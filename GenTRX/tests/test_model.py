"""Smoke tests for OrderModel — forward, backward, generate, FiLM, tokenizer compat."""

import pytest
import torch

from GenTRX.src.model import OrderModel, ModelConfig, compute_loss
from GenTRX.src.tokenizer import BinConfig, TokenizerConfig, _symmetric_log_edges


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T = 2, 64


@pytest.fixture
def model():
    return OrderModel(ModelConfig())


@pytest.fixture
def dummy_inputs():
    return dict(
        order_types=torch.randint(0, 3, (B, T)),
        price_bins=torch.randint(0, 100, (B, T)),
        vol_int_bins=torch.randint(0, 64, (B, T)),
        vol_dec_bins=torch.randint(0, 8, (B, T)),
        interval_bins=torch.randint(0, 64, (B, T)),
        lob_volumes=torch.randn(B, T, 20),
        time_of_day=torch.randint(0, 17280, (B, T)),
        mid_deltas=torch.randint(0, 4001, (B, T)),
    )


@pytest.fixture
def dummy_inputs_no_cond():
    return dict(
        order_types=torch.randint(0, 3, (B, T)),
        price_bins=torch.randint(0, 100, (B, T)),
        vol_int_bins=torch.randint(0, 64, (B, T)),
        vol_dec_bins=torch.randint(0, 8, (B, T)),
        interval_bins=torch.randint(0, 64, (B, T)),
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestOrderModel:
    def test_forward_shapes(self, model, dummy_inputs):
        logits = model(**dummy_inputs)
        assert logits["order_type"].shape == (B, T, 3)
        assert logits["price"].shape == (B, T, 100)
        assert logits["vol_int"].shape == (B, T, 64)
        assert logits["vol_dec"].shape == (B, T, 8)
        assert logits["interval"].shape == (B, T, 64)

    def test_forward_no_conditioning(self, model, dummy_inputs_no_cond):
        """FiLM should be skipped gracefully when conditioning is absent."""
        logits = model(**dummy_inputs_no_cond)
        assert logits["order_type"].shape == (B, T, 3)

    def test_backward(self, model, dummy_inputs):
        logits = model(**dummy_inputs)
        labels = {k: torch.randint(0, v.shape[-1], (B, T)) for k, v in logits.items()}
        loss, details = compute_loss(logits, labels)
        loss.backward()
        assert loss.isfinite()
        assert set(details.keys()) == {
            "order_type",
            "price",
            "vol_int",
            "vol_dec",
            "interval",
        }

    def test_generate(self, model, dummy_inputs):
        model.eval()
        # Use batch=1 for generate
        inputs = {k: v[:1] for k, v in dummy_inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=3)
        assert len(generated) == 3
        for g in generated:
            assert set(g.keys()) == {
                "order_type",
                "price",
                "vol_int",
                "vol_dec",
                "interval",
            }

    def test_param_count(self, model):
        n = model.n_params
        assert 12_000_000 < n < 13_000_000, f"Expected ~12.1M params, got {n:,}"

    def test_film_layers_present(self, model):
        assert hasattr(model, "film")
        assert set(model.film.keys()) == {"2", "5", "7"}

    def test_film_init_near_identity(self, model):
        """FiLM gamma bias should start at 1.0 (identity scaling)."""
        for key, film_layer in model.film.items():
            bias = film_layer.proj[-1].bias.data
            d = model.config.d_model
            gamma_bias = bias[:d]
            beta_bias = bias[d:]
            assert torch.allclose(
                gamma_bias, torch.ones_like(gamma_bias)
            ), f"FiLM {key} gamma not near 1"
            assert torch.allclose(
                beta_bias, torch.zeros_like(beta_bias)
            ), f"FiLM {key} beta not near 0"


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestSymmetricLogBins:
    def test_edge_count(self):
        edges = _symmetric_log_edges(100, 500)
        assert len(edges) == 100

    def test_symmetry(self):
        edges = _symmetric_log_edges(100, 500)
        # Negative and positive halves should be mirror images
        neg = edges[:50]
        pos = edges[50:]
        assert pytest.approx(neg, abs=1e-6) == (-pos[::-1]).tolist()

    def test_zero_band(self):
        """Values in [-1, +1) should land in the same bin (the zero band)."""
        cfg = BinConfig(100, -500, 500, symmetric_log=True)
        import numpy as np

        vals = np.array([-0.5, 0.0, 0.5, 0.99])
        bins = cfg.digitize(vals)
        assert len(set(bins)) == 1, f"Zero band values should be same bin, got {bins}"

    def test_extremes(self):
        cfg = BinConfig(100, -500, 500, symmetric_log=True)
        import numpy as np

        bins = cfg.digitize(np.array([-500.0, 500.0]))
        assert bins[0] == 0
        assert bins[1] == 99

    def test_near_mid_resolution(self):
        """Higher resolution (bins per tick) near mid than deep."""
        cfg = BinConfig(100, -500, 500, symmetric_log=True)
        import numpy as np

        # Near mid: ±5 ticks uses N bins over 10-tick range
        near = cfg.digitize(np.array([-5, 5]))
        near_bins_per_tick = (near[1] - near[0]) / 10.0

        # Deep: ±100..500 ticks uses M bins over 800-tick range
        deep = cfg.digitize(np.array([-500, -100, 100, 500]))
        deep_bins_per_tick = (deep[3] - deep[0]) / 1000.0

        assert near_bins_per_tick > deep_bins_per_tick, (
            f"Near-mid should have higher resolution: "
            f"{near_bins_per_tick:.3f} vs {deep_bins_per_tick:.3f} bins/tick"
        )


class TestTokenizerBackwardCompat:
    def test_old_config_no_symmetric_log(self):
        """Old checkpoint configs without symmetric_log should default to False."""
        old_dict = {
            "n_types": 3,
            "price": {"n_bins": 100, "lo": -500, "hi": 500, "log_scale": False},
            "vol_int": {"n_bins": 64, "lo": 0, "hi": 100, "log_scale": True},
            "vol_dec": {"n_bins": 8, "lo": 0.0, "hi": 1.0, "log_scale": False},
            "interval": {"n_bins": 64, "lo": 0, "hi": 50000000, "log_scale": True},
        }
        cfg = TokenizerConfig.from_dict(old_dict)
        assert cfg.price.symmetric_log is False

    def test_new_config_has_symmetric_log(self):
        cfg = TokenizerConfig()
        assert cfg.price.symmetric_log is True


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestLoss:
    def test_interval_weight_reduced(self):
        from GenTRX.src.model import _FIELD_WEIGHTS

        assert _FIELD_WEIGHTS["interval"] == 0.3

    def test_field_weights_present(self):
        from GenTRX.src.model import _FIELD_WEIGHTS

        assert set(_FIELD_WEIGHTS.keys()) == {
            "order_type",
            "price",
            "vol_int",
            "vol_dec",
            "interval",
        }
