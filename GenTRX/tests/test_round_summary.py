"""Tests for round-summary observability (batch A of next_up.md §1).

Pins the contract for `_log_event` aggregation events:

  - The new whitelist (n_assigned, n_delivered, n_collected,
    loss_improvement_pct, rollback_rate_10w/_50w, t_proposal_eval_s,
    t_save_ckpt_s, t_loader_build_s) lands in `_last_aggregation` so
    the validator-side Prometheus collector picks it up.
  - `_rollback_history` tracks True/False per round that made a real
    rollback decision; no-accepted and sibling-only paths skip it.
  - `rollback_rate_10w` and `rollback_rate_50w` compute the trailing
    average correctly across the deque.

Run: pytest GenTRX/tests/test_round_summary.py -v
"""

import pytest


@pytest.fixture
def aggregator(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator

    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
    )


def _agg_event(**overrides):
    """Minimal aggregation event with sensible defaults."""
    event = {
        "type": "aggregation",
        "round": 1,
        "n_assigned": 5,
        "n_delivered": 3,
        "n_collected": 2,
        "n_scored": 2,
        "n_accepted": 1,
        "loss_before": 10.0,
        "loss_after": 8.0,
        "loss_improvement_pct": 0.2,
        "t_proposal_eval_s": 1.5,
        "t_save_ckpt_s": 0.3,
        "t_loader_build_s": 0.5,
        "rolled_back": False,
        "version": 7,
    }
    event.update(overrides)
    return event


def test_log_event_whitelist_passes_new_keys(aggregator):
    """All batch-A keys should land in `_last_aggregation`."""
    aggregator._log_event(_agg_event())
    snap = aggregator._last_aggregation
    for key in (
        "round",
        "n_assigned",
        "n_delivered",
        "n_collected",
        "n_scored",
        "n_accepted",
        "loss_before",
        "loss_after",
        "loss_improvement_pct",
        "t_proposal_eval_s",
        "t_save_ckpt_s",
        "t_loader_build_s",
        "rolled_back",
        "version",
    ):
        assert key in snap, f"{key} missing from _last_aggregation"


def test_rollback_history_appends_on_decision(aggregator):
    """Events with `rolled_back` present push into history (success path)."""
    aggregator._log_event(_agg_event(rolled_back=False))
    aggregator._log_event(_agg_event(rolled_back=True))
    aggregator._log_event(_agg_event(rolled_back=False))
    assert list(aggregator._rollback_history) == [False, True, False]


def test_rollback_history_skips_no_accepted(aggregator):
    """No-accepted aggregation has no `rolled_back` key — skip the history."""
    no_accepted = {
        "type": "aggregation",
        "round": 1,
        "n_assigned": 5,
        "n_delivered": 3,
        "n_collected": 0,
        "n_scored": 0,
        "n_accepted": 0,
        "version": 7,
    }
    aggregator._log_event(no_accepted)
    assert len(aggregator._rollback_history) == 0


def test_rollback_history_skips_sibling_only(aggregator):
    """Sibling-only aggregation never makes a rollback decision."""
    sibling = _agg_event(sibling_only=True)
    sibling.pop("rolled_back")
    aggregator._log_event(sibling)
    assert len(aggregator._rollback_history) == 0


def test_rollback_history_sibling_only_with_explicit_rolled_back(aggregator):
    """Even with rolled_back set, sibling_only takes precedence and skips."""
    aggregator._log_event(_agg_event(sibling_only=True, rolled_back=False))
    assert len(aggregator._rollback_history) == 0


def test_rollback_rate_10w_50w_math(aggregator):
    """rollback_rate_10w looks at the last 10 entries; _50w at all up to 50."""
    for _ in range(5):
        aggregator._log_event(_agg_event(rolled_back=False))
    for _ in range(3):
        aggregator._log_event(_agg_event(rolled_back=True))
    snap = aggregator._last_aggregation
    assert snap["rollback_rate_10w"] == pytest.approx(3 / 8)
    assert snap["rollback_rate_50w"] == pytest.approx(3 / 8)


def test_rollback_rate_10w_lags_50w_when_recent_calm(aggregator):
    """After a burst of rollbacks then 10 clean rounds, 10w drops while 50w stays elevated."""
    for _ in range(5):
        aggregator._log_event(_agg_event(rolled_back=True))
    for _ in range(10):
        aggregator._log_event(_agg_event(rolled_back=False))
    snap = aggregator._last_aggregation
    assert snap["rollback_rate_10w"] == pytest.approx(0.0)
    assert snap["rollback_rate_50w"] == pytest.approx(5 / 15)


def test_rollback_history_capped_at_50(aggregator):
    """Deque has maxlen=50, oldest entries fall off."""
    for _ in range(60):
        aggregator._log_event(_agg_event(rolled_back=True))
    for _ in range(5):
        aggregator._log_event(_agg_event(rolled_back=False))
    assert len(aggregator._rollback_history) == 50
    snap = aggregator._last_aggregation
    assert snap["rollback_rate_50w"] == pytest.approx(45 / 50)


def test_rollbacks_total_increments_only_on_rollback(aggregator):
    """`_rollbacks_total` mirrors history but is a pure counter."""
    aggregator._log_event(_agg_event(rolled_back=False))
    aggregator._log_event(_agg_event(rolled_back=True))
    aggregator._log_event(_agg_event(rolled_back=True))
    aggregator._log_event(_agg_event(rolled_back=False))
    assert aggregator._rollbacks_total == 2


def test_rounds_aggregated_total_skips_rollbacks(aggregator):
    """Only successful, non-rolled-back rounds with accepted gradients count."""
    aggregator._log_event(_agg_event(rolled_back=False, n_accepted=2))
    aggregator._log_event(_agg_event(rolled_back=True, n_accepted=2))
    aggregator._log_event(_agg_event(rolled_back=False, n_accepted=0))
    aggregator._log_event(_agg_event(rolled_back=False, n_accepted=1))
    assert aggregator._rounds_aggregated_total == 2


def test_grad_norm_stats_flow_through_log_event(aggregator):
    """Grad-norm stat keys land in _last_aggregation when present in the event."""
    aggregator._log_event(_agg_event(
        grad_norm_mean=1.23,
        grad_norm_min=0.5,
        grad_norm_max=2.0,
        grad_norm_median=1.1,
        grad_norm_std=0.4,
    ))
    snap = aggregator._last_aggregation
    assert snap["grad_norm_mean"] == pytest.approx(1.23)
    assert snap["grad_norm_min"] == pytest.approx(0.5)
    assert snap["grad_norm_max"] == pytest.approx(2.0)
    assert snap["grad_norm_median"] == pytest.approx(1.1)
    assert snap["grad_norm_std"] == pytest.approx(0.4)


def test_grad_norm_stats_default_to_zero(aggregator):
    """Stable-shape contract: grad-norm keys exist from startup at 0 so the
    dashboard's gentrx_training{stat=grad_norm_*} series never gaps."""
    snap = aggregator._last_aggregation
    for key in (
        "grad_norm_mean",
        "grad_norm_min",
        "grad_norm_max",
        "grad_norm_median",
        "grad_norm_std",
    ):
        assert snap.get(key) == 0.0, f"{key} should default to 0.0"


# ---------------------------------------------------------------------------
# _deliver_scores carries was_rollback_winner + grad_norm per miner (§4.4)
# ---------------------------------------------------------------------------


def test_deliver_scores_includes_rollback_winner_and_norm(aggregator):
    """Per-miner payload exposes was_rollback_winner + grad_norm fields."""
    assignment = {
        "books": ["0"],
        "_score": 0.5,
        "_score_own": 0.5,
        "_score_held": 0.5,
        "_overfitting": False,
        "_grad_norm": 12.34,
        "_was_rollback_winner": True,
    }
    aggregator._assignments[7] = assignment
    aggregator._deliver_scores(
        scored=[(7, 1, 0.5, b"comp", assignment)],
        accepted=[(7, 1, 0.5, b"comp", assignment)],
        rejected=[],
        threshold=0.0,
        round_assignments=[(7, assignment)],
    )
    entry = aggregator._latest_scores["scores"]["7"]
    assert entry["was_rollback_winner"] is True
    assert entry["grad_norm"] == pytest.approx(12.34)


def test_deliver_scores_defaults_rollback_winner_to_false(aggregator):
    """Non-winning miners get was_rollback_winner=False by default."""
    assignment = {
        "books": ["0"],
        "_score": 0.5,
        "_score_own": 0.5,
        "_score_held": 0.5,
        "_overfitting": False,
        "_grad_norm": 5.0,
    }
    aggregator._assignments[3] = assignment
    aggregator._deliver_scores(
        scored=[(3, 1, 0.5, b"comp", assignment)],
        accepted=[(3, 1, 0.5, b"comp", assignment)],
        rejected=[],
        threshold=0.0,
        round_assignments=[(3, assignment)],
    )
    entry = aggregator._latest_scores["scores"]["3"]
    assert entry["was_rollback_winner"] is False
    assert entry["grad_norm"] == pytest.approx(5.0)


def test_deliver_scores_non_submitter_has_zero_grad_norm(aggregator):
    """Stable-shape contract: a non-submitter still gets numeric defaults
    (grad_norm=0.0, was_rollback_winner=False) so the dashboard's per-miner
    series never gaps."""
    assignment = {"books": ["1"]}
    aggregator._assignments[9] = assignment
    aggregator._deliver_scores(
        scored=[],
        accepted=[],
        rejected=[],
        threshold=0.0,
        round_assignments=[(9, assignment)],
    )
    entry = aggregator._latest_scores["scores"]["9"]
    assert entry["was_rollback_winner"] is False
    assert entry["grad_norm"] == 0.0
    assert entry["score_own"] == 0.0
    assert entry["score_held"] == 0.0


# ---------------------------------------------------------------------------
# Index-overlap collusion detector (§4.7)
# ---------------------------------------------------------------------------


def _fake_comp(sparse_dict):
    """Build a CompressedGradient-shaped object from a {name: (indices, vals, shape)} map."""
    import torch
    from GenTRX.src.gradient import CompressedGradient, GradientMetadata

    sparse = {
        name: (
            torch.tensor(idx, dtype=torch.int64),
            torch.tensor(vals, dtype=torch.float32),
            torch.Size(shape),
        )
        for name, (idx, vals, shape) in sparse_dict.items()
    }
    return CompressedGradient(sparse=sparse, metadata=GradientMetadata())


def _accepted_tuple(uid, comp):
    """Mimic the (uid, window, score, comp, assignment) shape used in _aggregate_accepted."""
    return (uid, 1, 0.5, comp, {"books": [str(uid)]})


def test_overlap_skips_below_two_miners(aggregator):
    """Need at least 2 miners to compute pairwise overlap."""
    comp = _fake_comp({"layer.weight": ([0, 1, 2], [1.0, 1.0, 1.0], [100])})
    assert aggregator._compute_index_overlap([_accepted_tuple(1, comp)]) == {}


def test_overlap_identical_indices(aggregator):
    """Two miners with identical top-k indices → overlap = 1.0."""
    comp_a = _fake_comp({"layer.weight": ([0, 1, 2, 3, 4], [1.0] * 5, [1000])})
    comp_b = _fake_comp({"layer.weight": ([0, 1, 2, 3, 4], [9.9] * 5, [1000])})
    stats = aggregator._compute_index_overlap(
        [_accepted_tuple(1, comp_a), _accepted_tuple(2, comp_b)]
    )
    assert stats["overlap_pairs_checked"] == 1
    assert stats["overlap_pairs_high"] == 1
    assert stats["overlap_mean"] == pytest.approx(1.0)
    assert stats["overlap_max"] == pytest.approx(1.0)


def test_overlap_disjoint_indices(aggregator):
    """Disjoint top-k → overlap = 0."""
    comp_a = _fake_comp({"layer.weight": ([0, 1, 2], [1.0] * 3, [1000])})
    comp_b = _fake_comp({"layer.weight": ([10, 11, 12], [1.0] * 3, [1000])})
    stats = aggregator._compute_index_overlap(
        [_accepted_tuple(1, comp_a), _accepted_tuple(2, comp_b)]
    )
    assert stats["overlap_pairs_high"] == 0
    assert stats["overlap_mean"] == pytest.approx(0.0)


def test_overlap_jaccard_math(aggregator):
    """Half-overlapping indices → Jaccard = 2/4 = 0.5."""
    comp_a = _fake_comp({"layer.weight": ([0, 1, 2], [1.0] * 3, [1000])})
    comp_b = _fake_comp({"layer.weight": ([1, 2, 3], [1.0] * 3, [1000])})
    stats = aggregator._compute_index_overlap(
        [_accepted_tuple(1, comp_a), _accepted_tuple(2, comp_b)]
    )
    assert stats["overlap_mean"] == pytest.approx(0.5)
    assert stats["overlap_pairs_high"] == 0


def test_overlap_skips_small_params(aggregator):
    """Parameters below min_param_size shouldn't contribute."""
    comp_a = _fake_comp({"bias": ([0, 1], [1.0] * 2, [50])})
    comp_b = _fake_comp({"bias": ([5, 6], [1.0] * 2, [50])})
    stats = aggregator._compute_index_overlap(
        [_accepted_tuple(1, comp_a), _accepted_tuple(2, comp_b)]
    )
    assert stats == {}


# ---------------------------------------------------------------------------
# Loader cache (§1.2)
# ---------------------------------------------------------------------------


def test_loader_cache_starts_empty(aggregator):
    """Fresh aggregator has empty cache and zero hit/miss counters."""
    assert aggregator._loader_cache == {}
    assert aggregator._loader_cache_hits == 0
    assert aggregator._loader_cache_misses == 0


def test_loader_cache_cleared_with_scoring_cache(aggregator):
    """Round-boundary clear drops both caches."""
    aggregator._loader_cache[("val", ((0, 100),))] = "fake_loader"
    aggregator._loader_cache_hits = 5
    aggregator._loader_cache_misses = 3
    aggregator._clear_scoring_cache()
    assert aggregator._loader_cache == {}
    assert aggregator._loader_cache_hits == 0
    assert aggregator._loader_cache_misses == 0


# ---------------------------------------------------------------------------
# Proposal-norm filter (§1.3)
# ---------------------------------------------------------------------------


def test_proposal_norm_ratio_default(aggregator):
    """Default ratio is conservative — should not filter typical proposals."""
    assert aggregator.proposal_norm_ratio == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Model-version stamp + mismatch counter (§4.8)
# ---------------------------------------------------------------------------


def test_gradient_metadata_model_v_trained_default():
    """Backwards-compat: existing miners without the field get 0."""
    from GenTRX.src.gradient import GradientMetadata

    meta = GradientMetadata()
    assert meta.model_v_trained == 0


def test_gradient_metadata_round_trips_model_v():
    """Field survives serialize → deserialize via .grad blob."""
    import torch
    from GenTRX.src.gradient import (
        CompressedGradient,
        GradientMetadata,
        serialize,
        deserialize,
    )

    sparse = {
        "layer.weight": (
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.Size([1000]),
        )
    }
    comp = CompressedGradient(
        sparse=sparse,
        metadata=GradientMetadata(model_v_trained=42),
    )
    data = serialize(comp)
    restored = deserialize(
        data, expected_shapes={"layer.weight": torch.Size([1000])}
    )
    assert restored.metadata.model_v_trained == 42


def test_log_event_passes_per_field_loss_prefix(aggregator):
    """Per-field loss keys (any name ending in a field id) flow through."""
    aggregator._log_event(_agg_event(
        per_field_loss_before_order_type=0.84,
        per_field_loss_before_price=3.92,
        per_field_loss_after_price=3.50,
        per_field_loss_after_interval=4.73,
    ))
    snap = aggregator._last_aggregation
    assert snap["per_field_loss_before_order_type"] == pytest.approx(0.84)
    assert snap["per_field_loss_before_price"] == pytest.approx(3.92)
    assert snap["per_field_loss_after_price"] == pytest.approx(3.50)
    assert snap["per_field_loss_after_interval"] == pytest.approx(4.73)


def test_log_event_rejects_unrelated_per_field_keys(aggregator):
    """Prefix is `per_field_loss_`; other per_field_* keys aren't whitelisted."""
    aggregator._log_event(_agg_event(per_field_accuracy_price=0.5))
    assert "per_field_accuracy_price" not in aggregator._last_aggregation


def test_gradient_metadata_legacy_payload_defaults_to_zero():
    """A blob serialised before the new field still deserialises cleanly."""
    import io
    import torch
    from GenTRX.src.gradient import deserialize

    legacy_payload = {
        "metadata": {
            "window_id": 1,
            "miner_uid": 2,
            "steps_trained": 50,
            "loss_before": 10.0,
            "loss_after": 8.0,
            "loss_trajectory": [],
        },
        "sparse": {
            "layer.weight": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([0.5], dtype=torch.float32),
                "shape": [1000],
            }
        },
    }
    buf = io.BytesIO()
    torch.save(legacy_payload, buf)
    restored = deserialize(
        buf.getvalue(),
        expected_shapes={"layer.weight": torch.Size([1000])},
    )
    assert restored.metadata.model_v_trained == 0
