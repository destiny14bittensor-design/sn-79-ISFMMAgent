# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Causal order model — decomposed embeddings, per-field output heads, LLaMA backbone.

Architecture (~12.1M params)
============================

Input: Each order at position t is represented by 8 embedding components (summed):

    Predicted fields (tokenized → embedding → also have output heads):
        emb_type       : Embedding(3, d)       — Bid(0) / Ask(1) / Cancel(2)
        emb_price      : Embedding(100, d)     — relative price bin (symmetric log-scale)
        emb_vol_int    : Embedding(64, d)      — integer part of volume (log-scale)
        emb_vol_dec    : Embedding(8, d)       — fractional part of volume
        emb_interval   : Embedding(64, d)      — time since previous order (log-scale ns bins)

    Conditioning fields (input only — not predicted, carry LOB/session context):
        time_proj      : MLP(2→64→d)          — cyclic sin/cos time-of-day (richer than v2)
        emb_mid_delta  : Embedding(4001, d)    — mid price delta from session open
        lob_proj       : Linear(20, d) + LN    — 10 ask + 10 bid volume levels

Backbone: HuggingFace LlamaForCausalLM (causal attention, RoPE).
    - lm_head removed — we use per-field output heads instead.
    - Defaults: d_model=288, 8 layers, 8 heads, d_ff=1152.

FiLM conditioning: At layers 2, 5, 7, time-of-day + LOB features are injected
    via Feature-wise Linear Modulation (scale+shift on hidden states). This lets
    the model learn regime-dependent behavior (e.g., market open vs lunch vs close)
    instead of just a constant additive offset. ~140K extra params.

Output heads: 5 independent Linear(d, n_bins) with LayerNorm:
    order_type(3) + price(100) + vol_int(64) + vol_dec(8) + interval(64) = 239 total

Loss: weighted sum of per-field cross-entropy losses.
    Position t predicts fields at position t+1 (standard causal LM shift).
    Class weights: bid=2.0, ask=4.0, cancel=0.5
    Field weights: order_type=2.0, price=1.5, vol_int=0.5, vol_dec=0.5, interval=0.3

Inference modes:
    1. Open-loop: model.generate() — sample each field, append to context, repeat.
    2. Closed-loop: generate_with_engine() — sample, execute in MatchingEngine,
       feed back real LOB state as conditioning for next step.

Key design choice: NO composite vocabulary. Fields are independent output heads,
not a single flattened token space. This avoids the combinatorial explosion of
3 × 100 × 64 × 8 × 64 = 9,830,400 composite tokens and allows per-field loss
weighting and accuracy tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM


@dataclass
class ModelConfig:
    # Per-field bin counts (must match tokenizer)
    n_types: int = 3
    n_price_bins: int = 100
    n_vol_int_bins: int = 64
    n_vol_dec_bins: int = 8
    n_interval_bins: int = 64

    # Model dims
    d_model: int = 288
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1152
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Conditioning
    lob_dim: int = 20
    max_mid_delta: int = 2000

    # FiLM conditioning: inject time+LOB as scale+shift at these backbone layers
    film_layers: tuple[int, ...] = (2, 5, 7)
    film_d_cond: int = 64  # hidden dim of FiLM projection MLP

    @property
    def mid_delta_buckets(self) -> int:
        return self.max_mid_delta * 2 + 1

    @property
    def field_sizes(self) -> dict[str, int]:
        return {
            "order_type": self.n_types,
            "price": self.n_price_bins,
            "vol_int": self.n_vol_int_bins,
            "vol_dec": self.n_vol_dec_bins,
            "interval": self.n_interval_bins,
        }


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: conditioning → (gamma, beta) → scale+shift.

    Given conditioning features c (per position), produces:
        gamma, beta = split(MLP(c))
        output = gamma * hidden + beta

    This is multiplicative conditioning — unlike additive projection, it can
    create genuinely different behavioral regimes (e.g., market open vs close).
    """

    def __init__(self, d_model: int, d_cond_input: int, d_cond_hidden: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_cond_input, d_cond_hidden),
            nn.GELU(),
            nn.Linear(d_cond_hidden, 2 * d_model),
        )
        # Init gamma near 1 (identity) and beta near 0 (no shift) so FiLM
        # starts as a no-op and the pretrained backbone isn't disrupted
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        # Set gamma bias to 1 (so initial gamma ≈ 1 → identity scaling)
        self.proj[-1].bias.data[:d_model] = 1.0

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            hidden: (B, T, d_model) — transformer hidden states
            cond: (B, T, d_cond_input) — conditioning features
        """
        gb = self.proj(cond)  # (B, T, 2*d_model)
        gamma, beta = gb.chunk(2, dim=-1)  # each (B, T, d_model)
        return gamma * hidden + beta


class OrderModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._param_count: int | None = None
        d = config.d_model

        # --- 8-component input embedding (summed) ---
        self.emb_type = nn.Embedding(config.n_types, d)
        self.emb_price = nn.Embedding(config.n_price_bins, d)
        self.emb_vol_int = nn.Embedding(config.n_vol_int_bins, d)
        self.emb_vol_dec = nn.Embedding(config.n_vol_dec_bins, d)
        self.emb_interval = nn.Embedding(config.n_interval_bins, d)
        self.emb_mid_delta = nn.Embedding(config.mid_delta_buckets, d)
        # Cyclic time-of-day: sin/cos → 2-layer MLP (richer than single Linear)
        self.time_proj = nn.Sequential(
            nn.Linear(2, config.film_d_cond),
            nn.GELU(),
            nn.Linear(config.film_d_cond, d),
        )
        self.lob_proj = nn.Sequential(nn.Linear(config.lob_dim, d), nn.LayerNorm(d))

        # --- LLaMA backbone ---
        llama_config = LlamaConfig(
            hidden_size=d,
            num_attention_heads=config.n_heads,
            intermediate_size=config.d_ff,
            num_hidden_layers=config.n_layers,
            attention_dropout=config.dropout,
            use_cache=False,
            vocab_size=2,  # placeholder, we don't use lm_head
        )
        self.backbone = LlamaForCausalLM(llama_config)
        del self.backbone.lm_head

        # --- FiLM conditioning layers ---
        # Injected after specific backbone layers to modulate hidden states
        # based on time-of-day + LOB context (22D input: 2 sin/cos + 20 LOB)
        film_input_dim = 2 + config.lob_dim  # time sin/cos + LOB volumes
        self.film_layers_idx = set(config.film_layers)
        self.film = nn.ModuleDict(
            {
                str(i): FiLMLayer(d, film_input_dim, config.film_d_cond)
                for i in config.film_layers
            }
        )

        # --- Per-field output heads ---
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, size))
                for name, size in config.field_sizes.items()
            }
        )

    @property
    def n_params(self) -> int:
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters())
        return self._param_count

    def param_summary(self) -> str:
        """Human-readable parameter summary by component."""
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            group = name.split(".")[0]
            groups[group] = groups.get(group, 0) + p.numel()
        total = self.n_params
        lines = [f"Total: {total:,} ({total/1e6:.1f}M)"]
        for g, count in sorted(groups.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"  {g:20s}: {count:>10,} ({pct:.1f}%)")
        return "\n".join(lines)

    @staticmethod
    def _time_sincos(time_of_day: torch.Tensor) -> torch.Tensor:
        """Convert time-of-day bin index to sin/cos features. (B, T) → (B, T, 2)."""
        tod_frac = time_of_day.float() * 5.0 / 86400.0  # 5s bins → fraction of day
        angle = tod_frac * 2.0 * 3.141592653589793
        return torch.stack([angle.sin(), angle.cos()], dim=-1)

    def _embed(
        self,
        order_types: torch.Tensor,
        price_bins: torch.Tensor,
        vol_int_bins: torch.Tensor,
        vol_dec_bins: torch.Tensor,
        interval_bins: torch.Tensor,
        lob_volumes: torch.Tensor | None = None,
        time_of_day: torch.Tensor | None = None,
        mid_deltas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build input embeddings and FiLM conditioning vector.

        Returns:
            embeds: (B, T, d_model) — summed input embeddings
            film_cond: (B, T, 22) — concatenated [time_sin, time_cos, lob_volumes]
                       for FiLM layers, or None if conditioning unavailable
        """
        x = (
            self.emb_type(order_types)
            + self.emb_price(price_bins)
            + self.emb_vol_int(vol_int_bins)
            + self.emb_vol_dec(vol_dec_bins)
            + self.emb_interval(interval_bins)
        )

        # Build FiLM conditioning: [time_sin, time_cos, lob_volumes]
        film_cond = None
        tod_features = None

        if time_of_day is not None:
            tod_features = self._time_sincos(time_of_day)  # (B, T, 2)
            x = x + self.time_proj(tod_features)

        if lob_volumes is not None:
            x = x + self.lob_proj(lob_volumes)

        # Concatenate conditioning for FiLM (only if both available)
        if tod_features is not None and lob_volumes is not None:
            film_cond = torch.cat([tod_features, lob_volumes], dim=-1)  # (B, T, 22)

        if mid_deltas is not None:
            x = x + self.emb_mid_delta(
                mid_deltas.clamp(0, self.config.mid_delta_buckets - 1)
            )
        return x, film_cond

    def _run_backbone(
        self,
        embeds: torch.Tensor,
        film_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run LLaMA backbone with FiLM conditioning injected at target layers.

        Replicates LlamaModel.forward but injects FiLM scale+shift after
        target decoder layers. This is the only way to condition mid-backbone
        without monkey-patching HF internals.
        """
        backbone = self.backbone.model
        hidden = embeds
        T = embeds.shape[1]
        device = embeds.device

        # Build position IDs and RoPE embeddings (same as LlamaModel.forward)
        cache_position = torch.arange(T, device=device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = backbone.rotary_emb(hidden, position_ids=position_ids)

        # No explicit causal mask needed — SDPA applies is_causal=True internally
        # when attention_mask is None (faster than materializing a T×T mask tensor).

        for i, layer in enumerate(backbone.layers):
            out = layer(
                hidden_states=hidden,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
            )
            # LlamaDecoderLayer returns a tensor directly (not a tuple)
            hidden = out if isinstance(out, torch.Tensor) else out[0]
            # Apply FiLM after target layers
            if film_cond is not None and i in self.film_layers_idx:
                hidden = self.film[str(i)](hidden, film_cond)

        hidden = backbone.norm(hidden)
        return hidden

    def forward(
        self,
        order_types: torch.Tensor,
        price_bins: torch.Tensor,
        vol_int_bins: torch.Tensor,
        vol_dec_bins: torch.Tensor,
        interval_bins: torch.Tensor,
        lob_volumes: torch.Tensor | None = None,
        time_of_day: torch.Tensor | None = None,
        mid_deltas: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass. Returns dict of per-field logits, each (B, T, field_size)."""
        embeds, film_cond = self._embed(
            order_types,
            price_bins,
            vol_int_bins,
            vol_dec_bins,
            interval_bins,
            lob_volumes,
            time_of_day,
            mid_deltas,
        )
        hidden = self._run_backbone(embeds, film_cond)

        return {name: head(hidden) for name, head in self.heads.items()}

    @torch.no_grad()
    def generate(
        self,
        order_types: torch.Tensor,
        price_bins: torch.Tensor,
        vol_int_bins: torch.Tensor,
        vol_dec_bins: torch.Tensor,
        interval_bins: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        lob_volumes: torch.Tensor | None = None,
        time_of_day: torch.Tensor | None = None,
        mid_deltas: torch.Tensor | None = None,
    ) -> list[dict[str, int]]:
        """Open-loop generation. Returns list of per-field sampled values.

        Conditioning tensors (lob_volumes, time_of_day, mid_deltas) are extended
        by repeating their last value for each generated step. For closed-loop
        generation with real LOB feedback, use generate_with_engine() instead.
        """
        generated = []
        for _ in range(max_new_tokens):
            T = min(order_types.shape[1], self.config.max_seq_len)
            logits = self.forward(
                order_types[:, -T:],
                price_bins[:, -T:],
                vol_int_bins[:, -T:],
                vol_dec_bins[:, -T:],
                interval_bins[:, -T:],
                lob_volumes[:, -T:] if lob_volumes is not None else None,
                time_of_day[:, -T:] if time_of_day is not None else None,
                mid_deltas[:, -T:] if mid_deltas is not None else None,
            )

            # Sample each field independently from last position
            sampled = {}
            for name, field_logits in logits.items():
                probs = F.softmax(field_logits[:, -1, :] / temperature, dim=-1)
                sampled[name] = torch.multinomial(probs, 1)  # (B, 1)

            generated.append({k: v.item() for k, v in sampled.items()})

            # Append to predicted field sequences
            order_types = torch.cat([order_types, sampled["order_type"]], dim=1)
            price_bins = torch.cat([price_bins, sampled["price"]], dim=1)
            vol_int_bins = torch.cat([vol_int_bins, sampled["vol_int"]], dim=1)
            vol_dec_bins = torch.cat([vol_dec_bins, sampled["vol_dec"]], dim=1)
            interval_bins = torch.cat([interval_bins, sampled["interval"]], dim=1)

            # Extend conditioning by repeating last value (open-loop approximation)
            if lob_volumes is not None:
                lob_volumes = torch.cat([lob_volumes, lob_volumes[:, -1:]], dim=1)
            if time_of_day is not None:
                time_of_day = torch.cat([time_of_day, time_of_day[:, -1:]], dim=1)
            if mid_deltas is not None:
                mid_deltas = torch.cat([mid_deltas, mid_deltas[:, -1:]], dim=1)

        return generated


# Order type class weights: deprioritize cancel, prioritize bid/ask direction
# Cancel (48% of data) at 0.5, bid (38%) at 2.0, ask (14%) at 4.0
_ORDER_TYPE_WEIGHTS = torch.tensor([2.0, 4.0, 0.5])  # [bid, ask, cancel]

# Per-field loss multipliers: order_type and price matter most for trading.
# Interval reduced from 1.0 → 0.3: it consumed ~30% of total loss at ~6% accuracy,
# wasting gradient budget. Interval timing may need a fundamentally different
# approach (continuous output / Poisson) rather than more weight on bad bins.
_FIELD_WEIGHTS = {
    "order_type": 2.0,
    "price": 1.5,
    "vol_int": 0.5,
    "vol_dec": 0.5,
    "interval": 0.3,
}


def compute_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-field CE losses. Returns (total_loss, per_field_losses)."""
    device = next(iter(logits.values())).device
    total = torch.tensor(0.0, device=device)
    details = {}
    for name, field_logits in logits.items():
        field_labels = labels[name]
        if name == "order_type":
            weight = _ORDER_TYPE_WEIGHTS.to(device)
            loss = F.cross_entropy(
                field_logits.reshape(-1, field_logits.size(-1)),
                field_labels.reshape(-1),
                weight=weight,
            )
        else:
            loss = F.cross_entropy(
                field_logits.reshape(-1, field_logits.size(-1)),
                field_labels.reshape(-1),
            )
        field_weight = _FIELD_WEIGHTS.get(name, 1.0)
        total = total + loss * field_weight
        details[name] = loss.item()
    return total, details
