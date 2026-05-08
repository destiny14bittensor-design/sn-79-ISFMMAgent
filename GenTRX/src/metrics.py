# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Training and evaluation metrics for GenTRX order model.

Provides human-interpretable metrics beyond raw CE loss:
  - Per-field top-k accuracy (especially order_type direction accuracy)
  - Mid-price direction accuracy (did the model predict the correct price move?)
  - Log-return forecast accuracy at 1-step and n-step horizons
  - Confusion matrix for order types

All functions work on batched tensors from the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class StepMetrics:
    """Accumulated metrics over a window of training/validation steps."""

    # Per-field accuracy accumulators
    field_correct: dict[str, int] = field(default_factory=dict)
    field_total: dict[str, int] = field(default_factory=dict)

    # Order type confusion: confusion[true][pred] += 1
    type_confusion: dict[int, dict[int, int]] = field(default_factory=dict)

    # Price direction accuracy (bid/ask → price went up/down)
    direction_correct: int = 0
    direction_total: int = 0

    # Mid-price return tracking for n-step horizon
    mid_prices: list[int] = field(default_factory=list)
    predicted_mid_prices: list[int] = field(default_factory=list)

    n_steps: int = 0

    def update(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        mid_prices: torch.Tensor | None = None,
    ) -> None:
        """Update metrics from a single batch.

        Args:
            logits: {field_name: (B, T, n_bins)} from model forward
            labels: {field_name: (B, T)} ground truth
            mid_prices: (B, T+1) mid prices for direction accuracy (optional)
        """
        self.n_steps += 1

        for name, field_logits in logits.items():
            preds = field_logits.argmax(dim=-1)  # (B, T)
            target = labels[name]
            correct = (preds == target).sum().item()
            total = target.numel()

            self.field_correct[name] = self.field_correct.get(name, 0) + correct
            self.field_total[name] = self.field_total.get(name, 0) + total

        # Order type confusion matrix
        if "order_type" in logits:
            ot_preds = logits["order_type"].argmax(dim=-1).reshape(-1)
            ot_labels = labels["order_type"].reshape(-1)
            for t, p in zip(ot_labels.tolist(), ot_preds.tolist()):
                if t not in self.type_confusion:
                    self.type_confusion[t] = {}
                self.type_confusion[t][p] = self.type_confusion[t].get(p, 0) + 1

        # Price direction accuracy: does predicted price bin agree with
        # the actual direction of mid-price movement?
        if mid_prices is not None and "price" in logits:
            self._update_direction(logits, labels, mid_prices)

    def _update_direction(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        mid_prices: torch.Tensor,
    ) -> None:
        """Check if the predicted price_bin direction matches actual mid movement.

        mid_prices shape: (B, T+1) — mid at each position plus the next step.
        A bid (type=0) with positive rel_price means price is above mid → bullish.
        We check: did mid actually move up in the next step?
        """
        # Predicted order types
        if "order_type" not in logits:
            return
        pred_types = logits["order_type"].argmax(dim=-1)  # (B, T)
        B, T = pred_types.shape

        if mid_prices.shape[1] < T + 1:
            return

        # Actual mid direction: +1 if mid went up, -1 if down, 0 if flat
        mid_current = mid_prices[:, :T].float()
        mid_next = mid_prices[:, 1 : T + 1].float()
        actual_dir = torch.sign(mid_next - mid_current)  # (B, T)

        # Predicted direction from order type: bid(0)→+1, ask(1)→-1, cancel(2)→0
        pred_dir = torch.zeros_like(pred_types, dtype=torch.float)
        pred_dir[pred_types == 0] = 1.0  # bid → bullish
        pred_dir[pred_types == 1] = -1.0  # ask → bearish

        # Only count non-zero predictions and non-zero actual moves
        mask = (pred_dir != 0) & (actual_dir != 0)
        if mask.sum() > 0:
            self.direction_correct += (pred_dir[mask] == actual_dir[mask]).sum().item()
            self.direction_total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        """Return human-readable metrics dict."""
        result: dict[str, float] = {}

        # Per-field accuracies
        for name in self.field_correct:
            total = self.field_total.get(name, 0)
            if total > 0:
                result[f"acc_{name}"] = self.field_correct[name] / total

        # Direction accuracy
        if self.direction_total > 0:
            result["acc_direction"] = self.direction_correct / self.direction_total

        # Order type per-class accuracy
        for true_type, pred_counts in sorted(self.type_confusion.items()):
            type_total = sum(pred_counts.values())
            type_correct = pred_counts.get(true_type, 0)
            type_name = {0: "bid", 1: "ask", 2: "cancel"}.get(true_type, str(true_type))
            if type_total > 0:
                result[f"acc_type_{type_name}"] = type_correct / type_total

        return result

    def reset(self) -> None:
        self.field_correct.clear()
        self.field_total.clear()
        self.type_confusion.clear()
        self.direction_correct = 0
        self.direction_total = 0
        self.mid_prices.clear()
        self.predicted_mid_prices.clear()
        self.n_steps = 0

    def format(self) -> str:
        """One-line summary for logging."""
        m = self.compute()
        parts = []
        for k, v in m.items():
            parts.append(f"{k}={v:.3f}")
        return " ".join(parts)


def compute_return_accuracy(
    generated_mids: list[int],
    actual_mids: list[int],
    horizons: list[int] = [1, 5, 10, 25],
) -> dict[str, float]:
    """Compare log-return direction accuracy between generated and actual mid prices.

    For each horizon h:
      - Compute log-return: r = log(mid[t+h] / mid[t]) for each t
      - Check if sign(r_generated) == sign(r_actual)
      - Report accuracy (fraction of correct sign predictions)

    Args:
        generated_mids: mid prices from model generation (one per generated order)
        actual_mids: actual mid prices from ground truth (same length or longer)
        horizons: list of step horizons to evaluate

    Returns:
        {"return_acc_1": 0.62, "return_acc_5": 0.58, ...}
    """
    n = min(len(generated_mids), len(actual_mids))
    if n < 2:
        return {}

    gen = np.array(generated_mids[:n], dtype=np.float64)
    act = np.array(actual_mids[:n], dtype=np.float64)

    # Replace zeros with nan to avoid log(0)
    gen[gen <= 0] = np.nan
    act[act <= 0] = np.nan

    result: dict[str, float] = {}

    for h in horizons:
        if n <= h:
            continue
        gen_ret = np.log(gen[h:] / gen[:-h])
        act_ret = np.log(act[h:] / act[:-h])

        # Only compare where both are finite
        valid = np.isfinite(gen_ret) & np.isfinite(act_ret) & (act_ret != 0)
        if valid.sum() == 0:
            continue

        gen_sign = np.sign(gen_ret[valid])
        act_sign = np.sign(act_ret[valid])
        acc = (gen_sign == act_sign).mean()
        result[f"return_acc_{h}"] = float(acc)

    return result
