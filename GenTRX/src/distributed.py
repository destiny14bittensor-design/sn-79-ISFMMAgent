# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Window-based distributed training primitives.

Wraps the existing training infrastructure to support the distributed protocol:
  1. train_window()  — train for N steps, return a GradientDelta
  2. apply_gradient() — apply an aggregated delta to the model
  3. evaluate_gradient() — score a gradient by loss improvement (for validators)

Usage:
    from GenTRX.src.distributed import train_window, WindowConfig

    # Miner side: train a window and get the gradient
    delta = train_window(model, dataloader, WindowConfig(n_steps=100), device)

    # Compress and share
    compressed = compress(delta, top_k_frac=0.01)
    data = serialize(compressed)  # → bytes for upload

    # Validator side: score a gradient
    score = evaluate_gradient(model, compressed, val_loader, device)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from GenTRX.src.gradient import (
    GradientDelta,
    GradientMetadata,
    CompressedGradient,
    snapshot_state,
    extract_delta,
    decompress,
    apply_gradient,
)
from GenTRX.src.model import OrderModel, compute_loss
from GenTRX.src.train import _forward_batch

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for a single training window."""

    n_steps: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    window_id: int = 0
    miner_uid: int = 0


def train_window(
    model: OrderModel,
    dataloader: DataLoader,
    config: WindowConfig,
    device: str = "cpu",
) -> GradientDelta:
    """Train model for N steps and return the parameter delta.

    Snapshots θ_before, trains N steps on the dataloader, computes
    Δθ = θ_after - θ_before. Model is left in the trained state.

    Args:
        model: The model to train (will be modified in-place).
        dataloader: Training data. Will cycle if fewer batches than n_steps.
        config: Window training configuration.
        device: Device to train on.

    Returns:
        GradientDelta with the parameter change and training metadata.
    """
    theta_before = snapshot_state(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    model.train()
    loss_trajectory: list[float] = []
    step = 0
    t0 = time.perf_counter()

    data_iter = iter(dataloader)
    while step < config.n_steps:
        # Cycle through dataloader if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        logits, labels = _forward_batch(model, batch, device)
        loss, _ = compute_loss(logits, labels)

        if not math.isfinite(loss.item()):
            logger.error("NaN/Inf loss at window step %d — stopping window", step)
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        loss_trajectory.append(loss.item())
        step += 1

    elapsed = time.perf_counter() - t0
    theta_after = snapshot_state(model)

    loss_before = loss_trajectory[0] if loss_trajectory else 0.0
    loss_after = loss_trajectory[-1] if loss_trajectory else 0.0

    metadata = GradientMetadata(
        window_id=config.window_id,
        miner_uid=config.miner_uid,
        steps_trained=step,
        loss_before=loss_before,
        loss_after=loss_after,
        loss_trajectory=loss_trajectory,
    )

    delta = extract_delta(theta_before, theta_after, metadata)

    logger.info(
        "Window %d (miner %d): %d steps in %.1fs, loss %.4f → %.4f, Δnorm=%.4f",
        config.window_id,
        config.miner_uid,
        step,
        elapsed,
        loss_before,
        loss_after,
        delta.norm,
    )

    return delta


def evaluate_gradient(
    model: OrderModel,
    gradient: CompressedGradient,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 50,
) -> float:
    """Score a gradient by the loss improvement it produces.

    Computes loss_before on the current model, applies the gradient,
    computes loss_after, then **rolls back** the model to its original state.

    Returns:
        score = loss_before - loss_after (positive = improvement)
    """
    # Snapshot current state for rollback
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    loss_before = _eval_loss(model, val_loader, device, max_batches)

    # Apply gradient
    delta = decompress(gradient)
    apply_gradient(model, delta)

    loss_after = _eval_loss(model, val_loader, device, max_batches)

    # Rollback
    model.load_state_dict(original_state)

    score = loss_before - loss_after
    logger.info(
        "Gradient eval (miner %d, window %d): loss %.4f → %.4f, score = %+.4f",
        gradient.metadata.miner_uid,
        gradient.metadata.window_id,
        loss_before,
        loss_after,
        score,
    )
    return score


def _eval_loss(
    model: OrderModel,
    loader: DataLoader,
    device: str,
    max_batches: int,
) -> float:
    """Compute average loss over up to max_batches."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            logits, labels = _forward_batch(model, batch, device)
            loss, _ = compute_loss(logits, labels)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)
