# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Gradient extraction, compression, and application for distributed training.

Core types:
  GradientDelta  — raw state dict delta (θ_after - θ_before) + metadata
  CompressedGradient — top-k sparsified, serializable to bytes

Compression: top-k sparsification per tensor. Keeps only the k largest
values (by absolute magnitude). Upgrade path to DCT transform later.

Usage:
    # Extract delta after training window
    delta = extract_delta(theta_before, theta_after, metadata)

    # Compress for transmission
    compressed = compress(delta, top_k_frac=0.01)

    # Decompress and apply
    apply_gradient(model, decompress(compressed))

    # Aggregate multiple deltas
    agg = aggregate([compressed_1, compressed_2, ...])
    apply_gradient(model, decompress(agg))
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class GradientMetadata:
    """Provenance info attached to a gradient delta."""

    window_id: int = 0
    miner_uid: int = 0
    steps_trained: int = 0
    loss_before: float = 0.0
    loss_after: float = 0.0
    # Per-step loss trajectory (for proof-of-training verification)
    loss_trajectory: list[float] = field(default_factory=list)


@dataclass
class GradientDelta:
    """Raw gradient delta: Δθ = θ_after - θ_before as a state dict."""

    delta: dict[str, torch.Tensor]
    metadata: GradientMetadata

    @property
    def n_params(self) -> int:
        return sum(t.numel() for t in self.delta.values())

    @property
    def norm(self) -> float:
        return sum(t.pow(2).sum().item() for t in self.delta.values()) ** 0.5


@dataclass
class CompressedGradient:
    """Top-k sparsified gradient — only stores indices + values per tensor.

    Each tensor is stored as (indices, values, original_shape).
    Total storage ≈ 2 * k * n_tensors (indices + values).
    """

    sparse: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Size]]
    metadata: GradientMetadata

    @property
    def n_nonzero(self) -> int:
        return sum(vals.numel() for _, (_, vals, _) in self.sparse.items())

    @property
    def compression_ratio(self) -> float:
        total = sum(shape.numel() for _, (_, _, shape) in self.sparse.items())
        return total / max(self.n_nonzero, 1)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Take a detached CPU copy of model parameters."""
    return {
        name: param.detach().cpu().clone() for name, param in model.named_parameters()
    }


def extract_delta(
    theta_before: dict[str, torch.Tensor],
    theta_after: dict[str, torch.Tensor],
    metadata: GradientMetadata | None = None,
) -> GradientDelta:
    """Compute Δθ = θ_after - θ_before."""
    delta = {}
    for name in theta_before:
        after = theta_after[name].cpu()
        before = theta_before[name].cpu()
        delta[name] = after - before
    return GradientDelta(delta=delta, metadata=metadata or GradientMetadata())


# ---------------------------------------------------------------------------
# Compression (top-k sparsification)
# ---------------------------------------------------------------------------


def compress(delta: GradientDelta, top_k_frac: float = 0.01) -> CompressedGradient:
    """Top-k compress a gradient delta.

    Keeps the top_k_frac fraction of values (by absolute magnitude) per tensor.
    E.g., top_k_frac=0.01 keeps 1% of values.
    """
    sparse = {}
    for name, tensor in delta.delta.items():
        flat = tensor.flatten()
        k = max(1, int(flat.numel() * top_k_frac))
        k = min(k, flat.numel())

        _, top_indices = torch.topk(flat.abs(), k)
        top_values = flat[top_indices]

        sparse[name] = (top_indices, top_values, tensor.shape)

    comp = CompressedGradient(sparse=sparse, metadata=delta.metadata)
    logger.debug(
        "Compressed: %d params → %d nonzero (%.1fx)",
        delta.n_params,
        comp.n_nonzero,
        comp.compression_ratio,
    )
    return comp


def decompress(comp: CompressedGradient) -> GradientDelta:
    """Decompress top-k back to dense tensors."""
    delta = {}
    for name, (indices, values, shape) in comp.sparse.items():
        flat = torch.zeros(shape.numel())
        flat[indices] = values
        delta[name] = flat.reshape(shape)
    return GradientDelta(delta=delta, metadata=comp.metadata)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(gradients: list[CompressedGradient]) -> CompressedGradient:
    """Average multiple compressed gradients by decompressing, summing, and re-averaging.

    Returns a dense GradientDelta (not re-compressed) wrapped as CompressedGradient
    with all values retained. Re-compress after if needed for transmission.
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")
    if len(gradients) == 1:
        return gradients[0]

    # Decompress all, average
    deltas = [decompress(g) for g in gradients]
    names = list(deltas[0].delta.keys())
    # n = len(deltas)

    avg_delta: dict[str, torch.Tensor] = {}
    for name in names:
        stacked = torch.stack([d.delta[name] for d in deltas])
        avg_delta[name] = stacked.mean(dim=0)

    # Metadata from first, annotate count
    meta = GradientMetadata(
        window_id=gradients[0].metadata.window_id,
        steps_trained=sum(g.metadata.steps_trained for g in gradients),
    )

    # Return as "fully dense" CompressedGradient (every index kept)
    sparse = {}
    for name, tensor in avg_delta.items():
        flat = tensor.flatten()
        indices = torch.arange(flat.numel())
        sparse[name] = (indices, flat, tensor.shape)

    return CompressedGradient(sparse=sparse, metadata=meta)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@torch.no_grad()
def apply_gradient(model: torch.nn.Module, delta: GradientDelta) -> None:
    """Add a gradient delta to model parameters in-place."""
    param_dict = dict(model.named_parameters())
    for name, d in delta.delta.items():
        if name in param_dict:
            param_dict[name].add_(d.to(param_dict[name].device))
        else:
            logger.warning("Skipping unknown param: %s", name)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize(comp: CompressedGradient) -> bytes:
    """Serialize CompressedGradient to bytes for network transmission."""
    buf = io.BytesIO()
    save_dict: dict[str, Any] = {
        "metadata": {
            "window_id": comp.metadata.window_id,
            "miner_uid": comp.metadata.miner_uid,
            "steps_trained": comp.metadata.steps_trained,
            "loss_before": comp.metadata.loss_before,
            "loss_after": comp.metadata.loss_after,
            "loss_trajectory": comp.metadata.loss_trajectory,
        },
        "sparse": {
            name: {
                "indices": indices,
                "values": values,
                "shape": list(shape),
            }
            for name, (indices, values, shape) in comp.sparse.items()
        },
    }
    torch.save(save_dict, buf)
    return buf.getvalue()


def deserialize(data: bytes) -> CompressedGradient:
    """Deserialize bytes back to CompressedGradient."""
    buf = io.BytesIO(data)
    d = torch.load(buf, map_location="cpu", weights_only=False)

    meta_d = d["metadata"]
    metadata = GradientMetadata(
        window_id=meta_d["window_id"],
        miner_uid=meta_d["miner_uid"],
        steps_trained=meta_d["steps_trained"],
        loss_before=meta_d["loss_before"],
        loss_after=meta_d["loss_after"],
        loss_trajectory=meta_d.get("loss_trajectory", []),
    )

    sparse = {}
    for name, sd in d["sparse"].items():
        sparse[name] = (sd["indices"], sd["values"], torch.Size(sd["shape"]))

    return CompressedGradient(sparse=sparse, metadata=metadata)
