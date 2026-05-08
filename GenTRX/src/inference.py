# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Closed-loop inference: generate → matching engine → LOB feedback → next step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from GenTRX.src.model import OrderModel
from GenTRX.src.orderbook import MatchingEngine, LobSnapshot
from GenTRX.src.tokenizer import OrderTokenizer


@dataclass
class GeneratedOrder:
    order_type: int
    price_bin: int
    vol_int_bin: int
    vol_dec_bin: int
    interval_bin: int
    mid_price: int
    lob_snapshot: LobSnapshot


def generate_with_engine(
    model: OrderModel,
    tokenizer: OrderTokenizer,
    engine: MatchingEngine,
    prompt: dict[str, torch.Tensor],
    n_orders: int = 100,
    temperature: float = 1.0,
    device: str = "cuda",
) -> list[GeneratedOrder]:
    """
    Generate orders autoregressively with matching engine feedback.

    After each sampled order the matching engine processes it and provides an
    updated LOB snapshot, which is fed back into the next model step.

    Args:
        model (OrderModel): Trained order generation model.
        tokenizer (OrderTokenizer): Tokenizer that defines vocabulary and config.
        engine (MatchingEngine): Matching engine for processing generated orders.
        prompt (dict[str, torch.Tensor]): Conditioning context with keys
            'order_types', 'price_bins', 'vol_int_bins', 'vol_dec_bins',
            'interval_bins', 'lob_volumes', 'time_of_day', 'mid_deltas' —
            each shaped (1, T).
        n_orders (int): Number of orders to generate. Defaults to 100.
        temperature (float): Sampling temperature. Defaults to 1.0.
        device (str): Torch device string. Defaults to 'cuda'.

    Returns:
        list[GeneratedOrder]: Generated order objects with their matching-engine
            LOB snapshots.
    """
    model.eval()
    cfg = tokenizer.config
    mcfg = model.config
    max_ctx = mcfg.max_seq_len

    # Move to device
    seqs = {k: v.to(device) for k, v in prompt.items()}

    # Track state for conditioning updates
    last_tod = int(seqs["time_of_day"][0, -1].item())
    snap = engine.snapshot()
    session_open_mid = snap.mid_price if snap.mid_price > 0 else None

    generated: list[GeneratedOrder] = []

    with torch.no_grad():
        for _ in range(n_orders):
            T = max_ctx
            logits = model(
                seqs["order_types"][:, -T:],
                seqs["price_bins"][:, -T:],
                seqs["vol_int_bins"][:, -T:],
                seqs["vol_dec_bins"][:, -T:],
                seqs["interval_bins"][:, -T:],
                seqs["lob_volumes"][:, -T:],
                seqs["time_of_day"][:, -T:],
                seqs["mid_deltas"][:, -T:],
            )

            # Sample each field
            sampled = {}
            for name, field_logits in logits.items():
                probs = F.softmax(field_logits[:, -1, :] / temperature, dim=-1)
                sampled[name] = torch.multinomial(probs, 1)

            otype = sampled["order_type"].item()
            p_bin = sampled["price"].item()
            vi_bin = sampled["vol_int"].item()
            vd_bin = sampled["vol_dec"].item()
            i_bin = sampled["interval"].item()

            # Convert to values for engine
            snap = engine.snapshot()
            mid = snap.mid_price
            price = _bin_to_price(p_bin, cfg.price, mid)
            volume = _bins_to_volume(vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec)
            is_buy = otype == 0

            if volume > 0:
                engine.process_order(otype, price, volume, is_buy)

            # Updated LOB state
            new_snap = engine.snapshot()
            new_lob = _snap_to_tensor(new_snap, cfg.lob_depth, device)

            # Update time-of-day
            interval_edges = np.linspace(
                cfg.interval.lo, cfg.interval.hi, cfg.interval.n_bins + 1
            )
            interval_ns = (interval_edges[i_bin] + interval_edges[i_bin + 1]) / 2
            last_tod = (last_tod + int(interval_ns / 1e9)) % 86400
            new_tod = torch.tensor([[last_tod // cfg.time_bin_seconds]], device=device)

            # Update mid delta
            new_mid = new_snap.mid_price
            if session_open_mid is None and new_mid > 0:
                session_open_mid = new_mid
            delta = (new_mid - session_open_mid) if session_open_mid else 0
            delta_clipped = max(-cfg.max_mid_delta, min(cfg.max_mid_delta, delta))
            new_md = torch.tensor([[delta_clipped + cfg.max_mid_delta]], device=device)

            # Append to sequences
            seqs["order_types"] = torch.cat(
                [seqs["order_types"], sampled["order_type"]], dim=1
            )
            seqs["price_bins"] = torch.cat(
                [seqs["price_bins"], sampled["price"]], dim=1
            )
            seqs["vol_int_bins"] = torch.cat(
                [seqs["vol_int_bins"], sampled["vol_int"]], dim=1
            )
            seqs["vol_dec_bins"] = torch.cat(
                [seqs["vol_dec_bins"], sampled["vol_dec"]], dim=1
            )
            seqs["interval_bins"] = torch.cat(
                [seqs["interval_bins"], sampled["interval"]], dim=1
            )
            seqs["lob_volumes"] = torch.cat(
                [seqs["lob_volumes"], new_lob.unsqueeze(0)], dim=1
            )
            seqs["time_of_day"] = torch.cat([seqs["time_of_day"], new_tod], dim=1)
            seqs["mid_deltas"] = torch.cat([seqs["mid_deltas"], new_md], dim=1)

            generated.append(
                GeneratedOrder(
                    order_type=otype,
                    price_bin=p_bin,
                    vol_int_bin=vi_bin,
                    vol_dec_bin=vd_bin,
                    interval_bin=i_bin,
                    mid_price=new_snap.mid_price,
                    lob_snapshot=new_snap,
                )
            )

    return generated


def _bin_to_price(bin_idx: int, cfg, mid_price: int) -> int:
    edges = np.linspace(cfg.lo, cfg.hi, cfg.n_bins + 1)
    center = (edges[bin_idx] + edges[bin_idx + 1]) / 2
    return mid_price + int(center)


def _bins_to_volume(vi_bin: int, vd_bin: int, vi_cfg, vd_cfg) -> int:
    """Reconstruct volume from int + dec bins for the matching engine (needs int ticks)."""
    vi_edges = np.linspace(vi_cfg.lo, vi_cfg.hi, vi_cfg.n_bins + 1)
    vi_center = (vi_edges[vi_bin] + vi_edges[vi_bin + 1]) / 2
    vd_edges = np.linspace(vd_cfg.lo, vd_cfg.hi, vd_cfg.n_bins + 1)
    vd_center = (vd_edges[vd_bin] + vd_edges[vd_bin + 1]) / 2
    return max(1, int(vi_center + vd_center))


def _snap_to_tensor(snap: LobSnapshot, depth: int, device: str) -> torch.Tensor:
    ask_vols = list(snap.ask_volumes[:depth])
    ask_vols += [0] * (depth - len(ask_vols))
    bid_vols = list(snap.bid_volumes[:depth])
    bid_vols += [0] * (depth - len(bid_vols))
    return torch.tensor([ask_vols + bid_vols], device=device, dtype=torch.float32)


def init_engine_from_data(
    encoded: dict[str, np.ndarray],
    tokenizer: OrderTokenizer,
    n_warmup: int = 100,
) -> MatchingEngine:
    """Initialize matching engine by replaying seed orders."""
    engine = MatchingEngine()
    cfg = tokenizer.config

    n = min(n_warmup, len(encoded["order_types"]))
    for i in range(n):
        otype = int(encoded["order_types"][i])
        p_bin = int(encoded["price_bins"][i])
        vi_bin = int(encoded["vol_int_bins"][i])
        vd_bin = int(encoded["vol_dec_bins"][i])

        snap = engine.snapshot()
        mid = snap.mid_price
        price = _bin_to_price(p_bin, cfg.price, mid)
        volume = _bins_to_volume(vi_bin, vd_bin, cfg.vol_int, cfg.vol_dec)
        is_buy = otype == 0

        if volume > 0:
            engine.process_order(otype, price, volume, is_buy)

    return engine
