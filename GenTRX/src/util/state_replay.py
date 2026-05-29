"""Replay a state-publish tick into a MatchingEngine.

Reset the engine, rebuild the LOB from the tick's L2 snapshot, then
apply the tick's events on top. Used by both the gradient server's
training pipeline and gentrx-serve's inference engine pool to keep
their state representations identical.

Wire shape mirrors `state_packager.StatePackager.extract_state`:

    {
      "bids":   [[price_decimal, volume_decimal], ...],
      "asks":   [[price_decimal, volume_decimal], ...],
      "events": [{"y": "o"|"c"|"t", "s": 0|1, "i": int,
                  "p": float, "q": float, "t": int, ...}],
    }

`q` on order events is the REMAINING quantity after fills. Original
size is recovered by adding the per-taker fill totals.
"""

from __future__ import annotations

from typing import Any


def replay_tick_to_engine(
    engine: Any,
    book_data: dict,
    price_scale: int,
    vol_scale: int,
) -> int:
    """Reset and rebuild `engine` from the tick's L2 snapshot, then
    apply non-trade events on top. Returns the number of events
    actually applied.

    The engine is mutated in place. Callers that need an isolated
    copy should deepcopy after this call returns.
    """
    from GenTRX.src.util.schema import ASK, BID, CANCEL

    engine.reset()

    bids = book_data.get("bids") or []
    asks = book_data.get("asks") or []
    for level in reversed(bids):
        try:
            p = round(float(level[0]) * price_scale)
            v = max(1, round(float(level[1]) * vol_scale))
        except (IndexError, TypeError, ValueError):
            continue
        if p > 0:
            engine.process_order(BID, p, v, is_buy=True)
    for level in reversed(asks):
        try:
            p = round(float(level[0]) * price_scale)
            v = max(1, round(float(level[1]) * vol_scale))
        except (IndexError, TypeError, ValueError):
            continue
        if p > 0:
            engine.process_order(ASK, p, v, is_buy=False)

    events = book_data.get("events") or []
    taker_fill_qty: dict[int, float] = {}
    for ev in events:
        if isinstance(ev, dict) and ev.get("y") == "t":
            tid = ev.get("Ti", 0)
            taker_fill_qty[tid] = taker_fill_qty.get(tid, 0.0) + float(
                ev.get("q", 0)
            )

    n_applied = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        y = ev.get("y", "o")
        if y == "t":
            continue
        side = int(ev.get("s", 0) or 0)
        eid = int(ev.get("i", 0) or 0)
        try:
            price = float(ev.get("p", 0) or 0)
            remaining = float(ev.get("q", 0) or 0)
        except (TypeError, ValueError):
            continue
        is_buy = side == 0
        if y == "c":
            order_type = CANCEL
            qty = remaining
        else:
            order_type = BID if is_buy else ASK
            qty = remaining + taker_fill_qty.get(eid, 0.0)

        price_ticks = round(price * price_scale)
        vol_ticks = max(1, round(qty * vol_scale))
        if price_ticks <= 0 or vol_ticks <= 0:
            continue
        engine.process_order(order_type, price_ticks, vol_ticks, is_buy)
        n_applied += 1

    return n_applied
