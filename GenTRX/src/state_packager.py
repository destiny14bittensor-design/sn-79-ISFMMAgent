# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""State extractor for the validator/proxy.

Walks a `MarketSimulationStateUpdate` (or compatible dict) and produces a
serialization-friendly tick packet that the gradient server can ingest.

State packet format:
    {
      "step": 0,
      "ts": 1234567890000,
      "books": {
        0: {
          "bids": [[price, qty], ...],
          "asks": [[price, qty], ...],
          "events": [
            {"y": "o", "s": 0, "i": 42, "p": 100.5, "q": 1.0, "t": ...},
            {"y": "t", "Ti": 42, "Mi": 10, "p": 100.5, "q": 0.5, "t": ..., "s": 0},
            {"y": "c", "s": 0, "i": 10, "p": 100.5, "q": 0.0, "t": ...},
          ]
        },
      }
    }

Event types: "o" = order (quantity is REMAINING after fills),
"t" = trade (Ti=taker_id, Mi=maker_id, q=filled quantity),
"c" = cancellation (i=orderId).

To reconstruct original order size: order.q + sum(trade.q where trade.Ti == order.i)

Transport: GenTRXService.push_state msgpack-encodes the packet and POSTs it
to the gradient server's `/gentrx/state` endpoint. The gradient server runs
as a separate process — single-machine deployments use a loopback URL.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _val(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from dict or object.

    Handles both pydantic objects (property accessors) and raw dicts
    (which may use short keys from the taos protocol).
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    v = getattr(obj, key, None)
    return v if v is not None else default


# Short-key aliases used by the taos protocol wire format.
# parse_dict() sets state.books to raw dicts with short keys (b/a/e/p/q/i/s/t).
# Pydantic objects use long-name @property accessors (bids/asks/events/price/...).
_SHORT = {
    "bids": "b",
    "asks": "a",
    "events": "e",
    "price": "p",
    "quantity": "q",
    "id": "i",
    "side": "s",
    "timestamp": "t",
    "orderId": "i",
    "taker_id": "Ti",
    "maker_id": "Mi",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get value trying long name first, then short alias."""
    v = _val(obj, key)
    if v is not None:
        return v
    short = _SHORT.get(key)
    if short is not None:
        return _val(obj, short, default)
    return default


def _get_price(level: Any) -> float:
    return _get(level, "price", 0)


def _get_qty(level: Any) -> float:
    return _get(level, "quantity", 0)


def _is_trade(event: Any) -> bool:
    if isinstance(event, dict):
        return "taker_id" in event or "takerAgentId" in event or "Ti" in event
    return hasattr(event, "taker_id") or hasattr(event, "takerAgentId")


def _is_cancellation(event: Any) -> bool:
    if isinstance(event, dict):
        return event.get("y") == "c" or "orderId" in event
    return hasattr(event, "orderId")


_SIM_EVENT_CODES = {"ESS", "ESE"}


def _extract_sim_events(state: Any) -> list[str]:
    """Return abbreviated sim-lifecycle event codes present on this tick.

    Walks `state.notices` (or the raw dict equivalent) and returns any
    'ESS' (SimulationStartEvent) or 'ESE' (SimulationEndEvent) codes
    found. Empty list when neither is present, which is the vast majority
    of ticks.
    """
    notices = _val(state, "notices", []) or []
    out: list[str] = []
    for n in notices:
        code = _val(n, "y") or _val(n, "type")
        if code in _SIM_EVENT_CODES and code not in out:
            out.append(code)
    return out


class StatePackager:
    """Extract a tick packet from the validator's state object.

    Pure transformation — no I/O. The returned dict is serialized to msgpack
    by GenTRXService.push_state and POSTed to the gradient server.
    """

    def __init__(self) -> None:
        self._step: int = 0
        self._config_saved: bool = False

    def extract_state(self, state: Any) -> dict:
        """Extract books + events + timestamp from state into a tick dict."""
        books = _val(state, "books") or {}
        packed_books = {}

        for book_id, book in books.items():
            bids = [
                [_get_price(level), _get_qty(level)]
                for level in (_get(book, "bids") or [])
            ]
            asks = [
                [_get_price(level), _get_qty(level)]
                for level in (_get(book, "asks") or [])
            ]
            events = self._extract_events(book)
            packed_books[int(book_id)] = {
                "bids": bids,
                "asks": asks,
                "events": events,
            }

        packet = {
            "step": self._step,
            "ts": _get(state, "timestamp", 0),
            "books": packed_books,
        }

        # Sim lifecycle markers. The simulator emits SimulationStartEvent
        # ('ESS') on a fresh sim and SimulationEndEvent ('ESE') when the
        # active sim finishes; both arrive via state.notices. The gradient
        # server uses these as the primary signal to reset in-memory tick
        # buffers and schedule data/ cleanup on S3; a backwards-time
        # heuristic in the gradient server handles the case where no ESE is emitted.
        sim_events = _extract_sim_events(state)
        if sim_events:
            packet["sim_events"] = sim_events

        # Include decimal config + simulation_id on first packet so the gradient
        # server knows the scale and can stamp run-identity onto its log events
        # without having to invent its own id.
        if not self._config_saved:
            config = _val(state, "config", {})
            sim_id = _get(config, "simulation_id", None)
            packet["config"] = {
                "priceDecimals": _get(config, "priceDecimals", 8),
                "volumeDecimals": _get(config, "volumeDecimals", 8),
                "simulation_id": sim_id,
            }
            # Only mark saved once we have a real sim_id so the gradient
            # server isn't stuck with sim_id=null if the first packet arrives
            # before the simulator has set its logDir.
            if sim_id is not None:
                self._config_saved = True

        self._step += 1
        return packet

    def _extract_events(self, book: Any) -> list[dict]:
        """Extract raw event dicts from book (no matching engine).

        Includes trade events (y="t") so the gradient server can reconstruct
        original order sizes: Order.quantity is the REMAINING size after fills,
        so we need Trade.taker_id + Trade.quantity to recover the full volume.
        """
        events = _get(book, "events") or []
        out = []
        for e in events:
            if _is_trade(e):
                out.append(
                    {
                        "y": "t",
                        "Ti": _get(e, "taker_id", 0),
                        "Mi": _get(e, "maker_id", 0),
                        "p": _get(e, "price", 0),
                        "q": float(_get(e, "quantity", 0)),
                        "t": int(_get(e, "timestamp", 0)),
                        "s": _get(e, "side", 0),
                    }
                )
            elif _is_cancellation(e):
                out.append(
                    {
                        "y": "c",
                        "s": _get(e, "side", 0),
                        "i": _get(e, "orderId", 0),
                        "p": _get(e, "price", 0),
                        "q": float(_get(e, "quantity", 0)),
                        "t": int(_get(e, "timestamp", 0)),
                    }
                )
            else:
                out.append(
                    {
                        "y": "o",
                        "s": _get(e, "side", 0),
                        "i": _get(e, "id", 0),
                        "p": _get(e, "price", 0),
                        "q": float(_get(e, "quantity", 0)),
                        "t": int(_get(e, "timestamp", 0)),
                    }
                )
        return out
