# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Minimal LOB matching engine for inference.

Adapted from mlib/core/orderbook.py (Microsoft MarS). Stripped to continuous
auction only — no call auction, no agent framework, no event loop.

Used at inference time to compute LOB state after each generated order.
Training data uses actual book state instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: int
    price: int
    volume: int
    is_buy: bool
    timestamp_ns: int = 0


@dataclass
class Level:
    price: int
    volume: int
    orders: list[Order] = field(default_factory=list)

    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.volume += order.volume

    def remove_volume(self, vol: int) -> list[tuple[int, int]]:
        """Remove vol from front of queue. Returns [(order_id, matched_vol), ...]."""
        matched: list[tuple[int, int]] = []
        remaining = vol
        while remaining > 0 and self.orders:
            o = self.orders[0]
            take = min(o.volume, remaining)
            o.volume -= take
            self.volume -= take
            remaining -= take
            matched.append((o.order_id, take))
            if o.volume == 0:
                self.orders.pop(0)
        return matched


@dataclass
class Trade:
    price: int
    volume: int
    aggressor_id: int
    passive_id: int
    is_buy: bool


@dataclass
class LobSnapshot:
    mid_price: int
    ask_prices: list[int]
    ask_volumes: list[int]
    bid_prices: list[int]
    bid_volumes: list[int]


class MatchingEngine:
    """Price-time priority LOB with continuous matching."""

    def __init__(self) -> None:
        self.bids: list[Level] = []  # descending by price
        self.asks: list[Level] = []  # ascending by price
        self._next_order_id = 0

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def snapshot(self, depth: int = 10) -> LobSnapshot:
        asks = self.asks[:depth]
        bids = self.bids[:depth]
        ask_p = [level.price for level in asks]
        ask_v = [level.volume for level in asks]
        bid_p = [level.price for level in bids]
        bid_v = [level.volume for level in bids]

        mid = 0
        if bid_p and ask_p:
            mid = (bid_p[0] + ask_p[0]) // 2
        elif bid_p:
            mid = bid_p[0]
        elif ask_p:
            mid = ask_p[0]

        return LobSnapshot(
            mid_price=mid,
            ask_prices=ask_p,
            ask_volumes=ask_v,
            bid_prices=bid_p,
            bid_volumes=bid_v,
        )

    def process_order(
        self,
        order_type: int,
        price: int,
        volume: int,
        is_buy: bool,
    ) -> tuple[list[Trade], LobSnapshot]:
        """Process a single order. Returns (trades, resulting_snapshot).

        order_type: 0=Bid, 1=Ask, 2=Cancel
        """
        trades: list[Trade] = []

        if order_type == 2:  # Cancel
            self._cancel(price, volume, is_buy)
        elif is_buy:
            trades = self._match_buy(price, volume)
        else:
            trades = self._match_sell(price, volume)

        return trades, self.snapshot()

    def _match_buy(self, price: int, volume: int) -> list[Trade]:
        trades: list[Trade] = []
        remaining = volume
        empty: list[int] = []

        for i, level in enumerate(self.asks):
            if level.price > price or remaining <= 0:
                break
            take = min(level.volume, remaining)
            matched = level.remove_volume(take)
            remaining -= take
            for passive_id, matched_vol in matched:
                trades.append(Trade(level.price, matched_vol, -1, passive_id, True))
            if level.volume == 0:
                empty.append(i)

        for i in reversed(empty):
            self.asks.pop(i)

        if remaining > 0:
            self._insert_bid(price, remaining)

        return trades

    def _match_sell(self, price: int, volume: int) -> list[Trade]:
        trades: list[Trade] = []
        remaining = volume
        empty: list[int] = []

        for i, level in enumerate(self.bids):
            if level.price < price or remaining <= 0:
                break
            take = min(level.volume, remaining)
            matched = level.remove_volume(take)
            remaining -= take
            for passive_id, matched_vol in matched:
                trades.append(Trade(level.price, matched_vol, -1, passive_id, False))
            if level.volume == 0:
                empty.append(i)

        for i in reversed(empty):
            self.bids.pop(i)

        if remaining > 0:
            self._insert_ask(price, remaining)

        return trades

    def _cancel(self, price: int, volume: int, is_buy: bool) -> None:
        levels = self.bids if is_buy else self.asks
        for i, level in enumerate(levels):
            if level.price == price:
                level.remove_volume(volume)
                if level.volume == 0:
                    levels.pop(i)
                return

    def correct_from_l2(
        self,
        bid_prices: list[int],  # descending (best first), integer ticks; 0 = no level
        bid_volumes: list[int],
        ask_prices: list[int],  # ascending (best first), integer ticks; 0 = no level
        ask_volumes: list[int],
    ) -> int:
        """Correct engine state against an authoritative L2 snapshot.

        Mirrors LimitOrderBook.correct_from_l2 logic from
        e-rnd-taos/src/dataprocessing/reconstruction.py:
          1. Remove phantom bids above L2 best bid.
          2. Remove phantom asks below L2 best ask.
          3. Insert missing bid levels above current engine best bid.
          4. Insert missing ask levels below current engine best ask.

        Returns number of corrections applied.
        """
        corrections = 0

        valid_bids = [
            (p, v) for p, v in zip(bid_prices, bid_volumes) if p > 0 and v > 0
        ]
        valid_asks = [
            (p, v) for p, v in zip(ask_prices, ask_volumes) if p > 0 and v > 0
        ]

        l2_best_bid = valid_bids[0][0] if valid_bids else None
        l2_best_ask = valid_asks[0][0] if valid_asks else None

        # Remove phantom bids above L2 best bid
        if l2_best_bid is not None:
            while self.bids and self.bids[0].price > l2_best_bid:
                self.bids.pop(0)
                corrections += 1

        # Remove phantom asks below L2 best ask
        if l2_best_ask is not None:
            while self.asks and self.asks[0].price < l2_best_ask:
                self.asks.pop(0)
                corrections += 1

        # Insert missing bid levels above current engine best bid
        our_best_bid = self.bids[0].price if self.bids else None
        if l2_best_bid is not None and (
            our_best_bid is None or our_best_bid < l2_best_bid
        ):
            existing = {lvl.price for lvl in self.bids}
            for p, v in valid_bids:
                if our_best_bid is not None and p <= our_best_bid:
                    break
                if p not in existing:
                    self._insert_bid(p, v)
                    existing.add(p)
                    corrections += 1

        # Insert missing ask levels below current engine best ask
        our_best_ask = self.asks[0].price if self.asks else None
        if l2_best_ask is not None and (
            our_best_ask is None or our_best_ask > l2_best_ask
        ):
            existing = {lvl.price for lvl in self.asks}
            for p, v in valid_asks:
                if our_best_ask is not None and p >= our_best_ask:
                    break
                if p not in existing:
                    self._insert_ask(p, v)
                    existing.add(p)
                    corrections += 1

        return corrections

    def _insert_bid(self, price: int, volume: int) -> None:
        order = Order(self._next_order_id, price, volume, True)
        self._next_order_id += 1
        for i, level in enumerate(self.bids):
            if price == level.price:
                level.add(order)
                return
            if price > level.price:
                self.bids.insert(i, Level(price, volume, [order]))
                return
        self.bids.append(Level(price, volume, [order]))

    def _insert_ask(self, price: int, volume: int) -> None:
        order = Order(self._next_order_id, price, volume, False)
        self._next_order_id += 1
        for i, level in enumerate(self.asks):
            if price == level.price:
                level.add(order)
                return
            if price < level.price:
                self.asks.insert(i, Level(price, volume, [order]))
                return
        self.asks.append(Level(price, volume, [order]))
