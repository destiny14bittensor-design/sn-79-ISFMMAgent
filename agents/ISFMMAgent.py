# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
ISF-MM (Imbalance-Skewed Fair-Price Market Making) agent for SN-79 (τaos).

Designed for lightweight miner infrastructure (8 vCPU, ~17 GB RAM, no GPU):
- All 128 books participate: top 62.5% = core_good (active MM), rest = maintenance (survival quoting)
- Dropped-book cleanup on rank refresh: cancel resting orders; promote exposure to reserve
- Fee-aware fair-price quotes (spread covers maker+taker fees and book spread)
- RoundTripGuard: FIFO lot state machine → Kappa realized-observation round-trips
- Account inventory (own_base vs miner_base_balance baseline) for skew/limits/risk
- FIFO drift reconcile; RESET_AGENTS/RDRA + process-restart-safe baseline
- Maker markout toxic flow: per-side bid_toxic/ask_toxic → edge, size, entry block
- Instruction pulse (optional, default off): cancel-only stale refresh; no toxic bypass
- Volume throttle: restrict entry makers; allow exit makers + guarded taker unwind
- Volume hard stop (cancel-only at cap)
- ≤5 instructions per book; minimal cancel/repost churn
- No StateHistoryManager / GenTRX (lazy_load + history_len=0 friendly)
"""
from __future__ import annotations

import gc
import math
import os
import traceback
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field

import bittensor as bt

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent, TradeEvent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import ResetAgentsEvent, SimulationStartEvent
from taos.im.protocol.instructions import *
from taos.im.protocol.models import *
from taos.im.utils import duration_from_timestamp

# FIFO lot: (open_ts, qty, price, open_fee)
FifoLot = tuple[int, float, float, float]
# Completed round-trip observation bucket: (sim_tick_ts, aggregated realized_pnl_quote)
RealizedObs = tuple[int, float]
# Pending maker markout: (book_id, maker_side, fill_px, fill_ts, due_ts)
# maker_side: BUY = bid fill, SELL = ask fill
PendingMarkout = tuple[int, int, float, int, int]

# Used when dendrite hotkey / onTrade validator arg is missing.
_FALLBACK_VALIDATOR = "_default"

# Per-request validator context (safe under concurrent validator forwards).
_validator_ctx: ContextVar[str | None] = ContextVar("isfmm_validator_key", default=None)
_sim_ts_ctx: ContextVar[int | None] = ContextVar("isfmm_sim_ts", default=None)


def _cfg_float(config: object, name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _cfg_int(config: object, name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _cfg_bool(config: object, name: str, default: bool) -> bool:
    val = getattr(config, name, default)
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes")
    return bool(val)


def _round_price(price: float, decimals: int) -> float:
    return round(price, decimals)


def _round_qty(qty: float, decimals: int) -> float:
    return round(qty, decimals)


def _floor_tick(price: float, tick: float, decimals: int) -> float:
    steps = int(price / tick + 1e-9)
    return _round_price(steps * tick, decimals)


def _ceil_tick(price: float, tick: float, decimals: int) -> float:
    steps = int((price + tick - 1e-9) / tick)
    return _round_price(steps * tick, decimals)


@dataclass
class RealizedObsState:
    """
    Validator-scoped FIFO mirror for Kappa realized-observation tracking only.

    Lots/deques feed realized PnL observations; they are NOT used for inventory
    skew, soft limits, or throttle exit-side decisions (account baseline does).
    """

    longs: deque[FifoLot] = field(default_factory=deque)
    shorts: deque[FifoLot] = field(default_factory=deque)
    # Validator keys realized PnL by state.timestamp and sums per tick per book.
    obs_by_ts: dict[int, float] = field(default_factory=dict)

    def fifo_net_qty(self) -> float:
        """Open FIFO net (reconcile only — not used for MM skew or risk)."""
        long_qty = sum(lot[1] for lot in self.longs)
        short_qty = sum(lot[1] for lot in self.shorts)
        return long_qty - short_qty

    def clear_lots(self) -> None:
        self.longs.clear()
        self.shorts.clear()

    def add_observation(self, ts: int, pnl: float, *, decimals: int = 8) -> None:
        """Accumulate realized PnL into a tick bucket (matches validator Kappa input)."""
        rounded = round(pnl, decimals)
        if rounded == 0.0:
            return
        current = self.obs_by_ts.get(ts, 0.0)
        new_value = round(current + rounded, decimals)
        if new_value != 0.0:
            self.obs_by_ts[ts] = new_value
        elif ts in self.obs_by_ts:
            del self.obs_by_ts[ts]

    def prune(self, sim_ts: int, lookback_ns: int) -> None:
        cutoff = sim_ts - lookback_ns
        while self.obs_by_ts:
            oldest = next(iter(self.obs_by_ts))
            if oldest >= cutoff:
                break
            del self.obs_by_ts[oldest]

    def _window_obs(self) -> list[RealizedObs]:
        return list(self.obs_by_ts.items())

    def non_zero_count(self) -> int:
        return sum(1 for pnl in self.obs_by_ts.values() if abs(pnl) > 1e-12)

    def positive_count(self) -> int:
        return sum(1 for pnl in self.obs_by_ts.values() if pnl > 1e-12)

    def negative_count(self) -> int:
        return sum(1 for pnl in self.obs_by_ts.values() if pnl < -1e-12)

    def obs_deficit(self, min_obs: int) -> int:
        return max(0, min_obs - self.non_zero_count())

    def _changeover_count(self, grace_period_ns: int) -> tuple[int, int]:
        """(num_changeover_gaps_ge_grace, max_gap_ns). Mirrors validator kappa.py gap detection."""
        if grace_period_ns <= 0 or len(self.obs_by_ts) < 2:
            return 0, 0
        sorted_ts = list(self.obs_by_ts)
        n_co = 0
        max_gap = 0
        for i in range(1, len(sorted_ts)):
            gap = sorted_ts[i] - sorted_ts[i - 1]
            if gap >= grace_period_ns:
                n_co += 1
                if gap > max_gap:
                    max_gap = gap
        return n_co, max_gap

    def effective_non_zero_count(self, grace_period_ns: int) -> int:
        """non_zero_count minus one obs per changeover gap (safety buffer matching validator)."""
        n_co, _ = self._changeover_count(grace_period_ns)
        return max(0, self.non_zero_count() - n_co)

    def effective_obs_deficit(self, min_obs: int, grace_period_ns: int) -> int:
        return max(0, min_obs - self.effective_non_zero_count(grace_period_ns))

    def net_realized(self) -> float:
        return sum(self.obs_by_ts.values())

    def last_realized_ts(self) -> int:
        if not self.obs_by_ts:
            return 0
        return max(self.obs_by_ts)

    def front_lot(self) -> tuple[bool, FifoLot] | None:
        """Oldest open FIFO lot: (is_long, (open_ts, qty, entry_price, entry_fee))."""
        if self.longs:
            return True, self.longs[0]
        if self.shorts:
            return False, self.shorts[0]
        return None


@dataclass
class BookSelection:
    """
    Per-validator book tiers for Kappa-safe participation.

    core    = core_good tier (active MM, top ≥62.5% by health score)
    reserve = maintenance tier (all remaining books; low-frequency survival quoting)
    skip    = ALWAYS EMPTY; hard-skip is abolished
    """

    skip: set[int] = field(default_factory=set)
    core: set[int] = field(default_factory=set)
    reserve: set[int] = field(default_factory=set)

    def traded(self) -> set[int]:
        return self.core | self.reserve


@dataclass
class BookFlowState:
    """Rolling market trade flow from Book.events (all agents, not only our fills)."""

    records: deque[tuple[int, int, float]] = field(default_factory=deque)
    seen_trade_ids: set[int] = field(default_factory=set)

    def add(self, trade_id: int, sim_ts: int, qty: float, lookback_ns: int, now_ts: int) -> None:
        if trade_id in self.seen_trade_ids or qty <= 0.0:
            return
        self.seen_trade_ids.add(trade_id)
        self.records.append((sim_ts, trade_id, qty))
        self.prune(now_ts, lookback_ns)

    def prune(self, sim_ts: int, lookback_ns: int) -> None:
        cutoff = sim_ts - lookback_ns
        while self.records and self.records[0][0] < cutoff:
            _, trade_id, _ = self.records.popleft()
            self.seen_trade_ids.discard(trade_id)

    def trade_count(self) -> int:
        return len(self.records)

    def trade_volume(self) -> float:
        return sum(qty for _, _, qty in self.records)

    def last_trade_ts(self) -> int:
        if not self.records:
            return 0
        return self.records[-1][0]


@dataclass
class BookState:
    """Per-book runtime state for quoting and weak-book widening."""

    inventory: float = 0.0
    edge_mult: float = 1.0
    stuck_since_ts: int = 0
    last_trade_ts: int = 0
    last_instruction_ts: int = 0
    quoted_bid: float | None = None
    quoted_ask: float | None = None
    bid_toxic: float = 0.0
    ask_toxic: float = 0.0
    bid_cooldown_until_ts: int = 0
    ask_cooldown_until_ts: int = 0
    last_toxic_decay_ts: int = 0
    inventory_open_ts: int = 0
    inventory_last_sign: int = 0
    exit_only_until_flat: bool = False
    neg_probe_cooldown_until_ts: int = 0
    neg_probe_consecutive_neg: int = 0
    deficit_probe_last_ts: int = 0


def _fifo_match(
    longs: deque[FifoLot],
    shorts: deque[FifoLot],
    is_buy: bool,
    quantity: float,
    price: float,
    fee: float,
    timestamp: int,
) -> tuple[float, float]:
    """Mirror validator _match_trade_fifo (approximate realized PnL + roundtrip qty)."""
    if quantity <= 0.0:
        return 0.0, 0.0

    if is_buy:
        if not shorts:
            open_fee = fee * (quantity / quantity)
            longs.append((timestamp, quantity, price, open_fee))
            return 0.0, 0.0
    else:
        if not longs:
            open_fee = fee * (quantity / quantity)
            shorts.append((timestamp, quantity, price, open_fee))
            return 0.0, 0.0

    realized_pnl = 0.0
    roundtrip_volume = 0.0
    remaining_qty = quantity
    quantity_inv = 1.0 / quantity

    if is_buy:
        while remaining_qty > 0 and shorts:
            old_ts, old_qty, old_price, old_fee = shorts[0]
            if old_qty <= remaining_qty:
                price_pnl = (old_price - price) * old_qty
                close_fee = fee * old_qty * quantity_inv
                realized_pnl += price_pnl - old_fee - close_fee
                roundtrip_volume += old_qty
                remaining_qty -= old_qty
                shorts.popleft()
            else:
                old_qty_inv = 1.0 / old_qty
                price_pnl = (old_price - price) * remaining_qty
                close_fee = fee
                open_fee = old_fee * remaining_qty * old_qty_inv
                realized_pnl += price_pnl - open_fee - close_fee
                roundtrip_volume += remaining_qty
                remaining_position_fee = old_fee - open_fee
                shorts[0] = (old_ts, old_qty - remaining_qty, old_price, remaining_position_fee)
                remaining_qty = 0.0
        if remaining_qty > 0:
            open_fee = fee * remaining_qty * quantity_inv
            longs.append((timestamp, remaining_qty, price, open_fee))
    else:
        while remaining_qty > 0 and longs:
            old_ts, old_qty, old_price, old_fee = longs[0]
            if old_qty <= remaining_qty:
                price_pnl = (price - old_price) * old_qty
                close_fee = fee * old_qty * quantity_inv
                realized_pnl += price_pnl - old_fee - close_fee
                roundtrip_volume += old_qty
                remaining_qty -= old_qty
                longs.popleft()
            else:
                old_qty_inv = 1.0 / old_qty
                price_pnl = (price - old_price) * remaining_qty
                close_fee = fee
                open_fee = old_fee * remaining_qty * old_qty_inv
                realized_pnl += price_pnl - open_fee - close_fee
                roundtrip_volume += remaining_qty
                remaining_position_fee = old_fee - open_fee
                longs[0] = (old_ts, old_qty - remaining_qty, old_price, remaining_position_fee)
                remaining_qty = 0.0
        if remaining_qty > 0:
            open_fee = fee * remaining_qty * quantity_inv
            shorts.append((timestamp, remaining_qty, price, open_fee))

    return realized_pnl, roundtrip_volume


class ISFMMAgent(FinanceSimulationAgent):
    """
    SN-79-aligned market maker with RoundTripGuard.

    Scoring alignment:
    - Kappa-3: min 3 non-zero realized PnL obs per book in lookback (~3 sim hours).
    - RoundTripGuard: FIFO tracks realized observations only; account inventory drives risk/skew.
    - All 128 books participate: top 62.5% = core_good (active MM), rest = maintenance (survival quoting)
    - Instruction pulse: optional cancel-only stale refresh (default off; core books only).
    """

    def initialize(self) -> None:
        if not hasattr(self.config, "lazy_load"):
            self.config.lazy_load = True
        else:
            self.config.lazy_load = bool(self.config.lazy_load)

        self.history_len = _cfg_int(self.config, "history_len", 0)

        self.core_min_ratio = _cfg_float(self.config, "core_min_ratio", 0.625)
        self.core_target_ratio = _cfg_float(self.config, "core_target_ratio", 0.70)
        # core_min_ratio=0.625 → 80 books min; core_target_ratio=0.70 → 90 books target for 128 books.
        # Retained for config backward-compat only; not used for enforcement or entry gating.
        self.core_budget_ratio = _cfg_float(self.config, "core_budget_ratio", 0.75)
        self.maintenance_budget_ratio = _cfg_float(self.config, "maintenance_budget_ratio", 0.15)
        self.maint_cap_stage_ratio = _cfg_float(self.config, "maint_cap_stage_ratio", 0.80)
        # emergency_buffer_ratio: used in VOLUME_BUDGET log for display only.
        self.emergency_buffer_ratio = _cfg_float(self.config, "emergency_buffer_ratio", 0.10)
        # profit_core: top N core books by realized PnL + markout quality; get volume priority.
        self.profit_core_n = _cfg_int(self.config, "profit_core_n", 25)
        # Sim-time interval for VOLUME_BUDGET log (default 10 sim-min = 600 s).
        self.volume_log_interval_ns = _cfg_int(self.config, "volume_log_interval_ns", 600_000_000_000)
        # NEGATIVE_RECOVERY config params
        self.recovery_probe_size_mult = _cfg_float(self.config, "recovery_probe_size_mult", 0.25)
        self.recovery_probe_min_spread_ticks = _cfg_int(self.config, "recovery_probe_min_spread_ticks", 2)
        self.recovery_probe_cooldown_ns = _cfg_int(self.config, "recovery_probe_cooldown_ns", 60_000_000_000)
        self.recovery_probe_max_consecutive_neg = _cfg_int(self.config, "recovery_probe_max_consecutive_neg", 2)
        self.hard_negative_min_gap = _cfg_int(self.config, "hard_negative_min_gap", 2)
        self.hard_negative_toxic_threshold = _cfg_float(self.config, "hard_negative_toxic_threshold", 0.8)
        self.neg_summary_interval_ns = _cfg_int(self.config, "neg_summary_interval_ns", 600_000_000_000)
        # Adaptive intensity (shadow mode by default — no behavior change until intensity_enabled=True)
        self.intensity_shadow_mode = _cfg_bool(self.config, "intensity_shadow_mode", True)
        self.intensity_enabled = _cfg_bool(self.config, "intensity_enabled", False)
        self.intensity_log_interval_ns = _cfg_int(self.config, "intensity_log_interval_ns", 600_000_000_000)
        self.intensity_aggressive_threshold = _cfg_float(self.config, "intensity_aggressive_threshold", 0.70)
        self.intensity_normal_threshold = _cfg_float(self.config, "intensity_normal_threshold", 0.40)
        self.intensity_light_threshold = _cfg_float(self.config, "intensity_light_threshold", 0.15)
        self.participation_floor_stale_ns = _cfg_int(self.config, "participation_floor_stale_ns", 180_000_000_000)
        self.low_opportunity_aggressive_frac = _cfg_float(self.config, "low_opportunity_aggressive_frac", 0.10)
        self.intensity_qty_min_mult = _cfg_float(self.config, "intensity_qty_min_mult", 0.15)
        self.intensity_aggressive_qty_mult = _cfg_float(self.config, "intensity_aggressive_qty_mult", 1.15)
        self.intensity_light_qty_mult = _cfg_float(self.config, "intensity_light_qty_mult", 0.35)
        self.intensity_light_refresh_factor = _cfg_int(self.config, "intensity_light_refresh_factor", 3)
        self.intensity_floor_refresh_factor = _cfg_int(self.config, "intensity_floor_refresh_factor", 8)
        # DEFICIT floor_probe: rapid postOnly probes to build 3 obs before changeover penalty
        self.deficit_floor_probe_interval_ns = _cfg_int(self.config, "deficit_floor_probe_interval_ns", 60_000_000_000)
        self.deficit_floor_probe_size_mult = _cfg_float(self.config, "deficit_floor_probe_size_mult", 0.25)
        self.deficit_floor_probe_spread_ticks = _cfg_int(self.config, "deficit_floor_probe_spread_ticks", 1)
        self.deficit_recovery_log_interval_ns = _cfg_int(self.config, "deficit_recovery_log_interval_ns", 600_000_000_000)
        # Auto intensity risk gate: automatically adjusts aggressiveness based on portfolio risk state.
        # intensity_enabled=True is the master switch; risk gate controls the rest automatically.
        self.risk_gate_enabled = _cfg_bool(self.config, "risk_gate_enabled", True)
        self.risk_gate_net_realized_threshold = _cfg_float(self.config, "risk_gate_net_realized_threshold", -0.3)
        self.risk_gate_below_obs_threshold = _cfg_int(self.config, "risk_gate_below_obs_threshold", 30)
        self.risk_gate_slow_probe_factor = _cfg_int(self.config, "risk_gate_slow_probe_factor", 3)
        self.risk_gate_log_interval_ns = _cfg_int(self.config, "risk_gate_log_interval_ns", 600_000_000_000)
        # R8: when deficit ≤ threshold, halve neg recovery probe cooldown to build obs faster
        self.risk_gate_few_deficit_threshold = _cfg_int(self.config, "risk_gate_few_deficit_threshold", 10)
        self.risk_gate_neg_probe_faster_factor = _cfg_int(self.config, "risk_gate_neg_probe_faster_factor", 2)
        self.inactive_buffer_books = max(0, _cfg_int(self.config, "inactive_buffer_books", 10))
        self.book_selection_mode = str(
            getattr(self.config, "book_selection_mode", "dynamic")
        ).lower()
        self.book_rank_refresh_steps = max(
            1, _cfg_int(self.config, "book_rank_refresh_steps", 60)
        )
        self.selection_lob_depth = max(
            1, _cfg_int(self.config, "selection_lob_depth", 5)
        )
        self.dead_book_gap_ns = _cfg_int(
            self.config, "dead_book_gap_ns", 600_000_000_000
        )
        self.book_flow_lookback_ns = _cfg_int(
            self.config, "book_flow_lookback_ns", 10_800_000_000_000
        )
        self.sel_w_realized = _cfg_float(self.config, "sel_w_realized", 1.0)
        self.sel_w_market = _cfg_float(self.config, "sel_w_market", 1.0)
        self.sel_w_passive = _cfg_float(self.config, "sel_w_passive", 1.0)
        self.sel_w_toxic = _cfg_float(self.config, "sel_w_toxic", 1.0)
        self.sel_w_inventory = _cfg_float(self.config, "sel_w_inventory", 1.0)
        self.sel_w_fee = _cfg_float(self.config, "sel_w_fee", 1.0)
        self.sel_w_activity = _cfg_float(self.config, "sel_w_activity", 1.0)
        self.order_qty = _cfg_float(self.config, "order_qty", 0.25)
        # gamma / imbalance_scale: max whole-tick skew at |q_norm|=1 / |imb|=1
        self.gamma = _cfg_float(self.config, "gamma", 0.5)
        self.imbalance_depth = _cfg_int(self.config, "imbalance_depth", 5)
        self.imbalance_scale = _cfg_float(self.config, "imbalance_scale", 1.0)
        self.max_skew_ticks = _cfg_int(self.config, "max_skew_ticks", 2)
        self.edge_ticks = _cfg_int(self.config, "edge_ticks", 1)
        self.min_spread_ticks = _cfg_int(self.config, "min_spread_ticks", 1)
        self.expiry_period = _cfg_int(self.config, "expiry_period", 6_000_000_000)
        self.max_maker_fee_rate = _cfg_float(self.config, "max_maker_fee_rate", 0.005)
        self.volume_throttle_ratio = _cfg_float(self.config, "volume_throttle_ratio", 0.90)
        self.volume_hard_stop_ratio = _cfg_float(self.config, "volume_hard_stop_ratio", 0.92)
        self.capital_turnover_cap = _cfg_float(self.config, "capital_turnover_cap", 10.0)
        self.inventory_soft_ratio = _cfg_float(self.config, "inventory_soft_ratio", 0.40)
        self.position_frac = _cfg_float(self.config, "position_frac", 0.015)
        self.quote_interval = max(1, _cfg_int(self.config, "quote_interval", 2))
        self.stuck_edge_ramp_ns = _cfg_int(self.config, "stuck_edge_ramp_ns", 30_000_000_000)
        self.reprice_ticks = _cfg_int(self.config, "reprice_ticks", 1)
        self.max_instructions_per_book = _cfg_int(self.config, "max_instructions_per_book", 5)

        # RoundTripGuard — matches scoring.kappa defaults (validator config).
        self.min_realized_observations = _cfg_int(self.config, "min_realized_observations", 3)
        self.kappa_lookback_ns = _cfg_int(self.config, "kappa_lookback_ns", 10_800_000_000_000)
        self.changeover_grace_period_ns = _cfg_int(self.config, "changeover_grace_period_ns", 600_000_000_000)
        self.loss_budget_frac = _cfg_float(self.config, "loss_budget_frac", 0.001)
        # Round-trip exit state machine (per FIFO lot, fee-inclusive).
        self.target_profit_ticks = _cfg_int(self.config, "target_profit_ticks", 1)
        self.roundtrip_spread_buffer_ticks = _cfg_int(self.config, "roundtrip_spread_buffer_ticks", 1)
        self.max_holding_time_ns = _cfg_int(self.config, "max_holding_time_ns", 600_000_000_000)
        self.adverse_move_ticks = _cfg_int(self.config, "adverse_move_ticks", 2)
        self.force_unwind_loss_budget_mult = _cfg_float(self.config, "force_unwind_loss_budget_mult", 5.0)
        self.force_unwind_age_mult = _cfg_float(self.config, "force_unwind_age_mult", 2.0)
        self.unwind_min_qty = _cfg_float(self.config, "unwind_min_qty", 0.01)
        self.unwind_cancel_first = _cfg_bool(self.config, "unwind_cancel_first", True)
        self.max_unwind_qty_per_slice = _cfg_float(self.config, "max_unwind_qty_per_slice", self.order_qty)
        self.global_drain_old_age_ns = _cfg_int(self.config, "global_drain_old_age_ns", 900_000_000_000)
        self.global_drain_min_books = _cfg_int(self.config, "global_drain_min_books", 8)
        self.max_forced_unwind_books_per_tick = _cfg_int(self.config, "max_forced_unwind_books_per_tick", 10)

        # Production / endpoint controls.
        self.production_mode = _cfg_bool(self.config, "production_mode", True)
        self.stats_enabled = _cfg_bool(self.config, "stats_enabled", False)

        # Logging verbosity controls.
        self.log_detail = _cfg_bool(self.config, "log_detail", False)
        self.log_summary_interval_ns = _cfg_int(self.config, "log_summary_interval_ns", 60_000_000_000)
        self.log_max_books = _cfg_int(self.config, "log_max_books", 10)

        # Instruction pulse (optional): cancel-only by default — never bypass toxic/throttle.
        self.instruction_pulse_enabled = _cfg_bool(self.config, "instruction_pulse_enabled", False)
        self.instruction_pulse_cancel_only = _cfg_bool(
            self.config, "instruction_pulse_cancel_only", True
        )
        self.touch_max_gap_ns = _cfg_int(self.config, "touch_max_gap_ns", 30_000_000_000)

        self.fifo_drift_tolerance_ratio = _cfg_float(
            self.config, "fifo_drift_tolerance_ratio", 0.05
        )
        # Drift quarantine: abs(drift) > max(large_drift_abs_threshold, large_drift_tol_mult * tol)
        # OR fifo_net * account_net < 0 with abs(drift) > tol → quarantine + clear.
        self.large_drift_abs_threshold = _cfg_float(
            self.config, "large_drift_abs_threshold", 20.0 * self.order_qty
        )  # absolute floor; default ≈ 5 lots for 0.25-lot books
        self.large_drift_tol_mult = _cfg_float(
            self.config, "large_drift_tol_mult", 100.0
        )  # multiplier of tol; default 100 × tol ≈ 1.25 for 0.25-lot books
        self.drift_quarantine_ns = _cfg_int(
            self.config, "drift_quarantine_ns", 300_000_000_000
        )  # 5 min max quarantine lifetime
        self.drift_warn_rate_limit_ns = _cfg_int(
            self.config, "drift_warn_rate_limit_ns", 60_000_000_000
        )  # 60 s per-book warning rate limit
        self.drift_stable_ticks_required = max(
            1, _cfg_int(self.config, "drift_stable_ticks_required", 3)
        )  # consecutive stable ticks needed to release quarantine / severe
        self.min_order_size = _cfg_float(self.config, "min_order_size", self.order_qty)

        # Maker markout toxic flow (sim-step horizon, not wall clock).
        self.markout_horizon_steps = max(1, _cfg_int(self.config, "markout_horizon_steps", 3))
        self.toxic_markout_ticks = _cfg_int(self.config, "toxic_markout_ticks", 1)
        self.toxic_ema_alpha = _cfg_float(self.config, "toxic_ema_alpha", 0.3)
        self.toxic_entry_block = _cfg_float(self.config, "toxic_entry_block", 0.6)
        self.toxic_cooldown_steps = max(1, _cfg_int(self.config, "toxic_cooldown_steps", 5))
        self.toxic_edge_scale = _cfg_float(self.config, "toxic_edge_scale", 0.8)
        self.toxic_skew_ticks = _cfg_int(self.config, "toxic_skew_ticks", 1)
        self.toxic_min_qty_scale = _cfg_float(self.config, "toxic_min_qty_scale", 0.35)
        self.toxic_decay_half_life_steps = max(
            1, _cfg_int(self.config, "toxic_decay_half_life_steps", 20)
        )

        # Per-validator book tiers (validator hotkey → BookSelection).
        self._book_selection: dict[str, BookSelection] = {}
        self._last_book_rank_ts: dict[str, int] = {}
        self._pending_dropped_books: dict[str, set[int]] = {}
        # Per-validator state (validator hotkey → book_id → …).
        self._book_state: dict[str, dict[int, BookState]] = {}
        self._realized_obs: dict[str, dict[int, RealizedObsState]] = {}
        self._initial_base: dict[str, dict[int, float]] = {}
        self._last_quote_refresh_ts: dict[str, int] = {}
        self._pending_markouts: dict[str, deque[PendingMarkout]] = {}
        self._book_flow: dict[str, dict[int, BookFlowState]] = {}
        self._book_mode_cache: dict[str, dict[int, str]] = {}

        # Dashboard stats tracking (validator hotkey → book_id → accumulated value).
        self._maker_volume: dict[str, dict[int, float]] = {}
        self._taker_volume: dict[str, dict[int, float]] = {}
        self._roundtrip_volume: dict[str, dict[int, float]] = {}
        self._sim_ts: int = 0
        self._last_summary_ts: dict[str, int] = {}
        self._last_volume_log_ts: dict[str, int] = {}
        self._neg_probed_books: dict[str, set] = {}
        self._neg_probe_counts: dict[str, dict[int, int]] = {}
        self._neg_summary_buf: dict[str, dict] = {}
        self._last_neg_summary_ts: dict[str, int] = {}
        self._intensity_summary_buf: dict[str, dict] = {}
        self._last_intensity_summary_ts: dict[str, int] = {}
        self._book_intensity_last: dict[str, dict[int, str]] = {}  # per-book current intensity class
        self._intensity_interval_data: dict[str, dict] = {}  # per-interval unique-book sets
        self._intensity_qualify_counts: dict[str, int] = {}  # consecutive qualify ticks per validator
        self._unwind_log_buffer: list[tuple] = []  # (book_id, side, qty, inv, age_s) — cleared per tick
        self._book_count: int = 128
        # DEFICIT_RECOVERY tracking
        self._deficit_recovery_buf: dict[str, dict] = {}
        self._last_deficit_recovery_ts: dict[str, int] = {}
        # Auto risk gate cross-tick state
        self._risk_gate_prev_forced_unwind: dict[str, int] = {}
        self._last_risk_summary_ts: dict[str, int] = {}

        # Per-book fill / order counters (reset on restart, not on validator reset).
        self._bid_fill_count: dict[str, dict[int, int]] = {}
        self._ask_fill_count: dict[str, dict[int, int]] = {}
        self._short_exit_fill_count: dict[str, dict[int, int]] = {}
        self._short_exit_order_count: dict[str, dict[int, int]] = {}
        self._log_step: dict[str, int] = {}

        # Drift quarantine state (FIFO/account structural mismatch).
        self._drift_quarantine: dict[str, dict[int, int]] = {}   # {vkey: {book_id: quarantine_start_ts}}
        self._severe_drift_books: dict[str, set[int]] = {}       # {vkey: {book_id}} — swap pattern or opposite-sign
        self._drift_warn_ts: dict[str, dict[int, int]] = {}      # {vkey: {book_id: last_warn_ts}}
        self._drift_summary_buf: dict[str, dict] = {}            # accumulated per tick for summary log
        self._last_drift_summary_ts: dict[str, int] = {}         # {vkey: ts}
        self._first_tick_done: set[str] = set()                  # validators seen at least once
        self._pending_baseline_reconcile: set[str] = set()       # pending 1-shot reconcile
        # Swap-pattern detection: remember prev-tick fifo_net / account_net per book.
        self._prev_fifo_net: dict[str, dict[int, float]] = {}      # {vkey: {book_id: fifo_net}}
        self._prev_account_net: dict[str, dict[int, float]] = {}   # {vkey: {book_id: account_net}}
        # Stable-tick counter: consecutive ticks with drift <= tol (for quarantine release).
        self._drift_stable_ticks: dict[str, dict[int, int]] = {}   # {vkey: {book_id: count}}
        # Tracks which quarantine books had ALL orders cancelled on first quarantine entry.
        self._quarantine_all_cancelled: dict[str, set[int]] = {}   # {vkey: {book_id}}

        if self.stats_enabled:
            self.router.add_api_route("/stats", self.get_stats, methods=["GET"])
        if not self.production_mode:
            self.router.add_api_route("/dashboard", self.get_dashboard, methods=["GET"])
        self._memsnap_tick = 0

    def _bind_request_context_if_needed(
        self, state: MarketSimulationStateUpdate
    ) -> tuple[object, object] | None:
        """Bind per-request context when handle() was bypassed (tests); no-op if already set."""
        if _validator_ctx.get() is not None:
            return None
        if not getattr(self, "_warned_handle_bypass", False):
            self._warned_handle_bypass = True
            bt.logging.warning(
                "ISFMM: update/respond entered without handle() — binding local context "
                "(miner/proxy should call POST /handle)"
            )
        vkey = self._validator_key(state=state)
        tok_v = _validator_ctx.set(vkey if vkey != _FALLBACK_VALIDATOR else None)
        tok_ts = _sim_ts_ctx.set(state.timestamp)
        return (tok_v, tok_ts)

    @staticmethod
    def _release_request_context(tokens: tuple[object, object] | None) -> None:
        if tokens is None:
            return
        tok_v, tok_ts = tokens
        _validator_ctx.reset(tok_v)
        _sim_ts_ctx.reset(tok_ts)

    def handle(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """Entry for miner forward and proxy POST /handle; binds isolated request context."""
        vkey = self._validator_key(state=state)
        tok_v = _validator_ctx.set(vkey if vkey != _FALLBACK_VALIDATOR else None)
        tok_ts = _sim_ts_ctx.set(state.timestamp)
        if not getattr(self, "_logged_handle_entry", False):
            self._logged_handle_entry = True
            label = vkey[:16] if vkey != _FALLBACK_VALIDATOR else "?"
            bt.logging.debug(f"ISFMM handle() entry: request context bound (validator={label})")
        try:
            return super().handle(state)
        finally:
            _validator_ctx.reset(tok_v)
            _sim_ts_ctx.reset(tok_ts)

    def _reset_all_validators(self) -> None:
        self._book_selection.clear()
        self._last_book_rank_ts.clear()
        self._pending_dropped_books.clear()
        self._book_flow.clear()
        self._book_mode_cache.clear()
        self._book_state.clear()
        self._realized_obs.clear()
        self._initial_base.clear()
        self._last_quote_refresh_ts.clear()
        self._pending_markouts.clear()
        self._neg_probed_books.clear()
        self._neg_probe_counts.clear()
        self._neg_summary_buf.clear()
        self._last_neg_summary_ts.clear()
        self._intensity_summary_buf.clear()
        self._last_intensity_summary_ts.clear()
        self._book_intensity_last.clear()
        self._intensity_interval_data.clear()
        self._intensity_qualify_counts.clear()
        self._risk_gate_prev_forced_unwind.clear()
        self._last_risk_summary_ts.clear()

    def _reset_validator_state(
        self,
        validator_key: str,
        state: MarketSimulationStateUpdate | None = None,
        *,
        book_ids: list[int] | None = None,
    ) -> None:
        """Clear validator-scoped FIFO/book state and re-seed inventory baselines from account."""
        cfg = state.config if state is not None else getattr(self, "simulation_config", None)
        accounts: dict[int, Account] = {}
        if state is not None and state.accounts and self.uid in state.accounts:
            accounts = state.accounts[self.uid]
        elif getattr(self, "accounts", None):
            accounts = self.accounts

        if book_ids is None:
            book_ids = list(accounts.keys()) if accounts else list(
                self._realized_obs.get(validator_key, {}).keys()
            )

        self._realized_obs[validator_key] = {}
        self._book_state.setdefault(validator_key, {}).clear()
        self._last_quote_refresh_ts.pop(validator_key, None)
        self._book_selection.pop(validator_key, None)
        self._last_book_rank_ts.pop(validator_key, None)
        self._pending_dropped_books.pop(validator_key, None)
        self._pending_markouts.pop(validator_key, None)
        self._book_flow.pop(validator_key, None)
        self._book_mode_cache.pop(validator_key, None)
        # Drift quarantine reset: clear all drift state and trigger 1-shot baseline reconcile.
        self._drift_quarantine.pop(validator_key, None)
        self._severe_drift_books.pop(validator_key, None)
        self._drift_warn_ts.pop(validator_key, None)
        self._drift_summary_buf.pop(validator_key, None)
        self._last_drift_summary_ts.pop(validator_key, None)
        self._prev_fifo_net.pop(validator_key, None)
        self._prev_account_net.pop(validator_key, None)
        self._drift_stable_ticks.pop(validator_key, None)
        self._quarantine_all_cancelled.pop(validator_key, None)
        self._neg_probed_books.pop(validator_key, None)
        self._neg_probe_counts.pop(validator_key, None)
        self._neg_summary_buf.pop(validator_key, None)
        self._last_neg_summary_ts.pop(validator_key, None)
        self._intensity_summary_buf.pop(validator_key, None)
        self._last_intensity_summary_ts.pop(validator_key, None)
        self._book_intensity_last.pop(validator_key, None)
        self._intensity_interval_data.pop(validator_key, None)
        self._risk_gate_prev_forced_unwind.pop(validator_key, None)
        self._last_risk_summary_ts.pop(validator_key, None)
        self._first_tick_done.discard(validator_key)
        self._pending_baseline_reconcile.add(validator_key)

        initial = self._initial_base.setdefault(validator_key, {})
        expected = self._config_initial_own_base(cfg)
        for bid in book_ids:
            if accounts and bid in accounts:
                own = self._account_own_base(accounts[bid])
                initial[bid] = expected if expected is not None else own
            elif expected is not None:
                initial[bid] = expected
            else:
                initial.pop(bid, None)

    def onStart(self, event: SimulationStartEvent) -> None:
        self._reset_all_validators()

    def _flow(self, validator_key: str, book_id: int) -> BookFlowState:
        return self._book_flow.setdefault(validator_key, {}).setdefault(
            book_id, BookFlowState()
        )

    def _ingest_market_flow(
        self, validator_key: str, state: MarketSimulationStateUpdate
    ) -> None:
        """Accumulate per-book market trades from Book.events (incremental per tick)."""
        if not state.books:
            return
        sim_ts = state.timestamp
        lookback = self.book_flow_lookback_ns
        for book_id, book in state.books.items():
            flow = self._flow(validator_key, book_id)
            flow.prune(sim_ts, lookback)
            events = book.events
            if not events:
                continue
            for evt in events:
                if getattr(evt, "type", None) != "t":
                    continue
                trade_id = getattr(evt, "id", None)
                if trade_id is None:
                    continue
                flow.add(
                    int(trade_id),
                    int(evt.timestamp),
                    float(evt.quantity),
                    lookback,
                    sim_ts,
                )

    def report(self, state: MarketSimulationStateUpdate, response: FinanceAgentResponse) -> None:
        if not self.production_mode:
            super().report(state, response)

    def update(self, state: MarketSimulationStateUpdate) -> None:
        ctx_tokens = self._bind_request_context_if_needed(state)
        try:
            validator_key = self._set_validator_context(state)
            self._ingest_market_flow(validator_key, state)
            notices = state.notices.get(self.uid, []) if state.notices else []
            for event in notices:
                etype = getattr(event, "type", None)
                if etype in ("RESET_AGENTS", "RA"):
                    self._reset_validator_state(validator_key, state)
                elif etype in (
                    "RDRA",
                    "ERDRA",
                    "RESPONSE_DISTRIBUTED_RESET_AGENT",
                    "ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT",
                ):
                    if self._reset_event_applies_to_self(event):
                        self._reset_validator_state(validator_key, state)
            self._sim_ts = state.timestamp
            cfg = state.config
            if cfg:
                self._book_count = int(getattr(cfg, "book_count", self._book_count))
            # Replace base class update() to avoid its O(books×events) logging loop.
            # Set the fields the rest of the code depends on, then dispatch events in
            # a single O(events) pass instead of O(books × events).
            self.simulation_config = cfg
            self.accounts = state.accounts.get(self.uid, {})
            self.events = state.notices.get(self.uid, [])
            _hotkey = getattr(getattr(state, "dendrite", None), "hotkey", None)
            for _ev in self.events:
                _etype = getattr(_ev, "type", None)
                if _etype in ("EVENT_TRADE", "ET"):
                    self.onTrade(_ev, _hotkey)
        finally:
            self._release_request_context(ctx_tokens)

    def _reset_event_applies_to_self(self, event: ResetAgentsEvent | object) -> bool:
        resets = getattr(event, "resets", None)
        if resets:
            for reset in resets:
                if (
                    getattr(reset, "agentId", None) == self.uid
                    and getattr(reset, "success", False)
                ):
                    return True
            return False
        agent_id = getattr(event, "agentId", None)
        return agent_id is None or agent_id == self.uid

    def _validator_key(
        self,
        validator: str | None = None,
        state: MarketSimulationStateUpdate | None = None,
    ) -> str:
        if validator:
            return validator
        if state is not None:
            dendrite = getattr(state, "dendrite", None)
            hotkey = getattr(dendrite, "hotkey", None) if dendrite is not None else None
            if hotkey:
                return hotkey
        ctx_key = _validator_ctx.get()
        if ctx_key:
            return ctx_key
        return _FALLBACK_VALIDATOR

    def _set_validator_context(self, state: MarketSimulationStateUpdate) -> str:
        return self._validator_key(state=state)

    def _observation_bucket_ts(self, event_ts: int) -> int:
        """Match validator realized_pnl_history key (state.timestamp per tick)."""
        ctx_ts = _sim_ts_ctx.get()
        if ctx_ts is not None:
            return ctx_ts
        pub = self._publish_interval_ns()
        return (event_ts // pub) * pub

    def _obs(self, validator_key: str, book_id: int) -> RealizedObsState:
        return self._realized_obs.setdefault(validator_key, {}).setdefault(
            book_id, RealizedObsState()
        )

    def _book_st(self, validator_key: str, book_id: int) -> BookState:
        return self._book_state.setdefault(validator_key, {}).setdefault(book_id, BookState())

    @staticmethod
    def _balance_field(balance: object, field: str, default: float = 0.0) -> float:
        if balance is None:
            return default
        val = getattr(balance, field, None)
        if val is not None:
            return float(val)
        if isinstance(balance, dict):
            aliases = {
                "free": ("free", "f"),
                "total": ("total", "t"),
                "reserved": ("reserved", "r"),
            }
            for key in aliases.get(field, (field,)):
                if key in balance and balance[key] is not None:
                    return float(balance[key])
        return default

    def _account_own_base(self, account: Account) -> float:
        own = getattr(account, "own_base", None)
        if own is not None:
            return float(own)
        total = self._balance_field(getattr(account, "base_balance", None), "total")
        loan = float(getattr(account, "base_loan", 0.0) or 0.0)
        collateral = float(getattr(account, "base_collateral", 0.0) or 0.0)
        return total - loan + collateral

    def _account_free_base(self, account: Account) -> float:
        bb = getattr(account, "base_balance", None)
        free = self._balance_field(bb, "free")
        if free > 0.0 or bb is not None:
            return free
        return max(0.0, self._account_own_base(account))

    @staticmethod
    def _config_initial_own_base(cfg: object | None) -> float | None:
        if cfg is None:
            return None
        val = getattr(cfg, "miner_base_balance", None)
        if val is None:
            return None
        return float(val)

    def _account_free_quote(self, account: Account) -> float:
        own = getattr(account, "own_quote", None)
        if own is not None:
            return float(own)
        qb = getattr(account, "quote_balance", None)
        free = self._balance_field(qb, "free")
        if free > 0.0 or qb is not None:
            return free
        total = self._balance_field(qb, "total")
        loan = float(getattr(account, "quote_loan", 0.0) or 0.0)
        collateral = float(getattr(account, "quote_collateral", 0.0) or 0.0)
        return total - loan + collateral

    def _net_inventory(
        self,
        validator_key: str,
        book_id: int,
        account: Account,
        cfg: object | None = None,
    ) -> float:
        """Signed base position vs baseline (+ long, − short). Account is source of truth."""
        own = self._account_own_base(account)
        initial = self._initial_base.setdefault(validator_key, {})
        if book_id not in initial:
            expected = self._config_initial_own_base(cfg)
            if expected is None and cfg is None:
                expected = self._config_initial_own_base(getattr(self, "simulation_config", None))
            initial[book_id] = expected if expected is not None else own
        return own - initial[book_id]

    def _fifo_drift_tolerance(self, volume_decimals: int) -> float:
        return max(
            _round_qty(self.order_qty * self.fifo_drift_tolerance_ratio, volume_decimals),
            10.0 ** (-volume_decimals),
        )

    @staticmethod
    def _trim_fifo_tail(
        lots: deque[FifoLot],
        amount: float,
        volume_decimals: int,
    ) -> float:
        """Reduce qty from the newest open lot; return unconsumed trim amount."""
        remaining = amount
        while remaining > 1e-12 and lots:
            ts, qty, px, fee = lots[-1]
            if qty <= remaining + 1e-12:
                remaining -= qty
                lots.pop()
            else:
                new_qty = _round_qty(qty - remaining, volume_decimals)
                fee_scale = new_qty / qty if qty > 0 else 0.0
                lots[-1] = (ts, new_qty, px, fee * fee_scale)
                remaining = 0.0
        return remaining

    def _adjust_fifo_tail_lot(
        self,
        obs: RealizedObsState,
        signed_drift: float,
        volume_decimals: int,
    ) -> bool:
        """
        Nudge FIFO toward account_net without clearing front lots.

        signed_drift = fifo_net − account_net.
        Primary: trim the same-side tail lot.
        Fallback: if same side is empty/insufficient, extend the opposite side with
        a synthetic lot — so fifo_net always converges to account_net in one step.
        """
        if signed_drift > 0:
            # fifo too long: trim longs; if insufficient, extend shorts
            leftover = self._trim_fifo_tail(obs.longs, signed_drift, volume_decimals)
            if leftover > 1e-12:
                obs.shorts.append((0, round(leftover, volume_decimals), 0.0, 0.0))
        else:
            # fifo too short: trim shorts; if insufficient, extend longs
            leftover = self._trim_fifo_tail(obs.shorts, abs(signed_drift), volume_decimals)
            if leftover > 1e-12:
                obs.longs.append((0, round(leftover, volume_decimals), 0.0, 0.0))
        return True

    def _reconcile_fifo_drift(
        self,
        validator_key: str,
        book_id: int,
        obs: RealizedObsState,
        account_net: float,
        volume_decimals: int,
    ) -> bool:
        """
        Reconcile FIFO vs account.

        Quarantine triggers:
          (a) abs(drift) > max(large_drift_abs_threshold, large_drift_tol_mult × tol)
          (b) fifo_net × account_net < 0 AND abs(drift) > tol   [opposite-sign = swap risk]
        On quarantine: clear FIFO lots; NO synthetic lot creation of any kind.
          Synthetic lot would cause swap oscillation: fifo reseeded to 92 → account fills flat
          → next tick fifo=92/account=0 → roles reversed endlessly.
        Severe = opposite-sign OR swap_oscillation pattern detected.
        Release: drift <= tol for drift_stable_ticks_required consecutive ticks,
                 account flat, no open orders.  Checked in _process_book_mm.
        All logs rate-limited; per-tick summary only.
        """
        fifo_net = obs.fifo_net_qty()
        signed_drift = fifo_net - account_net
        abs_drift = abs(signed_drift)
        tol = self._fifo_drift_tolerance(volume_decimals)
        now_ns = self._sim_ts

        # Update prev-tick snapshots (used for swap detection next tick).
        pf = self._prev_fifo_net.setdefault(validator_key, {})
        pa = self._prev_account_net.setdefault(validator_key, {})
        prev_fifo = pf.get(book_id)
        prev_acc = pa.get(book_id)
        pf[book_id] = fifo_net
        pa[book_id] = account_net

        stab = self._drift_stable_ticks.setdefault(validator_key, {})

        if abs_drift <= tol:
            # Stable tick: increment counter only.
            # Actual release (quarantine + severe) happens in _process_book_mm
            # where account.orders is available for the open-orders check.
            stab[book_id] = stab.get(book_id, 0) + 1
            return False

        # Any drift above tol resets stable counter.
        stab[book_id] = 0

        if abs_drift <= tol * 2.0:
            bt.logging.debug(
                f"FIFO drift (log only) v={validator_key[:12]} book={book_id}: "
                f"fifo_net={fifo_net:.6f} account_net={account_net:.6f} "
                f"drift={signed_drift:.6f} tol={tol:.6f}"
            )
            return False

        # --- Quarantine trigger evaluation ---
        large_thresh = max(self.large_drift_abs_threshold, self.large_drift_tol_mult * tol)
        is_large = abs_drift > large_thresh

        # Opposite-sign: FIFO long while account short (or vice-versa).
        is_opposite = (
            (fifo_net > tol and account_net < -tol)
            or (fifo_net < -tol and account_net > tol)
        )
        is_quarantine_trigger = is_large or (is_opposite and abs_drift > tol)

        # Swap pattern: prev_fifo ≈ current_account AND prev_account ≈ current_fifo.
        swap_tol = max(tol * 5, tol)
        swap_oscillation = (
            prev_fifo is not None and prev_acc is not None
            and abs(prev_fifo - account_net) <= swap_tol
            and abs(prev_acc - fifo_net) <= swap_tol
            and abs_drift > tol
        )
        is_severe = is_opposite or swap_oscillation

        if is_quarantine_trigger:
            # Accumulate into per-tick summary buffer — no per-book WARNING spam.
            sbuf = self._drift_summary_buf.setdefault(validator_key, {
                "affected": set(), "max_drift": 0.0, "severe": 0
            })
            sbuf["affected"].add(book_id)
            sbuf["max_drift"] = max(sbuf["max_drift"], abs_drift)
            if is_severe:
                sbuf["severe"] += 1

            # Rate-limited per-book WARNING.
            warn_ts = self._drift_warn_ts.setdefault(validator_key, {})
            if now_ns - warn_ts.get(book_id, 0) >= self.drift_warn_rate_limit_ns:
                warn_ts[book_id] = now_ns
                bt.logging.warning(
                    f"[DRIFT_QUARANTINE] v={validator_key[:12]} book={book_id}"
                    f" fifo_net={fifo_net:.6f} account_net={account_net:.6f}"
                    f" drift={signed_drift:.6f} severe={is_severe} swap_oscillation={swap_oscillation}"
                )

            # Clear FIFO lots — structural mismatch, data not trustworthy.
            # NO synthetic lot creation: reseeding fifo to account_net causes swap oscillation
            # (account fills flat next tick → fifo holds stale synthetic value → roles reversed).
            obs.longs.clear()
            obs.shorts.clear()

            if is_severe:
                # Opposite-sign or swap_oscillation: mark for flat-first-only exit path.
                self._severe_drift_books.setdefault(validator_key, set()).add(book_id)

            # Enter / refresh quarantine regardless of severe flag.
            self._drift_quarantine.setdefault(validator_key, {})[book_id] = now_ns
            return True

        # Non-quarantine drift: tail-adjust (normal async fill ordering skew).
        self._adjust_fifo_tail_lot(obs, signed_drift, volume_decimals)
        tag = f"v={validator_key[:12]} book={book_id}"
        detail = (
            f"fifo_net={fifo_net:.6f} account_net={account_net:.6f} "
            f"drift={signed_drift:.6f} tol={tol:.6f}"
        )
        if abs_drift <= tol * 20:
            bt.logging.debug(f"FIFO drift adjusted {tag}: {detail}")
        elif abs_drift <= tol * 100:
            bt.logging.info(f"FIFO drift adjusted {tag}: {detail}")
        else:
            # tol*100 < drift ≤ large_thresh: rate-limited warning only.
            warn_ts = self._drift_warn_ts.setdefault(validator_key, {})
            if now_ns - warn_ts.get(book_id, 0) >= self.drift_warn_rate_limit_ns:
                warn_ts[book_id] = now_ns
                bt.logging.warning(f"FIFO drift (medium) adjusted {tag}: {detail}")
        return True

    def _is_drift_quarantined(self, validator_key: str, book_id: int) -> bool:
        """True if book is in drift quarantine and timeout has not elapsed."""
        qmap = self._drift_quarantine.get(validator_key)
        if not qmap or book_id not in qmap:
            return False
        if self._sim_ts - qmap[book_id] > self.drift_quarantine_ns:
            del qmap[book_id]
            return False
        return True

    def _is_severe_drift(self, validator_key: str, book_id: int) -> bool:
        """True if book has severe drift (opposite-direction fifo/account) — flat-first exit only."""
        return book_id in self._severe_drift_books.get(validator_key, set())

    def _run_baseline_reconcile(self, validator_key: str, cfg: object) -> None:
        """1-shot full FIFO/account reconcile after reset/RDRA/restart. Drift books → quarantine."""
        vol_dec = int(getattr(cfg, "volumeDecimals", 8))
        for bid, account in self.accounts.items():
            if account is None:
                continue
            obs = self._obs(validator_key, bid)
            net = self._net_inventory(validator_key, bid, account, cfg)
            self._reconcile_fifo_drift(validator_key, bid, obs, net, vol_dec)
        n_q = len(self._drift_quarantine.get(validator_key, {}))
        n_s = len(self._severe_drift_books.get(validator_key, set()))
        bt.logging.info(
            f"[DRIFT_BASELINE] v={validator_key[:12]} quarantined={n_q} severe={n_s}"
        )

    @staticmethod
    def _throttle_qty_scale(
        volume_traded: float,
        throttle_cap: float,
        hard_stop_cap: float,
    ) -> float:
        if volume_traded < throttle_cap:
            return 1.0
        span = max(hard_stop_cap - throttle_cap, 1.0)
        scale = 1.0 - (volume_traded - throttle_cap) / span
        return max(0.0, min(1.0, scale))

    def _sync_account_inventory(
        self,
        validator_key: str,
        book_id: int,
        account: Account,
        st: BookState,
        cfg: object,
        volume_decimals: int,
        sim_ts: int = 0,
    ) -> float:
        net = self._net_inventory(validator_key, book_id, account, cfg)
        obs = self._obs(validator_key, book_id)
        self._reconcile_fifo_drift(validator_key, book_id, obs, net, volume_decimals)
        st.inventory = net
        # If FIFO was reseeded (inherited position, no open_ts) set open_ts so
        # exit_only and GLOBAL_DRAIN can track and unwind the position.
        if st.inventory_open_ts == 0 and abs(net) >= self._fifo_drift_tolerance(volume_decimals) and sim_ts > 0:
            st.inventory_open_ts = sim_ts
        return net

    def onTrade(self, event: TradeEvent, validator: str | None = None) -> None:
        book_id = event.bookId
        if book_id is None:
            return

        if event.takerAgentId == self.uid:
            is_buy = event.side == OrderDirection.BUY
            fee = event.takerFee
        elif event.makerAgentId == self.uid:
            is_buy = event.side == OrderDirection.SELL
            fee = event.makerFee
        else:
            return

        vkey = self._validator_key(validator)
        if validator:
            if not getattr(self, "_logged_ontrade_hotkey", False):
                self._logged_ontrade_hotkey = True
                bt.logging.debug(
                    f"onTrade: framework passed validator={validator[:16]}"
                )
        elif vkey == _FALLBACK_VALIDATOR:
            if not getattr(self, "_warned_default_validator", False):
                self._warned_default_validator = True
                bt.logging.error(
                    "onTrade: Kappa observation mis-bucket risk — validator arg "
                    "missing and no request context; FIFO/markout under _default "
                    "(confirm handle() is the entry point)"
                )
        elif not getattr(self, "_logged_ctx_fallback", False):
            self._logged_ctx_fallback = True
            bt.logging.debug(
                f"onTrade: validator arg missing; used request context {vkey[:16]}"
            )
        obs = self._obs(vkey, book_id)
        had_shorts = bool(obs.shorts)
        realized, rt_vol = _fifo_match(
            obs.longs, obs.shorts, is_buy, event.quantity, event.price, fee, event.timestamp
        )
        if abs(realized) > 1e-12:
            sim_cfg = getattr(self, "simulation_config", None)
            vol_dec = int(getattr(sim_cfg, "volumeDecimals", 8)) if sim_cfg else 8
            obs.add_observation(
                self._observation_bucket_ts(event.timestamp),
                realized,
                decimals=vol_dec,
            )

        fill_quote = event.quantity * event.price
        if event.makerAgentId == self.uid:
            bk = self._maker_volume.setdefault(vkey, {})
            bk[book_id] = bk.get(book_id, 0.0) + fill_quote
        else:
            bk = self._taker_volume.setdefault(vkey, {})
            bk[book_id] = bk.get(book_id, 0.0) + fill_quote

        if rt_vol > 0.0:
            rt_quote = rt_vol * event.price
            bk2 = self._roundtrip_volume.setdefault(vkey, {})
            bk2[book_id] = bk2.get(book_id, 0.0) + rt_quote

        if event.makerAgentId == self.uid:
            # bid fill = maker bought (taker sold, event.side==SELL → is_buy=True)
            # ask fill = maker sold  (taker bought, event.side==BUY  → is_buy=False)
            if is_buy:
                bk2 = self._bid_fill_count.setdefault(vkey, {})
                bk2[book_id] = bk2.get(book_id, 0) + 1
                if had_shorts and rt_vol > 0.0:
                    bk3 = self._short_exit_fill_count.setdefault(vkey, {})
                    bk3[book_id] = bk3.get(book_id, 0) + 1
            else:
                bk2 = self._ask_fill_count.setdefault(vkey, {})
                bk2[book_id] = bk2.get(book_id, 0) + 1

        st = self._book_st(vkey, book_id)
        st.last_trade_ts = event.timestamp

        if event.makerAgentId == self.uid:
            self._record_maker_markout(vkey, book_id, event)

    def _publish_interval_ns(self, cfg: object | None = None) -> int:
        if cfg is not None:
            return int(getattr(cfg, "publish_interval", 1_000_000_000))
        sim_cfg = getattr(self, "simulation_config", None)
        if sim_cfg is not None:
            return int(getattr(sim_cfg, "publish_interval", 1_000_000_000))
        return 1_000_000_000

    def _record_maker_markout(
        self, validator_key: str, book_id: int, event: TradeEvent
    ) -> None:
        """Queue maker fill for delayed mid markout (toxic flow scoring)."""
        pub = self._publish_interval_ns()
        due_ts = event.timestamp + self.markout_horizon_steps * pub
        # Taker BUY → our ask fill; taker SELL → our bid fill.
        maker_side = (
            OrderDirection.SELL
            if event.side == OrderDirection.BUY
            else OrderDirection.BUY
        )
        self._pending_markouts.setdefault(validator_key, deque()).append(
            (book_id, maker_side, event.price, event.timestamp, due_ts)
        )

    def _markout_ticks(
        self, maker_side: int, fill_px: float, mid: float, tick: float
    ) -> float:
        if tick <= 0.0:
            return 0.0
        if maker_side == OrderDirection.BUY:
            return (mid - fill_px) / tick
        return (fill_px - mid) / tick

    def _toxic_sample(self, markout_ticks: float) -> float:
        if markout_ticks >= -self.toxic_markout_ticks:
            return 0.0
        excess = -markout_ticks - self.toxic_markout_ticks
        return min(1.0, excess / max(self.toxic_markout_ticks, 1))

    def _decay_toxic_scores(self, st: BookState, sim_ts: int, publish_interval: int) -> None:
        if st.last_toxic_decay_ts == 0:
            st.last_toxic_decay_ts = sim_ts
            return
        dt = sim_ts - st.last_toxic_decay_ts
        if dt <= 0:
            return
        half_life = self.toxic_decay_half_life_steps * publish_interval
        decay = 0.5 ** (dt / half_life) if half_life > 0 else 1.0
        st.bid_toxic *= decay
        st.ask_toxic *= decay
        st.last_toxic_decay_ts = sim_ts

    def _apply_toxic_sample(
        self,
        st: BookState,
        maker_side: int,
        sample: float,
        sim_ts: int,
        publish_interval: int,
    ) -> None:
        if maker_side == OrderDirection.BUY:
            st.bid_toxic = (
                self.toxic_ema_alpha * sample
                + (1.0 - self.toxic_ema_alpha) * st.bid_toxic
            )
            if st.bid_toxic >= self.toxic_entry_block:
                cooldown_until = sim_ts + self.toxic_cooldown_steps * publish_interval
                st.bid_cooldown_until_ts = max(st.bid_cooldown_until_ts, cooldown_until)
        else:
            st.ask_toxic = (
                self.toxic_ema_alpha * sample
                + (1.0 - self.toxic_ema_alpha) * st.ask_toxic
            )
            if st.ask_toxic >= self.toxic_entry_block:
                cooldown_until = sim_ts + self.toxic_cooldown_steps * publish_interval
                st.ask_cooldown_until_ts = max(st.ask_cooldown_until_ts, cooldown_until)

    def _process_due_markouts(
        self,
        validator_key: str,
        state: MarketSimulationStateUpdate,
        tick: float,
    ) -> None:
        pending = self._pending_markouts.get(validator_key)
        if not pending:
            return

        sim_ts = state.timestamp
        publish_interval = self._publish_interval_ns(state.config)
        kept: deque[PendingMarkout] = deque()
        for book_id, maker_side, fill_px, _fill_ts, due_ts in pending:
            if due_ts > sim_ts:
                kept.append((book_id, maker_side, fill_px, _fill_ts, due_ts))
                continue
            if book_id not in state.books:
                continue
            book = state.books[book_id]
            if not book.bids or not book.asks:
                continue
            mid = (book.bids[0].price + book.asks[0].price) / 2.0
            markout = self._markout_ticks(maker_side, fill_px, mid, tick)
            sample = self._toxic_sample(markout)
            if sample <= 0.0:
                continue
            st = self._book_st(validator_key, book_id)
            self._apply_toxic_sample(st, maker_side, sample, sim_ts, publish_interval)
            if self.log_detail:
                bt.logging.debug(
                    f"BOOK {book_id} toxic markout side={'bid' if maker_side == OrderDirection.BUY else 'ask'} "
                    f"markout={markout:.2f}t sample={sample:.2f} "
                    f"bid_tox={st.bid_toxic:.2f} ask_tox={st.ask_toxic:.2f}"
                )
        self._pending_markouts[validator_key] = kept

    def _side_edge_mult(self, st: BookState, side: int) -> float:
        inv_mult = st.edge_mult
        tox = st.bid_toxic if side == OrderDirection.BUY else st.ask_toxic
        return inv_mult * (1.0 + tox * self.toxic_edge_scale)

    def _toxic_qty_scale(self, st: BookState, side: int) -> float:
        tox = st.bid_toxic if side == OrderDirection.BUY else st.ask_toxic
        floor = max(0.0, min(1.0, self.toxic_min_qty_scale))
        return max(floor, 1.0 - tox * (1.0 - floor))

    @staticmethod
    def _is_entry_side(side: int, net_inventory: float) -> bool:
        if net_inventory > 0 and side == OrderDirection.SELL:
            return False
        if net_inventory < 0 and side == OrderDirection.BUY:
            return False
        return True

    def _toxic_allows_entry(
        self,
        st: BookState,
        side: int,
        sim_ts: int,
        net_inventory: float,
    ) -> bool:
        """Block toxic-side entry makers; always allow inventory exit side."""
        if not self._is_entry_side(side, net_inventory):
            return True
        cooldown = (
            st.bid_cooldown_until_ts
            if side == OrderDirection.BUY
            else st.ask_cooldown_until_ts
        )
        if sim_ts < cooldown:
            return False
        tox = st.bid_toxic if side == OrderDirection.BUY else st.ask_toxic
        return tox < self.toxic_entry_block

    def _allows_entry_quote(
        self,
        st: BookState,
        side: int,
        sim_ts: int,
        net_inventory: float,
        throttled: bool,
    ) -> bool:
        """Entry maker allowed only when not throttled and not toxic/cooldown blocked."""
        if throttled and self._is_entry_side(side, net_inventory):
            return False
        return self._toxic_allows_entry(st, side, sim_ts, net_inventory)

    def _book_tier_counts(self, book_count: int) -> tuple[int, int, int, int]:
        """
        Returns (n_skip=0, n_eligible=book_count, n_core, n_maintenance).

        All books participate. core = target (round) with min floor.
        Default: core_min_ratio=0.625→80 min, core_target_ratio=0.70→90 target for 128 books.
        """
        n_skip = 0
        n_eligible = book_count
        n_core_min = max(1, int(book_count * self.core_min_ratio))
        n_core = max(n_core_min, min(round(book_count * self.core_target_ratio), book_count))
        n_reserve = max(0, book_count - n_core)
        return n_skip, n_eligible, n_core, n_reserve

    def _effective_obs_info(
        self,
        book_id: int,
        obs: RealizedObsState,
        sim_ts: int,
    ) -> tuple[int, int]:
        """(raw_obs, effective_obs). Subtracts one obs per changeover gap. Logs on detection."""
        grace = self.changeover_grace_period_ns
        raw = obs.non_zero_count()
        n_co, max_gap_ns = obs._changeover_count(grace)
        effective = max(0, raw - n_co)
        if n_co > 0 and self.log_detail:
            bt.logging.debug(
                f"[KAPPA_CHANGEOVER_LOCAL] book={book_id} gap_s={max_gap_ns / 1e9:.1f}"
                f" raw_obs={raw} effective_obs={effective}"
            )
        return raw, effective

    def _realized_selection_score(
        self, obs: RealizedObsState, sim_ts: int
    ) -> float:
        obs.prune(sim_ts, self.kappa_lookback_ns)
        grace = self.changeover_grace_period_ns
        return (
            obs.effective_non_zero_count(grace) * 15.0
            - obs.effective_obs_deficit(self.min_realized_observations, grace) * 25.0
            - obs.negative_count() * 8.0
            + min(obs.net_realized() * 100.0, 20.0)
        )

    def _book_mode(
        self,
        book_id: int,
        selection: "BookSelection",
        st: BookState,
        obs: RealizedObsState,
        sim_ts: int,
        unwind_min_qty: float,
    ) -> str:
        """Classify: EXIT_ONLY / DEFICIT / NEGATIVE / CORE_GOOD / MAINTENANCE."""
        has_inv = abs(st.inventory) >= unwind_min_qty
        age_exceeded = (
            has_inv
            and st.inventory_open_ts > 0
            and (sim_ts - st.inventory_open_ts) >= self.max_holding_time_ns
        )
        has_fifo = obs.front_lot() is not None
        account = self.accounts.get(book_id)
        has_open_orders = bool(account and account.orders)

        if (has_inv or has_fifo or has_open_orders) and age_exceeded:
            return "EXIT_ONLY"

        obs.prune(sim_ts, self.kappa_lookback_ns)
        _, effective_obs = self._effective_obs_info(book_id, obs, sim_ts)
        if effective_obs < self.min_realized_observations:
            return "DEFICIT"

        if obs.negative_count() > obs.positive_count() or obs.net_realized() < 0.0:
            return "NEGATIVE"

        return "CORE_GOOD" if book_id in selection.core else "MAINTENANCE"

    def _log_book_mode(
        self,
        validator_key: str,
        book_id: int,
        selection: "BookSelection",
        st: BookState,
        obs: RealizedObsState,
        sim_ts: int,
        unwind_min_qty: float,
    ) -> None:
        mode = self._book_mode(book_id, selection, st, obs, sim_ts, unwind_min_qty)
        cache = self._book_mode_cache.setdefault(validator_key, {})
        if cache.get(book_id) == mode:
            return
        cache[book_id] = mode

        if not self.log_detail:
            return
        tier = "core_good" if book_id in selection.core else "maintenance"
        obs.prune(sim_ts, self.kappa_lookback_ns)
        age_s = (sim_ts - st.inventory_open_ts) / 1e9 if st.inventory_open_ts > 0 else 0.0
        bt.logging.info(
            f"[BOOK_MODE] book={book_id} mode={mode} tier={tier}"
            f" obs={obs.non_zero_count()} pos={obs.positive_count()} neg={obs.negative_count()}"
            f" pnl={obs.net_realized():.4f} inv={st.inventory:.4f} age_s={age_s:.1f}"
        )

    def _lob_depth_qty(self, book: Book, depth: int) -> tuple[float, float, float]:
        """Return (top_bid_qty, top_ask_qty, summed depth over `depth` levels per side)."""
        bids = book.bids[:depth]
        asks = book.asks[:depth]
        top_bid = bids[0].quantity if bids else 0.0
        top_ask = asks[0].quantity if asks else 0.0
        total = sum(level.quantity for level in bids) + sum(level.quantity for level in asks)
        return top_bid, top_ask, total

    def _passive_spread_depth_score(self, book: Book | None, tick: float) -> float:
        """
        LOB quality for passive MM: spread, multi-level depth, thin top-of-book penalty.
        """
        if book is None or not book.bids or not book.asks:
            return -1_000.0

        spread_ticks = self._book_spread_ticks(book, tick)
        top_bid_qty, top_ask_qty, book_depth = self._lob_depth_qty(
            book, self.selection_lob_depth
        )

        spread_score = min(spread_ticks, 8) * 0.5
        if spread_ticks < self.min_spread_ticks:
            spread_score -= (self.min_spread_ticks - spread_ticks) * 2.0

        depth_score = math.log1p(book_depth)

        thin_penalty = 0.0
        thin_threshold = max(self.order_qty * 0.5, 1e-9)
        if top_bid_qty < thin_threshold:
            thin_penalty += 1.5
        if top_ask_qty < thin_threshold:
            thin_penalty += 1.5

        return spread_score + depth_score - thin_penalty

    def _book_activity_penalty(self, st: BookState, sim_ts: int) -> float:
        """Penalize books with no recent fills on our account (agent-local dead)."""
        if st.last_trade_ts == 0:
            return 3.0
        idle_ns = sim_ts - st.last_trade_ts
        if idle_ns < self.dead_book_gap_ns:
            return 0.0
        excess = idle_ns / max(self.dead_book_gap_ns, 1)
        return min(8.0, excess * 3.0)

    def _market_flow_score(self, flow: BookFlowState | None, sim_ts: int) -> float:
        """Reward active market-wide trade flow; penalize idle/dead books."""
        if flow is None or flow.trade_count() == 0:
            return -10.0
        volume = flow.trade_volume()
        count = flow.trade_count()
        score = math.log1p(volume) * 0.6 + min(count, 30) * 0.08
        idle_ns = sim_ts - flow.last_trade_ts()
        if idle_ns >= self.dead_book_gap_ns:
            score -= min(12.0, (idle_ns / max(self.dead_book_gap_ns, 1)) * 5.0)
        return score

    def _selection_fee_penalty(self, account: Account | None) -> float:
        if account is None:
            return 0.00023 * 100.0
        maker, taker = self._fee_rates(account)
        return (max(0.0, maker) + max(0.0, taker)) * 100.0

    def _book_health_score(
        self,
        validator_key: str,
        book_id: int,
        book: Book | None,
        tick: float,
        sim_ts: int,
        net_inventory: float,
        account: Account | None,
    ) -> float:
        """
        Dynamic book rank (higher = prefer core tier).

        book_health = w_r*realized + w_p*passive + w_m*market_flow − w_t*toxic
                      − w_i*inventory − w_f*fee − w_a*activity
        """
        obs = self._obs(validator_key, book_id)
        st = self._book_st(validator_key, book_id)
        flow = self._book_flow.get(validator_key, {}).get(book_id)

        realized_score = self._realized_selection_score(obs, sim_ts)
        passive_score = self._passive_spread_depth_score(book, tick)
        market_flow_score = self._market_flow_score(flow, sim_ts)
        toxic_score = (st.bid_toxic + st.ask_toxic) * 10.0 + max(
            0.0, st.edge_mult - 1.0
        ) * 5.0
        inventory_risk = min(abs(net_inventory) * 3.0, 15.0)
        fee_penalty = self._selection_fee_penalty(account)
        activity_penalty = self._book_activity_penalty(st, sim_ts)

        score = (
            self.sel_w_realized * realized_score
            + self.sel_w_passive * passive_score
            + self.sel_w_market * market_flow_score
            - self.sel_w_toxic * toxic_score
            - self.sel_w_inventory * inventory_risk
            - self.sel_w_fee * fee_penalty
            - self.sel_w_activity * activity_penalty
        )

        # Inventory risk: unrealized mark-to-market in ticks hurts the score,
        # rewarding books where our position is profitable or flat.
        net = net_inventory
        if book and book.bids and book.asks and abs(net) > 1e-9 and tick > 0:
            mid_price = (book.bids[0].price + book.asks[0].price) / 2.0
            front = self._obs(validator_key, book_id).front_lot()
            if front is not None:
                is_long, (_, _fqty, entry_px, _) = front
                if entry_px > 0:
                    pnl_ticks = (mid_price - entry_px) / tick if is_long else (entry_px - mid_price) / tick
                    score += max(-15.0, min(5.0, pnl_ticks * 0.4)) * self.sel_w_inventory

        return score

    def _min_trade_qty(self, cfg: object) -> float:
        vol_dec = int(getattr(cfg, "volumeDecimals", 8))
        return _round_qty(self.min_order_size, vol_dec)

    @staticmethod
    def _has_open_orders(account: Account | None) -> bool:
        return bool(account and account.orders)

    def _must_trade_books(
        self,
        validator_key: str,
        book_count: int,
        cfg: object,
        state: MarketSimulationStateUpdate,
    ) -> set[int]:
        """
        Books that must stay traded (≥ reserve): open exposure cannot be hard-skipped.
        """
        min_qty = self._min_trade_qty(cfg)
        must_trade: set[int] = set()
        accounts: dict[int, Account] = {}
        if state.accounts and self.uid in state.accounts:
            accounts = state.accounts[self.uid]
        obs_by_book = self._realized_obs.get(validator_key, {})
        for book_id in range(book_count):
            account = accounts.get(book_id)
            if account is not None:
                net = self._net_inventory(validator_key, book_id, account, cfg)
                if abs(net) >= min_qty:
                    must_trade.add(book_id)
                    continue
                if self._has_open_orders(account):
                    must_trade.add(book_id)
                    continue
            obs = obs_by_book.get(book_id)
            if obs is not None and obs.front_lot() is not None:
                must_trade.add(book_id)
        return must_trade

    def _finalize_book_selection(
        self,
        ranked: list[int],
        must_trade: set[int],
        book_count: int,
    ) -> BookSelection:
        """
        Tier split from a best-first ranking.

        must_trade books are never skipped and remain at least in reserve (core ok).
        """
        n_skip, n_eligible, n_core, _n_reserve = self._book_tier_counts(book_count)
        must_trade = must_trade & set(range(book_count))
        rank_index = {bid: idx for idx, bid in enumerate(ranked)}

        eligible_set = set(must_trade)
        for bid in ranked:
            if bid in eligible_set:
                continue
            if len(eligible_set) < n_eligible:
                eligible_set.add(bid)

        while len(eligible_set) > n_eligible:
            droppable = eligible_set - must_trade
            if not droppable:
                break
            worst = max(droppable, key=lambda b: rank_index[b])
            eligible_set.remove(worst)

        eligible_ranked = [bid for bid in ranked if bid in eligible_set]
        core = set(eligible_ranked[:n_core])
        reserve = set(eligible_ranked[n_core:])

        # n_skip=0 always (hard-skip abolished); skip stays empty.
        # The loop body is harmless: with n_skip=0 the break fires immediately.
        skip: set[int] = set()
        for bid in reversed(ranked):
            if bid in eligible_set:
                continue
            skip.add(bid)
            if len(skip) >= n_skip:
                break
        skip -= must_trade  # assertion: skip is always empty since n_skip=0

        for bid in must_trade:
            if bid in skip:
                skip.remove(bid)
            if bid not in core and bid not in reserve:
                reserve.add(bid)

        return BookSelection(skip=set(), core=core, reserve=reserve)

    def _fixed_book_selection(
        self,
        validator_key: str,
        book_count: int,
        cfg: object,
        state: MarketSimulationStateUpdate,
    ) -> BookSelection:
        must_trade = self._must_trade_books(validator_key, book_count, cfg, state)
        ranked = list(range(book_count))
        return self._finalize_book_selection(ranked, must_trade, book_count)

    def _dynamic_book_selection(
        self,
        validator_key: str,
        book_count: int,
        cfg: object,
        sim_ts: int,
        state: MarketSimulationStateUpdate,
    ) -> BookSelection:
        must_trade = self._must_trade_books(validator_key, book_count, cfg, state)
        tick = self._tick_size(cfg.priceDecimals)
        scores: dict[int, float] = {}
        for book_id in range(book_count):
            account = None
            if self.uid in state.accounts and book_id in state.accounts[self.uid]:
                account = state.accounts[self.uid][book_id]
            net = 0.0
            if account is not None:
                net = self._net_inventory(validator_key, book_id, account, cfg)
            book = state.books.get(book_id) if state.books else None
            scores[book_id] = self._book_health_score(
                validator_key, book_id, book, tick, sim_ts, net, account
            )

        ranked = sorted(
            range(book_count),
            key=lambda bid: (scores[bid], -bid),
            reverse=True,
        )
        return self._finalize_book_selection(ranked, must_trade, book_count)

    def _compute_book_selection(
        self,
        validator_key: str,
        book_count: int,
        cfg: object,
        sim_ts: int,
        state: MarketSimulationStateUpdate,
    ) -> BookSelection:
        if self.book_selection_mode == "fixed":
            return self._fixed_book_selection(validator_key, book_count, cfg, state)
        return self._dynamic_book_selection(
            validator_key, book_count, cfg, sim_ts, state
        )

    def _book_selection_for(
        self,
        validator_key: str,
        book_count: int,
        cfg: object,
        sim_ts: int,
        state: MarketSimulationStateUpdate,
    ) -> BookSelection:
        refresh_ns = self.book_rank_refresh_steps * int(cfg.publish_interval)
        last = self._last_book_rank_ts.get(validator_key, 0)
        sel = self._book_selection.get(validator_key)
        must_trade = self._must_trade_books(validator_key, book_count, cfg, state)
        n_skip, n_eligible, n_core, n_reserve = self._book_tier_counts(book_count)
        needs_refresh = (
            sel is None
            or len(sel.core) != n_core
            or len(sel.reserve) != n_reserve
            or sim_ts - last >= refresh_ns
            or not must_trade.issubset(sel.traded() if sel else set())
        )
        if needs_refresh:
            prev_traded = sel.traded() if sel else set()
            prev_core = sel.core if sel else set()
            sel = self._compute_book_selection(
                validator_key, book_count, cfg, sim_ts, state
            )
            self._book_selection[validator_key] = sel
            self._last_book_rank_ts[validator_key] = sim_ts
            dropped_from_traded = prev_traded - sel.traded()
            if dropped_from_traded:
                pending = self._pending_dropped_books.setdefault(validator_key, set())
                pending.update(dropped_from_traded)
            if prev_core and prev_core != sel.core:
                dropped_core = prev_core - sel.core
                added = sel.core - prev_core
                bt.logging.info(
                    f"Book rank refresh v={validator_key[:12]}: "
                    f"core_good={len(sel.core)} maintenance={len(sel.reserve)} "
                    f"dropped_core={sorted(dropped_core)[:8]}{'...' if len(dropped_core) > 8 else ''} "
                    f"added={sorted(added)[:8]}{'...' if len(added) > 8 else ''}"
                )
        return sel

    def _cleanup_dropped_books(
        self,
        response: FinanceAgentResponse,
        validator_key: str,
        dropped: set[int],
        selection: BookSelection,
        cfg: object,
        instr_used: dict[int, int],
    ) -> BookSelection:
        """
        One cleanup pass for books that moved between core_good and maintenance.

        Cancel resting orders; re-promote to reserve if exposure remains.
        Books no longer move to skip (hard-skip abolished).
        """
        if not dropped:
            return selection

        min_qty = self._min_trade_qty(cfg)
        still_pending: set[int] = set()
        promoted: list[int] = []

        for book_id in sorted(dropped):
            if book_id not in self.accounts:
                continue
            account = self.accounts[book_id]

            if (
                account.orders
                and instr_used.get(book_id, 0) < self.max_instructions_per_book
            ):
                batch = [o.id for o in account.orders[:8]]
                if batch:
                    response.cancel_orders(book_id, batch)
                    instr_used[book_id] = instr_used.get(book_id, 0) + 1
                    still_pending.add(book_id)

            if account.orders:
                still_pending.add(book_id)

            net = self._net_inventory(validator_key, book_id, account, cfg)
            obs = self._obs(validator_key, book_id)
            if (
                abs(net) >= min_qty
                or obs.front_lot() is not None
                or self._has_open_orders(account)
            ):
                selection.skip.discard(book_id)
                selection.core.discard(book_id)
                selection.reserve.add(book_id)
                promoted.append(book_id)

        if promoted:
            self._book_selection[validator_key] = selection
            bt.logging.info(
                f"Dropped-book cleanup v={validator_key[:12]}: "
                f"promoted_to_reserve={promoted[:12]}"
                f"{'...' if len(promoted) > 12 else ''}"
            )

        if still_pending:
            pending = self._pending_dropped_books.setdefault(validator_key, set())
            pending.update(still_pending)

        return selection

    def _tick_size(self, price_decimals: int) -> float:
        return 10.0 ** (-price_decimals)

    def _volume_cap(self, miner_wealth: float, volume_decimals: int) -> float:
        return round(self.capital_turnover_cap * miner_wealth, volume_decimals)

    def _book_volume_traded(
        self,
        account: Account,
        validator_key: str | None = None,
        book_id: int | None = None,
    ) -> float:
        """
        Per-book/account traded volume (informational/debug only; used for per-book cap enforcement).

        Resolution order:
        1. account.traded_volume — 24h per-book assessment window (primary)
        2. dict / lazy raw aliases (traded_volume, v)
        3. local per-book maker+taker volume tracker — local fallback
        4. account.fees.volume_traded — fee-tier window only; last-resort fallback
        """
        traded = getattr(account, "traded_volume", None)
        if traded is not None:
            return float(traded)

        try:
            traded = account["traded_volume"]  # type: ignore[index]
            if traded is not None:
                return float(traded)
        except (TypeError, KeyError):
            pass

        raw = getattr(account, "_raw", None)
        if isinstance(raw, dict):
            traded = raw.get("v")
            if traded is not None:
                return float(traded)

        if validator_key is not None and book_id is not None:
            local_mk = self._maker_volume.get(validator_key, {}).get(book_id, 0.0)
            local_tk = self._taker_volume.get(validator_key, {}).get(book_id, 0.0)
            local_total = local_mk + local_tk
            if local_total > 0.0:
                return local_total

        fees = getattr(account, "fees", None)
        if fees is not None:
            vol = getattr(fees, "volume_traded", None)
            if vol is not None:
                return float(vol)
            if isinstance(fees, dict):
                vol = fees.get("volume_traded")
                if vol is None:
                    vol = fees.get("v")
                if vol is not None:
                    return float(vol)

        return 0.0

    def _classify_profit_core(
        self,
        validator_key: str,
        selection: "BookSelection",
        sim_ts: int,
    ) -> set[int]:
        """
        Identify top profit_core_n books within core tier by realized PnL + markout quality.

        Criteria (ranked, descending):
          1. net_realized > 0 and effective_obs >= min_realized_observations
          2. bid_toxic + ask_toxic low
          3. recent market flow activity
        Books without sufficient obs or with negative net_realized rank last.
        Returns up to self.profit_core_n book IDs from selection.core.
        """
        n = max(1, self.profit_core_n)
        candidates = list(selection.core)
        if not candidates:
            return set()

        grace = self.changeover_grace_period_ns

        def _score(bid: int) -> float:
            obs = self._obs(validator_key, bid)
            obs.prune(sim_ts, self.kappa_lookback_ns)
            st = self._book_st(validator_key, bid)
            flow = self._book_flow.get(validator_key, {}).get(bid)

            eff_obs = obs.effective_non_zero_count(grace)
            net_r = obs.net_realized()
            toxic = st.bid_toxic + st.ask_toxic
            flow_score = self._market_flow_score(flow, sim_ts) if flow else -5.0

            # Primary: positive realized, sufficient obs
            pnl_score = min(net_r * 200.0, 30.0) if net_r > 0 else max(net_r * 50.0, -20.0)
            obs_score = min(eff_obs * 5.0, 25.0)
            toxic_penalty = toxic * 15.0
            return pnl_score + obs_score - toxic_penalty + flow_score * 0.5

        ranked = sorted(candidates, key=_score, reverse=True)
        return set(ranked[:n])

    def _is_hard_negative(
        self,
        obs: "RealizedObsState",
        st: "BookState",
        loss_budget: float,
    ) -> bool:
        """Return True if this book is too damaged to probe for NEGATIVE_RECOVERY.

        Conditions (all required):
          - neg_gap >= hard_negative_min_gap (default 2)
          - net_realized <= loss_budget (deeply in the red)
          - avg_markout_ticks < -1.0 (persistent adverse fill quality)
          - total toxic >= hard_negative_toxic_threshold (default 0.8)
        """
        neg_gap = obs.negative_count() - obs.positive_count()
        if neg_gap < self.hard_negative_min_gap:
            return False
        if obs.net_realized() > loss_budget:
            return False
        avg_markout = obs.avg_markout_ticks() if hasattr(obs, "avg_markout_ticks") else 0.0
        if avg_markout >= -1.0:
            return False
        if st.bid_toxic + st.ask_toxic < self.hard_negative_toxic_threshold:
            return False
        return True

    def _neg_summary_acc(
        self,
        validator_key: str,
        book_id: int,
        *,
        active_neg: bool = False,
        recovery_candidate: bool = False,
        probe_sent: bool = False,
        recovered: bool = False,
        hard_neg: bool = False,
    ) -> None:
        """Track unique books in NEGATIVE_SUMMARY buffer. probe_sent increments a counter."""
        buf = self._neg_summary_buf.setdefault(validator_key, {
            "active_negative": set(),
            "recovery_candidates": set(),
            "probes_sent": 0,
            "recovered": set(),
            "hard_negative": set(),
        })
        if active_neg:
            buf["active_negative"].add(book_id)
        if recovery_candidate:
            buf["recovery_candidates"].add(book_id)
        if probe_sent:
            buf["probes_sent"] = buf.get("probes_sent", 0) + 1
        if recovered:
            buf["recovered"].add(book_id)
        if hard_neg:
            buf["hard_negative"].add(book_id)

    def _compute_book_intensity(
        self,
        validator_key: str,
        book_id: int,
        obs: "RealizedObsState",
        st: "BookState",
        sim_ts: int,
        net: float,
    ) -> float:
        """Compute trade_intensity 0.0–1.0 for a book. Higher = more aggressive quoting."""
        obs.prune(sim_ts, self.kappa_lookback_ns)
        score = 0.5

        # Realized PnL
        net_r = obs.net_realized()
        if net_r > 0:
            score += min(net_r * 150.0, 0.20)
        else:
            score += max(net_r * 80.0, -0.25)

        # Obs quality: positive vs negative count
        pos_count = obs.positive_count()
        neg_count = obs.negative_count()
        eff_obs = obs.effective_non_zero_count(self.changeover_grace_period_ns)
        if pos_count > neg_count:
            score += 0.10
        elif neg_count > pos_count:
            score -= 0.15
        if eff_obs >= self.min_realized_observations:
            score += 0.05
        else:
            score -= 0.10  # DEFICIT penalty

        # Toxic score
        toxic = st.bid_toxic + st.ask_toxic
        score -= toxic * 0.20

        # Flow score
        flow = self._book_flow.get(validator_key, {}).get(book_id)
        if flow is not None:
            flow_score = self._market_flow_score(flow, sim_ts)
            score += flow_score * 0.05

        # Old inventory penalty
        if st.inventory_open_ts > 0:
            age_s = (sim_ts - st.inventory_open_ts) / 1e9
            if age_s > 600:
                score -= 0.20
            elif age_s > 300:
                score -= 0.10

        # Drift quarantine
        if self._is_drift_quarantined(validator_key, book_id) or self._is_severe_drift(validator_key, book_id):
            score -= 0.40

        return max(0.0, min(1.0, score))

    def _intensity_class(self, intensity: float) -> str:
        """Classify intensity into named bucket."""
        if intensity >= self.intensity_aggressive_threshold:
            return "aggressive"
        elif intensity >= self.intensity_normal_threshold:
            return "normal"
        elif intensity >= self.intensity_light_threshold:
            return "light"
        else:
            return "floor_only"

    def _compute_risk_gate(
        self,
        global_drain_mode: bool,
        old_inv_books: int,
        books_below_min_obs: int,
        tick_avg_net_realized: float,
        prev_forced_unwind_count: int,
        low_opp_fraction_below_threshold: bool,
    ) -> dict:
        """Compute per-tick automatic intensity risk gate.

        All rules are independent and additive. Returns a dict of gate flags
        and the effective aggressive_qty_mult cap. Only active when
        intensity_enabled=True and risk_gate_enabled=True.

        Rules:
          R2: forced_unwind > 0 or drain=True → ban aggressive class
          R3: old_inv_books > 1              → cap aggressive_qty_mult to 1.0
          R4: avg_net_realized < threshold   → demote aggressive to light
          R6: books_below_min_obs <= 30      → slow floor_probe frequency
          R7: LOW_OPP fraction below thresh  → full low-opportunity mode
        """
        agg_mult_cap = self.intensity_aggressive_qty_mult
        active_rules: list[str] = []

        ban_aggressive = False
        if prev_forced_unwind_count > 0 or global_drain_mode:
            ban_aggressive = True
            active_rules.append(f"R2:fu={prev_forced_unwind_count},drain={global_drain_mode}")

        if old_inv_books > 1:
            agg_mult_cap = min(agg_mult_cap, 1.0)
            active_rules.append(f"R3:old_inv={old_inv_books}")

        demote_to_light = False
        if tick_avg_net_realized < self.risk_gate_net_realized_threshold:
            demote_to_light = True
            active_rules.append(f"R4:nr={tick_avg_net_realized:.3f}")

        slow_floor_probe = False
        if books_below_min_obs <= self.risk_gate_below_obs_threshold:
            slow_floor_probe = True
            active_rules.append(f"R6:below_obs={books_below_min_obs}")

        full_low_opp_mode = False
        if low_opp_fraction_below_threshold:
            full_low_opp_mode = True
            active_rules.append("R7:LOW_OPP")

        return {
            "ban_aggressive": ban_aggressive,
            "aggressive_qty_mult_cap": agg_mult_cap,
            "demote_to_light": demote_to_light,
            "slow_floor_probe": slow_floor_probe,
            "full_low_opp_mode": full_low_opp_mode,
            "active_rules": active_rules,
        }

    def _classify_laggards(
        self,
        validator_key: str,
        selection: "BookSelection",
        sim_ts: int,
        hard_stop_cap: float,
        n_bottom: int = 10,
    ) -> set[int]:
        """
        Identify healthy low-volume books for quote-refresh boost.

        Computes per-book volume (maker+taker from account or local tracker).
        Logs VOLUME_MIN (dashboard 24H Vol proxy = min over active books),
        VOLUME_LAGGARDS (bottom-N list), VOLUME_BOOST, VOLUME_NO_BOOST.
        Returns a set of book_ids to force-refresh this tick.
        """
        active_books = selection.traded()
        if not active_books:
            return set()

        rt_map = self._roundtrip_volume.get(validator_key, {})
        unwind_min_qty = max(self.unwind_min_qty, 1e-8)

        vols: dict[int, float] = {}
        for bid in active_books:
            if bid in selection.skip:
                continue
            account = self.accounts.get(bid)
            if account is None:
                continue
            vols[bid] = self._book_volume_traded(account, validator_key, bid)

        if not vols:
            return set()

        min_bid = min(vols, key=lambda b: vols[b])
        min_vol = vols[min_bid]
        min_rt = rt_map.get(min_bid, 0.0)
        bt.logging.debug(
            f"[VOLUME_MIN] book={min_bid} vol={min_vol:.2f} rt={min_rt:.2f}"
        )

        sorted_by_vol = sorted(vols.items(), key=lambda x: x[1])
        bottom = sorted_by_vol[:n_bottom]
        if self.log_detail:
            bt.logging.debug(
                f"[VOLUME_LAGGARDS]"
                f" books={[b for b, _ in bottom]}"
                f" vols={[round(v, 2) for _, v in bottom]}"
                f" rts={[round(rt_map.get(b, 0.0), 2) for b, _ in bottom]}"
            )

        boost_set: set[int] = set()
        for bid, book_vol in bottom:
            st = self._book_st(validator_key, bid)
            obs = self._obs(validator_key, bid)

            if book_vol >= hard_stop_cap:
                if self.log_detail:
                    bt.logging.debug(
                        f"[VOLUME_NO_BOOST] book={bid} reason=hard_stop vol={book_vol:.2f}"
                    )
                continue

            if (
                abs(st.inventory) >= unwind_min_qty
                and st.inventory_open_ts > 0
                and (sim_ts - st.inventory_open_ts) >= self.max_holding_time_ns
            ):
                if self.log_detail:
                    bt.logging.debug(
                        f"[VOLUME_NO_BOOST] book={bid} reason=old_inventory vol={book_vol:.2f}"
                    )
                continue

            if (
                st.bid_toxic >= self.toxic_entry_block
                or st.ask_toxic >= self.toxic_entry_block
            ):
                if self.log_detail:
                    bt.logging.debug(
                        f"[VOLUME_NO_BOOST] book={bid} reason=toxic"
                        f" bid_tox={st.bid_toxic:.3f} ask_tox={st.ask_toxic:.3f}"
                    )
                continue

            obs.prune(sim_ts, self.kappa_lookback_ns)
            if obs.net_realized() < 0.0 or obs.negative_count() > obs.positive_count():
                if self.log_detail:
                    bt.logging.debug(
                        f"[VOLUME_NO_BOOST] book={bid} reason=negative"
                        f" pnl={obs.net_realized():.4f}"
                        f" neg={obs.negative_count()} pos={obs.positive_count()}"
                    )
                continue

            boost_set.add(bid)
            if self.log_detail:
                bt.logging.debug(
                    f"[VOLUME_BOOST] book={bid} reason=healthy_laggard"
                    f" vol={book_vol:.2f} rt={rt_map.get(bid, 0.0):.2f}"
                )

        return boost_set

    def _loss_budget_quote(self, miner_wealth: float) -> float:
        return miner_wealth * self.loss_budget_frac

    def _fee_rates(self, account: Account) -> tuple[float, float]:
        """Return (maker_rate, taker_rate); positive = cost."""
        if account.fees is None:
            return 0.0, 0.00023
        maker = float(account.fees.maker_fee_rate)
        taker = float(account.fees.taker_fee_rate)
        return maker, taker

    def _book_spread_ticks(self, book: Book, tick: float) -> int:
        if tick <= 0.0 or not book.bids or not book.asks:
            return 1
        spread = max(book.asks[0].price - book.bids[0].price, tick)
        return max(1, int(round(spread / tick)))

    def _fee_aware_half_spread_ticks(
        self,
        account: Account,
        book: Book,
        mid: float,
        tick: float,
        edge_mult: float,
    ) -> int:
        """
        Minimum half-spread in ticks so maker entry + likely taker exit can cover
        fees and at least half the visible book spread.
        """
        maker, taker = self._fee_rates(account)
        maker_cost = max(0.0, maker)
        taker_cost = max(0.0, taker)
        fee_ticks = int(math.ceil((maker_cost + taker_cost) * mid / tick)) if tick > 0 else 1
        spread_ticks = self._book_spread_ticks(book, tick)
        rt_ticks = fee_ticks + (spread_ticks + 1) // 2 + self.roundtrip_spread_buffer_ticks
        return max(
            self.min_spread_ticks,
            int(round(self.edge_ticks * edge_mult)),
            rt_ticks,
        )

    def _min_roundtrip_profit_quote(
        self,
        account: Account,
        book: Book,
        mid: float,
        tick: float,
        qty: float,
    ) -> float:
        """Fee- and spread-aware minimum realized PnL to accept a round-trip exit."""
        maker, taker = self._fee_rates(account)
        notional = mid * qty
        fee_floor = notional * (max(0.0, maker) + max(0.0, taker))
        spread_buf = self._book_spread_ticks(book, tick) * tick * qty * 0.5
        buffer = self.roundtrip_spread_buffer_ticks * tick * qty
        tick_goal = self.target_profit_ticks * tick * qty
        return max(tick_goal, fee_floor + spread_buf + buffer)

    @staticmethod
    def _lot_net_pnl(
        is_long: bool,
        qty: float,
        entry: float,
        entry_fee: float,
        exit_px: float,
        exit_fee_rate: float,
    ) -> float:
        """Fee-inclusive QUOTE PnL if this lot exits at exit_px."""
        close_fee = exit_px * qty * exit_fee_rate
        if is_long:
            return (exit_px - entry) * qty - entry_fee - close_fee
        return (entry - exit_px) * qty - entry_fee - close_fee

    def _should_seek_roundtrip(
        self,
        obs: RealizedObsState,
        min_qty: float,
        soft_limit: float,
        deficit: int,
        account_inventory: float,
    ) -> bool:
        """True when a FIFO lot should be evaluated for Kappa-aligned round-trip completion."""
        front = obs.front_lot()
        if front is None:
            return False
        _is_long, (_ts, qty, _px, _fee) = front
        if qty < min_qty:
            return False
        if deficit > 0:
            return True
        return abs(account_inventory) >= soft_limit

    def _plan_roundtrip_exit(
        self,
        obs: RealizedObsState,
        book: Book,
        account: Account,
        tick: float,
        sim_ts: int,
        soft_limit: float,
        min_qty: float,
        loss_budget: float,
        deficit: int,
        account_inventory: float,
    ) -> tuple[OrderDirection, float, float, bool, str] | None:
        """
        Round-trip exit state machine on the oldest FIFO lot.

        Returns (direction, qty, limit_price, use_market, reason) or None to defer.
        """
        front = obs.front_lot()
        if front is None or not book.bids or not book.asks:
            return None

        is_long, (open_ts, lot_qty, entry_px, entry_fee) = front
        if lot_qty < min_qty:
            return None

        maker_rate = account.fees.maker_fee_rate if account.fees else 0.0
        taker_rate = account.fees.taker_fee_rate if account.fees else 0.00023
        bid = book.bids[0].price
        ask = book.asks[0].price

        if is_long:
            direction = OrderDirection.SELL
            exit_maker_px = ask  # passive sell (offer side)
            exit_taker_px = bid  # aggressive sell into bids
            adverse_ticks = (entry_px - bid) / tick if tick > 0 else 0.0
        else:
            direction = OrderDirection.BUY
            exit_maker_px = bid  # passive buy (bid side)
            exit_taker_px = ask  # aggressive buy lifting offers
            adverse_ticks = (ask - entry_px) / tick if tick > 0 else 0.0

        mid = (bid + ask) / 2.0
        pnl_maker = self._lot_net_pnl(is_long, lot_qty, entry_px, entry_fee, exit_maker_px, maker_rate)
        pnl_taker = self._lot_net_pnl(is_long, lot_qty, entry_px, entry_fee, exit_taker_px, taker_rate)
        target_profit = self._min_roundtrip_profit_quote(account, book, mid, tick, lot_qty)
        holding_ns = sim_ts - open_ts
        kappa_loss_cap = loss_budget * 0.5 if deficit > 0 else 0.0
        kappa_urgent_ns = self.max_holding_time_ns // 3

        # 1) Fee-inclusive target profit at maker touch.
        if pnl_maker >= target_profit:
            return direction, lot_qty, exit_maker_px, False, "target_profit"

        # 2) Kappa obs deficit: prefer maker breakeven; taker only when urgent.
        if deficit > 0:
            if pnl_maker >= 0.0:
                return direction, lot_qty, exit_maker_px, False, "kappa_breakeven"
            if holding_ns >= kappa_urgent_ns and pnl_taker >= -kappa_loss_cap:
                return direction, lot_qty, exit_taker_px, True, "kappa_urgent_exit"

        # 3) Max holding time exceeded.
        if holding_ns >= self.max_holding_time_ns:
            if pnl_maker >= 0.0:
                return direction, lot_qty, exit_maker_px, False, "max_hold_profit"
            if pnl_taker >= -loss_budget:
                return direction, lot_qty, exit_taker_px, True, "max_hold_stop"
            # 3a) Forced: age >= max_hold * force_unwind_age_mult — accept larger loss.
            forced_budget = loss_budget * self.force_unwind_loss_budget_mult
            if holding_ns >= self.max_holding_time_ns * self.force_unwind_age_mult:
                if pnl_taker >= -forced_budget:
                    return direction, lot_qty, exit_taker_px, True, "max_hold_forced"
            # 3b) Emergency: age >= max_hold * 3 — exit regardless of loss.
            if holding_ns >= self.max_holding_time_ns * 3:
                return direction, lot_qty, exit_taker_px, True, "max_hold_emergency"

        # 4) Adverse move stop (price moved against entry).
        if adverse_ticks >= self.adverse_move_ticks:
            if pnl_taker >= -loss_budget:
                return direction, lot_qty, exit_taker_px, True, "adverse_stop"

        # 5) Account inventory risk: exit within loss budget (not FIFO net).
        if abs(account_inventory) >= soft_limit and pnl_taker >= -loss_budget:
            if pnl_maker >= 0.0:
                return direction, lot_qty, exit_maker_px, False, "inventory_profit"
            return direction, lot_qty, exit_taker_px, True, "inventory_stop"

        return None

    def _guard_priority(
        self,
        obs: RealizedObsState,
        account_inventory: float,
        sim_ts: int,
        book_id: int = 0,
    ) -> tuple[int, float, int, int]:
        obs.prune(sim_ts, self.kappa_lookback_ns)
        _, effective_obs = self._effective_obs_info(book_id, obs, sim_ts)
        return (
            max(0, self.min_realized_observations - effective_obs),
            abs(account_inventory),
            obs.negative_count(),
            -obs.last_realized_ts(),
        )

    def _try_roundtrip_guard(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        book: Book,
        account: Account,
        obs: RealizedObsState,
        st: BookState,
        cfg: object,
        tick: float,
        sim_ts: int,
        soft_limit: float,
        min_qty: float,
        at_hard_stop: bool,
        throttled: bool,
        budget_left,
        spend,
    ) -> bool:
        """
        Complete realized round-trips on the oldest FIFO lot (Kappa-aligned).

        Not mere inventory reduction: each exit is gated on entry price, fees,
        target profit, holding time, adverse move, and Kappa obs deficit.
        """
        if not budget_left():
            return False

        obs.prune(sim_ts, self.kappa_lookback_ns)
        _, effective_obs = self._effective_obs_info(book_id, obs, sim_ts)
        deficit = max(0, self.min_realized_observations - effective_obs)
        account_inventory = st.inventory
        if not self._should_seek_roundtrip(
            obs, min_qty, soft_limit, deficit, account_inventory
        ):
            return False

        loss_budget = self._loss_budget_quote(cfg.miner_wealth)
        plan = self._plan_roundtrip_exit(
            obs, book, account, tick, sim_ts, soft_limit, min_qty, loss_budget, deficit,
            account_inventory,
        )
        if plan is None:
            return False

        direction, lot_qty, exit_px, use_market, reason = plan
        close_qty = _round_qty(max(lot_qty, min_qty), cfg.volumeDecimals)
        close_qty = self._affordable_qty(account, direction, exit_px, close_qty, cfg.volumeDecimals)
        if close_qty < min_qty:
            return False

        # Throttle: maker exit always ok; taker only for deficit / account inventory risk.
        if use_market and throttled and deficit <= 0 and abs(account_inventory) < soft_limit:
            use_market = False

        if use_market:
            response.market_order(
                book_id=book_id,
                direction=direction,
                quantity=close_qty,
                stp=STP.CANCEL_OLDEST,
            )
        else:
            response.limit_order(
                book_id=book_id,
                direction=direction,
                quantity=close_qty,
                price=exit_px,
                stp=STP.CANCEL_OLDEST,
                timeInForce=TimeInForce.GTT,
                expiryPeriod=self.expiry_period,
                postOnly=True,
            )

        if self.log_detail:
            bt.logging.debug(f"BOOK {book_id} roundtrip exit: {reason} qty={close_qty} market={use_market}")
        spend()
        st.last_instruction_ts = sim_ts
        return True

    def _instruction_pulse(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        book: Book,
        account: Account,
        st: BookState,
        cfg: object,
        tick: float,
        sim_ts: int,
        net_inventory: float,
        at_hard_stop: bool,
        throttled: bool,
        order_qty: float,
        min_qty: float,
        budget_left,
        spend,
    ) -> bool:
        """
        Instruction pulse: optional presence tick without bypassing toxic/throttle.

        Default mode is cancel-only (stale quote pull). Limit placement is opt-in
        and respects _allows_entry_quote on each side.
        """
        if at_hard_stop:
            if account.orders and budget_left():
                response.cancel_orders(book_id, [account.orders[0].id])
                spend()
                st.last_instruction_ts = sim_ts
                return True
            return False

        if not budget_left():
            return False

        if not book.bids or not book.asks:
            return False

        mid = (book.bids[0].price + book.asks[0].price) / 2.0
        bid_px = _floor_tick(mid - tick, tick, cfg.priceDecimals)
        ask_px = _ceil_tick(mid + tick, tick, cfg.priceDecimals)

        if self.instruction_pulse_cancel_only:
            stale_ids = (
                self._stale_order_ids(account.orders, OrderDirection.BUY, bid_px, tick)
                + self._stale_order_ids(account.orders, OrderDirection.SELL, ask_px, tick)
            )
            if stale_ids:
                response.cancel_orders(book_id, stale_ids[:1])
                spend()
                st.last_instruction_ts = sim_ts
                return True
            return False

        if throttled:
            return False

        if self._allows_entry_quote(
            st, OrderDirection.BUY, sim_ts, net_inventory, throttled
        ):
            bid_qty = self._affordable_qty(
                account, OrderDirection.BUY, bid_px, order_qty, cfg.volumeDecimals
            )
            if bid_qty >= min_qty:
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=bid_qty,
                    price=bid_px,
                    stp=STP.CANCEL_BOTH,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=self.expiry_period,
                    postOnly=True,
                )
                spend()
                st.last_instruction_ts = sim_ts
                return True

        if self._allows_entry_quote(
            st, OrderDirection.SELL, sim_ts, net_inventory, throttled
        ):
            ask_qty = self._affordable_qty(
                account, OrderDirection.SELL, ask_px, order_qty, cfg.volumeDecimals
            )
            if ask_qty >= min_qty:
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=ask_qty,
                    price=ask_px,
                    stp=STP.CANCEL_BOTH,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=self.expiry_period,
                    postOnly=True,
                )
                spend()
                st.last_instruction_ts = sim_ts
                return True

        return False

    def _imbalance(self, book: Book, depth: int) -> float:
        bids = book.bids[:depth]
        asks = book.asks[:depth]
        bid_vol = sum(level.quantity for level in bids)
        ask_vol = sum(level.quantity for level in asks)
        denom = bid_vol + ask_vol
        if denom <= 0.0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _max_base_position(self, mid: float, miner_wealth: float) -> float:
        if mid <= 0.0:
            return self.order_qty
        return (miner_wealth * self.position_frac) / mid

    def _fair_quotes(
        self,
        book: Book,
        account: Account,
        st: BookState,
        tick: float,
        price_decimals: int,
        mid: float,
        max_base: float,
    ) -> tuple[float, float]:
        imb = self._imbalance(book, self.imbalance_depth)
        q_norm = 0.0
        if max_base > 0.0:
            q_norm = max(-1.0, min(1.0, st.inventory / max_base))

        # Integer-tick skew: sub-tick shifts are erased by floor/ceil rounding.
        inv_skew_ticks = int(round(self.gamma * q_norm))
        imb_skew_ticks = int(round(self.imbalance_scale * imb * self.edge_ticks))
        total_skew_ticks = imb_skew_ticks - inv_skew_ticks
        cap = self.max_skew_ticks
        total_skew_ticks = max(-cap, min(cap, total_skew_ticks))
        # Push quotes away on toxic side.
        if st.bid_toxic > 0.2:
            total_skew_ticks -= int(round(st.bid_toxic * self.toxic_skew_ticks))
        if st.ask_toxic > 0.2:
            total_skew_ticks += int(round(st.ask_toxic * self.toxic_skew_ticks))
        total_skew_ticks = max(-cap, min(cap, total_skew_ticks))

        bid_half_ticks = self._fee_aware_half_spread_ticks(
            account, book, mid, tick, self._side_edge_mult(st, OrderDirection.BUY)
        )
        ask_half_ticks = self._fee_aware_half_spread_ticks(
            account, book, mid, tick, self._side_edge_mult(st, OrderDirection.SELL)
        )

        reservation = mid + total_skew_ticks * tick
        bid = _floor_tick(reservation - bid_half_ticks * tick, tick, price_decimals)
        ask = _ceil_tick(reservation + ask_half_ticks * tick, tick, price_decimals)

        if bid >= ask:
            bid = _floor_tick(mid - bid_half_ticks * tick, tick, price_decimals)
            ask = _ceil_tick(mid + ask_half_ticks * tick, tick, price_decimals)
        if bid >= ask:
            bid = _floor_tick(mid - tick, tick, price_decimals)
            ask = _ceil_tick(mid + tick, tick, price_decimals)

        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        bid = min(bid, best_ask - tick)
        ask = max(ask, best_bid + tick)
        return bid, ask

    def _update_weak_book_edge(
        self,
        st: BookState,
        inventory: float,
        soft_limit: float,
        sim_ts: int,
    ) -> None:
        if abs(inventory) > soft_limit:
            if st.stuck_since_ts == 0:
                st.stuck_since_ts = sim_ts
            stuck_ns = sim_ts - st.stuck_since_ts
            ramp = max(self.stuck_edge_ramp_ns, 1)
            st.edge_mult = min(2.0, 1.0 + (stuck_ns / ramp) * 0.5)
        else:
            st.stuck_since_ts = 0
            st.edge_mult = max(1.0, st.edge_mult * 0.98)

    def _update_account_inventory_age(
        self,
        st: BookState,
        net: float,
        min_qty: float,
        sim_ts: int,
    ) -> None:
        if abs(net) < min_qty:
            st.inventory_open_ts = 0
            st.inventory_last_sign = 0
            return
        sign = 1 if net > 0 else -1
        if sign != st.inventory_last_sign:
            st.inventory_open_ts = sim_ts
            st.inventory_last_sign = sign

    def _update_exit_only_lock(
        self,
        st: BookState,
        net: float,
        unwind_min_qty: float,
        sim_ts: int,
    ) -> None:
        """Sticky lock: set when inventory age exceeds max_holding_time; cleared only when flat.

        Also clears for stranded positions (below exchange minOrderSize) since market-order
        exit is impossible — normal quoting will naturally unwind them.
        """
        if abs(net) < unwind_min_qty or abs(net) < self.min_order_size:
            st.exit_only_until_flat = False
            return
        if (
            st.inventory_open_ts > 0
            and (sim_ts - st.inventory_open_ts) >= self.max_holding_time_ns
        ):
            st.exit_only_until_flat = True

    def _try_forced_account_unwind(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        book: Book,
        account: Account,
        st: BookState,
        cfg: object,
        sim_ts: int,
        unwind_min_qty: float,
        at_hard_stop: bool,
        budget_left,
        spend,
    ) -> bool:
        net = st.inventory
        if abs(net) < unwind_min_qty:
            return False
        if st.inventory_open_ts <= 0:
            return False
        age_ns = sim_ts - st.inventory_open_ts
        if age_ns < self.max_holding_time_ns:
            return False

        age_s = age_ns / 1e9
        unwind_side = OrderDirection.SELL if net > 0 else OrderDirection.BUY
        side_name = "SELL" if unwind_side == OrderDirection.SELL else "BUY"

        # Step 1 — cancel-first: free up balance before placing market order.
        if self.unwind_cancel_first and account.orders and budget_left():
            all_ids = [o.id for o in account.orders[:8]]
            response.cancel_orders(book_id, all_ids)
            spend()
            if self.log_detail:
                bt.logging.debug(
                    f"BOOK {book_id} FORCED_UNWIND_CANCEL_FIRST"
                    f" inv={net} age_s={age_s:.1f} open_orders={len(account.orders)}"
                )
            return True

        # Step 2 — market order (runs on tick after cancel clears).
        if not budget_left():
            return False

        abs_inv = abs(net)
        max_slice = _round_qty(
            min(abs_inv, max(self.max_unwind_qty_per_slice, unwind_min_qty)),
            cfg.volumeDecimals,
        )
        # Guard: position below exchange minOrderSize cannot be exited via market order.
        exchange_min = _round_qty(self.min_order_size, cfg.volumeDecimals)
        if max_slice < exchange_min:
            bt.logging.warning(
                f"[UNWIND_STRANDED] book={book_id} side={side_name} inv={net:.4f}"
                f" max_slice={max_slice:.4f} exchange_min={exchange_min:.4f} age_s={age_s:.1f}"
            )
            return False

        # Use ask for BUY affordability check (market order hits asks).
        px_for_afford = book.asks[0].price if unwind_side == OrderDirection.BUY else 0.0
        affordable = self._affordable_qty(account, unwind_side, px_for_afford, max_slice, cfg.volumeDecimals)

        if affordable < unwind_min_qty:
            bt.logging.warning(
                f"[UNWIND_FAIL] book={book_id} side={side_name} inv={net:.4f} age_s={age_s:.1f}"
                f" affordable={affordable:.4f} reason=insufficient_balance"
                f" wanted={max_slice:.4f} free_base={self._account_free_base(account):.4f}"
                f" free_quote={self._account_free_quote(account):.2f}"
                f" open_orders={len(account.orders)}"
            )
            return False

        self._unwind_log_buffer.append((book_id, side_name, affordable, net, age_s))
        if self.log_detail:
            bt.logging.debug(
                f"BOOK {book_id} FORCED_UNWIND_MARKET side={side_name}"
                f" qty={affordable} inv={net} age_s={age_s:.1f}"
                f" at_hard_stop={at_hard_stop}"
            )
        response.market_order(
            book_id=book_id,
            direction=unwind_side,
            quantity=affordable,
            stp=STP.CANCEL_OLDEST,
        )
        spend()
        return True

    def _order_at_price(self, orders: list[Order], side: int, price: float, tick: float) -> Order | None:
        for order in orders:
            if order.side == side and order.price is not None and abs(order.price - price) < tick * 0.5:
                return order
        return None

    def _stale_order_ids(
        self,
        orders: list[Order],
        side: int,
        target: float,
        tick: float,
    ) -> list[int]:
        threshold = self.reprice_ticks * tick
        stale: list[int] = []
        for order in orders:
            if order.side != side or order.price is None:
                continue
            if abs(order.price - target) >= threshold:
                stale.append(order.id)
        return stale

    @staticmethod
    def _side_order_ids(orders: list[Order], side: int) -> list[int]:
        return [o.id for o in orders if o.side == side]

    def _pull_blocked_entry_orders(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        account: Account,
        st: BookState,
        sim_ts: int,
        net_inventory: float,
        throttled: bool,
        budget_left,
        spend,
    ) -> None:
        """Cancel all resting entry-side orders when throttle or toxic/cooldown blocks."""
        for side in (OrderDirection.BUY, OrderDirection.SELL):
            if self._allows_entry_quote(st, side, sim_ts, net_inventory, throttled):
                continue
            blocked_ids = self._side_order_ids(account.orders, side)
            if blocked_ids and budget_left():
                response.cancel_orders(book_id, blocked_ids[:8])
                spend()

    def _affordable_qty(
        self,
        account: Account,
        direction: OrderDirection,
        price: float,
        qty: float,
        volume_decimals: int,
    ) -> float:
        if direction == OrderDirection.BUY:
            if price <= 0.0:
                return 0.0
            max_qty = self._account_free_quote(account) / price
            return _round_qty(min(qty, max_qty), volume_decimals)
        max_qty = self._account_free_base(account)
        return _round_qty(min(qty, max_qty), volume_decimals)

    def _process_book_mm(
        self,
        response: FinanceAgentResponse,
        validator_key: str,
        book_id: int,
        book: Book,
        account: Account,
        st: BookState,
        cfg: object,
        tick: float,
        sim_ts: int,
        refresh_quotes: bool,
        volume_cap: float,
        throttle_cap: float,
        hard_stop_cap: float,
        instr_used: dict[int, int],
        mm_core: bool = True,
        global_drain_mode: bool = False,
        force_unwind_allowed: bool = True,
        is_profit_core: bool = False,
        volume_source: str = "per_book",
        intensity_class: str = "normal",
        intensity_value: float = 0.5,
        low_opportunity_mode: bool = False,
        rg_aggressive_qty_mult_cap: float | None = None,
        rg_floor_probe_interval: int | None = None,
        rg_neg_probe_faster: bool = False,
    ) -> None:
        def budget_left(cost: int = 1) -> bool:
            return instr_used.get(book_id, 0) + cost <= self.max_instructions_per_book

        def spend(cost: int = 1) -> None:
            instr_used[book_id] = instr_used.get(book_id, 0) + cost
            st.last_instruction_ts = sim_ts

        if volume_source == "per_book":
            book_volume = self._book_volume_traded(account, validator_key, book_id)
        else:
            # Pattern A (local_tracker): validator mirrors global value — use per-book local tracker.
            # "unknown": no account state available — use local tracker (may return 0).
            book_volume = (
                self._maker_volume.get(validator_key, {}).get(book_id, 0.0)
                + self._taker_volume.get(validator_key, {}).get(book_id, 0.0)
            )
        at_hard_stop = book_volume >= hard_stop_cap
        throttled = book_volume >= throttle_cap

        entry_allowed = not at_hard_stop
        _vb_msg = (
            f"[VOLUME_BOOK] b={book_id} vol={book_volume:.2f} src={volume_source}"
            f" throttle_cap={throttle_cap:.2f} hard_stop_cap={hard_stop_cap:.2f}"
            f" throttled={throttled} at_hard_stop={at_hard_stop}"
        )
        if throttled or at_hard_stop:
            bt.logging.info(_vb_msg)
        elif self.log_detail:
            bt.logging.debug(_vb_msg)

        mid = (book.bids[0].price + book.asks[0].price) / 2.0
        max_base = self._max_base_position(mid, cfg.miner_wealth)
        soft_limit = max(self.order_qty, max_base * self.inventory_soft_ratio)
        net = self._sync_account_inventory(
            validator_key, book_id, account, st, cfg, cfg.volumeDecimals, sim_ts
        )
        self._update_weak_book_edge(st, net, soft_limit, sim_ts)
        self._decay_toxic_scores(st, sim_ts, self._publish_interval_ns(cfg))
        obs = self._obs(validator_key, book_id)

        base_qty = self.order_qty
        min_qty = _round_qty(self.min_order_size, cfg.volumeDecimals)
        # unwind_min_qty: exit/forced-unwind threshold, separate from entry min_order_size.
        unwind_min_qty = max(self.unwind_min_qty, 10.0 ** (-cfg.volumeDecimals))
        qty_scale = self._throttle_qty_scale(book_volume, throttle_cap, hard_stop_cap)
        # order_qty: target size; min_qty: minimum acceptable fill to submit
        order_qty = _round_qty(max(min_qty, base_qty * qty_scale), cfg.volumeDecimals)

        # Intensity-based qty scaling (only when intensity_enabled=True).
        if self.intensity_enabled:
            if intensity_class == "aggressive":
                _eff_agg_mult = (
                    rg_aggressive_qty_mult_cap
                    if rg_aggressive_qty_mult_cap is not None
                    else self.intensity_aggressive_qty_mult
                )
                _iqm = _eff_agg_mult
            elif intensity_class == "normal":
                _iqm = 1.0
            elif intensity_class == "light":
                _iqm = self.intensity_light_qty_mult
            else:  # floor_only
                _iqm = self.intensity_qty_min_mult
            order_qty = _round_qty(max(min_qty, order_qty * _iqm), cfg.volumeDecimals)

        # Update account inventory age using unwind threshold (not entry min_qty).
        self._update_account_inventory_age(st, net, unwind_min_qty, sim_ts)
        self._update_exit_only_lock(st, net, unwind_min_qty, sim_ts)

        # Intensity-based quote refresh cadence (only when intensity_enabled=True).
        # aggressive: force refresh every tick; light/floor_only: throttle below global timer.
        if self.intensity_enabled:
            if intensity_class == "aggressive":
                refresh_quotes = True
            elif intensity_class in ("light", "floor_only") and refresh_quotes:
                _factor = (
                    self.intensity_light_refresh_factor if intensity_class == "light"
                    else self.intensity_floor_refresh_factor
                )
                _slow_interval_ns = self.quote_interval * cfg.publish_interval * _factor
                if sim_ts - st.last_instruction_ts < _slow_interval_ns:
                    refresh_quotes = False

        # EXIT_ONLY_LOCK: sticky old-inventory book — forced unwind only, no MM.
        if st.exit_only_until_flat:
            age_s = (sim_ts - st.inventory_open_ts) / 1e9 if st.inventory_open_ts > 0 else 0.0
            if self.log_detail:
                bt.logging.debug(
                    f"[EXIT_ONLY_BLOCK_MM] book={book_id} reason=exit_only_lock"
                    f" inv={net:.4f} age_s={age_s:.1f}"
                )
            if force_unwind_allowed:
                self._try_forced_account_unwind(
                    response, book_id, book, account, st, cfg, sim_ts,
                    unwind_min_qty, at_hard_stop, budget_left, spend,
                )
            else:
                # pace_limit or drain skip: cancel-only, no market order
                if account.orders and budget_left():
                    response.cancel_orders(book_id, [o.id for o in account.orders[:8]])
                    spend()
            return

        forced_unwind_mode = False  # never True here; exit_only books returned above

        # --- Drift quarantine / severe drift ---
        # Entry banned; FIFO RoundTripGuard exit prohibited; account_net flat-first only.
        is_drift_q = self._is_drift_quarantined(validator_key, book_id)
        is_severe_d = self._is_severe_drift(validator_key, book_id)
        if is_drift_q or is_severe_d:
            _dq_tol = self._fifo_drift_tolerance(cfg.volumeDecimals)
            _dq_flat = abs(net) < max(self.unwind_min_qty, _dq_tol)
            _dq_stable = (
                self._drift_stable_ticks.get(validator_key, {}).get(book_id, 0)
                >= self.drift_stable_ticks_required
            )
            _dq_no_orders = not account.orders

            # Check quarantine release: stable ticks + flat + no open orders.
            if _dq_stable and _dq_flat and _dq_no_orders:
                qmap = self._drift_quarantine.get(validator_key)
                if qmap:
                    qmap.pop(book_id, None)
                smap = self._severe_drift_books.get(validator_key)
                if smap:
                    smap.discard(book_id)
                self._quarantine_all_cancelled.get(validator_key, set()).discard(book_id)
                self._drift_stable_ticks.get(validator_key, {}).pop(book_id, None)
                # Released this tick — skip quarantine actions but don't do MM yet.
                return

            _dq_ac_set = self._quarantine_all_cancelled.setdefault(validator_key, set())
            if book_id not in _dq_ac_set:
                # First entry into quarantine: cancel ALL resting orders immediately.
                if account.orders and budget_left():
                    response.cancel_orders(book_id, [o.id for o in account.orders[:16]])
                    spend()
                _dq_ac_set.add(book_id)
            else:
                # Subsequent ticks: cancel entry-side orders only; exit orders may stand.
                if budget_left():
                    for _dq_side in (OrderDirection.BUY, OrderDirection.SELL):
                        if not self._is_entry_side(_dq_side, net):
                            continue
                        _dq_ids = self._side_order_ids(account.orders, _dq_side)
                        if _dq_ids and budget_left():
                            response.cancel_orders(book_id, _dq_ids[:8])
                            spend()

            # account_net flat-first exit: limit at touch price, no FIFO matching.
            if abs(net) >= min_qty and budget_left():
                _dq_exit_side = OrderDirection.SELL if net > 0 else OrderDirection.BUY
                _dq_exit_px = (
                    book.asks[0].price if _dq_exit_side == OrderDirection.BUY
                    else book.bids[0].price
                )
                _dq_exit_qty = _round_qty(min(abs(net), order_qty), cfg.volumeDecimals)
                _dq_exit_qty = self._affordable_qty(
                    account, _dq_exit_side, _dq_exit_px, _dq_exit_qty, cfg.volumeDecimals
                )
                if (
                    _dq_exit_qty >= min_qty
                    and self._order_at_price(
                        account.orders, _dq_exit_side, _dq_exit_px, tick
                    ) is None
                ):
                    response.limit_order(
                        book_id=book_id,
                        direction=_dq_exit_side,
                        quantity=_dq_exit_qty,
                        price=_dq_exit_px,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period,
                        postOnly=True,
                    )
                    spend()
            return

        def side_qty(side: int, px: float, qty: float) -> float:
            scaled = _round_qty(
                qty * self._toxic_qty_scale(st, side), cfg.volumeDecimals
            )
            return self._affordable_qty(account, side, px, scaled, cfg.volumeDecimals)

        # --- RoundTripGuard (FIFO-based) — hard stop no longer blocks exit ---
        self._try_roundtrip_guard(
            response, book_id, book, account, obs, st, cfg, tick, sim_ts,
            soft_limit, min_qty, at_hard_stop, throttled, budget_left, spend,
        )

        # Maintenance tier — survival quoting for Kappa obs/round-trip.
        if not mm_core:
            # Determine book mode for this tick (cache already updated by _log_book_mode).
            obs.prune(sim_ts, self.kappa_lookback_ns)
            _, eff_obs = self._effective_obs_info(book_id, obs, sim_ts)
            is_deficit = eff_obs < self.min_realized_observations
            is_negative = (
                obs.negative_count() > obs.positive_count() or obs.net_realized() < 0.0
            )

            # EXIT_ONLY lock already handled above; fall through means no age issue.
            # NEGATIVE: entry banned — exit-side only.
            if is_negative:
                if obs.front_lot() is None and abs(net) >= min_qty and budget_left():
                    exit_side = OrderDirection.SELL if net > 0 else OrderDirection.BUY
                    exit_px = book.asks[0].price if exit_side == OrderDirection.BUY else book.bids[0].price
                    exit_qty = _round_qty(min(abs(net), order_qty), cfg.volumeDecimals)
                    exit_qty = self._affordable_qty(account, exit_side, exit_px, exit_qty, cfg.volumeDecimals)
                    if exit_qty >= min_qty and self._order_at_price(account.orders, exit_side, exit_px, tick) is None:
                        response.limit_order(
                            book_id=book_id,
                            direction=exit_side,
                            quantity=exit_qty,
                            price=exit_px,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()
                if refresh_quotes and budget_left():
                    # Cancel entry-side orders that shouldn't exist in NEGATIVE mode.
                    for side in (OrderDirection.BUY, OrderDirection.SELL):
                        if not self._is_entry_side(side, net):
                            continue
                        entry_ids = self._side_order_ids(account.orders, side)
                        if entry_ids and budget_left():
                            response.cancel_orders(book_id, entry_ids[:8])
                            spend()
                return

            # DEFICIT floor_probe: tiny postOnly limit probes at tight spread to generate 3 obs
            # within the changeover_grace_period_ns window so eff_obs reaches min_realized_observations.
            # Only active while eff_obs < min_realized_observations (DEFICIT) and not throttled.
            if (
                is_deficit
                and not is_negative
                and refresh_quotes
                and not throttled
                and budget_left()
                and sim_ts - st.deficit_probe_last_ts >= (
                    rg_floor_probe_interval
                    if rg_floor_probe_interval is not None
                    else self.deficit_floor_probe_interval_ns
                )
            ):
                _fp_ticks = max(1, self.deficit_floor_probe_spread_ticks)
                _fp_qty_base = max(min_qty, order_qty * self.deficit_floor_probe_size_mult)
                _fp_qty = _round_qty(_fp_qty_base, cfg.volumeDecimals)
                _fp_placed = 0

                def _make_fp_price(direction: OrderDirection) -> float:
                    raw = (mid - _fp_ticks * tick) if direction == OrderDirection.BUY else (mid + _fp_ticks * tick)
                    if tick > 0:
                        raw = round(raw / tick) * tick
                    return round(raw, cfg.priceDecimals)

                if abs(net) < min_qty:
                    # Flat: probe both sides to set up a complete round trip quickly.
                    for _fp_dir in (OrderDirection.BUY, OrderDirection.SELL):
                        _fp_px = _make_fp_price(_fp_dir)
                        _fq = self._affordable_qty(account, _fp_dir, _fp_px, _fp_qty, cfg.volumeDecimals)
                        if (
                            _fq >= min_qty
                            and budget_left()
                            and self._order_at_price(account.orders, _fp_dir, _fp_px, tick) is None
                        ):
                            response.limit_order(
                                book_id=book_id,
                                direction=_fp_dir,
                                quantity=_fq,
                                price=_fp_px,
                                stp=STP.CANCEL_BOTH,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=self.expiry_period,
                                postOnly=True,
                            )
                            spend()
                            _fp_placed += 1
                elif net > min_qty:
                    # Long position: tight ASK to close quickly and generate obs.
                    _fp_px = _make_fp_price(OrderDirection.SELL)
                    _fq = self._affordable_qty(account, OrderDirection.SELL, _fp_px, _fp_qty, cfg.volumeDecimals)
                    if (
                        _fq >= min_qty
                        and budget_left()
                        and self._order_at_price(account.orders, OrderDirection.SELL, _fp_px, tick) is None
                    ):
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=_fq,
                            price=_fp_px,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()
                        _fp_placed += 1
                else:
                    # Short position: tight BID to close quickly and generate obs.
                    _fp_px = _make_fp_price(OrderDirection.BUY)
                    _fq = self._affordable_qty(account, OrderDirection.BUY, _fp_px, _fp_qty, cfg.volumeDecimals)
                    if (
                        _fq >= min_qty
                        and budget_left()
                        and self._order_at_price(account.orders, OrderDirection.BUY, _fp_px, tick) is None
                    ):
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=_fq,
                            price=_fp_px,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()
                        _fp_placed += 1

                if _fp_placed > 0:
                    st.deficit_probe_last_ts = sim_ts
                    _dbuf = self._deficit_recovery_buf.setdefault(validator_key, {
                        "floor_probes_sent": 0, "forced_unwind_count": 0,
                    })
                    _dbuf["floor_probes_sent"] = _dbuf.get("floor_probes_sent", 0) + 1

            # Per-book maintenance entry gate: only per-book hard_stop blocks entry.
            maint_entry_ok = not at_hard_stop
            # LOW_OPPORTUNITY_MODE suppresses non-deficit maintenance entry.
            if self.intensity_enabled and low_opportunity_mode and not is_deficit:
                maint_entry_ok = False

            # Flat + sufficient obs: reduce quoting cadence — save volume for core books.
            # Only quote on refresh ticks; no boost quoting for idle healthy maintenance books.
            maint_is_idle = (not is_deficit) and abs(net) < min_qty
            if maint_is_idle and not refresh_quotes:
                # Between refreshes: only clean up stale orders; no new quotes.
                if budget_left():
                    stale_all = self._stale_order_ids(account.orders, OrderDirection.BUY, mid - tick, tick)
                    stale_all += self._stale_order_ids(account.orders, OrderDirection.SELL, mid + tick, tick)
                    if stale_all:
                        response.cancel_orders(book_id, stale_all[:6])
                        spend()
                return
            if not refresh_quotes and not is_deficit:
                # Non-idle non-deficit maintenance between refreshes: stale cleanup only.
                if budget_left():
                    stale_all = self._stale_order_ids(account.orders, OrderDirection.BUY, mid - tick, tick)
                    stale_all += self._stale_order_ids(account.orders, OrderDirection.SELL, mid + tick, tick)
                    if stale_all:
                        response.cancel_orders(book_id, stale_all[:6])
                        spend()
                return

            if not throttled and budget_left():
                # Compute maintenance quotes: wider than core (add extra tick for safety).
                maint_bid_half = self._fee_aware_half_spread_ticks(
                    account, book, mid, tick, max(st.edge_mult, 1.0)
                ) + 1
                maint_ask_half = maint_bid_half
                reservation = mid
                maint_bid = _floor_tick(reservation - maint_bid_half * tick, tick, cfg.priceDecimals)
                maint_ask = _ceil_tick(reservation + maint_ask_half * tick, tick, cfg.priceDecimals)
                if maint_bid >= maint_ask:
                    maint_bid = _floor_tick(mid - tick, tick, cfg.priceDecimals)
                    maint_ask = _ceil_tick(mid + tick, tick, cfg.priceDecimals)
                best_bid = book.bids[0].price
                best_ask = book.asks[0].price
                maint_bid = min(maint_bid, best_ask - tick)
                maint_ask = max(maint_ask, best_bid + tick)

                # Cancel stale orders.
                stale_bid = self._stale_order_ids(account.orders, OrderDirection.BUY, maint_bid, tick)
                stale_ask = self._stale_order_ids(account.orders, OrderDirection.SELL, maint_ask, tick)
                if (stale_bid or stale_ask) and budget_left():
                    response.cancel_orders(book_id, (stale_bid + stale_ask)[:6])
                    spend()

                # Place bid if entry is permitted by staged gate.
                if (
                    maint_entry_ok
                    and budget_left()
                    and self._allows_entry_quote(st, OrderDirection.BUY, sim_ts, net, throttled=False)
                    and self._order_at_price(account.orders, OrderDirection.BUY, maint_bid, tick) is None
                ):
                    bid_qty = _round_qty(order_qty * self._toxic_qty_scale(st, OrderDirection.BUY), cfg.volumeDecimals)
                    bid_qty = self._affordable_qty(account, OrderDirection.BUY, maint_bid, bid_qty, cfg.volumeDecimals)
                    if bid_qty >= min_qty:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=bid_qty,
                            price=maint_bid,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()

                # Place ask if entry is permitted by staged gate.
                if (
                    maint_entry_ok
                    and budget_left()
                    and self._allows_entry_quote(st, OrderDirection.SELL, sim_ts, net, throttled=False)
                    and self._order_at_price(account.orders, OrderDirection.SELL, maint_ask, tick) is None
                ):
                    ask_qty = _round_qty(order_qty * self._toxic_qty_scale(st, OrderDirection.SELL), cfg.volumeDecimals)
                    ask_qty = self._affordable_qty(account, OrderDirection.SELL, maint_ask, ask_qty, cfg.volumeDecimals)
                    if ask_qty >= min_qty:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=ask_qty,
                            price=maint_ask,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()
            return

        maker_fee = account.fees.maker_fee_rate if account.fees else 0.0
        if maker_fee > self.max_maker_fee_rate:
            cancel_ids: list[int] = []
            for side in (OrderDirection.BUY, OrderDirection.SELL):
                touch = book.bids[0].price if side == OrderDirection.BUY else book.asks[0].price
                for order in account.orders:
                    if order.side == side and order.price is not None and abs(order.price - touch) <= tick:
                        cancel_ids.append(order.id)
            if cancel_ids and budget_left():
                response.cancel_orders(book_id, cancel_ids[:4])
                spend()
            return

        if at_hard_stop:
            # Cancel entry-side orders only; exit-side resting orders are allowed.
            entry_side = OrderDirection.BUY if net > 0 else OrderDirection.SELL if net < 0 else None
            if entry_side is not None:
                entry_ids = self._side_order_ids(account.orders, entry_side)
                if entry_ids and budget_left():
                    response.cancel_orders(book_id, entry_ids[:8])
                    spend()
            else:
                stale_all = [o.id for o in account.orders]
                if stale_all and budget_left():
                    response.cancel_orders(book_id, stale_all[:8])
                    spend()
            return

        # Global drain mode: cancel entry-side orders, no new MM quotes.
        if global_drain_mode:
            entry_side = OrderDirection.BUY if net > 0 else OrderDirection.SELL if net < 0 else None
            if entry_side is not None:
                entry_ids = self._side_order_ids(account.orders, entry_side)
                if entry_ids and budget_left():
                    response.cancel_orders(book_id, entry_ids[:8])
                    spend()
            return

        # NEGATIVE_RECOVERY: negative core books get a probe path instead of permanent ban.
        core_is_negative = (
            obs.negative_count() > obs.positive_count() or obs.net_realized() < 0.0
        )
        if core_is_negative:
            # Always cancel entry-side resting orders first.
            if refresh_quotes and budget_left():
                for _neg_side in (OrderDirection.BUY, OrderDirection.SELL):
                    if not self._is_entry_side(_neg_side, net):
                        continue
                    _neg_ids = self._side_order_ids(account.orders, _neg_side)
                    if _neg_ids and budget_left():
                        response.cancel_orders(book_id, _neg_ids[:8])
                        spend()

            # Accumulate stats for NEGATIVE_SUMMARY.
            _neg_loss_budget = self._loss_budget_quote(cfg.miner_wealth)
            _is_hard_neg = self._is_hard_negative(obs, st, _neg_loss_budget)
            self._neg_summary_acc(
                validator_key, book_id,
                active_neg=True,
                hard_neg=_is_hard_neg,
            )

            if _is_hard_neg:
                # Too damaged — no probe, pure exit-only.
                return

            # Check consecutive failure cooldown.
            # rg_neg_probe_faster: halve effective cooldown so deficit-light state recovers faster.
            _neg_cd_offset = (
                self.recovery_probe_cooldown_ns
                - self.recovery_probe_cooldown_ns // self.risk_gate_neg_probe_faster_factor
                if rg_neg_probe_faster else 0
            )
            if st.neg_probe_consecutive_neg >= self.recovery_probe_max_consecutive_neg:
                if sim_ts < st.neg_probe_cooldown_until_ts - _neg_cd_offset:
                    return
                st.neg_probe_consecutive_neg = 0

            # Check per-book probe cooldown.
            if sim_ts < st.neg_probe_cooldown_until_ts - _neg_cd_offset:
                return

            # Only probe if we have enough budget and it's a refresh tick.
            if not refresh_quotes or not budget_left():
                return

            # Eligibility: no open inventory past soft_limit (avoid doubling down).
            _neg_soft = max(self.order_qty, max_base * self.inventory_soft_ratio)
            if abs(net) >= _neg_soft:
                return

            # Choose probe side: opposite to existing inventory to reduce exposure.
            if net > min_qty:
                _probe_side = OrderDirection.SELL
            elif net < -min_qty:
                _probe_side = OrderDirection.BUY
            else:
                # Flat — probe the side that historically lost less (or buy by default).
                _probe_side = OrderDirection.BUY

            # Compute probe price: at least recovery_probe_min_spread_ticks inside mid.
            _probe_ticks = max(self.recovery_probe_min_spread_ticks, self.edge_ticks)
            if _probe_side == OrderDirection.BUY:
                _probe_px = mid - _probe_ticks * tick
            else:
                _probe_px = mid + _probe_ticks * tick
            if tick > 0:
                _probe_px = round(_probe_px / tick) * tick
            _probe_px = round(_probe_px, cfg.priceDecimals)

            _probe_qty = _round_qty(
                max(min_qty, order_qty * self.recovery_probe_size_mult),
                cfg.volumeDecimals,
            )
            _probe_qty = self._affordable_qty(
                account, _probe_side, _probe_px, _probe_qty, cfg.volumeDecimals
            )

            if _probe_qty >= min_qty and budget_left():
                response.limit_order(
                    book_id=book_id,
                    direction=_probe_side,
                    quantity=_probe_qty,
                    price=_probe_px,
                    stp=STP.CANCEL_BOTH,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=self.expiry_period,
                    postOnly=True,
                )
                spend()
                st.neg_probe_cooldown_until_ts = sim_ts + self.recovery_probe_cooldown_ns
                self._neg_probed_books.setdefault(validator_key, set()).add(book_id)
                _pct = self._neg_probe_counts.setdefault(validator_key, {})
                _pct[book_id] = _pct.get(book_id, 0) + 1
                self._neg_summary_acc(validator_key, book_id, recovery_candidate=True, probe_sent=True)
                bt.logging.debug(
                    f"[NEGATIVE_RECOVERY] book={book_id} probe_side={_probe_side}"
                    f" probe_px={_probe_px} probe_qty={_probe_qty:.4f}"
                    f" neg_count={obs.negative_count()} pos_count={obs.positive_count()}"
                    f" net_realized={obs.net_realized():.6f}"
                )
            else:
                self._neg_summary_acc(validator_key, book_id, recovery_candidate=True)
            return

        bid_px, ask_px = self._fair_quotes(
            book, account, st, tick, cfg.priceDecimals, mid, max_base
        )

        if tick > 0 and self.log_detail:
            bid_half_ticks = int(round((mid - bid_px) / tick))
            ask_half_ticks = int(round((ask_px - mid) / tick))
            spread_ticks = bid_half_ticks + ask_half_ticks
            maker_fee, taker_fee = self._fee_rates(account)
            bt.logging.debug(
                f"BOOK {book_id} QUOTE"
                f" edge_ticks={self.edge_ticks}"
                f" bid_half={bid_half_ticks} ask_half={ask_half_ticks}"
                f" spread_ticks={spread_ticks}"
                f" maker_fee={maker_fee:.5f} taker_fee={taker_fee:.5f}"
                f" inv={net:.4f} edge_mult={st.edge_mult:.3f}"
                f" bid_tox={st.bid_toxic:.3f} ask_tox={st.ask_toxic:.3f}"
            )

        # Every tick: strip entry-side quotes when throttle or toxic/cooldown blocks.
        self._pull_blocked_entry_orders(
            response, book_id, account, st, sim_ts, net, throttled, budget_left, spend
        )

        if not refresh_quotes:
            return

        if refresh_quotes and budget_left():
            if throttled:
                # Throttle: cancel stale quotes on exit side only.
                if net > 0:
                    stale_ids = self._stale_order_ids(
                        account.orders, OrderDirection.SELL, ask_px, tick
                    )
                elif net < 0:
                    stale_ids = self._stale_order_ids(
                        account.orders, OrderDirection.BUY, bid_px, tick
                    )
                else:
                    stale_ids = []
            else:
                stale_ids = (
                    self._stale_order_ids(account.orders, OrderDirection.BUY, bid_px, tick)
                    + self._stale_order_ids(account.orders, OrderDirection.SELL, ask_px, tick)
                )
            if stale_ids:
                response.cancel_orders(book_id, stale_ids[:6])
                spend()

        if throttled:
            # Throttle: no new entry makers; exit-side maker allowed.
            if net > 0 and budget_left():
                if self._order_at_price(account.orders, OrderDirection.SELL, ask_px, tick) is None:
                    ask_qty = side_qty(OrderDirection.SELL, ask_px, order_qty)
                    if ask_qty >= min_qty:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=ask_qty,
                            price=ask_px,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()
                        st.quoted_ask = ask_px
            elif net < 0 and budget_left():
                if self._order_at_price(account.orders, OrderDirection.BUY, bid_px, tick) is None:
                    bid_qty = side_qty(OrderDirection.BUY, bid_px, order_qty)
                    if bid_qty >= min_qty:
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=bid_qty,
                            price=bid_px,
                            stp=STP.CANCEL_BOTH,
                            timeInForce=TimeInForce.GTT,
                            expiryPeriod=self.expiry_period,
                            postOnly=True,
                        )
                        spend()
                        st.quoted_bid = bid_px
                        bk = self._short_exit_order_count.setdefault(validator_key, {})
                        bk[book_id] = bk.get(book_id, 0) + 1
            return

        # LOW_OPPORTUNITY_MODE: suppress new entry quotes; stale cancel + exit already done above.
        if self.intensity_enabled and low_opportunity_mode:
            return

        if not forced_unwind_mode and budget_left() and self._allows_entry_quote(
            st, OrderDirection.BUY, sim_ts, net, throttled=False
        ):
            if self._order_at_price(account.orders, OrderDirection.BUY, bid_px, tick) is None:
                bid_qty = side_qty(OrderDirection.BUY, bid_px, order_qty)
                if bid_qty >= min_qty:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=bid_qty,
                        price=bid_px,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period,
                        postOnly=True,
                    )
                    spend()
                    st.quoted_bid = bid_px

        if not forced_unwind_mode and budget_left() and self._allows_entry_quote(
            st, OrderDirection.SELL, sim_ts, net, throttled=False
        ):
            if self._order_at_price(account.orders, OrderDirection.SELL, ask_px, tick) is None:
                ask_qty = side_qty(OrderDirection.SELL, ask_px, order_qty)
                if ask_qty >= min_qty:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=ask_qty,
                        price=ask_px,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period,
                        postOnly=True,
                    )
                    spend()
                    st.quoted_ask = ask_px

    def get_dashboard(self):
        """Serve the dashboard HTML page."""
        from fastapi.responses import HTMLResponse
        import os
        html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        with open(html_path, "r") as f:
            content = f.read()
        return HTMLResponse(content=content)

    def get_stats(self):
        """Dashboard stats endpoint: compute per-agent metrics matching mainnet dashboard."""
        import numpy as np
        from taos.im.utils.kappa import kappa_3

        sim_ts = self._sim_ts
        book_count = self._book_count

        kappa_norm_min = -2.5
        kappa_norm_max = 2.5
        kappa_tau = 0.0
        lookback_steps = self.kappa_lookback_ns  # nanoseconds — matches validator call convention
        min_lookback = 5_400_000_000_000         # 1.5 hours in ns — matches validator default
        max_inactive_ratio = 1.0 - self.core_target_ratio

        agents_out = []
        for vkey in list(self._realized_obs.keys()):
            obs_map = self._realized_obs[vkey]

            # Build realized_pnl_history in the format kappa_3 expects: {ts: {book_id: pnl}}
            pnl_history: dict[int, dict[int, float]] = {}
            for book_id, obs in obs_map.items():
                for ts, pnl in obs.obs_by_ts.items():
                    pnl_history.setdefault(ts, {})[book_id] = pnl

            # Realized PnL
            total_realized_pnl = sum(
                obs.net_realized() for obs in obs_map.values()
            )

            # Roundtrip volume (24H RT QUOTE) and total volume (maker + taker, 24H QUOTE)
            # vol_24h = min per-book (maker+taker) matching mainnet dashboard metric.
            # rt_vol_24h = roundtrip volume of that min-volume book.
            rt_map = self._roundtrip_volume.get(vkey, {})
            mk_map = self._maker_volume.get(vkey, {})
            tk_map = self._taker_volume.get(vkey, {})
            all_book_ids = set(mk_map) | set(tk_map)
            per_book_vol = {
                bid: mk_map.get(bid, 0.0) + tk_map.get(bid, 0.0)
                for bid in all_book_ids
            }
            active_vols = {bid: v for bid, v in per_book_vol.items() if v > 0.0}
            if active_vols:
                min_vol_bid = min(active_vols, key=lambda b: active_vols[b])
                total_mk_vol = active_vols[min_vol_bid]
                total_rt_vol = rt_map.get(min_vol_bid, 0.0)
            else:
                total_mk_vol = 0.0
                total_rt_vol = 0.0

            # Kappa3 computation
            kappa_result = None
            median_kappa3 = None
            kappa_penalty = None
            kappa3_score = None
            if pnl_history:
                try:
                    kappa_result = kappa_3(
                        uid=0,
                        realized_pnl_values=pnl_history,
                        tau=kappa_tau,
                        lookback=lookback_steps,
                        norm_min=kappa_norm_min,
                        norm_max=kappa_norm_max,
                        min_lookback=min_lookback,
                        min_realized_observations=self.min_realized_observations,
                        grace_period=0,
                        deregistered_uids=[],
                        book_count=book_count,
                        cache=None,
                    )
                except Exception:
                    kappa_result = None

            if kappa_result and kappa_result.get("books"):
                books_raw = kappa_result["books"]
                norm_range = kappa_norm_max - kappa_norm_min
                norm_inv = 1.0 / norm_range if norm_range else 0.0
                norm_kappas = {}
                for bid, v in books_raw.items():
                    if v is not None:
                        norm_kappas[bid] = max(0.0, min(1.0, (v - kappa_norm_min) * norm_inv))

                all_books = list(range(book_count))
                scores_with = [norm_kappas[b] for b in all_books if b in norm_kappas]
                books_no_kappa = [b for b in all_books if b not in norm_kappas]
                max_inactive = int(max_inactive_ratio * book_count)
                if len(books_no_kappa) <= max_inactive:
                    data = np.array(scores_with) if scores_with else np.array([0.0])
                else:
                    excess = len(books_no_kappa) - max_inactive
                    data = np.array(scores_with + [0.0] * excess) if scores_with else np.array([0.0] * excess)

                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                eff_iqr = max(iqr, 0.01)
                lower = q1 - 1.5 * eff_iqr
                outliers = data[data < lower]
                if len(outliers) > 0 and np.median(outliers) < 0.5:
                    base_pen = (0.5 - np.median(outliers)) / 1.5
                    cons_bonus = 1.0 - np.exp(-5 * iqr)
                    kappa_penalty = float(base_pen * cons_bonus)
                else:
                    kappa_penalty = 0.0

                median_kappa3 = float(np.median(data))
                kappa3_score = float(max(0.0, median_kappa3 - kappa_penalty))

            # Activity factor: clamp(rt_volume / activity_cap, 0, 1)
            cfg = getattr(self, "simulation_config", None)
            miner_wealth = float(getattr(cfg, "miner_wealth", 50000.0)) if cfg else 50000.0
            vol_dec = int(getattr(cfg, "volumeDecimals", 4)) if cfg else 4
            activity_cap = miner_wealth * self.capital_turnover_cap / book_count
            activity = float(min(1.0, total_rt_vol / activity_cap)) if activity_cap > 0 else 0.0

            # Trading score = kappa only (pnl weight = 0 unless configured)
            trading_score = kappa3_score if kappa3_score is not None else 0.0

            # Account balances (from self.accounts = {book_id: account})
            accounts = getattr(self, "accounts", {})
            total_base = sum(
                float(getattr(getattr(acc, "base_balance", None) or {}, "total", 0.0) or 0.0)
                for acc in accounts.values()
            )
            total_base_loan = sum(
                float(getattr(acc, "base_loan", None) or 0.0)
                for acc in accounts.values()
            )
            total_base_coll = sum(
                float(getattr(acc, "base_collateral", None) or 0.0)
                for acc in accounts.values()
            )

            # Inventory (sum of net BASE positions across books)
            total_inventory = sum(
                self._book_st(vkey, bid).inventory
                for bid in self._book_state.get(vkey, {})
            )

            # Per-book fill / order counter maps
            bid_fc = self._bid_fill_count.get(vkey, {})
            ask_fc = self._ask_fill_count.get(vkey, {})
            sx_fill = self._short_exit_fill_count.get(vkey, {})
            sx_ord  = self._short_exit_order_count.get(vkey, {})

            # Per-book details — only include books with actual activity.
            # obs_map keys are excluded from active detection (empty entries created for all books).
            mk_map_all = self._maker_volume.get(vkey, {})
            tk_map_all = self._taker_volume.get(vkey, {})
            active_bids = (
                {b for b, v in mk_map_all.items() if v > 0}
                | {b for b, v in tk_map_all.items() if v > 0}
                | {b for b, v in bid_fc.items() if v > 0}
                | {b for b, v in ask_fc.items() if v > 0}
                | {b for b, obs in obs_map.items() if obs.non_zero_count() > 0 or obs.longs or obs.shorts}
            )
            books_out = []
            for bid in sorted(active_bids):
                obs = obs_map.get(bid)
                if obs is None:
                    from dataclasses import dataclass
                    obs = RealizedObsState()
                bst = self._book_st(vkey, bid)
                kappa_per_book = None
                raw_kappa = None
                if kappa_result and bid in kappa_result.get("books", {}):
                    raw_kappa = kappa_result["books"][bid]
                    if raw_kappa is not None:
                        kappa_per_book = round(max(0.0, min(1.0, (raw_kappa - kappa_norm_min) * (1.0 / (kappa_norm_max - kappa_norm_min)))), 4)
                # Inventory age: age of oldest open FIFO lot (seconds)
                inv_age_s = 0.0
                obs_state = self._realized_obs.get(vkey, {}).get(bid)
                if obs_state is not None and sim_ts:
                    if obs_state.longs:
                        inv_age_s = round((sim_ts - obs_state.longs[0][0]) / 1e9, 1)
                    elif obs_state.shorts:
                        inv_age_s = round((sim_ts - obs_state.shorts[0][0]) / 1e9, 1)
                books_out.append({
                    "book_id": bid,
                    "realized_pnl": round(obs.net_realized(), 6),
                    "non_zero_obs": obs.non_zero_count(),
                    "inventory": round(bst.inventory, 6),
                    "kappa3": kappa_per_book,
                    "kappa3_raw": round(raw_kappa, 4) if raw_kappa is not None else None,
                    "bid_toxic": round(bst.bid_toxic, 4),
                    "ask_toxic": round(bst.ask_toxic, 4),
                    "rt_vol": round(rt_map.get(bid, 0.0), 4),
                    "mk_vol": round(mk_map.get(bid, 0.0), 4),
                    "bid_fill_count": bid_fc.get(bid, 0),
                    "ask_fill_count": ask_fc.get(bid, 0),
                    "short_exit_fill_count": sx_fill.get(bid, 0),
                    "short_exit_order_count": sx_ord.get(bid, 0),
                    "inventory_age_s": inv_age_s,
                })

            # Summary analytics
            import statistics as _stats
            sorted_by_inv = sorted(books_out, key=lambda b: b["inventory"])
            top_short_books = [
                {"book_id": b["book_id"], "inventory": b["inventory"],
                 "inventory_age_s": b["inventory_age_s"], "realized_pnl": b["realized_pnl"]}
                for b in sorted_by_inv[:5] if b["inventory"] < 0
            ]
            top_long_books = [
                {"book_id": b["book_id"], "inventory": b["inventory"],
                 "inventory_age_s": b["inventory_age_s"], "realized_pnl": b["realized_pnl"]}
                for b in sorted_by_inv[-5:][::-1] if b["inventory"] > 0
            ]
            negative_kappa_books = [
                b["book_id"] for b in books_out
                if b.get("kappa3_raw") is not None and b["kappa3_raw"] < 0
            ]
            pnl_values = [b["realized_pnl"] for b in books_out]
            median_book_realized_pnl = round(_stats.median(pnl_values), 6) if pnl_values else 0.0

            agents_out.append({
                "uid": self.uid,
                "validator": vkey,
                "vol_24h": round(total_mk_vol, 2),
                "rt_vol_24h": round(total_rt_vol, 2),
                "activity": round(activity, 4),
                "realized_pnl": round(total_realized_pnl, 4),
                "median_kappa3": round(median_kappa3, 4) if median_kappa3 is not None else None,
                "kappa_penalty": round(kappa_penalty, 4) if kappa_penalty is not None else None,
                "kappa3_score": round(kappa3_score, 4) if kappa3_score is not None else None,
                "trading_score": round(trading_score, 4),
                "gentrx_score": 0.0,
                "score": round(trading_score, 4),
                "delta_inv": round(total_inventory, 6),
                "base_balance": round(total_base, 4),
                "base_loan": round(total_base_loan, 4),
                "base_coll": round(total_base_coll, 4),
                "top_short_books": top_short_books,
                "top_long_books": top_long_books,
                "negative_kappa_books": negative_kappa_books,
                "median_book_realized_pnl": median_book_realized_pnl,
                "books": books_out,
            })

        return {
            "sim_ts": sim_ts,
            "sim_time": duration_from_timestamp(sim_ts) if sim_ts else "00:00:00",
            "book_count": book_count,
            "agents": agents_out,
        }

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        ctx_tokens = self._bind_request_context_if_needed(state)
        try:
            return self._respond_impl(state)
        finally:
            self._release_request_context(ctx_tokens)

    def _respond_impl(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        self._unwind_log_buffer = []  # reset per tick

        self._memsnap_tick += 1
        if self._memsnap_tick % 3000 == 0:
            gc.collect()
            rss_mb = 0
            try:
                with open(f"/proc/{os.getpid()}/status") as _f:
                    for _ln in _f:
                        if _ln.startswith("VmRSS:"):
                            rss_mb = int(_ln.split()[1]) / 1024
                            break
            except Exception:
                pass
            import collections
            counts = collections.Counter(type(o).__name__ for o in gc.get_objects())
            top = counts.most_common(15)
            lines = [f"[MEMSNAP] tick={self._memsnap_tick} RSS={rss_mb:.0f}MB gc_objects={sum(counts.values())}"]
            for name, cnt in top:
                lines.append(f"  {cnt:>8d}  {name}")
            bt.logging.info("\n".join(lines))

        cfg = state.config
        sim_ts = state.timestamp
        validator_key = self._set_validator_context(state)

        # First tick per validator (or after reset/RDRA): run 1-shot baseline reconcile.
        if validator_key not in self._first_tick_done:
            self._first_tick_done.add(validator_key)
            self._pending_baseline_reconcile.add(validator_key)
        if validator_key in self._pending_baseline_reconcile:
            self._pending_baseline_reconcile.discard(validator_key)
            self._run_baseline_reconcile(validator_key, cfg)

        tick = self._tick_size(cfg.priceDecimals)
        volume_cap = self._volume_cap(cfg.miner_wealth, cfg.volumeDecimals)
        throttle_cap = volume_cap * self.volume_throttle_ratio
        hard_stop_cap = volume_cap * self.volume_hard_stop_ratio


        instr_used: dict[int, int] = {}

        selection = self._book_selection_for(
            validator_key, cfg.book_count, cfg, sim_ts, state
        )
        dropped_pending = self._pending_dropped_books.pop(validator_key, set())
        selection = self._cleanup_dropped_books(
            response, validator_key, dropped_pending, selection, cfg, instr_used
        )
        traded_books = selection.traded()
        self._process_due_markouts(validator_key, state, tick)

        # Profit-core: top profit_core_n books within core tier by realized PnL + markout quality.
        profit_core_books = self._classify_profit_core(validator_key, selection, sim_ts)

        quote_interval_ns = self.quote_interval * cfg.publish_interval
        last_refresh = self._last_quote_refresh_ts.get(validator_key, 0)
        refresh_quotes = (
            last_refresh == 0
            or sim_ts - last_refresh >= quote_interval_ns
        )

        boost_books = self._classify_laggards(
            validator_key, selection, sim_ts, hard_stop_cap
        )

        # Pattern A/B detection: if all non-zero account.traded_volume values are within 1%
        # of each other the validator is mirroring a single global value across books
        # (Pattern A) — use local per-book maker+taker tracker to avoid treating the global
        # mirrored value as each book's individual cap.  If values differ (Pattern B), each
        # book's own value is meaningful and is used directly.
        _pat_acc_vols = [
            self._book_volume_traded(acc, validator_key, bid)
            for bid, acc in self.accounts.items()
            if acc is not None
        ]
        _pat_nonzero = [v for v in _pat_acc_vols if v > 0.0]
        if _pat_nonzero:
            _pat_lo, _pat_hi = min(_pat_nonzero), max(_pat_nonzero)
            _book_vol_source = (
                "local_tracker"
                if (_pat_lo > 0 and (_pat_hi - _pat_lo) / _pat_hi <= 0.01)
                else "per_book"
            )
        else:
            _book_vol_source = "unknown"

        def _book_inventory(bid: int) -> float:
            if bid not in self.accounts:
                return 0.0
            return self._net_inventory(validator_key, bid, self.accounts[bid], cfg)

        unwind_min_qty_g = max(self.unwind_min_qty, 10.0 ** (-cfg.volumeDecimals))
        exchange_min_g = _round_qty(self.min_order_size, cfg.volumeDecimals)

        # Scan all traded books for old inventory (age >= max_holding_time_ns).
        # Exclude stranded positions (below exchange minOrderSize) — market exit is impossible.
        old_unwind_candidates: list[tuple[int, int, float]] = []  # (book_id, age_ns, abs_net)
        for bid in traded_books:
            if bid not in self.accounts:
                continue
            acc_scan = self.accounts[bid]
            if acc_scan is None:
                continue
            st_scan = self._book_st(validator_key, bid)
            net_scan = self._net_inventory(validator_key, bid, acc_scan, cfg)
            abs_net = abs(net_scan)
            if abs_net >= unwind_min_qty_g and abs_net >= exchange_min_g and st_scan.inventory_open_ts > 0:
                age_ns = sim_ts - st_scan.inventory_open_ts
                if age_ns >= self.max_holding_time_ns:
                    old_unwind_candidates.append((bid, age_ns, abs_net))

        old_unwind_candidates.sort(key=lambda x: x[1], reverse=True)
        max_age_ns = old_unwind_candidates[0][1] if old_unwind_candidates else 0

        global_drain_mode = (
            len(old_unwind_candidates) >= self.global_drain_min_books
            or max_age_ns >= self.global_drain_old_age_ns
        )

        selected_unwind_books: set[int] = set()
        if global_drain_mode:
            selected_unwind_books = {
                bid for bid, _, _ in old_unwind_candidates[: self.max_forced_unwind_books_per_tick]
            }
            bt.logging.debug(
                f"[GLOBAL_DRAIN] active=True old_books={len(old_unwind_candidates)}"
                f" max_age_s={max_age_ns / 1e9:.1f} selected={len(selected_unwind_books)}"
            )
            if self.log_detail:
                bt.logging.info(f"[GLOBAL_DRAIN_SELECTED] books={sorted(selected_unwind_books)}")
                for rank, (bid, age_ns, abs_net) in enumerate(
                    old_unwind_candidates[: self.max_forced_unwind_books_per_tick], start=1
                ):
                    bt.logging.info(
                        f"[FORCED_UNWIND_PACED] book={bid} rank={rank}"
                        f" age_s={age_ns / 1e9:.1f} inv={abs_net:.4f}"
                    )
                for bid, age_ns, abs_net in old_unwind_candidates[self.max_forced_unwind_books_per_tick :]:
                    bt.logging.debug(
                        f"[GLOBAL_DRAIN_SKIP_UNWIND] book={bid} reason=pace_limit"
                        f" inv={abs_net:.4f} age_s={age_ns / 1e9:.1f}"
                    )

        # Priority: Kappa obs deficit + account inventory (not FIFO net).
        traded_sorted = sorted(
            traded_books,
            key=lambda bid: self._guard_priority(
                self._obs(validator_key, bid), _book_inventory(bid), sim_ts, bid
            ),
            reverse=True,
        )

        old_unwind_set = {bid for bid, _, _ in old_unwind_candidates}

        # --- Adaptive intensity: pre-compute per-book class before main MM loop ---
        # Runs in both shadow and enabled modes; only enabled mode applies results to orders.
        _class_map = self._book_intensity_last.setdefault(validator_key, {})
        _idata = self._intensity_interval_data.setdefault(validator_key, {
            "below_obs": set(),
            "floor_probe": set(),
            "hard_neg": set(),
            "low_opp_ticks": 0,
            "net_realized_sum": 0.0,
            "net_realized_n": 0,
        })
        _intensity_vals: dict[int, tuple[str, float]] = {}  # book_id → (class, value)
        _cur_below_obs = 0    # current-tick count of books below min_realized_observations
        _tick_nr_sum = 0.0    # current-tick net_realized sum (for risk gate R4)
        _tick_nr_n = 0
        _risk_gate: dict | None = None
        if self.intensity_shadow_mode or self.intensity_enabled:
            _all_accounts_i = self.accounts if self.accounts else {}
            _neg_loss_budget_i = self._loss_budget_quote(cfg.miner_wealth)
            for _i_bid in traded_sorted:
                if _i_bid not in state.books:
                    continue
                _i_st = self._book_st(validator_key, _i_bid)
                _i_obs = self._obs(validator_key, _i_bid)
                _i_obs.prune(sim_ts, self.kappa_lookback_ns)
                _i_acc = _all_accounts_i.get(_i_bid)
                _i_net = self._net_inventory(validator_key, _i_bid, _i_acc, cfg) if _i_acc else 0.0
                _intensity_v = self._compute_book_intensity(
                    validator_key, _i_bid, _i_obs, _i_st, sim_ts, _i_net
                )
                _cls = self._intensity_class(_intensity_v)
                _class_map[_i_bid] = _cls
                _intensity_vals[_i_bid] = (_cls, _intensity_v)
                # Interval-level unique-book sets
                _i_eff_obs = _i_obs.effective_non_zero_count(self.changeover_grace_period_ns)
                if _i_eff_obs < self.min_realized_observations:
                    _idata["below_obs"].add(_i_bid)
                    _cur_below_obs += 1
                stale_threshold = sim_ts - self.participation_floor_stale_ns
                if _i_st.last_trade_ts > 0 and _i_st.last_trade_ts < stale_threshold:
                    _idata["floor_probe"].add(_i_bid)
                if self._is_hard_negative(_i_obs, _i_st, _neg_loss_budget_i):
                    _idata["hard_neg"].add(_i_bid)
                _nr_val = _i_obs.net_realized()
                _idata["net_realized_sum"] += _nr_val
                _idata["net_realized_n"] += 1
                _tick_nr_sum += _nr_val
                _tick_nr_n += 1
            # LOW_OPPORTUNITY_MODE: counted per-tick for the interval summary
            _agg_now = sum(1 for c in _class_map.values() if c == "aggressive")
            _total_now = max(1, len(_class_map))
            if _agg_now / _total_now < self.low_opportunity_aggressive_frac:
                _idata["low_opp_ticks"] = _idata.get("low_opp_ticks", 0) + 1
            _low_opp_mode_tick = (
                self.intensity_enabled
                and _agg_now / _total_now < self.low_opportunity_aggressive_frac
            )
            # Auto risk gate: computed once per tick when intensity is enabled
            if self.intensity_enabled and self.risk_gate_enabled:
                _tick_nr_avg = _tick_nr_sum / max(1, _tick_nr_n)
                _low_opp_frac_below = _agg_now / _total_now < self.low_opportunity_aggressive_frac
                _prev_fu = self._risk_gate_prev_forced_unwind.get(validator_key, 0)
                _risk_gate = self._compute_risk_gate(
                    global_drain_mode=global_drain_mode,
                    old_inv_books=len(old_unwind_candidates),
                    books_below_min_obs=_cur_below_obs,
                    tick_avg_net_realized=_tick_nr_avg,
                    prev_forced_unwind_count=_prev_fu,
                    low_opp_fraction_below_threshold=_low_opp_frac_below,
                )
        else:
            _low_opp_mode_tick = False

        for book_id in traded_sorted:
            if book_id not in state.books:
                continue
            book = state.books[book_id]
            try:
                if book_id not in self.accounts:
                    continue
                if len(book.bids) == 0 or len(book.asks) == 0:
                    continue
                account = self.accounts[book_id]
                st = self._book_st(validator_key, book_id)
                self._log_book_mode(
                    validator_key, book_id, selection, st,
                    self._obs(validator_key, book_id), sim_ts,
                    unwind_min_qty_g,
                )
                is_old_inventory = book_id in old_unwind_set
                force_unwind_allowed = (not global_drain_mode) or (book_id in selected_unwind_books)
                _i_cls, _i_val = _intensity_vals.get(book_id, ("normal", 0.5))

                # Apply auto risk gate overrides (only when intensity_enabled=True)
                _eff_agg_mult_cap: float | None = None
                _eff_floor_interval: int | None = None
                _eff_low_opp = _low_opp_mode_tick
                # R8: always active (independent of intensity_enabled) — when deficit ≤ threshold,
                # halve neg probe cooldown so remaining negative books get fills faster.
                _eff_neg_probe_faster = (
                    0 < _cur_below_obs <= self.risk_gate_few_deficit_threshold
                )
                if self.intensity_enabled and _risk_gate is not None:
                    _eff_agg_mult_cap = _risk_gate["aggressive_qty_mult_cap"]
                    if _risk_gate["slow_floor_probe"]:
                        _eff_floor_interval = (
                            self.deficit_floor_probe_interval_ns * self.risk_gate_slow_probe_factor
                        )
                    if _risk_gate["full_low_opp_mode"]:
                        _eff_low_opp = True
                    # R4 (demote to light) takes precedence over R2 (demote to normal)
                    if _risk_gate["demote_to_light"] and _i_cls == "aggressive":
                        _i_cls = "light"
                    elif _risk_gate["ban_aggressive"] and _i_cls == "aggressive":
                        _i_cls = "normal"

                self._process_book_mm(
                    response, validator_key, book_id, book, account, st, cfg, tick, sim_ts,
                    refresh_quotes or (book_id in boost_books),
                    volume_cap, throttle_cap, hard_stop_cap, instr_used,
                    mm_core=book_id in selection.core,
                    global_drain_mode=global_drain_mode and not is_old_inventory,
                    force_unwind_allowed=force_unwind_allowed,
                    is_profit_core=book_id in profit_core_books,
                    volume_source=_book_vol_source,
                    intensity_class=_i_cls,
                    intensity_value=_i_val,
                    low_opportunity_mode=_eff_low_opp,
                    rg_aggressive_qty_mult_cap=_eff_agg_mult_cap,
                    rg_floor_probe_interval=_eff_floor_interval,
                    rg_neg_probe_faster=_eff_neg_probe_faster,
                )
            except Exception as ex:
                bt.logging.error(
                    f"BOOK {book_id}: error at {duration_from_timestamp(state.timestamp)} "
                    f"(T={state.timestamp}): {ex}\n{traceback.format_exc()}"
                )

        if refresh_quotes:
            self._last_quote_refresh_ts[validator_key] = sim_ts

        # Store forced_unwind count for next tick's risk gate (R2 check)
        self._risk_gate_prev_forced_unwind[validator_key] = len(self._unwind_log_buffer)

        # --- Drift summary log (rate-limited; fires when any quarantine drift occurred this tick) ---
        drift_buf = self._drift_summary_buf.pop(validator_key, None)
        if drift_buf and drift_buf.get("affected"):
            last_ds = self._last_drift_summary_ts.get(validator_key, 0)
            if sim_ts - last_ds >= self.drift_warn_rate_limit_ns:
                self._last_drift_summary_ts[validator_key] = sim_ts
                n_dq = len(self._drift_quarantine.get(validator_key, {}))
                n_sv = len(self._severe_drift_books.get(validator_key, set()))
                bt.logging.warning(
                    f"[DRIFT_SUMMARY] v={validator_key[:12]}"
                    f" drift_quarantine_count={n_dq}"
                    f" severe_drift_count={n_sv}"
                    f" affected_books={sorted(drift_buf['affected'])}"
                    f" max_abs_drift={drift_buf['max_drift']:.4f}"
                )

        # --- Tick summary logging ---
        _summary_interval = (
            max(self.log_summary_interval_ns, 300_000_000_000)
            if self.production_mode
            else self.log_summary_interval_ns
        )
        should_log_summary = (
            sim_ts - self._last_summary_ts.get(validator_key, 0) >= _summary_interval
        )
        if should_log_summary:
            self._last_summary_ts[validator_key] = sim_ts
            n_instr = len(response.instructions)
            market_count = sum(
                1 for i in response.instructions if isinstance(i, PlaceMarketOrderInstruction)
            )
            limit_count = sum(
                1 for i in response.instructions if isinstance(i, PlaceLimitOrderInstruction)
            )
            cancel_count = sum(
                1 for i in response.instructions if isinstance(i, CancelOrdersInstruction)
            )
            forced_count = len(self._unwind_log_buffer)
            if forced_count > 0:
                _dbuf_fu = self._deficit_recovery_buf.setdefault(validator_key, {
                    "floor_probes_sent": 0, "forced_unwind_count": 0,
                })
                _dbuf_fu["forced_unwind_count"] = _dbuf_fu.get("forced_unwind_count", 0) + forced_count
            active_books_count = sum(
                1 for bid in traded_books
                if bid in self.accounts
                and (
                    self._obs(validator_key, bid).non_zero_count() > 0
                    or abs(self._book_st(validator_key, bid).inventory) >= unwind_min_qty_g
                )
            )
            sim_s = sim_ts / 1e9
            bt.logging.info(
                f"[SUMMARY] t={sim_s:.0f} instr={n_instr} active_books={active_books_count}"
                f" old_inv_books={len(old_unwind_candidates)} max_age_s={max_age_ns / 1e9:.1f}"
                f" drain={global_drain_mode}"
                f" forced_unwind={forced_count} market_orders={market_count}"
                f" cancels={cancel_count} limit_orders={limit_count}"
            )
            if self._unwind_log_buffer:
                top = sorted(self._unwind_log_buffer, key=lambda x: x[4], reverse=True)[: self.log_max_books]
                oldest = [
                    (bid, f"{age_s:.0f}s", f"{abs(inv):.3f}", side, f"{qty:.4f}")
                    for bid, side, qty, inv, age_s in top
                ]
                bt.logging.info(f"[UNWIND_SUMMARY] count={forced_count} oldest={oldest}")

        # Volume computation for VOLUME_BUDGET log (informational only).
        _vb_mk_map = self._maker_volume.get(validator_key, {})
        _vb_tk_map = self._taker_volume.get(validator_key, {})
        def _bvol(bid: int) -> float:
            acc = self.accounts.get(bid)
            if acc is not None:
                v = self._book_volume_traded(acc, validator_key, bid)
                if v > 0:
                    return v
            return _vb_mk_map.get(bid, 0.0) + _vb_tk_map.get(bid, 0.0)
        profit_core_vol = sum(_bvol(b) for b in profit_core_books)
        safe_core_vol = sum(_bvol(b) for b in selection.core if b not in profit_core_books)
        maint_vol = sum(_bvol(b) for b in selection.reserve)

        # Periodic volume budget log (every volume_log_interval_ns sim time).
        last_vol_log = self._last_volume_log_ts.get(validator_key, 0)
        if sim_ts - last_vol_log >= self.volume_log_interval_ns:
            self._last_volume_log_ts[validator_key] = sim_ts
            emergency_reserved = volume_cap * self.emergency_buffer_ratio
            bt.logging.info(
                f"[VOLUME_BUDGET] v={validator_key[:12]}"
                f" profit_core_vol={profit_core_vol:.1f} (n={len(profit_core_books)})"
                f" safe_core_vol={safe_core_vol:.1f} (n={len(selection.core) - len(profit_core_books)})"
                f" maint_vol={maint_vol:.1f} (n={len(selection.reserve)})"
                f" emergency_reserved={emergency_reserved:.1f}"
                f" cap={volume_cap:.1f}"
            )
            if maint_vol > profit_core_vol and profit_core_vol > 0:
                bt.logging.warning(
                    f"[VOLUME_BUDGET_WARN] v={validator_key[:12]}"
                    f" maint_vol={maint_vol:.1f} > profit_core_vol={profit_core_vol:.1f}"
                    f" — maintenance is dominating profitable core books"
                )

        # NEGATIVE_SUMMARY: periodic log of recovery activity.
        last_neg_log = self._last_neg_summary_ts.get(validator_key, 0)
        if sim_ts - last_neg_log >= self.neg_summary_interval_ns:
            self._last_neg_summary_ts[validator_key] = sim_ts
            _ns_buf = self._neg_summary_buf.get(validator_key, {})
            _ns_probed = self._neg_probed_books.get(validator_key, set())
            _ns_recovered = 0
            for _rb in list(_ns_probed):
                _rb_obs = self._obs(validator_key, _rb)
                _rb_obs.prune(sim_ts, self.kappa_lookback_ns)
                if _rb_obs.positive_count() >= _rb_obs.negative_count() and _rb_obs.net_realized() >= 0.0:
                    _ns_recovered += 1
            _ns_an = _ns_buf.get("active_negative", set())
            _ns_rc = _ns_buf.get("recovery_candidates", set())
            _ns_hn = _ns_buf.get("hard_negative", set())
            _pct_map = self._neg_probe_counts.get(validator_key, {})
            _top5 = sorted(_pct_map.items(), key=lambda x: -x[1])[:5]
            _top5_str = " ".join(f"b{b}:{n}" for b, n in _top5) if _top5 else "none"
            bt.logging.info(
                f"[NEGATIVE_SUMMARY] v={validator_key[:12]}"
                f" active_negative={len(_ns_an) if isinstance(_ns_an, set) else _ns_an}"
                f" recovery_candidates={len(_ns_rc) if isinstance(_ns_rc, set) else _ns_rc}"
                f" probes_sent={_ns_buf.get('probes_sent', 0)}"
                f" recovered={_ns_recovered}"
                f" hard_negative={len(_ns_hn) if isinstance(_ns_hn, set) else _ns_hn}"
                f" tracked_probed={len(_ns_probed)}"
                f" top_probed=[{_top5_str}]"
            )
            # Reset per-interval counters but keep probed set for recovery tracking.
            self._neg_summary_buf[validator_key] = {
                "active_negative": set(),
                "recovery_candidates": set(),
                "probes_sent": 0,
                "recovered": set(),
                "hard_negative": set(),
            }

        # INTENSITY_SUMMARY: periodic log of adaptive intensity distribution.
        last_int_log = self._last_intensity_summary_ts.get(validator_key, 0)
        if (self.intensity_shadow_mode or self.intensity_enabled) and sim_ts - last_int_log >= self.intensity_log_interval_ns:
            self._last_intensity_summary_ts[validator_key] = sim_ts
            _cmap = self._book_intensity_last.get(validator_key, {})
            _idat = self._intensity_interval_data.get(validator_key, {})
            _agg_n = sum(1 for c in _cmap.values() if c == "aggressive")
            _norm_n = sum(1 for c in _cmap.values() if c == "normal")
            _light_n = sum(1 for c in _cmap.values() if c == "light")
            _floor_n = sum(1 for c in _cmap.values() if c == "floor_only")
            _hn_n = len(_idat.get("hard_neg", set()))
            _below_n = _cur_below_obs  # current-tick count (not interval-cumulative set)
            _fprobe_n = len(_idat.get("floor_probe", set()))
            _low_opp_ticks = _idat.get("low_opp_ticks", 0)
            _low_opp_now = (_agg_n / max(1, len(_cmap))) < self.low_opportunity_aggressive_frac
            _nr_n = max(1, _idat.get("net_realized_n", 1))
            _nr_avg = _idat.get("net_realized_sum", 0.0) / _nr_n
            bt.logging.info(
                f"[INTENSITY_SUMMARY] v={validator_key[:12]}"
                f" mode={'shadow' if self.intensity_shadow_mode else 'enabled'}"
                f" aggressive={_agg_n}"
                f" normal={_norm_n}"
                f" light={_light_n}"
                f" floor_only={_floor_n}"
                f" hard_negative={_hn_n}"
                f" books_below_min_obs={_below_n}"
                f" needs_floor_probe={_fprobe_n}"
                f" drift_quarantine={len(self._drift_quarantine.get(validator_key, {}))}"
                f" LOW_OPPORTUNITY_MODE={_low_opp_now}"
                f" low_opp_ticks={_low_opp_ticks}"
                f" avg_net_realized={_nr_avg:.6f}"
            )
            # Reset interval data (keep _class_map — it's per-book current state)
            self._intensity_interval_data[validator_key] = {
                "below_obs": set(),
                "floor_probe": set(),
                "hard_neg": set(),
                "low_opp_ticks": 0,
                "net_realized_sum": 0.0,
                "net_realized_n": 0,
            }

            # --- Auto intensity state machine ---
            _total_books = max(1, len(_cmap))
            _agg_frac = _agg_n / _total_books

            if self.intensity_shadow_mode and not self.intensity_enabled:
                # shadow → enabled: 보수적 승격 (모든 validator 3회 연속 조건 충족)
                _qualify = (
                    _nr_avg > 0.3
                    and _agg_frac >= 0.70
                    and _cur_below_obs <= 5
                    and _hn_n == 0
                    and not _low_opp_now
                )
                if _qualify:
                    self._intensity_qualify_counts[validator_key] = (
                        self._intensity_qualify_counts.get(validator_key, 0) + 1
                    )
                else:
                    self._intensity_qualify_counts[validator_key] = 0

                _known_validators = set(self._last_intensity_summary_ts.keys())
                if (
                    len(_known_validators) >= 2
                    and all(
                        self._intensity_qualify_counts.get(v, 0) >= 3
                        for v in _known_validators
                    )
                ):
                    self.intensity_shadow_mode = False
                    self.intensity_enabled = True
                    self._intensity_qualify_counts.clear()
                    bt.logging.info(
                        f"[AUTO_INTENSITY] shadow → enabled"
                        f" agg={_agg_frac:.2f} nr_avg={_nr_avg:.3f}"
                        f" below_obs={_cur_below_obs}"
                    )

            elif self.intensity_enabled and not self.intensity_shadow_mode:
                # enabled → shadow: 즉각 강등 (손실 방어 우선)
                _demote = (
                    _nr_avg < -0.2
                    or _agg_frac < 0.30
                    or _hn_n >= 10
                    or _low_opp_now
                )
                if _demote:
                    self.intensity_enabled = False
                    self.intensity_shadow_mode = True
                    self._intensity_qualify_counts.clear()
                    bt.logging.warning(
                        f"[AUTO_INTENSITY] enabled → shadow"
                        f" agg={_agg_frac:.2f} nr_avg={_nr_avg:.3f} hn={_hn_n}"
                        f" low_opp={_low_opp_now}"
                    )

        # INTENSITY_RISK_SUMMARY: periodic log of auto risk gate state (fires with same cadence).
        if self.intensity_enabled and self.risk_gate_enabled and _risk_gate is not None:
            last_rs = self._last_risk_summary_ts.get(validator_key, 0)
            if sim_ts - last_rs >= self.risk_gate_log_interval_ns:
                self._last_risk_summary_ts[validator_key] = sim_ts
                bt.logging.info(
                    f"[INTENSITY_RISK_SUMMARY] v={validator_key[:12]}"
                    f" ban_agg={_risk_gate['ban_aggressive']}"
                    f" demote_light={_risk_gate['demote_to_light']}"
                    f" slow_probe={_risk_gate['slow_floor_probe']}"
                    f" full_low_opp={_risk_gate['full_low_opp_mode']}"
                    f" eff_agg_mult={_risk_gate['aggressive_qty_mult_cap']:.2f}"
                    f" drain={global_drain_mode}"
                    f" old_inv={len(old_unwind_candidates)}"
                    f" prev_fu={self._risk_gate_prev_forced_unwind.get(validator_key, 0)}"
                    f" below_obs={_cur_below_obs}"
                    f" active=[{','.join(_risk_gate['active_rules'])}]"
                )

        # DEFICIT_RECOVERY_SUMMARY: periodic snapshot of obs-accumulation progress.
        last_dr_log = self._last_deficit_recovery_ts.get(validator_key, 0)
        if sim_ts - last_dr_log >= self.deficit_recovery_log_interval_ns:
            self._last_deficit_recovery_ts[validator_key] = sim_ts
            _dr_buf = self._deficit_recovery_buf.get(validator_key, {})
            _dr_obs_all = self._realized_obs.get(validator_key, {})
            _dr_eff0 = _dr_eff1 = _dr_eff2 = _dr_eff3p = 0
            _dr_below = 0
            _dr_gap_sum = 0.0
            _dr_gap_n = 0
            _dr_co_sum = 0
            _dr_grace = self.changeover_grace_period_ns
            for _dr_bid in traded_books:
                _dr_obs_st = _dr_obs_all.get(_dr_bid)
                if _dr_obs_st is None:
                    _dr_eff0 += 1
                    _dr_below += 1
                    continue
                _dr_obs_st.prune(sim_ts, self.kappa_lookback_ns)
                _dr_eff = _dr_obs_st.effective_non_zero_count(_dr_grace)
                if _dr_eff == 0:
                    _dr_eff0 += 1
                elif _dr_eff == 1:
                    _dr_eff1 += 1
                elif _dr_eff == 2:
                    _dr_eff2 += 1
                else:
                    _dr_eff3p += 1
                if _dr_eff < self.min_realized_observations:
                    _dr_below += 1
                    _dr_co, _ = _dr_obs_st._changeover_count(_dr_grace)
                    _dr_co_sum += _dr_co
                    _sorted_ts = sorted(_dr_obs_st.obs_by_ts)
                    if len(_sorted_ts) >= 2:
                        for _gi in range(1, len(_sorted_ts)):
                            _dr_gap_sum += (_sorted_ts[_gi] - _sorted_ts[_gi - 1]) / 1e9
                            _dr_gap_n += 1
            _dr_avg_gap = _dr_gap_sum / _dr_gap_n if _dr_gap_n > 0 else 0.0
            bt.logging.info(
                f"[DEFICIT_RECOVERY_SUMMARY] v={validator_key[:12]}"
                f" books_below_min_obs={_dr_below}"
                f" eff_obs_0={_dr_eff0}"
                f" eff_obs_1={_dr_eff1}"
                f" eff_obs_2={_dr_eff2}"
                f" eff_obs_3plus={_dr_eff3p}"
                f" floor_probes_sent={_dr_buf.get('floor_probes_sent', 0)}"
                f" avg_probe_gap_s={_dr_avg_gap:.1f}"
                f" changeover_count={_dr_co_sum}"
                f" forced_unwind_count={_dr_buf.get('forced_unwind_count', 0)}"
            )
            self._deficit_recovery_buf[validator_key] = {
                "floor_probes_sent": 0, "forced_unwind_count": 0,
            }

        # Instruction pulse: books with no instructions this tick (last resort).
        if self.instruction_pulse_enabled:
            touched: set[int] = set()
            for instr in response.instructions:
                book_id = getattr(instr, "bookId", None)
                if book_id is not None:
                    touched.add(book_id)

            pulse_candidates = [
                bid for bid in traded_sorted
                if bid in selection.core
                and bid not in touched
                and sim_ts - self._book_st(validator_key, bid).last_instruction_ts
                >= self.touch_max_gap_ns
            ]

            def pulse_budget(book_id: int) -> bool:
                return instr_used.get(book_id, 0) < self.max_instructions_per_book

            def pulse_spend(book_id: int) -> None:
                instr_used[book_id] = instr_used.get(book_id, 0) + 1

            for book_id in pulse_candidates:
                if book_id not in state.books or book_id not in self.accounts:
                    continue
                book = state.books[book_id]
                if not book.bids or not book.asks:
                    continue
                account = self.accounts[book_id]
                st = self._book_st(validator_key, book_id)
                if _book_vol_source == "per_book":
                    vt = self._book_volume_traded(account, validator_key, book_id)
                else:
                    vt = (
                        self._maker_volume.get(validator_key, {}).get(book_id, 0.0)
                        + self._taker_volume.get(validator_key, {}).get(book_id, 0.0)
                    )
                pulse_base = self.order_qty
                pulse_min = _round_qty(pulse_base * 0.25, cfg.volumeDecimals)
                pulse_scale = self._throttle_qty_scale(vt, throttle_cap, hard_stop_cap)
                pulse_order = _round_qty(max(pulse_min, pulse_base * pulse_scale), cfg.volumeDecimals)
                self._instruction_pulse(
                    response, book_id, book, account, st, cfg, tick, sim_ts,
                    net_inventory=self._net_inventory(validator_key, book_id, account, cfg),
                    at_hard_stop=vt >= hard_stop_cap,
                    throttled=vt >= throttle_cap,
                    order_qty=pulse_order,
                    min_qty=pulse_min,
                    budget_left=lambda: pulse_budget(book_id),
                    spend=lambda: pulse_spend(book_id),
                )

        return response

if __name__ == "__main__":
    launch(ISFMMAgent)
