# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

"""Volume-Bucket Momentum/Mean-Reversion Agent.

Categorizes market activity into high/low volume buckets across buy/sell sides,
predicts directional signals from volume imbalances, then executes either:

  - Momentum trades  (imbalance and recent price move agree)
  - Mean-reversion trades (imbalance and recent price move disagree)

Signal strength is computed as log_return × mean_imbalance:

  - ``> 0``  →  MOMENTUM  (agree: ride the move)
  - ``< 0``  →  REVERSION (disagree: fade the move)
  - low confidence  →  NOISE / HOLD (do nothing)
"""

from copy import deepcopy
import os
from pathlib import Path
import pickle
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from enum import IntEnum
from datetime import datetime
from typing import Any, Optional

import numpy as np
import bittensor as bt
import optuna
from optuna import Study, Trial
import pandas as pd
import requests

from taos.common.agents import launch
from taos.im.agents import GenTRXAgent
from taos.im.agents import (
    Positions,
    Signals,
    TimestampedPrice,
)
from taos.im.protocol.events import SimulationEndEvent, TradeEvent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse


# ---------------------------------------------------------------------------
# Discord notification helper
# ---------------------------------------------------------------------------

@dataclass
class DiscordNotifier:
    """Batched Discord webhook notifications for trade activity.

    Trades are accumulated within each ``respond()`` cycle and flushed as a
    single message (or multiple messages when the 2000-char limit is hit).
    Emergency trades are never reported.

    Attributes:
        webhook_url: Discord webhook URL.
        enabled: Master on/off switch.
        rate_limit_seconds: Minimum wall-clock seconds between outgoing messages.
        last_notification_time: Unix timestamp of the last successful send.
        target_validator: When set, only trades for this validator hotkey are
            reported.
        pending_trades: One-liner strings accumulated since the last flush.
    """

    webhook_url: Optional[str] = None
    enabled: bool = False
    rate_limit_seconds: int = 1
    last_notification_time: float = 0.0
    target_validator: Optional[str] = None
    pending_trades: list = field(default_factory=list)

    def should_send(self) -> bool:
        """Check whether the rate-limit window has elapsed.

        Returns:
            True if notifications are enabled and enough time has passed since
            the last send; False otherwise.
        """
        if not self.enabled or not self.webhook_url:
            return False
        return (time.time() - self.last_notification_time) >= self.rate_limit_seconds

    def send_notification(
        self,
        content: str,
        username: str = "algorithm_0",
        bypass_rate_limit: bool = False,
    ) -> bool:
        """POST a plain-text message to the webhook.

        Args:
            content: Message body (plain text or markdown).
            username: Display name shown in Discord.
            bypass_rate_limit: When ``True``, skip the rate-limit check.  Used
                when sending multiple batches back-to-back.

        Returns:
            True on HTTP 204 (success), False otherwise.
        """
        if not bypass_rate_limit and not self.should_send():
            return False
        if not self.enabled or not self.webhook_url:
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json={"username": username, "content": content},
                timeout=5,
            )
            if response.status_code == 204:
                self.last_notification_time = time.time()
                return True
            bt.logging.warning(f"Discord webhook returned status {response.status_code}")
            return False
        except Exception as e:
            bt.logging.error(f"Failed to send Discord notification: {e}")
            return False

    def add_trade(
        self,
        action: str,
        book_id: int,
        total_books: int,
        direction: OrderDirection,
        quantity: float,
        price: float,
    ) -> None:
        """Append one formatted trade line to the pending batch.

        Args:
            action: ``"BUY"`` or ``"SELL"``.
            book_id: Zero-based book identifier.
            total_books: Total number of books in the simulation (for display).
            direction: ``OrderDirection`` enum value.
            quantity: Order size in base units.
            price: Execution price (typically the current mid-quote).
        """
        self.pending_trades.append(
            self.format_trade_oneliner(action, book_id, total_books, direction, quantity, price)
        )

    def flush_trades(self) -> bool:
        """Send all pending trades, splitting into multiple messages if needed.

        Discord enforces a 2000-character limit per message.  Lines are packed
        greedily and excess lines spill into a new batch.  Batches are delivered
        sequentially with a short inter-batch pause.

        Returns:
            True if every batch was delivered successfully, False if any failed.
        """
        if not self.pending_trades:
            return True

        MAX_CHARS = 2000
        batches: list[str] = []
        current_lines: list[str] = []
        current_len = 0

        for line in self.pending_trades:
            line_len = len(line) + 1  # +1 for the joining newline
            if current_len + line_len > MAX_CHARS and current_lines:
                batches.append("\n".join(current_lines))
                current_lines = [line]
                current_len = line_len
            else:
                current_lines.append(line)
                current_len += line_len

        if current_lines:
            batches.append("\n".join(current_lines))

        self.pending_trades.clear()

        all_success = True
        for i, batch in enumerate(batches):
            success = self.send_notification(batch, bypass_rate_limit=True)
            if not success:
                all_success = False
                bt.logging.warning(f"Failed to send Discord batch {i + 1}/{len(batches)}")
            elif i < len(batches) - 1:
                time.sleep(0.5)  # Brief pause to respect Discord's own rate limits

        return all_success

    def format_trade_oneliner(
        self,
        action: str,
        book_id: int,
        total_books: int,
        direction: OrderDirection,
        quantity: float,
        price: float,
    ) -> str:
        """Format a single trade as a fixed-width one-liner.

        Example::

            🟢 BUY  [2025-02-12 14:23:15] SIM #5/64    10.50@285.45
            🔴 SELL [2025-02-12 14:23:18] SIM #41/64    8.25@142.78

        Args:
            action: ``"BUY"`` or ``"SELL"``.
            book_id: Zero-based book identifier.
            total_books: Total books in the simulation.
            direction: ``OrderDirection`` enum value (unused directly; ``action``
                string drives the emoji).
            quantity: Order size in base units.
            price: Execution price.

        Returns:
            A single formatted string suitable for Discord plain-text output.
        """
        emoji = "🟢" if "BUY" in action else "🔴"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        book_display = f"SIM #{book_id + 1}/{total_books}"
        return f"{emoji} {action:<12} [{timestamp}] {book_display:<10} {quantity:.2f}@{price:.2f}"


# ---------------------------------------------------------------------------
# Volume-bucket data structures
# ---------------------------------------------------------------------------

class VolumeCategory(IntEnum):
    """Classification of a single trade's size relative to a threshold.

    Attributes:
        HIGH: Trade size exceeds the category threshold.
        LOW: Trade size is at or below the category threshold.
    """

    HIGH = 0
    LOW = 1


@dataclass
class VolumeBucket:
    """Accumulated volume split by HIGH/LOW category for one trade direction.

    Attributes:
        high: Total volume of HIGH-category trades.
        low: Total volume of LOW-category trades.
        count: Total number of trades accumulated (both categories).
    """

    high: float = 0.0
    low: float = 0.0
    count: int = 0

    def update(self, volume: float, category: VolumeCategory) -> None:
        """Add one trade's volume to the appropriate category bucket.

        Args:
            volume: Trade size to accumulate.
            category: Whether the trade is HIGH or LOW category.
        """
        if category == VolumeCategory.HIGH:
            self.high += volume
        else:
            self.low += volume
        self.count += 1

    def get_volume(self, category: VolumeCategory) -> float:
        """Return the accumulated volume for a given category.

        Args:
            category: The volume category to query.

        Returns:
            Accumulated volume for ``category``, or ``0.0`` for an unrecognised
            value.
        """
        if category == VolumeCategory.HIGH:
            return self.high
        if category == VolumeCategory.LOW:
            return self.low
        return 0.0


@dataclass
class PredictorEntry:
    """One time-bucket of aggregated buy/sell volume data.

    Attributes:
        buy: Volume bucket for buy-side trades.
        sell: Volume bucket for sell-side trades.
        timestamp: Simulation timestamp (ns) when this bucket was opened.
    """

    buy: VolumeBucket = field(default_factory=VolumeBucket)
    sell: VolumeBucket = field(default_factory=VolumeBucket)
    timestamp: int = 0

    def update(self, volume: float, direction: OrderDirection, category: VolumeCategory) -> None:
        """Route a trade into the correct directional bucket.

        Args:
            volume: Trade size.
            direction: Whether the trade was a buy or a sell.
            category: HIGH or LOW volume category.
        """
        if direction == OrderDirection.BUY:
            self.buy.update(volume, category)
        elif direction == OrderDirection.SELL:
            self.sell.update(volume, category)

    def imbalance(self, category: VolumeCategory) -> float:
        """Compute the normalised buy-sell volume imbalance for a category.

        Args:
            category: The volume category to compute imbalance for.

        Returns:
            Imbalance in ``[-1, +1]`` where ``+1`` is pure buy flow and ``-1``
            is pure sell flow.  Returns ``0.0`` when no volume has been recorded.
        """
        buy = self.buy.get_volume(category)
        sell = self.sell.get_volume(category)
        total = buy + sell
        return (buy - sell) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Trade tracking
# ---------------------------------------------------------------------------

@dataclass
class MyTrade:
    """One of the agent's own fills with running P&L and volume accumulators.

    Attributes:
        price: Fill price.
        qty: Fill quantity in base units.
        side: ``1`` if the agent bought base currency, ``0`` if it sold.
        fee: Fee paid for this fill (always positive).
        rolling_quote: Cumulative signed quote P&L over the lookback window.
            Positive values indicate the agent has collected more quote than it
            has spent (net profit in quote currency).
        rolling_volume: Cumulative traded notional over the lookback window.
        timestamp: Simulation timestamp (ns) of this fill.
    """

    price: float
    qty: float
    side: int
    fee: float
    rolling_quote: float
    rolling_volume: float
    timestamp: int

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Last Trade: {pd.to_timedelta(self.timestamp)}\n"
            f"Rolling Quote/PnL estimate: ¤ {self.rolling_quote:.4f}\n"
            f"Rolling Volume: ¤ {self.rolling_volume:.2f}"
        )


# ---------------------------------------------------------------------------
# Per-book state
# ---------------------------------------------------------------------------

@dataclass
class AgentDataStorage:
    """All mutable state for one (validator, book) pair.

    Attributes:
        predictors: Rolling window of volume-bucket snapshots used for signal
            generation.
        midquotes: Recent mid-price history used for log-return calculation.
        returns: Own-trade history used for P&L tracking and the Optuna
            objective.
        positions: Current open position tracker.
        last_signal: The most recent signal emitted for this book.
        last_processed_timestamp: Highest event timestamp already ingested;
            used to avoid double-processing events.
        last_emergency_timestamp: Simulation timestamp of the last emergency
            trade; used to rate-limit emergency actions.
        startup_timestamp: Simulation timestamp when this buffer was created;
            used to enforce the ``validator_lookback`` warm-up period.
        traded_volume: Latest ``traded_volume`` snapshot from the exchange
            account, refreshed each tick.
    """

    predictors: list[PredictorEntry]
    midquotes: list[TimestampedPrice]
    returns: list[MyTrade]
    positions: Positions = field(default_factory=Positions)
    last_signal: Signals = Signals.NOISE
    last_processed_timestamp: int = 0
    last_emergency_timestamp: int = 0
    startup_timestamp: int = 0
    traded_volume: float = 0.0

    def append_predictor(self, entry: PredictorEntry) -> None:
        """Append a new predictor bucket to the rolling window.

        Args:
            entry: The ``PredictorEntry`` to append.
        """
        self.predictors.append(entry)

    def update_last_predictor(
        self, volume: float, direction: OrderDirection, category: VolumeCategory
    ) -> None:
        """Forward a trade into the most recently opened predictor bucket.

        Args:
            volume: Trade size.
            direction: Buy or sell direction.
            category: HIGH or LOW volume category.
        """
        self.predictors[-1].update(volume, direction, category)

    def new_entry(
        self,
        volume: float,
        direction: OrderDirection,
        category_threshold: float,
        timestamp: int,
        sampling_interval: int,
        samples: int,
    ) -> None:
        """Ingest one market trade into the rolling predictor window.

        Opens a new time-bucket when ``sampling_interval`` nanoseconds have
        elapsed since the last bucket, then prunes the oldest bucket if the
        window exceeds ``samples``.

        Args:
            volume: Trade size in base units.
            direction: Whether the aggressor was a buyer or seller.
            category_threshold: Volume level that separates HIGH from LOW
                trades.
            timestamp: Simulation timestamp (ns) of the trade.
            sampling_interval: Bucket width in nanoseconds.
            samples: Maximum number of buckets to retain.
        """
        category = VolumeCategory.HIGH if volume > category_threshold else VolumeCategory.LOW

        if not self.predictors or (timestamp - self.predictors[-1].timestamp) > sampling_interval:
            self.append_predictor(PredictorEntry(timestamp=timestamp))

        self.update_last_predictor(volume, direction, category)

        while len(self.predictors) > samples:
            self.predictors.pop(0)


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Output of the signal-generation step.

    Attributes:
        name: Human-readable label describing the strategy variant that
            produced this prediction.
        direction: Suggested trade direction (BUY or SELL).
        confidence: Magnitude of the mean volume imbalance, in ``[0, 1]``.
            Higher values indicate stronger conviction.
        strength: Product of log-return and mean imbalance.  Positive values
            indicate momentum; negative values indicate mean-reversion.
        timestamp: Simulation timestamp (ns) at which the prediction was made.
    """

    name: str
    direction: OrderDirection
    confidence: float
    strength: float
    timestamp: int

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"{self.name} = Strength({self.strength:.4f}) "
            f"Confidence({self.confidence:.4f}) "
            f"Signal({self.direction.name})"
        )


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

@dataclass
class TradingParameters:
    """All tunable knobs for the trading strategy.

    Time-based fields use nanoseconds (simulation time units) unless noted.

    Attributes:
        quantity: Base order size in base units.
        expiry_period: Order expiry duration (ns); reserved for future use.
        rolling_window_min: Minimum rolling-window size; reserved.
        rolling_window_max: Maximum rolling-window size; reserved.
        samples: Number of predictor buckets to retain in the rolling window.
        sampling_interval: Bucket width and minimum signal cadence (ns).
        model_threshold: Minimum ``|imbalance|`` required to generate a signal.
        signal_threshold: Centre of the momentum/reversion dead-band.
        signal_tolerance: Half-width of the dead-band around
            ``signal_threshold``.
        category_threshold: Volume level separating HIGH from LOW trades.
        strategy_category: Which volume category drives the imbalance signal.
        reverse_strategy: When ``True``, flip the imbalance-to-direction
            mapping (buy when sell flow dominates, and vice-versa).
        skip_extend: When ``True``, never add to an open position; wait for
            an exit signal instead.
        validator_lookback: Rolling window length for P&L pruning (ns).
        validator_vol_cap: Maximum annualised notional volume expressed as a
            multiple of ``miner_wealth``.
        advanced: When ``True``, use strength-based (momentum/reversion)
            signals; when ``False``, use simple directional signals.
    """

    quantity: float = 2.0
    expiry_period: int = int(120e9)
    rolling_window_min: int = 10
    rolling_window_max: int = 20
    samples: int = 1
    sampling_interval: int = int(120e9)
    model_threshold: float = 0.0
    signal_threshold: float = 0.0
    signal_tolerance: float = 0.0
    category_threshold: float = 1.5
    strategy_category: VolumeCategory = VolumeCategory.LOW
    reverse_strategy: bool = True
    skip_extend: bool = True
    validator_lookback: int = int(3 * 60 * 60e9)
    validator_vol_cap: int = int(10)
    advanced: bool = True


# ---------------------------------------------------------------------------
# Optuna ask-tell wrapper
# ---------------------------------------------------------------------------

@dataclass
class StudyParams:
    """Container for one validator's Optuna study and bookkeeping metadata.

    Attributes:
        study_file: Path to the persisted study pickle file.
        study_init_time: Wall-clock Unix time when the study was created.
        study: The live Optuna ``Study`` object.
        trial_time: Simulation timestamp (ns) of the last optimisation step.
        trial_cache: Map of trial number to ``FrozenTrial`` for the currently
            active (not yet scored) trial.
        opt_interval: Minimum simulation-time gap between optimisation steps
            (ns).
    """

    study_file: Path
    study_init_time: float
    study: Study
    trial_time: int
    trial_cache: dict[int, optuna.trial.FrozenTrial]
    opt_interval: int


@dataclass
class HyperParams:
    """Subset of ``TradingParameters`` explored by Optuna.

    Attributes:
        sampling_interval: Bucket width and signal cadence (ns).
        model_threshold: Minimum ``|imbalance|`` required to act.
        strategy_category: Which volume category drives the imbalance signal.
        reverse_strategy: Whether to invert the imbalance-to-direction mapping.
    """

    sampling_interval: int = int(120e9)
    model_threshold: float = 0.0
    strategy_category: VolumeCategory = VolumeCategory.LOW
    reverse_strategy: bool = True

    def from_dict(self, kv: dict) -> "HyperParams":
        """Populate fields from a dictionary, casting values to existing types.

        Only keys that match an existing field name are applied; unknown keys
        are silently ignored.

        Args:
            kv: Mapping of field name to value.

        Returns:
            ``self``, updated in-place (for chaining).
        """
        for k, v in kv.items():
            if hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v))
        return self


def _create_or_load_study(study_file: Path) -> Study:
    """Load an existing Optuna study from disk, or create a new one.

    Args:
        study_file: Path to a pickle file produced by ``_persist_study``.

    Returns:
        A ``Study`` configured to maximise its objective, using TPE sampling
        and median pruning.
    """
    if study_file.exists():
        with open(study_file, "rb") as f:
            return pickle.load(f)
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )


def _persist_study(study: Study, study_file: Path) -> None:
    """Atomically write a study to disk via a temp-file rename.

    Writing to a temporary file and renaming is atomic on POSIX systems,
    preventing partial writes from corrupting the saved study.

    Args:
        study: The Optuna ``Study`` to persist.
        study_file: Destination path.
    """
    with tempfile.NamedTemporaryFile(dir=study_file.parent, delete=False) as tmp:
        pickle.dump(study, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.replace(tmp.name, study_file)


def ask_trial(study: Study) -> tuple[Trial, HyperParams]:
    """Sample the next hyperparameter trial from the study.

    Args:
        study: An active Optuna ``Study``.

    Returns:
        A tuple of ``(trial, hp)`` where ``trial`` is the Optuna ``Trial``
        object (needed later for ``tell_study``) and ``hp`` contains the
        sampled hyperparameter values.
    """
    trial = study.ask()
    hp = HyperParams(
        sampling_interval=trial.suggest_int("sampling_interval", int(10e9), int(400e9), step=int(1e9)),
        model_threshold=trial.suggest_float("model_threshold", 0.0, 0.99, step=0.01),
        strategy_category=VolumeCategory(trial.suggest_int("strategy_category", 0, 1)),
        reverse_strategy=bool(trial.suggest_int("reverse_strategy", 0, 1)),
    )
    return trial, hp


def seed_study(study: Study, seeds: HyperParams) -> tuple[Trial, HyperParams]:
    """Enqueue one trial using known-good seed values, then ask for it.

    Enqueuing forces Optuna to evaluate the seed params before exploring new
    regions of the search space.

    Args:
        study: An active Optuna ``Study``.
        seeds: ``HyperParams`` instance whose field values are used as the seed.

    Returns:
        A tuple of ``(trial, seeds)`` where ``trial`` is the trial object for
        the seeded params and ``seeds`` is returned unchanged.
    """
    for key, seed in seeds.__dict__.items():
        value = int(seed) if seed % 1 == 0 else seed
        study.enqueue_trial({key: value})
    return study.ask(), seeds


def tell_study(study: Study, gain: float, trial: Trial, study_file: Path) -> Study:
    """Report a trial result to the study and persist it to disk.

    Args:
        study: The Optuna ``Study`` that owns the trial.
        gain: Objective value achieved by the trial (higher is better).
        trial: The trial object returned by a previous ``ask_trial`` call.
        study_file: Path used to persist the updated study.

    Returns:
        The updated ``Study`` object.
    """
    study.tell(trial, gain)
    _persist_study(study, study_file)
    return study


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RevengAgent(GenTRXAgent):
    """Volume-bucket momentum / mean-reversion agent.

    Combines order-flow imbalance signals with recent price returns to classify
    each book into a momentum or mean-reversion regime.  Hyperparameters are
    continuously self-optimised via an Optuna ask-tell loop.  Optional Discord
    webhook notifications report non-emergency trades in real time.
    """

    def __init__(self, uid, config, log_dir=None):
        if not hasattr(config, "lazy_load"):
            config.lazy_load = False
        super().__init__(uid, config, log_dir)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up all agent state.

        Called once by the framework before the first tick.  Reads optional
        config overrides for all ``TradingParameters`` fields and initialises
        per-validator dicts, the agent-ID filter, and the Discord notifier.
        """
        # GenTRX is opt-in: only activates when explicitly configured.
        if not hasattr(self.config, 'gtx_training_enabled'):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, 'gtx_collect_data'):
            self.config.gtx_collect_data = False
        super().initialize()
        bt.logging.set_info()

        # Build default trading params, then apply any config-level overrides
        self.init_tparams = TradingParameters()
        for k, v in self.init_tparams.__dict__.items():
            if hasattr(self.config, k):
                setattr(self.init_tparams, k, type(v)(getattr(self.config, k)))

        self.tparams: dict[str, TradingParameters] = {}   # per-validator params
        self.studies: dict[str, StudyParams] = {}          # per-validator Optuna studies
        self.opt_interval: int = getattr(self.config, "opt_interval", int(10 * 60e9))

        # buffers[validator_hotkey][book_id] -> AgentDataStorage
        self.buffers: dict[str, dict[int, AgentDataStorage]] = {}

        # Only observe trades from other agents; exclude our own fills from signals
        agent_range = getattr(self.config, "agent_range", 256)
        self.filter_range = [i for i in range(agent_range) if i != self.uid]

        # Discord notifier setup
        self.discord = DiscordNotifier(
            webhook_url=getattr(self.config, "discord_webhook_url", None),
            enabled=bool(getattr(self.config, "discord_enabled", False)),
            rate_limit_seconds=getattr(self.config, "discord_rate_limit", 1),
            target_validator=getattr(self.config, "discord_target_validator", None),
        )
        if self.discord.enabled:
            if self.discord.webhook_url:
                bt.logging.info(f"Discord enabled (rate limit: {self.discord.rate_limit_seconds}s)")
                if self.discord.target_validator:
                    bt.logging.info(f"Discord filter: {self.discord.target_validator[:16]}...")
                self.discord.send_notification("🚀 **algorithm_0 Started**")
            else:
                bt.logging.warning("Discord enabled but no webhook_url provided — disabling.")
                self.discord.enabled = False

    # ------------------------------------------------------------------
    # Buffer / study lifecycle
    # ------------------------------------------------------------------

    def buffers_init(self, state: MarketSimulationStateUpdate, book_id: int) -> None:
        """Ensure buffers and a study exist for a (validator, book) pair.

        Creates missing entries on first encounter.  If a timestamp regression
        is detected (indicating a new simulation on the same connection), the
        book buffer and study are both reset cleanly.

        Args:
            state: The current market state update from the validator.
            book_id: Identifier of the order book being initialised.
        """
        hotkey = state.dendrite.hotkey

        if hotkey not in self.buffers:
            self.buffers[hotkey] = {}
        if hotkey not in self.studies:
            self.reset_study(hotkey, state.config.simulation_id)

        if book_id not in self.buffers[hotkey]:
            self.buffers[hotkey][book_id] = self._empty_buffer(state.timestamp)
        elif (
            self.buffers[hotkey][book_id].predictors
            and state.timestamp < self.buffers[hotkey][book_id].predictors[-1].timestamp
        ):
            bt.logging.info("Timestamp regression detected — resetting book buffer and study.")
            self.reset_study(hotkey, state.config.simulation_id)
            self.buffers[hotkey][book_id] = self._empty_buffer(state.timestamp)

    def _empty_buffer(self, startup_timestamp: int) -> AgentDataStorage:
        """Construct a freshly initialised ``AgentDataStorage``.

        Args:
            startup_timestamp: Simulation timestamp (ns) to record as the
                buffer creation time.

        Returns:
            A new ``AgentDataStorage`` with empty lists and the simulation's
            initial price as the sole midquote seed.
        """
        return AgentDataStorage(
            predictors=[],
            midquotes=[TimestampedPrice(timestamp=0, price=self.simulation_config.init_price)],
            returns=[],
            startup_timestamp=startup_timestamp,
        )

    def reset_study(self, validator: str, sim_id: str) -> None:
        """Create a new Optuna study for a validator, seeded from previous best params.

        If a study already exists its best params are carried forward as the
        seed for the first trial of the new study.  Any pending (unscored) trial
        is evaluated before the old study is discarded.

        Args:
            validator: Validator hotkey string.
            sim_id: Simulation identifier used as part of the study file path,
                allowing studies to be grouped per simulation run.
        """
        filename = f'study_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
        study_file = Path(self.data_dir) / str(self.uid) / validator / sim_id / filename
        study_file.parent.mkdir(exist_ok=True, parents=True)

        if validator not in self.studies:
            best_params = self.init_trading_params(validator)
        else:
            sp = self.studies[validator]
            if sp.trial_cache:
                # Score the pending trial before discarding the study
                self.evaluate_and_update_hp(validator)
            try:
                best_params = dict(sp.study.best_params)
            except ValueError:
                # No completed trials yet — fall back to current params
                bt.logging.info(
                    "No completed trials to seed from — using current params.",
                    prefix="[STUDY]",
                )
                best_params = self.init_trading_params(validator)

        study = _create_or_load_study(study_file)
        _persist_study(study, study_file)

        self.studies[validator] = StudyParams(
            study_file=study_file,
            study_init_time=time.time(),
            study=study,
            trial_time=0,
            trial_cache={},
            opt_interval=self.opt_interval,
        )

        trial, hp = seed_study(study, HyperParams().from_dict(best_params))
        self.change_trading_params(validator, hp)
        self.studies[validator].trial_cache[trial.number] = trial

    def init_trading_params(self, validator: str) -> dict[str, Any]:
        """Ensure trading params exist for a validator and return a dict copy.

        Creates a deep copy of the global defaults on first call for a given
        validator.  Always returns a snapshot copy rather than a live reference
        to prevent accidental mutation.

        Args:
            validator: Validator hotkey string.

        Returns:
            A plain ``dict`` copy of the validator's current
            ``TradingParameters`` fields.
        """
        if validator not in self.tparams:
            bt.logging.info(
                f"Initialising trading parameters for {self.validator_to_str(validator)}.",
                prefix="[Trading parameters]",
            )
            self.tparams[validator] = deepcopy(self.init_tparams)
        return dict(self.tparams[validator].__dict__)

    def change_trading_params(self, validator: str, hp: HyperParams) -> None:
        """Apply ``HyperParams`` fields onto a validator's ``TradingParameters``.

        Only fields present in both ``hp`` and the target ``TradingParameters``
        are updated; unrecognised fields are silently ignored.  Values are cast
        to the existing field type to preserve dataclass type invariants.

        Args:
            validator: Validator hotkey string.
            hp: ``HyperParams`` instance containing the new values to apply.
        """
        if validator not in self.tparams:
            self.init_trading_params(validator)
        for k, v in hp.__dict__.items():
            if hasattr(self.tparams[validator], k):
                setattr(self.tparams[validator], k, type(getattr(self.tparams[validator], k))(v))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def onTrade(self, event: TradeEvent, validator: str) -> None:
        """Record the agent's own fills for rolling P&L tracking.

        Accumulates ``rolling_quote`` and ``rolling_volume`` as a running sum
        across fills, then prunes entries older than ``validator_lookback``
        nanoseconds.  The rolling quote is re-zeroed at the new window boundary
        after pruning so relative P&L comparisons remain valid.

        Self-trades (``takerAgentId == makerAgentId``) are silently ignored.

        Args:
            event: The trade event emitted by the exchange.
            validator: Hotkey of the validator that delivered this event.
        """

        def build_trade(event: TradeEvent, prev_volume: float) -> MyTrade:
            """Construct a ``MyTrade`` from a raw ``TradeEvent``.

            Args:
                event: Raw fill event.
                prev_volume: Cumulative volume before this fill; used to
                    initialise ``rolling_volume`` on the first fill.

            Returns:
                A ``MyTrade`` with signed ``rolling_quote`` and initialised
                ``rolling_volume``.
            """
            taker = event.takerAgentId == self.uid
            side = event.side if taker else int(abs(event.side - 1))
            fee = event.takerFee if taker else event.makerFee
            notional = event.price * event.quantity
            # Sells receive quote; buys spend quote; fees are always a cost
            quote_delta = notional - fee if side == 1 else -(notional + fee)
            return MyTrade(
                price=event.price,
                qty=event.quantity,
                side=side,
                fee=fee,
                rolling_quote=quote_delta,
                rolling_volume=prev_volume if prev_volume is not None else notional,
                timestamp=event.timestamp,
            )

        book_id = event.bookId
        if event.takerAgentId == event.makerAgentId:
            return  # Self-trade — ignore

        buf = self.buffers[validator][book_id]
        trade = build_trade(event, buf.traded_volume)
        buf.returns.append(trade)

        # Accumulate rolling totals from previous entry
        if len(buf.returns) > 1:
            prev = buf.returns[-2]
            trade.rolling_volume = prev.rolling_volume + trade.rolling_volume
            trade.rolling_quote = prev.rolling_quote + trade.rolling_quote

        # Prune entries outside the lookback window
        cutoff = event.timestamp - self.tparams[validator].validator_lookback
        pruned = [t for t in buf.returns if t.timestamp >= cutoff]

        if len(pruned) < len(buf.returns) and pruned:
            # Re-zero rolling_quote at the new window boundary
            offset = buf.returns[0].rolling_quote
            for t in pruned:
                t.rolling_quote -= offset

        buf.returns = pruned

    def onEnd(self, event: SimulationEndEvent) -> None:
        """Score pending trials, clear all buffers, and reset studies.

        Called by the framework when the simulation ends.  Ensures the final
        trial result is recorded in the study before state is cleared, so
        learning is preserved across simulation restarts.

        Args:
            event: The simulation-end event (content unused).
        """
        bt.logging.info("[SIMULATION END] Evaluating trials and clearing state.")
        for validator, buffer in self.buffers.items():
            self.evaluate_and_update_hp(validator)
            for book_id in buffer:
                buffer[book_id] = self._empty_buffer(startup_timestamp=0)
            self.reset_study(validator, "backup")
        import gc
        gc.collect()

    # ------------------------------------------------------------------
    # Per-tick updates
    # ------------------------------------------------------------------

    def update_predictors(self, validator: str, book: Book, timestamp: int) -> None:
        """Ingest new market trades from the book event log into the predictor window.

        Events are filtered to the current sampling interval and skipped if
        they have already been processed or originate from this agent.  The
        processed-timestamp watermark is advanced to ``timestamp`` after the
        loop regardless of how many events were consumed.

        Args:
            validator: Validator hotkey string.
            book: The order book whose event log is being processed.
            timestamp: Current simulation timestamp (ns); used as the new
                watermark and to derive the lookback cutoff.
        """
        buffer = self.buffers[validator][book.id]
        cutoff = timestamp - self.tparams[validator].sampling_interval

        events_processed = 0
        skipped = {"not_trade": 0, "too_old": 0, "already_processed": 0, "not_other_agent": 0}

        for event in book.events:
            if event.type != "t":
                skipped["not_trade"] += 1
                continue
            if event.timestamp < cutoff:
                skipped["too_old"] += 1
                continue
            if event.timestamp <= buffer.last_processed_timestamp:
                skipped["already_processed"] += 1
                continue
            if event.taker_agent_id not in self.filter_range:
                skipped["not_other_agent"] += 1
                continue

            buffer.new_entry(
                event.quantity,
                event.side,
                self.tparams[validator].category_threshold,
                event.timestamp,
                self.tparams[validator].sampling_interval,
                self.tparams[validator].samples,
            )
            events_processed += 1
            buffer.last_processed_timestamp = max(buffer.last_processed_timestamp, event.timestamp)

        # Advance the watermark to the current state timestamp
        buffer.last_processed_timestamp = timestamp

        if events_processed == 0 and not buffer.predictors:
            bt.logging.debug(
                f"BOOK {book.id} | No events ingested: total={len(book.events)} "
                f"skipped={skipped} cutoff={cutoff}"
            )

    def midquote_tracking(
        self,
        timestamp: int,
        midquote: float,
        buffer: AgentDataStorage,
        sampling_interval: int,
    ) -> None:
        """Append the current mid-price and prune stale history.

        Retains at least three entries and discards any entry whose timestamp
        predates ``2 × sampling_interval`` nanoseconds ago.

        Args:
            timestamp: Current simulation timestamp (ns).
            midquote: Current mid-price ``(best_bid + best_ask) / 2``.
            buffer: The book's ``AgentDataStorage`` to update.
            sampling_interval: Look-back horizon for pruning (ns); entries
                older than ``2 × sampling_interval`` are removed.
        """
        buffer.midquotes.append(TimestampedPrice(timestamp=timestamp, price=midquote))
        cutoff = timestamp - sampling_interval * 2
        while len(buffer.midquotes) > 3 and buffer.midquotes[0].timestamp < cutoff:
            buffer.midquotes.pop(0)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def predict(
        self,
        buffer: AgentDataStorage,
        tparams: TradingParameters,
        current_timestamp: int,
    ) -> Prediction:
        """Derive a directional prediction from recent volume imbalance and price return.

        Confidence equals ``|mean_imbalance|`` (how lopsided buy vs sell flow
        is).  In advanced mode, strength equals ``log_return × mean_imbalance``:

        - ``> 0`` — flow direction and price move agree → momentum signal.
        - ``< 0`` — flow direction and price move disagree → reversion signal.

        In simple mode, ``strength`` is set equal to ``mean_imbalance`` and
        no price history is required.

        Args:
            buffer: Per-book state containing predictor buckets and mid-price
                history.
            tparams: Trading parameters controlling strategy behaviour.
            current_timestamp: Simulation timestamp (ns) at the time of
                prediction.

        Returns:
            A ``Prediction`` instance.  Returns a zero-confidence prediction
            named ``"NoData"`` or ``"InsufficientPriceHistory"`` when
            prerequisites are not met.
        """
        imbalances = [
            p.imbalance(tparams.strategy_category)
            for p in buffer.predictors
            if p.timestamp > 0
        ]

        if not imbalances:
            return Prediction("NoData", OrderDirection.BUY, 0.0, 0.0, current_timestamp)

        mean_imbalance = float(np.mean(imbalances))
        confidence = abs(mean_imbalance)

        # Map imbalance sign to a trade direction (optionally inverted)
        buy_condition = mean_imbalance > 0 if tparams.reverse_strategy else mean_imbalance < 0
        direction = OrderDirection.BUY if buy_condition else OrderDirection.SELL

        name = (
            f'{tparams.strategy_category.name}Vol'
            f'{"Reversed" if tparams.reverse_strategy else ""}'
        )

        if not tparams.advanced:
            # Simple mode: strength == imbalance, no price history required
            return Prediction(name, direction, confidence, mean_imbalance, current_timestamp)

        # Advanced mode: incorporate recent log-return
        if len(buffer.midquotes) < 2:
            return Prediction("InsufficientPriceHistory", direction, 0.0, 0.0, current_timestamp)

        target_ts = current_timestamp - tparams.sampling_interval
        prev_price = buffer.midquotes[0].price  # Fallback: oldest available
        for mq in reversed(buffer.midquotes[:-1]):
            if mq.timestamp <= target_ts:
                prev_price = mq.price
                break

        current_price = buffer.midquotes[-1].price
        log_return = (
            float(np.log(current_price / prev_price))
            if prev_price > 0 and current_price > 0
            else 0.0
        )

        return Prediction(name, direction, confidence, log_return * mean_imbalance, current_timestamp)

    def signal(self, prediction: Prediction, tparams: TradingParameters) -> Signals:
        """Map a ``Prediction`` to a discrete trading signal.

        Simple mode (``tparams.advanced == False``):

        - ``direction == SELL``  →  ``BEARISH``
        - ``direction == BUY``   →  ``BULLISH``

        Advanced mode:

        - ``confidence < model_threshold``  →  ``NOISE`` (low conviction)
        - ``strength > threshold + tolerance``  →  ``MOMENTUM``
        - ``strength < threshold - tolerance``  →  ``REVERSION``
        - otherwise  →  ``HOLD``

        Args:
            prediction: Output of ``predict()``.
            tparams: Trading parameters defining the signal thresholds.

        Returns:
            A ``Signals`` enum value.
        """
        if not tparams.advanced:
            return Signals.BEARISH if prediction.direction == OrderDirection.SELL else Signals.BULLISH

        if prediction.confidence < tparams.model_threshold:
            return Signals.NOISE
        if prediction.strength > tparams.signal_threshold + tparams.signal_tolerance:
            return Signals.MOMENTUM
        if prediction.strength < tparams.signal_threshold - tparams.signal_tolerance:
            return Signals.REVERSION
        return Signals.HOLD

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def trading_logic(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        signal: Signals,
        prediction: Prediction,
    ) -> FinanceAgentResponse:
        """Route a signal to the appropriate entry or exit action.

        Simple signals (BULLISH/BEARISH) flip the position when the signal
        direction opposes the open position, or enter/extend otherwise.
        Advanced signals follow momentum or reversion logic.
        NOISE and HOLD are no-ops.

        Args:
            response: Accumulator for market orders to be submitted.
            validator: Validator hotkey string.
            book_id: Target order book.
            signal: Discrete signal from ``signal()``.
            prediction: Underlying prediction (for direction and confidence).

        Returns:
            The updated ``response`` with any new market orders appended.
        """
        pos = self.buffers[validator][book_id].positions

        if signal in (Signals.BULLISH, Signals.BEARISH):
            # Simple mode: flip if opposing an open position, otherwise enter
            if pos.open and pos.direction != prediction.direction:
                response, amount, close_dir = self.generate_exit_response(
                    response, validator, book_id, prediction=prediction
                )
                bt.logging.debug(
                    f"EXIT Book={book_id} dir={close_dir} amount={amount} {prediction}",
                    prefix="[TRADE]",
                    suffix=f"[{self.validator_to_str(validator)}]",
                )
            else:
                response = self.entry_or_extend(
                    response, validator, book_id, prediction.direction, prediction=prediction
                )

        elif signal == Signals.REVERSION:
            # Close any existing position (it was in the wrong direction)
            if pos.open:
                response, amount, close_dir = self.generate_exit_response(
                    response, validator, book_id, prediction=prediction
                )
                bt.logging.debug(
                    f"REVERSION EXIT Book={book_id} dir={close_dir} amount={amount} {prediction}",
                    prefix="[TRADE]",
                    suffix=f"[{self.validator_to_str(validator)}]",
                )
            else:
                response = self.entry_or_extend(
                    response, validator, book_id, prediction.direction, prediction=prediction
                )

        elif signal == Signals.MOMENTUM:
            response = self.entry_or_extend(
                response, validator, book_id, prediction.direction, prediction=prediction
            )

        return response

    def entry_or_extend(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        direction: OrderDirection,
        multiplier: float = 1.0,
        prediction: Optional["Prediction"] = None,
        is_emergency: bool = False,
    ) -> FinanceAgentResponse:
        """Open a new position or extend an existing one via a market order.

        A volume-cap guard prevents trading when the annualised notional
        run-rate would exceed ``validator_vol_cap × miner_wealth``.  The guard
        is skipped for emergency trades and at timestamp zero.

        When the open position's direction conflicts with the requested
        direction, a stop-loss exit is triggered if confidence is sufficient;
        otherwise the trade is rejected with an error log.

        Args:
            response: Accumulator for market orders.
            validator: Validator hotkey string.
            book_id: Target order book.
            direction: Desired trade direction.
            multiplier: Scales ``quantity`` from ``TradingParameters``; useful
                for emergency sizing.
            prediction: The ``Prediction`` driving this trade.  May be ``None``
                for emergency trades.
            is_emergency: When ``True``, suppresses Discord notifications and
                bypasses the volume-cap check.

        Returns:
            The updated ``response``.  Unchanged if the trade was blocked by
            the volume cap or a direction conflict.
        """
        pos = self.buffers[validator][book_id].positions
        qty = self.tparams[validator].quantity * multiplier

        # Volume-cap guard (skipped for emergency trades and at timestamp zero)
        trade_history = self.buffers[validator][book_id].returns
        if (
            trade_history
            and prediction is not None
            and prediction.timestamp > 0
            and self.simulation_config.duration > 0
        ):
            fraction = prediction.timestamp / self.simulation_config.duration
            volume_estimate = trade_history[-1].rolling_volume / fraction
            cap = self.simulation_config.miner_wealth * self.tparams[validator].validator_vol_cap
            if volume_estimate > cap:
                bt.logging.error(
                    f"Volume cap exceeded — skipping trade.\n"
                    f"  Annualised estimate: {volume_estimate:.2f}  Cap: {cap:.2f}\n"
                    f"  {trade_history[-1]}"
                )
                return response

        if pos.open:
            if pos.direction != direction:
                # Direction conflict: trigger a stop-loss exit if confidence is sufficient
                if prediction and prediction.confidence > self.tparams[validator].model_threshold:
                    response, _, _ = self.generate_exit_response(
                        response, validator, book_id, prediction, is_emergency=True
                    )
                else:
                    bt.logging.error(
                        f"Direction conflict on Book={book_id}: "
                        f"open={pos.direction.name} requested={direction.name} — skipping."
                    )
                return response

            if self.tparams[validator].skip_extend:
                return response  # Configured not to add to existing positions

            pos.amount += 1
            pos.qty += qty
            label = "EXTEND"
        else:
            pos.direction = direction
            pos.amount = 1
            pos.qty = qty
            pos.open = True
            label = "ENTRY"

        response.market_order(book_id, pos.direction, qty)
        bt.logging.info(
            f"{label} Book={book_id} qty={qty} {prediction}",
            prefix="[TRADE]",
            suffix=f"[{self.validator_to_str(validator)}]",
        )

        if self.discord.enabled and not is_emergency and (
            not self.discord.target_validator or self.discord.target_validator == validator
        ):
            buf = self.buffers[validator][book_id]
            self.discord.add_trade(
                action="BUY" if direction == OrderDirection.BUY else "SELL",
                book_id=book_id,
                total_books=getattr(self.simulation_config, "book_count", 64),
                direction=direction,
                quantity=qty,
                price=buf.midquotes[-1].price if buf.midquotes else 0.0,
            )

        return response

    def generate_exit_response(
        self,
        response: FinanceAgentResponse,
        validator: str,
        book_id: int,
        prediction: Optional["Prediction"] = None,
        is_emergency: bool = False,
    ) -> tuple[FinanceAgentResponse, float, OrderDirection]:
        """Close the entire open position in a single market order.

        Resets the position tracker to flat and optionally queues a Discord
        notification for non-emergency exits.

        Args:
            response: Accumulator for market orders.
            validator: Validator hotkey string.
            book_id: Target order book.
            prediction: The prediction that triggered the exit; used only for
                Discord context.  May be ``None``.
            is_emergency: When ``True``, suppresses Discord notifications.

        Returns:
            A tuple of ``(response, total_qty, close_direction)`` where
            ``total_qty`` is the quantity submitted and ``close_direction``
            is the direction of the closing order.
        """
        pos = self.buffers[validator][book_id].positions
        pos.open = False

        close_dir = OrderDirection(abs(pos.direction - 1))
        total_qty = pos.qty
        pos.amount = 0
        pos.qty = 0.0

        response.market_order(book_id, close_dir, total_qty)

        if self.discord.enabled and not is_emergency and (
            not self.discord.target_validator or self.discord.target_validator == validator
        ):
            buf = self.buffers[validator][book_id]
            self.discord.add_trade(
                action="BUY" if close_dir == OrderDirection.BUY else "SELL",
                book_id=book_id,
                total_books=getattr(self.simulation_config, "book_count", 64),
                direction=close_dir,
                quantity=total_qty,
                price=buf.midquotes[-1].price if buf.midquotes else 0.0,
            )

        return response, total_qty, close_dir

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """Process one validator state update and return trading instructions.

        For each book in the update:

        1. Initialise buffers if needed.
        2. Ingest new market trades into the predictor window.
        3. Track the current mid-quote.
        4. If insufficient data, consider an emergency trade.
        5. Rate-limit signal generation to ``sampling_interval``.
        6. Generate a prediction, derive a signal, and execute trading logic.

        After all books are processed, the Optuna hyperparameter optimisation
        step runs if ``opt_interval`` nanoseconds have elapsed, and any pending
        Discord notifications are flushed.

        Args:
            state: The market state update delivered by the validator.

        Returns:
            A ``FinanceAgentResponse`` containing all market orders to submit.
        """
        # GenTRX: data collection + training trigger (runs even when training disabled).
        response = super().respond(state)
        validator = state.dendrite.hotkey
        start = time.time()

        # One-shot Discord smoke-test triggered by the ``run_discord_test`` config flag
        if (
            self.discord.enabled
            and getattr(self.config, "run_discord_test", False)
            and not hasattr(self, "_discord_test_completed")
        ):
            if any(book.bids and book.asks for book in state.books.values()):
                bt.logging.info("Running Discord notification test...")
                result = self._test_discord_with_live_data(state)
                self._discord_test_completed = True
                return result

        for book_id, book in state.books.items():
            try:
                best_bid = book.bids[0].price if book.bids else 0.0
                best_ask = (
                    book.asks[0].price if book.asks
                    else best_bid + 10 ** (-self.simulation_config.priceDecimals)
                )
                midquote = (best_bid + best_ask) / 2

                self.buffers_init(state, book_id)
                self.update_predictors(validator, book, state.timestamp)

                buffer = self.buffers[validator][book_id]
                buffer.traded_volume = state.accounts[self.uid][book_id].traded_volume
                self.midquote_tracking(
                    state.timestamp, midquote, buffer, self.tparams[validator].sampling_interval
                )

                # Not enough predictor data yet — consider an emergency trade
                if len(buffer.predictors) < self.tparams[validator].samples:
                    self._maybe_emergency_trade(
                        response, state, validator, book_id, midquote, buffer
                    )
                    continue

                # Rate-limit signals to the sampling interval
                if buffer.returns and (
                    state.timestamp - self.tparams[validator].sampling_interval
                    < buffer.returns[-1].timestamp
                ):
                    continue

                prediction = self.predict(buffer, self.tparams[validator], state.timestamp)
                sig = self.signal(prediction, self.tparams[validator])

                bt.logging.info(
                    f"Book={book_id} {prediction} signal={sig.name} mid={midquote:.4f}",
                    prefix="[SIGNAL]",
                )
                buffer.last_signal = sig
                response = self.trading_logic(response, validator, book_id, sig, prediction)

            except Exception as e:
                bt.logging.error(
                    f"Book {book_id} processing failed: {e}",
                    prefix=f"[{self.validator_to_str(validator)}]",
                )
                bt.logging.error(traceback.format_exc())

        # Periodic hyperparameter optimisation
        if state.timestamp - self.studies[validator].trial_time > self.studies[validator].opt_interval:
            self.evaluate_and_update_hp(validator)
            self.studies[validator].trial_time = state.timestamp

        if self.discord.enabled:
            self.discord.flush_trades()

        bt.logging.debug(f"respond() took {time.time() - start:.2f}s")
        return response

    def _maybe_emergency_trade(
        self,
        response: FinanceAgentResponse,
        state: MarketSimulationStateUpdate,
        validator: str,
        book_id: int,
        midquote: float,
        buffer: AgentDataStorage,
    ) -> None:
        """Fire an emergency open or close when no signal data is available.

        Ensures the agent has a position while waiting for the predictor window
        to fill.  Three conditions must all be true before an emergency trade
        fires:

        1. The full ``validator_lookback`` warm-up period has elapsed since
           buffer creation.
        2. The trade history is thin (fewer than 3 fills) or the last fill is
           older than ``2/3 × validator_lookback``.
        3. At least ``validator_lookback / 6`` nanoseconds have elapsed since
           the last emergency trade (rate limit).

        Emergency trades are never reported to Discord.

        Args:
            response: Accumulator for market orders; modified in-place.
            state: Current market state update.
            validator: Validator hotkey string.
            book_id: Target order book.
            midquote: Current mid-price, used to choose direction on entry.
            buffer: Per-book state for this (validator, book) pair.
        """
        tp = self.tparams[validator]
        emergency_interval = tp.validator_lookback // 6
        time_since_startup = state.timestamp - buffer.startup_timestamp

        has_waited = time_since_startup >= tp.validator_lookback
        needs_trade = len(buffer.returns) < 3 or (
            buffer.returns
            and (state.timestamp - buffer.returns[-1].timestamp) > int(tp.validator_lookback * 2 / 3)
        )
        can_fire = has_waited and (
            state.timestamp - buffer.last_emergency_timestamp
        ) >= emergency_interval

        if not (needs_trade and can_fire):
            bt.logging.info(
                f"BOOK {book_id} | Waiting for data: "
                f"{len(buffer.predictors)}/{tp.samples} buckets, "
                f"{time_since_startup / 1e9:.0f}s / {tp.validator_lookback / 1e9:.0f}s elapsed",
            )
            return

        buffer.last_emergency_timestamp = state.timestamp

        if buffer.positions.open:
            self.generate_exit_response(response, validator, book_id, is_emergency=True)
            bt.logging.info(f"BOOK {book_id} | Emergency close (no signal data).")
        else:
            direction = (
                OrderDirection.BUY if midquote >= buffer.midquotes[0].price else OrderDirection.SELL
            )
            self.entry_or_extend(response, validator, book_id, direction, is_emergency=True)
            bt.logging.info(f"BOOK {book_id} | Emergency open dir={direction.name}.")

    # ------------------------------------------------------------------
    # Optuna optimisation
    # ------------------------------------------------------------------

    def evaluate_trial(self, validator: str) -> float:
        """Estimate strategy performance as median rolling-quote P&L across books.

        Computes the change in ``rolling_quote`` between the first and last
        recorded fill for each book, then returns the median across all books.
        Returns ``0.0`` when no trade history exists, which penalises idle
        hyperparameter configurations.

        Args:
            validator: Validator hotkey string.

        Returns:
            Median rolling-quote P&L across all books with at least two fills,
            or ``0.0`` if no such books exist.
        """
        buffer = self.buffers.get(validator)
        if not buffer:
            return 0.0

        returns = [
            agent_data.returns[-1].rolling_quote - agent_data.returns[0].rolling_quote
            for agent_data in buffer.values()
            if len(agent_data.returns) > 1
        ]
        return float(np.median(returns)) if returns else 0.0

    def evaluate_and_update_hp(self, validator: str) -> None:
        """Score the active trial, update the study, and sample the next trial.

        If no study exists for the validator (e.g. after ``onEnd`` cleared
        state), a new one is initialised before returning to avoid crashes.
        If the trial cache is empty, a warning is logged and a new trial is
        sampled anyway to keep the optimisation loop running.

        Args:
            validator: Validator hotkey string.
        """
        if validator not in self.studies:
            # Reinitialise rather than crash if the study was removed
            self.reset_study(validator, "backup")
            return

        gain = self.evaluate_trial(validator)
        sp = self.studies[validator]
        trial_id = None  # Stays None when the cache is empty

        if sp.trial_cache:
            trial_id, trial = sp.trial_cache.popitem()
            if trial is not None:
                sp.study = tell_study(sp.study, gain, trial, sp.study_file)
                bt.logging.info(f"Trial {trial_id} scored {gain:.4f}.", prefix="[TRIAL]")
            else:
                bt.logging.warning(f"Trial {trial_id} was None — skipped.", prefix="[TRIAL]")
        else:
            bt.logging.warning("No pending trial found.", prefix="[TRIAL]")

        next_trial, hp = ask_trial(sp.study)
        sp.trial_cache[next_trial.number] = next_trial
        bt.logging.info(f"Next trial params: {hp}", prefix="[TRIAL]")

        try:
            bt.logging.info(
                f"Study best: value={sp.study.best_value:.4f} "
                f"params={sp.study.best_params} trial={sp.study.best_trial.number}",
                prefix="[TRIAL]",
            )
        except ValueError:
            pass  # No completed trials yet — normal at startup

        self.change_trading_params(validator, hp)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def validator_to_str(self, validator: str) -> str:
        """Return a short display form of a validator hotkey.

        Args:
            validator: Full validator hotkey string.

        Returns:
            ``"AAAAA...ZZ"`` for hotkeys longer than 10 characters, or the
            original string otherwise.
        """
        return f"{validator[:5]}...{validator[-2:]}" if len(validator) > 10 else validator

    # ------------------------------------------------------------------
    # Discord smoke-test
    # ------------------------------------------------------------------

    def _test_discord_with_live_data(
        self, state: MarketSimulationStateUpdate
    ) -> FinanceAgentResponse:
        """Send six test notifications using real book data to verify the Discord pipeline.

        Fires four reported trades (two buy/sell pairs) followed by two silent
        emergency trades.  The four reported trades are batched and flushed as
        a single Discord message.  Activates when ``run_discord_test=True`` is
        set in the agent config.

        Args:
            state: The current market state, used to find a book with live
                bids and asks.

        Returns:
            A ``FinanceAgentResponse`` containing the test market orders (these
            are real orders and will be executed by the exchange).
        """
        response = FinanceAgentResponse(agent_id=self.uid)
        if not self.discord.enabled:
            bt.logging.warning("Discord test skipped — notifications disabled.")
            return response

        test_book_id = next(
            (bid for bid, book in state.books.items() if book.bids and book.asks), None
        )
        if test_book_id is None:
            bt.logging.error("No book with bids and asks found for Discord test.")
            return response

        validator = state.dendrite.hotkey
        book = state.books[test_book_id]
        self.buffers_init(state, test_book_id)
        self.update_predictors(validator, book, state.timestamp)
        buf = self.buffers[validator][test_book_id]

        midquote = (book.bids[0].price + book.asks[0].price) / 2
        buf.midquotes.append(TimestampedPrice(timestamp=state.timestamp, price=midquote))

        # Use a real prediction when data permits; fall back to a synthetic one
        if len(buf.predictors) >= self.tparams[validator].samples:
            pred = self.predict(buf, self.tparams[validator], state.timestamp)
        else:
            pred = Prediction(
                name="TestPrediction",
                direction=OrderDirection.BUY,
                confidence=0.5,
                strength=-0.1,
                timestamp=state.timestamp,
            )

        pred_sell = Prediction(
            pred.name, OrderDirection.SELL, pred.confidence, -pred.strength, state.timestamp
        )

        bt.logging.info("=== Discord test: 4 reported trades + 2 silent emergency trades ===")
        for direction, prediction, label in [
            (OrderDirection.BUY,  pred,      "BUY ENTRY"),
            (None,                pred,      "EXIT (SELL to close BUY)"),
            (OrderDirection.SELL, pred_sell, "SELL ENTRY"),
            (None,                pred_sell, "EXIT (BUY to close SELL)"),
        ]:
            bt.logging.info(f"  Test: {label}")
            if direction is not None:
                response = self.entry_or_extend(
                    response, validator, test_book_id, direction, prediction=prediction
                )
            else:
                response, _, _ = self.generate_exit_response(
                    response, validator, test_book_id, prediction=prediction
                )

        self.discord.flush_trades()

        for direction, label in [
            (OrderDirection.BUY,  "EMERGENCY OPEN (silent)"),
            (None,                "EMERGENCY CLOSE (silent)"),
        ]:
            bt.logging.info(f"  Test: {label}")
            if direction is not None:
                response = self.entry_or_extend(
                    response, validator, test_book_id, direction,
                    prediction=None, is_emergency=True,
                )
            else:
                response, _, _ = self.generate_exit_response(
                    response, validator, test_book_id, prediction=None, is_emergency=True
                )

        bt.logging.info("Discord test complete — check for 1 batched message with 4 lines.")
        return response


if __name__ == "__main__":
    """Launch RevengAgent for local testing.

    Example::

        python RevengAgent.py \\
            --port 8888 \\
            --agent_id 0 \\
            --params \\
                discord_enabled=True \\
                discord_webhook_url=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN \\
                discord_target_validator=5GrwvaEF... \\
                discord_rate_limit=1 \\
                quantity=2.0 \\
                samples=1 \\
                sampling_interval=120 \\
                run_discord_test=True

    Discord parameters:

    - ``discord_enabled``: Enable notifications (default: ``False``).
    - ``discord_webhook_url``: Webhook URL.
    - ``discord_target_validator``: If set, only trades for this validator are sent.
    - ``discord_rate_limit``: Minimum seconds between outgoing messages (default: ``1``).
    - ``run_discord_test``: Fire a one-shot 6-trade test on the first viable tick.

    Notes:
        Trades are batched per ``respond()`` call into a single message.
        Emergency trades are never reported to Discord.
    """
    launch(RevengAgent)