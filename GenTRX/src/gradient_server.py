# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRX gradient exchange — event-driven aggregation server.

HTTP endpoints:
  POST /gentrx/state                  — receive sim state ticks from validator
  GET  /gentrx/assignment?miner_uid=N — fetch assignment for a miner
  GET  /gentrx/version                — check current model version
  GET  /gentrx/scores?since_round=N   — poll latest scores
  GET  /gentrx/metrics                — Prometheus metrics (scrape target for Grafana)

Two bucket types, each committed on-chain (state arrives over HTTP — no state bucket):
  - Validator bucket (GENTRX_VALIDATOR_S3_*): checkpoints/ + data/ + proposals/
  - Per-miner buckets (GENTRX_AGENT_S3_*): gradients/ — one per miner

Run standalone:
    python -m GenTRX.src.gradient_server --checkpoint checkpoints/GenTRX/best.pt \\
        --val-data data/server --subtensor-network local --netuid 1
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger("GenTRX.src.gradient_server")


def _tag_to_ns(tag: str) -> int:
    """Convert ddHHMMSS tag back to nanoseconds.

    Also handles plain integer strings (sim format fallback).
    """
    if len(tag) == 8:
        dd = int(tag[0:2])
        hh = int(tag[2:4])
        mm = int(tag[4:6])
        ss = int(tag[6:8])
        if hh < 24 and mm < 60 and ss < 60:
            total_secs = dd * 86400 + hh * 3600 + mm * 60 + ss
            return total_secs * 1_000_000_000
    return int(tag)


def _ts_to_tag(ts_ns: int) -> str:
    """Convert nanosecond timestamp to ddHHMMSS tag for filenames."""
    total_secs = ts_ns // 1_000_000_000
    days = total_secs // 86400
    remainder = total_secs % 86400
    hours = remainder // 3600
    minutes = (remainder % 3600) // 60
    seconds = remainder % 60
    return f"{days:02d}{hours:02d}{minutes:02d}{seconds:02d}"


def _filter_by_timestamp(
    files: list[Path],
    ts_start: int,
    ts_end: int,
) -> list[Path]:
    """Filter interval parquet files by timestamp range (nanoseconds).

    Supports two filename formats:
      - ddHHMMSS-ddHHMMSS.parquet (live format, e.g. "01013000-01020000.parquet")
      - NNNNNNNN-NNNNNNNN.parquet (sim format, raw integers)

    A file is included if its range overlaps with [ts_start, ts_end].
    ts_start=0 means from beginning, ts_end=0 means to end.
    """
    result = []
    for f in files:
        parts = f.stem.split("-")
        if len(parts) != 2:
            result.append(f)  # can't parse, include by default
            continue
        try:
            file_start = _tag_to_ns(parts[0])
            file_end = _tag_to_ns(parts[1])
        except ValueError:
            result.append(f)
            continue

        if ts_end and file_start >= ts_end:
            continue
        if ts_start and file_end <= ts_start:
            continue
        result.append(f)
    return result


class GradientAggregator:
    """Thread-safe gradient accumulator + periodic aggregation.

    Assignment protocol:
      - Global val book split (10% held out, deterministic from seed).
      - Each round, assigns miners specific books + timestamp window.
      - Beta(alpha, beta) samples a start point in [0, max_data_time - window],
        biased toward recent data. No interval index abstraction.
      - Books distributed round-robin across miners per round.
      - Round advances after each aggregation cycle.
    """

    # Default training window: 5 minutes of sim time
    DEFAULT_WINDOW_NS = 300_000_000_000

    def __init__(
        self,
        checkpoint_path: str,
        val_data_path: str,
        output_path: str | None = None,
        log_path: str | None = None,
        books_per_miner: int = 3,
        val_fraction: float = 0.10,
        min_score: float = -0.1,
        warmup_rounds: int = 5,
        rollback: bool = True,
        interval: float = 30.0,
        max_val_batches: int = 10,
        beta_alpha: float = 2.0,
        beta_beta: float = 5.0,
        seed: int = 42,
        window_ns: int | None = None,
        chain: Any | None = None,
        validator_store: Any | None = None,
        is_aggregator: bool = True,
        parquet_interval_ns: int = 300_000_000_000,  # 5 min, matches training window
        loop_sleep_s: float = 5.0,
        round_grace_s: float = 30.0,
        max_gradient_bytes: int = 10 * 1024 * 1024,
        overfit_ratio: float = 3.0,
        overfit_penalty: float = 0.1,
        keep_checkpoints: int = 10,
        keep_proposals: int = 10,
        blocks_per_round: int = 25,
        block_time_s: float = 12.0,
        validator_uid: str = "",
        bucket_prefix: str = "",
    ):
        # Default to "0" so unscoped invocations (proxy/localnet) write
        # proposals/scores under a stable path; multi-validator setups
        # must pass --validator-uid to avoid collisions in shared buckets.
        self._validator_uid: str = str(validator_uid) if validator_uid not in (None, "") else "0"
        self._bucket_prefix: str = bucket_prefix
        self.checkpoint_path = Path(checkpoint_path)
        self.val_data_path = val_data_path
        self.output_path = Path(output_path or checkpoint_path)
        self.log_path = (
            Path(log_path)
            if log_path
            else self.output_path.parent / "aggregation.jsonl"
        )
        self.books_per_miner = books_per_miner
        self.val_fraction = val_fraction
        self.min_score = min_score
        self.warmup_rounds = warmup_rounds
        self.rollback = rollback
        self.interval = interval
        self.max_val_batches = max_val_batches
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.window_ns = window_ns or self.DEFAULT_WINDOW_NS
        # Retention on the validator bucket. 0 disables pruning (objects
        # accumulate indefinitely; operator handles cleanup themselves).
        # Defaults: 10 checkpoints (~470 MB at 47 MB each), 10 proposals.
        self.keep_checkpoints = keep_checkpoints
        self.keep_proposals = keep_proposals
        # Round-completion fallback timing. The validator drives round
        # closure via POST /gentrx/round (block-sync); these only feed the
        # fallback that fires when the validator stops pushing.
        self.blocks_per_round = blocks_per_round
        self.block_time_s = block_time_s

        # Single validator bucket: checkpoints/ + data/ + proposals/ all in one place.
        # Committed to chain so miners and the aggregator discover it without
        # pre-configuration. 
        self.validator_store = validator_store
        # When True (uid-0 aggregator): reads cross-validator scores from chain,
        # aggregates gradients, publishes checkpoint.
        # When False (sibling validator): scores miners, publishes scores only.
        self.is_aggregator: bool = is_aggregator

        # On-chain bucket discovery (optional). When set, gradients are
        # collected from per-miner buckets using committed read credentials
        # instead of from the validator's single bucket.
        self._chain = chain
        self._miner_buckets: dict[int, Any] = {}
        # Static buckets registered via /register_bucket (benchmark miners that
        # can't commit to chain). Merged into _miner_buckets after each refresh.
        self._static_buckets: dict[int, Any] = {}
        # Cooldown so we don't re-query chain every loop iteration when the
        # result is empty (no miners committed yet). Both call sites go through
        # _refresh_miner_buckets().
        self._miner_buckets_queried_at: float = 0.0
        self._miner_buckets_refresh_s: float = 30.0
        self._version = 0
        self._agg_round = 0
        self._last_round_log_ts: float = 0.0
        self._last_push_block: int | None = None  # block number from most recent POST /round
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Monitoring identity. sim_id is the simulator-issued identifier for
        # the current sim run, sourced from the first state packet's config
        # block (state.config.simulation_id). sim_epoch bumps when we observe
        # sim time going backwards inside one gradient server lifetime —
        # catches sim restarts the simulator forgot to re-id. Both are stamped
        # on every aggregation.jsonl event so the dashboard can tell sessions
        # apart instead of blending them.
        self._sim_id: str | None = None
        self._sim_epoch: int = 0
        self._last_seen_sim_ts: int = 0
        # Sim id from the pending-rows staging file. Cleared once the
        # first config packet arrives and the sim-swap check has run.
        self._restored_sim_id: str | None = None
        # Sim transition handling. Set to True when an 'ESE' marker is seen
        # (or when the heuristic detects a restart without an explicit
        # marker). Picked up by the aggregation loop on its next tick,
        # which wipes stale parquets under the validator bucket's data/
        # prefix. Checkpoints and proposals are intentionally not touched.
        self._data_cleanup_pending: bool = False
        # Last model version stamped onto an assignment. When _version advances
        # past this, the round-completion fallback timer is extended so miners
        # have time to download the new checkpoint before uploading.
        self._last_assigned_version: int = 0
        # Multiplier applied to the round-completion fallback estimate when
        # the checkpoint just rolled.
        self._round_rollover_mult: float = 1.5

        # Latest scores (in-memory, served via GET /gentrx/scores)
        self._latest_scores: dict | None = None

        # Aggregation snapshot — updated on each aggregation; embedded in
        # the scores payload so the validator-side Prometheus collector can
        # track training progress without needing a second HTTP call.
        self._rounds_aggregated_total: int = 0
        self._rollbacks_total: int = 0
        self._last_aggregation: dict = {}

        # ---- Assignment lifecycle tracking ----
        # States: PENDING → DATA_READY → DELIVERED → GRADIENT_IN → SCORED
        # miner_uid → MinerAssignment dict — HOLDS ONLY THE CURRENT ROUND.
        self._assignments: dict[int, dict] = {}
        # Closing-round assignments preserved across a POST /round that
        # advances _agg_round. Without this the previous round's _state /
        # _gradient_data would be overwritten by the new round's installation
        # before the aggregation loop had a chance to collect + score.
        # uid → MinerAssignment dict (round field on each assignment tells us
        # which round it belongs to). Cleared after aggregation completes.
        self._prev_round_assignments: dict[int, dict] = {}
        # Rounds queued for aggregation by POST /round. The aggregation loop
        # drains this set on each iteration so heavy aggregation never runs
        # on the FastAPI event loop.
        self._pending_aggregation_rounds: set[int] = set()
        # Wall-clock time when each round was queued by POST /round. Used
        # to bound the post-close grace window before stragglers are
        # scored 0.
        self._pending_aggregation_at: dict[int, float] = {}

        # Global val/train book split (deterministic from seed)
        self._all_books: list[str] = []
        self._val_book_set: set[str] | None = None
        self._seed = seed

        # Track simulation progress (updated as data arrives)
        self._max_timestamp_ns: int = 0

        # S3: track processed gradient keys to avoid double-scoring
        self._processed_grad_keys: set[str] = set()

        # Price/volume scaling from simulator config (set from first state packet)
        self._price_scale: int | None = None
        self._vol_scale: int | None = None

        # Parquet interval batching: accumulate rows per book, flush when
        # sim timestamp crosses an interval boundary.
        self._parquet_interval_ns: int = parquet_interval_ns
        self._loop_sleep_s: float = loop_sleep_s
        self._round_grace_s: float = round_grace_s
        self._max_gradient_bytes: int = max_gradient_bytes
        # Scoring-model cache for eager scoring. Loaded on the first eager
        # score per round, reused for siblings, evicted at aggregation
        # boundary (because aggregation mutates the model in place). None
        # means "not loaded yet for this round".
        # Shape: {"round": int, "model": ..., "model_cfg": ..., "tokenizer_cfg": ...,
        #         "tokenizer": ..., "device": str}
        self._scoring_cache: dict | None = None
        self.overfit_ratio: float = overfit_ratio
        self.overfit_penalty: float = overfit_penalty
        self._pending_rows: dict[int, list[dict]] = {}  # book_id → rows
        self._pending_interval_start: dict[int, int] = {}  # book_id → interval start ts
        # Registry of written parquets (avoids S3 LIST for data-readiness checks)
        # book_id → [(filename, ts_start_ns, ts_end_ns), ...]
        self._written_parquets: dict[int, list[tuple[str, int, int]]] = {}
        # Per-book engine state persists across _process_tick calls
        self._engines: dict[int, Any] = {}
        self._order_sides: dict[int, dict[int, bool]] = {}
        self._last_ts: dict[int, int] = {}
        self._session_open_mid: dict[int, int | None] = {}
        # Local staging file for in-progress parquet buffer: persisted
        # periodically so a restart can continue filling the current window
        # rather than starting a fresh 5-min accumulation.
        self._pending_staging_path: Path = self.checkpoint_path.parent / "pending_rows.msgpack"
        self._last_pending_save_ts: float = 0.0

        # S3 data cache: downloaded parquets are cached locally to avoid
        # re-downloading every aggregation cycle. Key: S3 key → local path.
        self._s3_cache_dir: Path | None = None
        self._s3_cached_files: dict[str, Path] = {}

        # Fresh-model flag: set when we create a seed checkpoint (no prior
        # training).  Enables warmup — first N rounds accept all gradients
        # and skip val rollback so the model can bootstrap from random weights.
        self._fresh_start = False

        # Lazy-loaded
        self._model = None
        self._global_val_loader = None

    @property
    def _effective_min_score(self) -> float:
        """Min score threshold, disabled during warmup rounds after fresh start."""
        if self._fresh_start and self._agg_round < self.warmup_rounds:
            return float("-inf")
        return self.min_score

    @property
    def _effective_rollback(self) -> bool:
        """Rollback check, disabled during warmup rounds after fresh start."""
        if self._fresh_start and self._agg_round < self.warmup_rounds:
            return False
        return self.rollback

    def _discover_books(self) -> list[str]:
        """Discover available books from incoming state, S3, or filesystem.

        Sources in priority order — first non-empty wins:
          1. _written_parquets — definitive, set after first parquet flush.
          2. _pending_rows / _engines — books seen in state ticks but not yet
             flushed (lets us discover from tick #1, before the 5-min flush).
          3. data_store.list_books() — restart resilience: gradient server
             came back up against a populated S3 bucket.
          4. val_data_path filesystem scan — legacy local-only mode.

        Returns [] when nothing is known yet. Callers (_create_round_assignments,
        _create_assignment_for) early-return on empty, so the aggregator simply
        waits for state to arrive instead of inventing fake book IDs.

        Note: the live network is expected to run with ~128 books per validator
        with room to grow; nothing in this module hard-codes that count.
        """
        # 1. Flushed parquets (in-memory registry, no I/O)
        if self._written_parquets:
            return [str(bid) for bid in sorted(self._written_parquets.keys())]

        # 2. State ticks observed but not yet flushed
        if self._pending_rows:
            return [str(bid) for bid in sorted(self._pending_rows.keys())]

        # 3. S3 — restart against persistent bucket
        if self.validator_store is not None:
            try:
                book_ids = self.validator_store.list_books(self._validator_uid)
                if book_ids:
                    return [str(bid) for bid in book_ids]
            except Exception as exc:
                logger.debug("S3 book discovery failed: %s", exc)

        # 4. Local filesystem (legacy local-only mode)
        data_path = Path(self.val_data_path)
        if data_path.exists():
            books = [
                d.name for d in sorted(data_path.iterdir())
                if d.is_dir() and (d / "intervals").is_dir()
            ]
            if books:
                return books

        return []

    def _get_val_books(self) -> set[str]:
        """Global val book set (deterministic from seed, stable across rounds)."""
        if self._val_book_set is not None:
            return self._val_book_set

        if not self._all_books:
            self._all_books = self._discover_books()
        if not self._all_books:
            return set()

        n_val = max(1, int(len(self._all_books) * self.val_fraction))
        rng = random.Random(self._seed)
        self._val_book_set = set(
            rng.sample(self._all_books, min(n_val, len(self._all_books)))
        )
        logger.info(
            "Val books: %s (%d/%d)",
            sorted(self._val_book_set),
            len(self._val_book_set),
            len(self._all_books),
        )
        return self._val_book_set

    def _get_train_books(self) -> list[str]:
        """Train books (all books minus val books)."""
        if not self._all_books:
            self._all_books = self._discover_books()
        val = self._get_val_books()
        return [b for b in self._all_books if b not in val]

    def update_max_timestamp(self, ts_ns: int) -> None:
        """Update the maximum observed timestamp (called as data arrives)."""
        if ts_ns > self._max_timestamp_ns:
            self._max_timestamp_ns = ts_ns

    def get_assignment(self, miner_uid: int) -> dict | None:
        """Return an assignment for a miner, only if data is ready.

        Creates an assignment on demand if one doesn't exist for this UID.
        The gradient server doesn't need to know miner UIDs upfront —
        any UID can request an assignment and get one created.

        Returns None if data is not ready yet (PENDING state).
        Returns the assignment dict when state is DATA_READY.
        Also transitions to DELIVERED and stamps `_delivered_at`.
        """
        a = self._assignments.get(miner_uid)

        # Create on demand if no assignment exists for this UID
        if a is None or a.get("round") != self._agg_round:
            self._create_assignment_for(miner_uid)
            # Check data readiness immediately
            a = self._assignments.get(miner_uid)
            if a and a.get("_state") == "PENDING":
                data_keys = self._resolve_data_keys(a["books"], a["ts_start"], a["ts_end"])
                if data_keys:
                    a["data"] = data_keys
                    a["_state"] = "DATA_READY"

        a = self._assignments.get(miner_uid)
        if a is None:
            return None
        if a.get("_state") != "DATA_READY":
            return None

        # Transition to DELIVERED, stamp the delivery time. Used by the
        # heartbeat-loss fallback in _round_complete.
        a["_state"] = "DELIVERED"
        a["_delivered_at"] = time.time()
        logger.debug(
            "Assignment delivered: miner=%d round=%d books=%s data=%d files",
            miner_uid, a.get("round", 0), a.get("books", []), len(a.get("data", []))
        )
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def _create_round_assignments(self) -> None:
        """Create new PENDING assignments for all miners for the current round.

        Called after aggregation completes (or at startup).
        """
        self._all_books = self._discover_books()
        if not self._all_books:
            return

        max_ts = self._get_max_data_timestamp()
        if max_ts < self.window_ns:
            if not getattr(self, "_warned_waiting_for_data", False):
                logger.info(
                    "[GTX] Waiting for sim data. First round cannot be created "
                    "until max_ts >= window_ns (%.0f s). With the default sim "
                    "gracePeriod (10 min) plus window (%.0f s), expect the "
                    "first round around %.0f min of sim time.",
                    self.window_ns / 1e9,
                    self.window_ns / 1e9,
                    (600 + self.window_ns / 1e9) / 60,
                )
                self._warned_waiting_for_data = True
            else:
                logger.debug("Not enough data yet (max_ts=%d < window=%d)", max_ts, self.window_ns)
            return

        max_start = max_ts - self.window_ns
        train_books = self._get_train_books()
        if not train_books:
            train_books = list(self._all_books)

        book_rng = random.Random(
            hashlib.sha256(f"{self._agg_round}:books".encode()).hexdigest()
        )
        shuffled = list(train_books)
        book_rng.shuffle(shuffled)

        # Determine number of miners — chain commitments first, then previous
        # assignments, then a default. _refresh_miner_buckets() handles the
        # empty-cache cooldown so this is cheap even when no miners have
        # committed yet.
        self._refresh_miner_buckets()
        n_miners = max(len(self._miner_buckets), len(self._assignments), 2)

        for miner_uid in range(n_miners):
            miner_rng = random.Random(
                hashlib.sha256(f"{self._agg_round}:{miner_uid}:time".encode()).hexdigest()
            )
            beta_sample = 1.0 - miner_rng.betavariate(self.beta_alpha, self.beta_beta)
            ts_start = int(beta_sample * max_start)
            ts_end = ts_start + self.window_ns

            start = (miner_uid * self.books_per_miner) % len(shuffled)
            assigned = []
            for i in range(self.books_per_miner):
                assigned.append(shuffled[(start + i) % len(shuffled)])

            self._assignments[miner_uid] = {
                "round": self._agg_round,
                "model_version": self._version,
                "books": assigned,
                "ts_start": ts_start,
                "ts_end": ts_end,
                "data": [],  # resolved later by _check_data_readiness
                "data_source": "s3" if self.validator_store else "local",
                "data_endpoint": self.validator_store.endpoint_url if self.validator_store else "",
                "data_bucket": self.validator_store.bucket if self.validator_store else "",
                "data_access_key": self.validator_store.access_key if self.validator_store else "",
                "data_secret_key": self.validator_store.secret_key if self.validator_store else "",
                "_state": "PENDING",
                "_created_at": time.time(),
                "_delivered_at": None,
                "_gradient_data": None,
                "_score": None,
            }

        rolled = self._version > self._last_assigned_version
        self._last_assigned_version = self._version
        logger.info(
            "[GTX] round=%d: created %d assignments (books=%d, window=%ds%s)",
            self._agg_round, n_miners, len(shuffled),
            self.window_ns // 1_000_000_000,
            ", model rollover" if rolled else "",
        )

    def _create_assignment_for(self, miner_uid: int) -> None:
        """Create a PENDING assignment for a specific miner UID."""
        self._all_books = self._discover_books()
        if not self._all_books:
            return

        max_ts = self._get_max_data_timestamp()
        if max_ts < self.window_ns:
            return

        max_start = max_ts - self.window_ns
        train_books = self._get_train_books()
        if not train_books:
            train_books = list(self._all_books)

        book_rng = random.Random(
            hashlib.sha256(f"{self._agg_round}:books".encode()).hexdigest()
        )
        shuffled = list(train_books)
        book_rng.shuffle(shuffled)

        miner_rng = random.Random(
            hashlib.sha256(f"{self._agg_round}:{miner_uid}:time".encode()).hexdigest()
        )
        beta_sample = 1.0 - miner_rng.betavariate(self.beta_alpha, self.beta_beta)
        ts_start = int(beta_sample * max_start)
        ts_end = ts_start + self.window_ns

        start = (miner_uid * self.books_per_miner) % len(shuffled)
        assigned = [shuffled[(start + i) % len(shuffled)] for i in range(self.books_per_miner)]

        self._assignments[miner_uid] = {
            "round": self._agg_round,
            "model_version": self._version,
            "books": assigned,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "data": [],
            "data_source": "s3" if self.validator_store else "local",
            "data_endpoint": self.validator_store.endpoint_url if self.validator_store else "",
            "data_bucket": self.validator_store.bucket if self.validator_store else "",
            "data_access_key": self.validator_store.access_key if self.validator_store else "",
            "data_secret_key": self.validator_store.secret_key if self.validator_store else "",
            "_state": "PENDING",
            "_created_at": time.time(),
            "_delivered_at": None,
            "_gradient_data": None,
            "_score": None,
        }
        # Mark this version as assigned (idempotent: first assignment of
        # the round drives the rollover-bump check; on-demand creates
        # inherit the same value).
        self._last_assigned_version = max(self._last_assigned_version, self._version)
        logger.debug("Assignment created on demand: miner=%d round=%d books=%s", miner_uid, self._agg_round, assigned)

    def _round_estimate_s(self) -> float:
        """Server-side estimate of how long a round should take.

        Used only by the round-completion fallback in `_round_complete`,
        which fires when the validator stops pushing `POST /gentrx/round`.
        Derived from `blocks_per_round * block_time_s`. On a fresh
        checkpoint roll the estimate is bumped so miners have time to
        download the new model before uploading; the bump fires on the
        first round whose model_version advanced past the previous one.
        """
        base = self.blocks_per_round * self.block_time_s
        if self._version > self._last_assigned_version:
            return base * self._round_rollover_mult
        return base

    def _check_data_readiness(self) -> None:
        """Move PENDING assignments to DATA_READY when S3 data exists."""
        for uid, a in self._assignments.items():
            if a.get("_state") != "PENDING":
                continue
            data_keys = self._resolve_data_keys(a["books"], a["ts_start"], a["ts_end"])
            if data_keys:
                a["data"] = data_keys
                a["_state"] = "DATA_READY"
                logger.debug(
                    "Assignment data ready: miner=%d round=%d files=%d",
                    uid, a["round"], len(data_keys),
                )

    def _resolve_data_keys(
        self,
        book_ids: list[str],
        ts_start: int,
        ts_end: int,
    ) -> list[str]:
        """Resolve data keys for assigned books + timestamp range.

        Uses the in-memory registry of written parquets (no S3 LIST).
        Returns S3 key strings that miners can fetch from the data bucket.
        """
        keys: list[str] = []

        for book_id in book_ids:
            bid = int(book_id)
            parquets = self._written_parquets.get(bid, [])
            for fname, f_start, f_end in parquets:
                # Check overlap with [ts_start, ts_end]
                if ts_end and f_start >= ts_end:
                    continue
                if ts_start and f_end <= ts_start:
                    continue
                keys.append(f"data/{self._validator_uid}/{book_id}/intervals/{fname}")

        return keys

    def _get_max_data_timestamp(self) -> int:
        """Maximum sim timestamp (ns) we've observed.

        In-memory `_max_timestamp_ns` is the source of truth — it's updated by
        every tick processed in `_process_tick`. Only when we have nothing
        in-memory (fresh server, no state yet) do we consult S3 / filesystem
        for restart resilience. Stale data in S3 must not poison the live
        timestamp once state starts flowing.
        """
        if self._max_timestamp_ns > 0:
            return self._max_timestamp_ns

        if not self._all_books:
            self._all_books = self._discover_books()
        if not self._all_books:
            return 0

        max_ns = 0

        if self.validator_store is not None:
            for book_id in self._all_books[:3]:
                try:
                    filenames = self.validator_store.list_data(
                        self._validator_uid, book_id=int(book_id)
                    )
                    for fname in filenames:
                        parts = Path(fname).stem.split("-")
                        if len(parts) == 2:
                            try:
                                end_ns = _tag_to_ns(parts[1])
                                max_ns = max(max_ns, end_ns)
                            except ValueError:
                                pass
                except Exception:
                    continue
        else:
            data_path = Path(self.val_data_path)
            for book_id in self._all_books[:3]:
                book_dir = data_path / book_id / "intervals"
                if not book_dir.is_dir():
                    continue
                for f in book_dir.glob("*.parquet"):
                    parts = f.stem.split("-")
                    if len(parts) == 2:
                        try:
                            end_ns = _tag_to_ns(parts[1])
                            max_ns = max(max_ns, end_ns)
                        except ValueError:
                            pass

        return max_ns

    def _collect_val_files(self) -> list[Path]:
        """Collect all parquet files from val books.

        Uses the global val book split. Reads all available data for val books
        from S3 or local filesystem.
        """
        val_books = self._get_val_books()
        if not val_books:
            return []

        files = []
        for book_id in val_books:
            book_files = self._get_book_files(book_id)
            files.extend(book_files)

        logger.info(
            "Val files: %d files from %d val books",
            len(files),
            len(val_books),
        )
        return files

    # ------------------------------------------------------------------
    # S3 data loading
    # ------------------------------------------------------------------

    def _get_s3_cache_dir(self) -> Path:
        """Get or create the local cache directory for S3 parquets."""
        if self._s3_cache_dir is None:
            import tempfile

            self._s3_cache_dir = Path(tempfile.mkdtemp(prefix="gentrx_s3_cache_"))
            logger.info("S3 data cache: %s", self._s3_cache_dir)
        return self._s3_cache_dir

    def _fetch_s3_book_files(
        self,
        book_id: str,
        ts_start: int = 0,
        ts_end: int = 0,
    ) -> list[Path]:
        """List and download parquets for a book from S3, with timestamp filtering.

        Files are cached locally — only downloaded once per session.
        Returns list of local Path objects ready for OrderDataset.
        """
        if self.validator_store is None:
            return []

        try:
            filenames = self.validator_store.list_data(
                self._validator_uid, book_id=int(book_id)
            )
        except Exception as exc:
            logger.debug("S3 list_data failed for book %s: %s", book_id, exc)
            return []

        if not filenames:
            return []

        # Build Path-like objects for timestamp filtering
        # _filter_by_timestamp works on Path objects (uses .stem)
        pseudo_paths = [Path(f) for f in filenames]
        if ts_start or ts_end:
            pseudo_paths = _filter_by_timestamp(pseudo_paths, ts_start, ts_end)

        # Download each file to local cache
        cache_dir = self._get_s3_cache_dir()
        local_files = []
        for pp in pseudo_paths:
            fname = pp.name
            cache_key = f"{book_id}/{fname}"

            if cache_key in self._s3_cached_files:
                local_files.append(self._s3_cached_files[cache_key])
                continue

            local_path = cache_dir / str(book_id) / "intervals" / fname
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = self.validator_store.get_data(
                    self._validator_uid, book_id=int(book_id), filename=fname
                )
                local_path.write_bytes(data)
                self._s3_cached_files[cache_key] = local_path
                local_files.append(local_path)
            except Exception as exc:
                logger.warning(
                    "S3 download failed: book=%s file=%s: %s", book_id, fname, exc
                )

        return local_files

    def _get_book_files(
        self,
        book_id: str,
        ts_start: int = 0,
        ts_end: int = 0,
    ) -> list[Path]:
        """Get parquet files for a book — from S3 if data_store is set, else filesystem."""
        if self.validator_store is not None:
            return self._fetch_s3_book_files(book_id, ts_start, ts_end)

        # Filesystem fallback
        data_path = Path(self.val_data_path)
        book_dir = data_path / str(book_id) / "intervals"
        if not book_dir.is_dir():
            return []
        book_files = sorted(book_dir.glob("*.parquet"))
        if ts_start or ts_end:
            book_files = _filter_by_timestamp(book_files, ts_start, ts_end)
        return book_files

    # ------------------------------------------------------------------
    # DataLoader builders (use S3 or filesystem transparently)
    # ------------------------------------------------------------------

    def _build_val_loader_for_ranges(
        self,
        ts_ranges: list[tuple[int, int]],
        tokenizer,
        device: str,
    ):
        """Build a DataLoader from val books for specific timestamp ranges.

        For each time range that miners trained on, load the held-out val books
        that overlap. Tests whether the aggregated gradient generalizes to
        unseen books in the same time periods.
        """
        from GenTRX.src.dataloader import OrderDataset
        from torch.utils.data import DataLoader

        if not ts_ranges:
            return None

        val_books = self._get_val_books()
        if not val_books:
            return None

        files = []
        for ts_start, ts_end in ts_ranges:
            for book_id in val_books:
                book_files = self._get_book_files(book_id, ts_start, ts_end)
                files.extend(book_files)

        # Deduplicate (multiple ranges may overlap the same files)
        seen = set()
        unique_files = []
        for f in files:
            key = str(f)
            if key not in seen:
                seen.add(key)
                unique_files.append(f)

        if not unique_files:
            logger.info("  No val files found for %d time ranges", len(ts_ranges))
            return None

        logger.info(
            "  Val loader: %d files from %d ranges, %d val books",
            len(unique_files),
            len(ts_ranges),
            len(val_books),
        )
        ds = OrderDataset(unique_files, seq_len=256, tokenizer=tokenizer, max_cached=2)
        return DataLoader(ds, batch_size=32, shuffle=False)

    @property
    def version(self) -> int:
        return self._version

    def _log_event(self, event: dict) -> None:
        """Append a JSON event to the aggregation log.

        Every event is stamped with `sim_id` (from the simulator's config) and
        `sim_epoch` (incremented when sim time resets). Lets the dashboard
        separate runs that would otherwise blend together because version +
        round both reset on each gradient server restart.

        sim_id is None until the first state packet arrives — events emitted
        before that (e.g. the run_start marker) carry sim_id=null and are
        bucketed together by the dashboard.

        Also mirrors to wandb when `--wandb-project` is set (or
        `WANDB_PROJECT` env var). Soft no-op otherwise.
        """
        event["timestamp"] = time.time()
        event.setdefault("sim_id", self._sim_id)
        event.setdefault("sim_epoch", self._sim_epoch)
        with open(self.log_path, "a") as f:
            f.write(_json.dumps(event) + "\n")
        # Mirror aggregation events into the snapshot dict + counters so
        # the validator-side Prometheus collector can read them off the
        # next /gentrx/scores payload.
        if event.get("type") == "aggregation":
            self._last_aggregation.update({
                k: v for k, v in event.items()
                if k in ("round", "n_scored", "n_accepted", "loss_before",
                         "loss_after", "rolled_back", "sibling_only", "version")
            })
            if event.get("n_accepted", 0) > 0 and not event.get("rolled_back"):
                self._rounds_aggregated_total += 1
            if event.get("rolled_back"):
                self._rollbacks_total += 1
        try:
            from GenTRX.src import wandb_ops
            wandb_ops.log_event(event)
        except Exception:
            pass  # logging must never break the caller

    def _sync_from_uid0(self) -> bool:
        """Download the latest checkpoint from uid 0's chain-committed bucket.

        Called at startup (bootstrap) and at the start of each aggregation round
        (model sync).  Keeps sibling validators and their scoring aligned with
        the canonical model published by the aggregator.

        Returns True if a new checkpoint was downloaded, False otherwise.
        No-op if this server IS the aggregator, chain is not set, uid 0 has
        no bucket commitment, or the latest version is not newer than local.
        """
        if self.is_aggregator:
            return False  # we ARE uid 0 — we publish, not pull
        try:
            from GenTRX.src.gradient_store import GradientStore
            from GenTRX.src.chain import BucketInfo

            bucket_info = None
            if self._chain is not None:
                bucket_info = self._chain.get_bucket(0)

            # Fallback: GENTRX_AGGREGATOR_S3_* env vars. Operators running a
            # sibling validator set these to the aggregator's published read
            # credentials so checkpoint sync keeps working when the chain
            # commitment has not propagated (or chain RPC is flaky).
            if bucket_info is None:
                bucket_info = BucketInfo.from_aggregator_env()
                if bucket_info is not None:
                    logger.info(
                        "uid-0 chain bucket unavailable; using "
                        "GENTRX_AGGREGATOR_S3_* env fallback"
                    )

            if bucket_info is None:
                logger.debug(
                    "No uid-0 bucket available (chain empty and no "
                    "GENTRX_AGGREGATOR_S3_* fallback); skipping sync"
                )
                return False

            store = GradientStore(
                endpoint_url=bucket_info.endpoint_url,
                bucket=bucket_info.bucket_name,
                access_key=bucket_info.access_key_id,
                secret_key=bucket_info.secret_access_key,
                region=bucket_info.region,
                prefix=self._bucket_prefix,
            )
            version = store.get_latest_version(0)
            if version <= 0:
                logger.debug("uid-0 bucket has no checkpoint yet")
                return False
            if version <= self._version:
                return False  # already current

            logger.info(
                "Syncing from uid-0 checkpoint v%d → v%d at %s/%s",
                self._version, version,
                bucket_info.endpoint_url, bucket_info.bucket_name,
            )
            ckpt_bytes = store.get_checkpoint(0, version)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_bytes(ckpt_bytes)
            self._version = version
            logger.info("Model sync complete: checkpoint v%d written to %s", version, self.checkpoint_path)
            return True
        except Exception as exc:
            logger.warning("uid-0 sync failed: %s", exc)
            return False

    def start(self) -> None:
        """Start the background aggregation thread.

        On first start, uploads the initial checkpoint to S3 so agents can
        bootstrap without needing a local checkpoint file.
        """
        if self._running:
            return

        # Ensure a checkpoint exists.
        # Priority: (1) local file, (2) uid-0 bucket via chain, (3) fresh model.
        if not self.checkpoint_path.exists():
            self._sync_from_uid0()

        if not self.checkpoint_path.exists():
            logger.info(
                "No checkpoint at %s — initializing fresh model", self.checkpoint_path
            )
            try:
                import torch
                from dataclasses import asdict
                from GenTRX.src.model import OrderModel, ModelConfig
                from GenTRX.src.tokenizer import TokenizerConfig

                model_cfg = ModelConfig()
                tokenizer_cfg = TokenizerConfig()
                model = OrderModel(model_cfg)
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "tokenizer_config": asdict(tokenizer_cfg),
                        "epoch": 0,
                        "step": 0,
                        "loss": float("inf"),
                    },
                    self.checkpoint_path,
                )
                self._fresh_start = True
                logger.info(
                    "Fresh model checkpoint saved: %s (warmup=%d rounds)",
                    self.checkpoint_path,
                    self.warmup_rounds,
                )
            except Exception as exc:
                logger.error("Failed to initialize fresh model: %s", exc)

        # Upload initial checkpoint to aggregator bucket so agents can bootstrap.
        # On restart, scan for the highest existing version rather than always
        # resetting to v1 (which prune would immediately delete, leaving
        # latest.json pointing at a non-existent file).
        if self.validator_store is not None and self.checkpoint_path.exists():
            try:
                existing_version = self.validator_store.get_latest_existing_version(self._validator_uid)
                if existing_version > 0:
                    # Bucket already has checkpoints — resume from the highest.
                    # get_latest_existing_version() also repairs latest.json if stale.
                    self._version = existing_version
                    logger.info(
                        "Resumed from existing checkpoint v%d in aggregator bucket",
                        self._version,
                    )
                else:
                    # Fresh bucket — upload local checkpoint as v1.
                    data = self.checkpoint_path.read_bytes()
                    self._version = 1
                    self.validator_store.put_checkpoint(self._validator_uid, self._version, data)
                    logger.info(
                        "Initial checkpoint uploaded to aggregator (v%d, %.1f MB)",
                        self._version,
                        len(data) / 1e6,
                    )
                    self._prune_checkpoints()
            except Exception as exc:
                logger.error("Failed to upload initial checkpoint: %s", exc)

        # Restore in-progress parquet buffer from local staging file so a
        # restart continues filling the current window, not a fresh one.
        self._restore_pending_rows()
        # Restore written-parquet registry from S3 so a restart doesn't
        # force a full 5-min re-accumulation before assignments can be created.
        if self.validator_store is not None:
            self._restore_written_parquets()

        self._running = True
        self._thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._thread.start()
        logger.info("Aggregation thread started (interval=%ds)", self.interval)
        # Drop a synthetic event so dashboards see the gradient-server-start
        # boundary even before any aggregation has happened. sim_id is null
        # at this point — it gets attached as soon as the first state packet
        # arrives, and a separate `sim_bind` event will mark the moment.
        self._log_event({"type": "server_start", "interval": self.interval})

    def _save_pending_rows(self) -> None:
        """Atomically persist the in-progress parquet buffer to local disk.

        Called from the aggregation loop every ~60s. On restart,
        _restore_pending_rows() loads this file so the current parquet
        window continues accumulating rather than starting over.
        """
        if not self._pending_rows:
            return
        try:
            import msgpack as _mp
            payload = _mp.packb(
                {
                    "pending_rows": {str(bid): rows for bid, rows in self._pending_rows.items()},
                    "interval_start": {
                        str(bid): ts for bid, ts in self._pending_interval_start.items()
                    },
                    "last_ts": {str(bid): ts for bid, ts in self._last_ts.items()},
                    "max_ts": self._max_timestamp_ns,
                    # Lets a process restart tell whether the restored buffer
                    # belongs to the current sim before consuming it.
                    "sim_id": self._sim_id,
                },
                use_bin_type=True,
            )
            tmp = self._pending_staging_path.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.rename(self._pending_staging_path)
        except Exception as exc:
            logger.debug("pending rows persist failed: %s", exc)

    def _restore_pending_rows(self) -> None:
        """Restore in-progress parquet buffer from local disk after restart.

        Skipped if the staging file doesn't exist or is unreadable. The
        sim-id check in _process_tick (against `_restored_sim_id`) plus
        the backwards-time check (against `_max_timestamp_ns`) wipe the
        restored buffer if the sim was swapped between save and restore.
        """
        if not self._pending_staging_path.exists():
            return
        try:
            import msgpack as _mp
            from GenTRX.src.orderbook import MatchingEngine

            raw = _mp.unpackb(
                self._pending_staging_path.read_bytes(), raw=False, strict_map_key=False
            )
            pending = raw.get("pending_rows", {})
            interval_start = raw.get("interval_start", {})
            last_ts = raw.get("last_ts", {})
            max_ts = int(raw.get("max_ts", 0))
            self._restored_sim_id = raw.get("sim_id")

            for bid_str, rows in pending.items():
                bid = int(bid_str)
                self._pending_rows[bid] = rows
                self._pending_interval_start[bid] = int(interval_start.get(bid_str, 0))
                self._last_ts[bid] = int(last_ts.get(bid_str, 0))
                # Initialise engine placeholder so _process_tick's
                # "if book_id not in self._engines" branch does NOT fire
                # and overwrite our restored pending_rows with [].
                if bid not in self._engines:
                    self._engines[bid] = MatchingEngine()
                    self._order_sides[bid] = {}
                    self._session_open_mid[bid] = None

            if max_ts > self._max_timestamp_ns:
                self._max_timestamp_ns = max_ts

            total_rows = sum(len(r) for r in self._pending_rows.values())
            logger.info(
                "Restored pending buffer: %d books, %d rows, max_ts=%d",
                len(self._pending_rows), total_rows, max_ts,
            )
        except Exception as exc:
            logger.warning("pending rows restore failed: %s", exc)

    def _restore_written_parquets(self) -> None:
        """Scan S3 and rebuild _written_parquets from existing data/ parquets.

        Called once at startup so a restart doesn't force a fresh 5-min
        accumulation before the first assignment can be created.
        """
        try:
            book_ids = self.validator_store.list_books(self._validator_uid)
        except Exception as exc:
            logger.warning("Could not list S3 books for parquet restore: %s", exc)
            return

        restored = 0
        for bid in book_ids:
            try:
                fnames = self.validator_store.list_data(self._validator_uid, bid)
            except Exception as exc:
                logger.debug("list_data failed for book %d: %s", bid, exc)
                continue
            for fname in fnames:
                if not fname.endswith(".parquet"):
                    continue
                stem = fname[: -len(".parquet")]
                parts = stem.split("-")
                if len(parts) == 2:
                    try:
                        ts_min = _tag_to_ns(parts[0])
                        ts_max = _tag_to_ns(parts[1])
                    except Exception:
                        continue
                    if bid not in self._written_parquets:
                        self._written_parquets[bid] = []
                    self._written_parquets[bid].append((fname, ts_min, ts_max))
                    if ts_max > self._max_timestamp_ns:
                        self._max_timestamp_ns = ts_max
                    restored += 1

        if restored:
            logger.info(
                "Restored %d parquet(s) across %d book(s) from S3",
                restored, len(self._written_parquets),
            )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _aggregation_loop(self) -> None:
        """Event-driven aggregation loop.

        Round advancement is driven by the validator via POST /gentrx/round.
        This loop only handles the per-round work:
          1. Drain rounds queued for aggregation by POST /round
          2. Create assignments if none exist yet (needs data first)
          3. Check data readiness for PENDING assignments
          4. Collect gradients from miner buckets for DELIVERED assignments
          5. Aggregate when the current round is complete (all in or timer expired)
          6. Create next round's assignments after aggregation

        Polls every 5s — cheap when nothing is happening.
        """
        while self._running:
            try:
                # Sim transition cleanup: fires on the first loop tick after
                # an ESE marker (or a heuristic sim-restart) set the flag.
                # Wipes stale parquets from the validator bucket's data/
                # prefix; checkpoints and proposals are intentionally left
                # alone. Runs before the rest of the loop so a half-done
                # cleanup never overlaps a fresh assignment creation.
                if self._data_cleanup_pending:
                    self._run_data_cleanup()

                # Flush any book whose interval has elapsed by wall-clock sim
                # time even if no new event has arrived for it.  Without this,
                # sparse books hold _max_timestamp_ns − min(interval_starts) at
                # ≥100% indefinitely while active books continue to advance the
                # global clock.
                if self._max_timestamp_ns > 0:
                    for _bid in list(self._pending_rows):
                        _start = self._pending_interval_start.get(_bid, 0)
                        if (
                            _start > 0
                            and self._pending_rows[_bid]
                            and (self._max_timestamp_ns - _start) >= self._parquet_interval_ns
                        ):
                            self._flush_book_parquet(_bid)

                # Drain any rounds queued by POST /gentrx/round. The HTTP
                # handler installs new assignments and advances _agg_round
                # synchronously, but leaves aggregation of the closing round
                # to us so the event loop never blocks on torch work.
                if self._pending_aggregation_rounds:
                    # Give late-arriving gradients a final collection pass
                    # BEFORE aggregating. Without this, a miner whose
                    # upload lands after POST /round for the next round
                    # gets scored 0 even though the gradient is already
                    # in S3. Collection iterates both _assignments and
                    # _prev_round_assignments, so it catches the entries
                    # POST /round moved into _prev. Eager scoring runs
                    # inline on each newly-collected gradient.
                    self._collect_round_gradients()

                    # Drain a round when either:
                    #   1. every miner the validator delivered an
                    #      assignment to has been scored already (eager
                    #      path completed during the round) — score now,
                    #      no waiting; or
                    #   2. the post-close grace window has elapsed —
                    #      score whoever is in, leave stragglers at 0.
                    # The validator's 25-block round IS the training-time
                    # budget; the grace is just slack for uploads landing
                    # right after POST /round.
                    _now = time.time()
                    ready = {
                        rnd for rnd in self._pending_aggregation_rounds
                        if self._is_round_ready_for_aggregation(rnd, _now)
                    }

                    # Snapshot which rounds we're aggregating NOW. A
                    # concurrent POST /round can add new rounds to
                    # _pending_aggregation_rounds and new entries to
                    # _prev_round_assignments while we're working — we
                    # must not touch anything outside this snapshot.
                    draining = ready
                    for rnd in sorted(draining):
                        saved_round = self._agg_round
                        self._agg_round = rnd
                        try:
                            self._aggregate_round()
                        finally:
                            self._agg_round = saved_round
                        self._pending_aggregation_rounds.discard(rnd)
                        self._pending_aggregation_at.pop(rnd, None)
                    # Drop _prev_round_assignments entries for rounds we
                    # just aggregated only — leave any entries belonging to
                    # a later round that POST /round added concurrently.
                    for uid, a in list(self._prev_round_assignments.items()):
                        if a.get("round", -1) in draining:
                            self._prev_round_assignments.pop(uid, None)

                # State arrives over HTTP (POST /state) and is processed in
                # the FastAPI handler — this loop only manages the assignment
                # state machine + gradient collection.

                # Step 1: create assignments if none exist for current round
                has_current = any(
                    a.get("round") == self._agg_round
                    for a in self._assignments.values()
                )
                if not has_current:
                    self._create_round_assignments()

                # Step 2: check if PENDING assignments now have data
                self._check_data_readiness()

                # Step 3: collect gradients for DELIVERED assignments
                self._collect_round_gradients()

                # Periodic status log: every ~block (block_time_s seconds).
                _now = time.monotonic()
                if _now - self._last_round_log_ts >= self.block_time_s:
                    cur_assignments = [
                        a for a in self._assignments.values()
                        if a.get("round") == self._agg_round
                    ]
                    pending = sum(
                        1 for a in cur_assignments if a.get("_state") == "PENDING"
                    )
                    data_ready = sum(
                        1 for a in cur_assignments if a.get("_state") == "DATA_READY"
                    )
                    delivered = sum(
                        1 for a in cur_assignments
                        if a.get("_state") in ("DELIVERED", "GRADIENT_IN")
                    )
                    gradient_in = sum(
                        1 for a in cur_assignments if a.get("_state") == "GRADIENT_IN"
                    )
                    total = len(cur_assignments)
                    if delivered > 0:
                        # Active round: show countdown to next round
                        round_s = self._round_estimate_s()
                        delivered_times = [
                            a["_delivered_at"] for a in cur_assignments
                            if a.get("_delivered_at") and a.get("_state") in ("DELIVERED", "GRADIENT_IN")
                        ]
                        timing_suffix = ""
                        if self._last_push_block is not None and self.blocks_per_round > 0 and delivered_times:
                            elapsed_blocks = round(
                                (time.time() - min(delivered_times)) / self.block_time_s
                            )
                            cur_block = self._last_push_block + elapsed_blocks
                            next_block = (self._agg_round + 1) * self.blocks_per_round
                            secs = max(0.0, (next_block - cur_block) * self.block_time_s)
                            timing_suffix = (
                                f"; next round at block ~{next_block} (~{secs:.0f}s)"
                            )
                        elif delivered_times and round_s > 0:
                            elapsed_s = time.time() - min(delivered_times)
                            remaining_s = max(0.0, round_s - elapsed_s)
                            timing_suffix = f"; ~{remaining_s:.0f}s until round closes"
                        logger.info(
                            f"[GTX] round={self._agg_round}: {gradient_in}/{delivered} gradients in "
                            f"({delivered}/{total} delivered){timing_suffix}"
                        )
                    else:
                        # Waiting for validator to push round (data may still be accumulating)
                        data_age_s = ""
                        if self._written_parquets:
                            data_age_s = ", data ready"
                            if self._pending_rows and self._max_timestamp_ns > 0:
                                # min: pick the most advanced book (smallest interval
                                # start = largest elapsed) so the display shows how
                                # close we are to the NEXT parquet flush.
                                min_start = min(
                                    (self._pending_interval_start.get(bid, 0) for bid in self._pending_rows),
                                    default=0,
                                )
                                if min_start > 0:
                                    elapsed_ns = self._max_timestamp_ns - min_start
                                    pct = min(100, int(100 * elapsed_ns / self._parquet_interval_ns))
                                    data_age_s += f", next window {pct}%"
                        elif self._max_timestamp_ns > 0:
                            min_start = min(
                                (self._pending_interval_start.get(bid, 0) for bid in self._pending_rows),
                                default=0,
                            )
                            if min_start > 0:
                                elapsed_ns = self._max_timestamp_ns - min_start
                                pct = min(100, int(100 * elapsed_ns / self._parquet_interval_ns))
                                data_age_s = f", accumulating data ({pct}% to first window)"
                        states = f"pending={pending} data_ready={data_ready}"
                        logger.info(
                            f"[GTX] round={self._agg_round}: waiting for validator push "
                            f"({states}{data_age_s})"
                        )
                    self._last_round_log_ts = _now

                # Persist pending rows every loop iteration (~5s) so restarts
                # lose at most one loop tick of accumulation.
                self._save_pending_rows()
                self._last_pending_save_ts = _now

                # Step 4: check if current round is complete → aggregate
                # Fallback path when the validator stops pushing POST /round
                # (e.g. disconnect): the loop closes the round on its
                # heartbeat-loss timer so scoring doesn't stall.
                if self._round_complete():
                    self._aggregate_round()
                    self._agg_round += 1
                    # Siblings: sync canonical model from uid-0 before the new
                    # round so scoring uses the latest checkpoint. Once per round.
                    if not self.is_aggregator:
                        self._sync_from_uid0()
                    self._create_round_assignments()

            except Exception as exc:
                logger.error("Aggregation loop error: %s", exc)

            time.sleep(self._loop_sleep_s)  # short poll — most iterations are cheap

    def _refresh_miner_buckets(self) -> None:
        """Pull miner bucket commitments from chain, cooldown-rate-limited.

        Re-queries when the cache is non-empty (catches new registrations) at
        most once per `_miner_buckets_refresh_s`. Importantly also rate-limits
        the empty-cache case — a chain with no commitments would otherwise
        cause a query every loop iteration.
        """
        if self._chain is None:
            return
        now = time.time()
        if now - self._miner_buckets_queried_at < self._miner_buckets_refresh_s:
            return
        try:
            import asyncio
            t_start = time.time()
            loop = asyncio.new_event_loop()
            try:
                fresh = loop.run_until_complete(self._chain.get_miner_buckets())
            finally:
                loop.close()
            t_query = time.time() - t_start
            if fresh:
                self._miner_buckets = fresh
            elif not self._miner_buckets:
                self._miner_buckets = {}
            # Static buckets (benchmark miners not on-chain) always win.
            if self._static_buckets:
                self._miner_buckets.update(self._static_buckets)
            self._miner_buckets_queried_at = now
            logger.info(
                "[GTX] miner_buckets_refresh n=%d (static=%d) t=%.2fs",
                len(self._miner_buckets), len(self._static_buckets), t_query,
            )
            self._log_event({
                "type": "miner_buckets_refresh",
                "n_buckets": len(self._miner_buckets),
                "t_query_s": t_query,
            })
        except Exception as exc:
            logger.debug("Failed to refresh miner buckets: %s", exc)

    def _collect_round_gradients(self) -> None:
        """Check miner S3 buckets for gradients from DELIVERED assignments.

        Iterates both the live current-round (_assignments) and the preserved
        closing-round (_prev_round_assignments). A gradient uploaded for the
        closing round is still collectable until that round's aggregation
        finishes and clears _prev_round_assignments.
        """
        # Snapshot before iterating — push_round (HTTP handler) mutates these
        # dicts concurrently from the uvicorn event loop.
        for source in (dict(self._assignments), dict(self._prev_round_assignments)):
            for uid, a in source.items():
                if a.get("_state") != "DELIVERED":
                    continue
                if a.get("_gradient_data") is not None:
                    continue  # already collected

                # Try to read gradient from miner's bucket
                grad_data = self._try_read_miner_gradient(uid, a["round"])
                if grad_data is not None:
                    a["_gradient_data"] = grad_data
                    a["_state"] = "GRADIENT_IN"
                    logger.info(
                        "[GTX] round=%d gradient received: uid=%d (%.1f KB)",
                        a["round"], uid, len(grad_data) / 1024,
                    )
                    self._log_event({
                        "type": "gradient_received",
                        "round": a["round"],
                        "miner": uid,
                        "bytes": len(grad_data),
                    })
                    # Score eagerly so the round can drain the moment all
                    # expected gradients are in. Failure here is fine —
                    # the drain pass re-attempts via _score_round.
                    self._score_eagerly(uid, a, grad_data)

    def _try_read_miner_gradient(self, uid: int, round_id: int) -> bytes | None:
        """Try to read a gradient from a miner's S3 bucket. Returns None if not found."""
        dedup_key = f"miner_{uid}/round_{round_id}"
        if dedup_key in self._processed_grad_keys:
            return None

        # Per-miner buckets (production + localnet)
        if self._chain is not None:
            self._refresh_miner_buckets()
            bucket_info = self._miner_buckets.get(uid)
            if bucket_info is None:
                return None

            try:
                import boto3
                from botocore.config import Config as BotoConfig
                from botocore.exceptions import ClientError

                client = boto3.client(
                    "s3",
                    endpoint_url=bucket_info.endpoint_url,
                    aws_access_key_id=bucket_info.access_key_id,
                    aws_secret_access_key=bucket_info.secret_access_key,
                    region_name=getattr(bucket_info, "region", "auto"),
                    config=BotoConfig(connect_timeout=5, read_timeout=10),
                )
                key = f"{self._bucket_prefix}gradients/{uid}/{round_id:08d}.grad"
                try:
                    t_start = time.time()
                    resp = client.get_object(Bucket=bucket_info.bucket_name, Key=key)
                    data = resp["Body"].read()
                    t_get = time.time() - t_start
                    self._processed_grad_keys.add(dedup_key)
                    logger.info(
                        "[GTX] gradient_get uid=%d round=%d bytes=%d t=%.2fs",
                        uid, round_id, len(data), t_get,
                    )
                    return data
                except ClientError:
                    return None
            except Exception:
                return None

        return None

    def _is_round_ready_for_aggregation(self, rnd: int, now: float) -> bool:
        """Decide if a queued (POST /round-closed) round can drain now.

        Two triggers:
          1. Early complete: no DELIVERED or GRADIENT_IN remain for this
             round. Either every delivered miner uploaded and was already
             eagerly scored, or no miner was ever delivered to (an empty
             round). Nothing to wait for.
          2. Grace expired: at least `round_grace_s` seconds have elapsed
             since POST /round arrived for this round. Score whoever is
             in; leave stragglers at 0.
        """
        queued_at = self._pending_aggregation_at.get(rnd)
        if queued_at is not None and (now - queued_at) >= self._round_grace_s:
            return True

        for source in (self._assignments, self._prev_round_assignments):
            for a in source.values():
                if a.get("round") != rnd:
                    continue
                if a.get("_state") in ("DELIVERED", "GRADIENT_IN"):
                    return False
        return True

    def _round_complete(self) -> bool:
        """Check if the current round is ready for aggregation.

        A round is complete when all DELIVERED miners have submitted
        their gradients (GRADIENT_IN), or the heartbeat-loss fallback
        fires.

        The fallback exists only because the validator normally closes
        rounds via `POST /gentrx/round`. If the validator stops pushing
        (disconnect, restart, partition), the server force-closes the
        round at `oldest_delivery + _round_estimate_s() + _round_grace_s`
        so scoring does not stall indefinitely.
        """
        delivered = [
            a for a in self._assignments.values()
            if a.get("_state") in ("DELIVERED", "GRADIENT_IN")
            and a.get("round") == self._agg_round
        ]
        if not delivered:
            return False

        all_in = all(a["_state"] == "GRADIENT_IN" for a in delivered)
        if all_in:
            logger.info(
                "[GTX] round=%d complete: all %d gradients received",
                self._agg_round, len(delivered),
            )
            return True

        # Heartbeat-loss fallback: oldest delivery + estimate + grace
        delivered_times = [
            a["_delivered_at"] for a in delivered if a.get("_delivered_at")
        ]
        if not delivered_times:
            return False

        oldest_delivery = min(delivered_times)
        expected_round_s = self._round_estimate_s()
        elapsed = time.time() - oldest_delivery

        if elapsed > expected_round_s + self._round_grace_s:
            n_in = sum(1 for a in delivered if a["_state"] == "GRADIENT_IN")
            n_missing = len(delivered) - n_in
            logger.info(
                "[GTX] round=%d heartbeat-loss force-close (%.0fs > %.0fs): %d/%d gradients received, %d missing",
                self._agg_round, elapsed, expected_round_s + self._round_grace_s,
                n_in, len(delivered), n_missing,
            )
            return True

        return False

    def _aggregate_round(self) -> None:
        """Score and aggregate gradients for the current round.

        Collects all GRADIENT_IN assignments, scores them, aggregates
        accepted gradients into the model, and publishes a new checkpoint.
        Miners that didn't submit get score=0 for this round.

        Looks in BOTH _assignments (live current round) and
        _prev_round_assignments (closing round preserved across the last
        POST /round). Filters by `round == self._agg_round` so we only pick
        up entries that belong to the round we're aggregating.
        """
        round_assignments = []
        for source in (dict(self._assignments), dict(self._prev_round_assignments)):
            for uid, a in source.items():
                if (
                    a.get("round") == self._agg_round
                    and a.get("_state") in ("GRADIENT_IN", "DELIVERED", "SCORED")
                ):
                    round_assignments.append((uid, a))

        # Build pending list for the existing _drain_and_aggregate scoring logic.
        # Eager-scored entries (state=SCORED, _comp set) are passed through
        # so _score_round's idempotent path returns the cached score+comp
        # without re-running GPU work.
        pending = []
        for uid, a in round_assignments:
            grad_data = a.get("_gradient_data")
            state = a.get("_state")
            if grad_data is not None and state in ("GRADIENT_IN", "SCORED"):
                pending.append((uid, self._agg_round, a, grad_data))
            elif state == "DELIVERED":
                # No gradient submitted by round close, score 0
                a["_state"] = "SCORED"
                a["_score"] = 0.0
                logger.info("Miner %d: no gradient for round %d (score=0)", uid, self._agg_round)

        if not pending:
            logger.info("[GTX] round=%d: no gradients to aggregate", self._agg_round)
            self._log_event({
                "type": "aggregation",
                "round": self._agg_round,
                "n_scored": 0,
                "n_accepted": 0,
                "version": self._version,
            })
            return

        logger.info("[GTX] round=%d: aggregating %d gradients", self._agg_round, len(pending))

        # Reuse existing scoring + aggregation logic
        self._score_and_aggregate(pending)

        # Mark all as SCORED
        for uid, a in round_assignments:
            a["_state"] = "SCORED"

    def _score_and_aggregate(self, pending: list) -> None:
        """Orchestrator: reuse the eager-scoring cache when present, else load.

        Most rounds will arrive here with the model already loaded (the
        first eager score during the round triggered _get_scoring_cache).
        Stragglers go through _score_round, which is a no-op for already-
        scored gradients. Aggregation then runs once on the assembled
        scored list, and the cache is dropped so the next round loads
        the freshly-published checkpoint.
        """
        t_total_start = time.time()

        t_load_start = time.time()
        cache = self._get_scoring_cache(self._agg_round)
        if cache is None:
            logger.warning(
                "[GTX] aggregate_round=%d: scoring model unavailable, skipping",
                self._agg_round,
            )
            return
        model = cache["model"]
        model_cfg = cache["model_cfg"]
        tokenizer_cfg = cache["tokenizer_cfg"]
        tokenizer = cache["tokenizer"]
        device = cache["device"]
        t_load = time.time() - t_load_start

        t_score_start = time.time()
        scored = self._score_round(pending, model, tokenizer, device)
        t_score = time.time() - t_score_start

        t_agg_start = time.time()
        self._aggregate_accepted(scored, model, model_cfg, tokenizer_cfg, tokenizer, device)
        t_agg = time.time() - t_agg_start

        # Drop the cache so the next round picks up the freshly-published
        # checkpoint (aggregation mutated `model` in place, and a new
        # checkpoint was written by _aggregate_accepted).
        self._clear_scoring_cache()

        t_total = time.time() - t_total_start
        logger.info(
            "[GTX] aggregate_round=%d: n_pending=%d t_load=%.2fs t_score=%.2fs t_aggregate=%.2fs t_total=%.2fs",
            self._agg_round, len(pending), t_load, t_score, t_agg, t_total,
        )
        # Stash timings on the aggregator — _aggregate_accepted below will
        # also set loss_before/loss_after/rolled_back on this same dict.
        self._last_aggregation.update({
            "round": self._agg_round,
            "n_pending": len(pending),
            "t_load_s": t_load,
            "t_score_s": t_score,
            "t_aggregate_s": t_agg,
            "t_total_s": t_total,
            "timestamp": time.time(),
        })

    def _get_scoring_cache(self, rnd: int) -> dict | None:
        """Load (or reuse) the scoring model+tokenizer for round `rnd`.

        Cached for the lifetime of the round so eagerly-scored gradients
        don't each pay the checkpoint-load cost. Evicted at aggregation
        boundaries (`_clear_scoring_cache`) because aggregation mutates
        the model in place.
        """
        if (
            self._scoring_cache is not None
            and self._scoring_cache.get("round") == rnd
        ):
            return self._scoring_cache

        try:
            from GenTRX.src.model import OrderModel, ModelConfig
            from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig
            from GenTRX.src.train import load_checkpoint
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with self._lock:
                ckpt = load_checkpoint(str(self.checkpoint_path), device="cpu")
                raw_cfg = ckpt.get("model_config", {})
                model_cfg = ModelConfig(
                    **{
                        k: v
                        for k, v in raw_cfg.items()
                        if k in ModelConfig.__dataclass_fields__
                    }
                )
                tok_dict = ckpt.get("tokenizer_config")
                tokenizer_cfg = (
                    TokenizerConfig.from_dict(tok_dict) if tok_dict else TokenizerConfig()
                )
                model = OrderModel(model_cfg).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
            tokenizer = OrderTokenizer(tokenizer_cfg)
            self._scoring_cache = {
                "round": rnd,
                "model": model,
                "model_cfg": model_cfg,
                "tokenizer_cfg": tokenizer_cfg,
                "tokenizer": tokenizer,
                "device": device,
            }
            return self._scoring_cache
        except Exception as exc:
            logger.warning("Scoring model load failed for round %d: %s", rnd, exc)
            return None

    def _clear_scoring_cache(self) -> None:
        """Drop the cached scoring model. Called after aggregation."""
        self._scoring_cache = None

    def _score_eagerly(self, miner_uid: int, assignment: dict, grad_bytes: bytes) -> bool:
        """Score a freshly-collected gradient on the same loop tick it arrived.

        Returns True if scoring succeeded (assignment now stamped + state →
        SCORED), False otherwise (assignment stays GRADIENT_IN; the drain
        path will retry). Best-effort: failure here is not fatal because
        `_score_round` covers stragglers and any retries at drain time.
        """
        rnd = assignment.get("round")
        if rnd is None:
            return False
        cache = self._get_scoring_cache(rnd)
        if cache is None:
            return False
        result = self._score_one_gradient(
            miner_uid, assignment, grad_bytes,
            cache["model"], cache["tokenizer"], cache["device"],
        )
        if result is None:
            return False
        assignment["_state"] = "SCORED"
        return True

    def _score_one_gradient(
        self, miner_uid: int, assignment: dict, grad_bytes: bytes,
        model, tokenizer, device: str,
    ):
        """Evaluate one miner's gradient. Returns (score, comp) or None on failure.

        Stamps `_score_own`, `_score_held`, `_overfitting`, `_score`, and
        `_comp` on the assignment dict. Idempotent: callers that re-invoke
        with the same assignment get the cached result without re-running
        the GPU work, so we cleanly tolerate the inevitable race where a
        gradient is read on one loop tick and aggregation runs on the next.
        """
        from GenTRX.src.gradient import deserialize
        from GenTRX.src.distributed import evaluate_gradient

        cached_comp = assignment.get("_comp")
        cached_score = assignment.get("_score")
        if cached_comp is not None and cached_score is not None:
            return cached_score, cached_comp

        t_start = time.time()
        try:
            if len(grad_bytes) > self._max_gradient_bytes:
                logger.warning(
                    "  miner %d: gradient too large (%.1f MB)",
                    miner_uid, len(grad_bytes) / 1e6,
                )
                return None
            comp = deserialize(grad_bytes)
            model_param_names = set(name for name, _ in model.named_parameters())
            unknown = set(comp.sparse.keys()) - model_param_names
            if unknown:
                logger.warning(
                    "  miner %d: %d unknown param names", miner_uid, len(unknown)
                )
                return None
            miner_loader = self._build_miner_loader(assignment, tokenizer, device)
            if miner_loader is None:
                logger.warning("  miner %d: no data for assigned books", miner_uid)
                return None

            t_own_start = time.time()
            score_own = evaluate_gradient(
                model, comp, miner_loader, device, self.max_val_batches
            )
            t_own = time.time() - t_own_start

            miner_ranges = [(assignment.get("ts_start", 0), assignment.get("ts_end", 0))]
            held_loader = self._build_val_loader_for_ranges(
                miner_ranges, tokenizer, device
            )
            t_held = 0.0
            if held_loader is not None:
                t_held_start = time.time()
                score_held = evaluate_gradient(
                    model, comp, held_loader, device, self.max_val_batches
                )
                t_held = time.time() - t_held_start
                overfitting = score_own > score_held * self.overfit_ratio
                score = score_held * (self.overfit_penalty if overfitting else 1.0)
            else:
                score_held = None
                overfitting = False
                score = score_own

            assignment["_score_own"] = score_own
            assignment["_score_held"] = score_held
            assignment["_overfitting"] = overfitting
            assignment["_score"] = score
            assignment["_comp"] = comp

            t_total = time.time() - t_start
            logger.info(
                "[GTX] score_miner=%d: score_own=%+.4f score_held=%s%s combined=%+.4f t_own=%.2fs t_held=%.2fs t_total=%.2fs",
                miner_uid, score_own,
                f"{score_held:+.4f}" if score_held is not None else "n/a",
                " [OVERFIT]" if overfitting else "",
                score, t_own, t_held, t_total,
            )
            return score, comp
        except Exception as exc:
            logger.warning(
                "  miner %d: scoring failed: %s", miner_uid, exc, exc_info=True
            )
            return None

    def _score_round(self, pending: list, model, tokenizer, device: str) -> list:
        """Score gradients that haven't been scored yet; collect prior scores.

        Eager scoring (called from _collect_round_gradients) usually fills
        in `_score` and `_comp` on the assignment well before the round
        drains. This pass picks up any stragglers — gradients that landed
        inside the round-grace window — and assembles the full (uid, win,
        score, comp, assignment) tuple list the aggregator expects.
        """
        scored = []
        for miner_uid, window_id, assignment, grad_bytes in pending:
            result = self._score_one_gradient(
                miner_uid, assignment, grad_bytes, model, tokenizer, device,
            )
            if result is None:
                continue
            score, comp = result
            scored.append((miner_uid, window_id, score, comp, assignment))
        return scored

    def _aggregate_accepted(
        self,
        scored: list,
        model,
        model_cfg,
        tokenizer_cfg,
        tokenizer,
        device: str,
    ) -> None:
        """Aggregate accepted gradients and publish result.

        Aggregation-of-aggregations pattern:
          - ALL validators (aggregator + siblings): score miners, aggregate
            accepted gradients into a compressed delta, publish the delta as
            proposals/{block:08d}.grad to their own bucket.  This is proof of
            work — you can't produce a valid delta without actually scoring.
          - Aggregator (uid 0): also fetches proposals from sibling buckets,
            evaluates each against held-out val data, picks the best (or
            averages top proposals), and publishes the canonical checkpoint.

        Sibling validators never publish checkpoints — only proposals + scores.
        """
        from GenTRX.src.gradient import aggregate, decompress, apply_gradient, serialize
        from GenTRX.src.distributed import _eval_loss
        import torch

        threshold = self._effective_min_score

        accepted = [(m, w, s, c, a) for m, w, s, c, a in scored if s > threshold]
        rejected = [(m, w, s, a) for m, w, s, _, a in scored if s <= threshold]

        if self._fresh_start and self._agg_round < self.warmup_rounds:
            logger.info("  Warmup round %d/%d — min_score disabled, rollback disabled", self._agg_round + 1, self.warmup_rounds)

        # Log + deliver scores (all validators)
        for m, w, s, _, a in scored:
            self._log_event({
                "type": "gradient_score",
                "miner": m, "window": w, "score": s,
                "score_own": a.get("_score_own"),
                "score_held": a.get("_score_held"),
                "overfitting": a.get("_overfitting", False),
                "accepted": s > threshold,
                "books": a.get("books", []),
                "version": self._version,
            })

        self._deliver_scores(scored, accepted, rejected, threshold)

        if not accepted:
            logger.info("  No accepted gradients — model unchanged")
            self._log_event({
                "type": "aggregation", "round": self._agg_round,
                "n_scored": len(scored), "n_accepted": 0, "version": self._version,
            })
            return

        # --- Phase 1: Local aggregation (all validators) ---
        # Every validator aggregates its own accepted gradients into a delta.
        local_agg = aggregate([c for _, _, _, c, _ in accepted])
        local_delta = decompress(local_agg)

        # Publish the local delta as a proposal — proof of work.
        # Siblings publish proposals only; aggregator publishes proposals + checkpoint.
        if self.validator_store is not None:
            try:
                proposal_bytes = serialize(local_agg)
                self.validator_store.put_proposal(
                    self._validator_uid, self._agg_round, proposal_bytes
                )
                logger.info(
                    "  Published proposal for round %d (%.1f KB)",
                    self._agg_round, len(proposal_bytes) / 1024,
                )
                self._prune_proposals()
            except Exception as exc:
                logger.warning("  Failed to publish proposal: %s", exc)

        # --- Phase 2: Aggregator evaluates proposals (uid 0 only) ---
        # Fetch proposals from sibling validators, evaluate each against held-out
        # val data, pick the best delta (lowest val loss).  Falls back to local
        # aggregation if no sibling proposals are available.
        trained_ranges = [
            (a.get("ts_start", 0), a.get("ts_end", 0))
            for _, _, _, _, a in accepted if a.get("ts_start")
        ]
        val_loader = self._build_val_loader_for_ranges(trained_ranges, tokenizer, device)
        if val_loader is None:
            val_loader = self._global_val_loader

        best_delta = local_delta
        best_loss = float("inf")
        baseline_loss = 0.0

        if val_loader is not None:
            original_state = {k: v.clone() for k, v in model.state_dict().items()}
            baseline_loss = _eval_loss(model, val_loader, device, self.max_val_batches)

            if self.is_aggregator:
                # Evaluate all proposals (local + sibling) and pick the best
                proposals = self._fetch_validator_proposals(self._agg_round)
                # Always include our own local delta in the candidate set
                candidates = [("local", local_agg)]
                candidates.extend(proposals)

                for label, comp in candidates:
                    model.load_state_dict(original_state)
                    delta = decompress(comp)
                    apply_gradient(model, delta)
                    loss = _eval_loss(model, val_loader, device, self.max_val_batches)
                    logger.info("  Proposal %s: val loss %.4f (baseline %.4f)", label, loss, baseline_loss)
                    if loss < best_loss:
                        best_loss = loss
                        best_delta = delta

                # Restore and apply the winner
                model.load_state_dict(original_state)

            # Apply the chosen delta and check rollback
            apply_gradient(model, best_delta)
            new_loss = _eval_loss(model, val_loader, device, self.max_val_batches) if self.is_aggregator else best_loss

            if self._effective_rollback and new_loss > baseline_loss:
                model.load_state_dict(original_state)
                logger.warning("  Rollback: val %.4f → %.4f (worse)", baseline_loss, new_loss)
                self._log_event({
                    "type": "aggregation", "round": self._agg_round,
                    "n_scored": len(scored), "n_accepted": len(accepted),
                    "loss_before": baseline_loss, "loss_after": new_loss,
                    "rolled_back": True, "version": self._version,
                })
                return
        else:
            new_loss = 0.0
            if not self.is_aggregator:
                # Sibling without val data: we published the proposal, nothing more to do
                logger.info("  Sibling: proposal published, no val check")
                self._log_event({
                    "type": "aggregation", "round": self._agg_round,
                    "n_scored": len(scored), "n_accepted": len(accepted),
                    "version": self._version, "sibling_only": True,
                })
                return
            logger.warning("  No val data — applying local delta without check")
            apply_gradient(model, best_delta)

        # --- Phase 3: Save checkpoint (aggregator publishes canonical) ---
        t_save_start = time.time()
        with self._lock:
            from dataclasses import asdict
            import io as _io

            ckpt_dict = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "tokenizer_config": asdict(tokenizer_cfg),
                "step": self._version + 1,
                "loss": new_loss,
                "epoch": 0,
            }
            tmp_path = self.output_path.with_suffix(".tmp")
            torch.save(ckpt_dict, tmp_path)
            os.rename(tmp_path, self.output_path)
            self._version += 1
            t_local_save = time.time() - t_save_start

            # Only the aggregator (uid 0) publishes the canonical checkpoint.
            t_s3_put = 0.0
            ckpt_bytes = 0
            if self.validator_store is not None and self.is_aggregator:
                try:
                    t_s3_start = time.time()
                    buf = _io.BytesIO()
                    torch.save(ckpt_dict, buf)
                    ckpt_bytes = len(buf.getvalue())
                    self.validator_store.put_checkpoint(
                        self._validator_uid, self._version, buf.getvalue()
                    )
                    t_s3_put = time.time() - t_s3_start
                    self._prune_checkpoints()
                except Exception as exc:
                    logger.error("Failed to upload checkpoint to S3: %s", exc)
            logger.info(
                "[GTX] checkpoint_save v=%d t_local=%.2fs t_s3=%.2fs bytes=%d",
                self._version, t_local_save, t_s3_put, ckpt_bytes,
            )

        self._log_event({
            "type": "aggregation", "round": self._agg_round,
            "n_scored": len(scored), "n_accepted": len(accepted),
            "loss_before": baseline_loss, "loss_after": new_loss,
            "rolled_back": False, "version": self._version,
        })
        logger.info(
            "[GTX] round=%d aggregated %d/%d: loss %.4f → %.4f, new_version=%d",
            self._agg_round, len(accepted), len(scored), baseline_loss, new_loss, self._version,
        )

    def _deliver_scores(self, scored, accepted, rejected, threshold) -> None:
        """Stash scores in memory and publish to validator bucket.

        Memory: served via GET /gentrx/scores for the local validator.
        S3: written to scores/{round:08d}.json so the aggregator (uid 0) and
        other validators can read cross-validator scores via chain discovery.
        """
        if not scored:
            return
        if (self._latest_scores is not None and
                self._agg_round < self._latest_scores.get("round", -1)):
            logger.debug(
                "[GTX] _deliver_scores: dropping out-of-order round %d (latest=%d)",
                self._agg_round, self._latest_scores["round"],
            )
            return
        self._latest_scores = {
            "round": self._agg_round,
            "model_version": self._version,
            "scores": {
                str(m): {
                    "score": s,
                    "score_own": a.get("_score_own"),
                    "score_held": a.get("_score_held"),
                    "overfitting": a.get("_overfitting", False),
                    "accepted": s > threshold,
                    "books": a.get("books", []),
                }
                for m, w, s, _, a in scored
            },
            "n_scored": len(scored),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            # Snapshot of the last aggregation + counters for the validator-side
            # Prometheus collector. Included on the same polling path as scores
            # so no extra HTTP round-trip is needed.
            "aggregation": dict(self._last_aggregation),
            "counters": {
                "rounds_aggregated_total": self._rounds_aggregated_total,
                "rollbacks_total": self._rollbacks_total,
            },
        }
        # Scores are served in-memory via GET /gentrx/scores only.
        # No S3 publish — proposals (not raw scores) are the cross-validator
        # proof-of-work mechanism. See _fetch_validator_proposals().

    def _fetch_validator_proposals(self, round_id: int) -> list[tuple[str, Any]]:
        """Fetch proposals/{round:08d}.grad from sibling validator buckets.

        Returns [(label, CompressedGradient), ...] — one per validator that
        published a proposal for this round.  Filtered by validator_permit
        to prevent miners from injecting fake proposals.

        Only called by the aggregator.
        """
        if self._chain is None:
            return []

        try:
            import asyncio
            from GenTRX.src.gradient_store import GradientStore
            from GenTRX.src.gradient import deserialize

            loop = asyncio.new_event_loop()
            try:
                all_buckets: dict = loop.run_until_complete(
                    self._chain.get_miner_buckets()
                )
            finally:
                loop.close()

            if not all_buckets:
                return []

            metagraph = self._chain.metagraph
            validator_permits = getattr(metagraph, "validator_permit", None)

            proposals = []
            for uid, bucket_info in all_buckets.items():
                if validator_permits is not None:
                    if uid >= len(validator_permits) or not validator_permits[uid]:
                        continue
                try:
                    store = GradientStore(
                        endpoint_url=bucket_info.endpoint_url,
                        bucket=bucket_info.bucket_name,
                        access_key=bucket_info.access_key_id,
                        secret_key=bucket_info.secret_access_key,
                        region=getattr(bucket_info, "region", "auto"),
                        prefix=self._bucket_prefix,
                    )
                    raw = store.get_proposal(uid, round_id)
                    if raw is None:
                        continue
                    comp = deserialize(raw)
                    proposals.append((f"validator-{uid}", comp))
                    logger.debug("  Fetched proposal from validator uid-%d (%.1f KB)", uid, len(raw) / 1024)
                except Exception as exc:
                    logger.debug("Failed to read proposal from uid-%d: %s", uid, exc)

            if proposals:
                logger.info(
                    "  Fetched %d sibling proposals for round %d",
                    len(proposals), round_id,
                )
            return proposals

        except Exception as exc:
            logger.warning("Validator proposal fetch failed: %s", exc)
            return []

    def _build_miner_loader(
        self,
        assignment: dict,
        tokenizer,
        device: str,
    ):
        """Build a DataLoader for a miner's assigned books + timestamp range.

        Reads from S3 when store is set, otherwise from local filesystem.
        """
        from GenTRX.src.dataloader import OrderDataset
        from torch.utils.data import DataLoader

        books = assignment.get("books", [])
        if not books:
            return None

        ts_start = assignment.get("ts_start", 0)
        ts_end = assignment.get("ts_end", 0)
        files = []
        for book_id in books:
            book_files = self._get_book_files(book_id, ts_start, ts_end)
            files.extend(book_files)

        if not files:
            return None

        ds = OrderDataset(files, seq_len=256, tokenizer=tokenizer, max_cached=2)
        return DataLoader(ds, batch_size=32, shuffle=False)

    def receive_state(self, data: bytes) -> None:
        """Receive a msgpack-encoded state tick from the validator.

        Called from the POST /gentrx/state endpoint. Processes the tick
        in-memory and accumulates rows for parquet flushing.
        """
        import msgpack as _msgpack

        try:
            tick = _msgpack.unpackb(data, raw=False, use_list=True, strict_map_key=False)
        except Exception as exc:
            logger.warning("Failed to unpack state: %s", exc)
            return
        self._process_tick(tick)

    def _reset_sim_buffers(self) -> None:
        """Drop every in-memory tick buffer so the next sim starts clean.

        Does not touch the on-S3 bucket — that's scheduled separately via
        `_data_cleanup_pending` and executed by `_run_data_cleanup` on the
        next aggregation-loop tick. Checkpoints, proposals, round counters,
        and model version are all preserved.
        """
        self._pending_rows = {}
        self._pending_interval_start = {}
        self._written_parquets = {}
        self._last_seen_sim_ts = 0
        # Without this, the backwards-time check in _process_tick would
        # re-fire for every new-sim tick whose ts < old_max.
        self._max_timestamp_ns = 0
        self._restored_sim_id = None
        # Engines are per-book; drop them so the next sim replays events
        # against a fresh matching engine instead of stale depth.
        self._engines = {}
        # Invalidate the local pending-rows staging file so a subsequent
        # restart doesn't restore data from the old sim.
        try:
            if self._pending_staging_path.exists():
                self._pending_staging_path.unlink()
        except Exception:
            pass

    def _prune_checkpoints(self) -> None:
        """Trim this validator's checkpoints/<uid>/ to keep_checkpoints newest .pt files.

        Skips `latest.json` via the suffix filter — that pointer must
        survive every prune. No-op when keep_checkpoints<=0 or the
        validator store isn't wired (proxy-only / bootstrap states).
        """
        if self.validator_store is None or self.keep_checkpoints <= 0:
            return
        try:
            n = self.validator_store.prune_keep_latest(
                f"checkpoints/{self._validator_uid}/",
                keep=self.keep_checkpoints,
                suffix=".pt",
            )
            if n:
                logger.info(
                    "[GTX] pruned %d old checkpoint(s), keeping latest %d",
                    n, self.keep_checkpoints,
                )
        except Exception as exc:
            logger.debug("checkpoint prune failed: %s", exc)

    def _prune_proposals(self) -> None:
        """Trim this validator's proposals/<uid>/ to keep_proposals newest .grad files."""
        if self.validator_store is None or self.keep_proposals <= 0:
            return
        try:
            n = self.validator_store.prune_keep_latest(
                f"proposals/{self._validator_uid}/",
                keep=self.keep_proposals,
                suffix=".grad",
            )
            if n:
                logger.info(
                    "[GTX] pruned %d old proposal(s), keeping latest %d",
                    n, self.keep_proposals,
                )
        except Exception as exc:
            logger.debug("proposal prune failed: %s", exc)

    def _run_data_cleanup(self) -> int:
        """Delete stale parquets under data/ in the validator bucket.

        Returns the number of objects removed. No-op when no validator
        store is configured (proxy-only setups, or bootstrap before S3 is
        wired). Clears `_data_cleanup_pending` even on failure so one
        transient S3 error does not loop the cleanup attempt forever.
        """
        self._data_cleanup_pending = False
        if self.validator_store is None:
            return 0
        try:
            n = self.validator_store.delete_prefix(f"data/{self._validator_uid}/")
            if n:
                logger.info(
                    "[GTX] Sim transition cleanup: removed %d parquets from data/",
                    n,
                )
            return n
        except Exception as exc:
            logger.warning("Sim transition cleanup failed: %s", exc)
            return 0

    def _process_tick(self, tick: dict) -> None:
        """Process a single sim state tick into training rows.

        Replays events through MatchingEngine, accumulates rows per book,
        flushes to parquet when sim time crosses an interval boundary.
        Bumps `_sim_epoch` if the tick's timestamp is older than the last
        seen one — that's a sim restart inside this gradient server's lifetime.
        """
        from GenTRX.src.orderbook import MatchingEngine
        from GenTRX.src.util.schema import BID, ASK, CANCEL, LOB_DEPTH

        # Sim lifecycle markers from state.notices. 'ESE' flags a sim end:
        # clear in-memory buffers so no sim-A tail contaminates sim-B ticks,
        # queue the S3 data/ cleanup for the next aggregation-loop tick, and
        # return without touching books on this tick (ESE ticks carry no
        # trading data worth retaining). 'ESS' on its own is the new-sim
        # marker; if cleanup was still pending we run it as a safety net.
        sim_events = tick.get("sim_events") or []
        if "ESE" in sim_events:
            self._reset_sim_buffers()
            self._data_cleanup_pending = True
            logger.info(
                "[GTX] Sim end received; in-memory state cleared, "
                "S3 data/ cleanup queued for next aggregation-loop tick"
            )
            return
        if "ESS" in sim_events and self._data_cleanup_pending:
            # ESE was missed or arrived late — cleanup flag still set when
            # the new sim is already announcing itself. Treat as safety net.
            logger.info(
                "[GTX] Sim start received with cleanup still pending; "
                "running cleanup now before new sim data flushes"
            )
            self._run_data_cleanup()

        tick_ts = int(tick.get("ts", 0) or 0)
        # Two reference points: _last_seen_sim_ts catches an in-process sim
        # restart; _max_timestamp_ns catches a cross-process restart (it's
        # restored from the staging file, _last_seen_sim_ts is not).
        if tick_ts and (
            (self._last_seen_sim_ts and tick_ts < self._last_seen_sim_ts)
            or (self._max_timestamp_ns and tick_ts < self._max_timestamp_ns)
        ):
            prior = self._last_seen_sim_ts or self._max_timestamp_ns
            self._sim_epoch += 1
            # Heuristic fallback — an ESE should normally have landed before
            # this, but if the simulator didn't emit one we still want a
            # clean break.
            self._reset_sim_buffers()
            self._data_cleanup_pending = True
            logger.info(
                "Sim restart detected (ts %d → %d), advancing sim_epoch to %d "
                "and queueing data/ cleanup (no ESE marker seen)",
                prior, tick_ts, self._sim_epoch,
            )
        if tick_ts:
            self._last_seen_sim_ts = tick_ts

        def _pad(values, depth):
            return list(values[:depth]) + [0] * (depth - len(values))

        if self._price_scale is None:
            cfg = tick.get("config", {}) or {}
            pd = cfg.get("priceDecimals", 8)
            vd = cfg.get("volumeDecimals", 8)
            self._price_scale = 10**pd
            self._vol_scale = 10**vd

        # Bind sim_id from any packet that carries a config block — not just
        # the first. state_packager keeps resending config until sim_id is
        # non-None, so the second packet will have the real id even if the
        # first arrived before the simulator set its logDir.
        cfg = tick.get("config") or {}
        if cfg:
            new_sim_id = cfg.get("simulation_id")
            if new_sim_id and new_sim_id != self._sim_id:
                # Cross-process sim swap: restored staging buffer is from a
                # different sim than the one now pushing. Wipe before bind.
                if (
                    self._sim_id is None
                    and self._restored_sim_id is not None
                    and self._restored_sim_id != new_sim_id
                ):
                    self._sim_epoch += 1
                    self._reset_sim_buffers()
                    self._data_cleanup_pending = True
                    logger.info(
                        "sim_id mismatch on restart (staged=%s → live=%s), "
                        "advancing sim_epoch to %d and queueing data/ cleanup",
                        self._restored_sim_id, new_sim_id, self._sim_epoch,
                    )
                elif self._sim_id is not None:
                    # Different sim_id arriving without sim time going backwards
                    # — simulator was reseeded with a fresh id. Bump epoch so the
                    # dashboard treats it as a separate run within this process,
                    # and queue data/ cleanup in case the ESE path missed it.
                    self._sim_epoch += 1
                    self._reset_sim_buffers()
                    self._data_cleanup_pending = True
                    logger.info(
                        "sim_id changed (%s → %s), advancing sim_epoch to %d "
                        "and queueing data/ cleanup",
                        self._sim_id, new_sim_id, self._sim_epoch,
                    )
                self._restored_sim_id = None
                self._sim_id = new_sim_id
                logger.info("Bound to sim_id=%s (sim_epoch=%d)", self._sim_id, self._sim_epoch)
                self._log_event({"type": "sim_bind"})

        books = tick.get("books", {})

        for book_id, book_data in books.items():
            book_id = int(book_id)
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])
            events = book_data.get("events", [])

            if book_id not in self._engines:
                self._engines[book_id] = MatchingEngine()
                self._order_sides[book_id] = {}
                self._last_ts[book_id] = 0
                self._session_open_mid[book_id] = None
                self._pending_rows[book_id] = []
                self._pending_interval_start[book_id] = 0

            engine = self._engines[book_id]
            engine.reset()
            for level in reversed(bids):
                p = round(level[0] * self._price_scale)
                v = max(1, round(level[1] * self._vol_scale))
                if p > 0:
                    engine.process_order(BID, p, v, is_buy=True)
            for level in reversed(asks):
                p = round(level[0] * self._price_scale)
                v = max(1, round(level[1] * self._vol_scale))
                if p > 0:
                    engine.process_order(ASK, p, v, is_buy=False)

            taker_fill_qty: dict[int, float] = {}
            for ev in events:
                if ev.get("y") == "t":
                    tid = ev.get("Ti", 0)
                    taker_fill_qty[tid] = taker_fill_qty.get(tid, 0.0) + float(
                        ev.get("q", 0)
                    )

            for ev in events:
                y = ev.get("y", "o")
                if y == "t":
                    continue
                side = ev.get("s", 0)
                eid = ev.get("i", 0)
                price = float(ev.get("p") or 0)
                remaining = float(ev.get("q", 0))
                evt = int(ev.get("t", 0))

                is_buy = side == 0
                if y == "c":
                    order_type = CANCEL
                    qty = remaining
                else:
                    order_type = BID if is_buy else ASK
                    self._order_sides[book_id][eid] = is_buy
                    qty = remaining + taker_fill_qty.get(eid, 0.0)

                price_ticks = round(price * self._price_scale)
                vol_ticks = max(1, round(qty * self._vol_scale))

                snap = engine.snapshot()
                mid = snap.mid_price
                if self._session_open_mid[book_id] is None and mid > 0:
                    self._session_open_mid[book_id] = mid

                ask_vols = _pad(snap.ask_volumes, LOB_DEPTH)
                bid_vols = _pad(snap.bid_volumes, LOB_DEPTH)

                row = {
                    "timestamp": evt,
                    "order_type": order_type,
                    "rel_price": price_ticks - mid if mid > 0 else 0,
                    "volume_int": int(qty),
                    "volume_dec": qty - int(qty),
                    "interval_ns": (
                        evt - self._last_ts[book_id]
                        if self._last_ts[book_id] > 0
                        else 0
                    ),
                    "mid_price": mid,
                    "time_of_day_s": int((evt // 1_000_000_000) % 86400),
                    "mid_price_delta": (
                        int(mid - self._session_open_mid[book_id])
                        if self._session_open_mid[book_id]
                        else 0
                    ),
                }
                for i in range(LOB_DEPTH):
                    row[f"lob_ask_vol_{i + 1}"] = (
                        float(ask_vols[i]) / self._price_scale
                    )
                    row[f"lob_bid_vol_{i + 1}"] = (
                        float(bid_vols[i]) / self._price_scale
                    )

                self._pending_rows[book_id].append(row)
                engine.process_order(order_type, price_ticks, vol_ticks, is_buy)
                self._last_ts[book_id] = evt

                if evt > self._max_timestamp_ns:
                    self._max_timestamp_ns = evt

                # Set interval start on first row
                if self._pending_interval_start[book_id] == 0:
                    self._pending_interval_start[book_id] = evt

                # Flush when sim time crosses interval boundary
                interval_elapsed = evt - self._pending_interval_start[book_id]
                if interval_elapsed >= self._parquet_interval_ns:
                    self._flush_book_parquet(book_id)

    def _flush_book_parquet(self, book_id: int) -> None:
        """Flush accumulated rows for a book to a parquet file on S3."""
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import io as _io

        from GenTRX.src.util.schema import LOB_DEPTH, order_stream_schema

        rows = self._pending_rows.get(book_id, [])
        if not rows:
            return

        ts_min = rows[0]["timestamp"]
        ts_max = rows[-1]["timestamp"]
        tag_start = _ts_to_tag(ts_min)
        tag_end = _ts_to_tag(ts_max)
        pq_filename = f"{tag_start}-{tag_end}.parquet"

        try:
            columns = {
                "timestamp": pa.array(
                    [r["timestamp"] for r in rows], type=pa.timestamp("ns")
                ),
                "order_type": np.array(
                    [r["order_type"] for r in rows], dtype=np.int8
                ),
                "rel_price": np.array(
                    [r["rel_price"] for r in rows], dtype=np.int64
                ),
                "volume_int": np.array(
                    [r["volume_int"] for r in rows], dtype=np.int32
                ),
                "volume_dec": np.array(
                    [r["volume_dec"] for r in rows], dtype=np.float32
                ),
                "interval_ns": np.array(
                    [r["interval_ns"] for r in rows], dtype=np.int64
                ),
                "mid_price": np.array(
                    [r["mid_price"] for r in rows], dtype=np.int64
                ),
                "time_of_day_s": np.array(
                    [r["time_of_day_s"] for r in rows], dtype=np.int32
                ),
                "mid_price_delta": np.array(
                    [r["mid_price_delta"] for r in rows], dtype=np.int64
                ),
            }
            for i in range(LOB_DEPTH):
                k_ask = f"lob_ask_vol_{i + 1}"
                k_bid = f"lob_bid_vol_{i + 1}"
                columns[k_ask] = np.array(
                    [r[k_ask] for r in rows], dtype=np.float64
                )
                columns[k_bid] = np.array(
                    [r[k_bid] for r in rows], dtype=np.float64
                )

            table = pa.table(columns, schema=order_stream_schema())
            buf = _io.BytesIO()
            pq.write_table(table, buf)
            # Write parquets to data bucket (read by miners + validator)
            self.validator_store.put_data(
                self._validator_uid,
                book_id=book_id,
                filename=pq_filename,
                data=buf.getvalue(),
            )
            logger.debug(
                "Parquet flushed: book %d, %d rows, %s",
                book_id,
                len(rows),
                pq_filename,
            )
            # Register and clear only on success — failed writes leave rows
            # intact so the next interval flush retries with the accumulated data.
            if book_id not in self._written_parquets:
                self._written_parquets[book_id] = []
            self._written_parquets[book_id].append((pq_filename, ts_min, ts_max))
            self._pending_rows[book_id] = []
            self._pending_interval_start[book_id] = 0
        except Exception as exc:
            logger.error(
                "Failed to write parquet for book %d: %s — rows retained for next flush",
                book_id, exc,
            )


def add_api_key_middleware(app, api_key: str) -> None:
    """Install X-API-Key gate on every request if `api_key` is truthy.

    Requests without a matching `X-API-Key` header get a 401 before any
    route handler runs. No-op when `api_key` is empty — callers that need
    auth-off semantics rely on binding to 127.0.0.1 instead.
    """
    if not api_key:
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class APIKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.headers.get("X-API-Key") != api_key:
                return JSONResponse(
                    {"error": "Invalid or missing X-API-Key"},
                    status_code=401,
                )
            return await call_next(request)

    app.add_middleware(APIKeyMiddleware)


def create_gradient_router(
    checkpoint_path: str,
    val_data_path: str,
    output_path: str | None = None,
    **kwargs,
) -> tuple[APIRouter, GradientAggregator]:
    """Create a FastAPI router + aggregator for gradient exchange.

    Data collection is handled by the validator/proxy — it saves
    mvtrx state updates directly to S3. The gradient server
    reads from S3 for scoring.

    Returns (router, aggregator) — caller must call aggregator.start().
    """
    aggregator = GradientAggregator(
        checkpoint_path=checkpoint_path,
        val_data_path=val_data_path,
        output_path=output_path,
        **kwargs,
    )

    router = APIRouter(prefix="/gentrx", tags=["gentrx"])

    @router.post("/state")
    async def receive_state(request: Request):
        """Receive a state tick from the validator (msgpack bytes).

        Replaces S3-based state transfer — validator POSTs state directly,
        gradient server processes in-memory. No validator bucket needed.
        """
        body = await request.body()
        aggregator.receive_state(body)
        return {"status": "ok"}

    @router.get("/assignment")
    async def get_assignment(miner_uid: int = 0):
        """Get book/interval assignment for a miner."""
        return aggregator.get_assignment(miner_uid)

    @router.get("/health")
    async def health():
        return {"status": "ok", "version": aggregator.version}

    @router.get("/version")
    async def get_version():
        # Used by the validator service to detect new aggregation rounds.
        return {"version": aggregator.version}

    @router.get("/scores")
    async def get_scores(since_round: int = -1):
        """Get latest scores. Returns 204 if no new scores since since_round."""
        from fastapi.responses import Response as _Resp

        scores = aggregator._latest_scores
        if scores is None:
            return _Resp(status_code=204)
        if scores.get("round", -1) <= since_round:
            return _Resp(status_code=204)  # no new scores
        return scores

    @router.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics for Grafana scraping (pull model).

        Exposes training round progress, gradient acceptance rates, per-miner
        scores, loss improvement, and aggregation timing.  Scrape at:
            http://<host>:<port>/gentrx/metrics
        """
        from fastapi.responses import Response as _Resp
        try:
            from prometheus_client import CollectorRegistry, Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
        except ImportError:
            return _Resp(status_code=503, content="prometheus_client not installed")

        reg = CollectorRegistry()
        labels = ["netuid", "validator_uid"]
        lv = [str(aggregator._chain.netuid if aggregator._chain else ""), aggregator._validator_uid]

        def _g(name, doc, extra_labels=()):
            return Gauge(name, doc, labels + list(extra_labels), registry=reg)

        # Round / version
        g_round   = _g("gentrx_agg_round",            "Current aggregation round")
        g_version = _g("gentrx_checkpoint_version",   "Model checkpoint version (increments on accepted round)")
        g_rounds_total = _g("gentrx_rounds_aggregated_total", "Cumulative rounds aggregated")
        g_rollbacks    = _g("gentrx_rollbacks_total",         "Cumulative aggregations rolled back (no improvement)")

        # Last aggregation
        g_n_scored   = _g("gentrx_last_n_scored",   "Gradients scored in last round")
        g_n_accepted = _g("gentrx_last_n_accepted", "Gradients accepted in last round")
        g_loss_before = _g("gentrx_last_loss_before", "Model loss before last aggregation")
        g_loss_after  = _g("gentrx_last_loss_after",  "Model loss after last aggregation")
        g_loss_delta  = _g("gentrx_last_loss_improvement", "Loss reduction in last round (before − after)")
        g_t_score = _g("gentrx_last_score_duration_s",     "Gradient scoring duration (s) in last round")
        g_t_agg   = _g("gentrx_last_aggregate_duration_s", "Aggregation duration (s) in last round")
        g_t_total = _g("gentrx_last_total_duration_s",     "Total round duration (s) in last round")

        # Active miners
        g_active    = _g("gentrx_active_miners",    "Miners with non-zero gradient score this round")
        g_benchmark = _g("gentrx_benchmark_miners", "Benchmark (off-chain) miners registered with gradient server")

        # Per-miner scores — is_benchmark=1 for UIDs registered via /register_bucket
        # (benchmark agents tracked for scoring but excluded from on-chain rewards)
        g_score_own  = _g("gentrx_miner_score_own",  "Miner own-data score last round",  ("miner_uid", "is_benchmark"))
        g_score_held = _g("gentrx_miner_score_held", "Miner held-out score last round",  ("miner_uid", "is_benchmark"))
        g_score      = _g("gentrx_miner_score",      "Miner combined score last round",   ("miner_uid", "is_benchmark"))
        g_accepted   = _g("gentrx_miner_accepted",   "1 if miner gradient accepted last round", ("miner_uid", "is_benchmark"))

        g_round.labels(*lv).set(aggregator._agg_round)
        g_version.labels(*lv).set(aggregator._version)
        g_rounds_total.labels(*lv).set(aggregator._rounds_aggregated_total)
        g_rollbacks.labels(*lv).set(aggregator._rollbacks_total)

        agg = aggregator._last_aggregation
        g_n_scored.labels(*lv).set(agg.get("n_scored", 0))
        g_n_accepted.labels(*lv).set(agg.get("n_accepted", 0))
        lb = agg.get("loss_before")
        la = agg.get("loss_after")
        if lb is not None:
            g_loss_before.labels(*lv).set(lb)
        if la is not None:
            g_loss_after.labels(*lv).set(la)
        if lb is not None and la is not None:
            g_loss_delta.labels(*lv).set(lb - la)
        g_t_score.labels(*lv).set(agg.get("t_score_s", 0))
        g_t_agg.labels(*lv).set(agg.get("t_aggregate_s", 0))
        g_t_total.labels(*lv).set(agg.get("t_total_s", 0))

        benchmark_uids = set(aggregator._static_buckets.keys())
        g_benchmark.labels(*lv).set(len(benchmark_uids))

        scores = aggregator._latest_scores
        if scores:
            per_miner = scores.get("scores", {})
            g_active.labels(*lv).set(sum(1 for v in per_miner.values() if v.get("accepted")))
            for uid_str, info in per_miner.items():
                is_bm = "1" if int(uid_str) in benchmark_uids else "0"
                mlv = lv + [uid_str, is_bm]
                s_own  = info.get("score_own")
                s_held = info.get("score_held")
                s      = info.get("score", 0.0)
                if s_own is not None:
                    g_score_own.labels(*mlv).set(s_own)
                if s_held is not None:
                    g_score_held.labels(*mlv).set(s_held)
                g_score.labels(*mlv).set(s)
                g_accepted.labels(*mlv).set(1 if info.get("accepted") else 0)
        else:
            g_active.labels(*lv).set(0)

        return _Resp(content=generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @router.post("/register_bucket/{uid}")
    async def register_bucket(uid: int, request: Request):
        """Register a static bucket for a UID not on-chain (e.g. benchmark miners).

        Body (JSON): {
            "endpoint_url": "http://...",
            "bucket_name": "agent-0",
            "access_key_id": "...",
            "secret_access_key": "..."
        }
        The entry is merged into _miner_buckets and survives subsequent chain
        refreshes because _refresh_miner_buckets only overwrites the dict when
        the chain returns a non-empty result, but the merge happens after.
        """
        from GenTRX.src.chain import BucketInfo
        data = await request.json()
        bi = BucketInfo(
            account_id=data.get("account_id", data.get("bucket_name", "")),
            access_key_id=data["access_key_id"],
            secret_access_key=data["secret_access_key"],
            _endpoint_override=data.get("endpoint_url"),
            _bucket_override=data.get("bucket_name"),
        )
        aggregator._miner_buckets[uid] = bi
        aggregator._static_buckets[uid] = bi
        logger.info("[GTX] registered static bucket for uid=%d bucket=%s", uid, bi.bucket_name)
        return {"status": "ok", "uid": uid, "bucket": bi.bucket_name}

    @router.get("/data-status")
    async def get_data_status():
        """Return available data ranges per book.

        Used by the validator to create assignments from what actually exists.
        Returns:
            {
                "max_ts": int,        # global max sim timestamp (ns)
                "version": int,       # current model version
                "round": int,         # current aggregation round
                "books": {
                    "0": {"parquets": ["00000000-00000300.parquet", ...], "max_ts": int},
                    ...
                }
            }
        """
        books = {}
        for bid, plist in aggregator._written_parquets.items():
            fnames = [fname for fname, _, _ in plist]
            bmax = max((end for _, _, end in plist), default=0)
            books[str(bid)] = {"parquets": fnames, "max_ts": bmax}
        return {
            "max_ts": aggregator._max_timestamp_ns,
            "version": aggregator._version,
            "round": aggregator._agg_round,
            "books": books,
        }

    @router.post("/round")
    async def push_round(request: Request):
        """Accept an assignment plan from the validator.

        The validator drives round scheduling (block-based or timer-based),
        creates assignments from available data, and pushes the plan here.
        The gradient server records the assignments and manages the lifecycle
        (gradient collection, scoring, aggregation).

        Body (JSON):
            {
                "round": int,
                "assignments": {
                    "uid": {round, model_version, books, ts_start, ts_end, data, ...},
                    ...
                }
            }
        """
        import json as _json_mod
        body = await request.body()
        payload = _json_mod.loads(body)

        round_id = payload.get("round", aggregator._agg_round)
        assignments = payload.get("assignments", {})
        push_block = payload.get("block")
        if push_block is not None:
            aggregator._last_push_block = int(push_block)

        # Pure accounting — no heavy work on the event loop.
        # Aggregation of the closing round is handled by _aggregation_loop,
        # which picks up _pending_aggregation_rounds on its next iteration.
        if round_id > aggregator._agg_round:
            prior_round = aggregator._agg_round
            # Preserve the closing round's assignments in _prev_round_assignments
            # so the loop can still find them for collection + aggregation.
            # Without this, the new round's installation below overwrites
            # them and round N's gradients never get scored.
            closing = {
                uid: a for uid, a in aggregator._assignments.items()
                if a.get("round") == prior_round
            }
            if closing:
                aggregator._prev_round_assignments.update(closing)
                # Remove from live _assignments to keep it round-N+1-only.
                for uid in closing:
                    aggregator._assignments.pop(uid, None)

            # Skip-stale policy: if the validator jumped multiple rounds ahead,
            # drop assignments older than prior_round (both live and in history).
            gap = round_id - aggregator._agg_round
            if gap > 1:
                stale_live = [
                    uid for uid, a in aggregator._assignments.items()
                    if a.get("round", -1) < prior_round
                ]
                for uid in stale_live:
                    aggregator._assignments.pop(uid, None)
                stale_prev = [
                    uid for uid, a in aggregator._prev_round_assignments.items()
                    if a.get("round", -1) < prior_round
                ]
                for uid in stale_prev:
                    aggregator._prev_round_assignments.pop(uid, None)
                logger.warning(
                    "[GTX] round jump %d -> %d: dropped %d stale assignments",
                    prior_round, round_id, len(stale_live) + len(stale_prev),
                )
            if closing:
                aggregator._pending_aggregation_rounds.add(prior_round)
                aggregator._pending_aggregation_at.setdefault(prior_round, time.time())
            aggregator._agg_round = round_id

        # Install the validator's assignments as DELIVERED. The validator
        # has already sent them to miners via dendrite, so the
        # heartbeat-loss fallback timer starts now.
        now = time.time()
        for uid_str, a in assignments.items():
            uid = int(uid_str)
            a["_state"] = "DELIVERED"
            a["_created_at"] = now
            a["_delivered_at"] = now
            a["_gradient_data"] = None
            a["_score"] = None
            aggregator._assignments[uid] = a

        aggregator._last_round_log_ts = 0.0  # emit status on next loop tick
        logger.info(
            "[GTX] round=%d accepted from validator: %d assignments",
            round_id, len(assignments),
        )
        aggregator._log_event({
            "type": "round_delivered",
            "round": round_id,
            "n_assignments": len(assignments),
        })
        return {"status": "ok", "round": round_id, "n_assignments": len(assignments)}

    return router, aggregator


# ---------------------------------------------------------------------------
# Standalone server for testing
# ---------------------------------------------------------------------------


def _resolve_validator_uid(args, chain, validator_store) -> str:
    """Best-effort: derive this gradient server's UID from on-chain commitments.

    Matches the validator_store's (bucket, endpoint) against each UID's
    committed BucketInfo. Returns the operator's --validator-uid if set,
    otherwise the discovered UID, otherwise "0" (proxy mode, no chain,
    or pre-commit bootstrap).
    """
    import bittensor as bt
    if args.validator_uid:
        return str(args.validator_uid)
    if chain is None or validator_store is None:
        return "0"
    target_bucket = (validator_store.bucket or "").lower()
    target_endpoint = (validator_store.endpoint_url or "").lower()
    if not target_bucket:
        return "0"
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            buckets = loop.run_until_complete(chain.get_miner_buckets())
        finally:
            loop.close()
    except Exception as exc:
        bt.logging.warning(f"Validator UID discovery: chain query failed: {exc}")
        return "0"
    for uid, bi in buckets.items():
        if (bi.bucket_name or "").lower() == target_bucket and (
            (bi.endpoint_url or "").lower() == target_endpoint
        ):
            bt.logging.info(f"Validator UID resolved from chain commitment: {uid}")
            return str(uid)
    bt.logging.warning(
        f"Validator UID not found in {len(buckets)} chain commitments; "
        "defaulting to '0'. Pass --validator-uid explicitly if the validator "
        "process has not committed its bucket yet."
    )
    return "0"


if __name__ == "__main__":
    import argparse
    import uvicorn
    from fastapi import FastAPI

    parser = argparse.ArgumentParser(description="GenTRX gradient exchange server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--val-data", required=True, help="Data directory (collected parquets)"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--log-path",
        default=None,
        help="JSONL log for dashboard (default: next to output)",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 = localhost only. "
             "Use 0.0.0.0 for remote access, but set --api-key for security).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GENTRX_API_KEY", ""),
        help="Shared secret for validator↔gradient server auth. "
             "When set, all requests must include X-API-Key header. "
             "Also reads from GENTRX_API_KEY env var.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Aggregation interval in seconds (default: 30)",
    )
    parser.add_argument("--min-score", type=float, default=-0.1)
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=5,
        help="Rounds with no min_score/rollback after fresh model init (default: 5)",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=10,
        help="Max batches of validation data to evaluate each miner's gradient "
        "on. Default 10 — loss signal saturates fast, compute scales linearly. "
        "Bump to 30-50 only if you suspect noisy scores; drop to 5 for faster "
        "per-round aggregation on CPU-only gradient servers.",
    )
    parser.add_argument("--books-per-miner", type=int, default=3)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of books held for validation per interval",
    )
    parser.add_argument(
        "--parquet-interval-ns",
        type=int,
        default=300_000_000_000,
        help="Sim time per parquet file (default: 5min = 300_000_000_000 ns, matches training window)",
    )
    parser.add_argument(
        "--loop-sleep-s",
        type=float,
        default=5.0,
        help="Sleep between aggregation loop iterations (default: 5s)",
    )
    parser.add_argument(
        "--blocks-per-round",
        type=int,
        default=25,
        help="Server-side estimate of how many blocks make up one round. "
             "Used only by the heartbeat-loss fallback in _round_complete. "
             "Should match the validator's --gentrx.blocks_per_round.",
    )
    parser.add_argument(
        "--block-time-s",
        type=float,
        default=12.0,
        help="Assumed seconds per block on the target chain (default: 12s, "
             "finney). Combined with --blocks-per-round to estimate the "
             "round duration for the heartbeat-loss fallback.",
    )
    parser.add_argument(
        "--round-grace-s",
        type=float,
        default=30.0,
        help="Grace seconds added to the heartbeat-loss fallback estimate "
             "before force-closing a round (default: 30s). Only fires if "
             "the validator stops pushing POST /gentrx/round.",
    )
    parser.add_argument(
        "--max-gradient-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Reject gradients larger than this (default: 10 MB)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=10,
        help="Keep newest N checkpoints in the validator bucket (default: 10). "
             "Older checkpoints are deleted after a new one is published. "
             "Set 0 to disable pruning (you handle cleanup yourself — "
             "consider this if you mirror checkpoints to cold storage).",
    )
    parser.add_argument(
        "--keep-proposals",
        type=int,
        default=10,
        help="Keep newest N proposals in the validator bucket (default: 10). "
             "Older proposals are deleted after a new one is published. "
             "Set 0 to disable pruning.",
    )
    parser.add_argument(
        "--window-ns",
        type=int,
        default=None,
        help="Training window per assignment in ns (default: 5min = 300_000_000_000 ns)",
    )
    parser.add_argument(
        "--miner-buckets",
        default=None,
        help="JSON file with per-miner bucket config (LocalBucketConfig). "
        "Use only for proxy test (no chain). Localnet/production use --subtensor-network.",
    )
    parser.add_argument(
        "--subtensor-network",
        default=None,
        help="Bittensor network endpoint for chain-based bucket discovery "
        "(e.g. 'local', 'finney', 'wss://...'). Mutually exclusive with --miner-buckets.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Subnet netuid for chain commitment lookup (required with --subtensor-network).",
    )
    parser.add_argument(
        "--mode",
        default="simulation",
        choices=["simulation", "exchange"],
        help="Training mode shard for bucket keys (default: simulation). "
        "All keys live under gentrx/<network>/<mode>/. "
        "'exchange' reserves the prefix for future exchange-data training; "
        "no working data path today — operators should leave this at 'simulation'.",
    )
    parser.add_argument(
        "--network",
        default=None,
        choices=["mainnet", "testnet"],
        help="Explicit network shard for bucket keys (mainnet or testnet). "
             "Overrides the heuristic derived from --subtensor-network. "
             "Required when connecting via a custom wss:// endpoint to finney "
             "that is not automatically recognised. "
             "Equivalent to setting GENTRX_NETWORK in the environment.",
    )
    parser.add_argument(
        "--endpoint-override",
        default=os.environ.get("GENTRX_CHAIN_ENDPOINT_OVERRIDE") or None,
        help="Override S3 endpoint for all miner buckets (useful for MinIO localnet "
        "where on-chain commitments don't include the endpoint URL). "
        "Also reads from GENTRX_CHAIN_ENDPOINT_OVERRIDE env var.",
    )
    parser.add_argument(
        "--is-aggregator",
        action="store_true",
        default=True,
        help="This server is the uid-0 aggregator: aggregates gradients and "
        "publishes canonical checkpoints. Use --no-is-aggregator for sibling "
        "validators that only score and publish scores.",
    )
    parser.add_argument(
        "--no-is-aggregator",
        dest="is_aggregator",
        action="store_false",
        help="Run as a sibling validator: score miners and publish scores only, "
        "no gradient aggregation or checkpoint publish.",
    )
    parser.add_argument(
        "--validator-uid",
        default="",
        help="Validator UID on the subnet. Used for both Prometheus metric "
        "labels and the proposals/<uid>/ path scoping. Optional — when "
        "omitted, the gradient server self-discovers it by matching its "
        "validator_store bucket against on-chain commitments. Falls back to "
        "'0' if discovery fails (proxy mode, pre-commit bootstrap).",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", ""),
        help="Enable wandb mirroring of aggregation events to the named project. "
        "Requires WANDB_API_KEY. Empty (default) = wandb disabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME", ""),
        help="Optional wandb run name (default: wandb auto-generates).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log verbosity for GenTRX loggers (default: info).",
    )
    args = parser.parse_args()

    # CPU core affinity — pin to the last GRAD_CORES_COUNT cores (same pattern as
    # the validator via taos/im/utils/affinity.py).  The validator excludes these
    # cores from its own affinity so the two processes don't contend.
    _grad_cores_count = int(os.environ.get("GRAD_CORES_COUNT", "0"))
    if _grad_cores_count > 0:
        import multiprocessing as _mp
        _total_cpu = _mp.cpu_count()
        if _total_cpu > _grad_cores_count:
            _grad_cores = list(range(_total_cpu - _grad_cores_count, _total_cpu))
            try:
                os.sched_setaffinity(0, set(_grad_cores))
            except (AttributeError, OSError):
                pass  # not available on macOS / non-Linux
            import torch as _torch
            _torch.set_num_threads(_grad_cores_count)

    import bittensor as bt
    # Configure bt.logging before bt.Subtensor() so bt never defaults to
    # Warning level and suppresses our INFO messages.
    if args.log_level == "debug":
        bt.logging.set_debug()
    else:
        bt.logging.set_info()
    # enable_third_party_loggers is called once after bt.Subtensor so that all
    # loggers (including ones bt.Subtensor creates) are covered in a single pass.
    # Calling it here too would add a second QueueHandler to each logger and
    # cause duplicate log entries.

    # Bucket layout: gentrx/<network>/<mode>/...
    # network is derived from the connected subtensor (finney → mainnet,
    # everything else → testnet). mode is operator-selected via --mode.
    from GenTRX.src.gradient_store import (
        create_validator_store_from_env,
        gentrx_prefix,
        network_from_subtensor,
    )

    if args.network:
        os.environ["GENTRX_NETWORK"] = args.network
    network = network_from_subtensor(args.subtensor_network)
    bucket_prefix = gentrx_prefix(network, args.mode)
    bt.logging.info(
        f"Bucket prefix: {bucket_prefix} "
        f"(network={network}, mode={args.mode})"
    )

    # Single validator bucket: checkpoints/ + data/ + proposals/ all live
    # under the (network, mode) prefix. Committed to chain so miners +
    # aggregator discover it without pre-config.
    validator_store = create_validator_store_from_env(
        mode="write", prefix=bucket_prefix
    )
    if validator_store:
        bt.logging.info(f"S3 validator store (write): {validator_store.endpoint_url}/{validator_store.bucket}")
    else:
        parser.error("GENTRX_VALIDATOR_S3_* env vars required for gradient server")

    # Bucket discovery: chain (production/localnet) or local JSON (proxy test)
    chain = None
    if args.subtensor_network:
        if args.netuid is None:
            parser.error("--netuid is required when --subtensor-network is set")
        from GenTRX.src.chain import GenTRXChain

        sub = bt.Subtensor(network=args.subtensor_network)
        meta = sub.metagraph(args.netuid)
        chain = GenTRXChain(sub, args.netuid, meta)
        chain._endpoint_override = args.endpoint_override  # for MinIO localnet
        bt.logging.info(
            f"Chain bucket discovery: network={args.subtensor_network} "
            f"netuid={args.netuid} (override={args.endpoint_override or 'none'})"
        )
    elif args.miner_buckets:
        from GenTRX.src.chain import LocalBucketConfig

        chain = LocalBucketConfig(args.miner_buckets)
        bt.logging.info(f"Miner buckets loaded from {args.miner_buckets} (proxy test mode)")

    args.validator_uid = _resolve_validator_uid(args, chain, validator_store)

    # Route all non-bt loggers (GenTRX.src.*, root, …) through bt.logging so
    # gradient server log format matches the validator. Called once here, after
    # bt.Subtensor, so every logger bt creates during init is covered in a
    # single pass. Calling it earlier too would add duplicate QueueHandlers.
    bt.logging.enable_third_party_loggers()
    # Suppress noisy uvicorn access logs — must come after enable_third_party
    # since that call resets all levels to bt's current level.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    app = FastAPI(title="GenTRX Gradient Server")
    router, aggregator = create_gradient_router(
        checkpoint_path=args.checkpoint,
        val_data_path=args.val_data,
        output_path=args.output,
        log_path=args.log_path,
        interval=args.interval,
        min_score=args.min_score,
        warmup_rounds=args.warmup_rounds,
        max_val_batches=args.max_val_batches,
        books_per_miner=args.books_per_miner,
        val_fraction=args.val_fraction,
        validator_store=validator_store,
        chain=chain,
        is_aggregator=args.is_aggregator,
        parquet_interval_ns=args.parquet_interval_ns,
        loop_sleep_s=args.loop_sleep_s,
        round_grace_s=args.round_grace_s,
        max_gradient_bytes=args.max_gradient_bytes,
        window_ns=args.window_ns,
        keep_checkpoints=args.keep_checkpoints,
        keep_proposals=args.keep_proposals,
        blocks_per_round=args.blocks_per_round,
        block_time_s=args.block_time_s,
        validator_uid=args.validator_uid,
        bucket_prefix=bucket_prefix,
    )
    # API key middleware — reject unauthenticated requests when key is set
    api_key = args.api_key
    add_api_key_middleware(app, api_key)
    if api_key:
        bt.logging.info("API key auth enabled (X-API-Key header required)")
    elif args.bind != "127.0.0.1":
        bt.logging.warning(
            f"Gradient server bound to {args.bind} WITHOUT API key — "
            "anyone can push state and read scores!"
        )

    # Optional wandb dashboard. Soft-dep: no-op when wandb isn't installed
    # or WANDB_PROJECT / --wandb-project isn't set.
    try:
        from GenTRX.src import wandb_ops
        wandb_ops.init_wandb(
            project=args.wandb_project,
            run_name=args.wandb_run_name or None,
            config={
                "netuid": args.netuid,
                "interval": args.interval,
                "min_score": args.min_score,
                "books_per_miner": args.books_per_miner,
                "val_fraction": args.val_fraction,
                "is_aggregator": args.is_aggregator,
                "parquet_interval_ns": args.parquet_interval_ns,
            },
            tags=["aggregator" if args.is_aggregator else "sibling"],
        )
        import atexit
        atexit.register(wandb_ops.finish_wandb)
    except Exception:
        pass

    app.include_router(router)
    aggregator.start()

    # Single-line runtime version snapshot so bug reports carry exact
    # versions even though the Python pin is now open-ended (>=3.10).
    try:
        from GenTRX.src.service import _log_runtime_versions
        _log_runtime_versions()
    except Exception:
        pass

    bt.logging.info(f"Listening on {args.bind}:{args.port}")
    # timeout_graceful_shutdown=2 — the validator keeps a busy keep-alive
    # connection (state pushes every tick, scores + assignment polls). With
    # uvicorn's default `None`, SIGINT waits forever for "in-flight" requests
    # that never quiesce because new ones arrive faster than they finish.
    uvicorn.run(app, host=args.bind, port=args.port, timeout_graceful_shutdown=2)
