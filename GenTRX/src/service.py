# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""GenTRX service — validator-side orchestrator for distributed training.

The validator drives round scheduling (block-based or timer-based), creates
assignments from available data, pushes the assignment plan to the gradient
server, and delivers assignments to miners via dendrite.

The gradient server is passive for scheduling — it processes state ticks,
reports data availability, accepts assignment plans, and handles gradient
collection + scoring + aggregation.

HTTP contract with gradient server:
  POST /gentrx/state          — push sim state tick (msgpack, every tick)
  GET  /gentrx/data-status    — available data ranges per book
  POST /gentrx/round          — push assignment plan for a round
  GET  /gentrx/scores         — poll miner scores
  GET  /gentrx/version        — health check

All HTTP calls use the GENTRX_API_KEY shared secret (via X-API-Key header).
"""

from __future__ import annotations

import hashlib
import random
import time
from typing import Any

from GenTRX.src.bt_log import gtx_log

# Bittensor mainnet block time (~12 s/block). Used only for human-readable
# "next round in ~N min" log messages; not used for any scheduling logic.
_BITTENSOR_BLOCK_TIME_S = 12.0


def _log_runtime_versions() -> None:
    """One-shot startup log: Python + key dep versions.

    With the Python version pin now open-ended (>=3.10) and no requirements
    lock file, bug reports need to carry the exact versions the operator is
    actually running. Cheap to compute, single line per dep.
    """
    import sys
    lines = [f"python={sys.version.split()[0]}"]
    for name in ("bittensor", "torch", "httpx", "msgpack", "boto3", "fastapi", "uvicorn"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            lines.append(f"{name}={ver}")
        except Exception:
            lines.append(f"{name}=absent")
    gtx_log.info("runtime: %s", " ".join(lines))


class GenTRXService:
    """Validator-side orchestrator for GenTRX distributed training.

    Drives round scheduling, creates assignments from available data,
    pushes them to the gradient server, delivers to miners via dendrite.
    """

    # Assignment creation defaults (same as old gradient_server values)
    DEFAULT_BOOKS_PER_MINER = 3
    DEFAULT_WINDOW_NS = 300_000_000_000  # 5 min
    DEFAULT_BETA_ALPHA = 1.0
    DEFAULT_BETA_BETA = 3.0

    def __init__(
        self,
        packager: Any,
        gradient_server_url: str,
        api_key: str = "",
        poll_interval: float = 30.0,
        deliver_fn: Any | None = None,
        miner_uids: list[int] | None = None,
        miner_uids_fn: Any | None = None,
        log_path: str | None = None,
        # Round scheduling
        blocks_per_round: int = 0,
        get_block_fn: Any | None = None,
        # Assignment tunables
        books_per_miner: int = 3,
        window_ns: int = 0,
        val_fraction: float = 0.10,
        # Identity (used to scope data keys under data/<validator_uid>/)
        validator_uid: int | str = 0,
    ) -> None:
        if not gradient_server_url:
            raise ValueError(
                "GenTRXService requires a gradient_server_url. Single-machine "
                "deployments use a loopback URL like http://127.0.0.1:8100/gentrx; "
                "there is no in-process mode."
            )
        self._packager = packager
        self._server_url = gradient_server_url.rstrip("/")
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._deliver_fn = deliver_fn
        self._validator_uid = validator_uid
        self._miner_uids_static = miner_uids or []
        self._miner_uids_fn = miner_uids_fn

        # Round scheduling
        self._blocks_per_round = blocks_per_round
        self._get_block_fn = get_block_fn  # callable() -> int (current block number)
        self._current_round = 0
        self._last_round_push: float = 0.0
        self._last_known_block: int | None = None  # cached from last _should_advance_round query

        # Assignment tunables
        self._books_per_miner = books_per_miner or self.DEFAULT_BOOKS_PER_MINER
        self._window_ns = window_ns or self.DEFAULT_WINDOW_NS
        self._val_fraction = val_fraction
        self._val_books: set[str] | None = None  # lazy init from first data-status

        # log_path kept as a no-op for API compatibility. GenTRX records now
        # flow through bt.logging via the `gtx_log` shim — no separate
        # handler chain, no duplicate emits. Operators grep bt's stream for
        # `[GTX]` to isolate GenTRX activity.
        _ = log_path

        # State
        self._scores: dict[int, dict] = {}
        self._last_aggregation_stats: dict = {}
        self._last_poll: float = 0.0
        self._last_score_poll: float = 0.0
        self._last_score_round_seen: int = -1
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 60.0  # seconds between /health pings
        self._last_health_ok: bool | None = None   # None = never checked
        # Max sim timestamp we've pushed to the gradient server — lets us
        # skip the /data-status + /round HTTP chatter during the warmup
        # window (sim time < window_ns) before any parquet could exist.
        self._max_sim_ts_pushed: int = 0
        self._last_warmup_log: float = 0.0

    @property
    def _miner_uids(self) -> list[int]:
        if self._miner_uids_fn is not None:
            try:
                return self._miner_uids_fn() or []
            except Exception:
                return []
        return self._miner_uids_static

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        deliver_fn: Any | None = None,
        miner_uids: list[int] | None = None,
        miner_uids_fn: Any | None = None,
        get_block_fn: Any | None = None,
        validator_uid: int | str = 0,
    ) -> GenTRXService | None:
        """Create from validator/proxy config.

        Returns None when GenTRX is disabled or gradient server URL is missing.
        """
        gentrx_cfg = getattr(config, "gentrx", None)
        if gentrx_cfg is None or not getattr(gentrx_cfg, "enabled", False):
            return None

        try:
            from GenTRX.src.state_packager import StatePackager
        except ImportError as exc:
            gtx_log.warning("import failed — disabled: %s", exc)
            return None

        server_url = getattr(gentrx_cfg, "gradient_server_url", "")
        if not server_url:
            gtx_log.error(
                "enabled but --gentrx.gradient_server_url is empty. "
                "Set it to the gradient server endpoint (e.g. "
                "http://127.0.0.1:8100/gentrx for single-machine setups)."
            )
            return None

        import os as _os
        api_key = getattr(gentrx_cfg, "api_key", "") or _os.environ.get("GENTRX_API_KEY", "")
        interval = getattr(gentrx_cfg, "interval", 30)
        log_path = getattr(gentrx_cfg, "log_path", "data/gentrx/gentrx_service.log")
        blocks_per_round = getattr(gentrx_cfg, "blocks_per_round", 0) or 0

        service = cls(
            packager=StatePackager(),
            gradient_server_url=server_url,
            api_key=api_key,
            poll_interval=interval,
            deliver_fn=deliver_fn,
            miner_uids=miner_uids,
            miner_uids_fn=miner_uids_fn,
            log_path=log_path,
            blocks_per_round=blocks_per_round,
            get_block_fn=get_block_fn,
            validator_uid=validator_uid,
        )
        gtx_log.info(
            "service init: server=%s, poll=%ds, blocks_per_round=%d",
            server_url, interval, blocks_per_round,
        )
        if blocks_per_round > 0 and get_block_fn is not None:
            try:
                block = get_block_fn()
                current_round = block // blocks_per_round
                next_block = (current_round + 1) * blocks_per_round
                blocks_until = next_block - block
                mins = blocks_until * _BITTENSOR_BLOCK_TIME_S / 60
                gtx_log.info(
                    "[GTX] current block=%d (round=%d); next round at block ~%d (~%.0f min)",
                    block, current_round, next_block, mins,
                )
            except Exception:
                pass
        else:
            gtx_log.info(
                "[GTX] round scheduling: timer mode, first round in ~%.0fs",
                interval,
            )
        _log_runtime_versions()
        return service

    # ------------------------------------------------------------------
    # State packaging (called every tick)
    # ------------------------------------------------------------------

    def push_state(self, state: Any) -> None:
        """Extract a tick packet and POST it to the gradient server."""
        if self._packager is None:
            return
        try:
            packet = self._packager.extract_state(state)
            # Track max sim timestamp seen so poll_and_deliver can skip
            # /data-status calls before the first full training window.
            ts = packet.get("ts") if isinstance(packet, dict) else None
            if ts is not None and ts > self._max_sim_ts_pushed:
                self._max_sim_ts_pushed = int(ts)
            self._post_state(packet)
        except Exception as exc:
            gtx_log.warning("push_state failed: %s", exc)

    def _post_state(self, packet: dict) -> None:
        import msgpack
        data = msgpack.packb(packet, use_bin_type=True)
        t_start = time.time()
        try:
            self._http_post_sync("/state", data)
            t = time.time() - t_start
            # Only log slow state pushes — they fire every tick, noise budget matters
            if t > 0.2:
                gtx_log.info("state POST slow t=%.2fs bytes=%d", t, len(data))
        except Exception as exc:
            gtx_log.debug("state POST failed after %.2fs: %s", time.time() - t_start, exc)

    # ------------------------------------------------------------------
    # Round scheduling + assignment creation (validator-driven)
    # ------------------------------------------------------------------

    def _should_advance_round(self) -> int | None:
        """Check if a new round should start. Returns new round number or None.

        Block-synced mode: round = block // blocks_per_round.
        Timer mode: advance after poll_interval since last round push.
        """
        if self._blocks_per_round > 0 and self._get_block_fn is not None:
            try:
                block = self._get_block_fn()
                self._last_known_block = block
                block_round = block // self._blocks_per_round
                if block_round > self._current_round:
                    return block_round
            except Exception as exc:
                gtx_log.debug("block query failed: %s", exc)
            return None

        # Timer mode
        now = time.time()
        if now - self._last_round_push >= self._poll_interval:
            return self._current_round + 1
        return None

    async def _fetch_data_status(self) -> dict | None:
        """GET /gentrx/data-status — available data ranges per book."""
        t_start = time.time()
        try:
            resp = await self._http_get_async("/data-status")
            t = time.time() - t_start
            if t > 0.5:
                gtx_log.info("data-status fetch t=%.2fs", t)
            return resp.json()
        except Exception as exc:
            gtx_log.debug("data-status fetch failed after %.2fs: %s", time.time() - t_start, exc)
            return None

    def _create_assignments(self, data_status: dict) -> dict[int, dict]:
        """Create assignments for all miners from available data.

        Uses the same beta-distribution time sampling as the old gradient
        server logic, but driven by the validator.
        """
        books = data_status.get("books", {})
        if not books:
            return {}

        max_ts = data_status.get("max_ts", 0)
        if max_ts < self._window_ns:
            gtx_log.debug("not enough data yet (max_ts=%d < window=%d)", max_ts, self._window_ns)
            return {}

        model_version = data_status.get("version", 0)
        all_book_ids = sorted(books.keys(), key=lambda b: int(b))

        # Lazy init val books (deterministic split)
        if self._val_books is None:
            n_val = max(1, int(len(all_book_ids) * self._val_fraction))
            rng = random.Random(42)
            self._val_books = set(rng.sample(all_book_ids, min(n_val, len(all_book_ids))))

        train_books = [b for b in all_book_ids if b not in self._val_books]
        if not train_books:
            train_books = list(all_book_ids)

        # Shuffle books deterministically per round
        book_rng = random.Random(
            hashlib.sha256(f"{self._current_round}:books".encode()).hexdigest()
        )
        shuffled = list(train_books)
        book_rng.shuffle(shuffled)

        max_start = max_ts - self._window_ns
        miner_uids = self._miner_uids
        if not miner_uids:
            return {}

        # Resolve data bucket credentials from gradient server's validator store
        # (embedded in the assignment so miners don't need pre-configuration)
        # These come from the data-status response or from env vars
        import os
        data_endpoint = os.environ.get("GENTRX_VALIDATOR_S3_ENDPOINT_URL", "")
        data_bucket = os.environ.get("GENTRX_VALIDATOR_S3_BUCKET", "")
        data_access = os.environ.get("GENTRX_VALIDATOR_S3_READ_ACCESS_KEY", "")
        data_secret = os.environ.get("GENTRX_VALIDATOR_S3_READ_SECRET_KEY", "")

        assignments = {}
        for miner_uid in miner_uids:
            miner_rng = random.Random(
                hashlib.sha256(f"{self._current_round}:{miner_uid}:time".encode()).hexdigest()
            )
            beta_sample = 1.0 - miner_rng.betavariate(
                self.DEFAULT_BETA_ALPHA, self.DEFAULT_BETA_BETA
            )
            ts_start = int(beta_sample * max_start)
            ts_end = ts_start + self._window_ns

            start = (miner_uid * self._books_per_miner) % len(shuffled)
            assigned_books = [
                shuffled[(start + i) % len(shuffled)]
                for i in range(self._books_per_miner)
            ]

            # Resolve data keys from available parquets
            data_keys = []
            for book_id in assigned_books:
                book_info = books.get(book_id, {})
                for fname in book_info.get("parquets", []):
                    data_keys.append(f"data/{self._validator_uid}/{book_id}/intervals/{fname}")

            assignments[miner_uid] = {
                "round": self._current_round,
                "model_version": model_version,
                "books": assigned_books,
                "ts_start": ts_start,
                "ts_end": ts_end,
                "data": data_keys,
                "data_source": "s3",
                "data_endpoint": data_endpoint,
                "data_bucket": data_bucket,
                "data_access_key": data_access,
                "data_secret_key": data_secret,
            }

        return assignments

    async def _push_round(self, round_id: int, assignments: dict[int, dict]) -> bool:
        """POST /gentrx/round — push assignment plan to gradient server."""
        t_start = time.time()
        try:
            import json as _json
            body: dict = {
                "round": round_id,
                "assignments": {str(uid): a for uid, a in assignments.items()},
            }
            if self._last_known_block is not None:
                body["block"] = self._last_known_block
            payload = _json.dumps(body).encode()
            await self._http_post_async(
                "/round", payload, content_type="application/json"
            )
            t = time.time() - t_start
            self._last_round_push = time.time()
            gtx_log.info(
                "push_round round=%d n_assignments=%d bytes=%d t=%.2fs",
                round_id, len(assignments), len(payload), t,
            )
            return True
        except Exception as exc:
            gtx_log.warning("round push failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Main loop entry point (called every tick)
    # ------------------------------------------------------------------

    async def poll_and_deliver(self) -> None:
        """Drive round scheduling, deliver assignments, poll scores.

        Called every tick by handle_state. Rate-limited internally — most
        calls are no-ops.
        """
        if self._deliver_fn is None or not self._miner_uids:
            return

        # Warmup gate: before the sim has produced a full training window,
        # there's no data for assignments anyway. Skip the /data-status +
        # /round HTTP chatter until max_sim_ts crosses window_ns.
        if self._max_sim_ts_pushed < self._window_ns:
            now = time.time()
            if now - self._last_warmup_log >= 300.0:
                self._last_warmup_log = now
                pct = 100.0 * self._max_sim_ts_pushed / self._window_ns if self._window_ns else 0
                ns_remaining = self._window_ns - self._max_sim_ts_pushed
                mins_remaining = ns_remaining / 1e9 / 60
                gtx_log.info(
                    "[GTX] warmup: sim window %.1f%% full (%.0f min of sim time remaining "
                    "before first round can be assigned)",
                    pct, mins_remaining,
                )
            return

        # Check if a new round should start
        new_round = self._should_advance_round()
        if new_round is not None:
            data_status = await self._fetch_data_status()
            if data_status is not None:
                assignments = self._create_assignments(data_status)
                if assignments:
                    if await self._push_round(new_round, assignments):
                        # Advance only after a confirmed push so a simultaneous
                        # restart (gradient server not yet up) retries next tick
                        # rather than silently dropping the round.
                        self._current_round = new_round
                        gtx_log.info(
                            "round=%d: created %d assignments, pushing to miners",
                            new_round, len(assignments),
                        )
                        if self._blocks_per_round > 0:
                            next_block = (new_round + 1) * self._blocks_per_round
                            mins = self._blocks_per_round * _BITTENSOR_BLOCK_TIME_S / 60
                            gtx_log.info(
                                "[GTX] next assignment round at block ~%d (~%.0f min)",
                                next_block, mins,
                            )
                        else:
                            gtx_log.info(
                                "[GTX] next assignment round in ~%.0fs (timer mode)",
                                self._poll_interval,
                            )
                        try:
                            await self._deliver_fn(assignments)
                        except Exception as exc:
                            gtx_log.warning("delivery failed: %s", exc)
                            import traceback
                            gtx_log.debug(traceback.format_exc())
                    # if push failed (server down), don't advance — retry next tick
                else:
                    # No data to assign yet — advance to avoid thrashing data-status
                    self._current_round = new_round
                    gtx_log.debug("round=%d: no assignments (insufficient data)", new_round)

        # Score polling — rate-limited independently
        now = time.time()
        if now - self._last_score_poll >= self._poll_interval:
            self._last_score_poll = now
            await self.poll_scores()

        await self._check_health()

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    async def poll_scores(self) -> None:
        """Poll the gradient server for new scores."""
        t_start = time.time()
        try:
            resp = await self._http_get_async(
                f"/scores?since_round={self._last_score_round_seen}"
            )
            if resp.status_code == 204:
                return
            payload = resp.json()
            t = time.time() - t_start
            if t > 0.5:
                gtx_log.info("poll_scores t=%.2fs", t)
            round_id = payload.get("round", -1)
            if round_id > self._last_score_round_seen:
                self._last_score_round_seen = round_id
                self.receive_scores(payload)
        except Exception as exc:
            gtx_log.debug("poll_scores failed: %s", exc)

    def receive_scores(self, payload: dict) -> None:
        """Update local score store from a gradient-server response.

        The payload now carries an `aggregation` block + counters (see
        gradient_server._deliver_scores) so downstream consumers — Prometheus
        via ReportingService IPC, dashboards, etc. — have everything they
        need without a second HTTP call.
        """
        for uid_str, score_data in payload.get("scores", {}).items():
            self._scores[int(uid_str)] = score_data
        agg = payload.get("aggregation")
        if agg:
            self._last_aggregation_stats = dict(agg)
        counters = payload.get("counters")
        if counters:
            self._last_aggregation_stats.update(counters)
        scores_compact = {
            uid: f"{s['score']:.3f}{'✓' if s.get('accepted') else '✗'}"
            for uid, s in self._scores.items()
        }
        gtx_log.info(
            "scores: round=%s accepted=%s/%s scores=%s",
            payload.get("round"),
            payload.get("n_accepted"),
            payload.get("n_scored"),
            scores_compact,
        )

    def get_scores(self) -> dict[int, dict]:
        """Return current per-miner scores."""
        return dict(self._scores)

    def get_training_stats(self) -> dict:
        """Return last aggregation stats (loss, acceptance rate, version, timing)."""
        return dict(self._last_aggregation_stats)

    def register_benchmark_bucket(self, uid: int, bucket: dict) -> bool:
        """POST /gentrx/register_bucket/{uid} — inject static bucket for a benchmark miner.

        bucket keys: endpoint_url, bucket_name, access_key_id, secret_access_key
        """
        try:
            import httpx
            r = httpx.post(
                f"{self._server_url}/register_bucket/{uid}",
                json=bucket,
                headers=self._headers(),
                timeout=5.0,
            )
            return r.status_code == 200
        except Exception as exc:
            gtx_log.warning("register_benchmark_bucket uid=%d failed: %s", uid, exc)
            return False

    def is_healthy(self, timeout: float = 3.0) -> bool:
        """GET /health on the gradient server. Returns True if reachable and status=ok."""
        try:
            import httpx
            r = httpx.get(f"{self._server_url}/health", headers=self._headers(), timeout=timeout)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    async def _check_health(self) -> None:
        """Periodic health check — logs a warning on failure, info on recovery."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return
        self._last_health_check = now
        try:
            resp = await self._http_get_async("/health", timeout=3.0)
            ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception as exc:
            ok = False
            gtx_log.warning("health check failed: %s", exc)
        if ok and self._last_health_ok is not True:
            gtx_log.info("gradient server healthy at %s", self._server_url)
        elif not ok and self._last_health_ok is not False:
            gtx_log.warning("gradient server unreachable at %s", self._server_url)
        self._last_health_ok = ok

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    # Async variants (httpx) for the tick path so the validator's event loop
    # never blocks on HTTP. A tiny sync variant remains for push_state, which
    # is called from the state-packager thread, not the event loop.

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key} if self._api_key else {}

    async def _http_get_async(self, path: str, timeout: float = 5.0):
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(
                f"{self._server_url}{path}", headers=self._headers()
            )

    async def _http_post_async(
        self, path: str, data: bytes,
        content_type: str = "application/octet-stream", timeout: float = 5.0,
    ):
        import httpx
        headers = {"Content-Type": content_type, **self._headers()}
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                f"{self._server_url}{path}", content=data, headers=headers
            )

    def _http_post_sync(
        self, path: str, data: bytes,
        content_type: str = "application/octet-stream", timeout: float = 5.0,
    ):
        import httpx
        headers = {"Content-Type": content_type, **self._headers()}
        return httpx.post(
            f"{self._server_url}{path}",
            content=data, headers=headers, timeout=timeout,
        )
