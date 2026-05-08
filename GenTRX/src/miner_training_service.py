# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Miner-side training service.

Mirrors the validator's `gradient_server` topology on the miner side:
the trading agent stays thin and protocol-aware, the heavy work
(checkpoint download, train_window, gradient compress + upload)
runs in this service. Two deployment options:

- Same machine as the trading agent: loopback HTTP, no auth.
- Separate GPU host: set GENTRX_MINER_API_KEY on both sides.

This module is the in-process service class. The `__main__` driver
that exposes it over FastAPI is `GenTRX.src.miner_training_server`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from GenTRX.src.bt_log import gtx_log


logger = logging.getLogger("GenTRX.miner_training_service")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MinerTrainingConfig:
    """All gtx_-namespaced training knobs in one place.

    Field names drop the `gtx_` prefix here because the prefix is only
    needed when the keys travel through `--agent.params` and risk
    colliding with a strategy's own keys. Inside this module the names
    are unambiguous.
    """

    uid: int
    output_dir: Path = Path("data/live")
    gradient_dir: Path | None = None  # defaults to output_dir / "gradients"
    train_steps: int = 50
    train_batch_size: int = 16
    train_seq_len: int = 256
    train_lr: float = 1e-4
    top_k_frac: float = 0.01
    aggregator_uid: int = 0
    # Training mode shard: "simulation" (default) or "exchange". Combined
    # with the connected subtensor network to form the gentrx/<network>/<mode>/
    # bucket prefix. Leave at "simulation" unless instructed otherwise.
    mode: str = "simulation"
    # Explicit network shard override: "mainnet" or "testnet". When set, takes
    # precedence over the heuristic. Required for private finney node operators.
    # Equivalent to GENTRX_NETWORK env var.
    network_override: str = ""
    # Optional bootstrap: load this checkpoint at startup if it exists
    initial_checkpoint: Path | None = None
    # Retention: keep newest N gradient files under gradients/ in the
    # write bucket. 0 disables pruning (gradients accumulate forever
    # — operator handles cleanup themselves). Default 50 ≈ ~4 hours
    # of history at the standard round cadence.
    keep_gradients: int = 50


@dataclass
class _TrainingState:
    """Runtime state — not part of the public config."""

    pending_assignments: list[dict] = field(default_factory=list)
    training_thread: threading.Thread | None = None
    training_in_progress: bool = False
    train_window_id: int = 0
    model_version: int = 0
    last_uploaded_round: int | None = None
    last_loss_before: float | None = None
    last_loss_after: float | None = None
    retry_last_at: float = 0.0
    retry_cooldown: float = 30.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MinerTrainingService:
    """Owns the miner-side training loop.

    Lifecycle:
        svc = MinerTrainingService(MinerTrainingConfig(uid=7, ...))
        svc.attach_subtensor(subtensor, metagraph, netuid)  # for chain discovery
        svc.bootstrap_model()                               # pull latest checkpoint
        # then per assignment:
        svc.submit_assignment(payload)
        # query state:
        svc.get_status()
        # at shutdown:
        svc.shutdown(timeout=60)

    Thread safety:
        - submit_assignment() is safe from any thread.
        - All training work runs on a single dedicated background thread.
          Only one training window can be in flight at a time.
        - get_status() is read-only and safe from any thread.
    """

    def __init__(self, config: MinerTrainingConfig):
        self.cfg = config
        self.cfg.output_dir = Path(self.cfg.output_dir)
        if self.cfg.gradient_dir is None:
            self.cfg.gradient_dir = self.cfg.output_dir / "gradients"
        else:
            self.cfg.gradient_dir = Path(self.cfg.gradient_dir)
        self.cfg.gradient_dir.mkdir(parents=True, exist_ok=True)

        self.state = _TrainingState()
        self._lock = threading.Lock()  # protects pending_assignments + training_in_progress

        # Subtensor (set later via attach_subtensor for chain-based discovery)
        self._subtensor = None
        self._metagraph = None
        self._netuid: int | None = None

        # gentrx/<network>/<mode>/ prefix. network is finalized once
        # attach_subtensor() runs (subtensor.network → network_from_subtensor);
        # we start out with a testnet placeholder so unit-style tests that
        # never attach a subtensor still produce a well-formed prefix.
        from GenTRX.src.gradient_store import (
            gentrx_prefix,
            network_from_subtensor,
        )
        if self.cfg.network_override:
            import os as _os
            _os.environ.setdefault("GENTRX_NETWORK", self.cfg.network_override)
        self._bucket_prefix: str = gentrx_prefix(
            network_from_subtensor(None), self.cfg.mode
        )

        # Model / tokenizer (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.model_cfg = None
        self.tokenizer_cfg = None
        self.device = "cpu"

        # S3 stores
        self._store = None  # aggregator-bucket fallback (env-var)
        self._data_store = None  # default data store (== _store unless overridden)
        self._write_store = None  # per-miner write bucket
        self._discovered_aggregator_store = None  # cached chain-based discovery
        self._discovered_aggregator_uid: int = self.cfg.aggregator_uid
        self._s3_cache_dir: Path | None = None
        self._s3_cached_files: dict[str, Path] = {}

        self._tlog = self._build_logger()
        self._init_s3_stores()

        if self.cfg.initial_checkpoint and Path(self.cfg.initial_checkpoint).exists():
            try:
                self._load_model(str(self.cfg.initial_checkpoint))
            except Exception as exc:
                gtx_log.error(f"Failed to load initial checkpoint: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_subtensor(self, subtensor, metagraph, netuid: int) -> None:
        """Wire chain-based aggregator discovery.

        Also recomputes the bucket prefix from the connected subtensor's
        network identifier so all stores built after this call land in
        the right gentrx/<network>/<mode>/ shard.
        """
        from GenTRX.src.gradient_store import (
            gentrx_prefix,
            network_from_subtensor,
        )
        self._subtensor = subtensor
        self._metagraph = metagraph
        self._netuid = netuid
        if self.cfg.network_override:
            import os as _os
            _os.environ["GENTRX_NETWORK"] = self.cfg.network_override
        network_name = getattr(subtensor, "network", None)
        new_prefix = gentrx_prefix(
            network_from_subtensor(network_name), self.cfg.mode
        )
        if new_prefix != self._bucket_prefix:
            self._bucket_prefix = new_prefix
            for store in (self._store, self._data_store, self._write_store):
                if store is not None:
                    store.prefix = new_prefix
            gtx_log.info(
                f"GenTRX bucket prefix: {new_prefix} "
                f"(network={network_from_subtensor(network_name)}, mode={self.cfg.mode})"
            )

    def bootstrap_model(self) -> bool:
        """Pull the latest published checkpoint from the aggregator bucket.

        Returns True if a model is loaded after the call.
        """
        if self.model is not None:
            return True
        return self._ensure_model_version()

    def submit_assignment(self, payload: dict) -> dict:
        """Queue an assignment for training.

        Returns a status dict the HTTP layer can serialize as JSON.
        Never blocks on network or training — kicks off the training
        thread if one isn't already running.
        """
        if not isinstance(payload, dict) or "round" not in payload:
            return {"status": "rejected", "reason": "invalid payload"}

        with self._lock:
            self.state.pending_assignments.append(payload)
            in_progress = self.state.training_in_progress

        round_id = payload.get("round", "?")
        self._tlog.info(
            f"assignment queued: round={round_id} "
            f"validator_uid={payload.get('validator_uid', '?')} "
            f"books={payload.get('books', [])} "
            f"data_files={len(payload.get('data', []))} "
            f"in_progress={in_progress}"
        )

        if not in_progress:
            self._kick_training()
        return {"status": "queued", "round": round_id, "in_progress": in_progress}

    def get_status(self) -> dict:
        with self._lock:
            pending = len(self.state.pending_assignments)
            in_progress = self.state.training_in_progress
        return {
            "uid": self.cfg.uid,
            "training_in_progress": in_progress,
            "pending_assignments": pending,
            "model_version": self.state.model_version,
            "last_uploaded_round": self.state.last_uploaded_round,
            "last_loss_before": self.state.last_loss_before,
            "last_loss_after": self.state.last_loss_after,
            "train_window_id": self.state.train_window_id,
            "retry_cooldown_s": self.state.retry_cooldown,
            "pending_retry_count": self._pending_retry_count(),
        }

    def get_version_info(self) -> dict:
        return {
            "service": "miner_training_service",
            "uid": self.cfg.uid,
            "model_version": self.state.model_version,
        }

    def retry_pending_gradients(self) -> None:
        """Replay any locally-saved gradients from a previous failed S3 upload.

        Cooldown: 30 s → 300 s exponential. Reset to 30 s on success.
        Idempotent — call from any background tick.
        """
        if self._write_store is None:
            return
        pending_dir = self.cfg.gradient_dir / "pending"
        if not pending_dir.exists():
            return

        now = time.time()
        if now - self.state.retry_last_at < self.state.retry_cooldown:
            return
        self.state.retry_last_at = now

        any_failed = False
        for grad_path in sorted(pending_dir.glob("*.grad")):
            try:
                parts = grad_path.stem.split("_")
                round_id = int(parts[1])
                data = grad_path.read_bytes()
                self._write_store.put_gradient(
                    miner_uid=self.cfg.uid, round_id=round_id, data=data
                )
                grad_path.unlink()
                self._tlog.info(f"Retried pending gradient: round {round_id}")
            except Exception as exc:
                self._tlog.debug(f"Retry failed for {grad_path.name}: {exc}")
                any_failed = True

        if any_failed:
            self.state.retry_cooldown = min(self.state.retry_cooldown * 2, 300.0)
        else:
            self.state.retry_cooldown = 30.0

    def shutdown(self, timeout: float = 60.0) -> None:
        """Wait for any in-progress training thread to finish."""
        if self.state.training_thread and self.state.training_thread.is_alive():
            self._tlog.info("Shutdown: waiting for training thread...")
            self.state.training_thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal: setup
    # ------------------------------------------------------------------

    def _build_logger(self) -> logging.Logger:
        log = logging.getLogger(f"GenTRX.miner_training_service.{self.cfg.uid}")
        log.setLevel(logging.INFO)
        log.propagate = False
        if not log.handlers:
            fh = logging.FileHandler(self.cfg.gradient_dir / "train.log")
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
            )
            log.addHandler(fh)
        log.info(f"Training service initialized (uid={self.cfg.uid})")
        return log

    def _init_s3_stores(self) -> None:
        """Build the three logical stores from environment.

        Same conventions as `taos.im.agents.GenTRXAgent`:
          _store        : aggregator bucket fallback (env-var)
          _data_store   : default data store (overridden per-assignment)
          _write_store  : per-miner write bucket
        """
        try:
            from GenTRX.src.gradient_store import (
                create_aggregator_store_from_env,
                GradientStore,
            )
        except ImportError:
            return

        self._store = create_aggregator_store_from_env(prefix=self._bucket_prefix)
        if self._store:
            gtx_log.info(
                f"S3 aggregator bucket fallback: {self._store.endpoint_url}/{self._store.bucket}"
            )
        self._data_store = self._store

        agent_bucket = os.environ.get("GENTRX_AGENT_S3_BUCKET")
        if agent_bucket:
            self._write_store = GradientStore(
                endpoint_url=os.environ.get(
                    "GENTRX_AGENT_S3_ENDPOINT_URL",
                    self._store.endpoint_url if self._store else "",
                ),
                bucket=agent_bucket,
                access_key=os.environ.get("GENTRX_AGENT_S3_ACCESS_KEY", ""),
                secret_key=os.environ.get("GENTRX_AGENT_S3_SECRET_KEY", ""),
                region=os.environ.get("GENTRX_AGENT_S3_REGION", "auto"),
                prefix=self._bucket_prefix,
            )
            gtx_log.info(
                f"S3 write (gradients): {self._write_store.endpoint_url}/{self._write_store.bucket}"
            )

    # ------------------------------------------------------------------
    # Internal: training trigger
    # ------------------------------------------------------------------

    def _kick_training(self) -> None:
        """Spawn the background thread to consume pending assignments."""
        with self._lock:
            if self.state.training_in_progress:
                return
            assignments = self.state.pending_assignments
            self.state.pending_assignments = []
            if not assignments:
                return
            self.state.training_in_progress = True

        # Merge across validators
        all_data_keys: list[str] = []
        all_books: list[str] = []
        target_v = 0
        primary = assignments[0]
        for a in assignments:
            all_data_keys.extend(a.get("data", []))
            all_books.extend(a.get("books", []))
            v = int(a.get("model_version", 0) or 0)
            if v > target_v:
                target_v = v

        self._tlog.info(
            f"assignments: {len(assignments)} validator(s), "
            f"round={primary.get('round', '?')} "
            f"model_v={target_v} books={all_books} "
            f"data={len(all_data_keys)} files total"
        )

        if not all_data_keys:
            self._tlog.info("no data keys in any assignment — skipping")
            with self._lock:
                self.state.training_in_progress = False
            return
        if self.model is None and target_v <= 0:
            self._tlog.info(
                "no model loaded and assignment has no model_version — skipping (assignments preserved)"
            )
            with self._lock:
                self.state.pending_assignments = assignments + self.state.pending_assignments
                self.state.training_in_progress = False
            return

        self.state.training_thread = threading.Thread(
            target=self._download_and_train_background,
            args=(assignments, target_v, primary, all_books),
            daemon=True,
        )
        self.state.training_thread.start()

    def _download_and_train_background(
        self,
        assignments: list[dict],
        target_v: int,
        primary: dict,
        all_books: list[str],
    ) -> None:
        try:
            if self.model is None:
                self._tlog.info(f"no local model — bootstrapping from assignment (v{target_v})")
                if not self._ensure_model_version(target_v, primary) or self.model is None:
                    self._tlog.info("model bootstrap failed — skipping (assignments preserved)")
                    with self._lock:
                        self.state.pending_assignments = assignments + self.state.pending_assignments
                    return

            need_model = target_v > self.state.model_version and target_v > 0
            if need_model:
                self._tlog.info(
                    f"model rollover: v{self.state.model_version} → v{target_v}, downloading..."
                )

            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

            MODEL_DL_TIMEOUT_S = 60.0
            DATA_DL_TIMEOUT_S = 90.0
            with ThreadPoolExecutor(max_workers=1 + len(assignments)) as pool:
                f_model = (
                    pool.submit(self._ensure_model_version, target_v, primary)
                    if need_model else None
                )
                f_data = [pool.submit(self._download_assignment_data, a) for a in assignments]

                if f_model is not None:
                    try:
                        model_ok = f_model.result(timeout=MODEL_DL_TIMEOUT_S)
                    except FutTimeout:
                        model_ok = False
                        self._tlog.warning(
                            f"model v{target_v} download timed out after "
                            f"{MODEL_DL_TIMEOUT_S:.0f}s — training on v{self.state.model_version}"
                        )
                    if not model_ok:
                        self._tlog.warning(
                            f"model v{target_v} download failed — training on v{self.state.model_version}"
                        )
                parquet_files: list[Path] = []
                for f in f_data:
                    try:
                        parquet_files.extend(f.result(timeout=DATA_DL_TIMEOUT_S))
                    except FutTimeout:
                        self._tlog.warning(f"data download timed out after {DATA_DL_TIMEOUT_S:.0f}s")

            if not parquet_files:
                self._tlog.warning("download failed for all files — skipping")
                return

            self._tlog.info(f"{len(parquet_files)} files ready, deepcopy model...")
            train_model = copy.deepcopy(self.model)

            self._tlog.info(
                f"window {self.state.train_window_id} STARTED | "
                f"{len(parquet_files)} files from {len(assignments)} validator(s) | "
                f"books={all_books} | "
                f"{self.cfg.train_steps} steps | "
                f"model_v={target_v}"
            )
            self._train_background(parquet_files, train_model, primary)
        except Exception as exc:
            self._tlog.error(f"_download_and_train_background failed: {exc}")
            import traceback
            self._tlog.error(traceback.format_exc())
        finally:
            with self._lock:
                self.state.training_in_progress = False
            self._tlog.info("thread finished")
            self._clear_s3_cache()

    def _train_background(
        self, parquet_files: list[Path], train_model, assignment: dict | None = None
    ) -> None:
        try:
            from GenTRX.src.dataloader import OrderDataset, ChunkSampler
            from GenTRX.src.distributed import train_window, WindowConfig
            from GenTRX.src.gradient import compress, serialize
            from torch.utils.data import DataLoader

            self._tlog.info(f" building dataset from {len(parquet_files)} files...")
            dataset = OrderDataset(
                parquet_files,
                seq_len=self.cfg.train_seq_len,
                tokenizer=self.tokenizer,
                max_cached=2,
            )
            sampler = ChunkSampler(dataset, shuffle=True)
            loader = DataLoader(
                dataset,
                batch_size=self.cfg.train_batch_size,
                sampler=sampler,
                num_workers=0,
            )
            self._tlog.info(
                f"dataset ready: {dataset.total_orders} orders, "
                f"{len(loader)} batches, training {self.cfg.train_steps} steps..."
            )

            win_cfg = WindowConfig(
                n_steps=self.cfg.train_steps,
                lr=self.cfg.train_lr,
                window_id=self.state.train_window_id,
                miner_uid=self.cfg.uid,
            )
            delta = train_window(train_model, loader, win_cfg, self.device)
            self._tlog.info(
                f"training done: loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f}"
            )

            comp = compress(delta, top_k_frac=self.cfg.top_k_frac)
            data = serialize(comp)

            round_id = (assignment or {}).get("round", self.state.train_window_id)
            if self._write_store is not None:
                try:
                    self._write_store.put_gradient(
                        miner_uid=self.cfg.uid,
                        round_id=round_id,
                        data=data,
                    )
                    self._tlog.info(f"gradient uploaded to S3 (round={round_id})")
                    self.state.last_uploaded_round = round_id
                    self._prune_gradients()
                except Exception as exc:
                    self._tlog.warning(f"S3 upload failed: {exc} — saving for retry")
                    pending_dir = self.cfg.gradient_dir / "pending"
                    pending_dir.mkdir(parents=True, exist_ok=True)
                    pending_path = pending_dir / f"block_{round_id:08d}_miner_{self.cfg.uid}.grad"
                    pending_path.write_bytes(data)
            else:
                raise RuntimeError(
                    "No S3 store configured. Set GENTRX_AGENT_S3_* env vars to enable gradient upload."
                )

            self.state.last_loss_before = float(delta.metadata.loss_before)
            self.state.last_loss_after = float(delta.metadata.loss_after)
            self._tlog.info(
                f"window {self.state.train_window_id} COMPLETE | "
                f"loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f} | "
                f"gradient {len(data)/1024:.1f} KB"
            )
            self.state.train_window_id += 1

        except Exception as exc:
            self._tlog.error(f"FAILED: {exc}")
            import traceback
            self._tlog.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Internal: S3 store discovery + downloads
    # ------------------------------------------------------------------

    def _get_aggregator_store_for_assignment(self, assignment: dict | None):
        """Return a GradientStore pointing at a validator bucket with a
        published checkpoint. See taos.im.agents.GenTRXAgent for the equivalent
        inline logic; this is a duplicate by design.
        """
        if self._discovered_aggregator_store is not None:
            return self._discovered_aggregator_store

        from GenTRX.src.gradient_store import GradientStore

        def _build_store(bi) -> GradientStore:
            return GradientStore(
                endpoint_url=bi.endpoint_url,
                bucket=bi.bucket_name,
                access_key=bi.access_key_id,
                secret_key=bi.secret_access_key,
                region=os.environ.get("GENTRX_VALIDATOR_S3_REGION", "auto"),
                prefix=self._bucket_prefix,
            )

        try:
            if (
                self._subtensor is not None
                and self._metagraph is not None
                and self._netuid is not None
            ):
                from GenTRX.src.chain import GenTRXChain
                gtx_chain = GenTRXChain(self._subtensor, self._netuid, self._metagraph)

                configured_uid = self.cfg.aggregator_uid

                # Step 1: configured uid via chain
                try:
                    bucket_info = gtx_chain.get_bucket(configured_uid)
                    if bucket_info is not None:
                        store = _build_store(bucket_info)
                        latest = store.get_latest_version(configured_uid)
                        if latest > 0:
                            self._tlog.info(
                                f"Aggregator bucket discovered: uid={configured_uid} "
                                f"{bucket_info.endpoint_url}/{bucket_info.bucket_name} "
                                f"(latest v{latest})"
                            )
                            self._discovered_aggregator_store = store
                            self._discovered_aggregator_uid = configured_uid
                            return store
                except Exception as exc:
                    self._tlog.debug(f"uid={configured_uid} bucket probe failed: {exc}")

                # Step 2: env-var store
                if self._store is not None:
                    try:
                        if self._store.get_latest_version(configured_uid) > 0:
                            self._tlog.info(
                                f"Aggregator bucket from env: "
                                f"{self._store.endpoint_url}/{self._store.bucket}"
                            )
                            self._discovered_aggregator_store = self._store
                            self._discovered_aggregator_uid = configured_uid
                            return self._store
                    except Exception as exc:
                        self._tlog.debug(f"env-var store probe failed: {exc}")

                # Steps 3+4: sender then remaining metagraph
                scan_uids: list[int] = []
                sender = (assignment or {}).get("validator_uid")
                if sender is not None:
                    scan_uids.append(int(sender))
                try:
                    n = int(self._metagraph.n.item())
                except Exception:
                    n = 0
                for uid in range(n):
                    if uid != configured_uid and uid not in scan_uids:
                        scan_uids.append(uid)

                for uid in scan_uids:
                    try:
                        bucket_info = gtx_chain.get_bucket(uid)
                        if bucket_info is None:
                            continue
                        store = _build_store(bucket_info)
                        latest = store.get_latest_version(uid)
                        if latest > 0:
                            self._tlog.info(
                                f"Aggregator bucket discovered: uid={uid} "
                                f"{bucket_info.endpoint_url}/{bucket_info.bucket_name} "
                                f"(latest v{latest})"
                            )
                            self._discovered_aggregator_store = store
                            self._discovered_aggregator_uid = uid
                            return store
                    except Exception as exc:
                        self._tlog.debug(f"uid={uid} bucket probe failed: {exc}")
                        continue
        except Exception as exc:
            self._tlog.warning(f"Chain aggregator discovery failed: {exc}")

        return self._store

    def _assignment_data_store(self, assignment: dict):
        """Return a GradientStore for downloading training parquets."""
        bucket = assignment.get("data_bucket", "")
        if bucket:
            try:
                from GenTRX.src.gradient_store import GradientStore
                return GradientStore(
                    endpoint_url=assignment.get("data_endpoint", ""),
                    bucket=bucket,
                    access_key=assignment.get("data_access_key", ""),
                    secret_key=assignment.get("data_secret_key", ""),
                    region=os.environ.get("GENTRX_VALIDATOR_S3_REGION", "auto"),
                    prefix=self._bucket_prefix,
                )
            except Exception as exc:
                self._tlog.warning(f"Failed to build data store from assignment fields: {exc}")
        return self._data_store

    def _download_assignment_data(self, assignment: dict) -> list[Path]:
        data_keys = assignment.get("data", [])
        data_source = assignment.get("data_source", "local")

        if not data_keys:
            return []

        if data_source == "local":
            return [Path(k) for k in data_keys if Path(k).exists()]

        data_store = self._assignment_data_store(assignment)
        if data_store is None:
            self._tlog.warning("assignment has S3 data but no data_store configured")
            return []

        if self._s3_cache_dir is None:
            self._s3_cache_dir = self.cfg.output_dir / "_s3_cache"

        bucket_id = assignment.get("data_bucket", "") or "default"
        bucket_hash = hashlib.md5(bucket_id.encode()).hexdigest()[:8]
        cache_base = self._s3_cache_dir / bucket_hash

        local_files: list[Path] = []
        for key in data_keys:
            cache_key = f"{bucket_hash}/{key}"
            if cache_key in self._s3_cached_files:
                local_files.append(self._s3_cached_files[cache_key])
                continue

            parts = key.split("/")
            if len(parts) < 5 or parts[0] != "data" or parts[3] != "intervals":
                self._tlog.warning(f"  unexpected S3 key format: {key}")
                continue
            data_uid = parts[1]
            book_id = parts[2]
            filename = parts[-1]

            local_path = cache_base / book_id / "intervals" / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = data_store.get_data(
                    int(data_uid), book_id=int(book_id), filename=filename
                )
                local_path.write_bytes(data)
                self._s3_cached_files[cache_key] = local_path
                local_files.append(local_path)
            except Exception as exc:
                self._tlog.warning(f"  S3 download failed from data bucket: {key}: {exc}")

        return local_files

    def _ensure_model_version(
        self, target: int | None = None, assignment: dict | None = None
    ) -> bool:
        store = (
            self._get_aggregator_store_for_assignment(assignment)
            if assignment is not None
            else self._store
        )
        if store is None:
            return self.model is not None

        try:
            agg_uid = self._discovered_aggregator_uid
            latest = store.get_latest_version(agg_uid)
            if target is None:
                target = latest
            elif latest > 0 and latest > target:
                self._tlog.info(f"aggregator has v{latest} (assignment says v{target}) — using latest")
                target = latest

            if target <= 0:
                return self.model is not None
            if target <= self.state.model_version:
                return True

            gtx_log.info(f"Downloading checkpoint v{target} from aggregator bucket")
            ckpt_bytes = store.get_checkpoint(agg_uid, target)
            stage_dir = self.cfg.output_dir / "ckpt_cache"
            stage_dir.mkdir(parents=True, exist_ok=True)
            tmp = stage_dir / f"gentrx_ckpt_{self.cfg.uid}.pt"
            tmp.write_bytes(ckpt_bytes)
            self._load_model(str(tmp))
            self.state.model_version = target
            gtx_log.info(f"Model loaded: v{target}")
            return True
        except Exception as exc:
            gtx_log.warning(f"Checkpoint v{target} fetch failed: {exc}")
            return False

    def _load_model(self, checkpoint: str) -> None:
        import torch
        from GenTRX.src.model import OrderModel, ModelConfig
        from GenTRX.src.tokenizer import OrderTokenizer, TokenizerConfig

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_cfg = ckpt.get("model_config", ckpt.get("config", {}))
        self.model_cfg = ModelConfig(
            **{k: v for k, v in raw_cfg.items() if k in ModelConfig.__dataclass_fields__}
        )
        tok_dict = ckpt.get("tokenizer_config")
        self.tokenizer_cfg = (
            TokenizerConfig.from_dict(tok_dict) if tok_dict else TokenizerConfig()
        )

        self.model = OrderModel(self.model_cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.tokenizer = OrderTokenizer(self.tokenizer_cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        gtx_log.info(f"GenTRX model loaded from {checkpoint} on {self.device}")

    def _clear_s3_cache(self) -> None:
        import shutil
        if self._s3_cache_dir and self._s3_cache_dir.exists():
            try:
                shutil.rmtree(self._s3_cache_dir)
                self._tlog.debug("S3 cache cleared")
            except Exception as exc:
                self._tlog.debug(f"S3 cache clear failed: {exc}")
        self._s3_cached_files.clear()
        self._s3_cache_dir = None

    def _pending_retry_count(self) -> int:
        pending_dir = self.cfg.gradient_dir / "pending"
        if not pending_dir.exists():
            return 0
        return len(list(pending_dir.glob("*.grad")))

    def _prune_gradients(self) -> None:
        """Trim the gradients/ prefix to the configured retention.

        Hot bucket only — operators wanting long history should pull
        objects to cold storage on their own cadence.
        """
        if self._write_store is None or self.cfg.keep_gradients <= 0:
            return
        try:
            n = self._write_store.prune_keep_latest(
                f"gradients/{self.cfg.uid}/",
                keep=self.cfg.keep_gradients,
                suffix=".grad",
            )
            if n:
                self._tlog.info(
                    f"pruned {n} old gradient(s), keeping latest {self.cfg.keep_gradients}"
                )
        except Exception as exc:
            self._tlog.debug(f"gradient prune failed: {exc}")
