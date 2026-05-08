# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
CustomTrainingAgent — annotated template for custom GenTRX training and data
collection.

Two override points are demonstrated here.  Both have their full default logic
copied in so you can read, modify, and experiment without needing to trace into
the base class.

  train()                — one training window; runs on a background thread
                           after downloads complete.
  select_training_files() — choose which of the downloaded parquet files to
                           feed to the DataLoader.
  collect_row()          — build the row dict for one order event before it is
                           written to the local parquet buffer.

To use this template:
  1. Rename the class and this file to match your strategy.
  2. Add your trading signal to respond() below.
  3. Modify the three override methods as needed.
  4. Launch with gtx_training_enabled=true in --agent.params.

The full integration contract is in doc/gentrx/integration.md.
The data schema written by collect_row() is defined in
GenTRX/src/util/schema.py (order_stream_schema).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from taos.im.agents import GenTRXAgent

# Order-type constants and LOB types used in collect_row() overrides.
from GenTRX.src.util.schema import BID, ASK, CANCEL, LOB_DEPTH  # noqa: F401
from GenTRX.src.orderbook import LobSnapshot  # noqa: F401


class CustomTrainingAgent(GenTRXAgent):
    """Trading agent with fully custom training loop and data collection."""

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        # Opt-in guard: training is a no-op unless the operator explicitly
        # passes gtx_training_enabled=true in --agent.params.  Remove these
        # two lines if you always want training enabled.
        if not hasattr(self.config, "gtx_training_enabled"):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, "gtx_collect_data"):
            self.config.gtx_collect_data = False

        super().initialize()

        # Read any strategy-specific config here.

    # ------------------------------------------------------------------ #
    # Trading signal — customize here                                     #
    # ------------------------------------------------------------------ #

    def respond(self, state):
        # MUST call super() first: runs data collection, inference, and
        # queues training if an assignment arrived this tick.
        response = super().respond(state)

        # Add your order-placement logic here, modifying `response`.

        return response

    # ------------------------------------------------------------------ #
    # Data collection — what gets recorded per order event                #
    # ------------------------------------------------------------------ #

    def collect_row(
        self,
        book_id: int,
        ts: int,
        order_type: int,
        price_ticks: int,
        qty: float,
        snap: LobSnapshot,
        session_open_mid: int | None,
        *,
        interval_ns: int = 0,
    ) -> dict[str, Any] | None:
        """Build the row written to local parquet for one order event.

        Called once per BID, ASK, or CANCEL event when gtx_collect_data=true.
        Return the dict to buffer, or None to drop the event.

        The dict must satisfy order_stream_schema() — the fixed schema consumed
        by OrderDataset and expected by the model.  The required columns are the
        ones built below.  Extra keys are ignored at flush time; missing required
        keys raise at flush time.

        Customisation ideas:
          - Filter events by book: return None if book_id not in {12, 45}
          - Drop cancels:          return None if order_type == CANCEL
          - Normalise differently: change rel_price or volume encoding
          - Add extra columns:     they will be ignored by OrderDataset but are
                                   preserved in the parquet file for offline use
        """
        # --- Drop any event type you don't want recorded, e.g.: ----------
        # if order_type == CANCEL:
        #     return None

        # --- Filter to specific books, e.g.: -----------------------------
        # if book_id not in {12, 45, 78}:
        #     return None

        # book_id is available here for per-market branching — see commented
        # examples above.
        _ = book_id

        mid = snap.mid_price
        ask_vols = snap.ask_volumes[:LOB_DEPTH] + [0] * max(0, LOB_DEPTH - len(snap.ask_volumes))
        bid_vols = snap.bid_volumes[:LOB_DEPTH] + [0] * max(0, LOB_DEPTH - len(snap.bid_volumes))

        row: dict[str, Any] = {
            "timestamp":       ts,
            "order_type":      order_type,
            "rel_price":       price_ticks - mid if mid > 0 else 0,
            "volume_int":      int(qty),
            "volume_dec":      qty - int(qty),
            "interval_ns":     interval_ns,
            "mid_price":       mid,
            "time_of_day_s":   int((ts // 1_000_000_000) % 86400),
            "mid_price_delta": int(mid - session_open_mid) if session_open_mid else 0,
        }
        for i in range(LOB_DEPTH):
            row[f"lob_ask_vol_{i + 1}"] = float(ask_vols[i]) / self._gtx.vol_scale
            row[f"lob_bid_vol_{i + 1}"] = float(bid_vols[i]) / self._gtx.vol_scale

        return row

    # ------------------------------------------------------------------ #
    # Training — which files to use                                       #
    # ------------------------------------------------------------------ #

    def select_training_files(
        self, parquet_files: list[Path], assignment: dict | None
    ) -> list[Path]:
        """Choose which downloaded parquet files to pass to the DataLoader.

        Called at the start of train() before the dataset is built.
        Return an empty list to skip the training window entirely.

        parquet_files: all local files downloaded for this round
        assignment:    the primary assignment dict (keys: round, books,
                       ts_start, ts_end, validator_uid, data, …)

        Customisation ideas:
          - Train only on specific books (assignment["books"] lists them)
          - Skip very small files that would produce degenerate batches
          - Prefer locally-collected files over S3-fetched ones
        """
        # assignment keys: round, books, ts_start, ts_end, validator_uid, data, …
        # Use them to drive filtering logic, e.g.:
        #   books = set(assignment.get("books", [])) if assignment else set()
        _ = assignment

        # --- Filter to specific books, e.g.: -----------------------------
        # target_books = {"12", "45"}
        # return [f for f in parquet_files if f.parent.parent.name in target_books]

        # --- Skip files smaller than a minimum row count, e.g.: ----------
        # MIN_BYTES = 64 * 1024  # 64 KB
        # return [f for f in parquet_files if f.stat().st_size >= MIN_BYTES]

        return parquet_files  # default: use everything

    # ------------------------------------------------------------------ #
    # Training — the loop itself                                          #
    # ------------------------------------------------------------------ #

    def train(
        self,
        parquet_files: list[Path],
        train_model,
        assignment: dict | None = None,
    ) -> None:
        """Run one training window and upload the compressed gradient.

        Called on a background thread after select_training_files() is applied.
        train_model is a deep copy of self._gtx.model — safe to mutate in place.

        The required contract:
          - Produce a TrainingDelta (via train_window or your own loop).
          - Compress it with compress(delta, top_k_frac=...).
          - Serialize and upload via self._gtx.write_store.put_gradient().
        If you deviate from train_window(), keep the compress/serialize/upload
        block unchanged — the validator scores the gradient, not the model.

        Customisation ideas:
          - Change the optimizer or learning-rate schedule inside WindowConfig
          - Run multiple passes with different subsets of files
          - Apply data augmentation by wrapping the DataLoader
          - Use a different top_k_frac to control gradient sparsity
        """
        from GenTRX.src.dataloader import OrderDataset, ChunkSampler
        from GenTRX.src.distributed import train_window, WindowConfig
        from GenTRX.src.gradient import compress, serialize
        from torch.utils.data import DataLoader

        self._gtx.tlog.info(f"building dataset from {len(parquet_files)} files...")
        dataset = OrderDataset(
            parquet_files,
            seq_len=self._gtx.train_seq_len,
            tokenizer=self._gtx.tokenizer,
            max_cached=2,
        )
        sampler = ChunkSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=self._gtx.train_batch_size,
            sampler=sampler,
            num_workers=0,
        )
        self._gtx.tlog.info(
            f"dataset ready: {dataset.total_orders} orders, "
            f"{len(loader)} batches, training {self._gtx.train_steps} steps..."
        )

        # --- Customise WindowConfig to change optimizer / LR schedule: ---
        # win_cfg = WindowConfig(
        #     n_steps=self._gtx.train_steps * 2,   # train longer
        #     lr=self._gtx.train_lr * 0.5,          # smaller LR
        #     window_id=self._gtx.train_window_id,
        #     miner_uid=self.uid,
        # )
        win_cfg = WindowConfig(
            n_steps=self._gtx.train_steps,
            lr=self._gtx.train_lr,
            window_id=self._gtx.train_window_id,
            miner_uid=self.uid,
        )

        delta = train_window(train_model, loader, win_cfg, self._gtx.device)
        self._gtx.tlog.info(
            f"training done: loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f}"
        )

        # --- Compress: top_k_frac controls gradient sparsity (lower = smaller
        # gradient, potentially lower score; higher = larger upload) ----------
        comp = compress(delta, top_k_frac=self._gtx.top_k_frac)
        data = serialize(comp)

        # --- Upload — do not change this block unless you know what you're
        # doing; the validator reads this exact S3 path ----------------------
        if self._gtx.write_store is None:
            raise RuntimeError(
                "No S3 store configured. Set GENTRX_S3_* env vars to enable gradient upload."
            )

        round_id = (assignment or {}).get("round", self._gtx.train_window_id)
        try:
            self._gtx.write_store.put_gradient(
                miner_uid=self.uid,
                round_id=round_id,
                data=data,
            )
            self._gtx.tlog.info(f"gradient uploaded to S3 (round={round_id})")
            if self._gtx.keep_gradients > 0:
                try:
                    n = self._gtx.write_store.prune_keep_latest(
                        "gradients/", keep=self._gtx.keep_gradients, suffix=".grad"
                    )
                    if n:
                        self._gtx.tlog.info(
                            f"pruned {n} old gradient(s), keeping latest {self._gtx.keep_gradients}"
                        )
                except Exception as prune_exc:
                    self._gtx.tlog.debug(f"gradient prune failed: {prune_exc}")
        except Exception as exc:
            self._gtx.tlog.warning(f"S3 upload failed: {exc} — saving for retry")
            pending_dir = self._gtx.gradient_dir / "pending"
            pending_dir.mkdir(parents=True, exist_ok=True)
            pending_path = pending_dir / f"block_{round_id:08d}_miner_{self.uid}.grad"
            pending_path.write_bytes(data)
            self._gtx.last_gradient_path = pending_path

        self._gtx.tlog.info(
            f"window {self._gtx.train_window_id} COMPLETE | "
            f"loss {delta.metadata.loss_before:.4f} → {delta.metadata.loss_after:.4f} | "
            f"gradient {len(data)/1024:.1f} KB"
        )

        self._gtx.train_window_id += 1
