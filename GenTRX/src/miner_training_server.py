# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Miner training service — HTTP driver.

Exposes `MinerTrainingService` over FastAPI. Mirror of how
`gradient_server.py` exposes `GradientAggregator` to the validator.

HTTP endpoints:
  POST /miner/assignment   — queue an assignment for training
  GET  /miner/status       — current training state (training_in_progress, model_version, ...)
  GET  /miner/version      — service identity + model version

Run standalone:
    python -m GenTRX.src.miner_training_server \\
        --uid 7 \\
        --port 8200 \\
        --bind 127.0.0.1 \\
        --gtx-train-steps 50 --gtx-train-batch-size 8 --gtx-train-seq-len 256 \\
        --gtx-aggregator-uid 0 \\
        --subtensor-network finney --netuid 79

API key:
    --api-key (also reads $GENTRX_MINER_API_KEY). When set, every
    request must carry `X-API-Key`. Required when binding non-loopback.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request

from GenTRX.src.miner_training_service import (
    MinerTrainingConfig,
    MinerTrainingService,
)
# Reuse the validator-side API-key middleware — same auth surface, same behaviour.
from GenTRX.src.gradient_server import add_api_key_middleware


logger = logging.getLogger("GenTRX.miner_training_server")


def create_miner_router(svc: MinerTrainingService) -> APIRouter:
    """FastAPI router exposing /miner/* endpoints."""
    router = APIRouter()

    @router.post("/miner/assignment")
    async def receive_assignment(request: Request) -> dict:
        try:
            payload = await request.json()
        except Exception as exc:
            return {"status": "rejected", "reason": f"invalid JSON: {exc}"}
        return svc.submit_assignment(payload)

    @router.get("/miner/status")
    async def status() -> dict:
        return svc.get_status()

    @router.get("/miner/version")
    async def version() -> dict:
        return svc.get_version_info()

    return router


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GenTRX miner-side training service (HTTP driver)"
    )
    parser.add_argument("--uid", type=int, required=True, help="Miner UID (used in logs and gradient filenames)")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 = loopback). Use 0.0.0.0 for "
             "remote access; set --api-key when doing so.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GENTRX_MINER_API_KEY", ""),
        help="Shared secret for trading-agent → service auth. When set, all "
             "requests must include X-API-Key. Also reads $GENTRX_MINER_API_KEY.",
    )

    # MinerTrainingConfig knobs (gtx_-prefixed in --agent.params; here we
    # accept the dropped-prefix CLI form for clarity).
    parser.add_argument("--gtx-output-dir", default="data/live", type=Path)
    parser.add_argument("--gtx-gradient-dir", default=None, type=Path)
    parser.add_argument("--gtx-train-steps", type=int, default=50)
    parser.add_argument("--gtx-train-batch-size", type=int, default=16)
    parser.add_argument("--gtx-train-seq-len", type=int, default=256)
    parser.add_argument("--gtx-train-lr", type=float, default=1e-4)
    parser.add_argument("--gtx-top-k-frac", type=float, default=0.01)
    parser.add_argument("--gtx-aggregator-uid", type=int, default=0)
    parser.add_argument("--gtx-checkpoint", default=None, type=Path,
                        help="Optional local .pt to bootstrap from before chain discovery.")
    parser.add_argument(
        "--gtx-keep-gradients", type=int, default=50,
        help="Retain newest N gradients in the per-miner bucket (default 50). "
             "Set 0 to disable pruning (gradients accumulate; you handle cleanup).",
    )
    parser.add_argument(
        "--gtx-mode", default="simulation", choices=["simulation", "exchange"],
        help="Training mode shard for bucket keys (default: simulation). "
             "Combined with the connected subtensor network to form the "
             "gentrx/<network>/<mode>/ prefix. Leave at 'simulation' unless "
             "instructed otherwise.",
    )
    parser.add_argument(
        "--gtx-network", default="", dest="gtx_network",
        help="Explicit network shard for bucket keys: 'mainnet' or 'testnet'. "
             "Required when connecting to finney via a custom wss:// endpoint "
             "that is not automatically recognised. "
             "Leave empty to auto-detect from --subtensor-network.",
    )

    # Chain integration (optional — empty disables chain-based aggregator discovery)
    parser.add_argument("--subtensor-network", default=None,
                        help="e.g. 'finney', 'local', 'wss://...'")
    parser.add_argument("--netuid", type=int, default=None)

    return parser.parse_args(argv)


def _build_service(args: argparse.Namespace) -> MinerTrainingService:
    cfg = MinerTrainingConfig(
        uid=args.uid,
        output_dir=args.gtx_output_dir,
        gradient_dir=args.gtx_gradient_dir,
        train_steps=args.gtx_train_steps,
        train_batch_size=args.gtx_train_batch_size,
        train_seq_len=args.gtx_train_seq_len,
        train_lr=args.gtx_train_lr,
        top_k_frac=args.gtx_top_k_frac,
        aggregator_uid=args.gtx_aggregator_uid,
        mode=args.gtx_mode,
        network_override=args.gtx_network,
        initial_checkpoint=args.gtx_checkpoint,
        keep_gradients=args.gtx_keep_gradients,
    )
    svc = MinerTrainingService(cfg)

    if args.subtensor_network:
        if args.netuid is None:
            sys.exit("--netuid is required when --subtensor-network is set")
        import bittensor as bt
        sub = bt.Subtensor(network=args.subtensor_network)
        meta = sub.metagraph(args.netuid)
        svc.attach_subtensor(sub, meta, args.netuid)
        logger.info(
            "Chain discovery enabled: network=%s netuid=%d aggregator_uid=%d",
            args.subtensor_network, args.netuid, args.gtx_aggregator_uid,
        )

    # Best-effort bootstrap — failure here is fine, will retry on first assignment
    try:
        if svc.bootstrap_model():
            logger.info("Model bootstrapped: v%d", svc.state.model_version)
        else:
            logger.info("No checkpoint available yet — will pull on first assignment")
    except Exception as exc:
        logger.warning("Bootstrap attempt failed: %s", exc)

    return svc


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    if args.bind not in ("127.0.0.1", "localhost") and not args.api_key:
        logger.warning(
            "Service bound to %s without an API key — anyone reachable can "
            "submit assignments. Set GENTRX_MINER_API_KEY or --api-key.",
            args.bind,
        )

    svc = _build_service(args)

    app = FastAPI(title="GenTRX Miner Training Service")
    add_api_key_middleware(app, args.api_key)
    app.include_router(create_miner_router(svc))

    import uvicorn
    try:
        uvicorn.run(app, host=args.bind, port=args.port)
    finally:
        svc.shutdown(timeout=60)


if __name__ == "__main__":
    main()
