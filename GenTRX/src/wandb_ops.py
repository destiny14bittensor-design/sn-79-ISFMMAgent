# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Wandb integration for GenTRX gradient server.

Mirrors every aggregation event (already written to aggregation.jsonl) to a
Weights & Biases run. Soft dependency — import failure or missing project
config means the module is silently disabled; the gradient server runs
fine without it.

Enable by either:
  - Passing `--wandb-project <name>` to `python -m GenTRX.src.gradient_server`
  - Setting `WANDB_PROJECT=<name>` in the environment

The usual `WANDB_API_KEY` still governs auth. Offline runs (`WANDB_MODE=offline`)
are fine — wandb buffers to disk until the host is online.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


_wandb: Any = None          # the wandb module, only set when active
_enabled: bool = False


def init_wandb(
    project: str | None = None,
    run_name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Initialise a wandb run. Returns True when a run was created, False
    otherwise (not installed, no project, init failure). Never raises.

    Called once at gradient server startup; subsequent calls no-op.
    """
    global _wandb, _enabled

    if _enabled:
        return True

    project = project or os.environ.get("WANDB_PROJECT", "")
    if not project:
        logger.debug("wandb: no project configured, skipping")
        return False

    try:
        import wandb
    except ImportError:
        logger.info("wandb not installed — dashboard disabled (pip install wandb)")
        return False
    except Exception as exc:
        # Covers protobuf VersionError and similar broken-install states.
        # wandb.log calls from the broken module would raise too, so we
        # disable the whole integration cleanly here.
        logger.warning(
            "wandb import failed (%s: %s) — dashboard disabled. "
            "Try: pip install -U wandb protobuf",
            type(exc).__name__, exc,
        )
        return False

    # Privacy-hardened settings. Wandb captures a lot by default (hostname,
    # git info, full source code, stdout/stderr of the whole process) — all
    # of which ends up visible on the run page. For GenTRX runs that may
    # eventually be shared with a wider audience (subnet dashboards, public
    # project comparisons), we opt out of the automatic capture and keep
    # only the metrics we explicitly call wandb.log(...) with.
    #
    # Override via env vars at operator's risk:
    #   WANDB_CONSOLE=auto           (re-enables stdout/stderr capture)
    #   WANDB_DISABLE_CODE=false     (re-enables code upload)
    #   WANDB_DISABLE_GIT=false      (re-enables git commit/remote capture)
    try:
        settings_kwargs = {
            "console": os.environ.get("WANDB_CONSOLE", "off"),
            "save_code": False,
            "disable_code": os.environ.get("WANDB_DISABLE_CODE", "true").lower() == "true",
            "disable_git": os.environ.get("WANDB_DISABLE_GIT", "true").lower() == "true",
        }
        wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            reinit=True,
            settings=wandb.Settings(**settings_kwargs),
        )
    except TypeError:
        # Older wandb versions may not accept all settings kwargs; fall
        # back to a plain init with no settings override rather than
        # failing entirely.
        try:
            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
            logger.warning(
                "wandb: privacy settings not supported by this wandb version — "
                "stdout/code/git capture may be on. Upgrade wandb to opt out."
            )
        except Exception as exc:
            logger.warning("wandb init failed: %s — continuing without dashboard", exc)
            return False
    except Exception as exc:
        logger.warning("wandb init failed: %s — continuing without dashboard", exc)
        return False

    _wandb = wandb
    _enabled = True
    logger.info("[GTX] wandb active: project=%s run=%s", project, getattr(wandb.run, "name", "?"))
    return True


def log_event(event: dict) -> None:
    """Map a gradient-server event (same payload as aggregation.jsonl) to
    wandb metrics. No-op when wandb is not active.

    The existing event shapes are:
      - {type: server_start, interval}
      - {type: sim_bind}
      - {type: aggregation, round, n_scored, n_accepted, version,
         loss_before?, loss_after?, rolled_back?, sibling_only?}
      - {type: gradient_score, round, miner, window, score,
         score_own, score_held, overfitting}
    """
    if not _enabled or _wandb is None:
        return

    try:
        etype = event.get("type", "")

        # The "is training working" dashboard lives in training/*.
        # Per-miner detail in miners/* as a collective roll-up (median / best
        # / worst) to avoid N charts per round. Verbose per-miner detail is
        # gated behind WANDB_VERBOSE=true for operators who want it.
        verbose = os.environ.get("WANDB_VERBOSE", "").lower() == "true"

        if etype == "aggregation":
            n_scored = event.get("n_scored", 0) or 0
            n_accepted = event.get("n_accepted", 0) or 0
            metrics: dict[str, Any] = {
                "training/round": int(event.get("round", 0)),
                "training/model_version": event.get("version", 0),
                "training/accept_rate": (
                    n_accepted / n_scored if n_scored > 0 else 0.0
                ),
                "training/n_scored": n_scored,
                "training/n_accepted": n_accepted,
            }
            if event.get("loss_after") is not None:
                metrics["training/val_loss"] = event["loss_after"]
                if event.get("loss_before") is not None:
                    metrics["training/val_loss_delta"] = (
                        event["loss_after"] - event["loss_before"]
                    )
            if event.get("rolled_back"):
                metrics["training/rolled_back"] = 1
            _wandb.log(metrics)
            # Reset per-round collector for miner_score aggregation below.
            _reset_round_scores(int(event.get("round", 0)))

        elif etype == "gradient_score":
            round_id = int(event.get("round", 0))
            miner = event.get("miner", -1)
            score = event.get("score", 0.0) or 0.0
            _collect_miner_score(round_id, miner, score, event.get("overfitting", False))
            # Emit per-miner detail only in verbose mode.
            if verbose:
                metrics: dict[str, Any] = {
                    f"miners/{miner}/score": score,
                    f"miners/{miner}/overfitting": 1 if event.get("overfitting") else 0,
                }
                if event.get("score_own") is not None:
                    metrics[f"miners/{miner}/score_own"] = event["score_own"]
                if event.get("score_held") is not None:
                    metrics[f"miners/{miner}/score_held"] = event["score_held"]
                _wandb.log(metrics)
            # The roll-up is flushed exactly once per round, when the
            # aggregation event fires and _reset_round_scores forces it.
            # Flushing here too would log partial median/best/worst as
            # scores trickle in — wandb shows the last value per step,
            # so you'd see a jagged chart up to the true cohort value.

        elif etype == "server_start":
            _wandb.log({"server/running": 1})

        # Intentionally drop noisy events that pollute the dashboard:
        #   gradient_received, round_delivered, miner_buckets_refresh, sim_bind
        # They remain in aggregation.jsonl for log-level debugging.

    except Exception as exc:
        logger.debug("wandb log_event failed: %s", exc)


# Per-round miner-score collector for the miners/* roll-up metrics.
_current_round_scores: dict[int, list[tuple[int, float, bool]]] = {}


def _collect_miner_score(round_id: int, miner: int, score: float, overfitting: bool) -> None:
    _current_round_scores.setdefault(round_id, []).append((miner, score, bool(overfitting)))


def _reset_round_scores(new_round: int) -> None:
    # Flush any pending buckets (miners scored but not yet rolled up) first.
    for rnd in list(_current_round_scores.keys()):
        if rnd < new_round:
            _maybe_flush_miner_roll_up(rnd, _wandb, force=True)
    _current_round_scores.setdefault(new_round, [])


def _maybe_flush_miner_roll_up(round_id: int, wandb_mod: Any, force: bool = False) -> None:
    scores = _current_round_scores.get(round_id)
    if not scores:
        return
    # Only flush once per round — either on `force`, or when caller signals
    # "everyone's in" (currently we flush eagerly on each score, which means
    # median/best/worst update progressively during the round; final value
    # at round end is the true one).
    sorted_scores = sorted(s for _, s, _ in scores)
    n = len(sorted_scores)
    median = sorted_scores[n // 2]
    wandb_mod.log({
        "miners/n_scored": n,
        "miners/best_score": sorted_scores[-1],
        "miners/worst_score": sorted_scores[0],
        "miners/median_score": median,
        "miners/n_overfitting": sum(1 for _, _, of in scores if of),
    })
    if force:
        _current_round_scores.pop(round_id, None)


def finish_wandb() -> None:
    """Close the wandb run. No-op if not active."""
    global _enabled, _wandb
    if not _enabled or _wandb is None:
        return
    try:
        _wandb.finish()
    except Exception:
        pass
    _enabled = False
    _wandb = None


def is_active() -> bool:
    return _enabled
