# Weights and Biases dashboard

Wandb (Weights & Biases) is the **optional observability layer** for the gradient server. When enabled, every aggregation event - plus per-miner scores and the loss-delta trend - streams to a wandb project you control, giving operators a live web UI without running any extra infrastructure.

Skip this whole page if you only want file-based monitoring (`aggregation.jsonl` + `[GTX]` grep in the validator stream are always on).

---

## When to enable it

- **You operate the central aggregator** → strong yes. The dashboard makes it obvious when training stalls, whether rollbacks are firing, and which miners are drifting.
- **You run a sibling validator** → optional. Same metrics, but siblings skip the "rolled_back" signal.
- **You run miners only** → skip; wandb is only wired into the gradient server, not the agent.

---

## Setup

### 1. Get a wandb account + API key

1. Sign up at <https://wandb.ai> (free personal plan is fine).
2. Create a project, e.g. `gentrx-mainnet`. Keep it **private** - see [Privacy](#privacy-what-lands-on-wandbai) below.
3. Copy your API key from <https://wandb.ai/authorize>.

### 2. Install the client

```bash
venv/simulator/bin/pip install wandb
```

Soft-dependency: if `wandb` isn't installed, the gradient server logs a single line `wandb module not found - dashboard disabled` and keeps going. Preflight reports it as a WARN, not FAIL.

### 3. Configure env vars

Add to the gradient server host's `.env`:

```bash
WANDB_API_KEY=<your-key>            # or keep it in ~/.netrc
WANDB_PROJECT=gentrx-<env-name>     # e.g. gentrx-mainnet, gentrx-localnet
WANDB_RUN_NAME=<optional>           # defaults to wandb auto-generated name
```

All three can also be passed as CLI flags:

```bash
python -m GenTRX.src.gradient_server \
    --wandb-project gentrx-mainnet \
    --wandb-run-name aggregator-primary \
    ...
```

### 4. Launch

Start the gradient server the usual way. On first launch you'll see a `wandb init` line pointing at the run URL. Follow it - the project page should populate within a round.

---

## What gets logged

One run per gradient-server process, tagged `aggregator` or `sibling`. Metrics are grouped so the UI's auto-chart layout stays readable:

| Namespace | What |
|---|---|
| `training/` | `val_loss`, `val_loss_delta`, `model_version`, `round`, `accept_rate`, `n_scored`, `n_accepted`, `rolled_back` |
| `miners/` (roll-up) | `best_score`, `median_score`, `worst_score`, `n_scored`, `n_overfitting` |
| `miners/<uid>/` (per-miner detail) | `score`, `score_own`, `score_held`, `overfitting` - **only when `WANDB_VERBOSE=true`** |
| `server/` | `running`, startup config snapshot |

All validators pointed at the same `WANDB_PROJECT` show up side-by-side; filter by the `aggregator` / `sibling` tag to isolate one.

---

## Offline mode

For air-gapped hosts or intermittent connectivity:

```bash
export WANDB_MODE=offline
# run the gradient server as usual; events go to wandb/offline-run-*
# later, from any host with network:
wandb sync wandb/offline-run-*
```

---

## Privacy - what lands on wandb.ai

By default wandb captures much more than the explicit metrics: hostname, git commit/branch/remote, every `.py` file in the repo, `pip freeze`, and a mirror of the gradient server's stdout/stderr (which includes axon IPs, wallet names, and per-round event lines). If the project is ever made public, all of that becomes visible.

`GenTRX/src/wandb_ops.py` opts out at init time:

- `settings.console="off"` - no stdout/stderr mirror
- `settings.save_code=False`, `disable_code=True` - no source code upload
- `settings.disable_git=True` - no git commit / remote / diff

What still lands on the run page:

- **Hostname** (can't fully suppress; set a generic `WANDB_RUN_NAME` if you want to control the run's visible title)
- **pip freeze** + OS info (inventory only)
- **System metrics** (CPU / GPU / memory utilisation)
- **Our explicit metrics** (`training/*`, `miners/*`, …)

Override at your own risk:

```bash
WANDB_CONSOLE=auto            # re-enables stdout capture
WANDB_DISABLE_CODE=false      # re-enables code upload
WANDB_DISABLE_GIT=false       # re-enables git metadata
```

Project visibility is controlled in the wandb UI - private by default. **Don't flip to public without reviewing the run contents first.**

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `wandb module not found - dashboard disabled` | `wandb` not installed in the venv | `venv/simulator/bin/pip install wandb` |
| Init hangs / retries `localhost:8080` | Stale `~/.netrc` entry from an earlier `wandb local` install | Edit / delete the `machine localhost` block |
| `protobuf VersionError: gencode 6.32.1 runtime 6.31.1` | Version drift between wandb's protobuf and another package | `pip install --upgrade protobuf` or downgrade wandb |
| "Invalid entity / project" | `WANDB_PROJECT` doesn't exist under your account | Create it in the wandb UI first |
| Run appears but metrics are missing | `WANDB_VERBOSE` off (per-miner detail is off by default) | Export `WANDB_VERBOSE=true` if you need per-miner panels |
