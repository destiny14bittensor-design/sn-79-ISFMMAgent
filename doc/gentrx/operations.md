# GenTRX Operations Runbook

Production operations guide for GenTRX validators, gradient servers, and miner hosts. Covers what the setup docs leave out: process supervision, network and firewall requirements, TLS for cross-host deployments, failure-mode handling, upgrade procedures, backup strategy, and monitoring.

**Audience.** Operators standing GenTRX up on a long-running server under a process supervisor, with logs persisted off-host and a recovery story for the bucket.

**Prerequisites.** [`validator_setup.md`](validator_setup.md) (for validator + gradient-server hosts) or [`miner_setup.md`](miner_setup.md) (for miner hosts) is already complete: buckets created, on-chain commitments written, processes launching cleanly against the target network.

Topics:
- [Process supervision](#process-supervision)
- [Network / firewall](#network--firewall)
- [TLS termination](#tls-termination)
- [Failure semantics](#failure-semantics)
- [Versioning and upgrades](#versioning-and-upgrades)
- [Backups and disaster recovery](#backups-and-disaster-recovery)
- [Monitoring](#monitoring)

---

## Process supervision

GenTRX adds **one new long-running process** beyond the base validator setup: the gradient server. Both validator and gradient server should run under a process supervisor (pm2 or systemd) so they restart on crash and their logs rotate.

### pm2 (matches the existing `run_validator.sh` model)

For most operators, `run_gradients.sh` (same-machine) or `run_validator.sh -G` (all-in-one) handles pm2 lifecycle automatically — see [`validator_setup.md`](validator_setup.md). The raw commands below are for operators who manage pm2 outside the run scripts (e.g., custom startup order, separate restart policies).

```bash
# Validator (already covered by run_validator.sh)
pm2 start --name=validator "python validator.py ..."

# Gradient server - single-machine deployment (loopback)
pm2 start --name=gradient_server --time \
    "venv/simulator/bin/python -m GenTRX.src.gradient_server \
        --checkpoint /var/lib/gentrx/best.pt \
        --val-data /var/lib/gentrx/data \
        --output /var/lib/gentrx/latest.pt \
        --port 8100 \
        --bind 127.0.0.1 \
        --api-key $GENTRX_API_KEY \
        --interval 60 \
        --subtensor-network finney \
        --netuid 79 \
        --log-path /var/log/gentrx/gradient_server.log"

pm2 save
pm2 startup    # re-run once per host to enable boot resurrection
```

For multi-machine deployments, swap `--bind 127.0.0.1` for the host's public-facing interface (and read [TLS](#tls-termination) below).

### systemd (preferred for stable production hosts)

`/etc/systemd/system/gentrx-gradient-server.service`:

```ini
[Unit]
Description=GenTRX gradient server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=gentrx
WorkingDirectory=/opt/taos-im/sn-79
EnvironmentFile=/etc/gentrx/gradient_server.env
ExecStart=/opt/taos-im/sn-79/venv/simulator/bin/python -m GenTRX.src.gradient_server \
    --checkpoint /var/lib/gentrx/best.pt \
    --val-data /var/lib/gentrx/data \
    --output /var/lib/gentrx/latest.pt \
    --port 8100 \
    --bind 127.0.0.1 \
    --interval 60 \
    --subtensor-network finney --netuid 79 \
    --log-path /var/log/gentrx/gradient_server.log
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

`/etc/gentrx/gradient_server.env` contains `GENTRX_API_KEY=...` plus the six S3 env vars from [`validator_setup.md`](validator_setup.md). `chmod 600`.

```bash
systemctl daemon-reload
systemctl enable --now gentrx-gradient-server
journalctl -u gentrx-gradient-server -f
```

### Log rotation

The gradient server appends to a single `--log-path` file. Add `/etc/logrotate.d/gentrx`:

```
/var/log/gentrx/*.log {
    daily
    rotate 14
    missingok
    notifempty
    compress
    delaycompress
    copytruncate
}
```

`copytruncate` is safe because the gradient server holds the file open with append-only writes; no SIGHUP / restart needed.

### OOM handling

Gradient scoring runs forward passes on val data. With `--max-val-batches 10` (default) and the v1 model size, peak GPU memory is ~1 GB. On a CPU-only host, peak system RAM is ~3 GB. If you see OOM kills (`dmesg | grep -i kill`), the cheapest mitigations are:

- Drop `--max-val-batches` to 5 (roughly halves scoring memory; scores get slightly noisier, which is fine for small miner counts).
- Increase `--interval` so concurrent rounds don't overlap.
- Add a `MemoryHigh=` directive to the systemd unit if you want graceful pressure instead of OOM kill.

---

## Network / firewall

Required reachability:

| Source | Destination | Port | Protocol | Purpose |
|---|---|---|---|---|
| Validator | Gradient server | 8100 (default) | TCP / HTTP | Push state, pull assignments + scores |
| Gradient server | Subtensor | 9944 (or wss://) | TCP | Read miner bucket commitments |
| Gradient server | R2 / Hippius | 443 | TCP / HTTPS | Read per-miner gradients, write checkpoints + parquets |
| Miner | Validator bucket (R2/Hippius) | 443 | TCP / HTTPS | Pull checkpoints + training parquets |
| Miner | Own R2 bucket | 443 | TCP / HTTPS | Push gradients |
| Miner | Subtensor | 9944 (or wss://) | TCP | Commit bucket creds at startup |

For **single-machine deployments**, only the last 5 rows apply: the validator↔gradient-server traffic stays on loopback.

For **multi-machine deployments**:
- The gradient server's port (default 8100) must be reachable from the validator host. Restrict by source IP at the firewall level. Do not rely on `--api-key` alone (defence in depth).
- Outbound 443 to R2/Hippius is required from both the gradient server and every miner. Whitelist `*.r2.cloudflarestorage.com` if you're behind an egress proxy.

---

## TLS termination

`--api-key` authenticates the validator↔gradient-server traffic but the header travels in plain text. On loopback this is fine; on any non-loopback deployment, encrypt the channel. Three options from simplest to most involved:

### Option 1: Cloudflare Tunnel (cloudflared), recommended for most operators

No port forwarding, no certificate management, no public IP required on the gradient server host. The tunnel is outbound-only: the gradient server initiates a persistent connection to Cloudflare's edge, which the validator then reaches via a stable `*.trycloudflare.com` URL (or a custom domain on your Cloudflare zone). Fits naturally with R2 since you're already in the Cloudflare ecosystem.

```bash
# On the gradient server host, install once
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
    -o cloudflared.deb && sudo dpkg -i cloudflared.deb

# Start a quick tunnel (no login needed for trycloudflare.com)
cloudflared tunnel --url http://127.0.0.1:8100
# → prints a URL like https://random-words.trycloudflare.com
```

Keep the gradient server bound to loopback (`--bind 127.0.0.1`). On the validator host, set `--gentrx.gradient_server_url` to the printed URL.

For a stable domain (recommended for production): authenticate cloudflared with your Cloudflare account (`cloudflared tunnel login`), create a named tunnel, and point a DNS CNAME at it. The URL becomes deterministic.

```bash
# Run cloudflared under pm2 alongside the gradient server
pm2 start --name=cloudflared "cloudflared tunnel run <tunnel-name>"
pm2 save
```

Keep `--api-key` set. cloudflared provides encryption, the key provides authentication. Both are needed.

### Option 2: Tailscale / WireGuard

If both validator and gradient server live in the same Tailscale tailnet, the network is already encrypted point-to-point. Bind the gradient server to the Tailscale interface (`--bind 100.x.x.x`) and skip the reverse proxy. Still keep `--api-key` set as defence-in-depth.

### Option 3: nginx with LetsEncrypt

For operators who prefer a traditional reverse proxy and have a public IP with a DNS record:

```nginx
server {
    listen 443 ssl http2;
    server_name gentrx.example.com;

    ssl_certificate     /etc/letsencrypt/live/gentrx.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gentrx.example.com/privkey.pem;

    # State POSTs can be a few hundred KB; assignments + scores are tiny.
    client_max_body_size 4M;

    location /gentrx/ {
        proxy_pass         http://127.0.0.1:8100;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-API-Key         $http_x_api_key;
        proxy_read_timeout 60s;
        proxy_buffering    off;
    }
}
```

Bind the gradient server to loopback (`--bind 127.0.0.1`); only nginx hits it directly. Validator points at `https://gentrx.example.com/gentrx`.

---

## Failure semantics

What happens when each piece fails. Helps oncall know what to ignore vs. what to escalate.

### Gradient server is down

- **Validator side.** `push_state` POSTs fail; logged at debug (`[GTX] state POST failed`). State is dropped; the gradient server tolerates gaps when it comes back up. Assignment polls also fail; miners continue running their last assignment until the next poll succeeds.
- **Miner side.** No immediate impact. Miners only depend on the gradient server for new assignments and checkpoint discovery.
- **Recovery.** When the gradient server comes back it reads the latest checkpoint from R2 (where it was published before the crash) and resumes the next round. No state migration needed.

**Action**: alert if the gradient server is down for > 5 minutes during sim hours.

### Chain RPC is down

- Gradient server's `_refresh_miner_buckets` retries on a 30s cooldown. Delivered assignments using cached miner buckets continue to work.
- New miner registrations don't get picked up until the chain comes back.

**Action**: usually transient. Wait 5 minutes before paging.

### Validator bucket write fails mid-round

- Checkpoint write: `_aggregate_round` logs `Failed to upload checkpoint`. The round is effectively dropped: `_version` doesn't advance, miners keep training against the current model. Idempotent on retry.
- Parquet write: `_flush_book_parquet` logs an error. The parquet isn't in the in-memory `_written_parquets` registry, so subsequent assignments won't reference it. Some training data is lost for that book, but scoring continues against whatever did write.

**Action**: investigate R2 status / quota / IAM if failures persist more than two consecutive rounds.

### Miner registers mid-round

- The gradient server's chain-commitment cache refreshes on a 30s cooldown. The new miner's bucket appears at the next refresh.
- The validator's metagraph sync picks up the new miner on its own cadence (typically 12 blocks / ~150s on finney).
- First round including the new miner is the one created **after** both the cache refresh AND the metagraph sync. Practically: ~1-2 rounds of delay.

### Miner deregisters / loses uid

- Subsequent assignment delivery to that uid fails (axon dead) and is silently skipped at the validator dendrite layer.
- The gradient server's cached bucket entry becomes stale; it'll keep trying to read gradients from a bucket whose key no longer exists. After the 30s cache refresh, the entry is dropped if the new uid-holder hasn't committed yet.

### Simulation ends and a new one begins

Handled automatically. The simulator emits a `SimulationEndEvent` (type `ESE`) on the current sim's last tick and a `SimulationStartEvent` (`ESS`) on the next sim's first tick. Both travel to the gradient server inside the msgpack state packet via `state_packager`. On receipt:

1. **On `ESE`**: the gradient server drops every in-memory tick buffer (`_pending_rows`, `_pending_interval_start`, `_written_parquets`, per-book matching engines) so sim-A tail rows can't contaminate the sim-B stream. Up to 5 minutes of sim-time at the sim tail are discarded; the full simulator logs retain them. A cleanup flag is set for the aggregation loop.
2. **On the next aggregation-loop tick** (every 5 s): the stale `data/` prefix on the validator bucket is deleted via batched `delete_objects`. Runs in well under the new sim's grace-plus-window period, so sim-B's first parquet flush never races with sim-A's leftovers. Logs as `[GTX] Sim transition cleanup: removed N parquets`.
3. **Checkpoints, proposals, round counters, and model version are preserved**. Training continuity across sims is the point.
4. **Fallback**: if a simulator forgets to emit `ESE` (crash, forced restart), the heuristic detection path (sim-time going backwards, or a new `simulation_id` on the first packet) triggers the same reset and queues the same cleanup.

No operator intervention needed. In-flight sim-A assignments complete normally: miners already have their parquets on local disk at assignment-delivery time, so S3 deletion doesn't affect them.

---

## Versioning and upgrades

### Checkpoint format

Every `.pt` checkpoint embeds `model_config` and `tokenizer_config` as plain dicts. `train.load_checkpoint` reads them back and reconstructs the configs at load time.

**Known limitation**: the agent's `_load_model` builds an `OrderModel` with whatever dims the checkpoint specifies, but **does not verify** that those dims match the agent's local `ModelConfig` defaults. If the operator ships a checkpoint with different `d_model` / `n_layers` and miners download it, training will work, but the gradient produced has the checkpoint's shape, not the local defaults. This is fine in practice because the gradient is then applied back to the same checkpoint by the gradient server.

What can go wrong: if a miner has cached an *old* checkpoint locally and the operator ships a checkpoint with a different config, the miner needs to reload from S3 before the new shape is in use. The current code does this whenever `assignment.model_version > self._model_version`, so the path is covered. Just be aware that miners running mid-rollover may briefly train on the old config.

**Recommended rollout**:
1. Stage the new checkpoint to a non-production R2 bucket first.
2. Run a localnet test with the new checkpoint as the seed.
3. Bump `--gentrx.checkpoint` on the gradient server, restart it. The first round it publishes will roll the version forward. Miners pick it up via `model_version` on their next assignment.

### Code upgrades (validator + gradient server)

Both processes can be restarted independently:
- **Restart validator**: misses a few state ticks. Gradient server swallows the gap.
- **Restart gradient server**: `_written_parquets` registry is empty until state arrives again; it falls back to listing the validator bucket on first call. Validator's POSTs fail until the server is back; in-flight assignments still resolve from the validator bucket parquets.

For coordinated upgrades (breaking protocol changes), publish a `model_version` bump and require miners to redeploy by a deadline. The current protocol has no version negotiation; adding it is a Phase 3 item.

### Code upgrades (miners)

Per-miner. Backwards-compatible miner upgrades roll out asynchronously. For breaking changes, announce a deadline and the validator's chain-based discovery silently drops miners that fall off the new protocol.

---

## Backups and disaster recovery

Three categories of state, three different stories:

### 1. `aggregator/checkpoints/`: irreplaceable

This is the actual training output. If R2 loses the bucket, the project resets to the seed checkpoint.

**Backup strategy**:
- Enable **R2 versioning** on the bucket; keeps prior versions of each object for ~30 days. Cheapest layer.
- Cron a daily `aws s3 sync` (or `rclone copy` / `mc mirror`) to a second R2 bucket in a different region (or to local cold storage). Each `.pt` is ~140 MB; one sync per day is ~50 GB/year retained, trivial cost.
- Document the restore procedure: copy the latest `v*.pt` back into the primary bucket, restart the gradient server. The next round resumes from that version.

### 2. `data/<validator-uid>/{book}/intervals/*.parquet`: reproducible

Recreated by re-running the simulator. Don't bother backing up. Do note that wiping the data bucket while the gradient server is running mid-session causes orphaned references in already-issued assignments (miners get 404s on their parquet downloads). Coordinate any cleanup with a gradient server restart.

### 3. `gradients/<miner-uid>/{round_id:08d}.grad`: reproducible

Each miner re-uploads on the next assignment. No backup needed. Per-miner buckets do NOT need versioning enabled (would just waste storage).

---

## Monitoring

### Watch training progress: what a healthy run looks like

Three signals tell you if training is working. Check them in order:

**1. Aggregation completing.** Grep `aggregation.jsonl`:
```bash
tail -f data/localnet_test/aggregation.jsonl | jq -c 'select(.type=="aggregation")'
```
Healthy: a new `{"type":"aggregation", "round":N, "n_accepted":>=1, "loss_before":X, "loss_after":Y}` entry every `blocks_per_round` blocks. `loss_after < loss_before` means a round improved the model.

**2. Model version rolling.** `v00001`, `v00002`, `v00003` ... in the validator bucket's `checkpoints/` prefix. Each successful aggregation creates a new version. Version stuck at `v00001` for > 2 rounds means aggregation is not completing.

**3. Aggregation duration within budget.** Grep `aggregate_round` in the gradient server log:
```
[GTX] aggregate_round=N: n_pending=K t_load=X.Xs t_score=X.Xs t_aggregate=X.Xs t_total=X.Xs
```
`t_total` should be < `blocks_per_round × ~12s` (on mainnet). If consistently over, reduce `--max-val-batches` or raise the validator's `--gentrx.blocks_per_round`.

### Per-miner signals (wandb dashboard)

The `training/*` namespace answers "is the global model improving":
- `training/val_loss`: go-to chart. Trend should be monotonically decreasing.
- `training/val_loss_delta`: per-round improvement. Bars mostly below zero indicate the model is converging.
- `training/accept_rate`: fraction of miners whose delta was accepted. Stable or rising indicates a healthy network.
- `training/model_version`: strict monotonic increase indicates the aggregator is alive.
- `training/rolled_back`: spikes of 1 mean something published a worse delta. Isolated spikes are normal; sustained spikes are a problem.

The `miners/*` roll-up answers "are miners behaving":
- `miners/best_score`, `miners/median_score`, `miners/worst_score`: distribution of per-miner quality.
- `miners/n_overfitting`: how many were overfit-penalized this round.
- `miners/n_scored`: active miners in the round.

### Tuning knobs when aggregation lags

| Knob | Default | If `t_total` too high |
|---|---|---|
| `--max-val-batches` | 10 | Drop to 5 (~2× speedup, slightly noisier scores) |
| `--gentrx.blocks_per_round` (validator) | 25 | Raise to 50+ (more budget per round, slower training). |
| parallel scoring | off (sequential) | Not available; scoring runs serially per round |

### Built-in log inspector

```bash
venv/simulator/bin/python bin/gentrx_inspect --watch
```

Reads `aggregation.jsonl` (path is `<gradient-server-output-dir>/aggregation.jsonl`). Each event carries a `sim_id` and `sim_epoch` so multiple sim runs in the same log file render as separate entries (`--list` to see all, default view shows the most recent).

### Useful log greps

```bash
# Round summary lines
grep "round=.*aggregated" /var/log/gentrx/gradient_server.log

# Sim-bind events (when each new sim attached)
grep "Bound to sim_id" /var/log/gentrx/gradient_server.log

# Chain commitment refreshes (only fires when count > 0; empty results stay at debug)
grep "Retrieved.*miner bucket commitments" /var/log/gentrx/gradient_server.log

# Validator-side delivery success rate
# GenTRX records flow through bt.logging - grep the validator stream
# (pm2 logs / journalctl / tee / whatever you use) for the [GTX] prefix:
grep "\[GTX\] delivering to uids" /var/log/validator.log | tail -50
```

### Wandb dashboard

Optional live dashboard on wandb.ai. Mirrors every aggregation event plus per-miner scores to a web UI. Soft-dependency: if `wandb` isn't installed or no project is configured, the gradient server runs without it.

Full setup (install, env vars, metrics, privacy, offline mode, troubleshooting) lives in [`wandb.md`](wandb.md).

### Monitoring gaps

- **Prometheus metrics on the validator side are not yet exposed.** The gradient server returns an enriched payload on `GET /gentrx/scores` (round summary, per-miner scores, aggregation timings, rollback counter). If you need metrics, poll that endpoint and parse directly until Prometheus wiring lands.
- **No structured alerts** on consecutive aggregation failures or miner-bucket read failures. Wrap your process supervisor's failure notifier for coverage there.
