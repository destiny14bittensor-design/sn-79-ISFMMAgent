# GenTRX Testing Guide

One local test environment, plus unit tests.

| Test | Chain | Purpose | Runner |
|---|---|---|---|
| **Proxy test** | None (HTTP shortcut) | Fast iteration on the GenTRX flow without bittensor overhead | [`agents/proxy/README.md`](../../agents/proxy/README.md) |

The proxy test uses MinIO for S3 (runs locally in Docker). The runner README owns the execution detail (run command, tmux layout, stop, logs, troubleshooting); this guide covers what the test is for, what it does and does not exercise, and the common prerequisites.

---

## Common prerequisites

Full host setup (Python venv, taosim build, `.env`) lives in [`install.md`](install.md). Quick smoke check:

```bash
tmux -V                  # tmux installed
python -c "import bittensor, torch, boto3, msgpack, pyarrow; print('ok')"
taosim --help 2>&1 | head -1   # or confirm the binary under TAOS_BUILD/src/cpp
```

Copy `.env.example` to `.env` and fill in the local-paths section (`TAOS_VENV`, `TAOS_PROXY`, `TAOS_BUILD`).

---

## Proxy test

The fastest way to iterate on GenTRX. No subtensor, no wallets: `agents/proxy/proxy.py` is a stripped-down validator that talks to agents over HTTP instead of dendrite.

To run, follow [`agents/proxy/README.md`](../../agents/proxy/README.md).

### What this tests

- Full GenTRX training loop end-to-end.
- Assignment lifecycle, scoring, aggregation, checkpoint publishing.
- Agent training, gradient upload, checkpoint reload.

### What this does NOT test

- Bittensor dendrite / axon (proxy uses HTTP).
- On-chain bucket commitments (uses `LocalBucketConfig` from `miner_buckets.json`).
- Validator weight setting.

---

## Troubleshooting

Test-specific troubleshooting tables live in the runner READMEs:

- [`agents/proxy/README.md`](../../agents/proxy/README.md) (proxy: `taosim` PATH, MinIO connection, venv setup)

For pre-launch validation against any environment, run [`bin/gentrx_preflight`](preflight.md) first.
