# GenTRX preflight

`bin/gentrx_preflight` is a read-only launcher-side validator. It walks through every manual setup step and reports, for each, whether the host is ready to run. _Safe to re-run_. It never commits to the chain, never writes to S3 beyond a small self-cleaned probe object.

Run it **before** starting a fresh deployment, or any time a config file or env var changes. It catches 90% of the "why isn't it working" questions before the process ever starts.

---

## What it checks

Grouped by category. Required items fail the preflight; optional items only warn.

| Group | Checks |
|---|---|
| **Python + deps** | Python version, required packages (torch, bittensor, httpx, msgpack, boto3, fastapi, uvicorn), optional packages (wandb) |
| **Subtensor** | RPC reachable, block number advances, netuid resolvable, metagraph readable |
| **Wallet** | Coldkey + hotkey files present, wallet readable, registration on the target netuid, validator permit (validators only) |
| **GenTRX env vars** | `GENTRX_VALIDATOR_S3_*` for validators, `GENTRX_AGENT_S3_*` for miners, optional `GENTRX_API_KEY` for cross-machine setups |
| **S3 bucket** | Endpoint reachable, credentials work, write probe succeeds and gets cleaned up, read probe returns the written bytes |
| **Chain commitment** | Your bucket's read credentials are committed on-chain (miners + validators) and match the env vars |
| **Gradient server HTTP** | Gradient server reachable, `GET /gentrx/version` returns 200 (with the configured API key if set) |
| **taosim binary** | Compiled binary exists and is executable (`TAOSIM_BIN` / default path) |
| **Simulation XML** | Configured XML file exists and parses |
| **Ports** | Configured ports (`--axon.port`, gradient server port) are free on localhost |
| **Disk** | Target data + checkpoint directories exist and have room |

---

## Usage

Each `--env` value selects sensible defaults for `--chain-endpoint` and (for localnet / mainnet) `--netuid`. The chain endpoint is a bittensor SDK shortcut (`local`, `test`, `finney`); the SDK resolves it to the canonical websocket URL at connect time, so the default follows upstream changes automatically. Override either default with the matching flag or the `SUBTENSOR_ENDPOINT` / `NETUID` env vars (an explicit `wss://…` URL is also accepted).

| `--env` | Meaning | Chain shortcut | Default netuid |
|---|---|---|---|
| `localnet` | Local subtensor running on this host (Docker devnet-ready container or equivalent) | `local` | `1` |
| `testnet` | Public bittensor testnet | `test` | `79` |
| `mainnet` | Production bittensor (subnet 79) | `finney` | `79` |

Pass `--netuid <n>` if your testnet subnet registers under a number other than 79.

Examples
```bash
# Validator on mainnet, GenTRX enabled
bin/gentrx_preflight \
    --role validator \
    --env mainnet \
    --gentrx-enabled \
    --wallet-name <your-coldkey> \
    --wallet-hotkey <your-hotkey>

# Miner on mainnet
bin/gentrx_preflight \
    --role miner \
    --env mainnet \
    --wallet-name <your-coldkey> \
    --wallet-hotkey <your-hotkey>

# Miner on testnet
bin/gentrx_preflight \
    --role miner \
    --env testnet \
    --wallet-name <your-coldkey> \
    --wallet-hotkey <your-hotkey>

# Miner on localnet
bin/gentrx_preflight \
    --role miner \
    --env localnet \
    --wallet-name miner0

# Gradient server on localnet
bin/gentrx_preflight \
    --role gradient-server \
    --env localnet \
    --wallet-name validator0
```

All flags also read from env vars so you can call it from a launcher script without re-specifying everything. See `bin/gentrx_preflight --help` for the complete list.

The mainnet and testnet branches run the strict checks (validator permit, on-chain bucket commitment). The localnet branch relaxes both because the devnet-ready chain has `SubtokenDisabled` (no real staking) and miners typically have not committed buckets at that stage.

Exit codes:

- `0` - all required checks passed (warnings allowed)
- `1` - at least one required check failed
- `2` - preflight crashed internally (report please)

---

## Typical output

```
[PASS] python 3.10.9 ≥ 3.10
[PASS] torch 2.5.1 importable
[PASS] bittensor 10.2.0 importable
[WARN] wandb absent - dashboard disabled (optional)
[PASS] subtensor ws://localhost:9944 block=12345
[PASS] wallet validator/default coldkey+hotkey readable
[PASS] registered at UID 1 on netuid 1
[PASS] GENTRX_VALIDATOR_S3_* complete (6/6 vars)
[PASS] S3 bucket gentrx-localnet write+read probe OK
[PASS] chain commitment at UID 1 matches GENTRX_VALIDATOR_S3_*
[PASS] gradient server http://127.0.0.1:8100/gentrx version=3
[PASS] taosim /path/to/taosim executable
[PASS] simulation XML /…/simulation.xml parses
[PASS] axon port 8090 free
[PASS] checkpoints dir /…/checkpoints writable

14 passed · 1 warning · 0 failed - ready.
```

A single `[FAIL]` is enough to abort. Read the message, fix, rerun.

---

## Integrating into launchers

Production launcher scripts should invoke preflight before starting any long-lived processes. Running preflight adds ~5 seconds and catches misconfig before any real cost is paid.

---

## After preflight passes - first-run watchpoints

Preflight catches config-time errors. The following are runtime signals to confirm on the first round after starting validator + gradient server:

1. Gradient server log: `Bound to sim_id=<x>` and `Retrieved N miner bucket commitments` (N > 0).
2. Validator stream (pm2 logs / journalctl): `[GTX] round=0 v=N: M/M miners have data`.
3. After one round: `aggregation.jsonl` has a fresh entry with `n_accepted > 0`.
4. `bin/gentrx_inspect --list` shows one row for the new `sim_id`.

If any of these stalls > 5 minutes, see [`operations.md` "Failure semantics"](operations.md#failure-semantics).

