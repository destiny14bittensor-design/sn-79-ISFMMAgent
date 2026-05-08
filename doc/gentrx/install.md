# GenTRX Install Guide

GenTRX runs on top of a standard MVTRX (SN-79) miner or validator host. This guide covers a `uv`-based install that works across most Linux distros and macOS, needs sudo only for a tiny list of system packages, and pins the Python toolchain without touching your system Python.

If you would rather follow the project's shell scripts (`install_validator.sh`, `install_miner.sh`, `install_simulator.sh`), see the [README's Install section](../../README.md#install-) first and then jump to [§4 GenTRX-specific additions](#_4-gentrx-specific-additions) for Docker, `btcli`, and `wandb`. The scripts remain supported; this doc is an alternative path with a lighter sudo surface.

> **Running on WSL2?** It works for casual development, but the validator's POSIX-SHM IPC with the simulator and query subprocess has been observed to SIGBUS under memory pressure on WSL2. Native Linux is recommended for anything longer than a quick test.

---

## 1. System packages (minimal, sudo required)

Five things. Install whichever the package manager you have calls them.

| Name | Why |
|---|---|
| `tmux` | Multiplexing for long-running panes |
| Build tools (`build-essential` / `base-devel` / `Development Tools` / Xcode CLT) | Native extensions for some Python deps |
| `git` | Clone the repo |
| `curl` | Fetch the `uv` installer |
| CA certificates | TLS trust store (usually already present) |

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y tmux build-essential git curl ca-certificates

# Fedora / RHEL / Rocky
sudo dnf install -y tmux gcc-c++ make git curl ca-certificates

# Arch / Manjaro
sudo pacman -Syu --needed tmux base-devel git curl ca-certificates

# macOS (Xcode Command Line Tools + Homebrew)
xcode-select --install
brew install tmux git
# curl and CA certs ship with macOS.
```

Nothing else here requires sudo.

---

## 2. Install `uv`

`uv` is a single binary from Astral that handles Python version installation, virtualenv creation, and dependency resolution. No system Python needed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# The installer places uv at ~/.local/bin; restart your shell or:
source ~/.bashrc   # or the equivalent for zsh / fish
uv --version
```

Full docs: https://docs.astral.sh/uv/

macOS users can alternatively install via Homebrew: `brew install uv`.

---

## 3. Python env and dependencies

One command creates the venv with the pinned Python version, another installs the project in editable mode:

```bash
cd /path/to/taos-im/sn-79

uv venv --python 3.10.9 .venv/taos
source .venv/taos/bin/activate

uv pip install -U pip
uv pip install -e .
```

`uv` will download Python 3.10.9 if your system doesn't have it. This is why the system Python version does not matter.

All the GenTRX Python deps (`boto3`, `aiobotocore`, `transformers`, `polars`, `pyarrow`, `msgpack`, `structlog`, `sortedcontainers`, `tqdm` alongside the baseline `torch`, `bittensor`, `pandas`) are in `requirements.txt` and are pulled in by `pip install -e .`. No extras flag, no manual installs for the core GenTRX runtime.

### Verify

```bash
python -c "import bittensor; print(bittensor.__version__)"
python -c "import torch, boto3, msgpack, pyarrow; print('deps ok')"
```

### Why `.venv/taos`?

The launcher (`agents/proxy/run`) resolves the venv via `$VIRTUAL_ENV` (if active) or `$TAOS_VENV` (from `.env`). Any path works; this doc uses `.venv/taos` because `.venv/` is typically gitignored and hidden for convinience. Swap for `venv/simulator` or `~/venvs/taos` if you prefer another convention.

---

## 4. GenTRX-specific additions

Three extras sit on top of the Python env. Install whichever apply.

### bittensor-cli (`btcli`), typically already installed

No GenTRX operation strictly requires `btcli`. Bucket commitments are written by `bin/setup_miner_bucket.py` and verified by `bin/gentrx_preflight`, both of which use the `bittensor` Python package directly. Anyone running a miner or validator on MVTRX (SN-79) also already has `btcli` installed because subnet registration (`btcli subnet register`) is done before GenTRX enters the picture.

If `btcli` is missing:

```bash
uv pip install bittensor-cli
btcli --version
```

### wandb (optional dashboard)

The gradient server can stream per-round metrics to wandb.ai. Soft dependency: if `wandb` is not installed or `--wandb-project` is not set, the server logs one line and runs without it. See [`wandb.md`](wandb.md).

```bash
uv pip install wandb
```

---

## 5. `taosim` (C++ simulator)

Required for any simulation run. The build needs `g++-14`, `cmake 3.29.7`, and `vcpkg`. Two paths:

### Option A: the project script

```bash
# From the repo root:
sudo ./install_simulator.sh
```

Works out of the box on Ubuntu 22.04 / 24.04. **On 22.04 it compiles g++-14 from source and takes 1 to 2 hours.** On 24.04 it uses the apt-provided g++-14 and finishes much faster. Run inside `tmux` or `screen` to survive ssh drops.

### Option B: build manually (other distros or macOS)

Refer to `install_simulator.sh` as the reference. High-level steps:

1. Install a matching `g++-14` and `cmake 3.29.7` for your OS:
   - Ubuntu 24.04: `sudo apt install g++-14 cmake`
   - Fedora / RHEL: `sudo dnf install gcc-toolset-14 cmake`
   - Arch: `sudo pacman -S gcc cmake`
   - macOS: `brew install gcc@14 cmake`
2. Clone and bootstrap `vcpkg` under `simulate/trading/vcpkg/`:
   ```bash
   cd simulate/trading
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg && git reset --hard e140b1fde236eb682b0d47f905e65008a191800f && cd ..
   ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
   ```
3. Build:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 ..
   cmake --build . -j "$(nproc)"
   ```
4. Binary lands at `simulate/trading/build/src/cpp/taosim`.

After building, either add that directory to your `PATH` or point `.env`:

```bash
# .env
TAOS_BUILD=/abs/path/to/taos-im/sn-79/simulate/trading/build
```

Verify:

```bash
taosim --version 2>&1 | head -1
```

---

## 6. Point the launchers at your venv

The launchers resolve your venv in this order:

1. An already-active venv (`$VIRTUAL_ENV`).
2. `$TAOS_VENV` from your `.env`.

**Option A** (activate before running):
```bash
source .venv/taos/bin/activate
./agents/proxy/run /path/to/simulation.xml
```

**Option B** (set once in `.env`):
```bash
# .env
TAOS_VENV=/abs/path/to/.venv/taos
```

---

## 7. `.env` (one-time)

```bash
cp .env.example .env
$EDITOR .env
```

Minimum for the **proxy test**:
```bash
TAOS_VENV=/abs/path/to/<your-venv>
TAOS_BUILD=/abs/path/to/taos-im/sn-79/simulate/trading/build
# MinIO / S3 env vars are set automatically by setup_minio at run time.
```

Minimum for a **production validator**: see [`validator_setup.md`](validator_setup.md). Requires R2 or Hippius credentials for a single validator bucket.

Minimum for a **production miner**: see [`miner_setup.md`](miner_setup.md). Requires your own R2 bucket with two tokens (write + read).

---

## 8. Smoke test before first run

```bash
python -c "import torch, bittensor, boto3, msgpack, pyarrow; print('ok')"
taosim --version 2>&1 | head -1
```

---

## What the project scripts do (for reference)

If you took the script path, here is what they installed so you know what is already on the host:

| Script | Installs | When to run |
|---|---|---|
| `install_validator.sh` | pm2, tmux, pyenv + Python 3.10.9, taos, prometheus-node-exporter, vcpkg, g++-14, cmake 3.29.7, taosim build | Validator host (once) |
| `install_miner.sh` | pm2, tmux, pyenv + Python 3.10.9, taos, copies `agents/` into `~/.taos/agents` | Miner host (once) |
| `install_simulator.sh` | vcpkg, g++-14, cmake 3.29.7, taosim build only | Simulator rebuilds |

The scripts pin Python via `pyenv` rather than `uv`, and they `sudo apt-get install` build tools, pm2 (via nvm), and prometheus-node-exporter. If you used a script and want to switch to the `uv` path later, the project tree itself is not tied to either. Activate the `uv` venv instead of the pyenv one and point `$TAOS_VENV` at it.

---

## Next steps

- Full testing guide: [`testing.md`](testing.md)
- Running the proxy test: [`../../agents/proxy/README.md`](../../agents/proxy/README.md)
- Pre-launch validation (and first-run watchpoints): [`preflight.md`](preflight.md)
