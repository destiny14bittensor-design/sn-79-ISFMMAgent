#!/bin/bash
set -e
# run_validator.sh — launch MVTRX validator + simulator
#
# GenTRX distributed training (optional):
#   -G              enable as sibling validator (default; scores and proposes)
#   -G aggregator   enable as uid-0 aggregator (publishes canonical model; internal)
#   -Q <url>        gradient server URL — skip auto-start and use this address
#                   (e.g. http://gpu-host:8100/gentrx for a remote GPU machine)
#
# First run: prompts interactively for bucket credentials and gradient server
# choice; saves all answers to .env.
#
# Subsequent update runs: run without -G — the saved mode is restored from .env,
# all prompts are skipped, and the gradient server is restarted automatically.
# Pass -G to override the saved mode, or -Q to change the gradient server URL.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_PATH=~/.bittensor/wallets/
WALLET_NAME=taos
HOTKEY_NAME=validator
NETUID=79
CHECKPOINT=0
LOG_LEVEL=info
PD_KEY=""
PROM_PORT=9001
TIMEOUT=3.0
SIMULATION_CONFIG=simulation_0
PRESERVE_SIMULATOR=0
USE_TMUX=1
GENTRX_MODE=""    # aggregator | sibling  (empty = GenTRX disabled)
GRAD_URL=""       # explicit gradient server URL; empty = auto-start or prompt
GRAD_PORT=8100    # gradient server listen port (local or remote)
# GRAD_CORES_COUNT: reserve the last N CPU cores exclusively for the gradient
# server (CPU-only mode).  0 = no pinning (default; GPU mode or manual control).
# 8 is comfortable for CPU scoring; 4 is the minimum within 5-min round windows.
# Set in .env or export before running.  Validator + simulator are pinned to the
# remaining cores automatically when this is > 0.
GRAD_CORES_COUNT="${GRAD_CORES_COUNT:-0}"

# Track explicit CLI flags so they override saved config
_EXPLICIT_GRAD_URL=0

[ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"

# Normalize -G: if passed without a recognised mode value, default to 'sibling'.
# This allows `./run_validator.sh -G` without an explicit 'sibling' argument.
# Only `-G aggregator` (or its aliases a/agg) selects aggregator mode.
_norm_args=()
_ia=1
while [ "$_ia" -le "$#" ]; do
    _arg="${!_ia}"
    if [ "$_arg" = "-G" ]; then
        _ia=$(( _ia + 1 ))
        _nxt="${!_ia:-}"
        case "$_nxt" in
            aggregator|agg|a) _norm_args+=("-G" "$_nxt"); _ia=$(( _ia + 1 )) ;;
            sibling|sib|s)    _norm_args+=("-G" "sibling"); _ia=$(( _ia + 1 )) ;;
            *)                _norm_args+=("-G" "sibling") ;;  # default; don't consume _nxt
        esac
    else
        _norm_args+=("$_arg"); _ia=$(( _ia + 1 ))
    fi
done
set -- "${_norm_args[@]+"${_norm_args[@]}"}"

while getopts e:p:w:h:u:l:d:o:t:g:s:x:c:G:Q: flag; do
    case "${flag}" in
        e) ENDPOINT=${OPTARG};;
        p) WALLET_PATH=${OPTARG};;
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        u) NETUID=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
        d) PD_KEY=${OPTARG};;
        o) PROM_PORT=${OPTARG};;
        t) TIMEOUT=${OPTARG};;
        g) SIMULATION_CONFIG=${OPTARG};;
        s) PRESERVE_SIMULATOR=${OPTARG};;
        x) USE_TMUX=${OPTARG};;
        c) CHECKPOINT=${OPTARG};;
        G) GENTRX_MODE=${OPTARG};;
        Q) GRAD_URL=${OPTARG}; _EXPLICIT_GRAD_URL=1;;
    esac
done

# If -G not passed but we ran GenTRX before, restore saved mode so this
# update run also restarts the gradient server and keeps GENTRX_VAL_ARGS.
if [ -z "$GENTRX_MODE" ] && [ -n "${GENTRX_SAVED_MODE:-}" ]; then
    GENTRX_MODE="$GENTRX_SAVED_MODE"
fi

# -Q alone (no -G, no saved mode) still enables GenTRX; default to sibling.
if [ -n "$GRAD_URL" ] && [ -z "$GENTRX_MODE" ]; then
    GENTRX_MODE="sibling"
fi

# Export GRAD_CORES_COUNT so run_gradients.sh (called for local gradient server)
# inherits it, and so the validator Python process reads it via os.environ.
export GRAD_CORES_COUNT

echo "ENDPOINT: $ENDPOINT"
echo "WALLET_PATH: $WALLET_PATH"
echo "WALLET_NAME: $WALLET_NAME"
echo "HOTKEY_NAME: $HOTKEY_NAME"
echo "NETUID: $NETUID"
echo "PAGERDUTY KEY: $PD_KEY"
echo "PROMETHEUS PORT: $PROM_PORT"
echo "TIMEOUT: $TIMEOUT"
echo "SIMULATION_CONFIG: $SIMULATION_CONFIG"
echo "PRESERVE_SIMULATOR: $PRESERVE_SIMULATOR"
echo "USE_TMUX: $USE_TMUX"
echo "CHECKPOINT: $CHECKPOINT"
echo "GENTRX_MODE: ${GENTRX_MODE:-(disabled)}"
echo "GRAD_URL: ${GRAD_URL:-(auto)}"
echo "GENTRX_VAL_ARGS: ${GENTRX_VAL_ARGS:-<none>}"

# ── process cleanup ────────────────────────────────────────────────────────────
# Only kill gradient-server when it is managed locally by this script.
# An external gradient server (run_gradients.sh on same or remote host) must
# not be touched — it has its own lifecycle.
_kill_grad=""
if [ "${GENTRX_GRAD_LOCAL:-0}" = "1" ]; then
    _kill_grad="gradient-server"
fi
if [ "$PRESERVE_SIMULATOR" = "0" ]; then
    pm2 delete simulator validator ${_kill_grad} 2>/dev/null || true
    if [ "$USE_TMUX" = "1" ]; then
        tmux kill-session -t taos 2>/dev/null || true
    fi
else
    pm2 delete validator 2>/dev/null || true
fi

echo "Updating Validator"
cd "$REPO_ROOT"
git pull || { echo "WARNING: git pull failed (no tracking branch?). Continue without updating? [y/N]"; read -r _yn; [ "$_yn" = "y" ] || exit 1; }
pip install -e .

# ── simulator build ────────────────────────────────────────────────────────────
if [ "$PRESERVE_SIMULATOR" = "0" ]; then
    echo "Updating Simulator"
    export LD_LIBRARY_PATH="/usr/local/gcc-14.1.0/lib/../lib64:$LD_LIBRARY_PATH"
    cd simulate/trading
    cd vcpkg
    CURRENT_VCPKG_HASH=$(git rev-parse HEAD)
    EXPECTED_VCPKG_HASH="e140b1fde236eb682b0d47f905e65008a191800f"
    echo "Current vcpkg commit:  $CURRENT_VCPKG_HASH"
    echo "Expected vcpkg commit: $EXPECTED_VCPKG_HASH"
    if [ "$CURRENT_VCPKG_HASH" != "$EXPECTED_VCPKG_HASH" ]; then
        echo "Updating vcpkg..."
        git fetch --all --quiet
        git reset --hard "$EXPECTED_VCPKG_HASH"
        echo "Repo successfully reset to $EXPECTED_VCPKG_HASH."
        rm -r ../build
        mkdir ../build
        ./bootstrap-vcpkg.sh -disableMetrics
    else
        echo "Commit hash matches. No reset needed."
    fi
    cd ..
    if ! g++ -dumpversion | grep -q "14"; then
        cd build && cmake -DENABLE_TRACES=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 .. && cmake --build . -j "$(nproc)"
    else
        cd build && cmake -DENABLE_TRACES=1 -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j "$(nproc)"
    fi
    cd "$REPO_ROOT/taos/im/neurons"
else
    cd "$REPO_ROOT/taos/im/neurons"
fi

# ══════════════════════════════════════════════════════════════════════════════
# GenTRX setup (runs when -G is passed; defaults to sibling mode)
# ══════════════════════════════════════════════════════════════════════════════

_hr()         { echo; printf '%s\n' "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
_ok()         { printf ' \033[32m✓\033[0m  %s\n' "$*"; }
_warn()       { printf ' \033[33m⚠\033[0m  %s\n' "$*"; }
_step()       { printf '\n \033[1m─ %s\033[0m\n' "$*"; }
_info()       { printf '   %s\n' "$*"; }

# Write/update key=value in .env (handles special chars in value)
_env_write() {
    local key="$1" val="$2"
    [ -f "$REPO_ROOT/.env" ] || touch "$REPO_ROOT/.env"
    python3 -c "
import sys, os, shlex
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).readlines() if os.path.exists(path) else []
new = []; found = False
quoted = shlex.quote(val) if val else \"''\"
for l in lines:
    if l.strip().startswith(key + '=') and not l.strip().startswith('#'):
        new.append(key + '=' + quoted + '\n'); found = True
    else:
        new.append(l)
if not found:
    new.append(key + '=' + quoted + '\n')
open(path, 'w').writelines(new)
" "$REPO_ROOT/.env" "$key" "$val"
}

# Prompt for a value; skip if already set in current env; write to .env
_prompt() {
    local varname="$1" msg="$2" default="${3:-}"
    local current="${!varname:-}"
    if [ -n "$current" ]; then
        _info "$varname: [already set — skipping]"; return
    fi
    if [ -n "$default" ]; then
        read -r -p "    $msg [$default]: " _in
        printf -v "$varname" '%s' "${_in:-$default}"
    else
        read -r -p "    $msg: " _in
        printf -v "$varname" '%s' "$_in"
    fi
    [ -z "${!varname:-}" ] && { echo "ERROR: $varname is required."; exit 1; }
    _env_write "$varname" "${!varname}"
    export "$varname"
}

# Prompt for a secret (silent input)
_prompt_secret() {
    local varname="$1" msg="$2"
    local current="${!varname:-}"
    if [ -n "$current" ]; then
        _info "$varname: [already set — skipping]"; return
    fi
    read -r -s -p "    $msg: " _in; echo
    printf -v "$varname" '%s' "$_in"
    [ -z "${!varname:-}" ] && { echo "ERROR: $varname is required."; exit 1; }
    _env_write "$varname" "${!varname}"
    export "$varname"
}

_gentrx_validator_setup() {
    local mode="$1"

    # Normalize mode name
    case "$mode" in
        a|agg|aggregator) mode=aggregator ;;
        s|sib|sibling)    mode=sibling ;;
        *)
            echo "ERROR: -G mode must be 'aggregator' or 'sibling' (got '$mode')."
            echo "  Use -G            for sibling mode (default, for uid 1+)."
            echo "  Use -G aggregator if you are uid 0 (publishes the canonical model)."
            exit 1 ;;
    esac

    # Persist mode so subsequent update runs (no -G flag) reuse same mode
    _env_write "GENTRX_SAVED_MODE" "$mode"

    _hr
    printf ' \033[1mGenTRX Validator Setup — %s\033[0m\n' "$mode"
    _hr
    echo
    _info "Docs: doc/gentrx/validator_setup.md"
    echo

    # ── Validator S3 bucket ──────────────────────────────────────────────────
    _step "Validator S3 bucket"
    # Write credentials live on the gradient server host, not necessarily here.
    # Skip the wizard if:
    #   (a) write creds + bucket are all present (local gradient server setup), OR
    #   (b) an external gradient server URL is given (-Q) and the bucket is known
    #       (write creds stay on the remote GPU host; we only need read creds here).
    if ( [ -n "$GRAD_URL" ] && [ -n "${GENTRX_VALIDATOR_S3_BUCKET:-}" ] ) || \
       ( [ -n "${GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY:-}" ] && \
         [ -n "${GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY:-}" ] && \
         [ -n "${GENTRX_VALIDATOR_S3_BUCKET:-}" ] ); then
        _ok "Validator S3 bucket: ${GENTRX_VALIDATOR_S3_BUCKET} [already configured — skipping]"
    else
        _info "Each validator needs one R2 or Hippius bucket."
        _info "In your Cloudflare R2 (or Hippius) dashboard, create a bucket and"
        _info "generate TWO API tokens on it:"
        _info "  • Write token  (Object Read & Write) — stays on this host"
        _info "  • Read token   (Object Read only)    — committed on-chain"
        _info "From each token, copy the Access Key ID and Secret Access Key"
        _info "(NOT the Token Value — that bearer string is for a different API)."
        echo

        # Provider selection
        local _provider
        if [ -n "${GENTRX_VALIDATOR_S3_ACCOUNT_ID:-}" ]; then
            _provider="configured-r2"   # R2: account_id present
        elif [ -n "${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-}" ]; then
            _provider="2"               # custom endpoint (MinIO/Hippius): no account_id
        else
            printf '    Provider: (1) Cloudflare R2  (2) Hippius  [1]: '
            read -r _provider; _provider="${_provider:-1}"
        fi

        case "$_provider" in
            2|hippius|Hippius)
                _prompt GENTRX_VALIDATOR_S3_ENDPOINT_URL \
                    "Hippius endpoint URL" "https://s3.hippius.com"
                _prompt GENTRX_VALIDATOR_S3_BUCKET "Bucket name"
                ;;
            *)
                _prompt GENTRX_VALIDATOR_S3_ACCOUNT_ID \
                    "Cloudflare R2 account ID (32-char hex, from R2 dashboard)"
                local _default_bucket="${GENTRX_VALIDATOR_S3_ACCOUNT_ID:-}"
                _prompt GENTRX_VALIDATOR_S3_BUCKET "Bucket name" "$_default_bucket"
                # Derive endpoint URL from account ID if not set
                if [ -z "${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-}" ]; then
                    GENTRX_VALIDATOR_S3_ENDPOINT_URL="https://${GENTRX_VALIDATOR_S3_ACCOUNT_ID}.r2.cloudflarestorage.com"
                    _env_write "GENTRX_VALIDATOR_S3_ENDPOINT_URL" "$GENTRX_VALIDATOR_S3_ENDPOINT_URL"
                    export GENTRX_VALIDATOR_S3_ENDPOINT_URL
                    _info "Endpoint URL derived: $GENTRX_VALIDATOR_S3_ENDPOINT_URL"
                fi
                ;;
        esac
        echo
        _step "Write credentials (gradient server — stays on this host)"
        _prompt_secret GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY "Write access key ID"
        _prompt_secret GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY "Write secret access key"
    fi

    _step "Read credentials (committed on-chain — used by miners to discover you)"
    _prompt_secret GENTRX_VALIDATOR_S3_READ_ACCESS_KEY  "Read-only access key ID"
    _prompt_secret GENTRX_VALIDATOR_S3_READ_SECRET_KEY  "Read-only secret access key"
    echo
    _ok "Validator S3 credentials saved to .env"

    # ── Aggregator fallback (sibling only) — discover from chain ─────────────
    if [ "$mode" = "sibling" ]; then
        _step "uid-0 aggregator fallback"
        _info "Sibling validators pull the canonical checkpoint from uid-0's bucket."

        if [ -z "${GENTRX_AGGREGATOR_S3_BUCKET:-}" ]; then
            _info "Querying chain for uid-0 bucket commitment..."
            local _agg_info
            _agg_info=$(python3 - <<PYEOF 2>/dev/null
import sys
try:
    import bittensor as bt
    from GenTRX.src.chain import GenTRXChain
    sub = bt.Subtensor(network='$ENDPOINT')
    meta = sub.metagraph($NETUID)
    chain = GenTRXChain(sub, $NETUID, meta)
    b = chain.get_bucket(0)
    if b:
        print('|'.join([b.account_id.strip(), b.endpoint_url, b.bucket_name,
                         b.access_key_id.strip(), b.secret_access_key.strip()]))
    else:
        print('')
except Exception:
    print('')
PYEOF
)
            if [ -n "$_agg_info" ]; then
                IFS='|' read -r _agg_acid _agg_ep _agg_bucket _agg_ak _agg_sk <<< "$_agg_info"
                GENTRX_AGGREGATOR_S3_ACCOUNT_ID="$_agg_acid"
                GENTRX_AGGREGATOR_S3_ENDPOINT_URL="$_agg_ep"
                GENTRX_AGGREGATOR_S3_BUCKET="$_agg_bucket"
                GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY="$_agg_ak"
                GENTRX_AGGREGATOR_S3_READ_SECRET_KEY="$_agg_sk"
                _env_write "GENTRX_AGGREGATOR_S3_ACCOUNT_ID"      "$_agg_acid"
                _env_write "GENTRX_AGGREGATOR_S3_ENDPOINT_URL"    "$_agg_ep"
                _env_write "GENTRX_AGGREGATOR_S3_BUCKET"          "$_agg_bucket"
                _env_write "GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY" "$_agg_ak"
                _env_write "GENTRX_AGGREGATOR_S3_READ_SECRET_KEY" "$_agg_sk"
                export GENTRX_AGGREGATOR_S3_ACCOUNT_ID GENTRX_AGGREGATOR_S3_ENDPOINT_URL \
                       GENTRX_AGGREGATOR_S3_BUCKET GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY \
                       GENTRX_AGGREGATOR_S3_READ_SECRET_KEY
                _ok "Aggregator bucket loaded from chain: $GENTRX_AGGREGATOR_S3_BUCKET"
            else
                _warn "Could not read uid-0 bucket from chain — please enter manually."
                _info "Find these in the MVTRX Discord pinned message or SUBNET_BOOTSTRAP.md."
                echo
                _prompt        GENTRX_AGGREGATOR_S3_BUCKET             "uid-0 bucket name"
                _prompt        GENTRX_AGGREGATOR_S3_ACCOUNT_ID         "uid-0 R2 account ID (or bucket name for Hippius)"
                _prompt_secret GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY    "uid-0 read access key ID"
                _prompt_secret GENTRX_AGGREGATOR_S3_READ_SECRET_KEY    "uid-0 read secret access key"
                if [ -z "${GENTRX_AGGREGATOR_S3_ENDPOINT_URL:-}" ] && [ -n "${GENTRX_AGGREGATOR_S3_ACCOUNT_ID:-}" ]; then
                    GENTRX_AGGREGATOR_S3_ENDPOINT_URL="https://${GENTRX_AGGREGATOR_S3_ACCOUNT_ID}.r2.cloudflarestorage.com"
                    _env_write "GENTRX_AGGREGATOR_S3_ENDPOINT_URL" "$GENTRX_AGGREGATOR_S3_ENDPOINT_URL"
                    export GENTRX_AGGREGATOR_S3_ENDPOINT_URL
                fi
            fi
        else
            _ok "Aggregator bucket: ${GENTRX_AGGREGATOR_S3_BUCKET} [already set — skipping]"
        fi
    fi

    # ── Validator UID lookup ─────────────────────────────────────────────────
    _step "Looking up validator UID on chain"
    _info "Querying metagraph (netuid=$NETUID, network=$ENDPOINT)..."
    local _uid
    _uid=$(python3 -c "
import bittensor as bt, sys
try:
    sub = bt.Subtensor(network='$ENDPOINT')
    meta = sub.metagraph($NETUID)
    w = bt.Wallet(name='$WALLET_NAME', hotkey='$HOTKEY_NAME', path='$WALLET_PATH')
    print(list(meta.hotkeys).index(w.hotkey.ss58_address))
except Exception:
    print('')
" 2>/dev/null)
    if [ -n "$_uid" ]; then
        VALIDATOR_UID="$_uid"
        _env_write "VALIDATOR_UID" "$VALIDATOR_UID"
        export VALIDATOR_UID
        _ok "Validator UID: $VALIDATOR_UID"
    else
        _warn "Could not determine validator UID (chain unreachable or not yet registered)."
        _warn "Prometheus labels will lack validator_uid; re-run after chain sync."
        VALIDATOR_UID="${VALIDATOR_UID:-}"
    fi

    # ── Gradient server ──────────────────────────────────────────────────────
    _step "Gradient server"

    _start_local_gradients() {
        "$REPO_ROOT/run_gradients.sh" \
            -m "$mode" \
            -e "$ENDPOINT" \
            -u "$NETUID" \
            -V "${VALIDATOR_UID:-}" \
            -p "$GRAD_PORT" \
            -b 127.0.0.1 \
            -l "$LOG_LEVEL" \
            -x 0 \
            -n
        GRAD_URL="http://127.0.0.1:${GRAD_PORT}/gentrx"
        _env_write "GENTRX_GRAD_LOCAL" "1"
        _env_write "GENTRX_GRAD_HOST"  ""
        export GENTRX_GRAD_LOCAL=1
    }

    if [ -n "$GRAD_URL" ]; then
        # Explicit -Q override — trust the operator
        _ok "Gradient server URL: $GRAD_URL  (from -Q, not auto-starting)"
        _info "Ensure the gradient server is already running at that address before continuing."
        # Persist -Q so future runs without -Q still use the same server
        if [ "$_EXPLICIT_GRAD_URL" = "1" ]; then
            local _netloc
            _netloc=$(python3 -c "from urllib.parse import urlparse; print(urlparse('$GRAD_URL').netloc)" 2>/dev/null || true)
            if [ -n "$_netloc" ]; then
                GENTRX_GRAD_HOST="$_netloc"
                _env_write "GENTRX_GRAD_HOST"  "$GENTRX_GRAD_HOST"
                _env_write "GENTRX_GRAD_LOCAL" "0"
                export GENTRX_GRAD_HOST
            fi
        fi

    elif [ "${GENTRX_GRAD_LOCAL:-0}" = "1" ]; then
        # Previously chose local — restart it
        _ok "Restarting local gradient server (saved config)"
        # Backfill GENTRX_GRAD_CPU if not yet written (first run with new code)
        if [ -z "${GENTRX_GRAD_CPU:-}" ]; then
            local _gpu=""
            command -v nvidia-smi > /dev/null 2>&1 && \
                _gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
            [ -z "$_gpu" ] && GENTRX_GRAD_CPU=1 || GENTRX_GRAD_CPU=0
            _env_write "GENTRX_GRAD_CPU" "$GENTRX_GRAD_CPU"
            export GENTRX_GRAD_CPU
        fi
        _start_local_gradients

    elif [ -n "${GENTRX_GRAD_HOST:-}" ]; then
        # Previously chose remote — update GRAD_URL and print the update command
        local _gport="${GENTRX_GRAD_HOST##*:}"
        local _ghost="${GENTRX_GRAD_HOST%%:*}"
        [ "$_ghost" = "$_gport" ] && _gport="$GRAD_PORT"
        GRAD_URL="http://${GENTRX_GRAD_HOST}/gentrx"
        _ok "Gradient server: $GRAD_URL (remote, saved config)"
        _info "To restart / update the gradient server on the GPU machine, run:"
        echo
        _print_remote_run_cmd "$mode"
        echo

    else
        # First run — detect GPU and always prompt for gradient server placement.
        local _gpu=""
        if command -v nvidia-smi > /dev/null 2>&1; then
            _gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
        fi

        _info "The gradient server scores miner gradients via forward passes on held-out data."
        echo
        local _gs_choice
        if [ -n "$_gpu" ]; then
            _ok "GPU detected: $_gpu"
            printf '    Gradient server options:\n'
            printf '      (1) This machine — GPU [recommended]\n'
            printf '      (2) This machine — CPU (frees GPU for other workloads; ~5× slower)\n'
            printf '      (3) Separate GPU machine\n'
            printf '    Choice [1/2/3, default 1]: '
            read -r _gs_choice; _gs_choice="${_gs_choice:-1}"
        else
            _warn "No NVIDIA GPU detected."
            _info "CPU-only is supported but ~5× slower and may miss round deadlines at scale."
            echo
            printf '    Gradient server options:\n'
            printf '      (1) This machine (CPU-only, may be slow)\n'
            printf '      (2) Separate GPU machine [recommended]\n'
            printf '    Choice [1/2, default 1]: '
            read -r _gs_choice; _gs_choice="${_gs_choice:-1}"
        fi

        # Remote option: 3 when GPU present, 2 when no GPU
        local _remote_opt="3"
        [ -z "$_gpu" ] && _remote_opt="2"

        if [ "$_gs_choice" = "$_remote_opt" ]; then
            _gentrx_setup_remote_gradients "$mode"
        else
            # Local — CPU if: no GPU, or GPU+choice=2
            local _local_cpu=0
            { [ -n "$_gpu" ] && [ "$_gs_choice" = "2" ]; } && _local_cpu=1
            [ -z "$_gpu" ] && _local_cpu=1

            if [ "$_local_cpu" = "1" ]; then
                _warn "Starting gradient server on CPU. Monitor round completion in pm2 logs."
                if [ "${GRAD_CORES_COUNT:-0}" = "0" ]; then
                    read -r -p "    CPU cores to reserve for gradient server [8]: " _gc
                    GRAD_CORES_COUNT="${_gc:-8}"
                    _env_write "GRAD_CORES_COUNT" "$GRAD_CORES_COUNT"
                    export GRAD_CORES_COUNT
                    _ok "Gradient server will use the last $GRAD_CORES_COUNT CPU cores; validator + simulator use the rest"
                else
                    _info "GRAD_CORES_COUNT=$GRAD_CORES_COUNT [from .env — skipping]"
                fi
                _env_write "GENTRX_GRAD_CPU" "1"; export GENTRX_GRAD_CPU=1
            else
                _ok "Starting gradient server on GPU: $_gpu"
                _env_write "GENTRX_GRAD_CPU" "0"; export GENTRX_GRAD_CPU=0
            fi
            _start_local_gradients
        fi
    fi

    # ── Build GENTRX_VAL_ARGS ────────────────────────────────────────────────
    GENTRX_VAL_ARGS="--gentrx.enabled --gentrx.gradient_server_url $GRAD_URL"
    [ -n "${GENTRX_API_KEY:-}" ] && \
        GENTRX_VAL_ARGS="$GENTRX_VAL_ARGS --gentrx.api_key $GENTRX_API_KEY"
    _env_write "GENTRX_VAL_ARGS" "$GENTRX_VAL_ARGS"
    export GENTRX_VAL_ARGS

    _ok "GenTRX setup complete."
    _info "Prometheus scrape targets:"
    _info "  Validator:       http://localhost:$PROM_PORT/metrics/gentrx"
    _info "  Gradient server: http://localhost:${GRAD_PORT}/gentrx/metrics"

    _print_validator_cmd "$mode"
}

_print_validator_cmd() {
    local _mode="$1"
    _hr
    printf ' \033[1mSave this command — run it for future updates and new deployments:\033[0m\n'
    _hr
    echo
    printf '    \033[1m./run_validator.sh\033[0m \\\n'
    printf '        -G %s \\\n'    "$_mode"
    printf '        -e %s \\\n'    "$ENDPOINT"
    printf '        -w %s \\\n'    "$WALLET_NAME"
    printf '        -h %s \\\n'    "$HOTKEY_NAME"
    printf '        -u %s \\\n'    "$NETUID"
    printf '        -o %s \\\n'    "$PROM_PORT"
    printf '        -t %s'         "$TIMEOUT"
    [ "$SIMULATION_CONFIG" != "simulation_0" ] && \
        printf ' \\\n        -g %s' "$SIMULATION_CONFIG"
    [ -n "${GENTRX_GRAD_HOST:-}" ] && \
        printf ' \\\n        -Q http://%s/gentrx' "$GENTRX_GRAD_HOST"
    printf '\n'
    echo
    printf '  Note: bucket credentials and API key are in .env.\n'
    printf '  Back up .env alongside this command for any new deployment.\n'
    _hr
    echo
    printf ' \033[33m⚠\033[0m  Copy the command above, then press Enter to continue... '
    read -r
    echo
}

_print_remote_run_cmd() {
    local _mode="$1"
    local _s3_ep="${GENTRX_VALIDATOR_S3_ENDPOINT_URL:-https://${GENTRX_VALIDATOR_S3_ACCOUNT_ID:-<account_id>}.r2.cloudflarestorage.com}"
    local _h="${GENTRX_GRAD_HOST%%:*}"
    local _p="${GENTRX_GRAD_HOST##*:}"
    [ "$_h" = "$_p" ] && _p="$GRAD_PORT"
    printf '    \033[1m./run_gradients.sh\033[0m \\\n'
    printf '        -m %s \\\n'   "$_mode"
    printf '        -e %s \\\n'   "$ENDPOINT"
    printf '        -u %s \\\n'   "$NETUID"
    printf '        -V %s \\\n'   "${VALIDATOR_UID:-0}"
    printf '        -p %s \\\n'   "$_p"
    printf '        -b 0.0.0.0 \\\n'
    printf '        -E %s \\\n'   "$_s3_ep"
    printf '        -B %s \\\n'   "${GENTRX_VALIDATOR_S3_BUCKET:-<bucket>}"
    printf '        -w %s \\\n'   "${GENTRX_VALIDATOR_S3_WRITE_ACCESS_KEY:-<write-key>}"
    printf '        -W %s \\\n'   "${GENTRX_VALIDATOR_S3_WRITE_SECRET_KEY:-<write-secret>}"
    printf '        -k %s\n'      "${GENTRX_API_KEY:-<api-key>}"
}

_gentrx_setup_remote_gradients() {
    local mode="$1"

    _step "Remote gradient server setup"
    _info "You will run the gradient server on a GPU machine."
    echo

    # Generate / use API key (required for non-loopback)
    if [ -z "${GENTRX_API_KEY:-}" ]; then
        _info "Generating shared API key for validator ↔ gradient server auth..."
        GENTRX_API_KEY=$(openssl rand -hex 32 2>/dev/null || \
            python3 -c "import secrets; print(secrets.token_hex(32))")
        _env_write "GENTRX_API_KEY" "$GENTRX_API_KEY"
        export GENTRX_API_KEY
        _ok "API key generated and saved to .env"
    else
        _ok "Using existing GENTRX_API_KEY from .env"
    fi

    # Get host:port — skip if already saved
    local _host _port
    if [ -n "${GENTRX_GRAD_HOST:-}" ]; then
        _host="${GENTRX_GRAD_HOST%%:*}"
        _port="${GENTRX_GRAD_HOST##*:}"
        [ "$_host" = "$_port" ] && _port="$GRAD_PORT"
        _ok "GPU host: $_host:$_port (from .env — skipping)"
    else
        read -r -p "    Gradient server address (host or IP, e.g. gpu.example.com): " _host
        [ -z "$_host" ] && { echo "ERROR: host required."; exit 1; }
        read -r -p "    Port [$GRAD_PORT]: " _port; _port="${_port:-$GRAD_PORT}"
        GENTRX_GRAD_HOST="${_host}:${_port}"
        _env_write "GENTRX_GRAD_HOST"  "$GENTRX_GRAD_HOST"
        _env_write "GENTRX_GRAD_LOCAL" "0"
        export GENTRX_GRAD_HOST
    fi
    GRAD_URL="http://${_host}:${_port}/gentrx"

    _hr
    printf '\033[1m  Copy run_gradients.sh to the GPU machine and run:\033[0m\n'
    echo
    printf '    scp %s/run_gradients.sh %s:/path/to/\n' "$REPO_ROOT" "$_host"
    echo
    _print_remote_run_cmd "$mode"
    echo
    printf '  No .env needed on the GPU machine — all credentials are in the command above.\n'
    printf '  Open firewall: allow TCP port %s from this validator host only.\n' "$_port"
    echo
    printf '  Once the gradient server is running at %s, press Enter...\n' "$GRAD_URL"
    read -r
    _hr
}

# Run GenTRX setup if requested
if [ -n "$GENTRX_MODE" ]; then
    _gentrx_validator_setup "$GENTRX_MODE"
fi

# Fall back to env value if no GenTRX mode (allows GENTRX_VAL_ARGS from .env)
GENTRX_VAL_ARGS="${GENTRX_VAL_ARGS:-}"

# ── Benchmark agents ──────────────────────────────────────────────────────────
# Defaults to the production config. Override in .env for local testing:
#   BENCHMARK_AGENTS_CONFIG=$REPO_ROOT/taos/im/config/benchmark_agents_test.json
BENCHMARK_AGENTS_CONFIG="${BENCHMARK_AGENTS_CONFIG:-$REPO_ROOT/taos/im/config/benchmark_agents.json}"
BENCHMARK_ARGS="--benchmark.agents $BENCHMARK_AGENTS_CONFIG"

# ── Extra validator args ───────────────────────────────────────────────────────
# Pass arbitrary additional flags not covered by run_validator.sh CLI options.
# Set in .env or as an environment variable before running.  Example:
#   VALIDATOR_EXTRA_ARGS="--neuron.axon_off --neuron.epoch_length 10 --repo.remote dev --simulation.data_service_url http://localhost:8084"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"

# ── Launch validator ───────────────────────────────────────────────────────────
echo "Starting Validator"
pm2 start validator.py \
    --name=validator \
    --interpreter python \
    --cwd "$REPO_ROOT/taos/im/neurons" \
    -- \
    --netuid "$NETUID" \
    --subtensor.chain_endpoint "$ENDPOINT" \
    --wallet.path "$WALLET_PATH" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$HOTKEY_NAME" \
    --logging."$LOG_LEVEL" \
    --alerting.pagerduty.integration_key "$PD_KEY" \
    --prometheus.port "$PROM_PORT" \
    --neuron.timeout "$TIMEOUT" \
    --simulation.xml_config "$REPO_ROOT/simulate/trading/run/config/$SIMULATION_CONFIG.xml" \
    ${BENCHMARK_ARGS:+$BENCHMARK_ARGS} \
    ${GENTRX_VAL_ARGS:+$GENTRX_VAL_ARGS} \
    ${VALIDATOR_EXTRA_ARGS:+$VALIDATOR_EXTRA_ARGS}

# ── Launch simulator ───────────────────────────────────────────────────────────
if [ "$PRESERVE_SIMULATOR" = "0" ]; then
    echo "Starting Simulator"
    if [ "$CHECKPOINT" = "0" ]; then
        pm2 start --no-autorestart --name=simulator \
            --cwd "$REPO_ROOT/simulate/trading/run" \
            "../build/src/cpp/taosim" \
            -- -f "config/$SIMULATION_CONFIG.xml"
    else
        pm2 start --no-autorestart --name=simulator \
            --cwd "$REPO_ROOT/simulate/trading/run" \
            "../build/src/cpp/taosim" \
            -- -c "$CHECKPOINT"
    fi
    pm2 save
    pm2 startup

    if [ "$USE_TMUX" = "1" ]; then
        echo "Setting Up Tmux Session"
        tmux new-session -d -s taos -n 'validator' 'htop -F validator.py'
        # Enable mouse: click window tabs in the status bar to switch windows
        tmux set-option -t taos mouse on
        tmux split-window -h -t taos:validator 'htop -F taosim'
        tmux select-pane -t 0
        tmux split-window -v -t taos:validator 'pm2 logs validator'
        tmux select-pane -t 2
        tmux split-window -v -t taos:validator 'pm2 logs simulator'

        if [ -n "$GENTRX_MODE" ] && [ "${GENTRX_GRAD_LOCAL:-0}" = "1" ]; then
            # GenTRX window: resource monitor (top row) + gradient-server logs (bottom).
            # Only created when validator manages the gradient server locally.
            # External servers (run_gradients.sh) have their own gentrx tmux session.
            tmux new-window -t taos -n 'gentrx'
            if [ "${GENTRX_GRAD_CPU:-0}" = "1" ]; then
                # CPU mode: htop (left) | logs (right)
                tmux send-keys -t taos:gentrx 'htop' Enter
                tmux split-window -h -t taos:gentrx
                tmux send-keys -t taos:gentrx 'pm2 logs gradient-server' Enter
            else
                # GPU mode: htop | nvitop (top row) + logs (bottom)
                command -v nvitop > /dev/null 2>&1 || pip install nvitop -q || true
                tmux send-keys -t taos:gentrx 'htop' Enter
                tmux split-window -h -t taos:gentrx
                tmux send-keys -t taos:gentrx 'nvitop 2>/dev/null || watch -n2 nvidia-smi' Enter
                tmux select-pane -t taos:gentrx.0
                tmux split-window -v -t taos:gentrx
                tmux send-keys -t taos:gentrx 'pm2 logs gradient-server' Enter
            fi
            # Return to validator window; click tabs or use Ctrl+b l to toggle
            tmux select-window -t taos:validator
        fi
    fi
fi

# ── sysctl tuning ──────────────────────────────────────────────────────────────
setting_exists() {
    grep -q "^${1}=" /etc/sysctl.conf
}
SETTINGS="net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.core.rmem_default=8388608
net.core.wmem_default=8388608
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 87380 67108864
net.ipv4.tcp_tw_reuse=1
net.ipv4.tcp_fin_timeout=30"
echo "Checking and applying sysctl settings..."
echo "$SETTINGS" | while IFS= read -r line; do
    setting="${line%%=*}"
    value="${line#*=}"
    echo "Applying: ${setting}=${value}"
    sudo sysctl -w "${setting}=${value}"
done
needs_update=false
echo "$SETTINGS" | while IFS= read -r line; do
    setting="${line%%=*}"
    if ! setting_exists "$setting"; then
        echo "needs_update"
        break
    fi
done | grep -q "needs_update" && needs_update=true || true
if [ "$needs_update" = true ]; then
    echo "$SETTINGS" | while IFS= read -r line; do
        setting="${line%%=*}"
        value="${line#*=}"
        if ! setting_exists "$setting"; then
            echo "Adding: ${setting}=${value}"
            echo "${setting}=${value}" | sudo tee -a /etc/sysctl.conf > /dev/null
        else
            echo "Already present: ${setting}"
        fi
    done
    echo "Settings added to /etc/sysctl.conf."
fi
echo "Current settings:"
sysctl net.core.rmem_max net.core.wmem_max net.core.rmem_default net.core.wmem_default \
    net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.ipv4.tcp_tw_reuse net.ipv4.tcp_fin_timeout

if [ "$USE_TMUX" = "1" ]; then
    tmux attach-session -t taos
fi
