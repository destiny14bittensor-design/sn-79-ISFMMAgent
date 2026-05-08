# GenTRX Training Parameter Guide

The values below describe the model and training configuration as shipped in this repo. Tracking against a separate version number is intentionally avoided; the model evolves with the code.

## Model architecture

| Param | Value | Notes |
|---|---|---|
| `d_model` | 288 | Divisible by n_heads (288 / 8 = 36 per head) |
| `n_layers` | 8 | More depth gives better context modelling |
| `n_heads` | 8 | Standard for this model size |
| `d_ff` | 1152 | ~4× d_model |
| `dropout` | 0.1 | 0.15-0.2 also reasonable for stronger regularisation |
| `max_seq_len` | 2048 | Context window in orders |
| `film_layers` | (2, 5, 7) | FiLM conditioning injection points in backbone |
| `film_d_cond` | 64 | Hidden dim of FiLM projection MLP |
| **Total params** | **~12.1M** | Includes ~140K from FiLM and the richer time projection |

## Tokenizer bins

| Field | Bins | Range | Scale | Notes |
|---|---|---|---|---|
| `price` | 100 | [-500, 500] | symmetric log | Split 50 neg + 50 pos, log within each half. ~13 bins for ±5 ticks, dense near mid. |
| `vol_int` | 64 | [0, 100] | log | 81% in bin 0 (most orders vol = 0-1). Log helps the tail. |
| `vol_dec` | 8 | [0, 1] | linear | Fractional volume precision. |
| `interval` | 64 | [0, 50ms] | log | ~80% entropy across bins. |

**Symmetric log price binning.** `BinConfig(100, -500, 500, symmetric_log=True)`. Bin 50 is the zero band [-1, +1). Bins 0..49 cover negative (deep bid to near mid), bins 50..99 cover positive (near mid to deep ask). Log-spaced within each half gives ~13 bins per side for ±5 ticks.

Boundary behaviour:

- `vol_int` upper bound (100) clips ~0.4% of observations; may need adjustment for markets with markedly different volume distributions.
- `interval` upper bound (50 ms) clips ~4%; events with longer gaps collapse into the top bin.
- `price` range is fixed across symbols. An adaptive per-symbol range would help when tick sizes vary.

## Loss configuration

| Param | Value | Notes |
|---|---|---|
| Order type class weights | `[bid=2.0, ask=4.0, cancel=0.5]` | Bid/ask carry trading signal, cancel is the easiest and most common prediction. |
| Field loss weights | `order_type=2.0, price=1.5, interval=0.3, vol_int=0.5, vol_dec=0.5` | order_type and price carry the most actionable signal; interval is bin-quantised so its gradient is noisy and gets a smaller weight. |
| Label smoothing | none (0.05-0.1 reasonable) | Discourages overconfident predictions, helps generalisation. |

**Why these weights:**

- `cancel=0.5`: cancels are ~48% of the data and the easiest field to predict (mid-price + depth conditioning is enough). Higher cancel weight causes the model to flood predictions with cancels and drown out bid / ask signal.
- `bid=2.0`, `ask=4.0`: ask is upweighted because asks are only ~14% of the data and need the extra gradient to compete with bids. Together with the field-level `order_type=2.0` the model sees strong gradient on direction, which is the most useful signal for trading.
- `price=1.5`: price has 100 bins and the most complex distribution; extra signal helps.
- `interval=0.3`: interval is the noisiest field per gradient unit spent, partly because the bin scheme above quantises the distribution coarsely. The lower weight keeps it from dominating the loss while the model is still learning the easier fields.

## Standalone pretraining

Offline training of a seed checkpoint, separate from the live distributed loop. `--val-interval` here is a step counter (validate every N gradient steps), not a wall-clock cadence.

| Param | Quick test | Base checkpoint | Full training |
|---|---|---|---|
| `--lr` | 1e-4 | 1e-4 | 1e-4 |
| `--min-lr` | 1e-5 | 1e-6 | 1e-6 |
| `--warmup-steps` | 50 | 300 | 500 |
| `--batch-size` | 32 | 64 | 64 |
| `--seq-len` | 256 | 512 | 512 |
| `--max-books` | 3 | all | all |
| `--max-steps` | 500 | none | none |
| `--patience` | none | 3 | 5 |
| `--val-interval` | 100 | 500 | 500 |

## Distributed training (agent-side)

Training agents are triggered by an assignment arriving via dendrite. The assignment names the books, time window, and the exact `model_version` to train against. Parquets come from the validator bucket (credentials carried in the assignment payload). The agent downloads the named checkpoint from the validator's bucket (discovered via chain) when it is newer than the local one.

| Param | Default | Notes |
|---|---|---|
| `gtx_train_steps` | 50 | Steps per training window |
| `gtx_train_batch_size` | 16 | Bounded by GPU memory (shared with inference). Localnet / proxy launchers override to 8 for smaller GPUs. |
| `gtx_train_seq_len` | 256 | Shorter than pretrain; per-window speed matters more than long context. |
| `gtx_train_lr` | 1e-4 | Same as pretrain |
| `gtx_top_k_frac` | 0.01 | 1% retention, ~100× compression, ~500 KB gradient |

**Key trade-off.** `gtx_train_steps` controls how much each miner overfits per window. Shorter windows (50 steps) with more frequent aggregation generalise better than long windows (200+ steps) with fewer aggregations.

## Gradient server (aggregation)

| Param | Default | Notes |
|---|---|---|
| `--window-ns` | `300000000000` (5 min) | Sim-time window per assignment. See [Training window and sim grace period](#training-window-and-sim-grace-period) below. |
| `--min-score` | -0.1 | Stricter (e.g. -0.05 to 0.0) means fewer accepted gradients but safer. |
| `--rollback` | true | Always keep. Protects against regression. |
| `--max-val-batches` | 10 | More batches means more accurate scoring but slower. Range 10-30 is reasonable; raise if per-round score noise matters more than latency. |
| `--blocks-per-round` | 25 | Server-side estimate, used only by the heartbeat-loss fallback in `_round_complete`. Should match the validator's `--gentrx.blocks_per_round`. |
| `--block-time-s` | 12.0 | Assumed seconds per block on the target chain. |
| `--round-grace-s` | 30 | Grace seconds added to the heartbeat-loss estimate before force-closing a round. |

`--interval` (default 30 s) is a proxy / timer-mode knob and is ignored in block-synced production deployments; round closure is driven by `POST /gentrx/round` from the validator.

### Training window and sim grace period

The sim does not emit state immediately at startup. `simulation_0.xml` sets `gracePeriod="600000000000"` nanoseconds (10 minutes), during which the exchange accepts connections but publishes no state. First state messages arrive at `t = 10 minutes` of sim time.

The training window is a separate knob: `--window-ns` on the gradient server (default `300000000000` ns, 5 minutes). Each assignment covers one window of sim time, and the gradient server cannot create assignments until at least one window of data has accumulated.

In practice, the first training round lands at roughly `t = grace_period + window_ns`, so about 15 minutes of sim time under defaults. Operators watching startup should expect no aggregation events before that point. The gradient server logs `Not enough data yet` at debug level while it waits.

| Setting | Default | Where |
|---|---|---|
| Sim grace period | 10 min (`600000000000` ns) | `MultiBookExchangeAgent.gracePeriod` in the simulation XML |
| Training window | 5 min (`300000000000` ns) | `--window-ns` on the gradient server |
| First round lands at | ~15 min of sim time | Derived: `gracePeriod + window_ns` |

Combined with **field-level weights** (order_type=2.0), the model gets 4x more gradient signal from "is the next order a bid or ask?" compared to "what's the volume decimal?"

---
