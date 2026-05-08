# GenTRX bucket traffic

How many reads, writes, and stored bytes each GenTRX bucket sees in steady state. Use these to size your storage provider's budget, configure rate limiting, and recognise anomalies in the dashboard. Dollar cost depends on the provider (R2, Hippius, self-hosted) and their current pricing sheet; multiply the request counts and storage volumes below by the per-class request rate and per-GB storage rate the provider publishes.

Defaults assume `blocks_per_round=25` (≈5 min rounds, 288 rounds/day on a 12 s mainnet block) and 10 validators on a 7-day sim cycle. Scale pro-rata for your actual cadence and validator count.

---

## Two clocks

Training rounds run on wall-clock blocks (288/day at the default cadence). Parquet flushes run on sim-time: 5 min sim windows over ~24 h of sim time stretched across 7 wall-clock days, so ~3.4 h of sim-time per wall-clock day, giving ~41 parquet flushes per book per wall-clock day. Most request anomalies trace back to one of these two clocks running off-spec.

---

## Miner bucket

Per day, one miner:

| Operation | Count / day | Class | Notes |
|---|---|---|---|
| Gradient PUTs | 1 per round ≈ 288 | A (write) | One `.grad` per round, named by round_id. |
| Gradient GETs (validators) | `n_validators × 288` ≈ 2,900 | B (read) | Aggregator + each sibling pulls the gradient once per round. |

Storage is bounded by `gtx_keep_gradients` (default 50, ≈4 h of history at the standard cadence). At ~4 MB per file the hot bucket sits at ~200 MB. `gtx_keep_gradients=0` disables pruning, after which the bucket grows ~8 GB per 7-day sim cycle.

| Storage item | Volume |
|---|---|
| Gradient files (default retention, 50 files) | ~200 MB |
| Gradient files (retention disabled, 7-day sim) | ~8 GB |
| Checkpoint cache (model only, no optimiser) | ~47 MB |

The miner also downloads training parquets from the validator bucket each round; those reads count against the validator's bucket, not the miner's.

### Cold storage (recommended)

The hot bucket is for active training only. For long-term lineage (post-mortem, replay, audit) sync gradients to cheaper cold storage yourself: a nightly `aws s3 sync`, `rclone copy`, or `mc mirror` to Backblaze B2 / a local NAS / Glacier all speak the R2 / Hippius API. There is no bundled archival path; per-miner volumes are small and retention preferences vary.

---

## Validator bucket (aggregator)

Per day, aggregator with 256 miners on 128 books and 9 sibling validators:

| Operation | Count / day | Class | Notes |
|---|---|---|---|
| Data parquet PUTs | `128 × 41` ≈ 5,200 | A | Sim-time flush. |
| Proposal PUTs | 1 per round ≈ 288 | A | Round-based (wall-clock). |
| Checkpoint PUTs | < 200 typical | A | Suppressed by rollback when no improvement. |
| Data parquet GETs (miners) | `256 × 3 × 288` ≈ 221,000 | B | Per-round, no caching. With miner-side caching: ~32,000. |
| Data parquet GETs (siblings) | `9 × 128 × 288` ≈ 332,000 | B | Per-round without caching; ~47,000 with caching. |
| Checkpoint GETs | one per miner per roll × 256 ≈ 50,000 | B | Spikes on each version bump. |
| Proposal GETs (aggregator fan-in) | `9 × 288` ≈ 2,600 | B | Only the aggregator reads proposals. |

Training parquets accumulate through the 7-day sim run and are wiped on the next sim end (ESE) marker; no operator knob. Production-shape sim runs land at ~31 GB; full agent count at ~80 GB. Checkpoints are 47 MB each, pruned to `--keep-checkpoints` newest (default 10). Proposals are pruned to `--keep-proposals` newest (default 10, ~10 MB each).

| Storage item | Peak / retained | 7-day average |
|---|---|---|
| Training parquets (`data/`) | ~31 GB observed (80+ GB at full agent count) | ~15-40 GB |
| Checkpoints (`checkpoints/`) | ~470 MB (10 × 47 MB) | constant |
| Proposals (`proposals/`) | ~100 MB (10 × ~10 MB) | constant |

Two patterns drive aggregator-bucket request volume up quickly:

- **Disabled miner-side parquet caching.** Each round repeats the GETs from scratch instead of reusing the local cache, ~7× the baseline read traffic.
- **Aggressive checkpoint rolls.** Keep rollback enabled so a checkpoint publishes only when it improves on the previous one.

### Cold storage (recommended)

`--keep-checkpoints` / `--keep-proposals` cap the hot bucket. For a long-term checkpoint lineage (audit, retraining from a historical state) mirror `checkpoints/` to cold storage yourself; same `aws s3 sync` / `rclone` / `mc mirror` pattern as the miner side. `--keep-checkpoints=0` keeps everything in the hot bucket and grows storage unbounded.

---

## Sibling validator bucket

Sibling validators write `data/` and `proposals/` only; they do not publish checkpoints. Storage is dominated by training parquets at roughly the same volume as the aggregator (each validator has its own held-out books). Request volume is one PUT/round for proposals plus the parquet GETs the aggregator and other siblings make against this bucket, typically a low fraction of the aggregator totals because fewer participants read it.

---

## Monitoring (recommended)

Bucket read credentials are public on-chain by design, so an unfriendly or buggy participant can drive traffic via request storms. The numbers above bound the steady state; watch for anomalies.

### Provider-side

- **Per-bucket dashboard.** Most S3-compatible providers expose per-bucket request rate and storage on a daily cadence. Eyeball weekly during PoC, monthly once steady.
- **Billing alerts.** Set a hard cap above your expected monthly total (suggested: 3× the steady-state request count). Triggers an email before the invoice surprises you.
- **Per-bucket request anomaly.** Most providers do not ship a native alert. A small cron that posts to Slack or email when the daily Class B count exceeds a threshold is two short scripts.

### Validator + miner side

- `[GTX] pruned N old gradient(s) …` / `pruned N old checkpoint(s) …` log lines confirm retention is firing. Absence for >24 h despite uploads happening is a signal that something is wrong.
- `gradients/pending/` on the miner host should stay empty in steady state. If it grows, S3 is unreachable; the upload retry loop reconverges automatically but the connectivity is worth investigating.
- Validator: `aggregation.jsonl` growing without checkpoint version bumps means proposals keep getting rejected (no improvement). Not a traffic issue, but a quality signal worth watching alongside.

### Suggested thresholds

| Metric | Suggested threshold | Reason |
|---|---|---|
| Class B requests / day on miner bucket | 10× baseline (~3,000/day expected) | Validator-side bug or rogue scanner |
| Class A requests / day on validator bucket | 3× baseline | Checkpoint thrash or proposal flood |
| Bucket storage MoM growth | 2× | Retention disabled or misconfigured |

For tighter loops a self-hosted Prometheus exporter against the S3 metrics endpoint is cheap to add but not bundled here.
