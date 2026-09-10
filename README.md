# batch-api-distribution-tracker

A reproducible experiment that measures **OpenAI Batch API turnaround time as a function of requested
output length**, using one request per Batch job so that every observation has its own server-side
`created_at` / `in_progress_at` / `completed_at` timestamps.

* Model: `gpt-5.6-luna` via the Batch API, endpoint `/v1/responses`, `completion_window="24h"`
* Design: 5 output-token limits × 400 independent one-request Batch jobs = **2,000 production jobs**,
  submitted in one randomised wave (recorded seed), plus a 9-job pilot that is never counted as production.
* Primary measurements (server timestamps only; local polling time is never used as the completion time):

```text
turnaround_seconds = completed_at   - created_at
queue_seconds      = in_progress_at - created_at
active_seconds     = completed_at   - in_progress_at
```

## Deviation from the brief (read this first)

The brief asked for output-token limits `1, 10, 100, 1000, 10000`. **`gpt-5.6-luna` rejects
`max_output_tokens` below 16** on `/v1/responses` (HTTP 400 `integer_below_min_value`,
"Expected a value >= 16"; `/v1/chat/completions` rejects 1 as well). This was verified live on
2026-09-10 before implementation and again inside the pilot, whose 1- and 10-token jobs are kept in
`reports/pilot_report.md` as documentation of the constraint.

Rather than silently substitute, the production levels were changed to **`16, 100, 1000, 3000, 10000`**:
16 is the API floor (the closest feasible "tiny output" condition) and 3000 fills the log-scale gap between
1000 and 10000 so the design still has five distinct lengths. The worst-case cost rises from $2.67 to $3.40
(still under the $4.00 ceiling). Both the original and the actual levels are recorded in
`config/experiment.json` (`original_requested_levels`, `output_token_levels`) and in every report.
The levels are configurable: `python -m experiment prepare --levels ...` re-runs the same pipeline with any
set of feasible levels.

The model was **not** substituted: every request is `{"model": "gpt-5.6-luna", "input": <prompt>,
"max_output_tokens": N}` exactly as specified; reasoning is left at the model default (`medium`), and
reasoning tokens (which count toward `max_output_tokens`) are recorded separately.

## Quick start

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # hash-locked
export OPENAI_API_KEY=sk-...                # never written to any file in this repo

python -m experiment prepare --runs-per-group 400 --seed 20260910   # uploads inputs, manifest, DB, cost check; no batch creations
python -m experiment pilot --execute --max-cost-usd 0.10             # 9 pilot batches -> reports/pilot_report.md
python -m experiment launch --execute --concurrency 50 --max-cost-usd 4 --wait   # the timed wave (waits for the rolling-hour window)
python -m experiment monitor                                          # poll every 5-10 s until every job is terminal (resumable)
python -m experiment collect                                          # fetch output/error files, build observations
python -m experiment analyze                                          # stats, plots, reports/experiment_report.html
python -m experiment status
python -m experiment recover --execute                                # optional: labelled replacements for failed creation calls
```

Every command is idempotent: state lives in `data/state.sqlite`, raw event files are append-only, and
restarting any command never creates duplicate Batch jobs (`reconcile` adopts any batch the server created
whose local write was lost, matching on `metadata.observation_id`).

## Safety rails

* **Cost ceiling** — `prepare`, `pilot`, `launch` and `recover` project the worst case (every request emits its
  full limit) plus a 10 % margin and abort before any creation call if it exceeds `--max-cost-usd`
  (default $4.00; pilot default $0.10). Final cost is computed from actual usage.
* **Creation limit** — the documented limit is 2,000 batch creations per hour. The launcher counts creations
  in the rolling hour from both the local attempt log and `GET /v1/batches`, refuses to exceed the limit,
  stops the wave on a 429 that names the batch limit, never retries a failed creation during the wave, and
  `--wait` sleeps until the window allows the full wave.
* **Secrets** — the API key is read from the environment only; every log line and every raw record passes
  through a redaction filter, and the pilot scans all artifacts for key-shaped strings.

## Layout

```text
experiment/            Python package (python -m experiment <command>)
  config.py            ExperimentConfig (no secrets), pricing, API floor
  jsonl.py manifest.py cost.py limits.py redact.py     pure helpers
  store.py             SQLite state (jobs, creation_attempts, results, files, meta)
  api.py               async wrapper over the official openai SDK (raw responses, request ids)
  wave.py              the creation wave (bounded concurrency, no retries, 429 handling)
  prepare.py pilot.py launch.py monitor.py collect.py recover.py reconcile.py status.py analyze.py
config/experiment.json          the configuration actually used (+ experiment.example.json)
data/launch_manifest.csv        2,000 rows, shuffled with seed 20260910
data/raw/*.jsonl                append-only: creation events, poll events, final batch objects, responses, errors
data/processed/observations.{csv,parquet}   one row per Batch job (see "Data fields")
data/state.sqlite               transactional job state
reports/pilot_report.md         pilot checks
reports/experiment_report.html  statistics, plots, model fit, bias checks, cost, limitations
tests/                          pytest suite (fake API, no network)
```

## Data fields (`data/processed/observations.csv`)

`experiment_id, observation_id, attempt_id, phase, requested_output_tokens, api_max_output_tokens,
actual_output_tokens, reasoning_tokens, actual_input_tokens, total_tokens, randomized_launch_position,
local_create_started_at, local_create_finished_at, batch_id, input_file_id, output_file_id, error_file_id,
created_at, in_progress_at, finalizing_at, completed_at, failed_at, expired_at, cancelled_at, status,
turnaround_seconds, queue_seconds, active_seconds, server_created_at_offset_seconds, response_status,
finish_or_incomplete_reason, early_stop, http_status, openai_request_id, error_type, error_code,
error_message, estimated_cost_usd, valid_observation` (full list in `experiment/observations.py`).

`phase` is `pilot`, `prod` (intent-to-measure) or `replacement`; replacement rows carry
`parent_observation_id` and `attempt_id > 1` and are always reported separately.

## Tests

```bash
python -m pytest -q     # 112 tests, ~10 s, no network
```

The suite uses an in-memory fake of the Batch API (`tests/fake_api.py`) and covers: exactly 400 jobs per
group and 2,000 production jobs; randomised, reproducible launch order; JSONL validity; unique observation
identifiers; input-file reuse and the per-job-file fallback; timestamp calculations; creation-limit
enforcement; cost-ceiling enforcement; retry accounting; idempotent resume; response-usage extraction;
failed and expired jobs; malformed output files; absence of secrets from logs and artifacts; the analysis
and report on synthetic data; and the whole pipeline end to end.

## Results (final, 2026-09-10)

Launch: 2026-09-10T10:22:59Z to 2026-09-10T10:23:22Z (23 s local window, 11 s server `created_at` span, 50-way concurrency). 1,998 creation calls returned HTTP 200; 2 returned Cloudflare 504 but the server had created those batches, which `reconcile` adopted, so all 2,000 production batches ran. Every batch reached `completed` (0 failed, 0 expired); the last one finished 9 h 52 min after launch. 4 requests (0.2%) got a per-request HTTP 400 `invalid_prompt` from the usage-policy filter inside otherwise completed batches and are kept as non-valid rows in the intent-to-measure dataset. No replacements were needed, so there is no replacement dataset.

**Turnaround is bimodal (three waves) and set by the batch scheduler, not by output length.** Wave 1: 1020 jobs finished within 30 min of launch (median 193 s). Wave 2: 492 jobs were released together about 7.4 h after launch (26,347 to 26,820 s). Wave 3: 484 jobs about 9.7 h after launch (34,892 to 35,354 s). Which wave a job landed in is unrelated to its requested length (share per wave is 49 to 53%, 21 to 26%, 21 to 29% in every group). `in_progress_at` arrives about 2 s after creation for almost every batch, so the API's queue phase does not reflect the wait; the hours of waiting are inside the `in_progress` phase.

Per-group turnaround in seconds (valid production jobs; 95% percentile-bootstrap CIs, 2,000 resamples, seed 20260910):

| requested tokens | n valid | mean | sd | min | p10 | p25 | median [95% CI] | p75 | p90 | p95 [95% CI] | p99 | max | early stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 400 | 15,783 | 15,593 | 8 | 73 | 168 | 26,360 [362, 26,428] | 34,892 | 35,074 | 35,154 [35,129, 35,217] | 35,245 | 35,354 | 0.0% |
| 100 | 400 | 16,075 | 15,900 | 11 | 106 | 191 | 26,371 [429, 26,446] | 34,915 | 35,056 | 35,155 [35,128, 35,170] | 35,220 | 35,277 | 0.0% |
| 1,000 | 398 | 14,823 | 15,498 | 29 | 108 | 174 | 452 [349, 26,392] | 26,765 | 35,055 | 35,147 [35,088, 35,171] | 35,232 | 35,277 | 6.5% |
| 3,000 | 399 | 14,317 | 15,361 | 16 | 112 | 172 | 428 [335, 26,405] | 26,661 | 35,057 | 35,150 [35,092, 35,170] | 35,278 | 35,354 | 8.8% |
| 10,000 | 399 | 14,855 | 15,419 | 70 | 157 | 228 | 999 [411, 26,479] | 26,776 | 35,152 | 35,196 [35,166, 35,229] | 35,269 | 35,353 | 13.5% |

The medians jump between 452 s and 26,360 s across groups only because each group's mix of waves straddles the 50% mark; that is the bimodality, not a length effect. Within wave 1 the effect of length is small but visible:

| requested tokens | 16 | 100 | 1,000 | 3,000 | 10,000 |
|---|---|---|---|---|---|
| wave-1 median turnaround (s) | 162 | 177 | 186 | 178 | 234 |
| wave-1 p95 turnaround (s) | 396 | 463 | 417 | 462 | 676 |
| wave-1 median active time (s) | 160 | 163 | 172 | 169 | 225 |
| wave-2 median turnaround (s) | 26,513 | 26,504 | 26,514 | 26,513 | 26,568 |
| wave-3 median turnaround (s) | 35,048 | 35,034 | 35,036 | 35,054 | 35,147 |

**Exploratory model.** OLS of log(turnaround) on log1p(actual output tokens), launch position and server `created_at` offset (n = 1996, HC3 robust SEs): the token coefficient is 0.011 (robust 95% CI -0.040 to 0.062; bootstrap CI -0.041 to 0.064), R² = 0.003. A factor model on requested tokens gives the same picture (no level differs from 16 tokens at conventional levels). This is descriptive of one launch on one day and is not a causal or forward-looking estimate.

**Bias checks.** Submission order and server `created_at` are balanced across groups (Kruskal-Wallis p = 0.68 and p = 0.60; group medians of `created_at` within 1 s). Launch window 23 s. Failure rate 0% in every group. Flagged: early-stop rate rises with the limit (0%, 0%, 6.5%, 8.8%, 13.5%; chi-square p < 1e-20) and three groups have 398 or 399 valid rows instead of 400 because of the 4 `invalid_prompt` rejections. Actual output equals the requested limit for the median job in every group (ratio 1.00); the model used the default `medium` reasoning, which consumed all 16 tokens of the smallest group and about 37 tokens elsewhere.

**Monitoring gaps.** The container hosting the monitor restarted once; no poll events were recorded between 12:45 and 15:08 UTC. Server timestamps are unaffected, and only 3 batches changed state in that window. The poll archive is split into `data/raw/batch_poll_events.part01.jsonl.gz` and `data/raw/batch_poll_events.jsonl.gz` (rotation at 90 MB); the pilot's events are in the uncompressed `batch_poll_events.jsonl`.

## Cost statement

Computed from actual usage at the Batch rates (0.1 $/1M input, 0.01 $/1M cached input, 0.6 $/1M output; https://developers.openai.com/api/docs/pricing (Batch column, 2026-09-10)).

| phase | jobs with usage | input tokens | output tokens | cost (USD) |
|---|---|---|---|---|
| prod | 2000 | 89,775 | 5,101,522 | 3.0699 |
| pilot | 9 | 270 | 11,264 | 0.0068 |
| **total** | | | 5,202,831 total tokens | **3.0767** |

Projected worst case before launch was $3.3998 ($3.7398 with the 10% margin) against the $4.00 ceiling; actual production spend was 90% of the projection because early stops in the large groups used fewer tokens than the limit.

## Limitations

* One launch, one day, one project: the three-wave pattern describes the scheduler's behaviour during 2026-09-10 10:23 to 20:15 UTC and is not a forecast.
* One request per batch means per-batch scheduling overhead dominates; a multi-request batch may behave differently.
* Levels 1 and 10 could not be measured (API minimum is 16); 16 and 3000 were used instead, so the design differs from the brief.
* Server timestamps have 1-second resolution; queue and finalizing phases of 1 to 3 s are at the resolution floor.
* `max_output_tokens` caps reasoning plus visible tokens; with the default `medium` reasoning the 16-token group produced no visible text, and reasoning tokens (16 to 38 on average) are part of every group's output count.
* The model stopped early in 6.5 to 13.5% of jobs at 1,000 tokens and above, so actual length is slightly below the limit for those jobs; latency is analysed against both requested and actual tokens.
* Four requests were rejected by the usage-policy filter (`invalid_prompt`) with the identical prompt, which appears to be non-deterministic on the provider side.
* A container restart left a 2 h 23 min hole in the poll-event stream (server timestamps are unaffected).
* Cost figures are computed from reported usage at the documented Batch rates, not from the billing dashboard.
* The exploratory regression explains almost none of the variance (R² 0.003): the wave a job lands in, which we cannot predict from anything we recorded, is the dominant factor.

