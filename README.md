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
python -m pytest -q
```

The suite uses an in-memory fake of the Batch API (`tests/fake_api.py`) and covers: exactly 400 jobs per
group and 2,000 production jobs; randomised, reproducible launch order; JSONL validity; unique observation
identifiers; input-file reuse and the per-job-file fallback; timestamp calculations; creation-limit
enforcement; cost-ceiling enforcement; retry accounting; idempotent resume; response-usage extraction;
failed and expired jobs; malformed output files; absence of secrets from logs and artifacts; the analysis
and report on synthetic data; and the whole pipeline end to end.

## Results

See `reports/experiment_report.html` (and the summary at the end of this README once the run completes).
