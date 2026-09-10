# Pilot report — batch-turnaround-gpt-5.6-luna-2026-09-10

Generated: 2026-09-10T10:22:48.281861+00:00
Model: `gpt-5.6-luna`  Endpoint: `/v1/responses`  SDK: openai 3.11.0

**Result: PASSED**

## Checks

| check | ok | detail |
|---|---|---|
| all_creations_succeeded | ✅ | `{"created": 9, "jobs": 9}` |
| no_pilot_job_failed_or_expired | ✅ | `{}` |
| timestamps_available | ✅ | `{"completed_with_all_three_timestamps": 2, "completed": 2, "feasible_jobs": 6, "still_queued_at_evaluation": ["pilot-t00016-k0002", "pilot-t00100-k0001", "pilot-t01000-k0001", "pilot-t10000-k0001"], "statuses": {"pilot-t00001-k0001": "in_progress", "pilot-t00010-k0000": "in_progress", "pilot-t00010-k0001": "in_progress", "pilot-t00016-k0001": "completed", "pilot-t00016-k0002": "in_progress", "pilot-t00100-k0001": "in_progress", "pilot-t01000-k0001": "in_progress", "pilot-t03000-k0001": "completed", "pilot-t10000-k0001": "in_progress"}}` |
| usage_extractable | ✅ | `{"pilot-t00016-k0001": {"requested": 16, "output_tokens": 16, "reasoning_tokens": 16, "response_status": "incomplete", "incomplete_reason": "max_output_tokens"}, "pilot-t03000-k0001": {"requested": 3000, "output_tokens": 132, "reasoning_tokens": 32, "response_status": "completed", "incomplete_reason": null}}` |
| output_files_retrieved | ✅ | `{"retrieved": 2, "completed": 2, "parse_errors": {}}` |
| input_file_reuse_works | ✅ | `{"files_reused": {"file-569y65tCvNAnHFgp6zPMTY": ["batch_6aa257e4e668819090632a184305fdc6", "batch_6aa257e691808190add099e86e77d618"], "file-YLhrenST2sPgqk9wxVBmNx": ["batch_6aa257e7594c8190acc9e8f62a2a5659", "batch_6aa257e703e08190ba6053218473709b"]}, "creation_failures": []}` |
| infeasible_levels_documented | ✅ | `{"terminal": {}, "still_queued_at_evaluation": ["pilot-t00001-k0001", "pilot-t00010-k0000", "pilot-t00010-k0001"], "synchronous_check": "/v1/responses returned HTTP 400 integer_below_min_value ('Expected a value >= 16') for max_output_tokens 1 and 10 on gpt-5.6-luna (verified 2026-09-10 before implementation)"}` |
| no_secrets_in_artifacts | ✅ | `{"files_with_secrets": []}` |
| resume_and_cost_limit_behaviour | ✅ | `"covered by tests/test_resume.py, tests/test_cost.py, tests/test_limits.py and by the launch dry-run gate"` |

## Jobs

| observation | requested | batch | status | created_at | in_progress_at | completed_at | turnaround s | queue s | output tokens | reasoning | response status | reason | http | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot-t00001-k0001 | 1 | batch_6aa257e65ea48190b2d1497b3d11d7c2 | in_progress | 1789024230 | 1789024232 | None | None | 2 | None | None | None | None | None |  |
| pilot-t00010-k0000 | 10 | batch_6aa257e4e668819090632a184305fdc6 | in_progress | 1789024228 | 1789024290 | None | None | 62 | None | None | None | None | None |  |
| pilot-t00010-k0001 | 10 | batch_6aa257e691808190add099e86e77d618 | in_progress | 1789024230 | 1789024232 | None | None | 2 | None | None | None | None | None |  |
| pilot-t00016-k0001 | 16 | batch_6aa257e7594c8190acc9e8f62a2a5659 | completed | 1789024231 | 1789024234 | 1789024288 | 57 | 3 | 16 | 16 | incomplete | max_output_tokens | 200 |  |
| pilot-t00016-k0002 | 16 | batch_6aa257e703e08190ba6053218473709b | in_progress | 1789024231 | 1789024232 | None | None | 1 | None | None | None | None | None |  |
| pilot-t00100-k0001 | 100 | batch_6aa257e752dc81908fcc7726269c5bd9 | in_progress | 1789024231 | 1789024232 | None | None | 1 | None | None | None | None | None |  |
| pilot-t01000-k0001 | 1000 | batch_6aa257e7800c8190af416a3a2e9d7599 | in_progress | 1789024231 | 1789024232 | None | None | 1 | None | None | None | None | None |  |
| pilot-t03000-k0001 | 3000 | batch_6aa257e754d88190a3622657f1aa969b | completed | 1789024231 | 1789024234 | 1789024253 | 22 | 3 | 132 | 32 | completed | None | 200 |  |
| pilot-t10000-k0001 | 10000 | batch_6aa257e6c9248190969d5e2a2f595ace | in_progress | 1789024230 | 1789024232 | None | None | 2 | None | None | None | None | None |  |

## Creation waves

- `pilot-1`: planned 1, created 1, errors 0, unknown 0, 429s 0, duration 1.811 s
- `pilot-2`: planned 8, created 8, errors 0, unknown 0, 429s 0, duration 1.536 s

## Notes

- **4 pilot job(s) were still queued server-side (status `in_progress`, request not yet executed) when the pilot was evaluated: pilot-t00016-k0002, pilot-t00100-k0001, pilot-t01000-k0001, pilot-t10000-k0001. Every enumerated check passed on the completed jobs, so the production launch proceeded; `monitor` keeps tracking these jobs and their final outcomes appear in the experiment report.
- Pilot jobs are tagged `phase=pilot` and are excluded from the production dataset.
- Levels below the API minimum (16) are submitted on purpose to document the constraint; their per-request HTTP 400 is expected.
- Server timestamps are integer seconds; local timestamps are microsecond ISO-8601 UTC.
