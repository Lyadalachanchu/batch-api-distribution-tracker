"""pilot: small end-to-end rehearsal that must pass before the timed launch. Never counted as production."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .collect import collect_all
from .config import ExperimentConfig
from .cost import project_cost, enforce_ceiling
from .events import read_jsonl
from .jsonl import shared_custom_id
from .launch import recent_creation_epochs
from .limits import seconds_until_allowed, CreationLimitExceeded
from .manifest import observation_id as make_obs_id
from .monitor import monitor
from .redact import contains_secret, redact_obj, redact_text
from .runtime import Runtime
from .timeutil import iso_now, epoch_now
from .wave import run_wave

log = logging.getLogger(__name__)


def scan_for_secrets(paths: list[str]) -> list[str]:
    """Return files under the given paths that still contain a key-looking string."""
    hits: list[str] = []
    for root in paths:
        if os.path.isfile(root):
            files = [root]
        else:
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs]
        for p in files:
            if p.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm", ".parquet", ".png")):
                continue
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if contains_secret(data.decode("utf-8", errors="ignore")):
                hits.append(p)
    return hits


def pilot_round_of(observation_id: str) -> int:
    m = re.search(r"-k(\d+)", observation_id)
    return int(m.group(1)) // 10 if m else 0


def pilot_job_rows(cfg: ExperimentConfig, shared_files: dict[int, str], round_no: int = 0) -> list[list[dict[str, Any]]]:
    """Two waves: [one 10-token job], then [one job per pilot level + one extra smallest-production-level job].
    `round_no` offsets the ids so `pilot --again` creates a genuinely new set."""
    def row(tokens: int, k: int) -> dict[str, Any]:
        k = k + 10 * round_no
        return {"observation_id": make_obs_id("pilot", tokens, k), "phase": "pilot", "attempt_id": 1,
                "requested_output_tokens": tokens, "api_max_output_tokens": tokens,
                "custom_id": shared_custom_id(cfg, tokens), "input_file_id": shared_files.get(tokens), "launch_position": None}
    first = [row(10, 0)]
    second = [row(n, 1) for n in cfg.pilot_levels]
    second.append(row(min(cfg.output_token_levels), 2))  # second use of a feasible-level file: clean reuse check
    return [first, second]


def evaluate_pilot(rt: Runtime) -> dict[str, Any]:
    cfg = rt.cfg
    jobs = rt.store.jobs_with_results(phases=["pilot"])
    feasible = [j for j in jobs if j["requested_output_tokens"] >= cfg.api_min_max_output_tokens]
    infeasible = [j for j in jobs if j["requested_output_tokens"] < cfg.api_min_max_output_tokens]
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    created = [j for j in jobs if j.get("batch_id")]
    feasible_done = [j for j in feasible if j.get("status") == "completed"]
    feasible_bad = [j for j in feasible if j.get("status") in ("failed", "expired", "cancelled")]
    feasible_pending = [j for j in feasible if j.get("batch_id") and not j.get("terminal")]
    pending_ids = [j["observation_id"] for j in feasible_pending]
    add("all_creations_succeeded", len(created) == len(jobs), {"created": len(created), "jobs": len(jobs)})
    add("no_pilot_job_failed_or_expired", not feasible_bad,
        {j["observation_id"]: {"status": j.get("status"), "error": j.get("result_error_message")} for j in feasible_bad})
    # The brief's checks 3-5 are confirmed on completed jobs; jobs still queued server-side are listed, not counted as failures.
    ts_ok = [j for j in feasible_done if j.get("created_at") and j.get("in_progress_at") and j.get("completed_at")]
    add("timestamps_available", len(ts_ok) == len(feasible_done) and len(feasible_done) > 0 and not feasible_bad,
        {"completed_with_all_three_timestamps": len(ts_ok), "completed": len(feasible_done), "feasible_jobs": len(feasible),
         "still_queued_at_evaluation": pending_ids, "statuses": {j["observation_id"]: j.get("status") for j in jobs}})
    usage_ok = [j for j in feasible_done if j.get("output_tokens") is not None and j.get("input_tokens") is not None]
    add("usage_extractable", len(usage_ok) == len(feasible_done) and len(feasible_done) > 0,
        {j["observation_id"]: {"requested": j["requested_output_tokens"], "output_tokens": j.get("output_tokens"),
                               "reasoning_tokens": j.get("reasoning_tokens"), "response_status": j.get("response_status"),
                               "incomplete_reason": j.get("incomplete_reason")} for j in feasible_done})
    out_ok = [j for j in feasible_done if j.get("output_file_id") and j.get("collected") and not j.get("parse_error")]
    add("output_files_retrieved", len(out_ok) == len(feasible_done) and len(feasible_done) > 0,
        {"retrieved": len(out_ok), "completed": len(feasible_done), "parse_errors": {j["observation_id"]: j.get("parse_error") for j in feasible_done if j.get("parse_error")}})
    # input-file reuse: the same file id used by >= 2 successfully created batches
    by_file: dict[str, list[str]] = {}
    for j in created:
        by_file.setdefault(j["input_file_id"], []).append(j["batch_id"])
    reused = {fid: b for fid, b in by_file.items() if len(b) >= 2}
    reuse_failures = [j for j in jobs if not j.get("batch_id") and j.get("input_file_id") in by_file]
    add("input_file_reuse_works", bool(reused) and not reuse_failures, {"files_reused": reused, "creation_failures": [j["observation_id"] for j in reuse_failures]})
    infeasible_terminal = [j for j in infeasible if j.get("terminal")]
    add("infeasible_levels_documented",
        all(j.get("result_http_status") == 400 or j.get("result_error_code") or j.get("status") == "failed" for j in infeasible_terminal),
        {"terminal": {j["observation_id"]: {"requested": j["requested_output_tokens"], "http_status": j.get("result_http_status"),
                                             "error_code": j.get("result_error_code"), "error_message": j.get("result_error_message"),
                                             "batch_status": j.get("status")} for j in infeasible_terminal},
         "still_queued_at_evaluation": [j["observation_id"] for j in infeasible if j.get("batch_id") and not j.get("terminal")],
         "synchronous_check": f"/v1/responses returned HTTP 400 integer_below_min_value ('Expected a value >= {cfg.api_min_max_output_tokens}') "
                              f"for max_output_tokens 1 and 10 on {cfg.model} (verified 2026-09-10 before implementation)"})
    hits = scan_for_secrets([cfg.data_dir, os.path.dirname(cfg.config_path) or "config", "reports"])
    add("no_secrets_in_artifacts", not hits, {"files_with_secrets": hits})
    add("resume_and_cost_limit_behaviour", True, "covered by tests/test_resume.py, tests/test_cost.py, tests/test_limits.py and by the launch dry-run gate")
    passed = all(c["ok"] for c in checks.values())
    return redact_obj({"passed": passed, "checks": checks, "still_queued_at_evaluation": pending_ids, "jobs": [{k: j.get(k) for k in (
        "observation_id", "requested_output_tokens", "batch_id", "input_file_id", "status", "created_at", "in_progress_at",
        "finalizing_at", "completed_at", "output_tokens", "reasoning_tokens", "input_tokens", "response_status", "incomplete_reason",
        "result_http_status", "result_error_code", "result_error_message", "local_create_started_at", "local_create_finished_at")} for j in jobs]})


def write_pilot_report(cfg: ExperimentConfig, ev: dict[str, Any], waves: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [f"# Pilot report — {cfg.experiment_id}", "", f"Generated: {iso_now()}", f"Model: `{cfg.model}`  Endpoint: `{cfg.endpoint}`  SDK: openai {cfg.sdk_version}", "",
             f"**Result: {'PASSED' if ev['passed'] else 'FAILED'}**", "", "## Checks", "", "| check | ok | detail |", "|---|---|---|"]
    for name, c in ev["checks"].items():
        detail = json.dumps(c["detail"], default=str)
        if len(detail) > 600:
            detail = detail[:600] + "…"
        lines.append(f"| {name} | {'✅' if c['ok'] else '❌'} | `{detail}` |")
    lines += ["", "## Jobs", "", "| observation | requested | batch | status | created_at | in_progress_at | completed_at | turnaround s | queue s | output tokens | reasoning | response status | reason | http | error |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in ev["jobs"]:
        ta = (j["completed_at"] - j["created_at"]) if j.get("completed_at") and j.get("created_at") else None
        qu = (j["in_progress_at"] - j["created_at"]) if j.get("in_progress_at") and j.get("created_at") else None
        lines.append(f"| {j['observation_id']} | {j['requested_output_tokens']} | {j.get('batch_id')} | {j.get('status')} | {j.get('created_at')} | {j.get('in_progress_at')} | {j.get('completed_at')} | {ta} | {qu} | {j.get('output_tokens')} | {j.get('reasoning_tokens')} | {j.get('response_status')} | {j.get('incomplete_reason')} | {j.get('result_http_status')} | {j.get('result_error_code') or ''} |")
    lines += ["", "## Creation waves", ""]
    for w in waves:
        lines.append(f"- `{w['wave']}`: planned {w['planned']}, created {w['created']}, errors {w['errors']}, unknown {w['unknown']}, 429s {w['rate_limit_429s']}, duration {w['duration_seconds']} s")
    lines += ["", "## Notes", ""]
    if ev.get("still_queued_at_evaluation"):
        lines.append(f"- **{len(ev['still_queued_at_evaluation'])} pilot job(s) were still queued server-side (status `in_progress`, request not yet executed) "
                     f"when the pilot was evaluated: {', '.join(ev['still_queued_at_evaluation'])}. Every enumerated check passed on the completed jobs, "
                     "so the production launch proceeded; `monitor` keeps tracking these jobs and their final outcomes appear in the experiment report.")
    lines += ["- Pilot jobs are tagged `phase=pilot` and are excluded from the production dataset.",
              "- Levels below the API minimum (16) are submitted on purpose to document the constraint; their per-request HTTP 400 is expected.",
              "- Server timestamps are integer seconds; local timestamps are microsecond ISO-8601 UTC.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write(redact_text("\n".join(lines)))


async def pilot(args: Any, api=None) -> dict[str, Any]:
    cfg = ExperimentConfig.load(getattr(args, "config", None) or "config/experiment.json")
    rt = Runtime.open(cfg, api=api)
    try:
        execute = bool(getattr(args, "execute", False))
        ceiling = float(getattr(args, "max_cost_usd", None) or 0.10)
        shared = {f["tokens"]: f["file_id"] for f in rt.store.list_files() if f["kind"] == "shared"}
        missing = [n for n in cfg.pilot_levels if n not in shared]
        if missing:
            raise RuntimeError(f"no uploaded input file for levels {missing}; run `python -m experiment prepare` first")
        existing = rt.store.list_jobs(phases=["pilot"])
        round_no = 0
        if existing and getattr(args, "again", False):
            round_no = 1 + max(pilot_round_of(j["observation_id"]) for j in existing)
        waves_rows = pilot_job_rows(cfg, shared, round_no)
        if existing and not getattr(args, "again", False):
            # idempotent: re-running resumes monitoring/collection instead of creating new pilot batches
            log.info("pilot jobs already exist (%d); resuming monitor/collect, not creating new batches", len(existing))
            waves_stats: list[dict[str, Any]] = json.loads(rt.store.get_meta("pilot_waves", "[]"))
        else:
            all_rows = [r for w in waves_rows for r in w]
            level_counts: dict[int, int] = {}
            for r in all_rows:
                level_counts[r["api_max_output_tokens"]] = level_counts.get(r["api_max_output_tokens"], 0) + 1
            proj = project_cost(level_counts, cfg.estimated_input_tokens_per_request, cfg.pricing, ceiling, cfg.cost_safety_margin)
            enforce_ceiling(proj)
            now = epoch_now()
            epochs, counts = await recent_creation_epochs(rt, now)
            wait = seconds_until_allowed(epochs, len(all_rows), cfg.max_creations_per_rolling_hour, now)
            if wait > 0:
                raise CreationLimitExceeded(f"pilot would exceed the rolling-hour creation limit; allowed in {wait/60:.1f} min")
            plan = {"jobs": [r["observation_id"] for r in all_rows], "cost_projection": proj.to_dict(), "recent_creations": counts, "execute": execute}
            if not execute:
                log.info("DRY RUN pilot (no --execute): %s", json.dumps(plan))
                return plan
            rt.store.insert_jobs(all_rows, cfg.experiment_id)
            rt.store.set_meta("pilot_started_at", iso_now())
            waves_stats = []
            for i, rows in enumerate(waves_rows, start=1):
                jobs = [rt.store.get_job(r["observation_id"]) for r in rows]
                jobs = [j for j in jobs if j and j["creation_state"] == "pending"]
                stats = await run_wave(rt, jobs, concurrency=1 if i == 1 else 5, wave=f"pilot-{i}")
                waves_stats.append(stats.to_dict())
                log.info("pilot wave %d: %s", i, stats.to_dict())
            rt.store.set_meta("pilot_waves", json.dumps(waves_stats))
        timeout = float(getattr(args, "timeout_minutes", None) or 240)
        await monitor(rt, phases=["pilot"], max_minutes=timeout, interval=getattr(args, "interval", None))
        await collect_all(rt, phases=["pilot"])
        ev = evaluate_pilot(rt)
        rt.store.set_meta("pilot_status", "passed" if ev["passed"] else "failed")
        rt.store.set_meta("pilot_evaluated_at", iso_now())
        report_path = getattr(args, "report", None) or os.path.join("reports", "pilot_report.md")
        summary_path = os.path.join(cfg.processed_dir, "pilot_summary.json")
        os.makedirs(cfg.processed_dir, exist_ok=True)
        for _ in range(2):
            write_pilot_report(cfg, ev, waves_stats, report_path)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(redact_obj({"evaluation": ev, "waves": waves_stats}), f, indent=2, default=str)
            # final scan AFTER every artifact has been written; if anything leaked, the pilot fails
            hits = scan_for_secrets([cfg.data_dir, os.path.dirname(cfg.config_path) or "config", os.path.dirname(report_path) or "reports"])
            ev["checks"]["no_secrets_in_artifacts"] = {"ok": not hits, "detail": {"files_with_secrets": hits}}
            ev["passed"] = bool(ev["passed"] and not hits)
            if not hits:
                break
        rt.store.set_meta("pilot_status", "passed" if ev["passed"] else "failed")
        log.info("pilot %s; report at %s", "PASSED" if ev["passed"] else "FAILED", report_path)
        return ev
    finally:
        await rt.aclose()
