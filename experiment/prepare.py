"""prepare: config, JSONL inputs, uploads, randomised manifest, DB init, validation and cost checks. No batch creations."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from .config import ExperimentConfig
from .cost import project_cost, enforce_ceiling
from .jsonl import build_request_line, shared_custom_id, write_jsonl_file, validate_jsonl_bytes
from .manifest import build_manifest, write_manifest_csv, check_manifest
from .runtime import Runtime
from .timeutil import iso_now

log = logging.getLogger(__name__)


def build_config(args: Any, existing: ExperimentConfig | None) -> ExperimentConfig:
    cfg = existing or ExperimentConfig()
    for name in ("experiment_id", "model", "runs_per_group", "seed", "file_mode", "max_cost_usd", "launch_concurrency",
                 "poll_mode", "poll_interval_seconds", "reasoning_effort", "data_dir"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)
    if getattr(args, "levels", None):
        cfg.output_token_levels = [int(x) for x in args.levels]
        cfg.pilot_levels = sorted(set(cfg.original_requested_levels) | set(cfg.output_token_levels))
    import openai
    cfg.sdk_version = openai.__version__
    cfg.created_at = cfg.created_at or iso_now()
    if not cfg.notes:
        cfg.notes = [
            "Levels 1 and 10 from the brief are below the API minimum max_output_tokens=16 for gpt-5.6-luna "
            "(HTTP 400 integer_below_min_value, verified 2026-09-10); production uses 16 and 3000 in their place. "
            "The pilot still submits 1 and 10 to document the constraint.",
            "Request body is exactly {model, input, max_output_tokens}; reasoning is left at the model default (medium).",
        ]
    cfg.validate()
    return cfg


async def upload_shared_files(rt: Runtime, levels: list[int]) -> dict[int, dict[str, Any]]:
    """One one-line JSONL per level, uploaded once (idempotent on sha256)."""
    assert rt.api is not None
    out: dict[int, dict[str, Any]] = {}
    for n in sorted(levels):
        custom_id = shared_custom_id(rt.cfg, n)
        line = build_request_line(rt.cfg, custom_id, n)
        path = os.path.join(rt.cfg.inputs_dir, f"t{n:05d}.jsonl")
        data, sha = write_jsonl_file(path, [line])
        validate_jsonl_bytes(data, rt.cfg.endpoint, expect_lines=1)
        existing = rt.store.get_shared_file(n, sha)
        if existing:
            out[n] = existing
            continue
        res = await rt.api.upload_file(os.path.basename(path), data, purpose="batch")
        if not res.ok:
            raise RuntimeError(f"upload failed for level {n}: {res.http_status} {res.error_type} {res.error_message}")
        f = res.data
        rt.store.add_file(f["id"], "shared", n, f.get("filename"), f.get("bytes"), sha, f.get("created_at"))
        log.info("uploaded shared input for %d tokens: %s (%s bytes)", n, f["id"], f.get("bytes"))
        out[n] = rt.store.get_shared_file(n, sha)
    return out


async def upload_individual_files(rt: Runtime, jobs: list[dict[str, Any]], concurrency: int = 10) -> int:
    """Fallback when input-file reuse is unsupported: one file per job (custom_id = observation_id)."""
    assert rt.api is not None
    sem = asyncio.Semaphore(concurrency)
    uploaded = 0

    async def one(job: dict[str, Any]) -> None:
        nonlocal uploaded
        if rt.store.get_individual_file(job["observation_id"]):
            return
        line = build_request_line(rt.cfg, job["observation_id"], job["api_max_output_tokens"])
        path = os.path.join(rt.cfg.inputs_dir, "individual", f"{job['observation_id']}.jsonl")
        data, sha = write_jsonl_file(path, [line])
        validate_jsonl_bytes(data, rt.cfg.endpoint, expect_lines=1)
        async with sem:
            res = await rt.api.upload_file(os.path.basename(path), data, purpose="batch")
        if not res.ok:
            raise RuntimeError(f"upload failed for {job['observation_id']}: {res.http_status} {res.error_message}")
        f = res.data
        rt.store.add_file(f["id"], "individual", job["api_max_output_tokens"], f.get("filename"), f.get("bytes"), sha,
                          f.get("created_at"), observation_id=job["observation_id"])
        rt.store.set_job_input_file(job["observation_id"], f["id"])
        uploaded += 1

    await asyncio.gather(*(one(j) for j in jobs))
    return uploaded


async def prepare(args: Any, api=None) -> dict[str, Any]:
    existing = None
    cfg_path = getattr(args, "config", None) or "config/experiment.json"
    if os.path.exists(cfg_path) and not getattr(args, "reset_config", False):
        existing = ExperimentConfig.load(cfg_path)
    cfg = build_config(args, existing)
    cfg.config_path = cfg_path
    cfg.save(cfg_path)
    cfg.save(os.path.join(os.path.dirname(cfg_path) or ".", "experiment.example.json"))

    rt = Runtime.open(cfg, api=api, need_api=not getattr(args, "no_upload", False))
    try:
        rt.store.set_meta("experiment_id", cfg.experiment_id)
        rt.store.set_meta("prepared_at", iso_now())

        # 1-3. inputs + uploads
        levels_all = sorted(set(cfg.output_token_levels) | set(cfg.pilot_levels))
        shared: dict[int, dict[str, Any]] = {}
        if rt.api is not None:
            shared = await upload_shared_files(rt, levels_all)

        # 7. manifest (deterministic under the seed)
        custom_ids = {n: shared_custom_id(cfg, n) for n in cfg.output_token_levels}
        rows = build_manifest(cfg.output_token_levels, cfg.runs_per_group, cfg.seed, custom_ids, phase="prod")
        problems = check_manifest(rows, cfg.output_token_levels, cfg.runs_per_group)
        if problems:
            raise RuntimeError("manifest invalid: " + "; ".join(problems))
        write_manifest_csv(rows, cfg.manifest_path)

        # 8. persistent DB: insert (idempotent). Verify an existing DB matches the manifest.
        for r in rows:
            r["input_file_id"] = shared[r["requested_output_tokens"]]["file_id"] if (cfg.file_mode == "shared" and shared) else None
        inserted = rt.store.insert_jobs(rows, cfg.experiment_id)
        existing_jobs = rt.store.list_jobs(phases=["prod"])
        mismatches = 0
        by_id = {j["observation_id"]: j for j in existing_jobs}
        for r in rows:
            j = by_id.get(r["observation_id"])
            if j is None or j["launch_position"] != r["launch_position"] or j["requested_output_tokens"] != r["requested_output_tokens"]:
                mismatches += 1
            elif cfg.file_mode == "shared" and shared and not j.get("input_file_id"):
                rt.store.set_job_input_file(r["observation_id"], r["input_file_id"])
        if mismatches:
            raise RuntimeError(f"{mismatches} manifest rows disagree with the existing DB (different seed or levels?). "
                               "Use a fresh data_dir rather than overwriting an experiment in progress.")
        if len(existing_jobs) != len(rows):
            raise RuntimeError(f"DB has {len(existing_jobs)} production jobs but manifest has {len(rows)}")

        # 6. individual files fallback
        individual_uploaded = 0
        if cfg.file_mode == "individual" and rt.api is not None:
            individual_uploaded = await upload_individual_files(rt, rt.store.list_jobs(phases=["prod"]))

        # 9. cost check (worst case: every request emits its full limit)
        level_counts = {n: cfg.runs_per_group for n in cfg.output_token_levels}
        proj = project_cost(level_counts, cfg.estimated_input_tokens_per_request, cfg.pricing, cfg.max_cost_usd, cfg.cost_safety_margin)
        enforce_ceiling(proj)

        summary = {
            "prepared_at": iso_now(),
            "experiment_id": cfg.experiment_id,
            "model": cfg.model,
            "levels": cfg.output_token_levels,
            "pilot_levels": cfg.pilot_levels,
            "runs_per_group": cfg.runs_per_group,
            "total_production_jobs": len(rows),
            "seed": cfg.seed,
            "file_mode": cfg.file_mode,
            "shared_files": {str(n): f["file_id"] for n, f in shared.items()},
            "individual_files_uploaded_now": individual_uploaded,
            "jobs_inserted_now": inserted,
            "jobs_in_db": len(existing_jobs),
            "manifest_path": cfg.manifest_path,
            "cost_projection": proj.to_dict(),
            "batch_creations_made": 0,
        }
        os.makedirs(cfg.processed_dir, exist_ok=True)
        with open(os.path.join(cfg.processed_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("prepare complete: %d jobs, projected max cost $%.4f (ceiling $%.2f)", len(rows), proj.total_with_margin_usd, cfg.max_cost_usd)
        return summary
    finally:
        await rt.aclose()
