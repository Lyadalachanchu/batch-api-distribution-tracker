"""collect: retrieve and preserve output/error files for terminal jobs; parse usage into results."""
from __future__ import annotations

import json
import logging
from typing import Any

from .observations import parse_output_record
from .runtime import Runtime
from .timeutil import iso_now

log = logging.getLogger(__name__)


async def _fetch_lines(rt: Runtime, file_id: str) -> tuple[list[tuple[int, str, Any]], str | None]:
    assert rt.api is not None
    res = await rt.api.file_content(file_id)
    if not res.ok:
        return [], f"{res.http_status} {res.error_type}: {res.error_message}"
    text = res.data.decode("utf-8", errors="replace")
    out: list[tuple[int, str, Any]] = []
    for i, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            out.append((i, line, json.loads(line)))
        except json.JSONDecodeError as e:
            out.append((i, line, {"__malformed__": True, "error": str(e)}))
    return out, None


async def collect_job(rt: Runtime, job: dict[str, Any], force: bool = False) -> dict[str, Any]:
    obs = job["observation_id"]
    result: dict[str, Any] = {"batch_id": job.get("batch_id"), "collected_at": iso_now()}
    fetched_any = False
    for kind, fid, writer in (("output", job.get("output_file_id"), rt.responses), ("error", job.get("error_file_id"), rt.errors)):
        if not fid:
            continue
        lines, fetch_err = await _fetch_lines(rt, fid)
        if fetch_err:
            result["parse_error"] = f"{kind} file fetch failed: {fetch_err}"
            log.warning("fetch %s file %s for %s failed: %s", kind, fid, obs, fetch_err)
            continue
        fetched_any = True
        for line_no, raw, obj in lines:
            malformed = isinstance(obj, dict) and obj.get("__malformed__")
            rec: dict[str, Any] = {"observation_id": obs, "batch_id": job.get("batch_id"), "file_id": fid, "kind": kind,
                                   "line_no": line_no, "fetched_at": iso_now(), "recollect": bool(force), "record": obj}
            if malformed:
                rec["raw"] = raw  # keep the exact bytes of an unparseable line
            writer.append(rec)
            if malformed:
                result.setdefault("parse_error", f"{kind} line {line_no} malformed JSON: {obj.get('error')}")
                continue
            parsed = parse_output_record(obj, raw)
            # the output file wins over the error file when both exist (one-request batches have only one)
            if kind == "output" or result.get("response_status") is None:
                for k, v in parsed.items():
                    if v is not None or k not in result:
                        result[k] = v
                result["source_file_id"] = fid
                result["source_kind"] = kind
    if not fetched_any:
        errs = job.get("batch_errors")
        try:
            errs_obj = json.loads(errs) if errs else None
        except json.JSONDecodeError:
            errs_obj = None
        if errs_obj and isinstance(errs_obj, dict) and errs_obj.get("data"):
            first = errs_obj["data"][0]
            result["error_type"] = "batch_validation_error"
            result["error_code"] = first.get("code")
            result["error_message"] = first.get("message")
            result["source_kind"] = "batch.errors"
        elif job.get("status") in ("expired", "cancelled", "failed"):
            result["error_type"] = f"batch_{job.get('status')}"
            result["error_message"] = f"batch reached {job.get('status')} with no output or error file"
            result["source_kind"] = "batch.status"
        else:
            result["parse_error"] = result.get("parse_error") or "no output_file_id and no error_file_id"
    rt.store.upsert_result(obs, result)
    rt.store.mark_collected(obs)
    return result


async def collect_all(rt: Runtime, phases: list[str] | None = None, force: bool = False) -> dict[str, Any]:
    jobs = rt.store.list_jobs(phases=phases, terminal_uncollected=not force)
    if force:
        jobs = [j for j in rt.store.list_jobs(phases=phases) if j.get("terminal")]
    n_ok = n_err = 0
    for j in jobs:
        r = await collect_job(rt, j, force=force)
        if r.get("parse_error"):
            n_err += 1
        else:
            n_ok += 1
    return {"collected": n_ok, "with_parse_error": n_err, "total": len(jobs)}
