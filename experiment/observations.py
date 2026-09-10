"""Parse Batch output/error lines and build the one-row-per-job observation table."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .cost import actual_cost_usd
from .store import Store
from .timeutil import epoch_to_iso, iso_to_epoch

OBSERVATION_COLUMNS = [
    "experiment_id", "observation_id", "attempt_id", "phase", "parent_observation_id", "is_replacement",
    "requested_output_tokens", "api_max_output_tokens", "actual_output_tokens", "reasoning_tokens",
    "actual_input_tokens", "cached_input_tokens", "total_tokens",
    "randomized_launch_position", "local_create_started_at", "local_create_finished_at", "local_create_duration_ms",
    "batch_id", "input_file_id", "output_file_id", "error_file_id",
    "created_at", "in_progress_at", "finalizing_at", "completed_at", "failed_at", "expired_at", "cancelled_at", "expires_at",
    "created_at_iso", "in_progress_at_iso", "completed_at_iso",
    "status", "creation_state", "turnaround_seconds", "queue_seconds", "active_seconds", "finalizing_seconds",
    "server_created_at_offset_seconds", "local_terminal_seen_at",
    "response_status", "finish_or_incomplete_reason", "early_stop", "http_status", "openai_request_id",
    "error_type", "error_code", "error_message", "create_error_type", "create_error_message",
    "output_text_chars", "output_word_count", "parse_error", "estimated_cost_usd", "valid_observation",
]


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def extract_output_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(str(c.get("text") or ""))
    return "".join(parts)


def parse_output_record(obj: Any, raw_line: str | None = None) -> dict[str, Any]:
    """Map one Batch output/error JSONL record to result fields. Never raises: malformed -> parse_error."""
    r: dict[str, Any] = {"line_sha256": _sha(raw_line) if raw_line is not None else None}
    try:
        if not isinstance(obj, dict):
            r["parse_error"] = f"record is {type(obj).__name__}, not an object"
            return r
        resp = obj.get("response")
        err = obj.get("error")
        if isinstance(resp, dict):
            r["http_status"] = resp.get("status_code")
            r["openai_request_id"] = resp.get("request_id")
            body = resp.get("body")
            if isinstance(body, dict):
                if body.get("object") == "response" or "usage" in body or "status" in body:
                    r["response_status"] = body.get("status")
                    inc = body.get("incomplete_details") or {}
                    r["incomplete_reason"] = inc.get("reason") if isinstance(inc, dict) else None
                    u = body.get("usage") or {}
                    if isinstance(u, dict):
                        r["input_tokens"] = u.get("input_tokens")
                        r["cached_input_tokens"] = (u.get("input_tokens_details") or {}).get("cached_tokens")
                        r["output_tokens"] = u.get("output_tokens")
                        r["reasoning_tokens"] = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
                        r["total_tokens"] = u.get("total_tokens")
                    text = extract_output_text(body)
                    r["output_text_chars"] = len(text)
                    r["output_word_count"] = len(text.split())
                berr = body.get("error")
                if isinstance(berr, dict) and berr:
                    r["error_type"] = berr.get("type")
                    r["error_code"] = berr.get("code")
                    r["error_message"] = berr.get("message")
            elif body is not None:
                r["parse_error"] = "response.body is not an object"
        elif resp is not None:
            r["parse_error"] = "response is not an object"
        if isinstance(err, dict) and err:
            r["error_code"] = err.get("code") or r.get("error_code")
            r["error_message"] = err.get("message") or r.get("error_message")
            r["error_type"] = r.get("error_type") or "batch_request_error"
        if resp is None and not err:
            r["parse_error"] = r.get("parse_error") or "record has neither response nor error"
    except Exception as e:  # noqa: BLE001
        r["parse_error"] = f"{type(e).__name__}: {e}"
    return r


def _diff(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def build_observations(store: Store, cfg: ExperimentConfig) -> pd.DataFrame:
    launch_started_iso = store.get_meta("launch_started_at")
    launch_epoch = iso_to_epoch(launch_started_iso) if launch_started_iso else None
    rows: list[dict[str, Any]] = []
    for j in store.jobs_with_results():
        ca, ip, fz, co = j.get("created_at"), j.get("in_progress_at"), j.get("finalizing_at"), j.get("completed_at")
        fa, ea = j.get("failed_at"), j.get("expired_at")
        end = co if co is not None else (fa if fa is not None else ea)
        out_tokens = j.get("output_tokens")
        resp_status = j.get("response_status")
        req = j.get("requested_output_tokens")
        early_stop = None
        if resp_status is not None and out_tokens is not None:
            early_stop = bool(resp_status == "completed" and out_tokens < req)
        reason = j.get("incomplete_reason") or (resp_status if resp_status else None)
        lcs, lcf = j.get("local_create_started_at"), j.get("local_create_finished_at")
        dur_ms = None
        if lcs and lcf:
            dur_ms = round((iso_to_epoch(lcf) - iso_to_epoch(lcs)) * 1000.0, 3)
        valid = bool(j.get("status") == "completed" and resp_status in ("completed", "incomplete") and out_tokens is not None
                     and not j.get("parse_error"))
        rows.append({
            "experiment_id": j["experiment_id"],
            "observation_id": j["observation_id"],
            "attempt_id": j["attempt_id"],
            "phase": j["phase"],
            "parent_observation_id": j.get("parent_observation_id"),
            "is_replacement": j["phase"] == "replacement",
            "requested_output_tokens": req,
            "api_max_output_tokens": j.get("api_max_output_tokens"),
            "actual_output_tokens": out_tokens,
            "reasoning_tokens": j.get("reasoning_tokens"),
            "actual_input_tokens": j.get("input_tokens"),
            "cached_input_tokens": j.get("cached_input_tokens"),
            "total_tokens": j.get("total_tokens"),
            "randomized_launch_position": j.get("launch_position"),
            "local_create_started_at": lcs,
            "local_create_finished_at": lcf,
            "local_create_duration_ms": dur_ms,
            "batch_id": j.get("batch_id"),
            "input_file_id": j.get("input_file_id"),
            "output_file_id": j.get("output_file_id"),
            "error_file_id": j.get("error_file_id"),
            "created_at": ca, "in_progress_at": ip, "finalizing_at": fz, "completed_at": co,
            "failed_at": fa, "expired_at": ea, "cancelled_at": j.get("cancelled_at"), "expires_at": j.get("expires_at"),
            "created_at_iso": epoch_to_iso(ca), "in_progress_at_iso": epoch_to_iso(ip), "completed_at_iso": epoch_to_iso(co),
            "status": j.get("status") if j.get("batch_id") else ("creation_" + str(j.get("creation_state"))),
            "creation_state": j.get("creation_state"),
            "turnaround_seconds": _diff(co, ca),
            "queue_seconds": _diff(ip, ca),
            "active_seconds": _diff(co, ip),
            "finalizing_seconds": _diff(co, fz),
            "server_created_at_offset_seconds": _diff(ca, launch_epoch) if launch_epoch is not None else None,
            "local_terminal_seen_at": j.get("terminal_seen_at"),
            "response_status": resp_status,
            "finish_or_incomplete_reason": reason,
            "early_stop": early_stop,
            "http_status": j.get("result_http_status") if j.get("result_http_status") is not None else j.get("create_http_status"),
            "openai_request_id": j.get("openai_request_id"),
            "error_type": j.get("result_error_type") or j.get("create_error_type"),
            "error_code": j.get("result_error_code") or j.get("create_error_code"),
            "error_message": j.get("result_error_message") or j.get("create_error_message"),
            "create_error_type": j.get("create_error_type"),
            "create_error_message": j.get("create_error_message"),
            "output_text_chars": j.get("output_text_chars"),
            "output_word_count": j.get("output_word_count"),
            "parse_error": j.get("parse_error"),
            "estimated_cost_usd": actual_cost_usd(j.get("input_tokens"), j.get("cached_input_tokens"), out_tokens, cfg.pricing),
            "valid_observation": valid,
        })
    df = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    return df


def write_observations(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[str, str]:
    os.makedirs(cfg.processed_dir, exist_ok=True)
    csv_path = os.path.join(cfg.processed_dir, "observations.csv")
    pq_path = os.path.join(cfg.processed_dir, "observations.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    return csv_path, pq_path
