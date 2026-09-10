"""Batch JSONL input construction and validation."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .config import ExperimentConfig


class JsonlValidationError(ValueError):
    pass


def build_request_body(cfg: ExperimentConfig, max_output_tokens: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.model,
        "input": cfg.prompt,
        "max_output_tokens": int(max_output_tokens),
    }
    if cfg.reasoning_effort:
        body["reasoning"] = {"effort": cfg.reasoning_effort}
    return body


def build_request_line(cfg: ExperimentConfig, custom_id: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": cfg.endpoint,
        "body": build_request_body(cfg, max_output_tokens),
    }


def shared_custom_id(cfg: ExperimentConfig, max_output_tokens: int) -> str:
    return f"{cfg.experiment_id}:t{int(max_output_tokens):05d}"


def serialize_lines(lines: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(l, separators=(",", ":"), ensure_ascii=False) for l in lines) + "\n").encode("utf-8")


def validate_jsonl_bytes(data: bytes, endpoint: str, expect_lines: int | None = None) -> list[dict[str, Any]]:
    """Validate that data is well-formed Batch JSONL for the endpoint. Returns parsed lines."""
    if not data:
        raise JsonlValidationError("empty file")
    text = data.decode("utf-8")
    parsed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(text.split("\n"), start=1):
        if raw == "":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise JsonlValidationError(f"line {i}: invalid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise JsonlValidationError(f"line {i}: not an object")
        for key in ("custom_id", "method", "url", "body"):
            if key not in obj:
                raise JsonlValidationError(f"line {i}: missing '{key}'")
        if not isinstance(obj["custom_id"], str) or not obj["custom_id"]:
            raise JsonlValidationError(f"line {i}: custom_id must be a non-empty string")
        if obj["custom_id"] in seen_ids:
            raise JsonlValidationError(f"line {i}: duplicate custom_id {obj['custom_id']}")
        seen_ids.add(obj["custom_id"])
        if obj["method"] != "POST":
            raise JsonlValidationError(f"line {i}: method must be POST")
        if obj["url"] != endpoint:
            raise JsonlValidationError(f"line {i}: url {obj['url']!r} != endpoint {endpoint!r}")
        body = obj["body"]
        if not isinstance(body, dict) or "model" not in body:
            raise JsonlValidationError(f"line {i}: body must be an object with a model")
        if endpoint == "/v1/responses":
            if "input" not in body:
                raise JsonlValidationError(f"line {i}: body.input missing")
            mot = body.get("max_output_tokens")
            if not isinstance(mot, int) or isinstance(mot, bool) or mot <= 0:
                raise JsonlValidationError(f"line {i}: body.max_output_tokens must be a positive integer")
        parsed.append(obj)
    if not parsed:
        raise JsonlValidationError("no request lines")
    if expect_lines is not None and len(parsed) != expect_lines:
        raise JsonlValidationError(f"expected {expect_lines} lines, found {len(parsed)}")
    return parsed


def write_jsonl_file(path: str, lines: list[dict[str, Any]]) -> tuple[bytes, str]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = serialize_lines(lines)
    with open(path, "wb") as f:
        f.write(data)
    return data, hashlib.sha256(data).hexdigest()
