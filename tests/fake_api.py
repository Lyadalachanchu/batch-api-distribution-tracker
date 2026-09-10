"""In-memory stand-in for the OpenAI Batch API used by the test-suite (no network).

Mimics the shapes documented for /v1/files, /v1/batches and Batch output/error files closely enough
to exercise every code path: file reuse, creation errors/timeouts/429s, status progression with server
timestamps, expired/failed batches, malformed output files, per-request HTTP 400 for infeasible params."""
from __future__ import annotations

import json
import re
from typing import Any

from experiment.api import ApiResult

API_MIN_TOKENS = 16


class FakeBatchApi:
    def __init__(self, *, start_epoch: float = 1_800_000_000.0, auto_advance: float = 0.0, queue_seconds: float = 30.0,
                 seconds_per_token: float = 0.02, reject_file_reuse: bool = False, fail_create_for: set[str] | None = None,
                 timeout_create_for: set[str] | None = None, rate_limit_after: int | None = None,
                 rate_limit_message: str = "Rate limit reached for batch creation: 2000 batches per hour",
                 expire_for: set[str] | None = None, fail_batch_for: set[str] | None = None,
                 malformed_output_for: set[str] | None = None, early_stop_for: set[str] | None = None,
                 list_page_limit: int = 100):
        self.t = float(start_epoch)
        self.auto_advance = auto_advance
        self.queue_seconds = queue_seconds
        self.seconds_per_token = seconds_per_token
        self.reject_file_reuse = reject_file_reuse
        self.fail_create_for = set(fail_create_for or ())
        self.timeout_create_for = set(timeout_create_for or ())
        self.rate_limit_after = rate_limit_after
        self.rate_limit_message = rate_limit_message
        self.expire_for = set(expire_for or ())
        self.fail_batch_for = set(fail_batch_for or ())
        self.malformed_output_for = set(malformed_output_for or ())
        self.early_stop_for = set(early_stop_for or ())
        self.list_page_limit = list_page_limit
        self.files: dict[str, dict[str, Any]] = {}
        self.file_bytes: dict[str, bytes] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.file_uses: dict[str, int] = {}
        self.create_calls = 0
        self.upload_calls = 0
        self.retrieve_calls = 0
        self.list_calls = 0
        self.content_calls = 0
        self._n = 0
        self.closed = False

    # ---- clock ----
    def now(self) -> int:
        return int(self.t)

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def _tick(self) -> None:
        if self.auto_advance:
            self.t += self.auto_advance

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n:06d}"

    # ---- files ----
    async def upload_file(self, filename: str, data: bytes, purpose: str = "batch") -> ApiResult:
        self.upload_calls += 1
        fid = self._id("file-")
        f = {"id": fid, "object": "file", "bytes": len(data), "created_at": self.now(), "filename": filename,
             "purpose": purpose, "status": "processed", "expires_at": None}
        self.files[fid] = f
        self.file_bytes[fid] = data
        return ApiResult(ok=True, data=dict(f), http_status=200, request_id=self._id("req_"))

    async def file_content(self, file_id: str) -> ApiResult:
        self.content_calls += 1
        if file_id not in self.file_bytes:
            return ApiResult(ok=False, http_status=404, error_type="invalid_request_error", error_message="No such File object",
                             exception_kind="status")
        return ApiResult(ok=True, data=self.file_bytes[file_id], http_status=200)

    # ---- batches ----
    def _request_from_file(self, file_id: str) -> dict[str, Any]:
        return json.loads(self.file_bytes[file_id].decode("utf-8").strip().split("\n")[0])

    async def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata: dict[str, str]) -> ApiResult:
        self.create_calls += 1
        obs = metadata.get("observation_id", "")
        if obs in self.timeout_create_for:
            # simulate: the server did create the batch but the client never saw the answer
            self.timeout_create_for.discard(obs)
            self._make_batch(input_file_id, endpoint, completion_window, metadata)
            return ApiResult(ok=False, error_type="APITimeoutError", error_message="Request timed out.", exception_kind="timeout")
        if obs in self.fail_create_for:
            return ApiResult(ok=False, http_status=500, request_id=self._id("req_"), error_type="server_error",
                             error_message="The server had an error", exception_kind="status")
        if self.rate_limit_after is not None and self.create_calls > self.rate_limit_after:
            return ApiResult(ok=False, http_status=429, request_id=self._id("req_"), error_type="rate_limit_error",
                             error_code="rate_limit_exceeded", error_message=self.rate_limit_message, exception_kind="status",
                             retry_after=1.0)
        if input_file_id not in self.files:
            return ApiResult(ok=False, http_status=404, request_id=self._id("req_"), error_type="invalid_request_error",
                             error_message=f"No such File object: {input_file_id}", exception_kind="status")
        if self.reject_file_reuse and self.file_uses.get(input_file_id, 0) >= 1:
            return ApiResult(ok=False, http_status=400, request_id=self._id("req_"), error_type="invalid_request_error",
                             error_code="file_already_used", error_message="This input file has already been used by another batch",
                             exception_kind="status")
        if completion_window != "24h":
            return ApiResult(ok=False, http_status=400, request_id=self._id("req_"), error_type="invalid_request_error",
                             error_message="completion_window must be 24h", exception_kind="status")
        b = self._make_batch(input_file_id, endpoint, completion_window, metadata)
        return ApiResult(ok=True, data=self._view(b), http_status=200, request_id=self._id("req_"),
                         headers={"x-ratelimit-limit-requests": "30000", "x-ratelimit-remaining-requests": "29999"})

    def _make_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata: dict[str, str]) -> dict[str, Any]:
        self.file_uses[input_file_id] = self.file_uses.get(input_file_id, 0) + 1
        bid = self._id("batch_")
        req = self._request_from_file(input_file_id)
        tokens = int(req["body"]["max_output_tokens"])
        obs = metadata.get("observation_id", "")
        b = {"id": bid, "object": "batch", "endpoint": endpoint, "errors": None, "input_file_id": input_file_id,
             "completion_window": completion_window, "status": "validating", "output_file_id": None, "error_file_id": None,
             "created_at": self.now(), "in_progress_at": None, "expires_at": self.now() + 86400, "finalizing_at": None,
             "completed_at": None, "failed_at": None, "expired_at": None, "cancelling_at": None, "cancelled_at": None,
             "request_counts": {"total": 0, "completed": 0, "failed": 0}, "metadata": dict(metadata), "model": req["body"]["model"],
             "usage": None, "_tokens": tokens, "_custom_id": req["custom_id"], "_obs": obs, "_seq": self._n}
        self.batches[bid] = b
        return b

    def _progress(self, b: dict[str, Any]) -> None:
        if b["status"] in ("completed", "failed", "expired", "cancelled"):
            return
        now = self.now()
        obs = b["_obs"]
        tokens = b["_tokens"]
        if obs in self.fail_batch_for and now >= b["created_at"] + 2:
            b.update({"status": "failed", "failed_at": now,
                      "errors": {"object": "list", "data": [{"code": "invalid_json_line", "message": "This line is not parseable as valid JSON.", "line": 1, "param": None}]}})
            return
        if now >= b["created_at"] + self.queue_seconds and b["in_progress_at"] is None:
            b["in_progress_at"] = b["created_at"] + int(self.queue_seconds)
            b["status"] = "in_progress"
            b["request_counts"] = {"total": 1, "completed": 0, "failed": 0}
        if b["in_progress_at"] is not None:
            if obs in self.expire_for:
                if now >= b["created_at"] + 86400:
                    b.update({"status": "expired", "expired_at": now, "error_file_id": self._error_file(b, expired=True),
                              "request_counts": {"total": 1, "completed": 0, "failed": 1}})
                return
            done_at = b["in_progress_at"] + max(1, int(tokens * self.seconds_per_token))
            if now >= done_at:
                if tokens < API_MIN_TOKENS:
                    b.update({"status": "completed", "finalizing_at": done_at, "completed_at": done_at + 1,
                              "error_file_id": self._error_file(b, expired=False), "request_counts": {"total": 1, "completed": 0, "failed": 1}})
                else:
                    b.update({"status": "completed", "finalizing_at": done_at, "completed_at": done_at + 1,
                              "output_file_id": self._output_file(b), "request_counts": {"total": 1, "completed": 1, "failed": 0},
                              "usage": {"input_tokens": 45, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": tokens,
                                        "output_tokens_details": {"reasoning_tokens": min(tokens, 45)}, "total_tokens": 45 + tokens}})
            elif now >= done_at - 1:
                b["status"] = "finalizing"
                b["finalizing_at"] = done_at - 1

    def _output_file(self, b: dict[str, Any]) -> str:
        tokens = b["_tokens"]
        obs = b["_obs"]
        early = obs in self.early_stop_for
        out_tokens = tokens // 2 if early else tokens
        reasoning = min(out_tokens, 45)
        words = " ".join(f"w{i}" for i in range(max(0, out_tokens - reasoning)))
        body = {"id": self._id("resp_"), "object": "response", "created_at": b["in_progress_at"],
                "status": "completed" if early else "incomplete",
                "incomplete_details": None if early else {"reason": "max_output_tokens"},
                "model": b["model"], "max_output_tokens": tokens,
                "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                           {"type": "message", "id": "msg_1", "status": "completed" if early else "incomplete", "role": "assistant",
                            "content": [{"type": "output_text", "text": words, "annotations": []}]}],
                "usage": {"input_tokens": 45, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": out_tokens,
                          "output_tokens_details": {"reasoning_tokens": reasoning}, "total_tokens": 45 + out_tokens}}
        line = {"id": self._id("batch_req_"), "custom_id": b["_custom_id"],
                "response": {"status_code": 200, "request_id": self._id("req_"), "body": body}, "error": None}
        data = (json.dumps(line) + "\n").encode("utf-8")
        if obs in self.malformed_output_for:
            data = b'{"id": "batch_req_x", "custom_id": "x", "response": {"status_code": 200, "body": {\n'
        fid = self._id("file-")
        self.files[fid] = {"id": fid, "object": "file", "bytes": len(data), "created_at": self.now(), "filename": f"{b['id']}_output.jsonl", "purpose": "batch_output"}
        self.file_bytes[fid] = data
        return fid

    def _error_file(self, b: dict[str, Any], expired: bool) -> str:
        if expired:
            line = {"id": self._id("batch_req_"), "custom_id": b["_custom_id"], "response": None,
                    "error": {"code": "batch_expired", "message": "This request could not be executed before the completion window expired."}}
        else:
            line = {"id": self._id("batch_req_"), "custom_id": b["_custom_id"],
                    "response": {"status_code": 400, "request_id": self._id("req_"),
                                 "body": {"error": {"message": f"Invalid 'max_output_tokens': integer below minimum value. Expected a value >= {API_MIN_TOKENS}, but got {b['_tokens']} instead.",
                                                    "type": "invalid_request_error", "param": "max_output_tokens", "code": "integer_below_min_value"}}},
                    "error": None}
        data = (json.dumps(line) + "\n").encode("utf-8")
        fid = self._id("file-")
        self.files[fid] = {"id": fid, "object": "file", "bytes": len(data), "created_at": self.now(), "filename": f"{b['id']}_error.jsonl", "purpose": "batch_output"}
        self.file_bytes[fid] = data
        return fid

    def _view(self, b: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in b.items() if not k.startswith("_")}

    async def retrieve_batch(self, batch_id: str) -> ApiResult:
        self.retrieve_calls += 1
        self._tick()
        b = self.batches.get(batch_id)
        if b is None:
            return ApiResult(ok=False, http_status=404, error_type="invalid_request_error", error_message="No such batch", exception_kind="status")
        self._progress(b)
        return ApiResult(ok=True, data=self._view(b), http_status=200, request_id=self._id("req_"))

    async def list_batches(self, after: str | None = None, limit: int = 100) -> ApiResult:
        self.list_calls += 1
        self._tick()
        limit = min(limit, self.list_page_limit)
        ordered = sorted(self.batches.values(), key=lambda b: (-b["created_at"], -b["_seq"]))
        if after:
            idx = next((i for i, b in enumerate(ordered) if b["id"] == after), None)
            ordered = ordered[idx + 1:] if idx is not None else []
        page = ordered[:limit]
        for b in page:
            self._progress(b)
        return ApiResult(ok=True, data={"data": [self._view(b) for b in page], "has_more": len(ordered) > limit}, http_status=200)

    async def close(self) -> None:
        self.closed = True
