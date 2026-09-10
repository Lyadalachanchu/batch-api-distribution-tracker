"""Thin async wrapper around the official openai SDK (v3.x) that returns plain dicts plus HTTP metadata.

Creation calls use a client with max_retries=0 so that retry accounting is entirely ours; polling/collection
calls use bounded SDK retries. Nothing here logs request headers.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

RATE_LIMIT_HEADER_PREFIXES = ("x-ratelimit-", "retry-after", "openai-processing-ms")


@dataclass
class ApiResult:
    ok: bool
    data: Any = None
    http_status: int | None = None
    request_id: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    exception_kind: str | None = None  # status | timeout | connection | other
    headers: dict[str, str] = field(default_factory=dict)
    retry_after: float | None = None

    @property
    def outcome_unknown(self) -> bool:
        """True when the server may have processed the request even though we got no answer."""
        return (not self.ok) and self.exception_kind in ("timeout", "connection")


class BatchApi(Protocol):
    async def upload_file(self, filename: str, data: bytes, purpose: str = "batch") -> ApiResult: ...
    async def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata: dict[str, str]) -> ApiResult: ...
    async def retrieve_batch(self, batch_id: str) -> ApiResult: ...
    async def list_batches(self, after: str | None = None, limit: int = 100) -> ApiResult: ...
    async def file_content(self, file_id: str) -> ApiResult: ...
    async def close(self) -> None: ...


def _pick_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for k, v in headers.items():
            lk = str(k).lower()
            if lk.startswith(RATE_LIMIT_HEADER_PREFIXES):
                out[lk] = str(v)
    except Exception:
        pass
    return out


class OpenAIBatchApi:
    def __init__(self, api_key: str, timeout: float = 60.0, max_retries_default: int = 3):
        import openai  # local import keeps tests importable without network

        self._openai = openai
        self._create_client = openai.AsyncOpenAI(api_key=api_key, max_retries=0, timeout=timeout)
        self._client = openai.AsyncOpenAI(api_key=api_key, max_retries=max_retries_default, timeout=timeout)

    # ----- error mapping -----
    def _err(self, e: Exception) -> ApiResult:
        o = self._openai
        if isinstance(e, o.APIStatusError):
            body = getattr(e, "body", None)
            err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else (body if isinstance(body, dict) else {})
            headers = _pick_headers(getattr(getattr(e, "response", None), "headers", {}) or {})
            ra = headers.get("retry-after")
            return ApiResult(ok=False, http_status=e.status_code, request_id=getattr(e, "request_id", None),
                             error_type=str(err.get("type") or type(e).__name__), error_code=(str(err.get("code")) if err.get("code") is not None else None),
                             error_message=str(err.get("message") or getattr(e, "message", str(e))), exception_kind="status",
                             headers=headers, retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        if isinstance(e, o.APITimeoutError):
            return ApiResult(ok=False, error_type="APITimeoutError", error_message=str(e), exception_kind="timeout")
        if isinstance(e, o.APIConnectionError):
            return ApiResult(ok=False, error_type="APIConnectionError", error_message=str(e), exception_kind="connection")
        return ApiResult(ok=False, error_type=type(e).__name__, error_message=str(e), exception_kind="other")

    @staticmethod
    def _ok(raw: Any, data: Any) -> ApiResult:
        headers = getattr(raw, "headers", {}) or {}
        rid = None
        try:
            rid = headers.get("x-request-id")
        except Exception:
            pass
        return ApiResult(ok=True, data=data, http_status=getattr(raw, "status_code", None), request_id=rid, headers=_pick_headers(headers))

    # ----- operations -----
    async def upload_file(self, filename: str, data: bytes, purpose: str = "batch") -> ApiResult:
        try:
            raw = await self._client.files.with_raw_response.create(file=(filename, data), purpose=purpose)
            return self._ok(raw, raw.parse().model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            return self._err(e)

    async def create_batch(self, input_file_id: str, endpoint: str, completion_window: str, metadata: dict[str, str]) -> ApiResult:
        try:
            raw = await self._create_client.batches.with_raw_response.create(
                input_file_id=input_file_id, endpoint=endpoint, completion_window=completion_window, metadata=metadata)
            return self._ok(raw, raw.parse().model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            return self._err(e)

    async def retrieve_batch(self, batch_id: str) -> ApiResult:
        try:
            raw = await self._client.batches.with_raw_response.retrieve(batch_id)
            return self._ok(raw, raw.parse().model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            return self._err(e)

    async def list_batches(self, after: str | None = None, limit: int = 100) -> ApiResult:
        try:
            kwargs: dict[str, Any] = {"limit": limit}
            if after:
                kwargs["after"] = after
            raw = await self._client.batches.with_raw_response.list(**kwargs)
            page = raw.parse()
            items = [b.model_dump(mode="json") for b in page.data]
            return self._ok(raw, {"data": items, "has_more": bool(getattr(page, "has_more", False))})
        except Exception as e:  # noqa: BLE001
            return self._err(e)

    async def file_content(self, file_id: str) -> ApiResult:
        try:
            resp = await self._client.files.content(file_id)
            try:
                data = resp.content
            except Exception:
                data = await resp.aread()
            return ApiResult(ok=True, data=bytes(data), http_status=getattr(getattr(resp, "response", None), "status_code", None))
        except Exception as e:  # noqa: BLE001
            return self._err(e)

    async def close(self) -> None:
        await asyncio.gather(self._create_client.close(), self._client.close(), return_exceptions=True)
