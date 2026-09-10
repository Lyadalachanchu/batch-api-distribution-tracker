"""Secret redaction helpers. Every log line and every raw artifact passes through here."""
from __future__ import annotations

import re
from typing import Any

# OpenAI keys: "sk-..." (project keys look like sk-proj-...). Be generous: any sk- token >= 12 chars.
_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{12,}")

REDACTED = "sk-***REDACTED***"


def redact_text(text: str) -> str:
    if not text:
        return text
    text = _KEY_RE.sub(REDACTED, text)
    text = _BEARER_RE.sub(lambda m: m.group(1) + "***REDACTED***", text)
    return text


def contains_secret(text: str) -> bool:
    """True if the text still contains something that looks like a live key.

    The redaction marker contains '*' which the key regex does not accept, so redacted
    text never matches."""
    if not text:
        return False
    return bool(_KEY_RE.search(text)) or bool(_BEARER_RE.search(text))


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings inside dicts/lists/tuples; other types pass through."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
