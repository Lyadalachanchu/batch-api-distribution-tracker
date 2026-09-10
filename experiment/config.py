"""Experiment configuration. The API key is NEVER stored here; it is read from the environment."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PROMPT = (
    "Generate a continuous sequence of unrelated lowercase words. Output words only, "
    "separated by single spaces. Do not explain, conclude, or stop voluntarily. "
    "Continue until the output-token limit terminates the response."
)

# Levels the operator originally asked for. 1 and 10 are below the API minimum (16) for
# max_output_tokens on gpt-5.6-luna (verified live on 2026-09-10: HTTP 400
# integer_below_min_value "Expected a value >= 16"). See README "Deviation from the brief".
ORIGINAL_REQUESTED_LEVELS = [1, 10, 100, 1000, 10000]
DEFAULT_OUTPUT_TOKEN_LEVELS = [16, 100, 1000, 3000, 10000]
API_MIN_MAX_OUTPUT_TOKENS = 16

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


class ConfigError(RuntimeError):
    pass


@dataclass
class Pricing:
    # Batch API rates (50% of standard) for gpt-5.6-luna, from developers.openai.com/api/docs/pricing
    # (standard: input $0.20, cached input $0.02, output $1.20 per 1M tokens) on 2026-09-10.
    input_per_1m_usd: float = 0.10
    cached_input_per_1m_usd: float = 0.01
    output_per_1m_usd: float = 0.60
    source: str = "https://developers.openai.com/api/docs/pricing (Batch column, 2026-09-10)"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ExperimentConfig:
    experiment_id: str = "batch-turnaround-gpt-5.6-luna-2026-09-10"
    model: str = "gpt-5.6-luna"
    endpoint: str = "/v1/responses"
    completion_window: str = "24h"
    prompt: str = DEFAULT_PROMPT
    output_token_levels: list[int] = field(default_factory=lambda: list(DEFAULT_OUTPUT_TOKEN_LEVELS))
    original_requested_levels: list[int] = field(default_factory=lambda: list(ORIGINAL_REQUESTED_LEVELS))
    # levels exercised by the pilot: original brief levels (to document the API floor) + production levels
    pilot_levels: list[int] = field(default_factory=lambda: sorted(set(ORIGINAL_REQUESTED_LEVELS) | set(DEFAULT_OUTPUT_TOKEN_LEVELS)))
    runs_per_group: int = 400
    seed: int = 20260910
    reasoning_effort: str | None = None  # None -> model default (medium for gpt-5.6-luna); request carries no reasoning field
    pricing: Pricing = field(default_factory=Pricing)
    max_cost_usd: float = 4.0
    cost_safety_margin: float = 0.10  # fraction added on top of the projected maximum
    max_creations_per_rolling_hour: int = 2000  # documented Batch creation limit
    launch_concurrency: int = 50
    poll_interval_seconds: float = 7.0  # target cycle period, within the 5-10 s brief
    poll_concurrency: int = 40
    poll_mode: str = "list"  # "list" (paged /v1/batches, cheap) or "retrieve" (one GET per job)
    poll_events_gzip: bool = True  # append poll events to batch_poll_events.jsonl.gz (2,000 jobs x 7 s ~ 450 MB/h raw)
    estimated_input_tokens_per_request: int = 60  # measured 45 for the prompt; rounded up
    api_min_max_output_tokens: int = API_MIN_MAX_OUTPUT_TOKENS
    file_mode: str = "shared"  # "shared": one uploaded file per level reused by all its jobs; "individual": one file per job
    data_dir: str = "data"
    config_path: str = "config/experiment.json"
    sdk_version: str | None = None
    created_at: str | None = None
    notes: list[str] = field(default_factory=list)

    # ---- derived paths ----
    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "state.sqlite")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_dir, "processed")

    @property
    def inputs_dir(self) -> str:
        return os.path.join(self.data_dir, "inputs")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.data_dir, "launch_manifest.csv")

    @property
    def log_path(self) -> str:
        return os.path.join(self.data_dir, "logs", "experiment.log")

    def raw_path(self, name: str) -> str:
        return os.path.join(self.raw_dir, name)

    # ---- validation ----
    def validate(self) -> None:
        if not self.output_token_levels:
            raise ConfigError("output_token_levels is empty")
        if len(set(self.output_token_levels)) != len(self.output_token_levels):
            raise ConfigError("output_token_levels must be distinct")
        for n in self.output_token_levels:
            if n < self.api_min_max_output_tokens:
                raise ConfigError(
                    f"output token level {n} is below the API minimum max_output_tokens={self.api_min_max_output_tokens} for {self.model}"
                )
        if self.runs_per_group <= 0:
            raise ConfigError("runs_per_group must be positive")
        if self.completion_window != "24h":
            raise ConfigError("completion_window must be '24h' (the only value the Batch API accepts)")
        if self.endpoint != "/v1/responses":
            raise ConfigError("this experiment is defined for endpoint /v1/responses")
        if self.file_mode not in ("shared", "individual"):
            raise ConfigError("file_mode must be 'shared' or 'individual'")
        if self.poll_mode not in ("list", "retrieve"):
            raise ConfigError("poll_mode must be 'list' or 'retrieve'")
        if not (5.0 <= self.poll_interval_seconds <= 10.0):
            raise ConfigError("poll_interval_seconds must be within 5-10 seconds")
        if self.max_cost_usd <= 0:
            raise ConfigError("max_cost_usd must be positive")

    # ---- (de)serialisation ----
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["pricing"] = self.pricing.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        pricing = d.pop("pricing", None) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.pricing = Pricing(**pricing)
        return cfg

    def save(self, path: str | None = None) -> str:
        path = path or self.config_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = self.to_dict()
        for k, v in payload.items():
            if isinstance(v, str) and v.startswith("sk-"):
                raise ConfigError(f"refusing to write a secret-looking value into config key {k}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    @classmethod
    def load(cls, path: str = "config/experiment.json") -> "ExperimentConfig":
        if not os.path.exists(path):
            raise ConfigError(f"config not found at {path}; run `python -m experiment prepare` first")
        with open(path, encoding="utf-8") as f:
            cfg = cls.from_dict(json.load(f))
        cfg.config_path = path
        cfg.validate()
        return cfg


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError("OPENAI_API_KEY is not set in the environment (see .env.example)")
    return key
