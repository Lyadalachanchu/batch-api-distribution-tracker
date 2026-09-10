"""Shared runtime: config + store + API + append-only event writers."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .api import BatchApi, OpenAIBatchApi
from .config import ExperimentConfig, get_api_key
from .events import EventWriter
from .logging_utils import setup_logging
from .store import Store


@dataclass
class Runtime:
    cfg: ExperimentConfig
    store: Store
    api: BatchApi | None
    creation_events: EventWriter
    poll_events: EventWriter
    batch_objects: EventWriter
    responses: EventWriter
    errors: EventWriter

    @classmethod
    def open(cls, cfg: ExperimentConfig, api: BatchApi | None = None, need_api: bool = True) -> "Runtime":
        setup_logging(cfg.log_path)
        os.makedirs(cfg.raw_dir, exist_ok=True)
        os.makedirs(cfg.processed_dir, exist_ok=True)
        if api is None and need_api:
            api = OpenAIBatchApi(get_api_key())
        return cls(
            cfg=cfg,
            store=Store(cfg.db_path),
            api=api,
            creation_events=EventWriter(cfg.raw_path("batch_creation_events.jsonl")),
            poll_events=EventWriter(cfg.raw_path("batch_poll_events.jsonl")),
            batch_objects=EventWriter(cfg.raw_path("batch_objects.jsonl")),
            responses=EventWriter(cfg.raw_path("responses.jsonl")),
            errors=EventWriter(cfg.raw_path("errors.jsonl")),
        )

    async def aclose(self) -> None:
        for w in (self.creation_events, self.poll_events, self.batch_objects, self.responses, self.errors):
            w.close()
        self.store.close()
        if self.api is not None:
            await self.api.close()
