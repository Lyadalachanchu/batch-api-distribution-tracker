"""status: human-readable snapshot of the experiment state (no API calls unless --api)."""
from __future__ import annotations

import json
from typing import Any

from .config import ExperimentConfig
from .limits import WINDOW_SECONDS
from .runtime import Runtime
from .timeutil import epoch_now


def build_status(rt: Runtime) -> dict[str, Any]:
    from .launch import spent_or_committed_usd
    st = rt.store
    out: dict[str, Any] = {
        "experiment_id": rt.cfg.experiment_id,
        "model": rt.cfg.model,
        "levels": rt.cfg.output_token_levels,
        "prepared_at": st.get_meta("prepared_at"),
        "pilot_status": st.get_meta("pilot_status"),
        "pilot_started_at": st.get_meta("pilot_started_at"),
        "launch_started_at": st.get_meta("launch_started_at"),
        "launch_finished_at": st.get_meta("launch_finished_at"),
        "monitor_cycles": st.get_meta("monitor_cycles", 0),
        "creations_last_hour_local": st.count_creations_since(epoch_now() - WINDOW_SECONDS),
        "cost": spent_or_committed_usd(rt),
    }
    for phase in ("pilot", "prod", "replacement"):
        jobs = st.list_jobs(phases=[phase])
        if not jobs:
            continue
        by_status: dict[str, int] = {}
        by_group: dict[str, dict[str, int]] = {}
        for j in jobs:
            s = j.get("status") if j.get("batch_id") else f"creation_{j['creation_state']}"
            by_status[s] = by_status.get(s, 0) + 1
            g = by_group.setdefault(str(j["requested_output_tokens"]), {})
            g[s] = g.get(s, 0) + 1
        out[phase] = {"jobs": len(jobs), "by_status": by_status, "by_group": by_group,
                      "terminal": sum(1 for j in jobs if j.get("terminal")), "collected": sum(1 for j in jobs if j.get("collected"))}
    return out


async def status(args: Any, api=None) -> dict[str, Any]:
    cfg = ExperimentConfig.load(getattr(args, "config", None) or "config/experiment.json")
    rt = Runtime.open(cfg, api=api, need_api=False)
    try:
        out = build_status(rt)
        print(json.dumps(out, indent=2, default=str))
        return out
    finally:
        await rt.aclose()
