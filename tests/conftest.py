from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.config import ExperimentConfig  # noqa: E402
from tests.fake_api import FakeBatchApi  # noqa: E402


def make_args(**kw):
    base = dict(config=None, execute=False, max_cost_usd=None, concurrency=None, wait=False, resume=False,
                skip_pilot_gate=False, phases=None, once=False, max_minutes=None, no_collect=False, poll_mode=None,
                interval=0.01, force=False, timeout_minutes=5, again=False, report=None, bootstrap=50, out=None,
                runs_per_group=None, seed=None, levels=None, experiment_id=None, model=None, file_mode=None,
                launch_concurrency=None, poll_interval_seconds=None, reasoning_effort=None, data_dir=None,
                no_upload=False, reset_config=False)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated working directory so config/, data/, reports/ land under tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    return tmp_path


@pytest.fixture
def small_cfg(workdir):
    cfg = ExperimentConfig(experiment_id="exp-test", output_token_levels=[16, 100, 1000], runs_per_group=3, seed=7,
                           pilot_levels=[1, 10, 16, 100, 1000], data_dir=str(workdir / "data"),
                           config_path=str(workdir / "config" / "experiment.json"))
    cfg.validate()
    cfg.save()
    return cfg


@pytest.fixture
def fake_api():
    return FakeBatchApi(auto_advance=15.0)
