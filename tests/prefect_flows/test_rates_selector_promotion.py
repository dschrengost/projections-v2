from __future__ import annotations

import json
from pathlib import Path

from prefect_flows.rates_retrain import promote_rates_task


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_promote_rates_writes_runtime_selector_only_by_default(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    repo_config = tmp_path / "repo" / "config" / "rates_current_run.json"
    runtime_config = data_root / "control_plane" / "model_selectors" / "rates_current_run.json"

    repo_config.parent.mkdir(parents=True, exist_ok=True)
    repo_config.write_text('{"run_id":"old_run","feature_set":"stage5"}\n', encoding="utf-8")

    result = promote_rates_task.fn(
        data_root=data_root,
        source_config_path=repo_config,
        runtime_config_path=runtime_config,
        repo_config_path=repo_config,
        new_run_id="new_run",
        eval_summary_path=None,
        guardrail_result={"passed": True},
        sync_repo_selector=False,
    )

    runtime_payload = _read_json(runtime_config)
    assert runtime_payload["run_id"] == "new_run"
    assert runtime_payload["feature_set"] == "stage5"
    # Repo selector remains unchanged unless explicitly requested.
    assert _read_json(repo_config)["run_id"] == "old_run"
    assert result["runtime_selector_path"] == str(runtime_config)


def test_promote_rates_can_sync_repo_selector(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_config = tmp_path / "source" / "rates_current_run.json"
    repo_config = tmp_path / "repo" / "config" / "rates_current_run.json"
    runtime_config = data_root / "control_plane" / "model_selectors" / "rates_current_run.json"

    source_config.parent.mkdir(parents=True, exist_ok=True)
    source_config.write_text('{"run_id":"old_run","feature_set":"stage5"}\n', encoding="utf-8")

    result = promote_rates_task.fn(
        data_root=data_root,
        source_config_path=source_config,
        runtime_config_path=runtime_config,
        repo_config_path=repo_config,
        new_run_id="new_run",
        eval_summary_path=None,
        guardrail_result={"passed": True},
        sync_repo_selector=True,
    )

    assert _read_json(runtime_config)["run_id"] == "new_run"
    assert _read_json(repo_config)["run_id"] == "new_run"
    assert result["repo_selector_path"] == str(repo_config)
