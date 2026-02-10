from __future__ import annotations

from projections import model_selectors


def test_active_minutes_selector_prefers_runtime_when_present(tmp_path) -> None:
    project_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    repo_path = model_selectors.repo_minutes_selector_path(project_root=project_root)
    runtime_path = model_selectors.runtime_minutes_selector_path(data_root=data_root)

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.write_text('{"run_id":"repo"}\n', encoding="utf-8")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text('{"run_id":"runtime"}\n', encoding="utf-8")

    resolved = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=project_root,
    )
    assert resolved == runtime_path


def test_active_rates_selector_falls_back_to_repo(tmp_path) -> None:
    project_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    repo_path = model_selectors.repo_rates_selector_path(project_root=project_root)
    runtime_path = model_selectors.runtime_rates_selector_path(data_root=data_root)

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.write_text('{"run_id":"repo"}\n', encoding="utf-8")
    # Intentionally do not create runtime selector.
    assert not runtime_path.exists()

    resolved = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=project_root,
    )
    assert resolved == repo_path
