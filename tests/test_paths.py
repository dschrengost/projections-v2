from __future__ import annotations

from pathlib import Path

from projections import paths


def test_env_override(monkeypatch, tmp_path):
    custom_root = tmp_path / "shared-data"
    custom_root.mkdir()
    monkeypatch.setenv(paths.PROJECTIONS_DATA_ENV, str(custom_root))

    assert paths.get_data_root() == custom_root.resolve()


def test_repo_fallback(monkeypatch):
    monkeypatch.delenv(paths.PROJECTIONS_DATA_ENV, raising=False)
    expected = (paths.get_project_root() / "data").resolve()

    assert paths.get_data_root() == expected
