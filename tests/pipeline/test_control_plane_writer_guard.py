import json
from pathlib import Path

import pytest

from projections.pipeline import control_plane, writer_guard


def test_writer_guard_blocks_without_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(writer_guard.ENTRYPOINT_ENV, raising=False)
    monkeypatch.delenv(writer_guard.WRITER_TOKEN_ENV, raising=False)
    monkeypatch.delenv(writer_guard.LOCK_PATH_ENV, raising=False)
    monkeypatch.delenv(writer_guard.ALLOW_UNSAFE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="Refusing to publish pointers"):
        writer_guard.assert_can_write_pointers(purpose="test")


def test_writer_guard_allows_with_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(writer_guard.ALLOW_UNSAFE_ENV, raising=False)
    with writer_guard.PipelineWriterLock(data_root=tmp_path, run_id="TEST_RUN"):
        writer_guard.assert_can_write_pointers(purpose="test")

    with pytest.raises(RuntimeError):
        writer_guard.assert_can_write_pointers(purpose="after-exit")


def test_manifest_written_and_promoted_pointers_require_guard(tmp_path: Path) -> None:
    run_id = "20250101T000000Z"
    game_date = "2025-01-01"
    manifest = control_plane.write_run_manifest_start(
        data_root=tmp_path,
        game_date=game_date,
        run_id=run_id,
        as_of_ts="2025-01-01T00:00:00Z",
        sim_profile="sim_v3",
        entrypoint="prefect",
        minutes_current_run_path=Path("config/minutes_current_run.json"),
        rates_current_run_path=Path("config/rates_current_run.json"),
        ownership_current_run_path=Path("config/ownership_current_run.json"),
        slate={},
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["game_date"] == game_date
    assert payload["sim_profile"] == "sim_v3"
    assert payload["entrypoint"] == "prefect"
    assert payload["ownership_current_run_path"] == "config/ownership_current_run.json"

    dataset_dir = tmp_path / "gold" / "projections_minutes_v1" / f"game_date={game_date}"
    with pytest.raises(RuntimeError):
        control_plane.promote_run_pointer(dataset_dir=dataset_dir, run_id=run_id, manifest_path=manifest)

    with writer_guard.PipelineWriterLock(data_root=tmp_path, run_id=run_id):
        pointer = control_plane.promote_run_pointer(dataset_dir=dataset_dir, run_id=run_id, manifest_path=manifest)
        assert pointer.exists()
        assert (dataset_dir / "latest_run.json").exists()
        assert (dataset_dir / "LATEST" / "current.json").exists()
