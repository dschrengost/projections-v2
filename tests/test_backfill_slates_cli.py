from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from scripts import backfill_slates


def _write_schedule(
    data_root: Path, *, season: int, rows: list[dict[str, object]]
) -> Path:
    path = data_root / "silver" / "schedule" / f"season={season}" / "month=12" / "schedule.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def test_backfill_slates_calls_freezer_for_games_in_range(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def fake_freeze(
        schedule,
        *,
        snapshot_type: str,
        data_root: Path,
        out_root: Path,
        force: bool,
        history_days: int | None,
        require_history: bool = True,
    ) -> tuple[Path, Path]:
        calls.append((int(schedule.game_id), snapshot_type))
        out_path = (
            out_root
            / f"season={schedule.season}"
            / f"game_date={schedule.game_date.isoformat()}"
            / f"game_id={schedule.game_id}"
        )
        return out_path / f"{snapshot_type}.parquet", out_path / f"manifest.{snapshot_type}.json"

    monkeypatch.setattr(backfill_slates.freezer, "_freeze_game_snapshot", fake_freeze)

    _write_schedule(
        tmp_path,
        season=2025,
        rows=[
            {
                "game_id": 111,
                "tip_ts": pd.Timestamp("2025-12-12T00:00:00Z"),
                "game_date": pd.Timestamp("2025-12-12"),
            },
            {
                "game_id": 222,
                "tip_ts": pd.Timestamp("2025-12-13T00:00:00Z"),
                "game_date": pd.Timestamp("2025-12-13"),
            },
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        backfill_slates.app,
        [
            "--season",
            "2025",
            "--start",
            "2025-12-12",
            "--end",
            "2025-12-12",
            "--snapshot-type",
            "lock",
            "--data-root",
            str(tmp_path),
            "--out-root",
            str(tmp_path / "gold" / "slates"),
            "--no-dry-run",
            "--status-file",
            str(tmp_path / "status.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [(111, "lock")]


def test_backfill_slates_resumes_when_outputs_exist(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def fake_freeze(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        schedule = args[0]
        calls.append((int(schedule.game_id), kwargs["snapshot_type"]))
        raise AssertionError("should not be called when outputs exist")

    monkeypatch.setattr(backfill_slates.freezer, "_freeze_game_snapshot", fake_freeze)

    _write_schedule(
        tmp_path,
        season=2025,
        rows=[
            {
                "game_id": 111,
                "tip_ts": pd.Timestamp("2025-12-12T00:00:00Z"),
                "game_date": pd.Timestamp("2025-12-12"),
            },
        ],
    )

    out_root = tmp_path / "gold" / "slates"
    out_dir = out_root / "season=2025" / "game_date=2025-12-12" / "game_id=111"
    out_dir.mkdir(parents=True, exist_ok=True)
    for snap in ("lock", "pretip"):
        (out_dir / f"{snap}.parquet").write_bytes(b"placeholder")
        (out_dir / f"manifest.{snap}.json").write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        backfill_slates.app,
        [
            "--season",
            "2025",
            "--date",
            "2025-12-12",
            "--snapshot-type",
            "both",
            "--data-root",
            str(tmp_path),
            "--out-root",
            str(out_root),
            "--no-dry-run",
            "--status-file",
            str(tmp_path / "status.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == []
