from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.ops.overrides import apply_overrides_to_minutes_df


def _write_overrides(tmp_path: Path, *, game_date: date, overrides: list[dict]) -> Path:
    path = tmp_path / "artifacts" / "ops" / "overrides_v1" / f"game_date={game_date.isoformat()}" / "overrides.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "game_date": game_date.isoformat(),
        "updated_at": "2025-01-01T00:00:00Z",
        "overrides": overrides,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _make_minutes_df(*, game_date: date, game_id: str, team_id: str, rows: list[tuple[str, float]]) -> pd.DataFrame:
    recs: list[dict] = []
    for player_id, p50 in rows:
        recs.append(
            {
                "game_date": game_date.isoformat(),
                "game_id": game_id,
                "team_id": team_id,
                "player_id": player_id,
                "player_name": f"P{player_id}",
                "status": "active",
                "play_prob": 1.0,
                "minutes_p10": max(p50 - 5.0, 0.0),
                "minutes_p50": p50,
                "minutes_p90": min(p50 + 5.0, 48.0),
                "minutes_p10_cond": max(p50 - 5.0, 0.0),
                "minutes_p50_cond": p50,
                "minutes_p90_cond": min(p50 + 5.0, 48.0),
            }
        )
    return pd.DataFrame(recs)


@pytest.mark.usefixtures("monkeypatch")
def test_minutes_delta_locks_player_when_feasible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 2)
    gid = "100"
    tid = "10"
    pid_delta = "1"

    baseline = _make_minutes_df(
        game_date=slate_day,
        game_id=gid,
        team_id=tid,
        rows=[
            (pid_delta, 40.0),
            ("2", 35.0),
            ("3", 30.0),
            ("4", 30.0),
            ("5", 25.0),
            ("6", 25.0),
            ("7", 30.0),
            ("8", 25.0),
        ],
    )
    assert float(baseline["minutes_p50"].sum()) == 240.0

    _write_overrides(
        tmp_path,
        game_date=slate_day,
        overrides=[
            {
                "game_id": gid,
                "player_id": pid_delta,
                "fields": {"minutes_delta": 5.0},
                "updated_at": "2025-01-02T12:00:00Z",
                "sticky_fields": [],
            }
        ],
    )

    out = apply_overrides_to_minutes_df(
        baseline,
        game_date=slate_day,
        data_root=tmp_path,
        reconcile_team_minutes=True,
        log_diagnostics=False,
        force_reconcile=True,
    )

    # Team total is reconciled back to 240.
    assert abs(float(out["minutes_p50"].sum()) - 240.0) < 1e-6

    # Delta player stays at baseline+delta (locked), while other players absorb the residual.
    out_idx = out.set_index("player_id")
    assert abs(float(out_idx.loc[pid_delta, "minutes_p50"]) - 45.0) < 1e-6
    assert bool(out_idx.loc[pid_delta, "minutes_lock_eff"]) is True
    assert abs(float(out_idx.loc[pid_delta, "minutes_target_eff"]) - 45.0) < 1e-6
    others_sum = float(out.loc[out["player_id"] != pid_delta, "minutes_p50"].sum())
    assert abs(others_sum - (240.0 - 45.0)) < 1e-6


@pytest.mark.usefixtures("monkeypatch")
def test_minutes_target_implies_lock_and_holds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 2)
    gid = "300"
    tid = "30"
    pid_target = "1"

    baseline = _make_minutes_df(
        game_date=slate_day,
        game_id=gid,
        team_id=tid,
        rows=[
            (pid_target, 40.0),
            ("2", 35.0),
            ("3", 30.0),
            ("4", 30.0),
            ("5", 25.0),
            ("6", 25.0),
            ("7", 30.0),
            ("8", 25.0),
        ],
    )
    assert float(baseline["minutes_p50"].sum()) == 240.0

    _write_overrides(
        tmp_path,
        game_date=slate_day,
        overrides=[
            {
                "game_id": gid,
                "player_id": pid_target,
                "fields": {"minutes_target": 42.0},
                "updated_at": "2025-01-02T12:00:00Z",
                "sticky_fields": [],
            }
        ],
    )

    out = apply_overrides_to_minutes_df(
        baseline,
        game_date=slate_day,
        data_root=tmp_path,
        reconcile_team_minutes=True,
        log_diagnostics=False,
        force_reconcile=True,
    )

    assert abs(float(out["minutes_p50"].sum()) - 240.0) < 1e-6
    out_idx = out.set_index("player_id")
    assert abs(float(out_idx.loc[pid_target, "minutes_p50"]) - 42.0) < 1e-6
    assert bool(out_idx.loc[pid_target, "minutes_lock_eff"]) is True
    assert abs(float(out_idx.loc[pid_target, "minutes_target_eff"]) - 42.0) < 1e-6


@pytest.mark.usefixtures("monkeypatch")
def test_minutes_lock_without_target_locks_to_current_minutes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 2)
    gid = "400"
    tid = "40"
    pid_lock_only = "2"
    pid_delta = "1"

    baseline = _make_minutes_df(
        game_date=slate_day,
        game_id=gid,
        team_id=tid,
        rows=[
            (pid_delta, 40.0),
            (pid_lock_only, 35.0),
            ("3", 30.0),
            ("4", 30.0),
            ("5", 25.0),
            ("6", 25.0),
            ("7", 30.0),
            ("8", 25.0),
        ],
    )
    assert float(baseline["minutes_p50"].sum()) == 240.0

    _write_overrides(
        tmp_path,
        game_date=slate_day,
        overrides=[
            {
                "game_id": gid,
                "player_id": pid_lock_only,
                "fields": {"minutes_lock": True},
                "updated_at": "2025-01-02T12:00:00Z",
            },
            {
                "game_id": gid,
                "player_id": pid_delta,
                "fields": {"minutes_delta": 5.0},
                "updated_at": "2025-01-02T12:00:00Z",
            },
        ],
    )

    out = apply_overrides_to_minutes_df(
        baseline,
        game_date=slate_day,
        data_root=tmp_path,
        reconcile_team_minutes=True,
        log_diagnostics=False,
        force_reconcile=True,
    )

    assert abs(float(out["minutes_p50"].sum()) - 240.0) < 1e-6
    out_idx = out.set_index("player_id")
    assert abs(float(out_idx.loc[pid_lock_only, "minutes_p50"]) - 35.0) < 1e-6
    assert bool(out_idx.loc[pid_lock_only, "minutes_lock_eff"]) is True
    assert abs(float(out_idx.loc[pid_lock_only, "minutes_target_eff"]) - 35.0) < 1e-6


@pytest.mark.usefixtures("monkeypatch")
def test_locked_infeasible_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    slate_day = date(2025, 1, 2)
    gid = "200"
    tid = "20"

    baseline = _make_minutes_df(
        game_date=slate_day,
        game_id=gid,
        team_id=tid,
        rows=[
            ("1", 40.0),
            ("2", 40.0),
            ("3", 40.0),
            ("4", 40.0),
            ("5", 40.0),
            ("6", 40.0),
        ],
    )
    assert float(baseline["minutes_p50"].sum()) == 240.0

    # Delta pushes everyone to 48 (after clip) => locked_sum=288 > 240, so reconcile must reduce locked rows.
    overrides = []
    for pid in baseline["player_id"].astype(str).tolist():
        overrides.append(
            {"game_id": gid, "player_id": pid, "fields": {"minutes_delta": 10.0}, "updated_at": "2025-01-02T12:00:00Z"}
        )
    _write_overrides(tmp_path, game_date=slate_day, overrides=overrides)

    with pytest.raises(ValueError, match=r"infeasible locked minutes"):
        apply_overrides_to_minutes_df(
            baseline,
            game_date=slate_day,
            data_root=tmp_path,
            reconcile_team_minutes=True,
            log_diagnostics=False,
            force_reconcile=True,
        )
