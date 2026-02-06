from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnostics.audit_minutes_override_attribution import (
    _load_sim_minutes_mean,
    compute_attribution,
)


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


def test_compute_attribution_stage_totals_change_when_delta_breaks_240(tmp_path: Path) -> None:
    slate_day = date(2025, 1, 2)
    gid = "100"
    tid = "10"
    pid = "1"

    baseline = _make_minutes_df(
        game_date=slate_day,
        game_id=gid,
        team_id=tid,
        rows=[
            (pid, 40.0),
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

    overrides_payload = {
        "version": 1,
        "game_date": slate_day.isoformat(),
        "updated_at": "2025-01-02T12:00:00Z",
        "overrides": [
            {
                "game_id": gid,
                "player_id": pid,
                "fields": {"minutes_delta": 5.0},
                "updated_at": "2025-01-02T12:00:00Z",
                "sticky_fields": [],
            }
        ],
    }

    attrib, meta = compute_attribution(
        baseline_minutes=baseline,
        game_date=slate_day,
        data_root=tmp_path,
        overrides_payload=overrides_payload,
    )
    assert meta["minutes_col"] == "minutes_p50_cond"

    # m0 = 240 baseline, m1 = 245 pre-reconcile, m2 = 240 post-reconcile.
    assert abs(float(attrib["m0"].sum()) - 240.0) < 1e-6
    assert abs(float(attrib["m1"].sum()) - 245.0) < 1e-6
    assert abs(float(attrib["m2"].sum()) - 240.0) < 1e-6

    # Delta player stays at +5 even after reconciliation (locked).
    row = attrib.loc[attrib["player_id"] == pid].iloc[0]
    assert abs(float(row["d10"]) - 5.0) < 1e-6
    assert abs(float(row["d21"]) - 0.0) < 1e-6

    # Diagnostics were computed for the team.
    diag_df: pd.DataFrame = meta["reconcile_diagnostics"]
    assert not diag_df.empty
    assert int(diag_df.iloc[0]["n_players"]) == len(attrib)


@pytest.mark.usefixtures("monkeypatch")
def test_sim_minutes_mean_prefers_minutes_matrix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECTIONS_DATA_ROOT", str(tmp_path))

    game_date = "2025-01-02"
    run_id = "SIM_RUN"

    sim_dir = tmp_path / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}" / f"run={run_id}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir.parent / "latest_run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    # Minimal sim projections.parquet (fallback mean)
    pd.DataFrame(
        [
            {"game_id": "1", "team_id": "10", "player_id": "A", "minutes_sim_mean_uncond": 10.0, "minutes_sim_mean": 10.0},
            {"game_id": "1", "team_id": "10", "player_id": "B", "minutes_sim_mean_uncond": 20.0, "minutes_sim_mean": 20.0},
        ]
    ).to_parquet(sim_dir / "projections.parquet", index=False)

    # minutes_matrix.parquet overrides the projections mean when present.
    pd.DataFrame(
        [
            {"A": 12.0, "B": 18.0},
            {"A": 14.0, "B": 16.0},
        ]
    ).to_parquet(sim_dir / "minutes_matrix.parquet", index=False)

    df, meta = _load_sim_minutes_mean(data_root=tmp_path, game_date=game_date, sim_run_id=run_id)
    assert meta["minutes_source"] == "minutes_matrix.parquet"
    out = df.set_index("player_id")["minutes_m3"].to_dict()
    assert abs(float(out["A"]) - 13.0) < 1e-6
    assert abs(float(out["B"]) - 17.0) < 1e-6
