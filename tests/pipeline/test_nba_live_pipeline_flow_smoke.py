from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projections import model_selectors
from projections.pipeline import control_plane


def _arg_value(args: list[str], name: str) -> str:
    idx = args.index(name)
    return args[idx + 1]


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_nba_live_pipeline_flow_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prefect_flows import live_nba_pipeline
    from projections import paths

    game_date = "2025-01-01"
    monkeypatch.setenv("PROJECTIONS_ALLOW_DIRTY", "1")

    # Route all pipeline IO into a temporary data_root.
    monkeypatch.setattr(paths, "get_data_root", lambda: tmp_path)

    def fake_build_minutes_features_task(*, game_date: str, run_id: str, run_as_of_ts: str, data_root: Path) -> Path:  # noqa: ARG001
        out = data_root / "live" / "features_minutes_v1" / game_date / f"run={run_id}" / "features.parquet"
        df = pd.DataFrame(
            {
                "game_date": [game_date] * 20,
                "game_id": [1] * 20,
                "team_id": [100] * 10 + [200] * 10,
                "player_id": list(range(1, 21)),
                "feature_as_of_ts": ["2025-01-01T00:00:00Z"] * 20,
                "as_of_ts": ["2025-01-01T00:00:00Z"] * 20,
                "odds_as_of_ts": ["2025-01-01T00:00:00Z"] * 20,
                "injuries_as_of_ts": ["2025-01-01T00:00:00Z"] * 20,
            }
        )
        _write_parquet(out, df)
        return out

    monkeypatch.setattr(live_nba_pipeline, "build_minutes_features_task", fake_build_minutes_features_task)

    def fake_run_python_module(module: str, args: list[str], *, data_root: Path, timeout_s: int) -> None:  # noqa: ARG001
        run_id = None
        if "--run-id" in args:
            run_id = _arg_value(args, "--run-id")
        if module == "scripts.dk.run_daily_salaries":
            # Minimal DK salaries layout for select_main_draft_group_task
            out = (
                data_root
                / "gold"
                / "dk_salaries"
                / "site=dk"
                / f"game_date={game_date}"
                / "draft_group_id=123"
                / "salaries.parquet"
            )
            _write_parquet(out, pd.DataFrame({"player_id": list(range(1, 26)), "salary": [5000] * 25}))
            return

        if run_id is None:
            return

        if module in {"projections.cli.score_minutes_v1", "projections.cli.score_minutes_rotation_set_v1"}:
            out = (
                data_root
                / "artifacts"
                / "minutes_v1"
                / "daily"
                / game_date
                / f"run={run_id}"
                / "minutes.parquet"
            )
            minutes = [36.0] * 5 + [12.0] * 5
            df = pd.DataFrame(
                {
                    "game_date": [game_date] * 20,
                    "game_id": [1] * 20,
                    "team_id": [100] * 10 + [200] * 10,
                    "player_id": list(range(1, 21)),
                    "minutes_p50": minutes + minutes,
                    "minutes_p50_cond": minutes + minutes,
                }
            )
            _write_parquet(out, df)
            return

        if module == "projections.cli.build_rates_features_live":
            out = data_root / "live" / "features_rates_v1" / game_date / f"run={run_id}" / "features.parquet"
            df = pd.DataFrame(
                {
                    "game_date": [game_date] * 20,
                    "game_id": [1] * 20,
                    "team_id": [100] * 10 + [200] * 10,
                    "player_id": list(range(1, 21)),
                }
            )
            _write_parquet(out, df)
            return

        if module == "projections.cli.score_rates_live":
            out = data_root / "gold" / "rates_v1_live" / game_date / f"run={run_id}" / "rates.parquet"
            df = pd.DataFrame(
                {
                    "game_date": [game_date] * 20,
                    "game_id": [1] * 20,
                    "team_id": [100] * 10 + [200] * 10,
                    "player_id": list(range(1, 21)),
                    "pred_fg2_pct": [0.5] * 20,
                }
            )
            _write_parquet(out, df)
            return

        if module == "scripts.sim_v2.run_sim_live":
            assert _arg_value(args, "--profile-name") == "sim_v3"
            assert int(_arg_value(args, "--num-worlds")) == 10
            out = (
                data_root
                / "artifacts"
                / "sim_v2"
                / "worlds_fpts_v2"
                / f"game_date={game_date}"
                / f"run={run_id}"
                / "projections.parquet"
            )
            df = pd.DataFrame({"player_id": list(range(1, 21))})
            _write_parquet(out, df)
            return

        if module in {"projections.cli.score_ownership_live", "projections.cli.score_ownership_linestar"}:
            out = data_root / "silver" / "ownership_predictions" / game_date / f"run={run_id}" / "123.parquet"
            df = pd.DataFrame({"player_id": list(range(1, 21)), "pred_own_pct": [0.05] * 20})
            _write_parquet(out, df)
            # Minimal slates metadata written into the run dir (as enforced by CLI).
            slates = {"123": {"player_count": 20, "teams": [], "first_game_time": None, "is_locked": False}}
            (out.parent / "slates.json").write_text(json.dumps(slates, indent=2), encoding="utf-8")
            return

        if module == "projections.cli.finalize_projections":
            out = data_root / "artifacts" / "projections" / game_date / f"run={run_id}" / "projections.parquet"
            df = pd.DataFrame({"player_id": list(range(1, 21))})
            _write_parquet(out, df)
            return

    monkeypatch.setattr(live_nba_pipeline, "_run_python_module", fake_run_python_module)

    result: dict[str, str] = live_nba_pipeline.nba_live_pipeline_flow(game_date=game_date, sim_worlds=10)
    run_id = result["run_id"]

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["game_date"] == game_date
    assert manifest["sim_profile"] == "sim_v3"
    assert manifest["entrypoint"] == "prefect"
    assert Path(manifest["minutes_current_run_path"]).resolve() == model_selectors.active_minutes_selector_path(
        data_root=tmp_path
    )
    assert Path(manifest["rates_current_run_path"]).resolve() == model_selectors.active_rates_selector_path(
        data_root=tmp_path
    )
    assert Path(manifest["ownership_current_run_path"]).resolve() == model_selectors.active_ownership_selector_path(
        data_root=tmp_path
    )

    # Pointers are promoted atomically, but only by the guarded Prefect run.
    dataset_dir = tmp_path / "artifacts" / "projections" / game_date
    promoted = dataset_dir / control_plane.LATEST_DIRNAME / control_plane.CURRENT_POINTER_NAME
    assert promoted.exists()
    with pytest.raises(RuntimeError):
        control_plane.promote_run_pointer(dataset_dir=dataset_dir, run_id=run_id, manifest_path=manifest_path, extra={"test": True})
