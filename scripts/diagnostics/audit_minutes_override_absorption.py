"""Audit how ops minutes overrides (minutes_delta / OUT) redistribute team minutes.

This script is intentionally lightweight and does not modify any artifacts.

It computes a three-stage decomposition on the *same* baseline `minutes.parquet`:
  1) baseline: raw scorer output
  2) pre_reconcile: apply ops overrides + minutes_delta WITHOUT team-240 reconciliation
  3) post_reconcile: apply ops overrides + minutes_delta WITH team-240 reconciliation

Then it prints, per team, who absorbed the residual minutes during reconciliation.
"""

from __future__ import annotations

import json
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.ops.overrides import apply_overrides_to_minutes_df

app = typer.Typer(add_completion=False)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_minutes_run_dir(root: Path, *, game_date: str, run_id: str | None) -> tuple[Path, str | None]:
    day_dir = root / "artifacts" / "minutes_v1" / "daily" / game_date
    if not day_dir.exists():
        raise FileNotFoundError(f"Missing minutes day dir: {day_dir}")

    if run_id:
        run_dir = day_dir / f"run={run_id}"
        if not (run_dir / "minutes.parquet").exists():
            raise FileNotFoundError(f"Missing minutes.parquet under {run_dir}")
        return run_dir, run_id

    latest = _read_json(day_dir / "latest_run.json")
    latest_id = str(latest.get("run_id")) if latest and latest.get("run_id") else None
    if latest_id and (day_dir / f"run={latest_id}" / "minutes.parquet").exists():
        return day_dir / f"run={latest_id}", latest_id

    run_dirs = sorted([p for p in day_dir.glob("run=*") if p.is_dir()], reverse=True)
    for candidate in run_dirs:
        if (candidate / "minutes.parquet").exists():
            resolved = candidate.name.split("=", 1)[1] if candidate.name.startswith("run=") else None
            return candidate, resolved

    raise FileNotFoundError(f"No minutes.parquet found under {day_dir}")


def _pick_center_col(df: pd.DataFrame) -> str:
    for c in ("minutes_p50_cond", "minutes_p50", "minutes_final"):
        if c in df.columns:
            return c
    raise ValueError("Missing minutes center column (expected minutes_p50_cond/minutes_p50/minutes_final)")


def _coerce_id_str(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype="string")
    out = series.astype("string", copy=False).fillna("")
    numeric = pd.to_numeric(series, errors="coerce")
    int_like = numeric.notna() & (numeric % 1 == 0)
    if int_like.any():
        out = out.where(~int_like, numeric.where(int_like).astype("Int64").astype("string"))
    return out.str.replace(r"\.0$", "", regex=True)


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Minutes run_id (defaults to latest)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional override for PROJECTIONS_DATA_ROOT."),
    game_id: str | None = typer.Option(None, "--game-id", help="Filter to a single game_id."),
    team_id: str | None = typer.Option(None, "--team-id", help="Filter to a single team_id."),
    top: int = typer.Option(12, "--top", help="Players to show per team (sorted by reconcile delta)."),
    show_all_teams: bool = typer.Option(False, "--all-teams", help="Print teams even if no overrides apply."),
) -> None:
    root = Path(data_root) if data_root is not None else paths.data_path()
    game_day = date_cls.fromisoformat(date)

    run_dir, resolved_run_id = _resolve_minutes_run_dir(root, game_date=date, run_id=run_id)
    minutes_path = run_dir / "minutes.parquet"

    baseline = pd.read_parquet(minutes_path)
    if baseline.empty:
        raise typer.BadParameter(f"Baseline minutes empty: {minutes_path}")
    if "game_date" in baseline.columns:
        baseline["game_date"] = pd.to_datetime(baseline["game_date"], errors="coerce").dt.date
        baseline = baseline.loc[baseline["game_date"] == game_day].copy()

    required = {"game_id", "team_id", "player_id"}
    if not required <= set(baseline.columns):
        missing = sorted(required - set(baseline.columns))
        raise typer.BadParameter(f"minutes.parquet missing required columns: {missing}")

    baseline = baseline.copy()
    baseline["game_id"] = _coerce_id_str(baseline["game_id"])
    baseline["team_id"] = _coerce_id_str(baseline["team_id"])
    baseline["player_id"] = _coerce_id_str(baseline["player_id"])
    if "player_name" not in baseline.columns:
        baseline["player_name"] = ""

    base_center = _pick_center_col(baseline)

    pre = apply_overrides_to_minutes_df(
        baseline,
        game_date=game_day,
        data_root=root,
        reconcile_team_minutes=False,
        log_diagnostics=False,
        force_reconcile=False,
    )
    post = apply_overrides_to_minutes_df(
        baseline,
        game_date=game_day,
        data_root=root,
        reconcile_team_minutes=True,
        log_diagnostics=False,
        force_reconcile=True,
    )

    center_col = base_center if base_center in pre.columns and base_center in post.columns else _pick_center_col(post)

    for df in (pre, post):
        df["game_id"] = _coerce_id_str(df["game_id"])
        df["team_id"] = _coerce_id_str(df["team_id"])
        df["player_id"] = _coerce_id_str(df["player_id"])
        if "player_name" not in df.columns:
            df["player_name"] = ""

    merged = (
        baseline[["game_id", "team_id", "player_id", "player_name", center_col]]
        .rename(columns={center_col: "minutes_baseline"})
        .merge(
            pre[
                [
                    c
                    for c in (
                        "game_id",
                        "team_id",
                        "player_id",
                        center_col,
                        "ops_override_applied",
                        "minutes_delta",
                        "minutes_delta_applied",
                        "minutes_target",
                        "minutes_lock",
                        "minutes_target_eff",
                        "minutes_lock_eff",
                    )
                    if c in pre.columns
                ]
            ].rename(columns={center_col: "minutes_pre"}),
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
        .merge(
            post[["game_id", "team_id", "player_id", center_col]].rename(columns={center_col: "minutes_post"}),
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
    )

    merged["minutes_pre"] = pd.to_numeric(merged["minutes_pre"], errors="coerce").fillna(0.0)
    merged["minutes_post"] = pd.to_numeric(merged["minutes_post"], errors="coerce").fillna(0.0)
    merged["minutes_baseline"] = pd.to_numeric(merged["minutes_baseline"], errors="coerce").fillna(0.0)
    merged["direct_delta"] = merged["minutes_pre"] - merged["minutes_baseline"]
    merged["reconcile_delta"] = merged["minutes_post"] - merged["minutes_pre"]
    merged["total_delta"] = merged["minutes_post"] - merged["minutes_baseline"]
    merged["ops_override_applied"] = merged["ops_override_applied"].fillna(False).astype(bool)
    merged["minutes_delta_applied"] = merged["minutes_delta_applied"].fillna(False).astype(bool)

    if game_id is not None:
        merged = merged.loc[merged["game_id"] == str(game_id)].copy()
    if team_id is not None:
        merged = merged.loc[merged["team_id"] == str(team_id)].copy()

    if merged.empty:
        typer.echo("[audit] No rows after filters; nothing to report.")
        raise typer.Exit(0)

    header_bits = [f"date={date}", f"minutes_run_id={resolved_run_id or 'UNKNOWN'}", f"center_col={center_col}"]
    typer.echo("[audit] " + " ".join(header_bits))
    typer.echo(f"[audit] minutes_path={minutes_path}")
    typer.echo(f"[audit] overrides_path={root / 'artifacts' / 'ops' / 'overrides_v1' / f'game_date={date}' / 'overrides.json'}")

    group_cols = ["game_id", "team_id"]
    for (gid, tid), g in merged.groupby(group_cols, sort=False):
        has_any_override = bool(g["ops_override_applied"].any() or g["minutes_delta_applied"].any())
        if (not show_all_teams) and (not has_any_override):
            continue

        sum_base = float(g["minutes_baseline"].sum())
        sum_pre = float(g["minutes_pre"].sum())
        sum_post = float(g["minutes_post"].sum())
        sum_direct = float(g["direct_delta"].sum())
        sum_rec = float(g["reconcile_delta"].sum())

        typer.echo(
            f"\n[team] game_id={gid} team_id={tid} "
            f"sum_baseline={sum_base:.1f} sum_pre={sum_pre:.1f} sum_post={sum_post:.1f} "
            f"direct_sum={sum_direct:+.1f} reconcile_sum={sum_rec:+.1f}"
        )
        if abs(sum_post - 240.0) > 1e-3:
            typer.echo(f"[team] WARNING: post sum off 240 by {(sum_post - 240.0):+.3f}", err=True)

        cols = [
            "player_id",
            "player_name",
            "minutes_baseline",
            "minutes_target",
            "minutes_lock",
            "minutes_target_eff",
            "minutes_lock_eff",
            "minutes_delta",
            "minutes_delta_applied",
            "ops_override_applied",
            "minutes_pre",
            "minutes_post",
            "direct_delta",
            "reconcile_delta",
        ]
        cols = [c for c in cols if c in g.columns]
        view = g[cols].copy()
        view["abs_reconcile"] = view["reconcile_delta"].abs()
        view = view.sort_values(["abs_reconcile", "minutes_pre"], ascending=False)

        shown = view.head(int(max(top, 1)))
        for _, row in shown.iterrows():
            name = str(row.get("player_name") or "")
            delta = row.get("minutes_delta")
            delta_str = f"{float(delta):+.1f}" if pd.notna(delta) else ""
            flags: list[str] = []
            if bool(row.get("minutes_delta_applied", False)):
                flags.append("delta")
            if bool(row.get("ops_override_applied", False)):
                flags.append("ops")
            flag_str = ",".join(flags) if flags else "-"
            typer.echo(
                f"  - {row['player_id']} {name:24.24s} "
                f"base={row['minutes_baseline']:.1f} pre={row['minutes_pre']:.1f} post={row['minutes_post']:.1f} "
                f"direct={row['direct_delta']:+.1f} rec={row['reconcile_delta']:+.1f} "
                f"minutes_delta={delta_str:>5s} flags={flag_str}"
            )


if __name__ == "__main__":
    app()
