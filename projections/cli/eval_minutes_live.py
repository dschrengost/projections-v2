"""Evaluate live minutes predictions over a historical window."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import math
import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.eval_live import (
    MinutesLiveEvalDatasetBuilder,
    evaluate_minutes_live_run,
)

app = typer.Typer(help=__doc__)
DEFAULT_ARTIFACT_ROOT = Path("artifacts/minutes_eval_live")


def _normalize_date(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


def _fmt(value: float | int | None, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int,)) and decimals >= 0:
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def _overall_table(overall: dict[str, Any]) -> str:
    lines = [
        "## Overall Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "mae_minutes",
        "rmse_minutes",
        "smape_minutes",
        "coverage_p10_p90",
        "under_rate_p10",
        "over_rate_p90",
    ):
        lines.append(f"| {key} | {_fmt(overall.get(key))} |")
    lines.append("")
    return "\n".join(lines)


def _coverage_table(overall: dict[str, Any]) -> str:
    lines = [
        "### Conditional vs Unconditional Coverage",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "coverage_p10_p90",
        "under_rate_p10",
        "over_rate_p90",
        "coverage_p10_p90_cond",
        "under_rate_p10_cond",
        "over_rate_p90_cond",
    ):
        lines.append(f"| {key} | {_fmt(overall.get(key))} |")
    lines.append("")
    return "\n".join(lines)


def _slice_table(title: str, payload: list[dict[str, Any]]) -> str:
    lines = [f"### {title}", ""]
    if not payload:
        lines.append("_No rows in this slice._")
        lines.append("")
        return "\n".join(lines)
    lines.extend(
        [
            "| bucket | rows | mae | rmse | coverage | under | over |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload:
        lines.append(
            "| {bucket} | {rows} | {mae} | {rmse} | {coverage} | {under} | {over} |".format(
                bucket=row.get("bucket", ""),
                rows=row.get("rows", 0),
                mae=_fmt(row.get("mae_minutes"), 2),
                rmse=_fmt(row.get("rmse_minutes"), 2),
                coverage=_fmt(row.get("coverage_p10_p90")),
                under=_fmt(row.get("under_rate_p10")),
                over=_fmt(row.get("over_rate_p90")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _detailed_slice_table(title: str, payload: list[dict[str, Any]]) -> str:
    lines = [f"### {title}", ""]
    if not payload:
        lines.append("_No rows in this slice._")
        lines.append("")
        return "\n".join(lines)
    lines.extend(
        [
            "| bucket | rows | mae | rmse | coverage | coverage_cond | under | over | under_cond | over_cond |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload:
        lines.append(
            "| {bucket} | {rows} | {mae} | {rmse} | {coverage} | {coverage_cond} | {under} | {over} | {under_cond} | {over_cond} |".format(
                bucket=row.get("bucket", ""),
                rows=row.get("rows", 0),
                mae=_fmt(row.get("mae_minutes"), 2),
                rmse=_fmt(row.get("rmse_minutes"), 2),
                coverage=_fmt(row.get("coverage_p10_p90")),
                coverage_cond=_fmt(row.get("coverage_p10_p90_cond")),
                under=_fmt(row.get("under_rate_p10")),
                over=_fmt(row.get("over_rate_p90")),
                under_cond=_fmt(row.get("under_rate_p10_cond")),
                over_cond=_fmt(row.get("over_rate_p90_cond")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_report(
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    snapshot_mode: str,
    rows: int,
    games: int,
    metrics: dict[str, Any],
    snapshot_summary: dict[str, Any] | None,
    artifact_root: Path,
    data_root: Path,
) -> str:
    lines = [
        "# Minutes Live Evaluation",
        "",
        f"- Date window: {start_day.date().isoformat()} → {end_day.date().isoformat()}",
        f"- Snapshot mode: {snapshot_mode}",
        f"- Rows evaluated: {rows}",
        f"- Games evaluated: {games}",
        f"- Artifact root: {artifact_root}",
        f"- Data root: {data_root}",
        "",
    ]
    if snapshot_summary:
        lines.extend(
            [
                "## Snapshot Coverage",
                "",
                f"- Games in window: {snapshot_summary.get('total_games', 0)}",
                f"- Games with snapshots: {snapshot_summary.get('games_with_snapshots', 0)}",
                f"- Games skipped: {snapshot_summary.get('games_skipped', 0)}",
                "",
            ]
        )
        skipped = snapshot_summary.get("skipped_games", [])
        if skipped:
            lines.append("| game_id | game_date | reason |")
            lines.append("| ---: | --- | --- |")
            for item in skipped:
                lines.append(
                    f"| {item.get('game_id')} | {item.get('game_date')} | {item.get('reason')} |"
                )
            lines.append("")
    overall = metrics.get("overall", {})
    lines.append(_overall_table(overall))
    lines.append(_coverage_table(overall))

    slice_titles = {
        "starter_flag": "Starter vs bench",
        "spread_home": "Spread buckets",
        "minutes_p50": "Minutes projection buckets",
    }
    slices = metrics.get("slices", {})
    for key, title in slice_titles.items():
        lines.append(_slice_table(title, slices.get(key, [])))
    lines.append(_detailed_slice_table("DFS Rotation Slices", metrics.get("rotation_slices", [])))
    lines.append(_detailed_slice_table("Status Buckets (normalized)", metrics.get("status_slices", [])))
    lines.append(_detailed_slice_table("Injury Return Slices", metrics.get("injury_return_slices", [])))
    return "\n".join(lines)


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start-date", help="Inclusive start date (YYYY-MM-DD)."),
    end_date: datetime | None = typer.Option(None, "--end-date", help="Inclusive end date (YYYY-MM-DD)."),
    snapshot_mode: str = typer.Option(
        "last_before_tip",
        "--snapshot-mode",
        help="Snapshot selection mode (currently only last_before_tip).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing projections data (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    logs_root: Path | None = typer.Option(None, "--logs-root", help="Override prediction logs root."),
    labels_root: Path | None = typer.Option(None, "--labels-root", help="Override labels root."),
    schedule_root: Path | None = typer.Option(None, "--schedule-root", help="Override schedule root."),
    artifact_root: Path = typer.Option(
        DEFAULT_ARTIFACT_ROOT,
        "--artifact-root",
        help="Directory where evaluation artifacts should be written.",
    ),
) -> None:
    start_day = _normalize_date(start_date)
    end_input = end_date or start_date
    end_day = _normalize_date(end_input)
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on or after --start-date")

    data_root = data_root.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    logs_root = logs_root.expanduser().resolve() if logs_root else None
    labels_root = labels_root.expanduser().resolve() if labels_root else None
    schedule_root = schedule_root.expanduser().resolve() if schedule_root else None

    builder = MinutesLiveEvalDatasetBuilder(
        data_root=data_root,
        logs_root=logs_root,
        labels_root=labels_root,
        schedule_root=schedule_root,
        snapshot_mode=snapshot_mode,
    )
    frame = builder.build(start_day.date(), end_day.date())
    if frame.empty:
        typer.echo(
            "[minutes-eval] No evaluation rows found for the requested window.",
            err=True,
        )
        raise typer.Exit(code=1)

    metrics = evaluate_minutes_live_run(frame)
    games = int(frame["game_id"].nunique()) if "game_id" in frame else 0

    artifact_root.mkdir(parents=True, exist_ok=True)
    window_tag = f"{start_day.date().isoformat()}_{end_day.date().isoformat()}"
    base = f"{window_tag}_{builder.snapshot_mode}"
    json_path = artifact_root / f"eval_{base}.json"
    md_path = artifact_root / f"eval_{base}.md"
    snapshot_path = artifact_root / f"snapshot_{base}.parquet"

    frame.to_parquet(snapshot_path, index=False)
    payload = {
        "date_window": {"start": start_day.date().isoformat(), "end": end_day.date().isoformat()},
        "snapshot_mode": builder.snapshot_mode,
        "rows": int(len(frame)),
        "games": games,
        "data_root": str(data_root),
        "logs_root": str(builder.logs_root),
        "labels_root": str(builder.labels_root),
        "schedule_root": str(builder.schedule_root),
        "metrics": metrics,
        "snapshot_summary": builder.last_snapshot_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        _markdown_report(
            start_day=start_day,
            end_day=end_day,
            snapshot_mode=builder.snapshot_mode,
            rows=len(frame),
            games=games,
            metrics=metrics,
            snapshot_summary=builder.last_snapshot_summary,
            artifact_root=artifact_root,
            data_root=data_root,
        ),
        encoding="utf-8",
    )
    typer.echo(
        "[minutes-eval] Saved evaluation artifacts:\n"
        f"  - {json_path}\n"
        f"  - {md_path}\n"
        f"  - {snapshot_path}"
    )


if __name__ == "__main__":
    app()
