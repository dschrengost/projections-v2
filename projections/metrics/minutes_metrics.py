"""Emit CSV + HTML monitoring outputs for Minutes V1 predictions."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import math

import pandas as pd
import typer

from projections import paths
from projections.labels.derive_starters import derive_starter_flag_labels
from projections.minutes_v1.datasets import KEY_COLUMNS, ensure_columns
from projections.minutes_v1.monitoring import compute_monitoring_snapshot


app = typer.Typer(help=__doc__)

ALPHA_TARGET = 0.1
P90_TARGET = 1.0 - ALPHA_TARGET
MONTH_TOLERANCE = 0.04
SEGMENT_WARNING_TOLERANCE = 0.05
WILSON_Z = 1.96


@dataclass
class SegmentSpec:
    key: str
    labels: pd.Series
    title: str
    summary_filename: str
    daily_filename: str
    limit: int | None = None
    sort_by_delta: bool = False
    min_rows: int = 0
    note: str | None = None


@dataclass
class SegmentTable:
    title: str
    rows: list[dict[str, Any]]
    note: str | None = None


def _default_preds_path(data_root: Path, month_key: str) -> Path:
    return data_root / "preds" / "minutes_v1" / month_key / "minutes_pred.parquet"


def _segment_metrics(df: pd.DataFrame, *, prediction_col: str, label_col: str, segment: str, mask) -> dict[str, float] | None:
    scoped = df.loc[mask]
    if scoped.empty:
        return None
    recon_errors = scoped[prediction_col] - scoped[label_col]
    raw_errors = scoped["p50_raw"] - scoped[label_col]
    abs_recon = recon_errors.abs()
    abs_raw = raw_errors.abs()
    return {
        "segment": segment,
        "rows": float(len(scoped)),
        "mae_reconciled": float(abs_recon.mean()),
        "mae_raw": float(abs_raw.mean()),
        "p_gt_err6_reconciled": float((abs_recon > 6.0).mean()),
        "p_gt_err6_raw": float((abs_raw > 6.0).mean()),
        "p10_coverage": float((scoped[label_col] <= scoped["p10"]).mean()),
        "p90_coverage": float((scoped[label_col] <= scoped["p90"]).mean()),
        "p10_raw_coverage": float((scoped[label_col] <= scoped["p10_raw"]).mean()),
        "p90_raw_coverage": float((scoped[label_col] <= scoped["p90_raw"]).mean()),
    }


def _wilson_interval(successes: float, n: float) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    z = WILSON_Z
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) / n) + ((z**2) / (4.0 * n**2))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def _segment_specs(df: pd.DataFrame) -> list[SegmentSpec]:
    specs: list[SegmentSpec] = []

    injury_values = pd.to_numeric(df.get("injury_snapshot_missing"), errors="coerce")
    injury_labels = pd.Series(
        [f"injury_snapshot_missing={int(val)}" if not pd.isna(val) else "injury_snapshot_missing=unknown" for val in injury_values],
        index=df.index,
        name="injury_snapshot_missing",
    )
    specs.append(
        SegmentSpec(
            key="injury_snapshot_missing",
            labels=injury_labels,
            title="Coverage by injury_snapshot_missing",
            summary_filename="segment_coverage_injury_snapshot_missing.csv",
            daily_filename="segment_coverage_daily_injury_snapshot_missing.csv",
        )
    )

    abs_spread = pd.to_numeric(df.get("spread_home"), errors="coerce").abs()
    spread_labels = pd.Series(["|spread| unknown"] * len(df), index=df.index, name="spread_bucket")
    spread_labels.loc[abs_spread < 5] = "|spread| 0-4.5"
    spread_labels.loc[(abs_spread >= 5) & (abs_spread < 10)] = "|spread| 5-9.5"
    spread_labels.loc[abs_spread >= 10] = "|spread| >=10"
    specs.append(
        SegmentSpec(
            key="spread_bucket",
            labels=spread_labels,
            title="Coverage by |spread| buckets",
            summary_filename="segment_coverage_spread_bucket.csv",
            daily_filename="segment_coverage_daily_spread_bucket.csv",
        )
    )

    home_flag = df.get("home_flag")
    home_labels: list[str] = []
    for value in home_flag if home_flag is not None else []:
        if pd.isna(value):
            home_labels.append("home/away=unknown")
        elif bool(value):
            home_labels.append("home")
        else:
            home_labels.append("away")
    if home_labels:
        specs.append(
            SegmentSpec(
                key="home_flag",
                labels=pd.Series(home_labels, index=df.index, name="home_flag"),
                title="Coverage by home/away",
                summary_filename="segment_coverage_home_flag.csv",
                daily_filename="segment_coverage_daily_home_flag.csv",
            )
        )

    home_team = df.get("home_team_tricode", pd.Series(["" for _ in range(len(df))], index=df.index))
    away_team = df.get("away_team_tricode", pd.Series(["" for _ in range(len(df))], index=df.index))
    team_ids = pd.to_numeric(df.get("team_id", pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce")
    home_bool = pd.Series([bool(val) if not pd.isna(val) else False for val in home_flag], index=df.index) if home_flag is not None else pd.Series([False] * len(df), index=df.index)
    team_codes = pd.Series(
        [home if flag else away for home, away, flag in zip(home_team, away_team, home_bool)],
        index=df.index,
        name="team_code",
    )
    team_labels: list[str] = []
    for code, team_id in zip(team_codes, team_ids):
        base_str = "" if pd.isna(code) else str(code).strip()
        if not base_str or base_str.lower() == "nan":
            base_str = "team_unknown"
        if pd.isna(team_id):
            team_labels.append(base_str)
        else:
            team_labels.append(f"{base_str} (team_id={int(team_id)})")
    specs.append(
        SegmentSpec(
            key="team",
            labels=pd.Series(team_labels, index=df.index, name="team"),
            title="Coverage by team (top offenders)",
            summary_filename="segment_coverage_team.csv",
            daily_filename="segment_coverage_daily_team.csv",
            limit=8,
            sort_by_delta=True,
            min_rows=80,
            note="Top offenders by |delta| (n >= 80)",
        )
    )

    return specs


def _segment_table_for_spec(df: pd.DataFrame, spec: SegmentSpec, report_dir: Path) -> SegmentTable | None:
    labels = spec.labels.reindex(df.index)
    working = df[["game_date", "p10_hit", "p90_hit"]].copy()
    working["_segment"] = labels
    working = working.dropna(subset=["_segment"])  # Drop rows with no segment label.

    summary = (
        working.groupby("_segment", dropna=False)
        .agg(rows=("p10_hit", "size"), p10_hits=("p10_hit", "sum"), p90_hits=("p90_hit", "sum"))
        .reset_index()
        .rename(columns={"_segment": "segment"})
    )

    daily = (
        working.groupby(["game_date", "_segment"], dropna=False)
        .agg(rows=("p10_hit", "size"), p10_hits=("p10_hit", "sum"), p90_hits=("p90_hit", "sum"))
        .reset_index()
        .rename(columns={"_segment": "segment"})
    )

    if daily.empty:
        daily["p10_coverage"] = pd.Series(dtype=float)
        daily["p90_coverage"] = pd.Series(dtype=float)
    else:
        daily["p10_coverage"] = daily["p10_hits"] / daily["rows"]
        daily["p90_coverage"] = daily["p90_hits"] / daily["rows"]
    if not daily.empty:
        daily["game_date"] = pd.to_datetime(daily["game_date"]).dt.strftime("%Y-%m-%d")

    if summary.empty:
        summary_cols = [
            "segment",
            "rows",
            "p10_hits",
            "p10_coverage",
            "p10_ci_low",
            "p10_ci_high",
            "p90_hits",
            "p90_coverage",
            "p90_ci_low",
            "p90_ci_high",
            "p10_delta",
            "p90_delta",
            "warning",
        ]
        empty_summary = pd.DataFrame(columns=summary_cols)
        empty_summary.to_csv(report_dir / spec.summary_filename, index=False)
        daily.to_csv(report_dir / spec.daily_filename, index=False)
        return None

    summary["p10_coverage"] = summary["p10_hits"] / summary["rows"]
    summary["p90_coverage"] = summary["p90_hits"] / summary["rows"]
    p10_interval = [_wilson_interval(h, n) for h, n in zip(summary["p10_hits"], summary["rows"])]
    summary["p10_ci_low"] = [ci[0] for ci in p10_interval]
    summary["p10_ci_high"] = [ci[1] for ci in p10_interval]
    p90_interval = [_wilson_interval(h, n) for h, n in zip(summary["p90_hits"], summary["rows"])]
    summary["p90_ci_low"] = [ci[0] for ci in p90_interval]
    summary["p90_ci_high"] = [ci[1] for ci in p90_interval]
    summary["p10_delta"] = summary["p10_coverage"] - ALPHA_TARGET
    summary["p90_delta"] = summary["p90_coverage"] - P90_TARGET
    summary["warning"] = (
        summary["p10_delta"].abs() > SEGMENT_WARNING_TOLERANCE
    ) | (summary["p90_delta"].abs() > SEGMENT_WARNING_TOLERANCE)
    summary["max_abs_delta"] = summary[["p10_delta", "p90_delta"]].abs().max(axis=1)

    summary_out = summary.drop(columns=["max_abs_delta"])
    summary_out.to_csv(report_dir / spec.summary_filename, index=False)
    daily.to_csv(report_dir / spec.daily_filename, index=False)

    display_df = summary.copy()
    if spec.min_rows:
        display_df = display_df.loc[display_df["rows"] >= spec.min_rows]
    if display_df.empty:
        return None

    if spec.sort_by_delta:
        display_df = display_df.sort_values("max_abs_delta", ascending=False)
    else:
        display_df = display_df.sort_values("segment")
    if spec.limit:
        display_df = display_df.head(spec.limit)

    rows: list[dict[str, Any]] = []
    for _, row in display_df.iterrows():
        rows.append(
            {
                "segment": row["segment"],
                "rows": int(row["rows"]),
                "p10_coverage": float(row["p10_coverage"]),
                "p10_ci": (float(row["p10_ci_low"]), float(row["p10_ci_high"])),
                "p90_coverage": float(row["p90_coverage"]),
                "p90_ci": (float(row["p90_ci_low"]), float(row["p90_ci_high"])),
                "warning": bool(row["warning"]),
                "link": f"{spec.daily_filename}?segment={quote_plus(str(row['segment']))}",
            }
        )

    if not rows:
        return None

    return SegmentTable(title=spec.title, rows=rows, note=spec.note)


def _build_segment_tables(df: pd.DataFrame, specs: list[SegmentSpec], report_dir: Path) -> list[SegmentTable]:
    tables: list[SegmentTable] = []
    for spec in specs:
        table = _segment_table_for_spec(df, spec, report_dir)
        if table:
            tables.append(table)
    return tables


def _enforce_monthly_coverage(metrics: dict[str, float]) -> None:
    targets = {"p10_coverage": ALPHA_TARGET, "p90_coverage": 1.0 - ALPHA_TARGET}
    for bound, target in targets.items():
        value = metrics.get(bound)
        if value is None:
            continue
        delta = abs(value - target)
        if delta > MONTH_TOLERANCE:
            typer.echo(
                f"[coverage] Month {bound}={value:.3f} outside target "
                f"{target:.3f} ± {MONTH_TOLERANCE:.3f}",
                err=True,
            )
            raise typer.Exit(code=1)


def _format_ratio(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def _format_ci(ci: tuple[float, float]) -> str:
    low, high = ci
    if pd.isna(low) or pd.isna(high):
        return "—"
    return f"{float(low):.3f}-{float(high):.3f}"


def _render_segment_section(table: SegmentTable) -> str:
    header = f"<h3>{escape(table.title)}</h3>"
    note_html = f"<p class=\"segment-note\">{escape(table.note)}</p>" if table.note else ""
    rows_html = "".join(
        "<tr>"
        f"<td><a href=\"{escape(row['link'], quote=True)}\">{escape(str(row['segment']))}</a>"
        f"{'<span class=\"badge warning\">&#9888;</span>' if row['warning'] else ''}</td>"
        f"<td>{row['rows']}</td>"
        f"<td>{_format_ratio(row['p10_coverage'])}</td>"
        f"<td>{_format_ci(row['p10_ci'])}</td>"
        f"<td>{_format_ratio(row['p90_coverage'])}</td>"
        f"<td>{_format_ci(row['p90_ci'])}</td>"
        "</tr>"
        for row in table.rows
    )
    table_html = (
        "<table class=\"segment-table\">"
        "<thead><tr><th>Segment</th><th>Rows</th><th>p10 cov</th><th>p10 CI</th><th>p90 cov</th><th>p90 CI</th></tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )
    return f"<section class=\"segment-block\">{header}{note_html}{table_html}</section>"


def _render_html(overall: dict[str, float], rolling: pd.DataFrame, segment_tables: list[SegmentTable]) -> str:
    summary_items = "".join(
        f"<li><strong>{escape(str(key))}</strong>: {value:.3f}</li>" for key, value in overall.items() if isinstance(value, (int, float))
    )
    segments_block = ""
    if segment_tables:
        sections = "".join(_render_segment_section(table) for table in segment_tables)
        segments_block = (
            "<h2>Segmented Coverage</h2>"
            "<p class=\"tolerance-note\">Segments outside +/- 0.05 from target coverage are flagged.</p>"
            f"<div class=\"segments-wrapper\">{sections}</div>"
        )
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Minutes V1 Monitoring</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 0.4rem; text-align: right; }}
      th {{ background: #f0f0f0; }}
      .segment-table {{ margin-top: 0.5rem; }}
      .segment-table th:first-child, .segment-table td:first-child {{ text-align: left; }}
      .segments-wrapper {{ display: flex; flex-direction: column; gap: 1rem; }}
      .badge {{ background: #f39c12; color: #fff; border-radius: 0.25rem; padding: 0 0.35rem; margin-left: 0.35rem; font-size: 0.75rem; }}
      .segment-note {{ font-size: 0.9rem; color: #555; margin: 0.25rem 0; }}
      .tolerance-note {{ font-size: 0.9rem; color: #444; }}
      a {{ color: #0645ad; text-decoration: none; }}
    </style>
  </head>
  <body>
    <h1>Minutes V1 Monitoring Snapshot</h1>
    <ul>{summary_items}</ul>
    {segments_block}
    <h2>Rolling Daily Metrics</h2>
    {rolling.to_html(index=False)}
  </body>
</html>
"""


@app.command()
def main(
    month: str = typer.Option(..., help="Month key in YYYY-MM format."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Root directory for data outputs (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    reports_root: Path = typer.Option(Path("reports/minutes_v1"), help="Root directory for monitoring reports."),
    preds: Path | None = typer.Option(None, help="Optional explicit predictions parquet path."),
) -> None:
    """Generate CSV + HTML monitoring artifacts for a month of predictions."""

    try:
        period = pd.Period(month, freq="M")
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(f"Invalid month '{month}': {exc}")

    preds_path = preds or _default_preds_path(data_root, period.strftime("%Y-%m"))
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions parquet not found at {preds_path}")

    df = pd.read_parquet(preds_path)
    ensure_columns(
        df,
        [
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "minutes",
            "minutes_reconciled",
            "p50_raw",
            "p10_raw",
            "p90_raw",
            "p10",
            "p90",
            "tip_ts",
            "feature_as_of_ts",
            "injury_snapshot_missing",
            "spread_home",
            "home_flag",
            "home_team_tricode",
            "away_team_tricode",
        ],
    )
    unique_df = df.drop_duplicates(subset=list(KEY_COLUMNS)).copy()
    unique_df = derive_starter_flag_labels(unique_df)
    unique_df["game_date"] = pd.to_datetime(unique_df["game_date"]).dt.normalize()

    prediction_col = "minutes_reconciled"
    label_col = "minutes"

    snapshot = compute_monitoring_snapshot(
        unique_df,
        prediction_col=prediction_col,
        label_col=label_col,
        p10_col="p10",
        p90_col="p90",
    )

    segments: list[dict[str, float]] = []
    overall_metrics = _segment_metrics(
        unique_df,
        prediction_col=prediction_col,
        label_col=label_col,
        segment="overall",
        mask=slice(None),
    )
    if overall_metrics:
        segments.append(overall_metrics)
        _enforce_monthly_coverage(overall_metrics)
    starter_mask = unique_df["starter_flag_label"].astype(bool)
    starters = _segment_metrics(
        unique_df,
        prediction_col=prediction_col,
        label_col=label_col,
        segment="starters",
        mask=starter_mask,
    )
    if starters:
        segments.append(starters)
    bench = _segment_metrics(
        unique_df,
        prediction_col=prediction_col,
        label_col=label_col,
        segment="bench",
        mask=~starter_mask,
    )
    if bench:
        segments.append(bench)

    unique_df["p10_hit"] = (unique_df[label_col] <= unique_df["p10"]).astype(int)
    unique_df["p90_hit"] = (unique_df[label_col] <= unique_df["p90"]).astype(int)
    segment_specs = _segment_specs(unique_df)

    starter_counts = (
        unique_df.drop_duplicates(subset=["game_id", "team_id", "player_id"])
        .groupby(["game_id", "team_id"])["starter_flag_label"]
        .sum()
    )
    snapshot.overall["pct_games_with_exactly5_starters"] = float((starter_counts == 5).mean())

    report_dir = reports_root / period.strftime("%Y-%m")
    report_dir.mkdir(parents=True, exist_ok=True)
    segment_tables = _build_segment_tables(unique_df, segment_specs, report_dir)
    metrics_path = report_dir / "metrics.csv"
    pd.DataFrame(segments).to_csv(metrics_path, index=False)

    html_path = report_dir / "summary.html"
    html_path.write_text(_render_html(snapshot.overall, snapshot.rolling, segment_tables), encoding="utf-8")
    typer.echo(f"Wrote monitoring artifacts to {report_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
