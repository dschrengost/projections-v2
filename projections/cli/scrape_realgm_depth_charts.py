"""CLI for scraping/persisting RealGM NBA depth charts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from projections import paths
from scrapers.realgm_depth_charts import (
    realgm_dependency_report,
    save_realgm_depth_charts_bronze,
    scrape_realgm_depth_charts,
)

app = typer.Typer(help=__doc__)


@app.command()
def scrape(
    game_date: str | None = typer.Option(
        None,
        "--date",
        "-d",
        help="Partition date YYYY-MM-DD (defaults to today).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Data root (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--visible",
        help="Run browser headless.",
    ),
    timeout: float = typer.Option(
        60.0,
        "--timeout",
        help="RealGM page load timeout in seconds.",
    ),
) -> None:
    report = realgm_dependency_report()
    if not bool(report.get("available", False)):
        missing = ",".join(str(x) for x in report.get("missing", []))
        details = ",".join(str(x) for x in report.get("details", []))
        raise typer.BadParameter(
            "RealGM dependencies unavailable. "
            f"missing={missing} details={details}. "
            "Install: beautifulsoup4 lxml playwright && playwright install chromium"
        )

    target_date = date.fromisoformat(game_date) if game_date else date.today()
    df = scrape_realgm_depth_charts(headless=headless, timeout=timeout)
    if df.empty:
        typer.echo("[realgm] no depth chart rows returned")
        raise typer.Exit(code=0)

    outputs = save_realgm_depth_charts_bronze(
        df,
        game_date=target_date,
        data_root=data_root,
    )
    typer.echo(f"[realgm] rows={len(df)} history={outputs['history_path']}")


if __name__ == "__main__":
    app()
