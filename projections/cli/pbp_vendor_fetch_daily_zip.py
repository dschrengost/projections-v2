"""Download and stage daily vendor PBP ZIP for downstream ingest.

This command downloads a vendor-provided ZIP (e.g. BigDataBall daily export),
extracts CSVs, and stages them under:

  <data_root>/bronze/pbp_vendor/season_<YYYY>_<YY>/<YYYY>-<YYYY>_NBA_PbP_Logs/<YY>-<YY>-season/*.csv

The existing PBP ingest pipeline can then consume these files via glob.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import typer

from projections import paths

app = typer.Typer(help="Fetch daily vendor PBP ZIP and stage CSVs for ingest.")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_season_id() -> str:
    now = datetime.now(timezone.utc)
    start_year = int(now.year) if int(now.month) >= 8 else int(now.year) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _parse_season_id(season_id: str) -> tuple[int, int]:
    text = str(season_id).strip()
    if "-" not in text:
        raise ValueError(f"Invalid season_id={season_id!r}; expected YYYY-YY (example: 2025-26).")
    start_s, end_s = text.split("-", 1)
    start = int(start_s)
    end_two = int(end_s)
    end = 2000 + end_two if end_two < 100 else end_two
    if end < start:
        end += 100
    return start, end


def _season_dirs(data_root: Path, season_id: str) -> tuple[Path, Path, Path]:
    start, end = _parse_season_id(season_id)
    season_dir = data_root / "bronze" / "pbp_vendor" / f"season_{start}_{str(end)[-2:]}"
    logs_dir = season_dir / f"{start}-{end}_NBA_PbP_Logs"
    staged_dir = logs_dir / f"{start % 100:02d}-{end % 100:02d}-season"
    return season_dir, logs_dir, staged_dir


def _download_to_path(url: str, *, out_path: Path, timeout_seconds: int, user_agent: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout_seconds) as resp, out_path.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(resp, f)


@dataclass(frozen=True)
class FetchSummary:
    season_id: str
    source_url: str
    zip_path: str
    logs_dir: str
    staged_dir: str
    csv_found_in_zip: int
    copied_new: int
    copied_overwrite: int
    fetched_at_utc: str


@app.command("run")
def run(
    season_id: str = typer.Option(_default_season_id(), "--season-id", help="NBA season (YYYY-YY)."),
    url: str | None = typer.Option(
        None,
        "--url",
        help="Vendor daily ZIP URL. If omitted, uses env PBP_VENDOR_DAILY_URL.",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Projections data root.",
    ),
    timeout_seconds: int = typer.Option(600, "--timeout-seconds", help="HTTP timeout in seconds."),
    user_agent: str = typer.Option("projections-v2/pbp-daily-fetch", "--user-agent", help="HTTP User-Agent header."),
) -> None:
    # Prefer explicit option, then env var.
    resolved_url = str(url or "").strip()
    if not resolved_url:
        resolved_url = str(os.environ.get("PBP_VENDOR_DAILY_URL") or "").strip()
    if not resolved_url:
        raise typer.BadParameter("Missing --url and env PBP_VENDOR_DAILY_URL is not set.")

    season_dir, logs_dir, staged_dir = _season_dirs(data_root, season_id)
    logs_dir.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = season_dir / "_downloads" / f"date={datetime.now(timezone.utc).date().isoformat()}"
    zip_path = downloads_dir / f"daily_{_utc_now_compact()}.zip"

    typer.echo(f"[pbp-fetch] season_id={season_id} logs_dir={logs_dir} staged_dir={staged_dir}")
    typer.echo(f"[pbp-fetch] downloading -> {zip_path}")
    _download_to_path(
        resolved_url,
        out_path=zip_path,
        timeout_seconds=int(timeout_seconds),
        user_agent=user_agent,
    )

    copied_new = 0
    copied_overwrite = 0
    csv_paths: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix="pbp_vendor_zip_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            with ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            csv_paths = sorted(p for p in tmp_dir.rglob("*.csv") if p.is_file())
            if not csv_paths:
                raise RuntimeError(f"No CSV files found in ZIP: {zip_path}")
            for src in csv_paths:
                dst = staged_dir / src.name
                if dst.exists():
                    copied_overwrite += 1
                else:
                    copied_new += 1
                shutil.copy2(src, dst)
                flat_dst = logs_dir / src.name
                if flat_dst != dst:
                    shutil.copy2(src, flat_dst)
    except BadZipFile as exc:
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {zip_path}") from exc

    summary = FetchSummary(
        season_id=season_id,
        source_url=resolved_url,
        zip_path=str(zip_path),
        logs_dir=str(logs_dir),
        staged_dir=str(staged_dir),
        csv_found_in_zip=int(len(csv_paths)),
        copied_new=int(copied_new),
        copied_overwrite=int(copied_overwrite),
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    summary_path = downloads_dir / "fetch_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(
        "[pbp-fetch] staged "
        f"csv_found={summary.csv_found_in_zip} copied_new={summary.copied_new} "
        f"copied_overwrite={summary.copied_overwrite} summary={summary_path}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
