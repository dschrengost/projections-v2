"""
Scraper for NBA injury report PDFs hosted on ak-static.cms.nba.com.

The NBA publishes hourly snapshots (at :30 of every hour) during the season.
Each snapshot is a PDF with a consistent column schema but slightly different
layouts across seasons. This module downloads those PDFs, extracts the tables
with tabula-py, and normalizes the resulting rows into structured records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, List, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
try:  # pragma: no cover - import shim
    from tabula.io import read_pdf as tabula_read_pdf
except ImportError as exc:  # pragma: no cover - handled at runtime
    import tabula as _tabula

    if not hasattr(_tabula, "read_pdf"):
        raise ImportError("tabula-py is required for NBA injury scraping.") from exc
    tabula_read_pdf = _tabula.read_pdf
from PyPDF2 import PdfReader

# Expected columns inside the PDF tables.
EXPECTED_COLUMNS = [
    "Game Date",
    "Game Time",
    "Matchup",
    "Team",
    "Player Name",
    "Current Status",
    "Reason",
]

ET_TZ = ZoneInfo("America/New_York")
REPORT_BASE_URL = "https://ak-static.cms.nba.com/referee/injury"
REPORT_TEMPLATE = "Injury-Report_{date}_{time}.pdf"
REPORT_INTERVAL = timedelta(hours=1)
REPORT_SLOT_MINUTE = 30

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf",
    "Referer": "https://www.nba.com/stats/injury",
}


@dataclass(frozen=True)
class ReportLayout:
    """Tabula extraction parameters for a specific season or layout."""

    name: str
    start: datetime | None
    end: datetime | None
    head_area: Sequence[float]
    head_columns: Sequence[float]
    other_area: Sequence[float] | None = None
    other_columns: Sequence[float] | None = None

    def matches(self, timestamp: datetime) -> bool:
        after = self.start is None or timestamp >= self.start
        before = self.end is None or timestamp <= self.end
        return after and before


LAYOUTS: List[ReportLayout] = [
    ReportLayout(
        name="2022-23-first",
        start=datetime(2022, 10, 17, 0, 30, tzinfo=ET_TZ),
        end=datetime(2023, 5, 2, 17, 30, tzinfo=ET_TZ),
        head_area=[34.9956, -1.0, 566.5081, 843.1051],
        head_columns=[83.5685, 157.2435, 230.9185, 360.3760, 483.5185, 590.8735],
    ),
    ReportLayout(
        name="2022-23-rest",
        start=datetime(2023, 5, 2, 17, 30, tzinfo=ET_TZ),
        end=datetime(2023, 10, 23, 23, 30, tzinfo=ET_TZ),
        head_area=[73.1444, 1.7892, 530.9547, 841.6343],
        head_columns=[113.3475, 190.1755, 259.6363, 415.3971, 576.4200, 658.5102],
    ),
    ReportLayout(
        name="2023-24",
        start=datetime(2023, 10, 24, 17, 30, tzinfo=ET_TZ),
        end=datetime(2024, 10, 20, 23, 30, tzinfo=ET_TZ),
        head_area=[76.3017, 18.3124, 534.1120, 820.2698],
        head_columns=[108.8220, 183.5451, 255.1108, 371.9314, 543.4787, 655.0371],
    ),
    ReportLayout(
        name="2024-25",
        start=datetime(2024, 10, 21, 0, 30, tzinfo=ET_TZ),
        end=None,  # Default for 2024-25 onward (until updated).
        head_area=[76.3017, 18.3124, 534.1120, 827.6369],
        head_columns=[108.8220, 183.5451, 255.1108, 371.9314, 543.4787, 655.0371],
    ),
]


@dataclass(frozen=True)
class InjuryRecord:
    """Normalized representation of one row in an injury report."""

    report_time: datetime
    report_url: str
    game_date: date
    game_time_et: str
    matchup: str
    team: str
    player_name: str
    current_status: str
    reason: str


class PdfTableReader(Protocol):
    """Strategy for extracting pandas DataFrames from a PDF payload."""

    def extract(self, pdf_bytes: bytes, layout: ReportLayout) -> List[pd.DataFrame]:
        ...


class TabulaTableReader:
    """Default table reader built on top of tabula-py."""

    def extract(self, pdf_bytes: bytes, layout: ReportLayout) -> List[pd.DataFrame]:
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        num_pages = len(pdf_reader.pages)
        other_area = layout.other_area or layout.head_area
        other_cols = layout.other_columns or layout.head_columns

        with NamedTemporaryFile(suffix=".pdf", delete=True) as handle:
            handle.write(pdf_bytes)
            handle.flush()
            head_tables = tabula_read_pdf(
                handle.name,
                pages=1,
                stream=True,
                area=layout.head_area,
                columns=layout.head_columns,
            )
            if not head_tables:
                raise RuntimeError("tabula did not return any tables for page 1")
            other_tables: List[pd.DataFrame] = []
            if num_pages >= 2:
                other_tables = tabula_read_pdf(
                    handle.name,
                    pages=f"2-{num_pages}",
                    stream=True,
                    area=other_area,
                    columns=other_cols,
                    pandas_options={"header": None},
                )
        return head_tables + other_tables


def _normalize_timestamp(timestamp: datetime) -> datetime:
    """Round down to the nearest NBA report slot (:30 each hour, ET)."""
    ts = timestamp.astimezone(ET_TZ) if timestamp.tzinfo else timestamp.replace(tzinfo=ET_TZ)
    rounded = ts.replace(minute=REPORT_SLOT_MINUTE, second=0, microsecond=0)
    if ts.minute < REPORT_SLOT_MINUTE:
        rounded -= timedelta(hours=1)
    return rounded


def _build_filename(timestamp: datetime) -> str:
    """File names are keyed by timestamp minus 30 minutes."""
    window = timestamp - timedelta(minutes=REPORT_SLOT_MINUTE)
    date_part = window.strftime("%Y-%m-%d")
    time_part = window.strftime("%I%p")
    return REPORT_TEMPLATE.format(date=date_part, time=time_part)


def _select_layout(timestamp: datetime) -> ReportLayout:
    for layout in LAYOUTS:
        if layout.matches(timestamp):
            return layout
    return LAYOUTS[-1]


def _format_report_url(timestamp: datetime) -> str:
    return f"{REPORT_BASE_URL}/{_build_filename(timestamp)}"


def _strip(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and pd.isna(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_game_date(value: str) -> date | None:
    if not value:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _clean_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Remove duplicated headers, pad forward metadata, and merge split rows."""
    frames: List[pd.DataFrame] = []
    for table in tables:
        df = table.copy()
        df = df.iloc[:, : len(EXPECTED_COLUMNS)]
        df.columns = EXPECTED_COLUMNS[: df.shape[1]]
        if df.shape[1] < len(EXPECTED_COLUMNS):
            for col in EXPECTED_COLUMNS[df.shape[1] :]:
                df[col] = pd.NA
            df = df.reindex(columns=EXPECTED_COLUMNS)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

    # Drop rows that are repeated headers from downstream pages.
    lower_expected = [col.lower() for col in EXPECTED_COLUMNS]
    header_mask = combined.apply(
        lambda row: [str(value).strip().lower() for value in row.tolist()] == lower_expected,
        axis=1,
    )
    combined = combined.loc[~header_mask].copy()

    combined[["Game Date", "Game Time", "Matchup", "Team"]] = combined[
        ["Game Date", "Game Time", "Matchup", "Team"]
    ].ffill()

    normalized_rows: List[dict] = []
    for row in combined.to_dict("records"):
        if all((pd.isna(row[col]) for col in EXPECTED_COLUMNS)):
            continue
        cleaned_row = {col: _strip(row.get(col)) for col in EXPECTED_COLUMNS}
        is_continuation = cleaned_row["Player Name"] == "" and cleaned_row["Current Status"] == "" and cleaned_row[
            "Reason"
        ]
        if is_continuation and normalized_rows:
            prior = normalized_rows[-1]
            addition = cleaned_row.get("Reason", "")
            if addition:
                existing = prior.get("Reason", "")
                prior["Reason"] = (f"{existing} {addition}".strip() if existing else addition)
            continue
        if all(value == "" for value in cleaned_row.values()):
            continue
        normalized_rows.append(cleaned_row)

    result = pd.DataFrame(normalized_rows)
    for col in ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"]:
        if col in result:
            result[col] = result[col].apply(_strip)
    result.dropna(subset=["Player Name"], inplace=True)
    result = result[result["Player Name"] != ""]
    return result.reset_index(drop=True)


def _dataframe_to_records(df: pd.DataFrame) -> List[InjuryRecord]:
    records: List[InjuryRecord] = []
    for row in df.to_dict("records"):
        game_date = _parse_game_date(row["Game Date"])
        if not game_date:
            continue
        records.append(
            InjuryRecord(
                report_time=row["report_time"],
                report_url=row["report_url"],
                game_date=game_date,
                game_time_et=row["Game Time"],
                matchup=row["Matchup"],
                team=row["Team"],
                player_name=row["Player Name"],
                current_status=row["Current Status"],
                reason=row["Reason"],
            )
        )
    return records


class NBAInjuryScraper:
    """High-level facade for downloading and parsing NBA injury reports."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        table_reader: PdfTableReader | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._owns_client = client is None
        merged_headers = dict(DEFAULT_HEADERS)
        if headers:
            merged_headers.update(headers)
        self.client = client or httpx.Client(timeout=timeout, headers=merged_headers)
        self.table_reader = table_reader or TabulaTableReader()

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "NBAInjuryScraper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def build_report_url(self, timestamp: datetime) -> str:
        aligned = _normalize_timestamp(timestamp)
        return _format_report_url(aligned)

    def report_exists(self, timestamp: datetime) -> bool:
        url = self.build_report_url(timestamp)
        try:
            response = self.client.head(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404):
                return False
            raise
        return True

    def fetch_report(self, timestamp: datetime, *, as_dataframe: bool = False) -> List[InjuryRecord] | pd.DataFrame:
        aligned = _normalize_timestamp(timestamp)
        url = _format_report_url(aligned)
        response = self.client.get(url)
        response.raise_for_status()
        layout = _select_layout(aligned)
        tables = self.table_reader.extract(response.content, layout)
        df = _clean_tables(tables)
        df["report_time"] = aligned
        df["report_url"] = url
        if as_dataframe:
            return df
        return _dataframe_to_records(df)

    def fetch_range(
        self,
        start: datetime,
        end: datetime,
        *,
        progress: Callable[[datetime], None] | None = None,
    ) -> List[InjuryRecord]:
        if end < start:
            raise ValueError("end must be greater than or equal to start")
        records: List[InjuryRecord] = []
        ts = _normalize_timestamp(start)
        end_aligned = _normalize_timestamp(end)
        while ts <= end_aligned:
            if progress:
                progress(ts)
            try:
                result = self.fetch_report(ts)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (403, 404):
                    ts += REPORT_INTERVAL
                    continue
                raise
            if isinstance(result, pd.DataFrame):
                raise RuntimeError("Expected InjuryRecord list, received DataFrame.")
            records.extend(result)
            ts += REPORT_INTERVAL
        return records

    def fetch_daily_reports(
        self,
        target_date: date,
        *,
        progress: Callable[[datetime], None] | None = None,
    ) -> List[InjuryRecord]:
        day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=ET_TZ).replace(
            hour=0, minute=REPORT_SLOT_MINUTE
        )
        day_end = day_start + timedelta(hours=23)
        return self.fetch_range(day_start, day_end, progress=progress)

    def fetch_latest_report(
        self,
        *,
        as_of: datetime | None = None,
        hours_back: int = 72,
    ) -> List[InjuryRecord]:
        cursor = _normalize_timestamp(as_of or datetime.now(tz=ET_TZ))
        attempts = 0
        while attempts < hours_back:
            try:
                result = self.fetch_report(cursor)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (403, 404):
                    cursor -= REPORT_INTERVAL
                    attempts += 1
                    continue
                raise
            if isinstance(result, pd.DataFrame):
                raise RuntimeError("Expected InjuryRecord list, received DataFrame.")
            return result
        raise RuntimeError("Unable to find an available injury report within the requested window")
