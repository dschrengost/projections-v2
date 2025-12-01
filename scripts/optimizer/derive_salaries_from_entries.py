"""Derive a DK-style salaries CSV from a messy DK entries export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


COLUMN_CANDIDATES: Dict[str, list[str]] = {
    "Name": ["name", "player", "player name"],
    "Position": ["roster position", "position", "pos"],
    "Salary": ["salary"],
    "TeamAbbrev": ["teamabbrev", "team"],
    "ID": ["id", "player id", "playerid"],
}


def _find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lookup:
            return lookup[key]
    return None


def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "name": _find_column(df.columns, COLUMN_CANDIDATES["Name"]),
        "position": _find_column(df.columns, COLUMN_CANDIDATES["Position"]),
        "salary": _find_column(df.columns, COLUMN_CANDIDATES["Salary"]),
        "team": _find_column(df.columns, COLUMN_CANDIDATES["TeamAbbrev"]),
        "id": _find_column(df.columns, COLUMN_CANDIDATES["ID"]),
    }


def _parse_embedded_salaries(entries_path: Path) -> pd.DataFrame:
    """Fallback for DK entries export where the salaries table sits after blank columns."""

    with entries_path.open(newline="") as f:
        reader = list(csv.reader(f))

    header_idx = None
    header: list[str] = []
    for i, row in enumerate(reader):
        tail = row[-9:]
        if len(tail) >= 5 and tail[0].strip().lower() == "position" and tail[2].strip().lower() == "name":
            header_idx = i
            header = tail
            break

    if header_idx is None:
        return pd.DataFrame()

    rows = []
    for row in reader[header_idx + 1 :]:
        tail = row[-9:]
        if len(tail) < len(header):
            continue
        if all(not cell.strip() for cell in tail):
            continue
        rows.append(tail)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header)


def derive_salaries(entries_path: Path) -> pd.DataFrame:
    df = pd.read_csv(entries_path, engine="python", on_bad_lines="skip")
    detected = _detect_columns(df)

    if not detected["name"] or not detected["position"] or not detected["salary"]:
        fallback_df = _parse_embedded_salaries(entries_path)
        if not fallback_df.empty:
            df = fallback_df
            detected = _detect_columns(df)

    name_col = detected["name"]
    pos_col = detected["position"]
    salary_col = detected["salary"]
    team_col = detected["team"]
    id_col = detected["id"]

    missing = [label for label, col in {"Name": name_col, "Position": pos_col, "Salary": salary_col}.items() if col is None]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing}; found columns={list(df.columns)}"
        )

    salaries = pd.DataFrame()
    salaries["Name"] = df[name_col].astype(str).str.strip()
    salaries["Position"] = df[pos_col].astype(str).str.strip()
    salaries["Salary"] = pd.to_numeric(df[salary_col], errors="coerce").fillna(0).astype(int)
    if team_col:
        salaries["TeamAbbrev"] = df[team_col].astype(str).str.strip()
    if id_col:
        salaries["ID"] = df[id_col]

    salaries = salaries.dropna(subset=["Name", "Position"])
    salaries = salaries[salaries["Name"] != ""]
    salaries = salaries[salaries["Position"] != ""]
    salaries = salaries.drop_duplicates()
    return salaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive DK salaries CSV from entries export.")
    parser.add_argument("--entries", required=True, type=Path, help="Path to DKEntries-*.csv file")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the cleaned salaries CSV")
    args = parser.parse_args()

    if not args.entries.exists():
        raise FileNotFoundError(f"Entries CSV not found: {args.entries}")

    salaries = derive_salaries(args.entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    salaries.to_csv(args.out, index=False)

    print(f"Wrote {len(salaries)} rows to {args.out}")
    print("Head:\n" + salaries.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
