"""Build season-level bronze/silver datasets for Minutes V1."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths

from .constants import AvailabilityStatus
from .labels import freeze_boxscore_labels
from .schemas import (
    INJURIES_RAW_SCHEMA,
    INJURIES_SNAPSHOT_SCHEMA,
    ODDS_RAW_SCHEMA,
    ODDS_SNAPSHOT_SCHEMA,
    ROSTER_NIGHTLY_RAW_SCHEMA,
    ROSTER_NIGHTLY_SCHEMA,
    SCHEDULE_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from .snapshots import latest_pre_tip_snapshot, select_injury_snapshot
from .validation import hash_season_labels
from projections.etl.roster_nightly import SnapshotConfig as RosterSnapshotConfig, build_roster_snapshot

app = typer.Typer(help=__doc__)


def _normalize_key(value: str | None) -> str:
    if not value:
        return ""
    # Fold Unicode (e.g., Dondd -> Doncic) before stripping non-alphanumerics.
    normalized = unicodedata.normalize("NFKD", value)
    ascii_folded = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_folded.lower())


def _parse_minutes_iso(value: str | None) -> float:
    if not value:
        return 0.0
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?")
    match = pattern.fullmatch(value)
    if not match:
        return 0.0
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 60 + minutes + seconds / 60.0


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_from_raw(value: str | None) -> AvailabilityStatus:
    if not value:
        return AvailabilityStatus.UNKNOWN
    normalized = value.strip().lower()
    if normalized.startswith("out"):
        return AvailabilityStatus.OUT
    if normalized.startswith("quest"):
        return AvailabilityStatus.QUESTIONABLE
    if normalized.startswith("prob"):
        return AvailabilityStatus.PROBABLE
    if normalized.startswith("avail") or normalized.startswith("act"):
        return AvailabilityStatus.AVAILABLE
    return AvailabilityStatus.UNKNOWN


def _restriction_flag(notes: str | None) -> bool:
    if not notes:
        return False
    lowered = notes.lower()
    keywords = ("restriction", "limit", "minutes", "conditioning")
    return any(word in lowered for word in keywords)


def _ramp_flag(notes: str | None) -> bool:
    if not notes:
        return False
    lowered = notes.lower()
    return any(word in lowered for word in ("ramp", "recovery", "return to play"))


@dataclass
class TeamResolver:
    """Resolve team identifiers and schedule lookups."""

    schedule: pd.DataFrame

    def __post_init__(self) -> None:
        self.name_to_id: dict[str, int] = {}
        self.name_to_tricode: dict[str, str] = {}
        for row in self.schedule.itertuples():
            for team_id, name, city, tricode in [
                (row.home_team_id, row.home_team_name, row.home_team_city, row.home_team_tricode),
                (row.away_team_id, row.away_team_name, row.away_team_city, row.away_team_tricode),
            ]:
                city_alias = "".join(part[0] for part in re.sub(r"[.]", "", city).split() if part)
                alias_values = {
                    name,
                    f"{city} {name}",
                    tricode,
                }
                if city_alias:
                    alias_values.add(f"{city_alias} {name}")
                for key in alias_values:
                    normalized = _normalize_key(key)
                    if normalized and normalized not in self.name_to_id:
                        self.name_to_id[normalized] = team_id
                        self.name_to_tricode[normalized] = tricode
        self.game_lookup_local: dict[tuple[str, str, str], str] = {}
        self.game_lookup_tip: dict[tuple[str, str, str], str] = {}
        self.tip_lookup: dict[str, pd.Timestamp] = {}
        self.matchup_lookup: dict[tuple[str, str], list[str]] = {}
        for row in self.schedule.itertuples():
            key_local = (row.game_date.strftime("%Y-%m-%d"), row.away_team_tricode, row.home_team_tricode)
            key_tip = (row.tip_day.strftime("%Y-%m-%d"), row.away_team_tricode, row.home_team_tricode)
            self.game_lookup_local[key_local] = row.game_id
            self.game_lookup_tip[key_tip] = row.game_id
            self.tip_lookup[row.game_id] = row.tip_ts
            matchup_key = (row.away_team_tricode, row.home_team_tricode)
            self.matchup_lookup.setdefault(matchup_key, []).append(row.game_id)
        for key, game_ids in self.matchup_lookup.items():
            game_ids.sort(key=lambda gid: self.tip_lookup.get(gid, pd.Timestamp.min))

    def resolve_team_id(self, name: str | None) -> int | None:
        return self.name_to_id.get(_normalize_key(name))

    def resolve_tricode(self, name: str | None) -> str | None:
        return self.name_to_tricode.get(_normalize_key(name))

    def lookup_game_id(
        self,
        game_date: str,
        away_tricode: str,
        home_tricode: str,
        *,
        tip_ts: pd.Timestamp | None = None,
    ) -> str | None:
        if not (away_tricode and home_tricode):
            return None

        normalized_away = away_tricode.upper()
        normalized_home = home_tricode.upper()
        normalized_date = game_date
        tip_value: pd.Timestamp | None = None
        if tip_ts is not None:
            tip_value = pd.Timestamp(tip_ts).tz_convert("UTC")

        key = (normalized_date, normalized_away, normalized_home)
        candidate = self.game_lookup_local.get(key)
        if candidate and tip_value is not None:
            tip_candidate = self.tip_time(candidate)
            if tip_candidate is not None and abs(tip_candidate - tip_value) > pd.Timedelta(hours=12):
                candidate = None
        if candidate:
            return candidate

        if tip_value is not None:
            tip_key = (tip_value.tz_localize(None).strftime("%Y-%m-%d"), normalized_away, normalized_home)
            candidate = self.game_lookup_tip.get(tip_key)
            if candidate:
                return candidate

        matchup_key = (normalized_away, normalized_home)
        options = self.matchup_lookup.get(matchup_key, [])
        if not options:
            return None
        if tip_value is None:
            return options[0]

        best_id: str | None = None
        best_delta = pd.Timedelta.max
        for gid in options:
            tip_candidate = self.tip_time(gid)
            if tip_candidate is None:
                continue
            delta = abs(tip_candidate - tip_value)
            if delta < best_delta:
                best_delta = delta
                best_id = gid
        return best_id

    def tip_time(self, game_id: str) -> pd.Timestamp | None:
        return self.tip_lookup.get(game_id)


@dataclass
class PlayerResolver:
    """Resolve player ids from multiple name formats."""

    lookup: dict[str, int]

    @classmethod
    def from_labels(cls, labels_df: pd.DataFrame) -> "PlayerResolver":
        mapping: dict[str, int] = {}
        for row in labels_df.itertuples():
            if math.isnan(row.player_id):
                continue
            pid = int(row.player_id)
            name = row.player_name or ""
            tokens = name.split()
            aliases = {name}
            if len(tokens) >= 2:
                first = tokens[0]
                last = tokens[-1]
                aliases.add(f"{first} {last}")
                aliases.add(f"{last}, {first}")
                aliases.add(f"{last} {first}")
            for alias in aliases:
                norm = _normalize_key(alias)
                if norm and norm not in mapping:
                    mapping[norm] = pid
        return cls(mapping)

    def resolve(self, name: str | None) -> int | None:
        return self.lookup.get(_normalize_key(name))


@dataclass
class SeasonDatasetBuilder:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    data_dir: Path = paths.get_data_root()
    season_label: str = "2024"
    season_year: str = "2024-25"
    injuries_path: Path = paths.data_path("nba_injuries_2024_25.json")
    schedule_path: Path = paths.data_path("nba_schedule_2024_25.json")
    odds_path: Path = paths.data_path("raw", "oddstrader_season-2024-2025.json")
    boxscores_path: Path = paths.data_path("raw", "nba_boxscores_2024_25.json")

    def _month_slug(self) -> str:
        return self.start_date.strftime("%b").lower()

    def run(self) -> None:
        schedule_df = self._load_schedule()
        resolver = TeamResolver(schedule_df)

        labels_df = self._build_labels(schedule_df)
        player_resolver = PlayerResolver.from_labels(labels_df)

        injuries_raw = self._build_injuries_raw(resolver, player_resolver)
        odds_raw = self._build_odds_raw(resolver)
        roster_raw = self._build_roster_raw(labels_df, resolver)
        injuries_snapshot = self._build_injury_snapshot(injuries_raw, resolver)
        odds_snapshot = self._build_odds_snapshot(odds_raw, resolver)

        roster_snapshot = build_roster_snapshot(
            roster_raw.copy(),
            schedule_df.loc[:, ["game_id", "game_date", "tip_ts"]],
            config=RosterSnapshotConfig(start_date=self.start_date, end_date=self.end_date),
        )

        self._write_outputs(
            schedule_df,
            injuries_raw,
            odds_raw,
            roster_raw,
            roster_snapshot,
            injuries_snapshot,
            odds_snapshot,
            labels_df,
        )

    def _load_schedule(self) -> pd.DataFrame:
        with self.schedule_path.open() as handle:
            rows = json.load(handle)
        records: list[dict[str, Any]] = []
        for row in rows:
            tip_ts = pd.Timestamp(row["game_time_utc"]).tz_convert("UTC")
            tip_day = tip_ts.tz_localize(None).normalize()
            game_date = pd.Timestamp(row.get("local_game_date") or tip_day).normalize()
            if not (self.start_date <= tip_day <= self.end_date):
                continue
            arena_info = row.get("arena") or {}
            records.append(
                {
                    "game_id": row["game_id"],
                    "game_code": row.get("game_code"),
                    "season": row.get("season_year", self.season_year),
                    "game_date": game_date,
                    "tip_day": tip_day,
                    "tip_ts": tip_ts,
                    "home_team_id": row["home_team"]["team_id"],
                    "home_team_name": row["home_team"]["team_name"],
                    "home_team_city": row["home_team"]["team_city"],
                    "home_team_tricode": row["home_team"]["team_tricode"],
                    "away_team_id": row["away_team"]["team_id"],
                    "away_team_name": row["away_team"]["team_name"],
                    "away_team_city": row["away_team"]["team_city"],
                    "away_team_tricode": row["away_team"]["team_tricode"],
                    "arena_id": arena_info.get("arenaId") or row.get("arena_id"),
                    "arena_name": row.get("arena_name") or arena_info.get("arenaName"),
                    "arena_city": row.get("arena_city") or arena_info.get("arenaCity"),
                    "arena_state": row.get("arena_state") or arena_info.get("arenaState"),
                }
            )
        if not records:
            raise ValueError("No schedule rows found for requested window.")
        df = pd.DataFrame(records)
        return df.sort_values("game_date").reset_index(drop=True)

    def _load_boxscores(self) -> list[dict[str, Any]]:
        with self.boxscores_path.open() as handle:
            return json.load(handle)

    def _build_labels(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        schedule_ids = set(schedule_df["game_id"])
        boxscores = self._load_boxscores()
        records: list[dict[str, Any]] = []
        for game in boxscores:
            game_id = game["game_id"]
            if game_id not in schedule_ids:
                continue
            tip_ts = pd.Timestamp(game["game_time_utc"]).tz_convert("UTC")
            game_date = tip_ts.tz_localize(None).normalize()
            season = self.season_year
            for side in ("home", "away"):
                team = game[side]
                team_id = team["team_id"]
                for player in team.get("players", []):
                    minutes_raw = player.get("statistics", {}).get("minutes")
                    minutes = _parse_minutes_iso(minutes_raw)
                    records.append(
                        {
                            "game_id": game_id,
                            "player_id": int(player["person_id"]),
                            "player_name": player.get("name"),
                            "team_id": team_id,
                            "season": season,
                            "game_date": game_date,
                            "minutes": minutes,
                            "starter_flag": int(bool(player.get("starter"))),
                            "listed_pos": player.get("position"),
                            "source": "nba.com/boxscore",
                        }
                    )
        if not records:
            raise ValueError("Unable to build labels — no boxscore rows found for window.")
        labels = pd.DataFrame(records)
        return labels

    def _build_injuries_raw(
        self,
        resolver: TeamResolver,
        player_resolver: PlayerResolver,
    ) -> pd.DataFrame:
        with self.injuries_path.open() as handle:
            rows = json.load(handle)
        ingested_ts = pd.Timestamp.now(tz="UTC")
        data: list[dict[str, Any]] = []
        start_pad = self.start_date - pd.Timedelta(days=1)
        end_pad = self.end_date + pd.Timedelta(days=1)
        for idx, row in enumerate(rows):
            report_time = pd.Timestamp(row["report_time"]).tz_convert("UTC")
            report_day = report_time.tz_localize(None).normalize()
            report_date = report_day.date()
            if not (start_pad <= report_day <= end_pad):
                continue
            matchup = row.get("matchup", "")
            away_tri, home_tri = self._parse_matchup(matchup)
            game_date_str = row.get("game_date")
            game_id = resolver.lookup_game_id(game_date_str, away_tri, home_tri) if away_tri and home_tri else None
            player_id = player_resolver.resolve(row.get("player_name"))
            team_id = resolver.resolve_team_id(row.get("team"))
            data.append(
                {
                    "report_date": pd.Timestamp(report_date),
                    "as_of_ts": report_time,
                    "team_id": team_id,
                    "player_name": row.get("player_name"),
                    "player_id": player_id,
                    "status_raw": row.get("current_status"),
                    "notes_raw": row.get("reason"),
                    "game_id": game_id,
                    "ingested_ts": ingested_ts,
                    "source": row.get("report_url"),
                    "source_row_id": f"{int(report_time.timestamp())}_{idx}",
                }
            )
        columns = [
            "report_date",
            "as_of_ts",
            "team_id",
            "player_name",
            "player_id",
            "status_raw",
            "notes_raw",
            "game_id",
            "ingested_ts",
            "source",
            "source_row_id",
            "status",
            "restriction_flag",
            "ramp_flag",
            "games_since_return",
            "days_since_return",
        ]
        injuries = pd.DataFrame(data, columns=columns)
        if injuries.empty:
            return injuries
        injuries["status"] = injuries["status_raw"].apply(_status_from_raw).astype(str)
        injuries["restriction_flag"] = injuries["notes_raw"].apply(_restriction_flag)
        injuries["ramp_flag"] = injuries["notes_raw"].apply(_ramp_flag)
        injuries["games_since_return"] = pd.NA
        injuries["days_since_return"] = pd.NA
        return injuries

    def _build_odds_raw(self, resolver: TeamResolver) -> pd.DataFrame:
        with self.odds_path.open() as handle:
            events = json.load(handle)
        ingested_ts = pd.Timestamp.now(tz="UTC")
        records: list[dict[str, Any]] = []
        for event in events:
            scheduled = pd.Timestamp(event["scheduled"]).tz_convert("UTC")
            game_date = scheduled.tz_localize(None).strftime("%Y-%m-%d")
            game_day = pd.Timestamp(game_date)
            if not (self.start_date <= game_day <= self.end_date):
                continue
            home_tri = resolver.resolve_tricode(event.get("home_team"))
            away_tri = resolver.resolve_tricode(event.get("away_team"))
            if not (home_tri and away_tri):
                continue
            game_id = resolver.lookup_game_id(game_date, away_tri, home_tri, tip_ts=scheduled)
            if not game_id:
                continue
            spread_home = None
            total_points = None
            spread_ts = None
            total_ts = None
            markets = event.get("markets", {})
            spread = markets.get("spread", {})
            total = markets.get("total", {})
            home_spread = spread.get("home") or {}
            over_total = total.get("over") or {}
            if home_spread:
                spread_home = float(home_spread.get("point", 0.0))
                spread_ts = pd.Timestamp(home_spread.get("updated_at")).tz_convert("UTC")
            if over_total:
                total_points = float(over_total.get("point", 0.0))
                total_ts = pd.Timestamp(over_total.get("updated_at")).tz_convert("UTC")
            timestamps = [ts for ts in (spread_ts, total_ts) if ts is not None]
            if not timestamps:
                continue
            as_of_ts = min(max(timestamps), scheduled)
            records.append(
                {
                    "game_id": game_id,
                    "home_team_id": resolver.resolve_team_id(event.get("home_team")),
                    "away_team_id": resolver.resolve_team_id(event.get("away_team")),
                    "spread_home": spread_home,
                    "total": total_points,
                    "book": home_spread.get("book") if home_spread else over_total.get("book"),
                    "market": "spread_total",
                    "as_of_ts": as_of_ts,
                    "ingested_ts": ingested_ts,
                    "source": "oddstrader",
                }
            )
        return pd.DataFrame(records, columns=list(ODDS_RAW_SCHEMA.columns))

    def _build_roster_raw(self, labels_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
        schedule_long = pd.concat(
            [
                resolver.schedule[["game_id", "game_date", "tip_ts", "home_team_id"]].rename(
                    columns={"home_team_id": "team_id"}
                ),
                resolver.schedule[["game_id", "game_date", "tip_ts", "away_team_id"]].rename(
                    columns={"away_team_id": "team_id"}
                ),
            ],
            ignore_index=True,
        )
        roster = (
            labels_df.merge(schedule_long, on=["game_id", "team_id"], how="left")
            .assign(
                active_flag=1,
                ingested_ts=pd.Timestamp.now(tz="UTC"),
                source="nba.com/boxscore",
            )
        )
        roster["as_of_ts"] = roster["tip_ts"] - pd.Timedelta(hours=1)
        if "game_date_y" in roster.columns:
            roster["game_date"] = roster["game_date_y"].fillna(roster.get("game_date_x"))
        elif "game_date_x" in roster.columns:
            roster["game_date"] = roster["game_date_x"]
        else:
            roster["game_date"] = roster.get("game_date")
        roster = roster.drop(columns=[col for col in ("game_date_x", "game_date_y") if col in roster.columns])
        if "player_name" not in roster.columns:
            roster["player_name"] = pd.NA
        return roster[
            [
                "game_id",
                "team_id",
                "game_date",
                "player_id",
                "player_name",
                "active_flag",
                "starter_flag",
                "listed_pos",
                "ingested_ts",
                "source",
                "as_of_ts",
            ]
        ]

    def _build_injury_snapshot(self, injuries_raw: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
        merged = injuries_raw.dropna(subset=["game_id", "player_id"]).merge(
            resolver.schedule[["game_id", "tip_ts"]],
            on="game_id",
            how="left",
        )
        if merged.empty:
            return pd.DataFrame(columns=INJURIES_SNAPSHOT_SCHEMA.columns)
        snapshot = select_injury_snapshot(merged)
        return snapshot[
            [
                "game_id",
                "player_id",
                "as_of_ts",
                "status",
                "restriction_flag",
                "ramp_flag",
                "games_since_return",
                "days_since_return",
                "ingested_ts",
                "source",
                "selection_rule",
                "snapshot_missing",
            ]
        ]

    def _build_odds_snapshot(self, odds_raw: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
        merged = odds_raw.merge(resolver.schedule[["game_id", "tip_ts"]], on="game_id", how="left")
        if merged.empty:
            return pd.DataFrame(columns=ODDS_SNAPSHOT_SCHEMA.columns)
        merged.loc[merged["as_of_ts"] > merged["tip_ts"], "as_of_ts"] = merged["tip_ts"]
        snapshot = latest_pre_tip_snapshot(
            merged,
            group_cols=["game_id"],
            tip_ts_col="tip_ts",
            as_of_col="as_of_ts",
        )
        snapshot["book_pref"] = snapshot["book"]
        return snapshot[
            ["game_id", "as_of_ts", "spread_home", "total", "book", "book_pref", "ingested_ts", "source"]
        ]

    def _write_outputs(
        self,
        schedule_df: pd.DataFrame,
        injuries_raw: pd.DataFrame,
        odds_raw: pd.DataFrame,
        roster_raw: pd.DataFrame,
        roster_snapshot: pd.DataFrame,
        injuries_snapshot: pd.DataFrame,
        odds_snapshot: pd.DataFrame,
        labels_df: pd.DataFrame,
        ) -> None:
        month_token = f"month={self.start_date.month:02d}"
        bronze_injuries_dir = _ensure_dir(self.data_dir / "bronze" / "injuries_raw" / f"season={self.season_label}")
        bronze_odds_dir = _ensure_dir(self.data_dir / "bronze" / "odds_raw" / f"season={self.season_label}")
        silver_schedule_dir = _ensure_dir(
            self.data_dir / "silver" / "schedule" / f"season={self.season_label}" / month_token
        )
        silver_injuries_dir = _ensure_dir(
            self.data_dir / "silver" / "injuries_snapshot" / f"season={self.season_label}" / month_token
        )
        silver_odds_dir = _ensure_dir(
            self.data_dir / "silver" / "odds_snapshot" / f"season={self.season_label}" / month_token
        )
        bronze_roster_dir = _ensure_dir(
            self.data_dir / "bronze" / "roster_nightly" / f"season={self.season_label}" / month_token
        )
        silver_roster_dir = _ensure_dir(
            self.data_dir / "silver" / "roster_nightly" / f"season={self.season_label}" / month_token
        )
        labels_root = _ensure_dir(self.data_dir / "labels")

        schedule_df = enforce_schema(schedule_df, SCHEDULE_SCHEMA, allow_missing_optional=True)
        validate_with_pandera(schedule_df, SCHEDULE_SCHEMA)
        injuries_raw = enforce_schema(injuries_raw, INJURIES_RAW_SCHEMA)
        validate_with_pandera(injuries_raw, INJURIES_RAW_SCHEMA)
        odds_raw = enforce_schema(odds_raw, ODDS_RAW_SCHEMA)
        validate_with_pandera(odds_raw, ODDS_RAW_SCHEMA)
        roster_raw = enforce_schema(roster_raw, ROSTER_NIGHTLY_RAW_SCHEMA)
        validate_with_pandera(roster_raw, ROSTER_NIGHTLY_RAW_SCHEMA)
        roster_snapshot = enforce_schema(roster_snapshot, ROSTER_NIGHTLY_SCHEMA)
        validate_with_pandera(roster_snapshot, ROSTER_NIGHTLY_SCHEMA)
        injuries_snapshot = enforce_schema(injuries_snapshot, INJURIES_SNAPSHOT_SCHEMA)
        validate_with_pandera(injuries_snapshot, INJURIES_SNAPSHOT_SCHEMA)
        odds_snapshot = enforce_schema(odds_snapshot, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
        validate_with_pandera(odds_snapshot, ODDS_SNAPSHOT_SCHEMA)

        schedule_df.to_parquet(silver_schedule_dir / "schedule.parquet", index=False)
        month_slug = self._month_slug()
        injuries_raw.to_parquet(bronze_injuries_dir / f"injuries_{month_slug}.parquet", index=False)
        odds_raw.to_parquet(bronze_odds_dir / f"odds_{month_slug}.parquet", index=False)
        roster_raw.to_parquet(bronze_roster_dir / "roster_raw.parquet", index=False)
        roster_snapshot.to_parquet(silver_roster_dir / "roster.parquet", index=False)
        injuries_snapshot.to_parquet(silver_injuries_dir / "injuries_snapshot.parquet", index=False)
        odds_snapshot.to_parquet(silver_odds_dir / "odds_snapshot.parquet", index=False)

        drop_candidates = [col for col in ("listed_pos",) if col in labels_df.columns]
        labels_output = labels_df.drop(columns=drop_candidates).copy()
        labels_output["season"] = self.season_label
        season_file = labels_root / f"season={self.season_label}" / "boxscore_labels.parquet"
        if season_file.exists():
            existing = pd.read_parquet(season_file)
            labels_output = pd.concat([existing, labels_output], ignore_index=True, sort=False)
            labels_output.sort_values(["game_date", "game_id", "team_id", "player_id"], inplace=True)
            labels_output = labels_output.drop_duplicates(
                subset=["game_id", "team_id", "player_id"], keep="last"
            )
        written = freeze_boxscore_labels(labels_output, labels_root, overwrite=True)
        label_hashes = hash_season_labels(labels_root)
        for season in written:
            digest = label_hashes[season]
            season_dir = labels_root / f"season={season}"
            with (season_dir / "boxscore_labels.hash").open("w") as handle:
                handle.write(f"{season},{digest}\n")

        self._write_coverage_report(schedule_df, injuries_snapshot, odds_snapshot, roster_snapshot)

    def _write_coverage_report(
        self,
        schedule_df: pd.DataFrame,
        injuries_snapshot: pd.DataFrame,
        odds_snapshot: pd.DataFrame,
        roster: pd.DataFrame,
    ) -> None:
        games_total = schedule_df["game_id"].nunique()
        injuries_games = injuries_snapshot["game_id"].nunique()
        odds_games = odds_snapshot["game_id"].nunique()
        roster_players = roster["player_id"].nunique()
        coverage_rows = [
            {"metric": "games_total", "value": games_total},
            {"metric": "games_with_injuries_snapshot", "value": injuries_games},
            {"metric": "injuries_snapshot_pct", "value": injuries_games / games_total if games_total else 0.0},
            {"metric": "games_with_odds_snapshot", "value": odds_games},
            {"metric": "odds_snapshot_pct", "value": odds_games / games_total if games_total else 0.0},
            {"metric": "roster_unique_players", "value": roster_players},
            {"metric": "feature_rows", "value": 0},
            {"metric": "feature_coverage_pct", "value": 0.0},
        ]
        reports_dir = _ensure_dir(
            Path("reports") / "minutes_v1" / f"{self.start_date.year}-{self.start_date.month:02d}"
        )
        pd.DataFrame(coverage_rows).to_csv(reports_dir / "coverage.csv", index=False)

    @staticmethod
    def _parse_matchup(matchup: str | None) -> tuple[str | None, str | None]:
        if not matchup:
            return None, None
        if "@" in matchup:
            away, home = matchup.split("@", 1)
            return away.strip(), home.strip()
        if "vs" in matchup:
            home, away = matchup.split("vs", 1)
            return away.strip(), home.strip()
        return None, None


@app.command()
def build(
    start: datetime = typer.Option(..., help="Start date (inclusive), e.g. 2024-12-01."),
    end: datetime = typer.Option(..., help="End date (inclusive), e.g. 2024-12-31."),
) -> None:
    """Build the smoke-slice dataset."""

    builder = SeasonDatasetBuilder(
        start_date=pd.Timestamp(start).normalize(),
        end_date=pd.Timestamp(end).normalize(),
    )
    builder.run()


if __name__ == "__main__":
    app()
