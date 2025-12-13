"""Service layer for contest data parsing and analysis."""

import os
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel


class OwnershipMetrics(BaseModel):
    total_own: float
    avg_own: float
    min_own: float
    max_own: float
    num_under_10: int
    num_under_5: int
    num_over_50: int


class PlayerOwnership(BaseModel):
    player: str
    ownership_pct: float
    fpts: float | None
    win_pct: float | None  # % of winning lineups with this player
    top10_pct: float | None


class LeveragePlay(BaseModel):
    player: str
    ownership_pct: float
    win_rate: float
    leverage_score: float


class PlayerPair(BaseModel):
    player1: str
    player2: str
    count: int
    pct: float


class StackInfo(BaseModel):
    team: str
    stack_size: int
    count: int
    pct_of_top: float


class GameCorrelation(BaseModel):
    matchup: str
    players_in_winners: int
    total_fpts: float
    avg_fpts: float


class ContestSummary(BaseModel):
    contest_id: str
    contest_name: str
    entry_fee: float | None
    total_entries: int
    contest_type: str


class LineupOwnershipEntry(BaseModel):
    """Individual lineup with its ownership metrics."""
    rank: int
    entry_name: str
    points: float
    total_own: float  # Sum of all 8 players' ownership
    avg_own: float
    num_under_10: int
    num_under_5: int
    lineup_players: list[str]


class OwnershipDistribution(BaseModel):
    """Distribution bins for total ownership."""
    bin_label: str  # e.g., "100-110%", "110-120%"
    count: int
    pct_of_total: float


class DuplicateLineup(BaseModel):
    """A lineup that was entered multiple times."""
    lineup_key: str  # Sorted player names joined
    lineup_players: list[str]
    entry_count: int  # How many times this exact lineup was entered
    entry_names: list[str]  # All entry names that used this lineup
    ranks: list[int]  # Ranks of all entries with this lineup
    best_rank: int
    total_own: float
    avg_own: float


class DupesByOwnershipBand(BaseModel):
    """Dupe stats by ownership bucket."""
    ownership_band: str  # e.g., "100-110%"
    total_lineups: int
    unique_lineups: int
    duped_lineups: int
    dupe_rate: float  # % of lineups that were dupes


class DupeAnalysis(BaseModel):
    """Overall duplicate lineup analysis."""
    total_entries: int
    unique_lineups: int
    duplicate_entries: int  # Number of entries that were duplicates
    duplicate_rate: float  # % of entries that were dupes
    most_duped_lineup: DuplicateLineup | None
    top_duped_lineups: list[DuplicateLineup]  # Top 20 by entry count
    dupes_by_ownership: list[DupesByOwnershipBand]


class ContestDetail(BaseModel):
    contest_id: str
    contest_name: str
    total_entries: int
    winner: OwnershipMetrics | None
    top10: OwnershipMetrics | None
    top1pct: OwnershipMetrics | None
    cash_line: OwnershipMetrics | None
    field_avg: OwnershipMetrics | None
    winner_score: float | None
    top1pct_score: float | None
    cash_line_score: float | None
    leverage_plays: list[LeveragePlay]
    stacks: list[StackInfo]
    game_correlations: list[GameCorrelation]
    player_ownership: list[PlayerOwnership]
    # New lineup-level data
    top_lineups: list[LineupOwnershipEntry]  # Top 100 lineups with ownership
    ownership_distribution: list[OwnershipDistribution]  # Histogram of total_own
    dupe_analysis: DupeAnalysis  # Duplicate lineup analysis


class UserLineupEntry(BaseModel):
    contest_id: str
    contest_name: str
    entry_name: str
    rank: int
    points: float
    total_entries: int
    pct_finish: float
    lineup_players: list[str]
    ownership_metrics: OwnershipMetrics


# Cache for parsed contest data
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 300  # 5 minutes


def get_data_root() -> Path:
    """Get the data root directory from environment or default."""
    root = os.environ.get("PROJECTIONS_DATA_ROOT", str(Path.home() / "projections-data"))
    return Path(root)


def get_contest_data_dir() -> Path:
    """Get the contest data directory."""
    return get_data_root() / "bronze" / "dk_contests" / "nba_gpp_data"


def parse_lineup(lineup_str: str) -> list[str]:
    """Parse a lineup string into player names.
    
    Format: "POS1 Player1 POS2 Player2 ..." where positions are 
    C, F, G, PF, PG, SF, SG, UTIL followed by a space.
    
    Position markers must be at the start of string or after a space,
    not inside player names (e.g., "OG Anunoby" should not match "G ").
    """
    if pd.isna(lineup_str):
        return []
    
    # Pattern matches positions that are either at start of string or after whitespace
    # Uses alternation to match longest positions first (PG before G, SF before F, etc.)
    # This prevents "SG" from matching as "S" + "G"
    pattern = r'(?:^|\s)(PG|SG|SF|PF|UTIL|C|F|G)\s+([^A-Z].*?)(?=(?:\s(?:PG|SG|SF|PF|UTIL|C|F|G)\s)|$)'
    
    matches = re.findall(pattern, lineup_str, re.IGNORECASE)
    
    if matches:
        # Re.findall returns list of tuples: (position, player_name)
        players = [match[1].strip() for match in matches if match[1].strip()]
        if len(players) >= 7:  # Valid NBA DK lineup has 8 players
            return players
    
    # Fallback: split by known position markers more carefully
    # Order matters: check longer positions first to avoid partial matches
    positions_ordered = ["UTIL ", "PG ", "SG ", "SF ", "PF ", "G ", "F ", "C "]
    
    # Find all position marker locations (only when preceded by start or space)
    pos_locations = []
    for pos in positions_ordered:
        # Look for the position preceded by start of string or space
        idx = 0
        while True:
            # Find next occurrence of this position
            found_idx = lineup_str.find(pos, idx)
            if found_idx == -1:
                break
            # Check if it's at start or preceded by space
            if found_idx == 0 or (found_idx > 0 and lineup_str[found_idx - 1] == ' '):
                # Also make sure we haven't matched a longer position already
                # e.g., don't match "G " if we already matched "PG " at same location
                is_suffix_of_longer = False
                for longer_pos in ["PG ", "SG ", "SF ", "PF "]:
                    if pos in ["G ", "F "]:
                        longer_start = found_idx - 1
                        if longer_start >= 0 and lineup_str[longer_start:found_idx + len(pos)] == longer_pos:
                            is_suffix_of_longer = True
                            break
                if not is_suffix_of_longer:
                    pos_locations.append((found_idx, pos))
            idx = found_idx + 1
    
    # Sort by position in string
    pos_locations.sort(key=lambda x: x[0])
    
    players = []
    for i, (start_idx, pos) in enumerate(pos_locations):
        player_start = start_idx + len(pos)
        if i + 1 < len(pos_locations):
            player_end = pos_locations[i + 1][0]
        else:
            player_end = len(lineup_str)
        player = lineup_str[player_start:player_end].strip()
        if player:
            players.append(player)
    
    return players


def parse_contest_csv(path: Path) -> pd.DataFrame:
    """Parse a contest results CSV with caching."""
    cache_key = str(path)
    now = time.time()

    if cache_key in _cache:
        cached_time, cached_df = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_df.copy()

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "").str.replace("ï»¿", "")

    # Normalize ownership column
    own_col = None
    for col in ["%Drafted", "X.Drafted"]:
        if col in df.columns:
            own_col = col
            break

    if own_col:
        df["ownership_pct"] = df[own_col].apply(
            lambda x: float(str(x).replace("%", "")) if pd.notna(x) else 0.0
        )
    else:
        df["ownership_pct"] = 0.0

    _cache[cache_key] = (now, df)
    return df.copy()


def get_player_ownership_lookup(df: pd.DataFrame) -> dict[str, float]:
    """Build player -> ownership % lookup from contest data."""
    player_own = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("Player")) and row.get("ownership_pct", 0) > 0:
            player_own[row["Player"]] = row["ownership_pct"]
    return player_own


def get_player_fpts_lookup(df: pd.DataFrame) -> dict[str, float]:
    """Build player -> FPTS lookup from contest data."""
    player_fpts = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("Player")) and pd.notna(row.get("FPTS")):
            player_fpts[row["Player"]] = float(row["FPTS"])
    return player_fpts


def get_unique_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Get unique entries (first occurrence of each EntryId)."""
    return df[df["EntryId"].notna()].drop_duplicates(subset=["EntryId"])


def compute_lineup_ownership(
    lineup_players: list[str], player_own: dict[str, float]
) -> OwnershipMetrics | None:
    """Compute ownership metrics for a lineup."""
    ownerships = [player_own.get(p, 0) for p in lineup_players]
    valid_owns = [o for o in ownerships if o > 0]

    if not valid_owns:
        return None

    return OwnershipMetrics(
        total_own=sum(valid_owns),
        avg_own=float(np.mean(valid_owns)),
        min_own=min(valid_owns),
        max_own=max(valid_owns),
        num_under_10=sum(1 for o in valid_owns if o < 10),
        num_under_5=sum(1 for o in valid_owns if o < 5),
        num_over_50=sum(1 for o in valid_owns if o > 50),
    )


def compute_ownership_metrics_for_entries(
    entries_df: pd.DataFrame, player_own: dict[str, float]
) -> OwnershipMetrics | None:
    """Compute aggregate ownership metrics for a set of entries."""
    all_metrics = []

    for _, entry in entries_df.iterrows():
        lineup_players = parse_lineup(entry.get("Lineup", ""))
        if not lineup_players:
            continue
        metrics = compute_lineup_ownership(lineup_players, player_own)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        return None

    return OwnershipMetrics(
        total_own=float(np.mean([m.total_own for m in all_metrics])),
        avg_own=float(np.mean([m.avg_own for m in all_metrics])),
        min_own=float(np.mean([m.min_own for m in all_metrics])),
        max_own=float(np.mean([m.max_own for m in all_metrics])),
        num_under_10=int(np.mean([m.num_under_10 for m in all_metrics])),
        num_under_5=int(np.mean([m.num_under_5 for m in all_metrics])),
        num_over_50=int(np.mean([m.num_over_50 for m in all_metrics])),
    )


def compute_pairs(
    entries_df: pd.DataFrame, top_n: int = 10
) -> list[PlayerPair]:
    """Find most common player pairs in top N lineups."""
    top_entries = entries_df[entries_df["Rank"] <= top_n]
    pair_counts: Counter[tuple[str, str]] = Counter()

    for _, entry in top_entries.iterrows():
        lineup_players = parse_lineup(entry.get("Lineup", ""))
        if len(lineup_players) < 2:
            continue
        for p1, p2 in combinations(sorted(lineup_players), 2):
            pair_counts[(p1, p2)] += 1

    total_lineups = len(top_entries)
    pairs = []
    for (p1, p2), count in pair_counts.most_common(50):
        pairs.append(
            PlayerPair(
                player1=p1,
                player2=p2,
                count=count,
                pct=round(100 * count / total_lineups, 1) if total_lineups > 0 else 0,
            )
        )

    return pairs


def compute_leverage_scores(
    entries_df: pd.DataFrame,
    player_own: dict[str, float],
    top_n: int = 10,
) -> list[LeveragePlay]:
    """Compute leverage scores for players in top lineups."""
    top_entries = entries_df[entries_df["Rank"] <= top_n]
    total_top = len(top_entries)

    if total_top == 0:
        return []

    # Count player appearances in top lineups
    player_appearances: Counter[str] = Counter()
    for _, entry in top_entries.iterrows():
        lineup_players = parse_lineup(entry.get("Lineup", ""))
        for player in lineup_players:
            player_appearances[player] += 1

    leverage_plays = []
    for player, appearances in player_appearances.items():
        own_pct = player_own.get(player, 0)
        if own_pct <= 0:
            continue

        win_rate = 100 * appearances / total_top
        leverage = (win_rate - own_pct) / own_pct if own_pct > 0 else 0

        leverage_plays.append(
            LeveragePlay(
                player=player,
                ownership_pct=own_pct,
                win_rate=round(win_rate, 1),
                leverage_score=round(leverage, 2),
            )
        )

    # Sort by leverage score descending
    leverage_plays.sort(key=lambda x: x.leverage_score, reverse=True)
    return leverage_plays[:20]


def compute_stack_analysis(
    entries_df: pd.DataFrame,
    player_teams: dict[str, str],
    top_n: int = 10,
) -> list[StackInfo]:
    """Analyze team stacking patterns in top lineups."""
    top_entries = entries_df[entries_df["Rank"] <= top_n]
    total_top = len(top_entries)

    if total_top == 0:
        return []

    # Count stacks by team and size
    stack_counts: dict[tuple[str, int], int] = defaultdict(int)

    for _, entry in top_entries.iterrows():
        lineup_players = parse_lineup(entry.get("Lineup", ""))
        team_counts: Counter[str] = Counter()

        for player in lineup_players:
            team = player_teams.get(player)
            if team:
                team_counts[team] += 1

        for team, count in team_counts.items():
            if count >= 2:
                stack_counts[(team, count)] += 1

    stacks = []
    for (team, stack_size), count in sorted(
        stack_counts.items(), key=lambda x: (-x[1], -x[0][1])
    ):
        stacks.append(
            StackInfo(
                team=team,
                stack_size=stack_size,
                count=count,
                pct_of_top=round(100 * count / total_top, 1),
            )
        )

    return stacks[:20]


def compute_game_correlations(
    entries_df: pd.DataFrame,
    player_teams: dict[str, str],
    player_fpts: dict[str, float],
    top_n: int = 10,
) -> list[GameCorrelation]:
    """Compute game correlations for top lineups."""
    top_entries = entries_df[entries_df["Rank"] <= top_n]

    if len(top_entries) == 0:
        return []

    # Group teams into matchups (we'll use team as proxy for now)
    team_stats: dict[str, dict] = defaultdict(lambda: {"players": 0, "fpts": 0.0})

    for _, entry in top_entries.iterrows():
        lineup_players = parse_lineup(entry.get("Lineup", ""))
        for player in lineup_players:
            team = player_teams.get(player, "UNK")
            team_stats[team]["players"] += 1
            team_stats[team]["fpts"] += player_fpts.get(player, 0)

    correlations = []
    for team, stats in sorted(team_stats.items(), key=lambda x: -x[1]["players"]):
        if stats["players"] > 0:
            correlations.append(
                GameCorrelation(
                    matchup=team,  # Using team as matchup proxy
                    players_in_winners=stats["players"],
                    total_fpts=round(stats["fpts"], 1),
                    avg_fpts=round(stats["fpts"] / stats["players"], 1)
                    if stats["players"] > 0
                    else 0,
                )
            )

    return correlations[:15]


def categorize_contest(name: str) -> str:
    """Categorize contest by type based on name."""
    name_lower = name.lower()
    if "high five" in name_lower:
        return "High Five"
    elif "hot shot" in name_lower:
        return "Hot Shot"
    elif "pick and roll" in name_lower:
        return "Pick and Roll"
    elif "elbow shot" in name_lower or "elbow" in name_lower:
        return "Elbow Shot"
    elif "milly" in name_lower or "million" in name_lower:
        return "Millionaire"
    elif "single entry" in name_lower:
        return "Single Entry"
    elif "max" in name_lower:
        return "Multi-Entry"
    else:
        return "Other"


def list_available_dates(limit: int = 30) -> list[str]:
    """List available contest dates (most recent first)."""
    data_dir = get_contest_data_dir()
    if not data_dir.exists():
        return []

    dates = []
    for d in data_dir.iterdir():
        if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name):
            results_dir = d / "results"
            if results_dir.exists() and any(results_dir.glob("contest_*_results.csv")):
                dates.append(d.name)

    dates.sort(reverse=True)
    return dates[:limit]


def list_contests_for_date(date: str) -> list[ContestSummary]:
    """List all contests for a given date."""
    data_dir = get_contest_data_dir() / date
    if not data_dir.exists():
        return []

    results_dir = data_dir / "results"
    if not results_dir.exists():
        return []

    # Try to load contest metadata from daily CSV
    contest_meta: dict[str, dict] = {}
    for csv_file in data_dir.glob("*.csv"):
        if "results" not in csv_file.name:
            try:
                meta_df = pd.read_csv(csv_file, encoding="utf-8-sig")
                meta_df.columns = meta_df.columns.str.strip()
                for _, row in meta_df.iterrows():
                    cid = str(row.get("ContestId", row.get("contest_id", "")))
                    contest_meta[cid] = {
                        "contest_name": row.get(
                            "ContestName", row.get("contest_name", "")
                        ),
                        "entry_fee": row.get("EntryFee", row.get("entry_fee")),
                    }
            except Exception:
                pass

    contests = []
    for results_file in results_dir.glob("contest_*_results.csv"):
        contest_id = results_file.stem.replace("contest_", "").replace("_results", "")

        try:
            df = parse_contest_csv(results_file)
            entries = get_unique_entries(df)
            total_entries = len(entries)

            if total_entries < 10:
                continue

            meta = contest_meta.get(contest_id, {})
            contest_name = meta.get("contest_name", f"Contest {contest_id}")
            entry_fee = meta.get("entry_fee")

            contests.append(
                ContestSummary(
                    contest_id=contest_id,
                    contest_name=contest_name,
                    entry_fee=float(entry_fee) if entry_fee else None,
                    total_entries=total_entries,
                    contest_type=categorize_contest(contest_name),
                )
            )
        except Exception:
            continue

    # Sort by total entries descending
    contests.sort(key=lambda x: x.total_entries, reverse=True)
    return contests


def compute_lineup_ownership_entries(
    entries_df: pd.DataFrame,
    player_own: dict[str, float],
    limit: int = 100,
) -> list[LineupOwnershipEntry]:
    """Extract lineup-level ownership data for top finishers."""
    lineup_entries = []

    for _, entry in entries_df.head(limit).iterrows():
        rank = int(entry["Rank"]) if pd.notna(entry.get("Rank")) else 0
        entry_name = str(entry.get("EntryName", ""))
        points = float(entry["Points"]) if pd.notna(entry.get("Points")) else 0
        lineup_players = parse_lineup(entry.get("Lineup", ""))

        if not lineup_players:
            continue

        ownership_metrics = compute_lineup_ownership(lineup_players, player_own)
        if not ownership_metrics:
            continue

        lineup_entries.append(
            LineupOwnershipEntry(
                rank=rank,
                entry_name=entry_name,
                points=points,
                total_own=ownership_metrics.total_own,
                avg_own=ownership_metrics.avg_own,
                num_under_10=ownership_metrics.num_under_10,
                num_under_5=ownership_metrics.num_under_5,
                lineup_players=lineup_players,
            )
        )

    return lineup_entries


def compute_dupe_analysis(
    entries_df: pd.DataFrame, player_own: dict[str, float], bin_size: int = 10
) -> DupeAnalysis:
    """Analyze duplicate lineups in the contest."""
    # Build a map of lineup_key -> list of entries
    lineup_map: dict[str, list[dict]] = defaultdict(list)

    for _, entry in entries_df.iterrows():
        rank = int(entry["Rank"]) if pd.notna(entry.get("Rank")) else 0
        entry_name = str(entry.get("EntryName", ""))
        points = float(entry["Points"]) if pd.notna(entry.get("Points")) else 0
        lineup_players = parse_lineup(entry.get("Lineup", ""))

        if not lineup_players:
            continue

        # Create a unique key for this lineup (sorted player names)
        lineup_key = "|".join(sorted(lineup_players))

        ownership_metrics = compute_lineup_ownership(lineup_players, player_own)

        lineup_map[lineup_key].append({
            "rank": rank,
            "entry_name": entry_name,
            "points": points,
            "lineup_players": lineup_players,
            "total_own": ownership_metrics.total_own if ownership_metrics else 0,
            "avg_own": ownership_metrics.avg_own if ownership_metrics else 0,
        })

    total_entries = len(entries_df)
    unique_lineups = len(lineup_map)
    duplicate_entries = total_entries - unique_lineups

    # Find duplicated lineups (entry_count > 1)
    duplicated_lineups = []
    for lineup_key, entries in lineup_map.items():
        if len(entries) > 1:
            first_entry = entries[0]
            duplicated_lineups.append(
                DuplicateLineup(
                    lineup_key=lineup_key,
                    lineup_players=first_entry["lineup_players"],
                    entry_count=len(entries),
                    entry_names=[e["entry_name"] for e in entries],
                    ranks=[e["rank"] for e in entries],
                    best_rank=min(e["rank"] for e in entries),
                    total_own=first_entry["total_own"],
                    avg_own=first_entry["avg_own"],
                )
            )

    # Sort by entry count descending
    duplicated_lineups.sort(key=lambda x: x.entry_count, reverse=True)
    most_duped = duplicated_lineups[0] if duplicated_lineups else None
    top_duped = duplicated_lineups[:20]

    # Dupes by ownership band
    ownership_bins: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "unique": set(), "duped": 0}
    )

    for lineup_key, entries in lineup_map.items():
        first_entry = entries[0]
        total_own = first_entry["total_own"]

        # Determine bin
        bin_start = int(total_own / bin_size) * bin_size
        bin_end = bin_start + bin_size
        bin_label = f"{bin_start}-{bin_end}%"

        ownership_bins[bin_label]["total"] += len(entries)
        ownership_bins[bin_label]["unique"].add(lineup_key)
        if len(entries) > 1:
            ownership_bins[bin_label]["duped"] += len(entries)

    dupes_by_ownership = []
    for bin_label in sorted(ownership_bins.keys()):
        stats = ownership_bins[bin_label]
        bin_total_lineups = stats["total"]
        bin_unique_lineups = len(stats["unique"])
        bin_duped_lineups = stats["duped"]
        dupe_rate = (bin_duped_lineups / bin_total_lineups * 100) if bin_total_lineups > 0 else 0

        dupes_by_ownership.append(
            DupesByOwnershipBand(
                ownership_band=bin_label,
                total_lineups=bin_total_lineups,
                unique_lineups=bin_unique_lineups,
                duped_lineups=bin_duped_lineups,
                dupe_rate=round(dupe_rate, 1),
            )
        )

    return DupeAnalysis(
        total_entries=total_entries,
        unique_lineups=unique_lineups,
        duplicate_entries=duplicate_entries,
        duplicate_rate=round(100 * duplicate_entries / total_entries, 1) if total_entries > 0 else 0,
        most_duped_lineup=most_duped,
        top_duped_lineups=top_duped,
        dupes_by_ownership=dupes_by_ownership,
    )


def compute_ownership_distribution(
    entries_df: pd.DataFrame, player_own: dict[str, float], bin_size: int = 10
) -> list[OwnershipDistribution]:
    """Compute histogram of total lineup ownership."""
    total_owns = []

    for _, entry in entries_df.iterrows():
        lineup_players = parse_lineup(entry.get("Lineup", ""))
        if not lineup_players:
            continue

        ownership_metrics = compute_lineup_ownership(lineup_players, player_own)
        if ownership_metrics:
            total_owns.append(ownership_metrics.total_own)

    if not total_owns:
        return []

    # Create bins
    min_own = int(min(total_owns) / bin_size) * bin_size
    max_own = int(max(total_owns) / bin_size) * bin_size + bin_size
    bins = list(range(min_own, max_own + bin_size, bin_size))

    # Count lineups in each bin
    hist, _ = np.histogram(total_owns, bins=bins)
    total_count = len(total_owns)

    distribution = []
    for i in range(len(hist)):
        if hist[i] > 0:
            bin_start = bins[i]
            bin_end = bins[i + 1]
            distribution.append(
                OwnershipDistribution(
                    bin_label=f"{bin_start}-{bin_end}%",
                    count=int(hist[i]),
                    pct_of_total=round(100 * hist[i] / total_count, 1),
                )
            )

    return distribution


def get_contest_detail(date: str, contest_id: str) -> ContestDetail | None:
    """Get detailed metrics for a specific contest."""
    results_path = (
        get_contest_data_dir() / date / "results" / f"contest_{contest_id}_results.csv"
    )
    if not results_path.exists():
        return None

    df = parse_contest_csv(results_path)
    entries = get_unique_entries(df)
    total_entries = len(entries)

    if total_entries < 10:
        return None

    player_own = get_player_ownership_lookup(df)
    player_fpts = get_player_fpts_lookup(df)

    # Build player -> team mapping from contest data
    # (We don't have team info in the CSV, so we'll use empty for now)
    # In a real implementation, you'd join with player roster data
    player_teams: dict[str, str] = {}

    # Get contest metadata
    contest_meta = {}
    date_dir = get_contest_data_dir() / date
    for csv_file in date_dir.glob("*.csv"):
        if "results" not in csv_file.name:
            try:
                meta_df = pd.read_csv(csv_file, encoding="utf-8-sig")
                meta_df.columns = meta_df.columns.str.strip()
                for _, row in meta_df.iterrows():
                    cid = str(row.get("ContestId", row.get("contest_id", "")))
                    if cid == contest_id:
                        contest_meta = {
                            "contest_name": row.get(
                                "ContestName", row.get("contest_name", "")
                            ),
                        }
                        break
            except Exception:
                pass

    contest_name = contest_meta.get("contest_name", f"Contest {contest_id}")

    # Compute ownership metrics for different tiers
    winner_entries = entries[entries["Rank"] == 1]
    top10_entries = entries[entries["Rank"] <= 10]
    top1pct_entries = entries[entries["Rank"] <= max(1, int(total_entries * 0.01))]
    cash_entries = entries[entries["Rank"] <= int(total_entries * 0.2)]

    winner_metrics = compute_ownership_metrics_for_entries(winner_entries, player_own)
    top10_metrics = compute_ownership_metrics_for_entries(top10_entries, player_own)
    top1pct_metrics = compute_ownership_metrics_for_entries(top1pct_entries, player_own)
    cash_metrics = compute_ownership_metrics_for_entries(cash_entries, player_own)
    field_metrics = compute_ownership_metrics_for_entries(entries, player_own)

    # Get scores
    winner_score = (
        float(winner_entries["Points"].iloc[0])
        if len(winner_entries) > 0 and pd.notna(winner_entries["Points"].iloc[0])
        else None
    )
    top1pct_score = (
        float(top1pct_entries["Points"].mean())
        if len(top1pct_entries) > 0
        else None
    )
    cash_line_rank = int(total_entries * 0.2)
    cash_line_entries = entries[entries["Rank"] == cash_line_rank]
    cash_line_score = (
        float(cash_line_entries["Points"].iloc[0])
        if len(cash_line_entries) > 0
        else None
    )

    # Compute player ownership with win/top10 rates
    player_in_winner: Counter[str] = Counter()
    player_in_top10: Counter[str] = Counter()

    for _, entry in winner_entries.iterrows():
        for player in parse_lineup(entry.get("Lineup", "")):
            player_in_winner[player] += 1

    for _, entry in top10_entries.iterrows():
        for player in parse_lineup(entry.get("Lineup", "")):
            player_in_top10[player] += 1

    num_winners = len(winner_entries)
    num_top10 = len(top10_entries)

    player_ownership_list = []
    for player, own_pct in sorted(player_own.items(), key=lambda x: -x[1]):
        player_ownership_list.append(
            PlayerOwnership(
                player=player,
                ownership_pct=own_pct,
                fpts=player_fpts.get(player),
                win_pct=round(100 * player_in_winner[player] / num_winners, 1)
                if num_winners > 0
                else None,
                top10_pct=round(100 * player_in_top10[player] / num_top10, 1)
                if num_top10 > 0
                else None,
            )
        )

    # Compute lineup-level data
    top_lineups = compute_lineup_ownership_entries(entries, player_own, limit=100)
    ownership_distribution = compute_ownership_distribution(entries, player_own, bin_size=10)
    dupe_analysis = compute_dupe_analysis(entries, player_own, bin_size=10)

    return ContestDetail(
        contest_id=contest_id,
        contest_name=contest_name,
        total_entries=total_entries,
        winner=winner_metrics,
        top10=top10_metrics,
        top1pct=top1pct_metrics,
        cash_line=cash_metrics,
        field_avg=field_metrics,
        winner_score=winner_score,
        top1pct_score=top1pct_score,
        cash_line_score=cash_line_score,
        leverage_plays=compute_leverage_scores(entries, player_own, top_n=10),
        stacks=compute_stack_analysis(entries, player_teams, top_n=10),
        game_correlations=compute_game_correlations(
            entries, player_teams, player_fpts, top_n=10
        ),
        player_ownership=player_ownership_list[:50],
        top_lineups=top_lineups,
        ownership_distribution=ownership_distribution,
        dupe_analysis=dupe_analysis,
    )


def get_pairs_analysis(
    date: str, contest_id: str, top_n: int = 10
) -> list[PlayerPair]:
    """Get player pair analysis for a contest."""
    results_path = (
        get_contest_data_dir() / date / "results" / f"contest_{contest_id}_results.csv"
    )
    if not results_path.exists():
        return []

    df = parse_contest_csv(results_path)
    entries = get_unique_entries(df)

    return compute_pairs(entries, top_n)


def find_user_lineups(date: str, pattern: str) -> list[UserLineupEntry]:
    """Find user's lineups across all contests for a date."""
    if not pattern:
        return []

    pattern_lower = pattern.lower()
    data_dir = get_contest_data_dir() / date
    results_dir = data_dir / "results"

    if not results_dir.exists():
        return []

    # Load contest metadata
    contest_meta: dict[str, str] = {}
    for csv_file in data_dir.glob("*.csv"):
        if "results" not in csv_file.name:
            try:
                meta_df = pd.read_csv(csv_file, encoding="utf-8-sig")
                meta_df.columns = meta_df.columns.str.strip()
                for _, row in meta_df.iterrows():
                    cid = str(row.get("ContestId", row.get("contest_id", "")))
                    contest_meta[cid] = row.get(
                        "ContestName", row.get("contest_name", f"Contest {cid}")
                    )
            except Exception:
                pass

    user_entries = []

    for results_file in results_dir.glob("contest_*_results.csv"):
        contest_id = results_file.stem.replace("contest_", "").replace("_results", "")

        try:
            df = parse_contest_csv(results_file)
            entries = get_unique_entries(df)
            total_entries = len(entries)

            if total_entries < 10:
                continue

            player_own = get_player_ownership_lookup(df)

            # Find matching entries
            for _, entry in entries.iterrows():
                entry_name = str(entry.get("EntryName", ""))
                if pattern_lower in entry_name.lower():
                    rank = int(entry["Rank"]) if pd.notna(entry.get("Rank")) else 0
                    points = (
                        float(entry["Points"]) if pd.notna(entry.get("Points")) else 0
                    )
                    lineup_players = parse_lineup(entry.get("Lineup", ""))
                    ownership = compute_lineup_ownership(lineup_players, player_own)

                    if ownership:
                        user_entries.append(
                            UserLineupEntry(
                                contest_id=contest_id,
                                contest_name=contest_meta.get(
                                    contest_id, f"Contest {contest_id}"
                                ),
                                entry_name=entry_name,
                                rank=rank,
                                points=points,
                                total_entries=total_entries,
                                pct_finish=round(100 * rank / total_entries, 2)
                                if total_entries > 0
                                else 0,
                                lineup_players=lineup_players,
                                ownership_metrics=ownership,
                            )
                        )
        except Exception:
            continue

    # Sort by rank
    user_entries.sort(key=lambda x: x.rank)
    return user_entries
