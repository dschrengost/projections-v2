from __future__ import annotations

import pandas as pd

GTV2_STATE_PREFIX = "gtv2_state_"
GTV2_AUX_COLUMNS = (
    "gtv2_minutes_deterministic",
    "gtv2_active_logit",
    "gtv2_active_prob_proxy",
)


def find_gtv2_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith(GTV2_STATE_PREFIX)]
    cols.extend([c for c in GTV2_AUX_COLUMNS if c in frame.columns])
    return sorted(dict.fromkeys(cols))


def merge_gtv2_embeddings(
    base_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], float]:
    """Left-join GTV2-derived embeddings on (game_date, player_id)."""
    if base_df.empty:
        raise ValueError("base_df must be non-empty")
    if embeddings_df.empty:
        return base_df.copy(), [], 0.0

    base = base_df.copy()
    emb = embeddings_df.copy()
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    emb["game_date"] = pd.to_datetime(emb["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce").fillna(0).astype("int64")
    emb["player_id"] = pd.to_numeric(emb["player_id"], errors="coerce").fillna(0).astype("int64")

    gtv2_cols = find_gtv2_feature_columns(emb)
    if not gtv2_cols:
        return base, [], 0.0

    keep_cols = ["game_date", "player_id", *gtv2_cols]
    dedup = emb[keep_cols].drop_duplicates(subset=["game_date", "player_id"], keep="last")
    merged = base.merge(dedup, on=["game_date", "player_id"], how="left")

    hit_mask = merged[gtv2_cols].notna().any(axis=1)
    coverage = float(hit_mask.mean()) if len(merged) else 0.0
    for col in gtv2_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged, gtv2_cols, coverage
