"""Export GTV2 player-state embeddings for ownership-model enrichment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections.pipeline.gtv2_inference_runtime import load_gtv2_model, resolve_torch_device
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    build_game_level_examples,
    collate_game_level_examples,
)


def _required_feature_columns(config) -> list[str]:
    cols = ["game_id", "player_id", "team_id", "game_date"]
    cols.extend(list(config.feature_columns))
    cols.extend(list(config.game_feature_columns))
    cols.extend(list(config.team_feature_columns))
    return sorted(dict.fromkeys(cols))


def _load_candidate_rows(
    *,
    ownership_base_path: Path,
    gtv2_features_path: Path,
    config,
) -> pd.DataFrame:
    ownership = pd.read_parquet(ownership_base_path, columns=["game_date", "player_id"])
    ownership["game_date"] = pd.to_datetime(ownership["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ownership["player_id"] = pd.to_numeric(ownership["player_id"], errors="coerce").fillna(0).astype("int64")
    ownership_keys = ownership.drop_duplicates()

    features = pd.read_parquet(gtv2_features_path)
    required = _required_feature_columns(config)
    missing = [col for col in required if col not in features.columns]
    for col in missing:
        features[col] = 0.0
    features = features[required].copy()
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    features["player_id"] = pd.to_numeric(features["player_id"], errors="coerce").fillna(0).astype("int64")
    features["team_id"] = pd.to_numeric(features["team_id"], errors="coerce").fillna(0).astype("int64")
    features["game_id"] = pd.to_numeric(features["game_id"], errors="coerce").fillna(0).astype("int64")
    merged = features.merge(ownership_keys, on=["game_date", "player_id"], how="inner")
    return merged.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)


def _score_player_states(
    frame: pd.DataFrame,
    *,
    config,
    model,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["minutes"] = 0.0
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    examples = build_game_level_examples(
        work,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        flow_label_columns=[],
        minutes_label_col="minutes",
        min_valid_players_per_team=max(1, int(config.min_active_count)),
        overflow_protected_prior_play_prob_floor=float(config.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(config.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(config.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(config.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(config.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(config.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(config.overflow_keep_weight_prior_minutes),
    )
    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )

    rows: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["player_features"].to(device=device),
                batch["player_valid_mask"].to(device=device),
                game_features=batch["game_features"].to(device=device),
                team_features=batch["team_features"].to(device=device),
                sample_active=False,
                run_flow=False,
            )

            valid = batch["player_valid_mask"].cpu().numpy().astype(bool)
            player_ids = batch["player_ids"].cpu().numpy().astype(np.int64)
            team_ids = batch["team_ids"].cpu().numpy().astype(np.int64)
            game_ids = [int(v) for v in batch["game_id_norm"]]
            game_dates = [str(v) for v in batch["game_date"]]

            player_states = out.player_states.detach().cpu().numpy()
            minutes = out.minutes.minutes.detach().cpu().numpy()
            active_logits = out.active.player_logits.detach().cpu().numpy()
            active_prob = 1.0 / (1.0 + np.exp(-np.clip(active_logits, -40.0, 40.0)))

            for b_idx in range(player_states.shape[0]):
                valid_flat = np.concatenate([valid[b_idx, 0], valid[b_idx, 1]], axis=0)
                player_flat = np.concatenate([player_ids[b_idx, 0], player_ids[b_idx, 1]], axis=0)
                team_flat = np.concatenate(
                    [
                        np.full((15,), int(team_ids[b_idx, 0]), dtype=np.int64),
                        np.full((15,), int(team_ids[b_idx, 1]), dtype=np.int64),
                    ],
                    axis=0,
                )
                for idx in np.where(valid_flat)[0]:
                    row: dict[str, float | int | str] = {
                        "game_date": game_dates[b_idx],
                        "game_id": int(game_ids[b_idx]),
                        "team_id": int(team_flat[idx]),
                        "player_id": int(player_flat[idx]),
                        "gtv2_minutes_deterministic": float(minutes[b_idx, idx]),
                        "gtv2_active_logit": float(active_logits[b_idx, idx]),
                        "gtv2_active_prob_proxy": float(active_prob[b_idx, idx]),
                    }
                    state = player_states[b_idx, idx]
                    for j, value in enumerate(state.tolist()):
                        row[f"gtv2_state_{j:03d}"] = float(value)
                    rows.append(row)

    if not rows:
        raise RuntimeError("no GTV2 embeddings were produced")
    return pd.DataFrame(rows).drop_duplicates(subset=["game_date", "player_id"], keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GTV2 player-state embeddings for ownership modeling")
    parser.add_argument("--ownership-base", type=Path, required=True)
    parser.add_argument("--gtv2-features-parquet", type=Path, required=True)
    parser.add_argument(
        "--gtv2-bundle-dir",
        type=Path,
        default=Path("/home/daniel/projections-data/artifacts/game_transformer_v2_experiments/bundle_current"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = resolve_torch_device(str(args.device) if args.device != "auto" else None)
    config, model = load_gtv2_model(args.gtv2_bundle_dir, device=device)
    candidate_rows = _load_candidate_rows(
        ownership_base_path=args.ownership_base,
        gtv2_features_path=args.gtv2_features_parquet,
        config=config,
    )
    print(
        f"[gtv2_ownership_embed] candidate rows={len(candidate_rows):,} "
        f"dates={candidate_rows['game_date'].nunique()} players={candidate_rows['player_id'].nunique()}"
    )
    scored = _score_player_states(
        candidate_rows,
        config=config,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    print(f"[gtv2_ownership_embed] wrote {len(scored):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
