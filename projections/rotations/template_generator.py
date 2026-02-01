from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from projections.rotations.generator import RotationWorlds, TeamContext
from projections.rotations.priors_humility import HumilityConfig, apply_prior_humility, humility_config_as_dict
from projections.rotations.rotation_gate import GateConfig, apply_rotation_gate, gate_config_as_dict
from projections.rotations.schemas import LINEUP_COLS


DEFAULT_ROT_ARTIFACTS_ROOT = Path("/home/daniel/projections-data/artifacts/rot_v1")


def _resolve_bundle_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file():
        run_id = path.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError(f"Empty rot bundle pointer: {path}")
        resolved = path.parent / run_id
        if not resolved.exists():
            raise FileNotFoundError(f"Pointer {path} -> {resolved} does not exist")
        return resolved
    raise FileNotFoundError(f"rot bundle not found: {path}")


def _regime_from_count(count_ge_5: int) -> str:
    if int(count_ge_5) <= 8:
        return "tight"
    if int(count_ge_5) <= 10:
        return "normal"
    return "deep"


def _choose_regime_label(ctx: TeamContext, *, candidate_ids: Optional[list[int]]) -> str:
    if ctx.regime_label in {"tight", "normal", "deep"}:
        return str(ctx.regime_label)
    prior = ctx.minutes_prior or {}
    if candidate_ids is None:
        # No gating: choose a reasonable default.
        return "normal"
    if prior:
        count_ge_5 = sum(1 for pid in candidate_ids if float(prior.get(int(pid), 0.0)) >= 5.0)
        return _regime_from_count(count_ge_5)
    # Without priors, use candidate set size as a proxy.
    return _regime_from_count(len(candidate_ids))


def _unique_preserve_order(values: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        iv = int(v)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def _sort_candidates_by_prior(candidate_ids: list[int], prior: dict[int, float]) -> list[int]:
    return sorted(candidate_ids, key=lambda pid: (-float(prior.get(int(pid), 0.0)), int(pid)))


def _summary_stats_1d(values: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(values, errors="coerce").astype(np.float64)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    arr = s.to_numpy(dtype=np.float64)
    q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9]).tolist()
    return {"mean": float(arr.mean()), "p10": float(q10), "p50": float(q50), "p90": float(q90)}


@dataclass(frozen=True)
class _Template:
    team_id: int
    game_id: str
    regime_label: str
    lineups: np.ndarray  # shape (n_segments, 5) of template player ids
    durations_sec: np.ndarray  # shape (n_segments,) int64
    starters: tuple[int, int, int, int, int]
    players_by_minutes: tuple[int, ...]
    total_seconds: int


class TemplateRotationGenerator:
    """MVP rotation sampler based on historical stint templates.

    This is Phase 2.0/2.1 base plumbing: load rot_v1 bundle, sample templates, map roles.
    """

    def __init__(
        self,
        *,
        rot_bundle: Optional[Path] = None,
        duration_jitter_std_sec: float = 0.0,
        max_attempts_per_world: int = 50,
        humility_config: HumilityConfig | None = None,
        gate_config: GateConfig | None = None,
        gate_preds: pd.DataFrame | None = None,
    ) -> None:
        self._rot_bundle_path = rot_bundle or (DEFAULT_ROT_ARTIFACTS_ROOT / "LATEST_PUBLISHED")
        self._duration_jitter_std_sec = float(duration_jitter_std_sec)
        self._max_attempts_per_world = int(max_attempts_per_world)
        self._humility_config = humility_config or HumilityConfig()
        self._gate_config = gate_config or GateConfig()

        self._loaded = False
        self._templates: dict[tuple[int, str], _Template] = {}
        self._templates_by_team_regime: dict[tuple[int, str], list[tuple[int, str]]] = {}
        self._templates_by_regime: dict[str, list[tuple[int, str]]] = {}
        self._gate_preds_by_team_game: dict[tuple[int, str], pd.DataFrame] = {}

        if gate_preds is not None and not gate_preds.empty:
            required = {"game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"}
            missing = sorted([c for c in required if c not in gate_preds.columns])
            if missing:
                raise ValueError(f"gate_preds missing required columns: {missing}")
            tmp = gate_preds[["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"]].copy()
            tmp["game_id"] = tmp["game_id"].astype("string")
            tmp["team_id"] = pd.to_numeric(tmp["team_id"], errors="coerce").astype("Int64")
            tmp["player_id"] = pd.to_numeric(tmp["player_id"], errors="coerce").astype("Int64")
            tmp["p_ge5_pred"] = pd.to_numeric(tmp["p_ge5_pred"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
            tmp["p_ge15_pred"] = pd.to_numeric(tmp["p_ge15_pred"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
            tmp = tmp.dropna(subset=["game_id", "team_id", "player_id"]).copy()
            tmp["team_id"] = tmp["team_id"].astype(int)
            tmp["player_id"] = tmp["player_id"].astype(int)
            tmp = tmp.sort_values(["team_id", "game_id", "player_id"], kind="mergesort").drop_duplicates(
                subset=["team_id", "game_id", "player_id"],
                keep="last",
            )
            for (team_id, game_id), g in tmp.groupby(["team_id", "game_id"], sort=True):
                self._gate_preds_by_team_game[(int(team_id), str(game_id))] = g.copy()

    def _load(self) -> None:
        if self._loaded:
            return
        bundle_dir = _resolve_bundle_dir(Path(self._rot_bundle_path))
        events_path = bundle_dir / "rotation_events.parquet"
        labels_path = bundle_dir / "rotation_labels.parquet"
        if not events_path.exists():
            raise FileNotFoundError(f"Missing rotation_events: {events_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing rotation_labels: {labels_path}")

        events_cols = ["team_id", "game_id", "segment_idx", "duration_sec", *LINEUP_COLS]
        events = pd.read_parquet(events_path, columns=events_cols)
        labels = pd.read_parquet(labels_path, columns=["team_id", "game_id", "regime_label"])

        labels = labels.drop_duplicates(subset=["team_id", "game_id"], keep="first").copy()
        labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce").astype("int64")
        labels["game_id"] = labels["game_id"].astype("string")
        labels["regime_label"] = labels["regime_label"].astype("string")

        events["team_id"] = pd.to_numeric(events["team_id"], errors="coerce").astype("int64")
        events["game_id"] = events["game_id"].astype("string")
        events["segment_idx"] = pd.to_numeric(events["segment_idx"], errors="coerce").astype("int64")
        events["duration_sec"] = pd.to_numeric(events["duration_sec"], errors="coerce").astype("int64")
        for c in LINEUP_COLS:
            events[c] = pd.to_numeric(events[c], errors="coerce").astype("int64")

        events = events.sort_values(["team_id", "game_id", "segment_idx"], kind="mergesort").reset_index(drop=True)
        label_map = dict(
            ((int(r.team_id), str(r.game_id)), str(r.regime_label))
            for r in labels.itertuples(index=False)
        )

        templates: dict[tuple[int, str], _Template] = {}
        templates_by_team_regime: dict[tuple[int, str], list[tuple[int, str]]] = {}
        templates_by_regime: dict[str, list[tuple[int, str]]] = {}

        for (team_id, game_id), g in events.groupby(["team_id", "game_id"], sort=True):
            key = (int(team_id), str(game_id))
            regime = label_map.get(key, "unknown")

            segs = g.sort_values("segment_idx", kind="mergesort")
            lineups = segs.loc[:, list(LINEUP_COLS)].to_numpy(dtype=np.int64)
            durations = segs["duration_sec"].to_numpy(dtype=np.int64)
            total_seconds = int(durations.sum())

            # starters = first non-zero segment lineup (fallback to first segment).
            starter_idx = int(np.argmax(durations > 0)) if (durations > 0).any() else 0
            starters = tuple(int(v) for v in lineups[starter_idx].tolist())  # type: ignore[assignment]

            # role ordering by template minutes.
            player_seconds: dict[int, int] = {}
            for duration, lineup in zip(durations.tolist(), lineups.tolist()):
                for pid in lineup:
                    player_seconds[int(pid)] = int(player_seconds.get(int(pid), 0)) + int(duration)
            players_by_minutes = tuple(
                pid for pid, _ in sorted(player_seconds.items(), key=lambda kv: (-kv[1], kv[0]))
            )

            tmpl = _Template(
                team_id=int(team_id),
                game_id=str(game_id),
                regime_label=str(regime),
                lineups=lineups,
                durations_sec=durations,
                starters=starters,  # type: ignore[arg-type]
                players_by_minutes=players_by_minutes,
                total_seconds=total_seconds,
            )
            templates[key] = tmpl
            templates_by_team_regime.setdefault((tmpl.team_id, tmpl.regime_label), []).append(key)
            templates_by_regime.setdefault(tmpl.regime_label, []).append(key)

        # Deterministic ordering of template lists.
        for k in list(templates_by_team_regime.keys()):
            templates_by_team_regime[k] = sorted(templates_by_team_regime[k], key=lambda x: (x[0], x[1]))
        for k in list(templates_by_regime.keys()):
            templates_by_regime[k] = sorted(templates_by_regime[k], key=lambda x: (x[0], x[1]))

        self._templates = templates
        self._templates_by_team_regime = templates_by_team_regime
        self._templates_by_regime = templates_by_regime
        self._loaded = True

    def _candidate_ids(self, ctx: TeamContext) -> Optional[list[int]]:
        if ctx.candidate_player_ids:
            return _unique_preserve_order(ctx.candidate_player_ids)
        if ctx.minutes_prior:
            return _unique_preserve_order(ctx.minutes_prior.keys())
        if ctx.starter_candidates:
            return _unique_preserve_order(ctx.starter_candidates)
        return None

    def _choose_template_keys(self, *, team_id: int, regime_label: str) -> tuple[list[tuple[int, str]], str]:
        team_keys = self._templates_by_team_regime.get((int(team_id), str(regime_label)), [])
        if team_keys:
            return team_keys, "team"
        league_keys = self._templates_by_regime.get(str(regime_label), [])
        if league_keys:
            return league_keys, "league"
        # Last-resort fallback: anything for the team, else anything at all.
        any_team = sorted(
            [k for k in self._templates.keys() if k[0] == int(team_id)],
            key=lambda x: (x[0], x[1]),
        )
        if any_team:
            return any_team, "team_any_regime"
        return sorted(self._templates.keys(), key=lambda x: (x[0], x[1])), "league_any_regime"

    def _map_roles(
        self,
        *,
        template: _Template,
        ctx: TeamContext,
        rng: np.random.Generator,
        candidate_ids: list[int],
    ) -> Optional[dict[int, int]]:
        template_starters = list(template.starters)
        if len(set(template_starters)) != 5:
            return None

        candidate_ids = _unique_preserve_order(candidate_ids)
        if len(candidate_ids) < 5:
            return None

        prior = ctx.minutes_prior or {}

        starter_pool = _unique_preserve_order(ctx.starter_candidates) if ctx.starter_candidates else []
        chosen_starters: list[int] = []
        if starter_pool:
            chosen_starters.extend(starter_pool)
        if len(chosen_starters) < 5:
            fill_pool = [pid for pid in _sort_candidates_by_prior(candidate_ids, prior) if pid not in chosen_starters]
            chosen_starters.extend(fill_pool[: (5 - len(chosen_starters))])
        chosen_starters = chosen_starters[:5]
        if len(chosen_starters) != 5 or len(set(chosen_starters)) != 5:
            return None

        mapping: dict[int, int] = dict(zip(template_starters, chosen_starters))

        template_players = list(template.players_by_minutes)
        bench_template = [pid for pid in template_players if pid not in template_starters]
        remaining_candidates = [pid for pid in candidate_ids if pid not in chosen_starters]
        if len(remaining_candidates) < len(bench_template):
            return None

        if prior:
            remaining_candidates = _sort_candidates_by_prior(remaining_candidates, prior)
        else:
            remaining_candidates = [int(v) for v in rng.permutation(np.array(remaining_candidates, dtype=np.int64)).tolist()]

        bench_assign = remaining_candidates[: len(bench_template)]
        mapping.update(dict(zip(bench_template, bench_assign)))

        if len(mapping) != len(set(mapping.keys())):
            return None
        if len(set(mapping.values())) != len(mapping.values()):
            return None
        return mapping

    def _jitter_durations(self, *, durations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = float(self._duration_jitter_std_sec)
        if std <= 0:
            return durations.astype(np.float64)
        noise = rng.normal(loc=0.0, scale=std, size=durations.shape[0]).astype(np.float64)
        out = durations.astype(np.float64) + noise
        out = np.clip(out, 0.0, None)
        total = float(out.sum())
        target = float(durations.sum())
        if total <= 0:
            return durations.astype(np.float64)
        out = out * (target / total)
        # Force exact conservation in float space.
        drift = float(target - out.sum())
        out[-1] += drift
        return out

    def generate(self, ctx: TeamContext) -> RotationWorlds:
        self._load()

        effective_ctx = ctx
        n_worlds = int(ctx.n_worlds)
        if n_worlds <= 0:
            raise ValueError("n_worlds must be positive")

        candidate_ids = self._candidate_ids(ctx)
        humility_tier_counts = None
        heuristics_applied_n = None
        heuristics_applied_by_tier = None
        heuristics_stats = None
        gate_tier_counts = None
        gate_missing_preds_n = None
        gate_excluded_n = None
        gate_player_p_ge5 = None
        gate_player_p_ge15 = None
        gate_player_p_ge5_used = None
        gate_player_p_ge15_used = None
        gate_player_tier = None
        gate_player_reason = None
        gate_player_missing_pred = None
        gate_player_excluded = None
        gate_player_minutes_cap = None
        gate_player_play_prob_cap = None
        gate_player_minutes_prior_adj = None
        gate_player_play_prob_adj = None
        gate_minutes_cap_by_player: dict[int, float] = {}
        if (
            bool(self._humility_config.enabled)
            and candidate_ids is not None
            and (effective_ctx.minutes_prior is not None)
            and len(candidate_ids) > 0
        ):
            prior = effective_ctx.minutes_prior or {}
            p10 = effective_ctx.minutes_p10_prior
            p90 = effective_ctx.minutes_p90_prior
            pp = effective_ctx.play_prob_prior or {}
            starters = set(int(v) for v in (effective_ctx.starter_candidates or []))

            data: dict[str, list] = {
                "game_id": [str(effective_ctx.game_id)] * len(candidate_ids),
                "team_id": [int(effective_ctx.team_id)] * len(candidate_ids),
                "player_id": [int(pid) for pid in candidate_ids],
                "minutes_prior": [float(prior.get(int(pid), 0.0)) for pid in candidate_ids],
                "play_prob": [float(pp.get(int(pid), 1.0)) for pid in candidate_ids],
                "starter_candidate": [bool(int(pid) in starters) for pid in candidate_ids],
            }
            if p10 is not None:
                data["minutes_p10"] = [float(p10.get(int(pid), prior.get(int(pid), 0.0))) for pid in candidate_ids]
            if p90 is not None:
                data["minutes_p90"] = [float(p90.get(int(pid), prior.get(int(pid), 0.0))) for pid in candidate_ids]
            df_priors = pd.DataFrame(data)
            df_adj = apply_prior_humility(df_priors, self._humility_config)

            minutes_prior_adj = {
                int(pid): float(v)
                for pid, v in zip(df_adj["player_id"].tolist(), df_adj["minutes_prior_adj"].tolist())
            }
            minutes_p10_adj = {
                int(pid): float(v)
                for pid, v in zip(df_adj["player_id"].tolist(), df_adj["minutes_p10_adj"].tolist())
            }
            minutes_p90_adj = {
                int(pid): float(v)
                for pid, v in zip(df_adj["player_id"].tolist(), df_adj["minutes_p90_adj"].tolist())
            }
            play_prob_adj = {
                int(pid): float(v)
                for pid, v in zip(df_adj["player_id"].tolist(), df_adj["play_prob_adj"].tolist())
            }

            humility_tier_counts = (
                df_adj["humility_tier"].fillna("unknown").value_counts(dropna=False).to_dict()
                if "humility_tier" in df_adj.columns
                else None
            )

            if "rotation_prior_heuristics_applied" in df_adj.columns:
                applied_mask = df_adj["rotation_prior_heuristics_applied"].fillna(False).astype(bool)
                heuristics_applied_n = int(applied_mask.sum())
                if "humility_tier" in df_adj.columns:
                    tmp = df_adj.loc[applied_mask, ["humility_tier"]].copy()
                    heuristics_applied_by_tier = (
                        tmp["humility_tier"].fillna("unknown").value_counts(dropna=False).to_dict()
                        if not tmp.empty
                        else None
                    )
            if ("p_ge5_prior_heur" in df_adj.columns) and ("p_eq0_prior_heur" in df_adj.columns):
                heuristics_stats = {
                    "p_ge5_prior_heur": _summary_stats_1d(df_adj["p_ge5_prior_heur"]),
                    "p_eq0_prior_heur": _summary_stats_1d(df_adj["p_eq0_prior_heur"]),
                }

            effective_ctx = replace(
                effective_ctx,
                minutes_prior=minutes_prior_adj,
                minutes_p10_prior=minutes_p10_adj,
                minutes_p90_prior=minutes_p90_adj,
                play_prob_prior=play_prob_adj,
            )

        rng = np.random.default_rng(int(effective_ctx.rng_seed))

        regime_label = _choose_regime_label(effective_ctx, candidate_ids=candidate_ids)
        template_keys, source = self._choose_template_keys(team_id=int(effective_ctx.team_id), regime_label=regime_label)
        if not template_keys:
            raise RuntimeError("Template library is empty (no rot_v1 templates loaded).")

        # Rotation Gate: purely non-structural post-map capper.
        #
        # - NEVER changes candidate eligibility, mapping inputs, regime selection, or template selection.
        # - ONLY produces diagnostics + optional caps used during minutes/world generation below.
        if (
            bool(self._gate_config.enabled)
            and candidate_ids is not None
            and (effective_ctx.minutes_prior is not None)
            and len(candidate_ids) > 0
        ):
            prior = effective_ctx.minutes_prior or {}
            p10 = effective_ctx.minutes_p10_prior
            p90 = effective_ctx.minutes_p90_prior
            pp = effective_ctx.play_prob_prior or {}
            starters = set(int(v) for v in (effective_ctx.starter_candidates or []))

            data: dict[str, list] = {
                "game_id": [str(effective_ctx.game_id)] * len(candidate_ids),
                "team_id": [int(effective_ctx.team_id)] * len(candidate_ids),
                "player_id": [int(pid) for pid in candidate_ids],
                "minutes_prior": [float(prior.get(int(pid), 0.0)) for pid in candidate_ids],
                "play_prob": [float(pp.get(int(pid), 1.0)) for pid in candidate_ids],
            }
            if p10 is not None:
                data["minutes_p10"] = [float(p10.get(int(pid), prior.get(int(pid), 0.0))) for pid in candidate_ids]
            if p90 is not None:
                data["minutes_p90"] = [float(p90.get(int(pid), prior.get(int(pid), 0.0))) for pid in candidate_ids]
            df_priors = pd.DataFrame(data)

            preds_df = self._gate_preds_by_team_game.get((int(effective_ctx.team_id), str(effective_ctx.game_id)))
            if preds_df is None:
                preds_df = pd.DataFrame(columns=["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"])

            df_adj = apply_rotation_gate(
                df_priors,
                preds_df,
                starters_set=starters,
                cfg=self._gate_config,
                seed=int(effective_ctx.rng_seed),
            )

            gate_tier_counts = (
                df_adj["gate_tier"].fillna("unknown").value_counts(dropna=False).to_dict()
                if "gate_tier" in df_adj.columns
                else None
            )
            missing_mask = df_adj["gate_missing_pred"].fillna(False).astype(bool) if "gate_missing_pred" in df_adj.columns else None
            gate_missing_preds_n = int(missing_mask.sum()) if missing_mask is not None else None
            excluded_mask = df_adj["gate_excluded"].fillna(False).astype(bool) if "gate_excluded" in df_adj.columns else None
            gate_excluded_n = int(excluded_mask.sum()) if excluded_mask is not None else None

            gate_player_p_ge5 = (
                {
                    int(pid): float(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        pd.to_numeric(df_adj["p_ge5_pred"], errors="coerce").astype(np.float64).fillna(np.nan).tolist(),
                    )
                }
                if "p_ge5_pred" in df_adj.columns
                else None
            )
            gate_player_p_ge15 = (
                {
                    int(pid): float(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        pd.to_numeric(df_adj["p_ge15_pred"], errors="coerce").astype(np.float64).fillna(np.nan).tolist(),
                    )
                }
                if "p_ge15_pred" in df_adj.columns
                else None
            )
            gate_player_p_ge5_used = (
                {
                    int(pid): float(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        pd.to_numeric(df_adj["p_ge5_used"], errors="coerce").astype(np.float64).fillna(np.nan).tolist(),
                    )
                }
                if "p_ge5_used" in df_adj.columns
                else None
            )
            gate_player_p_ge15_used = (
                {
                    int(pid): float(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        pd.to_numeric(df_adj["p_ge15_used"], errors="coerce").astype(np.float64).fillna(np.nan).tolist(),
                    )
                }
                if "p_ge15_used" in df_adj.columns
                else None
            )
            gate_player_tier = (
                {
                    int(pid): str(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        df_adj["gate_tier"].fillna("unknown").astype("string").tolist(),
                    )
                }
                if "gate_tier" in df_adj.columns
                else None
            )
            gate_player_reason = (
                {
                    int(pid): str(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        df_adj["gate_reason"].fillna("").astype("string").tolist(),
                    )
                }
                if "gate_reason" in df_adj.columns
                else None
            )
            gate_player_missing_pred = (
                {int(pid): bool(v) for pid, v in zip(df_adj["player_id"].tolist(), missing_mask.tolist())}
                if ("player_id" in df_adj.columns and missing_mask is not None)
                else None
            )
            gate_player_excluded = (
                {int(pid): bool(v) for pid, v in zip(df_adj["player_id"].tolist(), excluded_mask.tolist())}
                if ("player_id" in df_adj.columns and excluded_mask is not None)
                else None
            )
            gate_player_minutes_cap = (
                {
                    int(pid): float(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        pd.to_numeric(df_adj["gate_minutes_cap"], errors="coerce").astype(np.float64).fillna(np.nan).tolist(),
                    )
                }
                if "gate_minutes_cap" in df_adj.columns
                else None
            )
            gate_player_play_prob_cap = (
                {
                    int(pid): float(v)
                    for pid, v in zip(
                        df_adj["player_id"].tolist(),
                        pd.to_numeric(df_adj["gate_play_prob_cap"], errors="coerce").astype(np.float64).fillna(np.nan).tolist(),
                    )
                }
                if "gate_play_prob_cap" in df_adj.columns
                else None
            )
            gate_player_minutes_prior_adj = {
                int(pid): float(v) for pid, v in zip(df_adj["player_id"].tolist(), df_adj["minutes_prior_adj"].tolist())
            }
            gate_player_play_prob_adj = {
                int(pid): float(v) for pid, v in zip(df_adj["player_id"].tolist(), df_adj["play_prob_adj"].tolist())
            }

            # Caps are applied during minutes/world generation (never used for mapping/template selection).
            if gate_player_minutes_cap is not None:
                gate_minutes_cap_by_player = {
                    int(pid): float(cap)
                    for pid, cap in gate_player_minutes_cap.items()
                    if cap is not None and np.isfinite(float(cap))
                }

        if candidate_ids is not None:
            minutes_by_player: dict[int, np.ndarray] = {
                int(pid): np.zeros(n_worlds, dtype=np.float64) for pid in candidate_ids
            }
        else:
            minutes_by_player = {}

        mapping_success = 0
        template_resamples = 0
        fallback_to_prior = 0

        for w in range(n_worlds):
            mapping: Optional[dict[int, int]] = None
            chosen_template_key: Optional[tuple[int, str]] = None

            for attempt in range(self._max_attempts_per_world):
                idx = int(rng.integers(low=0, high=len(template_keys)))
                key = template_keys[idx]
                tmpl = self._templates[key]

                if candidate_ids is None:
                    chosen_template_key = key
                    mapping = None
                    break

                mapping_try = self._map_roles(
                    template=tmpl,
                    ctx=effective_ctx,
                    rng=rng,
                    candidate_ids=candidate_ids,
                )
                if mapping_try is not None:
                    chosen_template_key = key
                    mapping = mapping_try
                    break
                template_resamples += 1

            if chosen_template_key is None:
                fallback_to_prior += 1
                # Fallback: allocate regulation minutes proportional to minutes_prior (if any), else zeros.
                prior = effective_ctx.minutes_prior or {}
                if (candidate_ids is None) or (not candidate_ids):
                    continue
                weights = np.array(
                    [max(float(prior.get(int(pid), 0.0)), 0.0) for pid in candidate_ids],
                    dtype=float,
                )
                if weights.sum() <= 0:
                    continue
                weights = weights / weights.sum()
                team_minutes = 240.0
                for pid, frac in zip(candidate_ids, weights.tolist()):
                    minutes_by_player[int(pid)][w] = team_minutes * float(frac)
                if gate_minutes_cap_by_player:
                    for pid, cap in gate_minutes_cap_by_player.items():
                        if int(pid) in minutes_by_player:
                            minutes_by_player[int(pid)][w] = float(np.minimum(minutes_by_player[int(pid)][w], cap))
                continue

            tmpl = self._templates[chosen_template_key]
            durations = self._jitter_durations(durations=tmpl.durations_sec, rng=rng)
            lineups = tmpl.lineups

            if candidate_ids is None:
                # Identity mapping to template player ids; union keys as we go.
                for duration, lineup in zip(durations.tolist(), lineups.tolist()):
                    for template_pid in lineup:
                        pid = int(template_pid)
                        if pid not in minutes_by_player:
                            minutes_by_player[pid] = np.zeros(n_worlds, dtype=np.float64)
                        minutes_by_player[pid][w] += float(duration) / 60.0
            else:
                assert mapping is not None
                mapping_success += 1
                for duration, lineup in zip(durations.tolist(), lineups.tolist()):
                    for template_pid in lineup:
                        pid = int(mapping[int(template_pid)])
                        add = float(duration) / 60.0
                        cap = gate_minutes_cap_by_player.get(int(pid), np.nan)
                        if np.isfinite(cap):
                            cur = float(minutes_by_player[pid][w])
                            if cur >= float(cap):
                                continue
                            minutes_by_player[pid][w] = float(min(cur + add, float(cap)))
                        else:
                            minutes_by_player[pid][w] += add

        diagnostics = {
            "regime_label": regime_label,
            "template_source": source,
            "mapping_success_rate": (mapping_success / n_worlds) if candidate_ids is not None else None,
            "template_resamples_total": int(template_resamples),
            "fallback_to_prior_worlds": int(fallback_to_prior),
            "humility_enabled": bool(self._humility_config.enabled),
            "humility_config": humility_config_as_dict(self._humility_config),
            "humility_tier_counts": humility_tier_counts,
            "rotation_prior_heuristics_applied_n": heuristics_applied_n,
            "rotation_prior_heuristics_applied_by_tier": heuristics_applied_by_tier,
            "rotation_prior_heuristics_stats": heuristics_stats,
            "gate_enabled": bool(self._gate_config.enabled),
            "gate_config": gate_config_as_dict(self._gate_config),
            "gate_tier_counts": gate_tier_counts,
            "gate_missing_preds_n": gate_missing_preds_n,
            "gate_excluded_n": gate_excluded_n,
            "gate_player_p_ge5_pred": gate_player_p_ge5,
            "gate_player_p_ge15_pred": gate_player_p_ge15,
            "gate_player_p_ge5_used": gate_player_p_ge5_used,
            "gate_player_p_ge15_used": gate_player_p_ge15_used,
            "gate_player_tier": gate_player_tier,
            "gate_player_reason": gate_player_reason,
            "gate_player_missing_pred": gate_player_missing_pred,
            "gate_player_excluded": gate_player_excluded,
            "gate_player_minutes_cap": gate_player_minutes_cap,
            "gate_player_play_prob_cap": gate_player_play_prob_cap,
            "gate_player_minutes_prior_adj": gate_player_minutes_prior_adj,
            "gate_player_play_prob_adj": gate_player_play_prob_adj,
        }

        return RotationWorlds(minutes_by_player=minutes_by_player, diagnostics=diagnostics)
