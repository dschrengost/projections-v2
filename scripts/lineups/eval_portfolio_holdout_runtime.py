"""Evaluate a selected portfolio on base-holdout + runtime-generated holdout worlds.

Option C: runtime holdouts are generated at evaluation time only (not a Prefect artifact).

Example:
  uv run python scripts/lineups/eval_portfolio_holdout_runtime.py \\
    --date 2026-01-10 --site dk \\
    --lineups-path artifacts/lineups/dk/game_date=2026-01-10/lineups.csv \\
    --base-worlds-path /home/daniel/projections-data/artifacts/sim_v2/worlds_fpts_v2/game_date=2026-01-10/run=ABC/worlds_matrix.parquet \\
    --seed 123 --k-runtime-holdouts 3 --num-worlds-runtime 10000
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer

from projections.eval.portfolio_holdout_runtime import (
    add_threshold_metric,
    compute_diversity_diagnostics,
    compute_threshold_from_train_max,
    compute_train_holdout_indices,
    load_worlds_matrix_parquet,
    parse_lineups_csv,
    score_portfolio_on_worlds,
    stable_hash_bytes,
    stable_hash_json,
    summarize_lineup_scores,
)
from projections.paths import get_data_root
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, load_sim_v2_profile
from projections.sim_v2.runtime_worlds import RuntimeWorldsResult, WorldsPerturbation, generate_worlds_matrix_sim_v2

app = typer.Typer(add_completion=False)


def _utc_ts() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_info() -> dict[str, Any]:
    info: dict[str, Any] = {"git_sha": None, "git_dirty": None}
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        info["git_sha"] = sha
    except Exception:
        return info

    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        info["git_dirty"] = bool(dirty)
    except Exception:
        info["git_dirty"] = None
    return info


def _resolve_worlds_matrix_path(base_worlds_path: Path) -> Path:
    if base_worlds_path.is_dir():
        candidate = base_worlds_path / "worlds_matrix.parquet"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing worlds_matrix.parquet under {base_worlds_path}")
    if base_worlds_path.exists():
        return base_worlds_path
    raise FileNotFoundError(f"Base worlds path not found: {base_worlds_path}")


def _format_float(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.2f}"


def _attach_lineup_ids(summary: dict[str, Any], lineup_ids: list[str]) -> dict[str, Any]:
    per = summary.get("per_lineup") or []
    for row in per:
        idx = int(row.get("lineup_idx", -1))
        if 0 <= idx < len(lineup_ids):
            row["lineup_id"] = lineup_ids[idx]
    return summary


def _collect_missing_players(lineups: list[list[str]], player_index: dict[str, int]) -> list[str]:
    missing: set[str] = set()
    for lu in lineups:
        for pid in lu:
            pid_s = str(pid).strip()
            if not pid_s:
                continue
            if pid_s not in player_index:
                missing.add(pid_s)
    return sorted(missing)


def _load_perturbation(path: Path) -> WorldsPerturbation:
    raw = _read_json(path)
    if not raw:
        raise ValueError(f"Invalid or empty perturb_cfg JSON: {path}")

    def _get(name: str, default: float = 1.0) -> float:
        val = raw.get(name, default)
        try:
            return float(val)
        except Exception as exc:
            raise ValueError(f"Invalid perturb_cfg[{name}]={val!r}") from exc

    return WorldsPerturbation(
        minutes_sigma_min_mult=_get("minutes_sigma_min_mult"),
        team_sigma_scale_mult=_get("team_sigma_scale_mult"),
        player_sigma_scale_mult=_get("player_sigma_scale_mult"),
        team_factor_sigma_mult=_get("team_factor_sigma_mult"),
        team_factor_gamma_mult=_get("team_factor_gamma_mult"),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _render_summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "set",
        "n_worlds",
        "max_mean",
        "max_p90",
        "max_p95",
        "max_p99",
        "p(max>T)",
    ]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(r.get("set")),
                    str(r.get("n_worlds")),
                    _format_float(r.get("max_mean")),
                    _format_float(r.get("max_p90")),
                    _format_float(r.get("max_p95")),
                    _format_float(r.get("max_p99")),
                    _format_float(r.get("p_max_gt_threshold")),
                ]
            )
            + " |"
        )
    return "\n".join(out)


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    site: str = typer.Option(..., "--site", help="dk|fd (used for labeling only)."),
    lineups_path: Path = typer.Option(..., "--lineups-path", exists=True, readable=True),
    base_worlds_path: Path = typer.Option(..., "--base-worlds-path"),
    train_frac: float = typer.Option(0.7, "--train-frac"),
    seed: int = typer.Option(123, "--seed", help="Base seed for runtime holdout generation."),
    k_runtime_holdouts: int = typer.Option(3, "--k-runtime-holdouts"),
    num_worlds_runtime: int = typer.Option(10000, "--num-worlds-runtime"),
    threshold: float | None = typer.Option(None, "--threshold", help="Optional fixed threshold T for P(max > T)."),
    threshold_quantile: float = typer.Option(
        0.95,
        "--threshold-quantile",
        help="If --threshold is unset, derive T as this quantile of base-train max scores.",
    ),
    profiles_path: Path | None = typer.Option(
        None, "--profiles-path", help="Override sim_v2 profiles JSON (default: config/sim_v2_profiles.json)."
    ),
    sim_profile: str | None = typer.Option(
        None,
        "--sim-profile",
        help="Override sim_v2 profile for runtime holdouts (default: inferred from base sim_manifest.json).",
    ),
    data_root: Path | None = typer.Option(None, "--data-root", help="Data root for sim_v2 generator inputs."),
    base_split_seed: int | None = typer.Option(
        None,
        "--base-split-seed",
        help="Override seed for base train/holdout split (default: base sim_manifest.seed if present, else --seed).",
    ),
    perturb: bool = typer.Option(False, "--perturb/--no-perturb"),
    perturb_cfg: Path | None = typer.Option(None, "--perturb-cfg", help="JSON file with perturbation multipliers."),
    persist_runtime_worlds: bool = typer.Option(
        False,
        "--persist-runtime-worlds/--no-persist-runtime-worlds",
        help="Persist generated runtime holdout worlds under the eval run directory (default: no).",
    ),
    output_root: Path | None = typer.Option(
        None, "--output-root", help="Override eval report output directory (default under data_root/artifacts/lineups_eval/...)."
    ),
) -> None:
    worlds_matrix_path = _resolve_worlds_matrix_path(base_worlds_path)
    base_run_dir = worlds_matrix_path.parent

    base_manifest = _read_json(base_run_dir / "sim_manifest.json") or {}
    base_profile_inferred = base_manifest.get("profile") or base_manifest.get("sim_profile")
    profile_name = str(sim_profile or base_profile_inferred or "baseline")

    profiles_path_eff = profiles_path or DEFAULT_PROFILES_PATH
    profile_cfg = load_sim_v2_profile(profile=profile_name, profiles_path=profiles_path_eff)
    profiles_hash = stable_hash_bytes(Path(profiles_path_eff).read_bytes()) if Path(profiles_path_eff).exists() else None

    git = _git_info()
    base_git_commit = base_manifest.get("git_commit")
    git_mismatch: dict[str, Any] | None = None
    if base_git_commit and git.get("git_sha"):
        base_git_s = str(base_git_commit).strip()
        cur_git = str(git["git_sha"]).strip()
        if base_git_s and cur_git and not cur_git.startswith(base_git_s):
            git_mismatch = {"base_git_commit": base_git_s, "current_git_sha": cur_git}

    run_ts = _utc_ts()
    root = output_root or (get_data_root() / "artifacts" / "lineups_eval" / f"game_date={date}" / f"run={run_ts}")
    root.mkdir(parents=True, exist_ok=True)

    parsed = parse_lineups_csv(lineups_path=lineups_path, site=site)
    diversity = compute_diversity_diagnostics(parsed.lineups)

    base_worlds_matrix, base_player_index, _base_player_ids = load_worlds_matrix_parquet(
        worlds_matrix_path=worlds_matrix_path
    )
    missing_in_base = _collect_missing_players(parsed.lineups, base_player_index)

    split_seed = int(
        base_split_seed
        if base_split_seed is not None
        else (base_manifest.get("seed") if base_manifest.get("seed") is not None else seed)
    )
    train_idx, holdout_idx, split_meta = compute_train_holdout_indices(
        n_worlds=int(base_worlds_matrix.shape[0]),
        train_frac=float(train_frac),
        seed=split_seed,
    )

    # === Selection-visible inputs (base-train only) ===
    train_worlds = base_worlds_matrix[train_idx, :]
    train_scores = score_portfolio_on_worlds(
        lineups=parsed.lineups, worlds_matrix=train_worlds, player_index=base_player_index
    )
    base_train_summary = _attach_lineup_ids(summarize_lineup_scores(train_scores), parsed.lineup_ids)

    threshold_info: dict[str, Any]
    if threshold is not None:
        threshold_info = {"threshold": float(threshold), "source": "user"}
    else:
        threshold_info = compute_threshold_from_train_max(train_scores=train_scores, quantile=float(threshold_quantile))
    base_train_threshold = add_threshold_metric(scores=train_scores, threshold=float(threshold_info["threshold"]))
    base_train_summary["threshold_metric"] = base_train_threshold

    # === Evaluation inputs (base-holdout + runtime holdouts) ===
    holdout_worlds = base_worlds_matrix[holdout_idx, :]
    holdout_scores = score_portfolio_on_worlds(
        lineups=parsed.lineups, worlds_matrix=holdout_worlds, player_index=base_player_index
    )
    base_holdout_summary = _attach_lineup_ids(summarize_lineup_scores(holdout_scores), parsed.lineup_ids)
    base_holdout_summary["threshold_metric"] = add_threshold_metric(
        scores=holdout_scores, threshold=float(threshold_info["threshold"])
    )

    runtime_worlds_dir = root / "runtime_worlds"
    runtime_holdouts: list[dict[str, Any]] = []
    runtime_agg_rows: list[dict[str, Any]] = []
    runtime_manifest: list[dict[str, Any]] = []

    perturbation: WorldsPerturbation | None = None
    if perturb:
        if perturb_cfg is None:
            raise typer.BadParameter("--perturb requires --perturb-cfg")
        perturbation = _load_perturbation(perturb_cfg)

    for i in range(int(max(0, k_runtime_holdouts))):
        seed_i = int(seed) + 1000 * int(i)
        run_id = f"runtime_holdout_{i}"

        output_root_i = runtime_worlds_dir if persist_runtime_worlds else None
        res: RuntimeWorldsResult = generate_worlds_matrix_sim_v2(
            game_date=date,
            num_worlds=int(num_worlds_runtime),
            seed=seed_i,
            profile_name=profile_name,
            profiles_path=profiles_path_eff,
            data_root=data_root,
            perturbation=None,
            output_root=output_root_i,
            run_id=run_id,
            persist_outputs=bool(persist_runtime_worlds),
        )
        scores_i = score_portfolio_on_worlds(
            lineups=parsed.lineups, worlds_matrix=res.worlds_matrix, player_index=res.player_index
        )
        summary_i = _attach_lineup_ids(summarize_lineup_scores(scores_i), parsed.lineup_ids)
        thresh_i = add_threshold_metric(scores=scores_i, threshold=float(threshold_info["threshold"]))
        summary_i["threshold_metric"] = thresh_i
        missing_i = _collect_missing_players(parsed.lineups, res.player_index)

        worlds_path = None
        if persist_runtime_worlds and res.provenance.run_dir:
            candidate = Path(res.provenance.run_dir) / "worlds_matrix.parquet"
            if candidate.exists():
                worlds_path = str(candidate)

        runtime_holdout_payload: dict[str, Any] = {
            "holdout_id": int(i),
            "seed": seed_i,
            "num_worlds": int(num_worlds_runtime),
            "cfg_hash": res.provenance.cfg_hash,
            "missing_players": missing_i,
            "metrics": summary_i,
        }

        runtime_holdouts.append(runtime_holdout_payload)

        runtime_manifest.append(
            {
                "holdout_id": int(i),
                "seed": seed_i,
                "num_worlds": int(num_worlds_runtime),
                "cfg_hash": res.provenance.cfg_hash,
                "git_sha": git.get("git_sha"),
                "perturbation": None,
                "output_path": worlds_path,
            }
        )

        perturbed_payload: dict[str, Any] | None = None
        if perturbation is not None:
            run_id_p = f"{run_id}_perturbed"
            res_p = generate_worlds_matrix_sim_v2(
                game_date=date,
                num_worlds=int(num_worlds_runtime),
                seed=seed_i,
                profile_name=profile_name,
                profiles_path=profiles_path_eff,
                data_root=data_root,
                perturbation=perturbation,
                output_root=output_root_i,
                run_id=run_id_p,
                persist_outputs=bool(persist_runtime_worlds),
            )
            scores_p = score_portfolio_on_worlds(
                lineups=parsed.lineups, worlds_matrix=res_p.worlds_matrix, player_index=res_p.player_index
            )
            summary_p = _attach_lineup_ids(summarize_lineup_scores(scores_p), parsed.lineup_ids)
            thresh_p = add_threshold_metric(scores=scores_p, threshold=float(threshold_info["threshold"]))
            summary_p["threshold_metric"] = thresh_p

            worlds_path_p = None
            if persist_runtime_worlds and res_p.provenance.run_dir:
                candidate_p = Path(res_p.provenance.run_dir) / "worlds_matrix.parquet"
                if candidate_p.exists():
                    worlds_path_p = str(candidate_p)

            perturbed_payload = {
                "seed": seed_i,
                "num_worlds": int(num_worlds_runtime),
                "cfg_hash": res_p.provenance.cfg_hash,
                "perturbation": asdict(perturbation),
                "metrics": summary_p,
                "deltas": {
                    "portfolio_max_mean": float(summary_p["portfolio_max"]["mean"] - summary_i["portfolio_max"]["mean"]),
                    "portfolio_max_p95": float(summary_p["portfolio_max"]["p95"] - summary_i["portfolio_max"]["p95"]),
                    "p_max_gt_threshold": float(
                        summary_p["threshold_metric"]["p_max_gt_threshold"] - summary_i["threshold_metric"]["p_max_gt_threshold"]
                    ),
                },
            }
            runtime_holdout_payload["perturbed"] = perturbed_payload

            runtime_manifest.append(
                {
                    "holdout_id": int(i),
                    "seed": seed_i,
                    "num_worlds": int(num_worlds_runtime),
                    "cfg_hash": res_p.provenance.cfg_hash,
                    "git_sha": git.get("git_sha"),
                    "perturbation": asdict(perturbation),
                    "output_path": worlds_path_p,
                }
            )

        runtime_agg_rows.append(
            {
                "set": f"runtime_{i}",
                "n_worlds": int(num_worlds_runtime),
                "max_mean": float(summary_i["portfolio_max"]["mean"]),
                "max_p90": summary_i["portfolio_max"].get("p90"),
                "max_p95": summary_i["portfolio_max"].get("p95"),
                "max_p99": summary_i["portfolio_max"].get("p99"),
                "p_max_gt_threshold": float(thresh_i["p_max_gt_threshold"]),
            }
        )

    # Aggregate stability across runtime holdouts (portfolio-level metrics).
    runtime_agg: dict[str, Any] | None = None
    if runtime_holdouts:
        means = np.array([h["metrics"]["portfolio_max"]["mean"] for h in runtime_holdouts], dtype=np.float64)
        p90s = np.array([h["metrics"]["portfolio_max"]["p90"] for h in runtime_holdouts], dtype=np.float64)
        p95s = np.array([h["metrics"]["portfolio_max"]["p95"] for h in runtime_holdouts], dtype=np.float64)
        p99s = np.array([h["metrics"]["portfolio_max"]["p99"] for h in runtime_holdouts], dtype=np.float64)
        p_gt = np.array(
            [h["metrics"]["threshold_metric"]["p_max_gt_threshold"] for h in runtime_holdouts], dtype=np.float64
        )
        runtime_agg = {
            "k": int(len(runtime_holdouts)),
            "portfolio_max": {
                "mean": {"mean": float(means.mean()), "std": float(means.std(ddof=0))},
                "p90": {"mean": float(p90s.mean()), "std": float(p90s.std(ddof=0))},
                "p95": {"mean": float(p95s.mean()), "std": float(p95s.std(ddof=0))},
                "p99": {"mean": float(p99s.mean()), "std": float(p99s.std(ddof=0))},
            },
            "p_max_gt_threshold": {"mean": float(p_gt.mean()), "std": float(p_gt.std(ddof=0))},
        }

    # Build markdown summary table.
    summary_rows = [
        {
            "set": "base_train",
            "n_worlds": int(train_idx.size),
            "max_mean": float(base_train_summary["portfolio_max"]["mean"]),
            "max_p90": base_train_summary["portfolio_max"].get("p90"),
            "max_p95": base_train_summary["portfolio_max"].get("p95"),
            "max_p99": base_train_summary["portfolio_max"].get("p99"),
            "p_max_gt_threshold": float(base_train_summary["threshold_metric"]["p_max_gt_threshold"]),
        },
        {
            "set": "base_holdout",
            "n_worlds": int(holdout_idx.size),
            "max_mean": float(base_holdout_summary["portfolio_max"]["mean"]),
            "max_p90": base_holdout_summary["portfolio_max"].get("p90"),
            "max_p95": base_holdout_summary["portfolio_max"].get("p95"),
            "max_p99": base_holdout_summary["portfolio_max"].get("p99"),
            "p_max_gt_threshold": float(base_holdout_summary["threshold_metric"]["p_max_gt_threshold"]),
        },
    ] + runtime_agg_rows

    md_lines = []
    md_lines.append(f"# Portfolio Holdout Evaluation ({date}, site={site})")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append(_render_summary_table(summary_rows))
    md_lines.append("")
    if runtime_agg is not None:
        md_lines.append("## Runtime Stability (K holdouts)")
        md_lines.append(
            f"- max_mean: {_format_float(runtime_agg['portfolio_max']['mean']['mean'])} ± {_format_float(runtime_agg['portfolio_max']['mean']['std'])}"
        )
        md_lines.append(
            f"- max_p95: {_format_float(runtime_agg['portfolio_max']['p95']['mean'])} ± {_format_float(runtime_agg['portfolio_max']['p95']['std'])}"
        )
        md_lines.append(
            f"- P(max>T): {_format_float(runtime_agg['p_max_gt_threshold']['mean'])} ± {_format_float(runtime_agg['p_max_gt_threshold']['std'])}"
        )
        md_lines.append("")

    md_lines.append("## Inputs / Provenance")
    md_lines.append(f"- lineups_path: `{lineups_path}`")
    md_lines.append(f"- base_worlds_path: `{worlds_matrix_path}`")
    md_lines.append(f"- base_run_dir: `{base_run_dir}`")
    md_lines.append(f"- draft_group_id: {parsed.draft_group_id}")
    md_lines.append(f"- sim_profile (runtime): `{profile_name}`")
    md_lines.append(f"- profiles_path: `{profiles_path_eff}`")
    md_lines.append(f"- profiles_sha256: `{profiles_hash}`")
    md_lines.append(f"- git_sha: `{git.get('git_sha')}` dirty={git.get('git_dirty')}")
    md_lines.append(f"- base_sim_git_commit: `{base_git_commit}`")
    if git_mismatch is not None:
        md_lines.append(f"- WARNING git mismatch vs base worlds: `{json.dumps(git_mismatch, sort_keys=True)}`")
    md_lines.append(f"- train_frac: {train_frac} (seed={split_seed}) n_train={int(train_idx.size)} n_holdout={int(holdout_idx.size)}")
    md_lines.append(f"- threshold: {_format_float(float(threshold_info['threshold']))} (source={threshold_info['source']})")
    md_lines.append(f"- runtime_seed_base: {seed} k={k_runtime_holdouts} num_worlds_runtime={num_worlds_runtime}")
    if perturbation is not None:
        md_lines.append(f"- perturbation: `{json.dumps(asdict(perturbation), sort_keys=True)}`")
    if missing_in_base:
        md_lines.append(f"- WARNING missing players in base worlds: {len(missing_in_base)}")
    md_lines.append("")

    md_lines.append("## Portfolio Diversity")
    md_lines.append(f"- n_lineups: {diversity.get('n_lineups')}")
    md_lines.append(f"- unique_players: {diversity.get('unique_players')}")
    md_lines.append(f"- exposure_hhi: {_format_float(diversity.get('exposure_hhi'))}")
    md_lines.append(f"- pairwise_overlap_mean: {_format_float(diversity.get('pairwise_overlap_mean'))}")
    md_lines.append(f"- pairwise_overlap_p95: {_format_float(diversity.get('pairwise_overlap_p95'))}")
    md_lines.append(f"- pairwise_overlap_max: {diversity.get('pairwise_overlap_max')}")
    md_lines.append(f"- duplicate_lineups: {diversity.get('duplicate_lineups')}")
    md_lines.append("")

    report_md_path = root / "eval_report.md"
    report_json_path = root / "eval_report.json"
    holdout_manifest_path = root / "runtime_holdout_manifest.json"

    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _write_json(holdout_manifest_path, runtime_manifest)

    eval_report = {
        "run": {"ts_utc": run_ts, "output_dir": str(root)},
        "provenance": {
            **git,
            "profiles_path": str(profiles_path_eff),
            "profiles_sha256": profiles_hash,
        },
        "selection_inputs": {
            "date": date,
            "site": site,
            "draft_group_id": parsed.draft_group_id,
            "lineups_path": str(lineups_path),
            "base_worlds_path": str(worlds_matrix_path),
            "base_sim_manifest": base_manifest or None,
            "base_split": split_meta,
            "threshold": threshold_info,
        },
        "eval_inputs": {
            "base_holdout": {"n_worlds": int(holdout_idx.size), "holdout_idx_sha256": split_meta["holdout_idx_sha256"]},
            "runtime_holdouts": {
                "k": int(k_runtime_holdouts),
                "num_worlds": int(num_worlds_runtime),
                "seed_base": int(seed),
                "seed_stride": 1000,
                "profile_name": profile_name,
                "profile_cfg_sha256": stable_hash_json(asdict(profile_cfg)),
                "persist_runtime_worlds": bool(persist_runtime_worlds),
                "perturbation": asdict(perturbation) if perturbation is not None else None,
            },
        },
        "warnings": {"missing_players_in_base_worlds": missing_in_base},
        "mismatch": {"git": git_mismatch},
        "portfolio": {"diversity": diversity},
        "metrics": {
            "base_train": base_train_summary,
            "base_holdout": base_holdout_summary,
            "runtime_holdouts": runtime_holdouts,
            "runtime_aggregate": runtime_agg,
        },
    }
    _write_json(report_json_path, eval_report)

    typer.echo(f"Wrote report: {report_md_path}")
    typer.echo(f"Wrote report JSON: {report_json_path}")
    typer.echo(f"Wrote runtime manifest: {holdout_manifest_path}")
    typer.echo("")
    typer.echo(_render_summary_table(summary_rows))


if __name__ == "__main__":
    app()
