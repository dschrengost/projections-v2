"""Runtime (non-Prefect) worlds generation helpers for evaluation-only workflows.

This module intentionally keeps a narrow surface area:
- Default backend calls the canonical sim_v2 generator (same code path as production),
  but can write outputs to a caller-controlled directory to avoid leaking into selection.
- A small synthetic generator is provided for unit tests and lightweight debugging.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from projections.paths import get_data_root
from projections.sim_v2.config import DEFAULT_PROFILES_PATH, SimV2Profile, load_sim_v2_profile

EVAL_ONLY_MARKER_FILENAME = "EVAL_ONLY_DO_NOT_USE_FOR_SELECTION.txt"


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_config_hash(payload: Any) -> str:
    """Return a stable sha256 hash for a JSON-serializable config payload."""
    encoded = _stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _looks_like_canonical_sim_artifacts(path: Path) -> bool:
    """Best-effort guard to avoid writing evaluation worlds into selection-visible locations."""
    lowered = [part.lower() for part in path.resolve().parts]
    return "artifacts" in lowered and "sim_v2" in lowered and "worlds_fpts_v2" in lowered


@dataclass(frozen=True)
class WorldsPerturbation:
    """Multiplicative perturbations applied to the base sim_v2 profile knobs."""

    minutes_sigma_min_mult: float = 1.0
    team_sigma_scale_mult: float = 1.0
    player_sigma_scale_mult: float = 1.0
    team_factor_sigma_mult: float = 1.0
    team_factor_gamma_mult: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "minutes_sigma_min_mult": float(self.minutes_sigma_min_mult),
            "team_sigma_scale_mult": float(self.team_sigma_scale_mult),
            "player_sigma_scale_mult": float(self.player_sigma_scale_mult),
            "team_factor_sigma_mult": float(self.team_factor_sigma_mult),
            "team_factor_gamma_mult": float(self.team_factor_gamma_mult),
        }


def apply_perturbation_overrides(
    profile: SimV2Profile, perturb: WorldsPerturbation | None
) -> dict[str, float]:
    """Return CLI override kwargs for the sim_v2 generator, derived from a perturbation."""
    if perturb is None:
        return {}

    overrides: dict[str, float] = {}
    overrides["minutes_sigma_min"] = float(profile.minutes_sigma_min) * float(perturb.minutes_sigma_min_mult)
    overrides["team_sigma_scale"] = float(getattr(profile, "team_sigma_scale", 1.0)) * float(perturb.team_sigma_scale_mult)
    overrides["player_sigma_scale"] = float(getattr(profile, "player_sigma_scale", 1.0)) * float(perturb.player_sigma_scale_mult)
    overrides["team_factor_sigma"] = float(profile.team_factor_sigma) * float(perturb.team_factor_sigma_mult)
    overrides["team_factor_gamma"] = float(profile.team_factor_gamma) * float(perturb.team_factor_gamma_mult)
    return overrides


@dataclass(frozen=True)
class RuntimeWorldsProvenance:
    """Minimal provenance for a runtime worlds generation call."""

    game_date: str
    profile_name: str
    profiles_path: str
    seed: int
    num_worlds: int
    cfg_hash: str
    perturbation: dict[str, float] | None

    backend: str = "sim_v2"
    output_dir: str | None = None
    run_dir: str | None = None
    sim_manifest: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "game_date": self.game_date,
            "profile_name": self.profile_name,
            "profiles_path": self.profiles_path,
            "seed": int(self.seed),
            "num_worlds": int(self.num_worlds),
            "cfg_hash": self.cfg_hash,
            "perturbation": self.perturbation,
            "output_dir": self.output_dir,
            "run_dir": self.run_dir,
            "sim_manifest": dict(self.sim_manifest) if self.sim_manifest is not None else None,
        }


@dataclass(frozen=True)
class RuntimeWorldsResult:
    worlds_matrix: np.ndarray
    player_ids: list[str]
    player_index: dict[str, int]
    provenance: RuntimeWorldsProvenance


def generate_worlds_matrix_sim_v2(
    *,
    game_date: str,
    num_worlds: int,
    seed: int,
    profile_name: str,
    profiles_path: Path | None = None,
    data_root: Path | None = None,
    perturbation: WorldsPerturbation | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    persist_outputs: bool = False,
    allow_unsafe_output_root: bool = False,
) -> RuntimeWorldsResult:
    """Generate a consolidated worlds matrix via the canonical sim_v2 generator.

    By default, outputs are written to a temporary directory and deleted after load.
    Callers can persist outputs by setting persist_outputs=True and providing output_root.
    """
    root = Path(data_root) if data_root is not None else get_data_root()
    profiles_path_eff = Path(profiles_path) if profiles_path is not None else DEFAULT_PROFILES_PATH
    profile = load_sim_v2_profile(profile=profile_name, profiles_path=profiles_path_eff)
    overrides = apply_perturbation_overrides(profile, perturbation)
    perturb_dict = perturbation.to_dict() if perturbation is not None else None
    cfg_hash = stable_config_hash({"profile": asdict(profile), "overrides": overrides})

    # Resolve a safe output_root. Never default to the canonical sim_v2 artifact root.
    tmp_dir_cm = None
    if output_root is None:
        tmp_dir_cm = tempfile.TemporaryDirectory(prefix="sim_v2_runtime_worlds_")
        output_root_eff = Path(tmp_dir_cm.name)
    else:
        output_root_eff = Path(output_root)

    if not allow_unsafe_output_root and _looks_like_canonical_sim_artifacts(output_root_eff):
        raise ValueError(
            f"Refusing to write runtime evaluation worlds under selection-visible sim_v2 artifacts: {output_root_eff}"
        )

    # Import here to avoid importing Typer-heavy scripts at module import time.
    from scripts.sim_v2.generate_worlds_fpts_v2 import main as sim_v2_generate_main

    run_id_eff = run_id or f"runtime_seed_{int(seed)}"
    run_dir = output_root_eff / f"game_date={game_date}" / f"run={run_id_eff}"

    try:
        sim_v2_generate_main(
            start_date=game_date,
            end_date=game_date,
            n_worlds=int(num_worlds),
            profile=profile_name,
            data_root=root,
            profiles_path=profiles_path_eff,
            output_root=output_root_eff,
            sim_run_id=run_id_eff,
            use_rates_noise=None,
            rates_noise_split=None,
            seed=int(seed),
            min_play_prob=None,
            use_minutes_noise=None,
            minutes_run_id=None,
            minutes_noise_run_id=None,
            minutes_sigma_min=float(overrides.get("minutes_sigma_min")) if overrides else None,
            team_sigma_scale=float(overrides.get("team_sigma_scale")) if overrides else None,
            player_sigma_scale=float(overrides.get("player_sigma_scale")) if overrides else None,
            rates_run_id=None,
            team_factor_sigma=float(overrides.get("team_factor_sigma")) if overrides else None,
            team_factor_gamma=float(overrides.get("team_factor_gamma")) if overrides else None,
            use_efficiency_scoring=None,
            export_attempt_means=False,
        )

        matrix_path = run_dir / "worlds_matrix.parquet"
        if not matrix_path.exists():
            raise FileNotFoundError(f"sim_v2 generator did not produce {matrix_path}")

        df = pd.read_parquet(matrix_path)
        player_ids = [str(c) for c in df.columns]
        worlds_matrix = df.to_numpy(dtype=np.float64, copy=False)
        player_index = {pid: idx for idx, pid in enumerate(player_ids)}

        sim_manifest_path = run_dir / "sim_manifest.json"
        sim_manifest = None
        if sim_manifest_path.exists():
            try:
                sim_manifest = json.loads(sim_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                sim_manifest = None

        # Add an explicit marker to discourage accidental reuse.
        try:
            (run_dir / EVAL_ONLY_MARKER_FILENAME).write_text(
                "EVAL ONLY: runtime holdout worlds generated for evaluation. Do not use for selection.\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        prov = RuntimeWorldsProvenance(
            game_date=game_date,
            profile_name=profile_name,
            profiles_path=str(profiles_path_eff),
            seed=int(seed),
            num_worlds=int(num_worlds),
            cfg_hash=cfg_hash,
            perturbation=perturb_dict,
            backend="sim_v2",
            output_dir=str(output_root_eff),
            run_dir=str(run_dir),
            sim_manifest=sim_manifest,
        )
        return RuntimeWorldsResult(
            worlds_matrix=worlds_matrix,
            player_ids=player_ids,
            player_index=player_index,
            provenance=prov,
        )
    finally:
        if tmp_dir_cm is not None:
            tmp_dir_cm.cleanup()
        elif not persist_outputs:
            # Caller provided output_root but asked us not to persist; clean up best-effort.
            try:
                shutil.rmtree(run_dir)
            except Exception:
                pass


def generate_synthetic_worlds_matrix(
    *,
    player_ids: list[str],
    means: np.ndarray,
    stdevs: np.ndarray,
    num_worlds: int,
    seed: int,
) -> RuntimeWorldsResult:
    """Generate independent Normal worlds for unit tests / lightweight debug.

    This is not a substitute for sim_v2 correlated worlds; it is intentionally simple.
    """
    if len(player_ids) <= 0:
        raise ValueError("player_ids must be non-empty")
    means = np.asarray(means, dtype=np.float64)
    stdevs = np.asarray(stdevs, dtype=np.float64)
    if means.shape != stdevs.shape or means.shape != (len(player_ids),):
        raise ValueError("means/stdevs must be shape (n_players,) matching player_ids")
    if int(num_worlds) <= 0:
        raise ValueError("num_worlds must be positive")

    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(size=(int(num_worlds), len(player_ids))).astype(np.float64)
    worlds = means[None, :] + z * stdevs[None, :]
    player_index = {str(pid): idx for idx, pid in enumerate(player_ids)}
    cfg_hash = stable_config_hash(
        {
            "backend": "synthetic_normal",
            "player_ids": [str(pid) for pid in player_ids],
            "means_sha256": hashlib.sha256(means.tobytes()).hexdigest(),
            "stdevs_sha256": hashlib.sha256(stdevs.tobytes()).hexdigest(),
        }
    )
    prov = RuntimeWorldsProvenance(
        game_date="synthetic",
        profile_name="synthetic_normal",
        profiles_path="n/a",
        seed=int(seed),
        num_worlds=int(num_worlds),
        cfg_hash=cfg_hash,
        perturbation=None,
        backend="synthetic_normal",
        output_dir=None,
        run_dir=None,
        sim_manifest=None,
    )
    return RuntimeWorldsResult(
        worlds_matrix=worlds,
        player_ids=[str(pid) for pid in player_ids],
        player_index=player_index,
        provenance=prov,
    )
