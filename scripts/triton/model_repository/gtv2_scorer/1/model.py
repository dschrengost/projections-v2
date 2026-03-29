from __future__ import annotations

import json
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.data import DataLoader


def _param(model_config: dict[str, Any], key: str, default: str) -> str:
    params = model_config.get("parameters")
    if not isinstance(params, dict):
        return str(default)
    item = params.get(key)
    if not isinstance(item, dict):
        return str(default)
    value = item.get("string_value")
    if value is None:
        return str(default)
    return str(value)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _parse_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path, *, row_group_size: int = 500_000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp")
    df.to_parquet(tmp, index=False, compression="zstd", row_group_size=row_group_size)
    os.replace(tmp, out_path)


def _json_response(payload: dict[str, Any]) -> pb_utils.InferenceResponse:
    tensor = pb_utils.Tensor(
        "response_json",
        np.asarray([json.dumps(payload, sort_keys=True)], dtype=object),
    )
    return pb_utils.InferenceResponse(output_tensors=[tensor])


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        self._model_config = json.loads(args["model_config"])
        project_root = _param(
            self._model_config,
            "project_root",
            "/home/daniel/prod/projections-v2",
        )
        if project_root and project_root not in os.sys.path:
            os.sys.path.insert(0, project_root)

        from projections.pipeline.gtv2_inference_runtime import (
            build_gtv2_inference_examples,
            load_gtv2_model,
            resolve_torch_device,
            score_gtv2_features_df,
            set_inference_seed,
        )
        from projections.rotation.game_transformer_v2 import (
            GameLevelDataset,
            collate_game_level_examples,
        )
        from projections.rotation.sample_worlds_v2 import (
            MakeModelConfig,
            MinutesUncertaintyConfig,
            sample_worlds_for_batch,
        )

        self._build_gtv2_inference_examples = build_gtv2_inference_examples
        self._load_gtv2_model = load_gtv2_model
        self._resolve_torch_device = resolve_torch_device
        self._score_gtv2_features_df = score_gtv2_features_df
        self._set_inference_seed = set_inference_seed
        self._GameLevelDataset = GameLevelDataset
        self._collate_game_level_examples = collate_game_level_examples
        self._MakeModelConfig = MakeModelConfig
        self._MinutesUncertaintyConfig = MinutesUncertaintyConfig
        self._sample_worlds_for_batch = sample_worlds_for_batch

        self._default_bundle_dir = Path(
            _param(
                self._model_config,
                "bundle_dir",
                "/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current",
            )
        )
        self._default_device = _param(self._model_config, "device", "cuda:0")
        self._default_num_worlds = _parse_int(
            _param(self._model_config, "num_worlds", "25000"),
            default=25000,
        )
        self._default_world_chunk_size = _parse_int(
            _param(self._model_config, "world_chunk_size", "5000"),
            default=5000,
        )

        self._cached_bundle_dir: Path | None = None
        self._cached_device_key: str | None = None
        self._cached_flow_scale_clip_override: float | None = None
        self._cached_config: Any = None
        self._cached_model: torch.nn.Module | None = None
        self._cached_device: torch.device | None = None

    def _ensure_model(
        self,
        *,
        bundle_dir: Path,
        requested_device: str | None,
        flow_scale_clip_override: float | None,
    ) -> tuple[Any, torch.nn.Module, torch.device]:
        device_key = str(requested_device or self._default_device or "").strip()
        device = self._resolve_torch_device(device_key or None)
        if (
            self._cached_model is not None
            and self._cached_config is not None
            and self._cached_bundle_dir == bundle_dir
            and self._cached_device_key == str(device)
            and self._cached_flow_scale_clip_override == flow_scale_clip_override
        ):
            return self._cached_config, self._cached_model, self._cached_device  # type: ignore[return-value]

        config, model = self._load_gtv2_model(
            bundle_dir=bundle_dir,
            device=device,
            flow_scale_clip_override=flow_scale_clip_override,
        )
        self._cached_bundle_dir = bundle_dir
        self._cached_device_key = str(device)
        self._cached_flow_scale_clip_override = flow_scale_clip_override
        self._cached_config = config
        self._cached_model = model
        self._cached_device = device
        return config, model, device

    def _handle_score(self, payload: dict[str, Any]) -> dict[str, Any]:
        game_date = str(payload["game_date"])
        features_path = Path(str(payload["features_path"]))
        out_path = Path(str(payload["out_path"]))
        summary_path_raw = payload.get("summary_path")
        summary_path = Path(str(summary_path_raw)) if summary_path_raw else None

        bundle_dir = Path(str(payload.get("bundle_dir") or self._default_bundle_dir))
        random_seed = _parse_int(payload.get("random_seed"), default=42)
        batch_size = _parse_int(payload.get("batch_size"), default=4)
        requested_device = str(payload.get("device") or self._default_device).strip()
        flow_scale_clip_override = payload.get("flow_scale_clip_override")
        clip = (
            _parse_float(flow_scale_clip_override, default=0.0)
            if flow_scale_clip_override is not None
            else None
        )

        self._set_inference_seed(random_seed)
        config, model, device = self._ensure_model(
            bundle_dir=bundle_dir,
            requested_device=requested_device,
            flow_scale_clip_override=clip,
        )

        features_df = pd.read_parquet(features_path)
        scores = self._score_gtv2_features_df(
            features_df=features_df,
            game_date=game_date,
            config=config,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        _atomic_write_parquet(scores, out_path)

        summary = {
            "rows": int(len(scores)),
            "games": int(scores["game_id"].nunique()),
            "players": int(scores["player_id"].nunique()),
            "bundle_dir": str(bundle_dir),
            "device": str(device),
            "action": "score",
        }
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return {
            "ok": True,
            "action": "score",
            "rows": int(len(scores)),
            "device": str(device),
            "bundle_dir": str(bundle_dir),
        }

    def _handle_worlds(self, payload: dict[str, Any]) -> dict[str, Any]:
        game_date = str(payload["game_date"])
        features_path = Path(str(payload["features_path"]))
        out_path = Path(str(payload["out_path"]))
        summary_path_raw = payload.get("summary_path")
        summary_path = Path(str(summary_path_raw)) if summary_path_raw else None

        bundle_dir = Path(str(payload.get("bundle_dir") or self._default_bundle_dir))
        random_seed = _parse_int(payload.get("random_seed"), default=42)
        requested_device = str(payload.get("device") or self._default_device).strip()
        num_worlds = _parse_int(
            payload.get("num_worlds"),
            default=self._default_num_worlds,
        )
        chunk_size = _parse_int(
            payload.get("world_chunk_size"),
            default=self._default_world_chunk_size,
        )
        active_temperature = _parse_float(
            payload.get("active_temperature"),
            default=1.0,
        )
        strict_world_contracts = _parse_bool(
            payload.get("strict_world_contracts"),
            default=True,
        )
        flow_scale_clip_override = payload.get("flow_scale_clip_override")
        clip = (
            _parse_float(flow_scale_clip_override, default=0.0)
            if flow_scale_clip_override is not None
            else None
        )
        make_model_mode = str(payload.get("make_model_mode") or "beta_binomial_all")
        make_model_use_learned_efficiency = _parse_bool(
            payload.get("make_model_use_learned_efficiency"),
            default=True,
        )
        minutes_uncertainty_config = self._MinutesUncertaintyConfig(
            enabled=_parse_bool(payload.get("minutes_uncertainty_enabled"), default=False),
            mode=str(payload.get("minutes_uncertainty_mode") or "gaussian"),
            gaussian_scale=_parse_float(
                payload.get("minutes_uncertainty_gaussian_scale"),
                default=1.0,
            ),
            min_sigma=_parse_float(
                payload.get("minutes_uncertainty_min_sigma"),
                default=0.75,
            ),
            max_sigma=_parse_float(
                payload.get("minutes_uncertainty_max_sigma"),
                default=6.0,
            ),
            fallback_sigma=_parse_float(
                payload.get("minutes_uncertainty_fallback_sigma"),
                default=1.5,
            ),
            use_hurdle_sigma=_parse_bool(
                payload.get("minutes_uncertainty_use_hurdle_sigma"),
                default=True,
            ),
            use_prior_std=_parse_bool(
                payload.get("minutes_uncertainty_use_prior_std"),
                default=True,
            ),
            preserve_top_k_per_team=_parse_int(
                payload.get("minutes_uncertainty_preserve_top_k_per_team"),
                default=3,
            ),
            full_sigma_at_minutes_or_below=_parse_float(
                payload.get("minutes_uncertainty_full_sigma_at_minutes_or_below"),
                default=24.0,
            ),
            zero_sigma_at_minutes_or_above=_parse_float(
                payload.get("minutes_uncertainty_zero_sigma_at_minutes_or_above"),
                default=32.0,
            ),
            apply_minutes_taper=_parse_bool(
                payload.get("minutes_uncertainty_apply_minutes_taper"),
                default=True,
            ),
            dirichlet_base_concentration=_parse_float(
                payload.get("minutes_uncertainty_dirichlet_base_concentration"),
                default=24.0,
            ),
            empirical_minutes_bin_edges=tuple(
                float(x) for x in list(payload.get("minutes_uncertainty_empirical_bin_edges") or [])
            ),
            empirical_sigma_by_bin=tuple(
                float(x) for x in list(payload.get("minutes_uncertainty_empirical_sigma_by_bin") or [])
            ),
            empirical_blend_alpha=_parse_float(
                payload.get("minutes_uncertainty_empirical_blend_alpha"),
                default=1.0,
            ),
        )

        self._set_inference_seed(random_seed)
        config, model, device = self._ensure_model(
            bundle_dir=bundle_dir,
            requested_device=requested_device,
            flow_scale_clip_override=clip,
        )

        features_df = pd.read_parquet(features_path)
        examples = self._build_gtv2_inference_examples(
            features_df=features_df,
            game_date=game_date,
            config=config,
        )
        loader = DataLoader(
            self._GameLevelDataset(examples),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_game_level_examples,
        )

        make_model_cfg = self._MakeModelConfig(
            mode=make_model_mode,
            use_learned_efficiency=bool(make_model_use_learned_efficiency),
        )

        world_frames: list[pd.DataFrame] = []
        contract_counter: Counter[str] = Counter()
        for batch in loader:
            df_batch, checks = self._sample_worlds_for_batch(
                model,
                batch,
                device=device,
                num_worlds=int(num_worlds),
                chunk_size=max(1, int(chunk_size)),
                active_temperature=float(active_temperature),
                strict_contracts=bool(strict_world_contracts),
                make_model_config=make_model_cfg,
                minutes_uncertainty_config=minutes_uncertainty_config,
            )
            world_frames.append(df_batch)
            contract_counter.update(checks)

        worlds_df = (
            pd.concat(world_frames, ignore_index=True)
            if world_frames
            else pd.DataFrame()
        )
        if worlds_df.empty:
            raise RuntimeError("GTV2 worlds generation produced zero rows")
        _atomic_write_parquet(worlds_df, out_path)

        contract_checks: dict[str, Any] = {
            str(key): (
                float(value) if isinstance(value, np.floating) else int(value)
                if isinstance(value, np.integer)
                else value
            )
            for key, value in dict(contract_counter).items()
        }
        summary = {
            "action": "worlds",
            "rows": int(len(worlds_df)),
            "bundle_dir": str(bundle_dir),
            "device": str(device),
            "contract_checks": contract_checks,
        }
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return {
            "ok": True,
            "action": "worlds",
            "world_rows": int(len(worlds_df)),
            "contract_checks": contract_checks,
            "device": str(device),
            "bundle_dir": str(bundle_dir),
        }

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "request_json")
                if input_tensor is None:
                    raise RuntimeError("request_json input missing")
                data = input_tensor.as_numpy()
                if data.size == 0:
                    raise RuntimeError("request_json input is empty")
                payload_raw = data.reshape(-1)[0]
                if isinstance(payload_raw, bytes):
                    payload_text = payload_raw.decode("utf-8")
                else:
                    payload_text = str(payload_raw)
                payload = json.loads(payload_text)
                if not isinstance(payload, dict):
                    raise RuntimeError("request_json must decode to object")

                action = str(payload.get("action") or "").strip().lower()
                if action == "score":
                    result = self._handle_score(payload)
                elif action == "worlds":
                    result = self._handle_worlds(payload)
                else:
                    raise RuntimeError(f"unsupported action: {action!r}")
                responses.append(_json_response(result))
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    _json_response(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=20),
                        }
                    )
                )
        return responses

    def finalize(self) -> None:
        self._cached_model = None
        self._cached_config = None
        self._cached_bundle_dir = None
        self._cached_device = None
        self._cached_device_key = None
