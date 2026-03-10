"""Create/update a Triton model repository entry for GTV2 scorer."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _render_config(
    *,
    project_root: Path,
    bundle_dir: Path,
    device: str,
    num_worlds: int,
    world_chunk_size: int,
) -> str:
    return f"""name: "gtv2_scorer"
backend: "python"

max_batch_size: 0

input [
  {{
    name: "request_json"
    data_type: TYPE_STRING
    dims: [1]
  }}
]

output [
  {{
    name: "response_json"
    data_type: TYPE_STRING
    dims: [1]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }}
]

parameters [
  {{
    key: "project_root"
    value: {{ string_value: "{project_root}" }}
  }},
  {{
    key: "bundle_dir"
    value: {{ string_value: "{bundle_dir}" }}
  }},
  {{
    key: "device"
    value: {{ string_value: "{device}" }}
  }},
  {{
    key: "num_worlds"
    value: {{ string_value: "{int(num_worlds)}" }}
  }},
  {{
    key: "world_chunk_size"
    value: {{ string_value: "{int(world_chunk_size)}" }}
  }}
]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path("/home/daniel/projections-data/triton_models"),
        help="Target Triton model repository root.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path visible to Triton runtime.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current"),
        help="Promoted GTV2 bundle directory.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-worlds", type=int, default=25000)
    parser.add_argument("--world-chunk-size", type=int, default=5000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model.py/config.pbtxt if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    model_repo = args.model_repo.expanduser().resolve()
    bundle_dir = args.bundle_dir.expanduser().resolve()

    template_root = project_root / "scripts" / "triton" / "model_repository" / "gtv2_scorer"
    template_model = template_root / "1" / "model.py"
    if not template_model.exists():
        raise FileNotFoundError(f"template model.py not found: {template_model}")

    target_model_root = model_repo / "gtv2_scorer"
    target_model_version = target_model_root / "1"
    target_model = target_model_version / "model.py"
    target_config = target_model_root / "config.pbtxt"

    target_model_version.mkdir(parents=True, exist_ok=True)

    if target_model.exists() and not args.force:
        raise RuntimeError(f"{target_model} exists; rerun with --force to overwrite")
    if target_config.exists() and not args.force:
        raise RuntimeError(f"{target_config} exists; rerun with --force to overwrite")

    shutil.copy2(template_model, target_model)
    target_config.write_text(
        _render_config(
            project_root=project_root,
            bundle_dir=bundle_dir,
            device=str(args.device),
            num_worlds=int(args.num_worlds),
            world_chunk_size=int(args.world_chunk_size),
        ),
        encoding="utf-8",
    )

    print(f"Model repository updated: {target_model_root}")
    print(f"- model.py: {target_model}")
    print(f"- config.pbtxt: {target_config}")


if __name__ == "__main__":
    main()
