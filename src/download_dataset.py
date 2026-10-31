"""Download one immutable Roboflow dataset version for local/Colab use."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .pipeline_utils import load_config, split_project_id
except ImportError:  # Support: python src/download_dataset.py
    from pipeline_utils import load_config, split_project_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--target", default="/content/datasets/baseline")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config["dataset"]
    workspace, project_slug = split_project_id(dataset_config["project_id"])

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set; do not place it in source code.")

    target = Path(args.target)
    existing_yaml = target / "data.yaml"
    if existing_yaml.exists() and not args.overwrite:
        print(existing_yaml)
        return

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    dataset = project.version(dataset_config["version"]).download(
        dataset_config["format"],
        location=str(target),
        overwrite=args.overwrite,
    )
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Roboflow download finished without {data_yaml}")
    print(data_yaml)


if __name__ == "__main__":
    main()

