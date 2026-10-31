"""Validate an exported YOLO dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .pipeline_utils import (
        load_config,
        validate_class_names,
        validate_split_counts,
    )
except ImportError:  # Support: python src/validate_dataset.py
    from pipeline_utils import (
        load_config,
        validate_class_names,
        validate_split_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dataset-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(args.dataset_root)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    classes = validate_class_names(data_yaml)
    splits = validate_split_counts(root, config["dataset"]["splits"])
    print(json.dumps({"classes": classes, "splits": splits}, indent=2))


if __name__ == "__main__":
    main()

