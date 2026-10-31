"""Train a configured Ultralytics detector on an exported dataset version."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .pipeline_utils import (
        environment_info,
        load_config,
        validate_class_names,
        validate_split_counts,
        write_json,
    )
except ImportError:  # Support: python src/train.py
    from pipeline_utils import (
        environment_info,
        load_config,
        validate_class_names,
        validate_split_counts,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/config and print resolved settings without training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch under a distinct run name.",
    )
    return parser.parse_args()


def build_train_args(config: dict[str, Any], smoke_test: bool) -> dict[str, Any]:
    """Translate the YAML train section to Ultralytics keyword arguments."""
    train_args = dict(config["train"])
    if smoke_test:
        train_args.update({"epochs": 1, "patience": 1})
    return train_args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = Path(args.dataset_root)
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    validate_class_names(data_yaml)
    split_counts = validate_split_counts(dataset_root, config["dataset"]["splits"])

    if args.dry_run:
        print(
            {
                "status": "dry-run-ok",
                "dataset_root": str(dataset_root),
                "splits": split_counts,
                "train_args": build_train_args(config, args.smoke_test),
            }
        )
        return

    output_root = Path(args.output_root or config["output"]["drive_root"])
    run_name = config["output"]["run_name"]
    if args.smoke_test:
        run_name = f"{run_name}-smoke"

    run_dir = output_root / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Choose another output root or archive the previous run."
        )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_test": args.smoke_test,
        "config": config,
        "environment": environment_info(),
    }
    write_json(output_root / "run_metadata" / f"{run_name}.json", metadata)

    from ultralytics import YOLO

    model = YOLO(config["model"]["checkpoint"])
    model.train(
        data=str(data_yaml),
        project=str(output_root / "runs"),
        name=run_name,
        exist_ok=False,
        **build_train_args(config, args.smoke_test),
    )


if __name__ == "__main__":
    main()


