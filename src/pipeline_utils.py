"""Shared, side-effect-free helpers for the training pipeline."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


EXPECTED_CLASSES = (
    "1000",
    "10000",
    "100000",
    "2000",
    "20000",
    "200000",
    "5000",
    "50000",
    "500000",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    for key in ("experiment_id", "name", "dataset", "model", "train", "output"):
        if key not in config:
            raise ValueError(f"Missing config key '{key}' in {config_path}")

    version = config["dataset"].get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError("dataset.version must be a positive integer")

    return config


def split_project_id(project_id: str) -> tuple[str, str]:
    """Return workspace and project slugs from a full Roboflow project ID."""
    parts = project_id.split("/", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("project_id must have the form 'workspace/project'")
    return parts[0], parts[1]


def read_class_names(data_yaml: str | Path) -> tuple[str, ...]:
    """Read class names from a YOLO data.yaml file."""
    with Path(data_yaml).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data.get("names")
    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names, key=lambda value: int(value))]
    elif isinstance(names, list):
        ordered = names
    else:
        raise ValueError("data.yaml must contain names as a list or indexed mapping")

    return tuple(str(name) for name in ordered)


def validate_class_names(
    data_yaml: str | Path, expected: tuple[str, ...] = EXPECTED_CLASSES
) -> tuple[str, ...]:
    """Fail fast if the downloaded dataset has an unexpected class mapping."""
    names = read_class_names(data_yaml)
    if names != expected:
        raise ValueError(
            "Unexpected class order. "
            f"Expected {expected}, received {names}."
        )
    return names


def count_split_images(dataset_root: str | Path) -> dict[str, int]:
    """Count images in Roboflow YOLO train/valid/test folders."""
    root = Path(dataset_root)
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing split image directory: {image_dir}")
        counts[split] = sum(
            1
            for item in image_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def validate_split_counts(
    dataset_root: str | Path, expected: dict[str, int]
) -> dict[str, int]:
    """Verify that an exported version contains the audited split sizes."""
    actual = count_split_images(dataset_root)
    normalized_expected = {key: int(expected[key]) for key in actual}
    if actual != normalized_expected:
        raise ValueError(
            f"Unexpected split counts. Expected {normalized_expected}, got {actual}."
        )
    return actual


def environment_info() -> dict[str, Any]:
    """Collect reproducibility metadata without requiring a GPU."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }
        )
    except ImportError:
        info["torch"] = None

    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        info["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    except OSError:
        info["git_commit"] = None

    return info


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON artifact, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

