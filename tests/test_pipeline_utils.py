from pathlib import Path
import tempfile
import unittest

import yaml

from src.pipeline_utils import (
    EXPECTED_CLASSES,
    load_config,
    split_project_id,
    validate_class_names,
    validate_split_counts,
)
from src.train import build_train_args


class PipelineUtilsTests(unittest.TestCase):
    def test_baseline_config_is_complete(self) -> None:
        config = load_config("configs/baseline.yaml")
        self.assertEqual(config["dataset"]["version"], 1)
        self.assertEqual(config["train"]["imgsz"], 640)
        self.assertEqual(config["train"]["seed"], 42)

    def test_split_project_id(self) -> None:
        self.assertEqual(
            split_project_id("workspace/project"), ("workspace", "project")
        )
        with self.assertRaises(ValueError):
            split_project_id("project-only")

    def test_smoke_test_overrides_only_runtime_length(self) -> None:
        config = load_config("configs/baseline.yaml")
        args = build_train_args(config, smoke_test=True)
        self.assertEqual(args["epochs"], 1)
        self.assertEqual(args["patience"], 1)
        self.assertEqual(args["imgsz"], 640)
        self.assertEqual(args["seed"], 42)

    def test_dataset_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.yaml").write_text(
                yaml.safe_dump({"names": list(EXPECTED_CLASSES)}),
                encoding="utf-8",
            )
            for split, count in {"train": 2, "valid": 1, "test": 1}.items():
                image_dir = root / split / "images"
                image_dir.mkdir(parents=True)
                for index in range(count):
                    (image_dir / f"{index}.jpg").write_bytes(b"test")

            self.assertEqual(
                validate_class_names(root / "data.yaml"), EXPECTED_CLASSES
            )
            self.assertEqual(
                validate_split_counts(
                    root, {"train": 2, "valid": 1, "test": 1}
                ),
                {"train": 2, "valid": 1, "test": 1},
            )

    def test_wrong_class_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.yaml"
            path.write_text(
                yaml.safe_dump({"names": list(reversed(EXPECTED_CLASSES))}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unexpected class order"):
                validate_class_names(path)


if __name__ == "__main__":
    unittest.main()

