#!/usr/bin/env python3
"""Train baseline YOLOv8n (A0) — chạy headless trên server.

Usage:
  source .venv/bin/activate
  python train_baseline.py
  python train_baseline.py --quick          # 3 epoch thử pipeline
  python train_baseline.py --epochs 50 --batch 32
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "Vietnamese Currency.v1i.yolov8"
CLEAN_DATA_DIR = ROOT / "datasets" / "vn_currency_yolo_bbox"
CLEAN_YAML = CLEAN_DATA_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
EXPORT_DIR = ROOT / "notebooks" / "baseline_exports"
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def line_to_bbox(line: str) -> str:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid label: {line!r}")
    class_id = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    if len(coords) == 4:
        xc, yc, w, h = coords
    else:
        if len(coords) % 2 != 0:
            raise ValueError(f"Odd polygon coords: {line!r}")
        xs, ys = coords[0::2], coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = max(x_max - x_min, 1e-6)
        h = max(y_max - y_min, 1e-6)
        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    w = float(np.clip(w, 1e-6, 1.0))
    h = float(np.clip(h, 1e-6, 1.0))
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def convert_label_file(src: Path, dst: Path) -> dict:
    stats = {"n_lines": 0, "n_bbox": 0, "n_polygon": 0}
    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_coord = len(line.split()) - 1
        stats["n_bbox" if n_coord == 4 else "n_polygon"] += 1
        out_lines.append(line_to_bbox(line))
        stats["n_lines"] += 1
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return stats


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def build_clean_dataset(force: bool = False) -> None:
    marker = CLEAN_DATA_DIR / ".converted_ok"
    if CLEAN_DATA_DIR.exists() and marker.exists() and not force:
        print(f"[data] Dùng lại dataset sạch: {CLEAN_DATA_DIR}")
        return

    if CLEAN_DATA_DIR.exists() and force:
        shutil.rmtree(CLEAN_DATA_DIR)

    assert RAW_DATA_DIR.exists(), f"Missing dataset: {RAW_DATA_DIR}"
    rows = []
    for split in SPLITS:
        src_img = RAW_DATA_DIR / split / "images"
        src_lbl = RAW_DATA_DIR / split / "labels"
        dst_img = CLEAN_DATA_DIR / split / "images"
        dst_lbl = CLEAN_DATA_DIR / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = [p for p in sorted(src_img.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        for img_path in images:
            ensure_link_or_copy(img_path, dst_img / img_path.name)
            lbl_src = src_lbl / f"{img_path.stem}.txt"
            lbl_dst = dst_lbl / f"{img_path.stem}.txt"
            if lbl_src.exists():
                st = convert_label_file(lbl_src, lbl_dst)
            else:
                lbl_dst.write_text("", encoding="utf-8")
                st = {"n_lines": 0, "n_bbox": 0, "n_polygon": 0}
            rows.append({"split": split, "image": img_path.name, **st})

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(CLEAN_DATA_DIR / "convert_stats.csv", index=False)
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    print(
        f"[data] Convert xong: images={len(stats_df)}, "
        f"objects={int(stats_df['n_lines'].sum())}, "
        f"polygon→bbox={int(stats_df['n_polygon'].sum())}"
    )


def write_data_yaml() -> list[str]:
    with open(RAW_DATA_DIR / "data.yaml", "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    class_names = list(raw_cfg["names"])
    data_yaml = {
        "path": str(CLEAN_DATA_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    CLEAN_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"[data] YAML: {CLEAN_YAML}")
    return class_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n baseline for VN currency")
    p.add_argument("--quick", action="store_true", help="3 epochs smoke test")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--name", type=str, default="vn_currency_yolov8n_baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-convert", action="store_true")
    p.add_argument("--eval-test", action="store_true", help="Also evaluate on test (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = args.epochs if args.epochs is not None else (3 if args.quick else 50)
    if args.batch is not None:
        batch = args.batch
    else:
        batch = 16 if torch.cuda.is_available() else 4

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    build_clean_dataset(force=args.force_convert)
    class_names = write_data_yaml()

    model = YOLO(args.model)
    model.train(
        data=str(CLEAN_YAML),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=4,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        seed=seed,
        deterministic=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    print(f"[train] RUN_DIR={run_dir}")
    print(f"[train] BEST_PT={best_pt}")

    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(
        data=str(CLEAN_YAML),
        split="val",
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        plots=True,
        verbose=True,
    )
    box = val_metrics.box
    baseline_val = {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    if getattr(box, "ap50", None) is not None:
        ap50 = np.array(box.ap50).reshape(-1)
        for i, name in enumerate(class_names):
            if i < len(ap50):
                baseline_val[f"AP50_{name}"] = float(ap50[i])

    baseline_test = None
    if args.eval_test:
        test_metrics = best_model.val(
            data=str(CLEAN_YAML),
            split="test",
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            plots=False,
            verbose=True,
        )
        tbox = test_metrics.box
        baseline_test = {
            "mAP50": float(tbox.map50),
            "mAP50-95": float(tbox.map),
            "precision": float(tbox.mp),
            "recall": float(tbox.mr),
        }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "run_name": args.name,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "data_yaml": str(CLEAN_YAML),
        "imgsz": args.imgsz,
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
        "device": str(device),
        "class_names": class_names,
        "quick": args.quick,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "val": baseline_val,
        "test": baseline_test,
    }
    metrics_row = {
        "model": "YOLOv8n baseline (A0)",
        "val_mAP50": baseline_val["mAP50"],
        "val_mAP50-95": baseline_val["mAP50-95"],
        "val_precision": baseline_val["precision"],
        "val_recall": baseline_val["recall"],
    }
    if baseline_test:
        metrics_row.update({
            "test_mAP50": baseline_test["mAP50"],
            "test_mAP50-95": baseline_test["mAP50-95"],
            "test_precision": baseline_test["precision"],
            "test_recall": baseline_test["recall"],
        })

    pd.DataFrame([metrics_row]).to_csv(EXPORT_DIR / "baseline_val_metrics.csv", index=False)
    with open(EXPORT_DIR / "baseline_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    shutil.copy2(best_pt, EXPORT_DIR / "yolov8n_baseline_best.pt")

    print("===== BASELINE DONE =====")
    print(json.dumps(baseline_val, indent=2))
    print(f"Exports: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
