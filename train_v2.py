#!/usr/bin/env python3
"""Train improved detector v2 (A1+A2) for VN currency.

Nhắm 4 lỗi quan sát trên camera LIVE:
1) Nền rối → chỉ nhận khi tờ che sát camera
2) Khó nhận ≥2 tờ cùng lúc
3) Nhầm 1000/2000/5000 ↔ 20000/50000
4) Yếu mặt sau (đặc biệt 5000)

Usage (server có GPU):
  conda activate vn_currency   # hoặc venv tương đương
  python train_v2.py
  python train_v2.py --epochs 100 --batch 24 --model yolov8s.pt
  python train_v2.py --quick
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

# Tái sử dụng convert từ baseline
from train_baseline import (
    CLEAN_DATA_DIR,
    CLEAN_YAML,
    ROOT,
    RUNS_DIR,
    build_clean_dataset,
    write_data_yaml,
)

EXPORT_DIR = ROOT / "notebooks" / "v2_exports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 improved v2")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--model", type=str, default="yolov8s.pt",
                    help="yolov8s.pt (khuyến nghị) hoặc yolov8n.pt / yolov8m.pt")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", type=str, default="vn_currency_yolov8s_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-convert", action="store_true")
    p.add_argument("--eval-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = args.epochs if args.epochs is not None else (5 if args.quick else 100)
    if args.batch is not None:
        batch = args.batch
    else:
        # v8s cần VRAM hơn n; 4090 24GB: 24–32 ổn
        batch = 24 if torch.cuda.is_available() else 4

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    build_clean_dataset(force=args.force_convert)
    class_names = write_data_yaml()

    # ===== Augment / train recipe nhắm lỗi LIVE =====
    # - scale rộng: tờ tiền nhỏ hơn trong khung (không cần che 70% cam)
    # - mosaic + mixup + copy_paste: nhiều tờ / chồng ngữ cảnh
    # - HSV + perspective: ánh sáng webcam, góc nghiêng
    # - erasing: che khuất một phần (tay / bóng)
    # - cls loss cao hơn mặc định một chút: giảm nhầm mệnh giá gần màu
    train_kwargs = dict(
        data=str(CLEAN_YAML),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=8,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        seed=seed,
        deterministic=True,
        patience=35,
        save=True,
        plots=True,
        verbose=True,
        cos_lr=True,
        # --- augmentation ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=12.0,
        translate=0.15,
        scale=0.6,          # quan trọng: object nhỏ/xa hơn trong frame
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.25,
        erasing=0.25,
        close_mosaic=20,    # tắt mosaic cuối để ổn định box
        # --- loss emphasis ---
        box=7.5,
        cls=1.0,            # mặc định ultralytics ~0.5; tăng để phân lớp tốt hơn
        dfl=1.5,
    )

    print("[train] recipe v2:", {k: train_kwargs[k] for k in [
        "epochs", "batch", "imgsz", "scale", "mosaic", "mixup", "copy_paste",
        "erasing", "cls", "hsv_s", "hsv_v", "degrees"
    ]})

    model = YOLO(args.model)
    model.train(**train_kwargs)

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
        "version": "v2",
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
        "train_aug": {
            "scale": 0.6,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.25,
            "erasing": 0.25,
            "cls": 1.0,
            "hsv_s": 0.7,
            "hsv_v": 0.5,
            "degrees": 12.0,
        },
        "targets": [
            "robust to cluttered background / smaller bills in frame",
            "better multi-bill detection",
            "reduce confusion among 1000/2000/5000 vs 20000/50000",
            "improve generalization (still need more back-side photos if weak)",
        ],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "val": baseline_val,
        "test": baseline_test,
    }
    metrics_row = {
        "model": f"{args.model} v2 improved",
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

    pd.DataFrame([metrics_row]).to_csv(EXPORT_DIR / "v2_val_metrics.csv", index=False)
    with open(EXPORT_DIR / "v2_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    shutil.copy2(best_pt, EXPORT_DIR / "yolov8s_v2_best.pt")

    # So sánh nhanh với baseline nếu có
    base_csv = ROOT / "notebooks" / "baseline_exports" / "baseline_val_metrics.csv"
    if base_csv.exists():
        base = pd.read_csv(base_csv).iloc[0].to_dict()
        cmp = pd.DataFrame([
            {
                "model": "baseline_yolov8n",
                "val_mAP50": base.get("val_mAP50"),
                "val_mAP50-95": base.get("val_mAP50-95"),
                "val_precision": base.get("val_precision"),
                "val_recall": base.get("val_recall"),
            },
            metrics_row,
        ])
        cmp.to_csv(EXPORT_DIR / "compare_baseline_vs_v2_val.csv", index=False)
        print(cmp)

    print("===== V2 DONE =====")
    print(json.dumps(baseline_val, indent=2))
    print(f"Exports: {EXPORT_DIR}")
    print(
        "\nSau train: cập nhật demo\n"
        f"  python demo_camera.py --weights {EXPORT_DIR / 'yolov8s_v2_best.pt'} --conf 0.30\n"
        "Và BỔ SUNG ảnh mặt sau 1000/2000/5000 + nền rối nếu vẫn yếu trên cam."
    )


if __name__ == "__main__":
    main()
