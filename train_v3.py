#!/usr/bin/env python3
"""Train improved detector v3 — fine-tune từ v2.

Nhắm residual errors sau v2 (camera LIVE):
1) 5000 ↔ 20000 nhầm lẫn cao
2) 100000 ↔ 500000 nhầm lẫn cao
3) Tờ để dọc (portrait) kém hơn tờ ngang

Chiến lược:
- Fine-tune từ yolov8s_v2_best.pt (không train lại từ COCO)
- Dataset mở rộng: thêm bản xoay 90°/270° (học hướng dọc)
- Oversample ảnh chứa class khó (5000, 20000, 100000, 500000)
- imgsz lớn hơn + cls loss cao hơn + degrees rộng

Usage:
  python train_v3.py
  python train_v3.py --epochs 80 --batch 16 --imgsz 800
  python train_v3.py --quick
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from train_baseline import (
    CLEAN_DATA_DIR,
    ROOT,
    RUNS_DIR,
    SPLITS,
    IMAGE_EXTS,
    build_clean_dataset,
)

EXPORT_DIR = ROOT / "notebooks" / "v3_exports"
V3_DATA_DIR = ROOT / "datasets" / "vn_currency_yolo_v3"
V3_YAML = V3_DATA_DIR / "data.yaml"

V2_WEIGHT_CANDIDATES = [
    ROOT / "notebooks" / "v2_exports" / "yolov8s_v2_best.pt",
    ROOT / "runs" / "detect" / "vn_currency_yolov8s_v2" / "weights" / "best.pt",
]

# class_id theo data.yaml gốc
HARD_CLASS_IDS = {
    6,  # 5000
    4,  # 20000
    2,  # 100000
    8,  # 500000
}


def rotate_yolo_box_90cw(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float]:
    # ảnh xoay 90° CW: (x,y) -> (1-y, x)
    return (1.0 - yc, xc, h, w)


def rotate_yolo_box_90ccw(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float]:
    # ảnh xoay 90° CCW: (x,y) -> (y, 1-x)
    return (yc, 1.0 - xc, h, w)


def transform_label_file(src: Path, dst: Path, mode: str) -> None:
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid, xc, yc, w, h = line.split()[:5]
        xc, yc, w, h = map(float, (xc, yc, w, h))
        if mode == "rot90cw":
            xc, yc, w, h = rotate_yolo_box_90cw(xc, yc, w, h)
        elif mode == "rot90ccw":
            xc, yc, w, h = rotate_yolo_box_90ccw(xc, yc, w, h)
        elif mode == "copy":
            pass
        else:
            raise ValueError(mode)
        xc = float(np.clip(xc, 0.0, 1.0))
        yc = float(np.clip(yc, 0.0, 1.0))
        w = float(np.clip(w, 1e-6, 1.0))
        h = float(np.clip(h, 1e-6, 1.0))
        lines_out.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def label_has_hard_class(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid = int(float(line.split()[0]))
        if cid in HARD_CLASS_IDS:
            return True
    return False


def build_v3_dataset(force: bool = False) -> dict:
    """Từ dataset bbox sạch → thêm xoay 90/270 + oversample class khó (chỉ train)."""
    marker = V3_DATA_DIR / ".v3_ready"
    if V3_DATA_DIR.exists() and marker.exists() and not force:
        print(f"[data] Dùng lại dataset v3: {V3_DATA_DIR}")
        return json.loads((V3_DATA_DIR / "build_stats.json").read_text(encoding="utf-8"))

    build_clean_dataset(force=False)
    if V3_DATA_DIR.exists():
        shutil.rmtree(V3_DATA_DIR)

    stats = {
        "train_base": 0,
        "train_rot90": 0,
        "train_rot270": 0,
        "train_hard_oversample": 0,
        "valid": 0,
        "test": 0,
    }

    with open(CLEAN_DATA_DIR / "data.yaml", "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    class_names = list(raw_cfg["names"])

    for split in SPLITS:
        src_img = CLEAN_DATA_DIR / split / "images"
        src_lbl = CLEAN_DATA_DIR / split / "labels"
        dst_img = V3_DATA_DIR / split / "images"
        dst_lbl = V3_DATA_DIR / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = [p for p in sorted(src_img.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        for img_path in images:
            stem = img_path.stem
            lbl_path = src_lbl / f"{stem}.txt"
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # bản gốc
            cv2.imwrite(str(dst_img / img_path.name), img)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / f"{stem}.txt")
            else:
                (dst_lbl / f"{stem}.txt").write_text("")
            stats["valid" if split == "valid" else ("test" if split == "test" else "train_base")] += 1

            # Chỉ mở rộng TRAIN
            if split != "train":
                continue

            # xoay 90 CW
            rot_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            name_cw = f"{stem}_rot90cw.jpg"
            cv2.imwrite(str(dst_img / name_cw), rot_cw)
            transform_label_file(lbl_path, dst_lbl / f"{Path(name_cw).stem}.txt", "rot90cw")
            stats["train_rot90"] += 1

            # xoay 90 CCW (=270 CW)
            rot_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            name_ccw = f"{stem}_rot90ccw.jpg"
            cv2.imwrite(str(dst_img / name_ccw), rot_ccw)
            transform_label_file(lbl_path, dst_lbl / f"{Path(name_ccw).stem}.txt", "rot90ccw")
            stats["train_rot270"] += 1

            # oversample class khó: nhân đôi bản gốc (copy) nếu có hard class
            if label_has_hard_class(lbl_path):
                name_os = f"{stem}_hardos.jpg"
                cv2.imwrite(str(dst_img / name_os), img)
                transform_label_file(lbl_path, dst_lbl / f"{Path(name_os).stem}.txt", "copy")
                stats["train_hard_oversample"] += 1

    data_yaml = {
        "path": str(V3_DATA_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(V3_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    with open(V3_DATA_DIR / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")

    n_train = stats["train_base"] + stats["train_rot90"] + stats["train_rot270"] + stats["train_hard_oversample"]
    print("[data] v3 dataset:", stats)
    print(f"[data] train images total ≈ {n_train}")
    print(f"[data] yaml: {V3_YAML}")
    return stats


def find_v2_weights(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for p in V2_WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Không tìm thấy weights v2. Chỉ định --weights hoặc train v2 trước.\n"
        + "\n".join(str(p) for p in V2_WEIGHT_CANDIDATES)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 v3 (finetune from v2)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--weights", type=str, default=None, help="Checkpoint v2 để fine-tune")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=800, help="Lớn hơn v2 để đọc chi tiết mệnh giá")
    p.add_argument("--name", type=str, default="vn_currency_yolov8s_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-build-data", action="store_true")
    p.add_argument("--eval-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = args.epochs if args.epochs is not None else (5 if args.quick else 80)
    # imgsz 800 + yolov8s: batch nhỏ hơn v2
    if args.batch is not None:
        batch = args.batch
    else:
        batch = 16 if torch.cuda.is_available() else 2

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    build_v3_dataset(force=args.force_build_data)
    with open(V3_YAML, "r", encoding="utf-8") as f:
        class_names = list(yaml.safe_load(f)["names"])

    v2_w = find_v2_weights(args.weights)
    print(f"[train] fine-tune from: {v2_w}")

    train_kwargs = dict(
        data=str(V3_YAML),
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
        patience=30,
        save=True,
        plots=True,
        verbose=True,
        cos_lr=True,
        # fine-tune: LR thấp hơn mặc định một chút
        lr0=0.005,
        lrf=0.01,
        # augment: xoay mạnh (dọc/ngang) + vẫn giữ scale/nền
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.6,
        degrees=90.0,       # quan trọng: tờ để dọc
        translate=0.12,
        scale=0.55,
        shear=3.0,
        perspective=0.0008,
        flipud=0.1,         # thêm biến thể hướng
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.2,
        erasing=0.3,
        close_mosaic=15,
        # nhấn mạnh phân lớp (giảm 5k↔20k, 100k↔500k)
        box=7.5,
        cls=1.5,
        dfl=1.5,
    )

    print("[train] recipe v3:", {k: train_kwargs[k] for k in [
        "epochs", "batch", "imgsz", "degrees", "scale", "cls", "hsv_s", "flipud", "lr0"
    ]})

    model = YOLO(str(v2_w))
    model.train(**train_kwargs)

    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    print(f"[train] RUN_DIR={run_dir}")
    print(f"[train] BEST_PT={best_pt}")

    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(
        data=str(V3_YAML),
        split="val",
        imgsz=args.imgsz,
        batch=max(1, batch // 2),
        device=device,
        plots=True,
        verbose=True,
    )
    box = val_metrics.box
    metrics_val = {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    if getattr(box, "ap50", None) is not None:
        ap50 = np.array(box.ap50).reshape(-1)
        for i, name in enumerate(class_names):
            if i < len(ap50):
                metrics_val[f"AP50_{name}"] = float(ap50[i])

    # In riêng các class đang lỗi
    focus = ["5000", "20000", "100000", "500000"]
    print("[val] focus classes:")
    for c in focus:
        print(f"  AP50_{c} = {metrics_val.get(f'AP50_{c}')}")

    metrics_test = None
    if args.eval_test:
        test_metrics = best_model.val(
            data=str(V3_YAML),
            split="test",
            imgsz=args.imgsz,
            batch=max(1, batch // 2),
            device=device,
            plots=False,
            verbose=True,
        )
        tbox = test_metrics.box
        metrics_test = {
            "mAP50": float(tbox.map50),
            "mAP50-95": float(tbox.map),
            "precision": float(tbox.mp),
            "recall": float(tbox.mr),
        }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": "v3",
        "finetune_from": str(v2_w),
        "run_name": args.name,
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "data_yaml": str(V3_YAML),
        "imgsz": args.imgsz,
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
        "device": str(device),
        "class_names": class_names,
        "targets": [
            "reduce 5000 ↔ 20000 confusion",
            "reduce 100000 ↔ 500000 confusion",
            "improve vertical (portrait) banknotes",
        ],
        "data_expansion": [
            "train + rot90cw + rot90ccw",
            "oversample images containing hard classes",
        ],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "val": metrics_val,
        "test": metrics_test,
    }
    row = {
        "model": "YOLOv8s v3 (finetune v2 + rot/oversample)",
        "val_mAP50": metrics_val["mAP50"],
        "val_mAP50-95": metrics_val["mAP50-95"],
        "val_precision": metrics_val["precision"],
        "val_recall": metrics_val["recall"],
    }
    for c in focus:
        row[f"AP50_{c}"] = metrics_val.get(f"AP50_{c}")

    pd.DataFrame([row]).to_csv(EXPORT_DIR / "v3_val_metrics.csv", index=False)
    with open(EXPORT_DIR / "v3_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    shutil.copy2(best_pt, EXPORT_DIR / "yolov8s_v3_best.pt")

    # Bảng so sánh nếu có baseline/v2
    cmp_rows = []
    for label, csv_path in [
        ("baseline", ROOT / "notebooks" / "baseline_exports" / "baseline_val_metrics.csv"),
        ("v2", ROOT / "notebooks" / "v2_exports" / "v2_val_metrics.csv"),
    ]:
        if csv_path.exists():
            d = pd.read_csv(csv_path).iloc[0].to_dict()
            cmp_rows.append({
                "model": label,
                "val_mAP50": d.get("val_mAP50"),
                "val_mAP50-95": d.get("val_mAP50-95"),
                "val_precision": d.get("val_precision"),
                "val_recall": d.get("val_recall"),
            })
    cmp_rows.append(row)
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(EXPORT_DIR / "compare_baseline_v2_v3_val.csv", index=False)
    print(cmp)

    print("===== V3 DONE =====")
    print(json.dumps(metrics_val, indent=2))
    print(f"Exports: {EXPORT_DIR}")
    print(
        "\nDemo:\n"
        f"  python demo_camera.py --weights {EXPORT_DIR / 'yolov8s_v3_best.pt'} --conf 0.30 --imgsz {args.imgsz}\n"
        "Test đặc biệt: tờ 5000 & 100000 để dọc trên nền rối."
    )


if __name__ == "__main__":
    main()
