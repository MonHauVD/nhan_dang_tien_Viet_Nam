#!/usr/bin/env python3
"""Train v4 — fine-tune từ v3, nhắm residual errors camera.

Sau v3:
1) 200000 hay thành 10000 / 500000  → oversample mạnh 200000 + hard-pair context
2) 50000/100000 bị 'nhận rồi mất'   → chủ yếu fix demo (temporal smooth);
                                       train thêm blur/nhiễu để ổn định hơn
3) 5000 ↔ 20000 vẫn nhầm            → oversample cặp này + cls cao
4) 1000/2000, 20000, 500000 đã ổn   → giữ bằng fine-tune từ v3 (không reset)

Usage:
  python train_v4.py
  python train_v4.py --epochs 60 --batch 16 --imgsz 800
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

from train_baseline import IMAGE_EXTS, ROOT, RUNS_DIR, SPLITS, build_clean_dataset
from train_v3 import V3_DATA_DIR, V3_YAML, build_v3_dataset

EXPORT_DIR = ROOT / "notebooks" / "v4_exports"
V4_DATA_DIR = ROOT / "datasets" / "vn_currency_yolo_v4"
V4_YAML = V4_DATA_DIR / "data.yaml"
REAL_CAM_DIR = ROOT / "datasets" / "real_cam_yolo"

V3_WEIGHT_CANDIDATES = [
    ROOT / "notebooks" / "v3_exports" / "yolov8s_v3_best.pt",
    ROOT / "runs" / "detect" / "vn_currency_yolov8s_v3" / "weights" / "best.pt",
]

# data.yaml names index
# 1000,10000,100000,2000,20000,200000,5000,50000,500000
FOCUS_OVERSAMPLE = {
    5: 4,  # 200000 ×4 (yếu nhất trên cam)
    6: 3,  # 5000 ×3
    4: 2,  # 20000 ×2 (để cạnh 5000)
    2: 2,  # 100000 ×2
    7: 2,  # 50000 ×2 (ổn định hơn trên cam)
}


def find_v3_weights(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for p in V3_WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Không thấy weights v3. Hãy chỉ định --weights")


def _class_ids_in_label(label_path: Path) -> set[int]:
    ids = set()
    if not label_path.exists():
        return ids
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(int(float(line.split()[0])))
    return ids


def _max_oversample_factor(cids: set[int]) -> int:
    fac = 1
    for cid in cids:
        fac = max(fac, FOCUS_OVERSAMPLE.get(cid, 1))
    return fac


def build_v4_dataset(force: bool = False) -> dict:
    """v3 data + oversample thêm các class đang yếu trên camera."""
    marker = V4_DATA_DIR / ".v4_ready"
    if V4_DATA_DIR.exists() and marker.exists() and not force:
        print(f"[data] Dùng lại dataset v4: {V4_DATA_DIR}")
        return json.loads((V4_DATA_DIR / "build_stats.json").read_text(encoding="utf-8"))

    # đảm bảo v3 data có (xoay sẵn)
    build_v3_dataset(force=False)
    if V4_DATA_DIR.exists():
        shutil.rmtree(V4_DATA_DIR)

    with open(V3_YAML, "r", encoding="utf-8") as f:
        class_names = list(yaml.safe_load(f)["names"])

    stats = {"copied": 0, "extra_oversample": 0, "motion_blur_copies": 0}

    for split in SPLITS:
        src_img = V3_DATA_DIR / split / "images"
        src_lbl = V3_DATA_DIR / split / "labels"
        dst_img = V4_DATA_DIR / split / "images"
        dst_lbl = V4_DATA_DIR / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = [p for p in sorted(src_img.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        for img_path in images:
            stem = img_path.stem
            lbl_path = src_lbl / f"{stem}.txt"
            # copy gốc
            shutil.copy2(img_path, dst_img / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / f"{stem}.txt")
            else:
                (dst_lbl / f"{stem}.txt").write_text("")
            stats["copied"] += 1

            if split != "train":
                continue

            cids = _class_ids_in_label(lbl_path)
            factor = _max_oversample_factor(cids)
            # đã có 1 bản → thêm (factor-1) bản nữa
            for k in range(1, factor):
                out_name = f"{stem}_os{k}.jpg"
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # jitter nhẹ màu/độ sáng để không copy y hệt
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.85, 1.15), 0, 255)
                jitter = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                cv2.imwrite(str(dst_img / out_name), jitter)
                shutil.copy2(lbl_path, dst_lbl / f"{Path(out_name).stem}.txt")
                stats["extra_oversample"] += 1

            # motion-blur copy cho ảnh có 50000/100000/200000 — chống mất nhận trên cam rung
            if cids & {5, 7, 2}:
                img = cv2.imread(str(img_path))
                if img is not None:
                    k = random.choice([5, 7, 9])
                    kernel = np.zeros((k, k))
                    kernel[k // 2, :] = 1.0 / k
                    blur = cv2.filter2D(img, -1, kernel)
                    out_name = f"{stem}_mblur.jpg"
                    cv2.imwrite(str(dst_img / out_name), blur)
                    shutil.copy2(lbl_path, dst_lbl / f"{Path(out_name).stem}.txt")
                    stats["motion_blur_copies"] += 1

    # ===== Merge ảnh cam thật (mờ / nền rối / front+back) =====
    stats["real_cam_base"] = 0
    stats["real_cam_aug"] = 0
    if (REAL_CAM_DIR / "images").exists() and (REAL_CAM_DIR / "labels").exists():
        real_imgs = sorted(
            p for p in (REAL_CAM_DIR / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        dst_img = V4_DATA_DIR / "train" / "images"
        dst_lbl = V4_DATA_DIR / "train" / "labels"
        for img_path in real_imgs:
            stem = img_path.stem
            lbl_path = REAL_CAM_DIR / "labels" / f"{stem}.txt"
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # nhân bản mạnh vì chỉ có ~18 ảnh cam domain
            variants = [("raw", img)]
            # downscale → upscale (cam độ phân giải thấp)
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            lowres = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            variants.append(("lowres", lowres))
            # gaussian blur
            variants.append(("gblur", cv2.GaussianBlur(img, (7, 7), 0)))
            variants.append(("gblur2", cv2.GaussianBlur(img, (11, 11), 0)))
            # motion blur
            k = 9
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            variants.append(("mblur", cv2.filter2D(img, -1, kernel)))
            # brightness
            bright = cv2.convertScaleAbs(img, alpha=1.0, beta=25)
            dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-25)
            variants.append(("bright", bright))
            variants.append(("dark", dark))
            # xoay 90 để cover tờ dọc trên cam
            variants.append(("rot90cw", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)))
            variants.append(("rot90ccw", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)))

            from train_v3 import transform_label_file

            for tag, arr in variants:
                out_stem = f"realcam_{stem}_{tag}"
                cv2.imwrite(str(dst_img / f"{out_stem}.jpg"), arr)
                if tag in {"rot90cw", "rot90ccw"}:
                    mode = "rot90cw" if tag == "rot90cw" else "rot90ccw"
                    transform_label_file(lbl_path, dst_lbl / f"{out_stem}.txt", mode)
                else:
                    shutil.copy2(lbl_path, dst_lbl / f"{out_stem}.txt")
                if tag == "raw":
                    stats["real_cam_base"] += 1
                else:
                    stats["real_cam_aug"] += 1
        print(
            f"[data] merged real_cam: base={stats['real_cam_base']} "
            f"aug={stats['real_cam_aug']}"
        )
    else:
        print(
            f"[data] Chưa có {REAL_CAM_DIR} — chạy trước: python label_real_images.py"
        )

    data_yaml = {
        "path": str(V4_DATA_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(V4_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    with open(V4_DATA_DIR / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    print("[data] v4:", stats)
    print("[data] yaml:", V4_YAML)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 v4 (finetune from v3)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--name", type=str, default="vn_currency_yolov8s_v4")
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
    epochs = args.epochs if args.epochs is not None else (5 if args.quick else 60)
    batch = args.batch if args.batch is not None else (16 if torch.cuda.is_available() else 2)

    print(f"[env] device={device}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    build_clean_dataset(force=False)
    build_v4_dataset(force=args.force_build_data)
    with open(V4_YAML, "r", encoding="utf-8") as f:
        class_names = list(yaml.safe_load(f)["names"])

    v3_w = find_v3_weights(args.weights)
    print(f"[train] fine-tune from: {v3_w}")

    model = YOLO(str(v3_w))
    model.train(
        data=str(V4_YAML),
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
        patience=25,
        save=True,
        plots=True,
        verbose=True,
        cos_lr=True,
        lr0=0.003,
        lrf=0.01,
        hsv_h=0.025,   # tách màu 5k↔20k, 200k↔500k tốt hơn một chút
        hsv_s=0.85,
        hsv_v=0.65,
        degrees=90.0,
        translate=0.12,
        scale=0.55,
        shear=3.0,
        perspective=0.001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.15,
        erasing=0.35,
        close_mosaic=12,
        box=7.5,
        cls=1.8,       # nhấn phân lớp
        dfl=1.5,
    )

    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))
    val_metrics = best_model.val(
        data=str(V4_YAML),
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

    focus = ["200000", "5000", "20000", "100000", "50000", "10000", "500000"]
    print("[val] focus:")
    for c in focus:
        print(f"  AP50_{c}={metrics_val.get(f'AP50_{c}')}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "model": "YOLOv8s v4 (finetune v3 + focus oversample)",
        "val_mAP50": metrics_val["mAP50"],
        "val_mAP50-95": metrics_val["mAP50-95"],
        "val_precision": metrics_val["precision"],
        "val_recall": metrics_val["recall"],
    }
    for c in focus:
        row[f"AP50_{c}"] = metrics_val.get(f"AP50_{c}")
    pd.DataFrame([row]).to_csv(EXPORT_DIR / "v4_val_metrics.csv", index=False)
    meta = {
        "version": "v4",
        "finetune_from": str(v3_w),
        "best_pt": str(best_pt),
        "run_dir": str(run_dir),
        "focus": list(FOCUS_OVERSAMPLE.keys()),
        "val": metrics_val,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Kèm demo temporal smooth để giảm mất nhận dạng 50k/100k giữa các frame",
    }
    with open(EXPORT_DIR / "v4_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    shutil.copy2(best_pt, EXPORT_DIR / "yolov8s_v4_best.pt")

    print("===== V4 DONE =====")
    print(json.dumps(metrics_val, indent=2))
    print(
        f"\nDemo:\n"
        f"  python demo_camera.py --weights {EXPORT_DIR / 'yolov8s_v4_best.pt'} "
        f"--mode live --conf 0.28 --imgsz {args.imgsz} --hold-frames 15\n"
    )


if __name__ == "__main__":
    main()
