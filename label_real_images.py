#!/usr/bin/env python3
"""Gắn bbox bán tự động cho real_image/ (ảnh cam test thật).

Quy ước tên file: {mệnh_giá}_{front|back}.jpg
  ví dụ: 200000_front.jpg, 5000_back.jpg

Class lấy từ TÊN FILE (đáng tin). Model chỉ đề xuất vị trí box.
Bạn nên xem thư mục review/ trước khi đưa vào train.

Usage:
  python label_real_images.py
  python label_real_images.py --weights notebooks/v3_exports/yolov8s_v3_best.pt --conf 0.15
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REAL_DIR = ROOT / "real_image"
OUT_DIR = ROOT / "datasets" / "real_cam_yolo"
REVIEW_DIR = OUT_DIR / "review"
CLASS_NAMES = ["1000", "10000", "100000", "2000", "20000", "200000", "5000", "50000", "500000"]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

WEIGHT_CANDIDATES = [
    ROOT / "notebooks" / "v3_exports" / "yolov8s_v3_best.pt",
    ROOT / "runs" / "detect" / "vn_currency_yolov8s_v3" / "weights" / "best.pt",
    ROOT / "notebooks" / "v2_exports" / "yolov8s_v2_best.pt",
    ROOT / "runs" / "detect" / "vn_currency_yolov8s_v2" / "weights" / "best.pt",
]

FNAME_RE = re.compile(r"^(?P<denom>\d+)_(?P<side>front|back)$", re.IGNORECASE)


def find_weights(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Không tìm thấy weights. Dùng --weights")


def parse_stem(stem: str) -> tuple[str, str]:
    m = FNAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"Tên file không đúng quy ước {{mệnh_giá}}_{{front|back}}: {stem}"
        )
    denom = m.group("denom")
    side = m.group("side").lower()
    if denom not in NAME_TO_ID:
        raise ValueError(f"Mệnh giá không có trong class list: {denom}")
    return denom, side


def xyxy_to_yolo(x1, y1, x2, y2, w, h) -> tuple[float, float, float, float]:
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    return (
        float(np.clip(xc, 0, 1)),
        float(np.clip(yc, 0, 1)),
        float(np.clip(bw / w, 1e-6, 1)),
        float(np.clip(bh / h, 1e-6, 1)),
    )


def pick_box(result, target_cls: int, conf_thr: float):
    """Chọn box tốt nhất: ưu tiên đúng class target, fallback conf cao nhất."""
    if result.boxes is None or len(result.boxes) == 0:
        return None, "no_detection"

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    same = [(i, confs[i]) for i in range(len(clss)) if clss[i] == target_cls and confs[i] >= conf_thr]
    if same:
        i = max(same, key=lambda t: t[1])[0]
        return (xyxy[i], confs[i], clss[i], "matched_class"), "ok"

    # fallback: bất kỳ box nào trên ngưỡng thấp hơn
    cand = [(i, confs[i]) for i in range(len(confs)) if confs[i] >= max(0.05, conf_thr * 0.5)]
    if not cand:
        i = int(np.argmax(confs))
        return (xyxy[i], confs[i], clss[i], "low_conf_override"), "override_class"
    i = max(cand, key=lambda t: t[1])[0]
    return (xyxy[i], confs[i], clss[i], "override_class"), "override_class"


def draw_review(img, box_xyxy, true_name: str, pred_name: str, conf: float, status: str):
    out = img.copy()
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    color = (0, 220, 0) if status == "ok" else (0, 165, 255)
    if status.startswith("fail"):
        color = (0, 0, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    label = f"GT:{true_name} pred:{pred_name} {conf:.2f} [{status}]"
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default=str(REAL_DIR))
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--conf", type=float, default=0.15, help="Ngưỡng thấp vì cam mờ")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    assert src.exists(), src

    weights = find_weights(args.weights)
    device = args.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    img_out = OUT_DIR / "images"
    lbl_out = OUT_DIR / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)
    REVIEW_DIR.mkdir(parents=True)

    model = YOLO(str(weights))
    print(f"[info] weights={weights}")
    print(f"[info] device={device} conf={args.conf} imgsz={args.imgsz}")

    rows = []
    images = sorted([p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for path in images:
        try:
            denom, side = parse_stem(path.stem)
        except ValueError as e:
            print("SKIP", path.name, e)
            continue
        target_cls = NAME_TO_ID[denom]
        img = cv2.imread(str(path))
        if img is None:
            print("FAIL read", path)
            continue
        h, w = img.shape[:2]

        result = model.predict(
            source=img, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False
        )[0]
        picked, status = pick_box(result, target_cls, args.conf)

        if picked is None:
            # fallback: box giữa ảnh (cần sửa tay trong review)
            margin = 0.08
            x1, y1 = w * margin, h * margin
            x2, y2 = w * (1 - margin), h * (1 - margin)
            conf = 0.0
            pred_cls = target_cls
            status = "fail_no_det_fullframe_guess"
            box_xyxy = np.array([x1, y1, x2, y2], dtype=float)
        else:
            box_xyxy, conf, pred_cls, _ = picked

        # Class luôn theo tên file
        xc, yc, bw, bh = xyxy_to_yolo(*box_xyxy, w, h)
        label_line = f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

        out_stem = f"{denom}_{side}"
        # copy ảnh
        dst_img = img_out / f"{out_stem}.jpg"
        cv2.imwrite(str(dst_img), img)
        (lbl_out / f"{out_stem}.txt").write_text(label_line, encoding="utf-8")

        review = draw_review(
            img,
            box_xyxy,
            true_name=denom,
            pred_name=CLASS_NAMES[pred_cls] if 0 <= pred_cls < len(CLASS_NAMES) else str(pred_cls),
            conf=float(conf),
            status=status,
        )
        cv2.imwrite(str(REVIEW_DIR / f"{out_stem}.jpg"), review)

        rows.append(
            {
                "file": path.name,
                "denom": denom,
                "side": side,
                "status": status,
                "det_conf": float(conf),
                "pred_cls_before_override": CLASS_NAMES[pred_cls]
                if 0 <= int(pred_cls) < len(CLASS_NAMES)
                else str(pred_cls),
            }
        )
        print(f"[{status:28s}] {path.name} conf={conf:.3f}")

    # data.yaml (để merge / train riêng)
    data_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "images",
        "val": "images",  # bộ này quá nhỏ — chỉ dùng train khi merge
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    with open(OUT_DIR / "label_report.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    ok_n = sum(1 for r in rows if r["status"] == "ok")
    ov_n = sum(1 for r in rows if r["status"] == "override_class")
    fail_n = sum(1 for r in rows if str(r["status"]).startswith("fail"))
    print("\n===== DONE =====")
    print(f"Total={len(rows)} | matched={ok_n} | override_class={ov_n} | fail/guess={fail_n}")
    print(f"Labels : {lbl_out}")
    print(f"Review : {REVIEW_DIR}")
    print("→ Mở review/*.jpg kiểm tra box. Sửa file labels/*.txt nếu box lệch.")
    print("→ Sau đó chạy: python train_v4.py  (đã merge real_cam)")


if __name__ == "__main__":
    main()
