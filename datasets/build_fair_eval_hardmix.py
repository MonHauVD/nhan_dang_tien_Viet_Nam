#!/usr/bin/env python3
"""Build fair_eval_hardmix: 16 Hard + 24 Normal real photos, polygon→bbox.

Selection rule (implementation detail): Hard = webcam-style difficult shots;
Normal = remaining real photos sampled randomly (seed fixed).
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

SEED = 42
N_NORMAL = 24
PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "datasets" / "new_valid_Vietnamese Currency.yolov8"
DST = PROJECT / "datasets" / "fair_eval_hardmix"
CLASS_NAMES = [
    "1000",
    "10000",
    "100000",
    "2000",
    "20000",
    "200000",
    "5000",
    "50000",
    "500000",
]


def poly_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = max(0.0, x_min), min(1.0, x_max)
    y_min, y_max = max(0.0, y_min), min(1.0, y_max)
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy, w, h


def convert_label_line(line: str) -> str:
    parts = line.strip().split()
    if not parts:
        return ""
    cls = parts[0]
    vals = [float(x) for x in parts[1:]]
    if len(vals) == 4:
        cx, cy, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        cx, cy, w, h = poly_to_bbox(vals)
    else:
        raise ValueError(f"Unsupported label format ({len(vals)} coords): {line[:80]}")
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_label_file(src: Path, dst: Path) -> dict[str, int]:
    raw = src.read_text(encoding="utf-8").strip()
    n_poly = n_bbox = 0
    out_lines: list[str] = []
    if raw:
        for line in raw.splitlines():
            n_coords = len(line.split()) - 1
            if n_coords == 4:
                n_bbox += 1
            elif n_coords > 4:
                n_poly += 1
            out_lines.append(convert_label_line(line))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return {"n_poly": n_poly, "n_bbox": n_bbox, "n_obj": len(out_lines)}


def paired_images() -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for split in ("train", "valid", "test"):
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                raise FileNotFoundError(f"Missing label for {img}")
            pairs.append((img, lbl, split))
    return pairs


def is_hard_shot(name: str) -> bool:
    """Difficult real-cam shots (glare / bend / clutter / backlight / rotation)."""
    return name.startswith("WIN")


def write_yaml(path: Path, root: Path) -> None:
    text = (
        f"path: {root.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "test: images\n"
        "nc: 9\n"
        "names:\n"
        + "".join(f"- '{n}'\n" for n in CLASS_NAMES)
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if DST.exists():
        shutil.rmtree(DST)
    img_out = DST / "images"
    lbl_out = DST / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    pairs = paired_images()
    hard = [(i, l, s) for i, l, s in pairs if is_hard_shot(i.name)]
    normal = [(i, l, s) for i, l, s in pairs if not is_hard_shot(i.name)]
    if len(hard) != 16:
        raise RuntimeError(f"Expected 16 Hard images, got {len(hard)}")
    if len(normal) < N_NORMAL:
        raise RuntimeError(f"Not enough Normal images: {len(normal)}")

    normal_sel = random.sample(normal, N_NORMAL)
    selected = hard + normal_sel
    manifest_rows: list[dict] = []
    stats = {"poly_converted": 0, "already_bbox": 0, "objects": 0}

    for img, lbl, split in selected:
        difficulty = "hard" if is_hard_shot(img.name) else "normal"
        shutil.copy2(img, img_out / img.name)
        info = convert_label_file(lbl, lbl_out / f"{img.stem}.txt")
        stats["poly_converted"] += info["n_poly"]
        stats["already_bbox"] += info["n_bbox"]
        stats["objects"] += info["n_obj"]
        manifest_rows.append(
            {
                "filename": img.name,
                "stem": img.stem,
                "difficulty": difficulty,
                "source_split": split,
                "n_objects": info["n_obj"],
                "n_poly_converted": info["n_poly"],
                "n_bbox_kept": info["n_bbox"],
            }
        )

    write_yaml(DST / "data.yaml", DST)

    with open(DST / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: (r["difficulty"], r["filename"])))

    for diff in ("hard", "normal"):
        di = DST / diff / "images"
        dl = DST / diff / "labels"
        di.mkdir(parents=True)
        dl.mkdir(parents=True)
        for row in manifest_rows:
            if row["difficulty"] != diff:
                continue
            shutil.copy2(img_out / row["filename"], di / row["filename"])
            shutil.copy2(lbl_out / f"{row['stem']}.txt", dl / f"{row['stem']}.txt")
        write_yaml(DST / f"data_{diff}.yaml", DST / diff)

    meta = {
        "seed": SEED,
        "n_hard": len(hard),
        "n_normal": len(normal_sel),
        "n_total": len(selected),
        "source": str(SRC),
        "stats": stats,
        "normal_selected": sorted(
            r["filename"] for r in manifest_rows if r["difficulty"] == "normal"
        ),
        "hard_selected": sorted(
            r["filename"] for r in manifest_rows if r["difficulty"] == "hard"
        ),
    }
    (DST / "build_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {DST} | Hard={len(hard)} Normal={len(normal_sel)} "
        f"poly→bbox={stats['poly_converted']}"
    )


if __name__ == "__main__":
    main()
