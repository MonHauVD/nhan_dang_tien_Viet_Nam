#!/usr/bin/env python3
"""Chỉnh bounding box YOLO bằng Tkinter (không dùng OpenCV HighGUI — tránh lỗi Qt trên Linux).

Phím / chuột:
  Kéo chuột trái : vẽ lại box
  n              : ảnh tiếp (auto-save nếu đã sửa)
  b              : ảnh trước
  s              : lưu
  a              : co box 8% mỗi cạnh
  r              : load lại box đã lưu
  q / Esc        : thoát

Chạy:
  python edit_bbox_gui.py
  python3 edit_bbox_gui.py --dir datasets/real_cam_yolo
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "datasets" / "real_cam_yolo"
CLASS_NAMES = ["1000", "10000", "100000", "2000", "20000", "200000", "5000", "50000", "500000"]
MAX_DISP_W = 1100


def yolo_to_xyxy(xc, yc, bw, bh, w, h):
    return [
        (xc - bw / 2) * w,
        (yc - bh / 2) * h,
        (xc + bw / 2) * w,
        (yc + bh / 2) * h,
    ]


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h


def load_label(path: Path, w: int, h: int):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return None
    line = path.read_text(encoding="utf-8").strip().splitlines()[0]
    parts = line.split()
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    return {"cls": cls, "xyxy": yolo_to_xyxy(xc, yc, bw, bh, w, h)}


def save_label(path: Path, cls: int, xyxy, w: int, h: int):
    xc, yc, bw, bh = xyxy_to_yolo(*xyxy, w, h)
    path.write_text(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def shrink_box(xyxy, ratio=0.08):
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    return [x1 + bw * ratio, y1 + bh * ratio, x2 - bw * ratio, y2 - bh * ratio]


class BBoxEditor(tk.Tk):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.title("Edit YOLO bbox (Tkinter)")
        self.data_dir = data_dir
        self.img_dir = data_dir / "images"
        self.lbl_dir = data_dir / "labels"
        self.review_dir = data_dir / "review"
        self.review_dir.mkdir(parents=True, exist_ok=True)

        self.paths = sorted(
            p for p in self.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise SystemExit(f"Không có ảnh trong {self.img_dir}")

        self.idx = 0
        self.dirty = False
        self.box = None  # xyxy trên ảnh gốc
        self.cls = 0
        self.pil_img = None
        self.disp_scale = 1.0
        self.tk_img = None
        self.drag_start = None  # (x,y) trên canvas

        # UI
        top = tk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        self.status = tk.Label(top, text="", anchor="w", font=("Sans", 11))
        self.status.pack(side="left", fill="x", expand=True)

        btns = tk.Frame(self)
        btns.pack(fill="x", padx=8, pady=4)
        for text, cmd in [
            ("Prev (b)", self.prev_img),
            ("Next (n)", self.next_img),
            ("Save (s)", self.save),
            ("Shrink 8% (a)", self.shrink),
            ("Reload (r)", self.reload),
            ("Quit (q)", self.on_quit),
        ]:
            tk.Button(btns, text=text, command=cmd).pack(side="left", padx=3)

        tip = tk.Label(
            self,
            text="Kéo chuột trái trên ảnh để vẽ box sát tờ tiền hơn. Class giữ theo tên file.",
            anchor="w",
            fg="#333",
        )
        tip.pack(fill="x", padx=8)

        self.canvas = tk.Canvas(self, bg="#222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.bind("<Key-s>", lambda e: self.save())
        self.bind("<Key-n>", lambda e: self.next_img())
        self.bind("<Key-b>", lambda e: self.prev_img())
        self.bind("<Key-a>", lambda e: self.shrink())
        self.bind("<Key-r>", lambda e: self.reload())
        self.bind("<Key-q>", lambda e: self.on_quit())
        self.bind("<Escape>", lambda e: self.on_quit())

        self.load_image()
        self.focus_force()

    def label_path(self) -> Path:
        return self.lbl_dir / f"{self.paths[self.idx].stem}.txt"

    def load_image(self):
        path = self.paths[self.idx]
        self.pil_img = Image.open(path).convert("RGB")
        w, h = self.pil_img.size
        self.disp_scale = min(1.0, MAX_DISP_W / max(w, 1))

        lab = load_label(self.label_path(), w, h)
        if lab:
            self.cls = lab["cls"]
            self.box = lab["xyxy"]
        else:
            denom = path.stem.split("_")[0]
            self.cls = CLASS_NAMES.index(denom) if denom in CLASS_NAMES else 0
            self.box = [w * 0.2, h * 0.2, w * 0.8, h * 0.8]

        self.dirty = False
        self.drag_start = None
        self.redraw()

    def redraw(self):
        assert self.pil_img is not None
        w, h = self.pil_img.size
        disp = self.pil_img.copy()
        draw = ImageDraw.Draw(disp)
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)

        name = CLASS_NAMES[self.cls] if 0 <= self.cls < len(CLASS_NAMES) else str(self.cls)
        flag = "MODIFIED*" if self.dirty else "saved"
        title = f"[{self.idx+1}/{len(self.paths)}] {self.paths[self.idx].name} | class={name} | {flag}"
        self.status.config(text=title)

        if self.disp_scale != 1.0:
            nw, nh = int(w * self.disp_scale), int(h * self.disp_scale)
            disp = disp.resize((nw, nh), Image.Resampling.BILINEAR)

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.size[0], height=disp.size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # vẽ lại box trên canvas cho rõ khi đang kéo
        if self.box is not None:
            x1, y1, x2, y2 = [v * self.disp_scale for v in self.box]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff66", width=2)

    def canvas_to_img(self, x, y):
        return x / self.disp_scale, y / self.disp_scale

    def on_press(self, event):
        self.drag_start = self.canvas_to_img(event.x, event.y)

    def on_drag(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.canvas_to_img(event.x, event.y)
        self.box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        self.dirty = True
        self.redraw()

    def on_release(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.canvas_to_img(event.x, event.y)
        if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
            self.box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            self.dirty = True
        self.drag_start = None
        self.redraw()

    def save(self):
        assert self.pil_img is not None and self.box is not None
        w, h = self.pil_img.size
        save_label(self.label_path(), self.cls, self.box, w, h)

        # review
        rev = self.pil_img.copy()
        draw = ImageDraw.Draw(rev)
        draw.rectangle(self.box, outline=(0, 220, 0), width=3)
        name = CLASS_NAMES[self.cls]
        draw.text((12, 10), f"GT:{name} [edited]", fill=(0, 200, 0))
        rev.save(self.review_dir / f"{self.paths[self.idx].stem}.jpg", quality=92)

        self.dirty = False
        self.redraw()
        print(f"[SAVE] {self.label_path()}")

    def shrink(self):
        if self.box:
            self.box = shrink_box(self.box, 0.08)
            self.dirty = True
            self.redraw()

    def reload(self):
        self.load_image()

    def _autosave_if_needed(self):
        if self.dirty:
            self.save()

    def next_img(self):
        self._autosave_if_needed()
        self.idx = (self.idx + 1) % len(self.paths)
        self.load_image()

    def prev_img(self):
        self._autosave_if_needed()
        self.idx = (self.idx - 1) % len(self.paths)
        self.load_image()

    def on_quit(self):
        if self.dirty:
            if messagebox.askyesno("Chưa lưu", "Lưu ảnh hiện tại trước khi thoát?"):
                self.save()
        edited = {"n_images": len(self.paths), "dir": str(self.data_dir)}
        (self.data_dir / "edited_ok.json").write_text(json.dumps(edited, indent=2), encoding="utf-8")
        self.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    args = ap.parse_args()
    data_dir = Path(args.dir)
    if not (data_dir / "images").exists():
        raise SystemExit(f"Thiếu {data_dir / 'images'}")

    # Đảm bảo có tkinter
    try:
        import tkinter  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Thiếu tkinter. Cài: sudo apt install python3-tk\n" + str(e)
        )

    app = BBoxEditor(data_dir)
    app.mainloop()
    print("Xong. Có thể chạy train_v4.py")


if __name__ == "__main__":
    main()
