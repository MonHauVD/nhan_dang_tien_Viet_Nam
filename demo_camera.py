#!/usr/bin/env python3
"""
Demo nhận dạng tiền VN từ camera / ảnh bằng YOLOv8 baseline.

Chế độ:
  1) LIVE   — camera trực tiếp (pause / continue)
  2) IMAGE  — chọn ảnh hoặc thư mục (kéo thả nếu cài tkinterdnd2)

Chạy:
  python demo_camera.py
  python demo_camera.py --mode live
  python demo_camera.py --mode image --source ./anh.jpg
  python demo_camera.py --weights path/to/best.pt --conf 0.35 --camera 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT_CANDIDATES = [
    ROOT / "notebooks" / "baseline_exports" / "yolov8n_baseline_best.pt",
    ROOT / "runs" / "detect" / "vn_currency_yolov8n_baseline" / "weights" / "best.pt",
]

# Phím điều khiển (OpenCV waitKey)
KEY_QUIT = {ord("q"), 27}          # q hoặc ESC
KEY_PAUSE = {ord("p"), ord(" ")}   # p hoặc Space
KEY_CONTINUE = {ord("c")}          # c
KEY_NEXT = {ord("n"), 83, 3}       # n hoặc →
KEY_PREV = {ord("b"), 81, 2}       # b hoặc ←
KEY_SAVE = {ord("s")}
KEY_HELP = {ord("h")}


def find_default_weights() -> Path | None:
    for p in DEFAULT_WEIGHT_CANDIDATES:
        if p.exists():
            return p
    return None


def load_model(weights: Path, device: str | int):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # warm-up nhỏ
    _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device)
    return model


def draw_overlay(frame: np.ndarray, paused: bool, fps: float, conf: float, mode: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = 56
    cv2.rectangle(out, (0, 0), (w, bar_h), (20, 20, 20), -1)
    status = "PAUSED" if paused else "LIVE"
    color = (0, 200, 255) if paused else (0, 220, 0)
    cv2.putText(out, f"{mode} | {status}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"FPS:{fps:.1f}  conf>={conf:.2f}  [p/Space]=pause  [c]=continue  [s]=save  [h]=help  [q]=quit",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return out


def print_help_live() -> None:
    print(
        """
===== Phím LIVE =====
  p / Space  : tạm dừng (giữ khung hình hiện tại)
  c          : tiếp tục
  s          : lưu khung hình + prediction
  h          : in hướng dẫn
  q / ESC    : thoát
"""
    )


def print_help_image() -> None:
    print(
        """
===== Phím IMAGE =====
  n / →      : ảnh tiếp theo (nếu chọn thư mục)
  b / ←      : ảnh trước
  s          : lưu ảnh đã vẽ box
  h          : in hướng dẫn
  q / ESC    : thoát
"""
    )


def predict_frame(model, frame_bgr: np.ndarray, conf: float, imgsz: int, device) -> np.ndarray:
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    plotted = results[0].plot()  # BGR
    return plotted


def run_live(model, camera_id: int, conf: float, imgsz: int, device, save_dir: Path) -> None:
    print_help_live()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được camera id={camera_id}. Thử --camera 1 hoặc kiểm tra quyền webcam."
        )

    # Ưu tiên độ phân giải vừa phải cho mượt
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "VN Currency — LIVE"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused = False
    last_vis = None
    fps = 0.0
    t_prev = time.time()
    frame_i = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Mất tín hiệu camera.")
                break
            vis = predict_frame(model, frame, conf, imgsz, device)
            last_vis = vis
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            t_prev = now
        else:
            vis = last_vis
            if vis is None:
                ok, frame = cap.read()
                if not ok:
                    break
                vis = predict_frame(model, frame, conf, imgsz, device)
                last_vis = vis

        show = draw_overlay(vis, paused=paused, fps=fps, conf=conf, mode="LIVE")
        cv2.imshow(win, show)
        key = cv2.waitKey(1) & 0xFF

        if key in KEY_QUIT:
            break
        if key in KEY_PAUSE:
            paused = True
            print("[LIVE] PAUSED")
        elif key in KEY_CONTINUE:
            paused = False
            print("[LIVE] CONTINUE")
        elif key in KEY_SAVE and last_vis is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"live_{int(time.time())}_{frame_i:04d}.jpg"
            cv2.imwrite(str(out), last_vis)
            print(f"[SAVE] {out}")
            frame_i += 1
        elif key in KEY_HELP:
            print_help_live()

    cap.release()
    cv2.destroyAllWindows()


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in exts]
    return files


def pick_path_with_dialog() -> Path | None:
    """Chọn file hoặc thư mục bằng hộp thoại tkinter (+ kéo thả nếu có tkinterdnd2)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        print("Không dùng được tkinter:", e)
        return None

    selected: dict[str, Path | None] = {"path": None}
    dnd_ok = False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD

        root = TkinterDnD.Tk()
        dnd_ok = True
    except Exception:
        root = tk.Tk()

    root.title("Chọn ảnh / thư mục — VN Currency Demo")
    root.geometry("560x320")
    root.resizable(False, False)

    info = tk.Label(
        root,
        text="Chọn 1 ảnh hoặc 1 thư mục chứa ảnh.\nCó thể kéo thả vào vùng bên dưới (nếu hỗ trợ).",
        font=("Sans", 11),
        justify="center",
    )
    info.pack(pady=12)

    drop = tk.Label(
        root,
        text="Kéo thả ảnh / thư mục vào đây\nhoặc bấm các nút bên dưới",
        relief="groove",
        width=50,
        height=6,
        bg="#f0f0f0",
        font=("Sans", 11),
    )
    drop.pack(padx=20, pady=8, fill="both", expand=True)

    status = tk.Label(root, text="", fg="#333")
    status.pack(pady=4)

    def set_path(p: Path):
        selected["path"] = p
        status.config(text=f"Đã chọn: {p}")

    def browse_file():
        p = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả", "*.*"),
            ],
        )
        if p:
            set_path(Path(p))

    def browse_dir():
        p = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if p:
            set_path(Path(p))

    def ok():
        if selected["path"] is None:
            messagebox.showwarning("Thiếu lựa chọn", "Hãy chọn ảnh hoặc thư mục trước.")
            return
        root.destroy()

    def cancel():
        selected["path"] = None
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(pady=8)
    tk.Button(btns, text="Chọn ảnh…", width=14, command=browse_file).grid(row=0, column=0, padx=6)
    tk.Button(btns, text="Chọn thư mục…", width=14, command=browse_dir).grid(row=0, column=1, padx=6)
    tk.Button(btns, text="Bắt đầu", width=12, command=ok).grid(row=0, column=2, padx=6)
    tk.Button(btns, text="Hủy", width=10, command=cancel).grid(row=0, column=3, padx=6)

    if dnd_ok:
        def on_drop(event):
            raw = event.data.strip()
            if raw.startswith("{") and raw.endswith("}"):
                raw = raw[1:-1]
            # Một số WM trả nhiều path trong {} riêng
            parts = []
            buf = ""
            in_brace = False
            for ch in raw:
                if ch == "{":
                    in_brace = True
                    buf = ""
                elif ch == "}":
                    in_brace = False
                    parts.append(buf)
                    buf = ""
                elif ch == " " and not in_brace:
                    if buf:
                        parts.append(buf)
                        buf = ""
                else:
                    buf += ch
            if buf:
                parts.append(buf)
            cand = Path(parts[0]) if parts else Path(raw)
            if cand.exists():
                set_path(cand)
            else:
                status.config(text=f"Không đọc được: {raw}")

        drop.drop_target_register(DND_FILES)
        drop.dnd_bind("<<Drop>>", on_drop)
        status.config(text="Kéo thả đã bật (tkinterdnd2)")
    else:
        status.config(
            text="Kéo thả chưa bật — dùng nút Chọn ảnh/thư mục (pip install tkinterdnd2)"
        )

    root.mainloop()
    return selected["path"]


def run_image(model, source: Path | None, conf: float, imgsz: int, device, save_dir: Path) -> None:
    if source is None:
        source = pick_path_with_dialog()
    if source is None:
        print("Không có ảnh nào được chọn.")
        return

    images = list_images(source)
    if not images:
        raise RuntimeError(f"Không tìm thấy ảnh trong: {source}")

    print_help_image()
    print(f"Số ảnh: {len(images)}")

    win = "VN Currency — IMAGE"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    idx = 0

    while True:
        path = images[idx]
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Không đọc được ảnh: {path}")
            idx = (idx + 1) % len(images)
            continue

        vis = predict_frame(model, frame, conf, imgsz, device)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(
            overlay,
            f"IMAGE [{idx+1}/{len(images)}] {path.name}  conf>={conf:.2f}  [n]/ [b]prev  [s]save  [q]quit",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(win, overlay)

        key = cv2.waitKey(0) & 0xFF
        if key in KEY_QUIT:
            break
        if key in KEY_NEXT:
            idx = (idx + 1) % len(images)
        elif key in KEY_PREV:
            idx = (idx - 1) % len(images)
        elif key in KEY_SAVE:
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"pred_{path.stem}.jpg"
            cv2.imwrite(str(out), vis)
            print(f"[SAVE] {out}")
        elif key in KEY_HELP:
            print_help_image()

    cv2.destroyAllWindows()


def interactive_mode_menu() -> str:
    print(
        """
==============================
  Demo nhận dạng tiền VN
==============================
  1) LIVE  — camera trực tiếp
  2) IMAGE — chọn ảnh / thư mục
  q) Thoát
"""
    )
    while True:
        choice = input("Chọn chế độ [1/2/q]: ").strip().lower()
        if choice in {"1", "live", "l"}:
            return "live"
        if choice in {"2", "image", "i"}:
            return "image"
        if choice in {"q", "quit", "exit"}:
            sys.exit(0)
        print("Lựa chọn không hợp lệ.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo camera/ảnh nhận dạng tiền VN (YOLOv8)")
    p.add_argument("--mode", choices=["live", "image", "ask"], default="ask")
    p.add_argument("--weights", type=str, default=None, help="Đường dẫn best.pt")
    p.add_argument("--source", type=str, default=None, help="Ảnh hoặc thư mục (mode image)")
    p.add_argument("--camera", type=int, default=0, help="Chỉ số webcam (thường 0)")
    p.add_argument("--conf", type=float, default=0.35, help="Ngưỡng confidence")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None, help="0 / cpu / cuda:0")
    p.add_argument(
        "--save-dir",
        type=str,
        default=str(ROOT / "demo_outputs"),
        help="Thư mục lưu ảnh khi nhấn s",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights) if args.weights else find_default_weights()
    if weights is None or not weights.exists():
        raise SystemExit(
            "Không tìm thấy file weights.\n"
            "Hãy chỉ định: python demo_camera.py --weights /path/to/best.pt\n"
            f"Các vị trí mặc định đã thử:\n" + "\n".join(f"  - {p}" for p in DEFAULT_WEIGHT_CANDIDATES)
        )

    # Device
    if args.device is not None:
        device = args.device
    else:
        try:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[INFO] weights = {weights}")
    print(f"[INFO] device  = {device}")
    print(f"[INFO] conf    = {args.conf}")

    model = load_model(weights, device)
    save_dir = Path(args.save_dir)

    mode = args.mode if args.mode != "ask" else interactive_mode_menu()

    if mode == "live":
        run_live(model, args.camera, args.conf, args.imgsz, device, save_dir)
    else:
        source = Path(args.source) if args.source else None
        run_image(model, source, args.conf, args.imgsz, device, save_dir)


if __name__ == "__main__":
    main()
