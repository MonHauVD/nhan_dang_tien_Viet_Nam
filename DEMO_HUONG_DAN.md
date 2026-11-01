# Hướng dẫn demo nhận dạng tiền VN (camera / ảnh)

Script: `demo_camera.py`  
Model khuyến nghị (mới nhất): `notebooks/v4_exports/yolov8s_v4_best.pt`

---

## 1. Cài đặt (máy local)

Trong thư mục dự án:

```bash
cd ~/Desktop/Note/Hoc_Tap/Nhan_dang_mau/nhan_dang_tien_Viet_Nam

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Torch (máy có NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Nếu chỉ CPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Demo cần OpenCV CÓ GUI (không dùng bản headless)
pip install ultralytics opencv-python pillow numpy

# Tuỳ chọn: kéo thả ảnh vào cửa sổ chọn file
pip install tkinterdnd2
```

Linux: nếu thiếu tkinter:

```bash
sudo apt update
sudo apt install python3-tk
```

Kiểm tra nhanh:

```bash
python - <<'PY'
import cv2, torch
from ultralytics import YOLO
from pathlib import Path
print("cv2", cv2.__version__)
print("cuda", torch.cuda.is_available())
print("weights ok", Path("notebooks/v4_exports/yolov8s_v4_best.pt").exists())
PY
```

---

## 2. Chạy demo (bản mới nhất)

### Lệnh khuyến nghị — LIVE camera

```bash
source .venv/bin/activate
cd ~/Desktop/Note/Hoc_Tap/Nhan_dang_mau/nhan_dang_tien_Viet_Nam

python demo_camera.py \
  --weights notebooks/v4_exports/yolov8s_v4_best.pt \
  --mode live \
  --conf 0.28 \
  --imgsz 800 \
  --hold-seconds 1
```

Một dòng:

```bash
python demo_camera.py --weights notebooks/v4_exports/yolov8s_v4_best.pt --mode live --conf 0.28 --imgsz 800 --hold-seconds 1
```

Giải thích nhanh tham số:

| Tham số | Ý nghĩa |
|---------|---------|
| `--weights` | Checkpoint final YOLOv8s v4 |
| `--mode live` | Webcam trực tiếp |
| `--conf 0.28` | Ngưỡng tin cậy (mặc định script cũng ~0.28) |
| `--imgsz 800` | Kích thước suy luận (khớp hướng fine-tune v3/v4) |
| `--hold-seconds 1` | Giữ box ~1 giây khi model miss tạm thời (giảm nhấp nháy) |

### Menu tương tác

```bash
python demo_camera.py --weights notebooks/v4_exports/yolov8s_v4_best.pt
```

Menu hỏi:

1. **LIVE** — camera  
2. **IMAGE** — chọn ảnh / thư mục  

### Các lệnh khác

```bash
# Đổi webcam nếu 0 không mở được
python demo_camera.py --weights notebooks/v4_exports/yolov8s_v4_best.pt --mode live --camera 1 --conf 0.28 --imgsz 800 --hold-seconds 1

# Một ảnh
python demo_camera.py --weights notebooks/v4_exports/yolov8s_v4_best.pt --mode image --source /đường/dẫn/anh.jpg --conf 0.28 --imgsz 800

# Cả thư mục ảnh
python demo_camera.py --weights notebooks/v4_exports/yolov8s_v4_best.pt --mode image --source /đường/dẫn/thu_muc_anh --conf 0.28 --imgsz 800

# So sánh nhanh với baseline (không khuyến nghị cho demo lớp)
python demo_camera.py --weights notebooks/baseline_exports/yolov8n_baseline_best.pt --mode live --conf 0.28 --imgsz 640
```

---

## 3. Phím điều khiển

### LIVE (camera)
| Phím | Chức năng |
|------|-----------|
| `p` / `Space` | Pause (giữ khung hiện tại) |
| `c` | Continue (tiếp tục nhận dạng) |
| `s` | Lưu khung hình đã vẽ box vào `demo_outputs/` |
| `h` | In hướng dẫn |
| `q` / `ESC` | Thoát |

### IMAGE (ảnh / thư mục)
| Phím | Chức năng |
|------|-----------|
| `n` / `→` | Ảnh tiếp theo |
| `b` / `←` | Ảnh trước |
| `s` | Lưu ảnh prediction |
| `h` | In hướng dẫn |
| `q` / `ESC` | Thoát |

Chọn ảnh: bấm **Chọn ảnh…** / **Chọn thư mục…**, hoặc **kéo thả** vào ô (cần `tkinterdnd2`).

---

## 4. Lưu ý khi demo trước lớp

- Ánh sáng đủ, tờ tiền nằm gọn trong khung, tránh bóng tay che quá nhiều.  
- Lệnh chuẩn: `--conf 0.28 --imgsz 800 --hold-seconds 1`. Nếu bỏ sót nhiều → hạ `conf` còn `0.20–0.25`; nếu nhiều box ảo → tăng `0.40–0.50`.  
- Webcam không mở được: thử `--camera 1`. Trên Linux có thể cần quyền user trong group `video`.  
- Ảnh đã lưu nằm ở `demo_outputs/` — dùng được cho slide / báo cáo.

---

## 5. Cấu trúc liên quan

```text
nhan_dang_tien_Viet_Nam/
  demo_camera.py                      # script demo
  DEMO_HUONG_DAN.md                   # file này
  notebooks/v4_exports/
    yolov8s_v4_best.pt                # checkpoint khuyến nghị (final)
  notebooks/baseline_exports/
    yolov8n_baseline_best.pt          # baseline (so sánh)
  demo_outputs/                       # ảnh lưu khi nhấn s
```
