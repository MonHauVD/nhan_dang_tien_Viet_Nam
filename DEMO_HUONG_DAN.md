# Hướng dẫn demo nhận dạng tiền VN (camera / ảnh)

Script: `demo_camera.py`  
Model mặc định: `notebooks/baseline_exports/yolov8n_baseline_best.pt`

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
print("cv2", cv2.__version__)
print("cuda", torch.cuda.is_available())
print("weights ok", __import__("pathlib").Path("notebooks/baseline_exports/yolov8n_baseline_best.pt").exists())
PY
```

---

## 2. Chạy demo

```bash
source .venv/bin/activate
cd ~/Desktop/Note/Hoc_Tap/Nhan_dang_mau/nhan_dang_tien_Viet_Nam

python demo_camera.py
```

Menu hỏi bạn chọn:

1. **LIVE** — camera  
2. **IMAGE** — chọn ảnh / thư mục  

### Lệnh tắt

```bash
# Camera trực tiếp
python demo_camera.py --mode live --camera 0 --conf 0.35

# Một ảnh
python demo_camera.py --mode image --source /đường/dẫn/anh.jpg

# Cả thư mục ảnh
python demo_camera.py --mode image --source /đường/dẫn/thu_muc_anh

# Chỉ định weights khác
python demo_camera.py --weights runs/detect/vn_currency_yolov8n_baseline/weights/best.pt
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
- `--conf 0.35` là mặc định hợp lý; nếu bỏ sót nhiều → hạ còn `0.25`; nếu nhiều box ảo → tăng `0.45`.  
- Webcam không mở được: thử `--camera 1`. Trên Linux có thể cần quyền user trong group `video`.  
- Ảnh đã lưu nằm ở `demo_outputs/` — dùng được cho slide / báo cáo.

---

## 5. Cấu trúc liên quan

```text
nhan_dang_tien_Viet_Nam/
  demo_camera.py                 # script demo
  DEMO_HUONG_DAN.md              # file này
  notebooks/baseline_exports/
    yolov8n_baseline_best.pt     # checkpoint dùng mặc định
  demo_outputs/                  # ảnh lưu khi nhấn s
```
