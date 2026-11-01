#!/usr/bin/env bash
# Chạy trên máy server, trong thư mục dự án:
#   cd ~/Documents/dungtv/nhan_dang_tien_Viet_Nam
#   bash setup_train_remote.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> GPU"
nvidia-smi || true

echo "==> Tạo venv"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "==> Cài PyTorch CUDA (thử cu128, fallback cu126)"
if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; then
  echo "cu128 thất bại → thử cu126"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

echo "==> Cài dependencies dự án"
pip install -r notebooks/requirements.txt

echo "==> Kiểm tra CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
    print("vram :", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), "GB")
else:
    raise SystemExit("CUDA không khả dụng — dừng lại")
PY

echo
echo "OK. Tiếp theo train baseline:"
echo "  source .venv/bin/activate"
echo "  python train_baseline.py"
echo "hoặc mở notebooks/02_Train_YOLO_Baseline.ipynb"
