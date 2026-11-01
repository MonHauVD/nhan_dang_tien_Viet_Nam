# Ảnh cam thật (`real_image/`) — gắn nhãn & đưa vào train v4

## Bạn đã có gì
- 18 ảnh `1280×720` chụp bằng **đúng camera demo** (mờ / nền rối)
- Đặt tên chuẩn: `{mệnh_giá}_{front|back}.jpg`  
  ví dụ `200000_front.jpg`, `5000_back.jpg`

Đây là dữ liệu **ROI cao nhất** lúc này (đúng domain cam), dù chưa có bbox.

## Đã chạy sẵn
```bash
python label_real_images.py --conf 0.12 --imgsz 800
```

Kết quả:
- Labels YOLO: `datasets/real_cam_yolo/labels/`
- Ảnh review (box vẽ sẵn): `datasets/real_cam_yolo/review/`
- **Class luôn theo tên file**; model chỉ gợi ý vị trí box

Thống kê lần auto-label với v3:
| status | ý nghĩa | số lượng |
|--------|---------|----------|
| `ok` | model đoán đúng class + có box | 13 |
| `override_class` | có box nhưng class sai → đã ép đúng theo tên file | 4 (`1000_back`, `200000_*`, `500000_front`) |
| `fail_no_det_fullframe_guess` | không detect → box tạm toàn khung | 1 (`50000_front`) |

→ Các file `override`/`fail` **bắt buộc xem review** và sửa box nếu lệch.

## Việc bạn cần làm (5–10 phút)
1. Mở thư mục `datasets/real_cam_yolo/review/`
2. Kiểm tra từng ảnh; ưu tiên:
   - `200000_front.jpg`, `200000_back.jpg`
   - `5000_*`, `20000_*`
   - `50000_front.jpg` (đang guess)
3. Nếu box lệch: sửa file tương ứng trong `datasets/real_cam_yolo/labels/`  
   Format 1 dòng: `class_id xc yc w h` (chuẩn hóa 0–1)  
   class_id:  
   `1000=0, 10000=1, 100000=2, 2000=3, 20000=4, 200000=5, 5000=6, 50000=7, 500000=8`

Hoặc import `datasets/real_cam_yolo/images` + labels vào Roboflow để chỉnh box cho nhanh.

## Train v4 (đã merge real_cam + blur/low-res/rotate)
`train_v4.py` sẽ nhân bản ảnh cam thật (blur, low-res, xoay 90°) rồi trộn vào train.

```bash
# sync cả real_image + datasets/real_cam_yolo + train_v4.py lên server
rsync -avh --exclude '.venv' --exclude '.git' --exclude 'runs' \
  ~/Desktop/Note/Hoc_Tap/Nhan_dang_mau/nhan_dang_tien_Viet_Nam/ \
  server_02@10.87.25.8:/home/server_02/Documents/dungtv/nhan_dang_tien_Viet_Nam/

# trên server
conda activate vn_currency
cd ~/Documents/dungtv/nhan_dang_tien_Viet_Nam
nohup python -u train_v4.py --epochs 60 --batch 16 --imgsz 800 > logs/train_v4.log 2>&1 &
```

## Gợi ý thu thập thêm (nếu còn thời gian)
Mỗi class yếu thêm 10–20 ảnh cam thật:
- `200000` (nhiều góc, gần/xa)
- cặp `5000` vs `20000` cạnh nhau
- `50000` / `100000` khi tay hơi run / hơi mờ
