# Kết quả validation — Baseline smoke test

## Thông tin run

- Run: `baseline-smoke`.
- Dataset: Roboflow version 1.
- Model: YOLOv8n.
- GPU: NVIDIA T4.
- Epochs: 1.
- Validation images: 439.
- Validation instances: 1.074.
- Model summary sau fuse: 73 layers, 3.134.651 parameters, 8,6 GFLOPs.
- Artifact: `MyDrive/nhan_dang_mau/runs/baseline-smoke`.

## Metric tổng thể

| Precision | Recall | F1 tính từ P/R | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0,428 | 0,476 | 0,451 | 0,431 | 0,342 |

F1 được tính bằng `2PR/(P+R)`. Đây là smoke test 1 epoch, không dùng làm kết quả baseline cuối cùng.

## Metric theo class

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 120 | 120 | 0,548 | 0,404 | 0,480 | 0,404 |
| 10.000 | 122 | 122 | 0,275 | 0,546 | 0,242 | 0,184 |
| 100.000 | 118 | 118 | 0,792 | 0,797 | 0,877 | 0,692 |
| 2.000 | 114 | 114 | 0,355 | 0,491 | 0,395 | 0,329 |
| 20.000 | 120 | 120 | 0,206 | 0,383 | 0,150 | 0,106 |
| 200.000 | 123 | 123 | 0,302 | 0,293 | 0,189 | 0,121 |
| 5.000 | 123 | 123 | 0,728 | 0,565 | 0,637 | 0,541 |
| 50.000 | 119 | 119 | 0,352 | 0,731 | 0,668 | 0,536 |
| 500.000 | 115 | 115 | 0,297 | 0,0696 | 0,242 | 0,167 |

## Tốc độ

| Preprocess | Inference | Loss | Postprocess | Tổng pipeline xấp xỉ |
|---:|---:|---:|---:|---:|
| 0,8 ms | 4,2 ms | 0,0 ms | 3,1 ms | 8,1 ms/ảnh |

Tốc độ này chỉ dùng để xác nhận pipeline; phép đo cuối phải thực hiện trong cùng môi trường cho baseline và phương án cải tiến.

## Kết luận

- Pipeline tải dữ liệu, class mapping, split, train, validation và lưu artifact hoạt động đúng.
- Số ảnh và số instance validation khớp audit.
- Metric thấp ở một số class là bình thường sau đúng 1 epoch.
- Không dùng smoke-test weights làm mô hình cuối.

