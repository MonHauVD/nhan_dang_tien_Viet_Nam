# Kết quả huấn luyện baseline E0

Nguồn: `results.csv` của run `baseline`, huấn luyện 100 epoch trên NVIDIA T4. Dataset là Roboflow version 1; model `yolov8n.pt`; ảnh 640×640; batch 16; seed 42; augmentation của Ultralytics trainer đã tắt.

Biểu đồ đường học: `reports/figures/baseline_training_curves.png`.

## Kiểm tra file kết quả

| Hạng mục | Kết quả |
|---|---:|
| Số dòng epoch | 100 |
| Khoảng epoch | 1–100 |
| Số cột | 15 |
| Giá trị NaN hoặc vô hạn | 0 |
| Thời gian tích lũy tại epoch 100 | 1.808,69 giây, xấp xỉ 30 phút 9 giây |

## Epoch tốt nhất trên validation

Epoch 95 có mAP50-95 cao nhất. Đây cũng là epoch có fitness cao nhất nếu áp dụng công thức Ultralytics Detection `0,1 × mAP50 + 0,9 × mAP50-95`.

| Epoch | Precision | Recall | F1 tính từ P/R | mAP50 | mAP50-95 | Val box loss | Val cls loss | Val DFL loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **95** | **0,98989** | **0,98338** | **0,98662** | **0,99406** | **0,91018** | 0,42408 | 0,29086 | 0,89604 |
| 100 | 0,99099 | 0,98366 | 0,98731 | 0,99407 | 0,90958 | 0,42350 | 0,29302 | 0,89722 |

F1 trong bảng được tính bằng `2PR / (P + R)` từ Precision và Recall tổng thể trong CSV; đây không phải một cột do Ultralytics xuất trực tiếp.

## Nhận xét

- Run hoàn tất đủ 100 epoch và không có giá trị lỗi trong CSV.
- mAP50-95 tăng từ 0,75469 ở epoch 1 lên cực đại 0,91018 ở epoch 95.
- Epoch 100 chỉ thấp hơn epoch 95 là 0,00060 mAP50-95. Sai khác rất nhỏ, cho thấy kết quả đã gần hội tụ ở cuối quá trình huấn luyện.
- Train loss tiếp tục giảm tới cuối run, trong khi validation loss dao động nhẹ ở các epoch cuối. Có một khoảng cách train/validation loss, vì vậy không nên dùng `last.pt` làm checkpoint chính chỉ vì nó là epoch cuối.
- Checkpoint dùng cho các bước tiếp theo là `runs/baseline/weights/best.pt`. Validation lại checkpoint này đã hoàn tất trên 439 ảnh và 1.074 instances.
- Các con số trên là validation metric. Chưa dùng chúng làm kết quả test chính thức và chưa chọn confidence threshold.

## Xác minh `best.pt` trên validation

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 439 | 1.074 | 0,990 | 0,983 | 0,994 | 0,910 |
| 1000 | 120 | 120 | 0,997 | 1,000 | 0,995 | 0,955 |
| 10000 | 122 | 122 | 0,965 | 0,951 | 0,990 | 0,859 |
| 100000 | 118 | 118 | 1,000 | 0,991 | 0,995 | 0,898 |
| 2000 | 114 | 114 | 0,994 | 0,982 | 0,995 | 0,954 |
| 20000 | 120 | 120 | 0,980 | 0,983 | 0,994 | 0,890 |
| 200000 | 123 | 123 | 1,000 | 0,989 | 0,995 | 0,894 |
| 5000 | 123 | 123 | 0,992 | 0,996 | 0,994 | 0,946 |
| 50000 | 119 | 119 | 0,992 | 0,975 | 0,995 | 0,892 |
| 500000 | 115 | 115 | 0,989 | 0,983 | 0,994 | 0,903 |

Giá trị tổng thể chính xác do cell trả về: Precision 0,98988196; Recall 0,98335840; F1 0,98660940; mAP50 0,99405322; mAP50-95 0,90999083. Chênh lệch mAP50-95 so với dòng epoch 95 trong CSV là khoảng 0,00019 và không làm thay đổi kết luận.

Tốc độ trên NVIDIA T4: 1,6 ms preprocess; 5,0 ms inference; 1,5 ms postprocess; tổng xấp xỉ 8,1 ms/ảnh. Giá trị `loss` 0,008 ms/ảnh không đáng kể.

Các class có mAP50-95 thấp nhất là `10000` (0,859), `20000` (0,890), `50000` (0,892) và `200000` (0,894). Đây là các class cần ưu tiên khi xem confusion matrix và phân tích lỗi.

## Cảnh báo định dạng annotation

Ultralytics báo validation có 338 segment và 1.074 box. Vì số segment không bằng số box, thư viện loại phần segment và chỉ dùng box cho bài toán detection. Run validation vẫn hoàn tất đủ 439 ảnh và 1.074 instances.

Cảnh báo không làm hỏng checkpoint hoặc metric detection hiện tại, nhưng cho thấy YOLO export chứa hỗn hợp dòng box và dòng polygon. Không sửa dataset theo phạm vi đã chốt. Cần giữ cùng export cho E0/E1 và ghi hạn chế này trong báo cáo; có thể chạy thống kê chỉ đọc trên các file label để xác định số dòng polygon ở từng split.

Tham chiếu kỹ thuật: [mã nguồn Ultralytics xử lý cảnh báo](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py) và [định dạng nhãn segmentation của Ultralytics](https://docs.ultralytics.com/datasets/segment/).

Thống kê chỉ đọc trên YOLO export:

| Split | Object rows | Box rows | Polygon rows | Tỷ lệ polygon | File có polygon | Tỷ lệ ảnh/file có polygon |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3.832 | 2.689 | 1.143 | 29,83% | 479 | 31,16% |
| Validation | 1.074 | 736 | 338 | 31,47% | 134 | 30,52% |
| Test | 558 | 385 | 173 | 31,00% | 68 | 31,19% |
| **Tổng** | **5.464** | **3.810** | **1.654** | **30,27%** | **681** | **31,04%** |

Các polygon row mẫu đều có 11 giá trị, tương ứng class ID và 5 cặp tọa độ. Tổng object row của export là 5.464, ít hơn tổng annotation trong metadata nguồn 1 đơn vị; train và validation khớp, còn test export có 558 so với 559 trong metadata nguồn. Không đủ bằng chứng để quy nguyên nhân, vì vậy ghi đây là chênh lệch export cần công bố và không tự sửa.

## Môi trường tái lập

| Thành phần | Giá trị |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| GPU | Tesla T4 |
| Ultralytics | 8.4.188 |
| Dataset version | 1 |
| Model | yolov8n.pt |
| Image size | 640 |
| Epochs | 100 |
| Batch | 16 |
| Patience | 20 |
| Seed | 42 |
| Workers | 2 |

## Thông tin chưa có trong CSV

`results.csv` không chứa Exact Count Accuracy, Exact Total Accuracy hoặc MAE tổng tiền. Confusion matrix, metadata môi trường và các artifact chuẩn đã được xác nhận có trên Drive. Các metric nghiệp vụ sẽ được tính sau khi chọn confidence threshold trên validation.
