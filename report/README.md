# Thư mục nộp báo cáo — Nhận dạng tiền Việt Nam

## Thành phần

| File / thư mục | Mô tả |
|----------------|--------|
| `Bao_cao_Nhan_dang_tien_Viet_Nam.docx` | Báo cáo Word (bìa, mục lục, đủ các mục theo yêu cầu học phần) |
| `notebooks/01_EDA_Bao_cao.ipynb` | EDA + thống kê nguồn ảnh train (Roboflow & camera thật) |
| `notebooks/02_Qua_trinh_Train_va_Cai_tien.ipynb` | Ghi lại baseline → v2 → v3 → v4 |
| `notebooks/03_So_sanh_Baseline_vs_Final.ipynb` | So sánh công bằng baseline vs final (metric, conf sweep, định tính) |
| `figures/` | Biểu đồ dùng trong báo cáo |
| `exports/` | CSV/JSON chỉ số so sánh |
| `build_report_docx.py` | Script tái tạo file `.docx` nếu cần chỉnh sửa hàng loạt |

## Cách mở mục lục trong Word

1. Mở `Bao_cao_Nhan_dang_tien_Viet_Nam.docx`
2. Vào trang **MỤC LỤC** → chọn vùng mục lục → chuột phải → **Cập nhật trường** → **Cập nhật toàn bộ mục lục**

## Chạy notebook

Từ thư mục gốc dự án (đã cài dependency YOLO / pandas / matplotlib):

```bash
cd nhan_dang_tien_Viet_Nam
jupyter notebook report/notebooks/
```

Notebook tự dò đường dẫn về `notebooks/*_exports` và `datasets/`.

## Tái tạo file Word

```bash
./.venv_report/bin/python report/build_report_docx.py
```
