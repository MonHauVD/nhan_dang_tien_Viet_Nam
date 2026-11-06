# Báo cáo — Nhận dạng tiền Việt Nam

## Thành phần

| File / thư mục | Mô tả |
|----------------|--------|
| `Bao_cao_Nhan_dang_tien_Viet_Nam.docx` | Báo cáo Word |
| `notebooks/01_EDA_Bao_cao.ipynb` | Phân tích sơ bộ dữ liệu (EDA) |
| `notebooks/02_Qua_trinh_Train_va_Cai_tien.ipynb` | Quá trình huấn luyện và cải tiến (baseline → v4) |
| `notebooks/03_So_sanh_Baseline_vs_Final.ipynb` | So sánh định lượng baseline và mô hình cuối trên `fair_eval_hardmix` |
| `figures/` | Hình minh họa trong báo cáo |
| `exports/` | Bảng chỉ số (CSV/JSON) |
| `build_report_docx.py` | Script sinh lại file Word từ nội dung đã chuẩn hóa |

Notebook đọc trọng số và kết quả từ `notebooks/*_exports` cùng `datasets/` ở thư mục gốc dự án.
