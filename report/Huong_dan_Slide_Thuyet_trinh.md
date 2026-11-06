# Hướng dẫn thiết kế slide thuyết trình

**Đề tài:** Phát hiện và nhận dạng mệnh giá tiền giấy/polymer Việt Nam bằng YOLOv8  
**Đối tượng:** Giảng viên học phần Nhận dạng mẫu (trình độ thạc sĩ)  
**Độ dài mục tiêu:** **20 slide** (cho phép 18–22 nếu cần gộp/tách nhẹ)  
**Thời lượng gợi ý:** 12–15 phút trình bày + Q&A  

Tài liệu này dành cho **agent thiết kế slide**. Chỉ dùng nội dung dưới đây; không thêm câu hướng dẫn chạy code, cài đặt, hay ghi chú nội bộ nhóm.

---

## 0. Yêu cầu chung cho agent thiết kế

### Giọng điệu & nội dung
- Học thuật, khách quan, ngắn gọn; mỗi slide **một ý chính**.
- Số liệu ưu tiên từ `fair_eval_hardmix` (ảnh chụp thực). Roboflow chỉ nêu như miền studio gần bão hòa.
- Thuật ngữ nhóm ảnh: **Hard** / **Normal** — **không** nhắc tên file `WIN*`.
- Không dùng emoji, không dùng lời thoại kiểu “chúng ta sẽ demo…”, không checklist “cài gì / chạy thế nào”.

### Bố cục hình ảnh
- Ưu tiên ảnh **3 cột so sánh**: `Nhãn thật (GT)` | `Baseline A0` | `Final v4`.
- Có thể thêm hàng ảnh Hard (khó) và Normal (thuận lợi) để đối chiếu.
- Chú thích hình dạng: *Hình x.y — mô tả ngắn, protocol (imgsz=640, conf=0.25)*.

### Nguồn asset sẵn có (dùng trước)
| File | Dùng cho |
|------|----------|
| `report/figures/fig_val_compare_overview.png` | So sánh chỉ số ALL (baseline vs final) |
| `report/figures/fig_test_compare_overview.png` | mAP50 Hard vs Normal |
| `report/figures/fig_val_AP50_per_class.png` | AP50 theo mệnh giá |
| `report/figures/fig_conf_sweep_val.png` | Quét ngưỡng tin cậy |
| `report/figures/fig_train_curves.png` | Đường cong học |
| `report/figures/fig_qualitative_val.png` | So sánh định tính Hard |
| `report/figures/fig_qualitative_real_cam.png` | Thêm ví dụ Hard |
| `report/exports/compare_summary.json` | Số liệu khóa |
| `datasets/fair_eval_hardmix/hard/images/` | Ảnh Hard để ghép GT–pred |
| `datasets/fair_eval_hardmix/normal/images/` | Ảnh Normal |
| `datasets/fair_eval_hardmix/*/labels/` | Nhãn GT bbox |

### Asset nên tạo thêm (nếu thiếu)
Agent thiết kế (hoặc pipeline phụ) nên tạo **2–4 tấm collage**:
1. Hard — 1 ảnh nhiều tờ: GT / Baseline / Final (overlay bbox + class).
2. Hard — 1 ảnh lóa sáng hoặc xoay: GT / Baseline / Final.
3. Normal — 1 ảnh điều kiện tốt: GT / Baseline / Final (cho thấy baseline cũng ổn trên Normal).
4. (Tuỳ chọn) Bảng lỗi điển hình: nhầm lớp 5k↔20k hoặc bỏ sót trên Hard.

Protocol vẽ pred: `imgsz=640`, `conf=0.25`; weights:
- Baseline: `notebooks/baseline_exports/yolov8n_baseline_best.pt`
- Final: `notebooks/v4_exports/yolov8s_v4_best.pt`

### Phong cách hình thức (slide)
- Nền sáng, chữ tối, accent 1 màu trung tính (xanh đậm / xám than) — tránh tím gradient “AI default”.
- Font rõ: tiêu đề lớn, thân ngắn; tránh đoạn văn dài.
- Logo/HV nếu có thì nhỏ góc; không chồng badge lên ảnh minh họa.

### Cấu trúc kể chuyện (narrative)
```
Bài toán → Dữ liệu & miền khó → Baseline → Cải tiến →
Đánh giá công bằng (Hard/Normal) → Minh chứng ảnh →
Hạn chế → Kết luận
```

---

## 1. Khung slide từng trang (20 trang)

### Slide 1 — Trang bìa
**Tiêu đề:** Phát hiện và nhận dạng mệnh giá tiền Việt Nam bằng YOLOv8  
**Phụ đề:** Bài tập lớn — Học phần Nhận dạng mẫu  
**Thông tin:** Nhóm 08 — Lớp M26CQKH02-B | HVCNBCVT  
**Thành viên (nhỏ):** Hoàng Minh Đức Anh · Trần Việt Dũng · Vũ Hoàng Lâm · Lê Khánh Minh  
**Visual:** 1 ảnh montage tờ tiền VN (trang trí, không cần bbox).

---

### Slide 2 — Nội dung trình bày
**Mục tiêu slide:** Lộ trình 7–8 mục, không chi tiết kỹ thuật.  
**Bullet gợi ý:**
1. Bài toán và mục tiêu  
2. Dữ liệu & phân tích sơ bộ  
3. Baseline YOLOv8n  
4. Các vòng cải tiến (v2→v4)  
5. Protocol đánh giá trên ảnh thực (Hard / Normal)  
6. Kết quả định lượng & định tính  
7. Hạn chế và hướng phát triển  

---

### Slide 3 — Phát biểu bài toán
**Ý chính:** Object detection đa lớp: định vị + gán mệnh giá.  
**Nội dung:**
- Đầu vào: ảnh RGB chứa 0–nhiều tờ tiền  
- Đầu ra: danh sách `(bbox, class, confidence)`  
- 9 lớp: 1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k (chưa có 500đ)  
**Visual:** Sơ đồ nhỏ Input → YOLO → Boxes + Labels.

---

### Slide 4 — Mục tiêu & phạm vi
**Bullet:**
- Xây pipeline tái lập: dữ liệu → baseline → cải tiến → đánh giá  
- So sánh khách quan baseline vs mô hình cuối trên miền gần thực tế  
- Demo camera minh họa suy luận thời gian thực  
**Phạm vi / giả định (nhỏ):** không phát hiện tiền giả; không OCR sê-ri; điều kiện cực đoan thuộc nhóm Hard.

---

### Slide 5 — Nguồn dữ liệu
**Bảng nhỏ:**

| Nguồn | Vai trò | Quy mô |
|-------|---------|--------|
| Roboflow Vietnamese Currency | Huấn luyện / theo dõi | ~1537 / 439 / 218 |
| Ảnh camera thật (`real_cam_yolo`) | Fine-tune thích nghi miền | ~18 gốc |
| `fair_eval_hardmix` | Đánh giá độc lập | 40 (16 Hard + 24 Normal) |

**Ghi chú 1 dòng:** Roboflow letterbox nền trắng → miền studio “dễ”; cần tập ảnh thực để đo độ bền.

---

### Slide 6 — EDA: phân phối & nhận xét then chốt
**Ý chính:** Hồ sơ dữ liệu trước train.  
**Bullet (chọn 4–5 ý, không nhồi số):**
- Đơn vị quan sát: mức ảnh vs mức đối tượng  
- Nhãn lẫn bbox / polygon → thống nhất bbox  
- Letterbox tạo artifact nền trắng  
- Một số cặp mệnh giá gần màu (5k↔20k) dễ nhầm  
- Thiếu lớp 500đ → thu hẹp phạm vi 9 lớp  
**Visual (tuỳ chọn):** histogram phân bố lớp hoặc ví dụ ảnh + bbox từ EDA.

---

### Slide 7 — Baseline A0 (YOLOv8n)
**Thiết lập:** COCO pretrained → fine-tune; `imgsz=640`; ~50 epoch; dữ liệu bbox Roboflow.  
**Quan sát:** Trên valid Roboflow, mAP50 ≈ **0,995** (miền studio gần bão hòa).  
**Hàm ý:** Cần đánh giá thêm trên ảnh thực; không kết luận chỉ từ Roboflow.  
**Visual:** Logo/kiến trúc YOLO nano (đơn giản) hoặc 1 ảnh pred baseline trên studio.

---

### Slide 8 — Hành trình cải tiến v2 → v3 → v4
**Dạng timeline / 3 cột:**

| Vòng | Thay đổi chính | Mục tiêu |
|------|----------------|----------|
| v2 | YOLOv8s + augment mạnh | Capacity + đa dạng biến dạng |
| v3 | imgsz↑, xoay 90°, oversample lớp khó | Giảm nhầm cặp màu gần |
| v4 | + ảnh camera thật; hold nhãn ~1s (demo) | Thu hẹp domain gap |

**Một câu chốt:** Cải tiến theo cả kiến trúc, dữ liệu và thích nghi miền — không chỉ “train lâu hơn”.

---

### Slide 9 — Protocol đánh giá công bằng
**Tiêu đề phụ:** `fair_eval_hardmix`  
**Nội dung cố định:**
- 40 ảnh chụp thực: **16 Hard** + **24 Normal** (seed=42)  
- Hard: lóa sáng, cong tờ, nền rối, hắt sáng, xoay/lật  
- Normal: điều kiện thuận lợi hơn  
- Đo chính: `imgsz=640`, `conf=0.25`  
- Checkpoint chọn khi train trên Roboflow; fair_eval dùng đánh giá độc lập sau khi khóa cấu hình  

**Visual:** 2 thumbnail Hard vs 2 thumbnail Normal (không cần pred).

---

### Slide 10 — Kết quả tổng quan (ALL 40 ảnh) ⭐
**Bảng số liệu khóa (làm nổi bật):**

| Mô hình | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---------|-------|----------|-----------|--------|-----|
| Baseline A0 | **0,399** | 0,325 | 0,708 | 0,435 | 0,539 |
| Final v4 | **0,929** | 0,644 | 0,921 | 0,913 | 0,917 |
| Δ (v4−A0) | **+0,530** | +0,319 | — | — | — |

**Visual:** `fig_val_compare_overview.png`  
**Câu diễn giải (1 dòng):** Trên ảnh thực, khoảng cách hai mô hình lộ rõ — khác với Roboflow (~0,99 cả hai).

---

### Slide 11 — Hard vs Normal ⭐
**Bảng:**

| Nhóm | Baseline mAP50 | Final mAP50 | Δ |
|------|----------------|-------------|---|
| Hard (16) | **0,189** | **0,875** | **+0,686** |
| Normal (24) | 0,573 | 0,955 | +0,382 |

**Visual:** `fig_test_compare_overview.png`  
**Câu chốt:** Lợi ích v4 tập trung ở miền Hard; baseline suy giảm mạnh khi điều kiện thực khó.

---

### Slide 12 — AP50 theo mệnh giá
**Ý:** Cải thiện không đồng đều theo lớp; các lớp từng yếu tăng rõ.  
**Visual:** `fig_val_AP50_per_class.png`  
**Ghi chú ngắn:** Nhóm dễ nhầm màu (5k / 20k / 200k / …) là trọng tâm các vòng cải tiến.

---

### Slide 13 — Minh chứng ảnh #1 (Hard, nhiều tờ) ⭐
**Layout bắt buộc 3 cột:**
| Nhãn thật (GT) | Baseline A0 | Final v4 |

**Nội dung nói:** Baseline bỏ sót / nhầm lớp; Final bám sát GT.  
**Nguồn ảnh:** chọn 1 ảnh Hard có ≥2–3 tờ từ `fair_eval_hardmix/hard/`.  
**Chú thích:** imgsz=640, conf=0.25.

---

### Slide 14 — Minh chứng ảnh #2 (Hard, điều kiện xấu) ⭐
**Layout 3 cột GT | Baseline | Final**  
**Chọn ảnh:** lóa sáng / xoay / nền rối / hắt sáng.  
**Ý:** Domain gap khiến mô hình studio-only thất bại; fine-tune + augment giúp ổn định.

---

### Slide 15 — Minh chứng ảnh #3 (Normal + collage sẵn)
**Nửa trên:** GT | Baseline | Final trên 1 ảnh Normal (baseline đã khá tốt).  
**Nửa dưới hoặc slide full:** dùng `fig_qualitative_val.png` / `fig_qualitative_real_cam.png` nếu collage mới chưa kịp.  
**Ý:** Trên Normal cả hai tốt hơn; khoảng cách thu hẹp — khẳng định Hard mới là thước đo phân biệt.

---

### Slide 16 — Ngưỡng tin cậy & đường cong học
**Hai panel:**
- Trái: `fig_conf_sweep_val.png` — chọn τ theo chi phí ($c_{FN}=3$, $c_{FP}=1$); τ* final ≈ **0,45**  
- Phải: `fig_train_curves.png` — tiến triển học qua vòng cải tiến  

**1 câu:** Ngưỡng tối ưu trên fair_eval; đánh giá so sánh chính vẫn cố định conf=0,25 để công bằng.

---

### Slide 17 — Phân tích lỗi & hạn chế
**Lỗi thường gặp:**
- Nhầm lớp cặp màu gần khi ánh sáng xấu  
- Bỏ sót khi mờ / chuyển động / che khuất  
- IoU thấp khi tờ nghiêng → ảnh hưởng mAP50-95  

**Hạn chế:**
- fair_eval còn nhỏ (40 ảnh) → khoảng tin cậy rộng  
- Ảnh camera fine-tune còn ít  
- Chưa có lớp 500đ; chưa phát hiện tiền giả / OCR  

---

### Slide 18 — Đóng góp chính (takeaways)
**3–4 ý, in đậm số liệu:**
1. Xây pipeline đầy đủ: EDA → baseline → cải tiến đa vòng → đánh giá độc lập.  
2. Trên ảnh thực, v4 vượt baseline rõ: mAP50 **0,40 → 0,93** (ALL); Hard **0,19 → 0,88**.  
3. Chỉ số Roboflow cao không đồng nghĩa độ bền camera — cần tập Hard/Normal.  
4. Cải tiến hiệu quả nhờ kết hợp capacity, augment, oversample và dữ liệu thực.

---

### Slide 19 — Hướng phát triển
- Mở rộng fair_eval và dữ liệu camera đa điều kiện  
- Bổ sung lớp 500đ  
- Hậu xử lý phân lớp trên crop / đặc trưng màu  
- Ngưỡng thích nghi theo ngữ cảnh  
- (Tuỳ nhu cầu) OCR sê-ri  

---

### Slide 20 — Kết luận & cảm ơn
**Kết luận 2 câu:**
- Đã hoàn thành hệ thống phát hiện mệnh giá tiền VN trên YOLOv8 với đánh giá công bằng trên ảnh thực.  
- Mô hình cuối (v4) thể hiện độ bền vượt trội so với baseline trên nhóm Hard.  

**Dòng cuối:** Cảm ơn Thầy/Cô và Quý vị đã lắng nghe.  
**Q&A.**

---

## 2. Biến thể độ dài (18 hoặc 22 slide)

### Nếu rút còn **18**
Gộp các cặp:
- 5+6 → một slide “Dữ liệu & EDA”  
- 16 bỏ hoặc gộp vào 12  
- 17+19 → “Hạn chế & hướng phát triển”

### Nếu kéo lên **22**
Tách thêm:
- Sau slide 8: slide “Augment & oversample” (minh họa biến dạng)  
- Sau slide 11: slide riêng Precision/Recall/F1 trên Hard  
- Sau slide 15: thêm 1 slide collage lỗi còn lại của v4 (trung thực)  
- Trước kết luận: slide demo camera (screenshot UI)

---

## 3. Checklist số liệu “không được sai”

Dùng đúng các số sau trên slide kết quả (làm tròn 3 chữ số thập phân kiểu Việt Nam):

| Metric | Baseline | Final v4 |
|--------|----------|----------|
| ALL mAP50 | 0,399 | 0,929 |
| ALL mAP50-95 | 0,325 | 0,644 |
| Hard mAP50 | 0,189 | 0,875 |
| Normal mAP50 | 0,573 | 0,955 |
| Hard F1 | 0,305 | 0,902 |

Protocol ghi chú chân trang khi có bảng số:  
`fair_eval_hardmix · imgsz=640 · conf=0,25 · 16 Hard + 24 Normal`

---

## 4. Gợi ý lời thuyết minh (speaker notes ngắn)

Không in lên slide; agent có thể gắn speaker notes:

| Slide | 15–25 giây nói |
|-------|----------------|
| 7 | Baseline rất mạnh trên Roboflow nên dễ tạo ảo giác “bài toán đã xong”. |
| 9 | Vì vậy chúng em dựng tập ảnh thực Hard/Normal để đo công bằng. |
| 10–11 | Đây là kết quả chính của báo cáo: delta lớn nhất nằm ở Hard. |
| 13–14 | Ảnh minh họa giúp thấy lỗi bỏ sót/nhầm lớp của baseline. |
| 17 | Nhóm vẫn trung thực về hạn chế quy mô đánh giá. |
| 20 | Mở Q&A: protocol, chọn YOLOv8s, dữ liệu camera, demo. |

---

## 5. Deliverable mong đợi từ agent thiết kế

1. File slide (PPTX hoặc PDF) **18–22 trang**, đúng narrative trên.  
2. Tối thiểu **3 slide ảnh** so sánh GT / Baseline / Final (slide 13–15).  
3. Nhúng các figure trong `report/figures/` ở đúng vị trí đã chỉ.  
4. Không thêm nội dung ngoài đề tài; không thêm slide “cách chạy repo”.  
5. Xuất kèm thư mục `slides_assets/` nếu có collage tự tạo (để tái sử dụng trong báo cáo).

---

## 6. Tham chiếu nội dung gốc

- Báo cáo: `report/Bao_cao_Nhan_dang_tien_Viet_Nam.docx`  
- Notebook: `report/notebooks/01_*.ipynb`, `02_*.ipynb`, `03_*.ipynb`  
- Số liệu: `report/exports/compare_summary.json`  
- Dataset đánh giá: `datasets/fair_eval_hardmix/`
