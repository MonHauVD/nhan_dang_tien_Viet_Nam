# Kế hoạch áp dụng kiến thức môn Nhận dạng mẫu vào bài nhận dạng tiền VN

> **Đề tài:** Phát hiện + nhận dạng mệnh giá tiền giấy/polymer Việt Nam (object detection)  
> **Dữ liệu hiện có:** `Vietnamese Currency.v1i.yolov8` — 2194 ảnh, 5464 object, **9 class** (thiếu `500`), nhãn lẫn bbox (~70%) và polygon (~30%)  
> **Yêu cầu môn:** mô hình cơ sở → ≥1 mô hình cải tiến/thay thế → EDA → phân tích lỗi → so sánh metric  
> **Nguồn lý thuyết:** Slide 1–6 (TS. Nguyễn Xuân Đức)

Lưu ý quan trọng: Khi làm cần thể hiện như 1 người làm chuyên nghiệp với các bước làm theo logic, chỉ là những chỗ nào có thể áp dụng được lý thuyết thì bổ sung nó vào. KHÔNG đưa dẫn chứng thể hiện là tôi dùng lý thuyết này ở slide nào.
---

## 0. Bản đồ nhanh: slide → chỗ dùng trong dự án

| Slide | Chủ đề thầy dạy | Áp dụng vào bài tiền như thế nào? | Ưu tiên |
|------:|-----------------|-----------------------------------|:-------:|
| 1 | Tổng quan PR, pipeline học từ dữ liệu | Khung toàn bộ BTL: xác định bài toán → dữ liệu → biểu diễn → model → đánh giá → phân tích lỗi & cải tiến | Bắt buộc |
| 2 | Xác suất, Bayes, loss/risk, ngưỡng | Chọn `conf` threshold theo chi phí FP/FN; giải thích confusion giữa mệnh giá gần nhau; MAP vs quyết định theo cost | Cao |
| 3 | Dữ liệu & đặc trưng, EDA, leakage, pipeline | EDA (đã có `01_EDA.ipynb`); QA nhãn; bias nguồn ảnh; feature cổ điển (màu/texture); fit chỉ trên train | Bắt buộc |
| 4 | PCA, LDA, curse of dimensionality | Trực quan hóa đặc trưng crop tờ tiền; baseline cổ điển (PCA/LDA + classifier); so với deep model | Trung bình–Cao |
| 5 | Hồi quy, baseline, CV, overfit, regularization | **Tinh thần quy trình** (baseline → tune trên val/CV → khóa → test 1 lần); không dùng linear regression làm detector chính | Cao (quy trình) |
| 6 | Logistic, Precision/Recall/F1, ROC/PR, threshold theo cost | Metric & bảng so sánh; confusion matrix theo class; sweep ngưỡng confidence; baseline “ngây thơ” | Bắt buộc |

**Lưu ý quan trọng:** bài toán của nhóm là **detection đa đối tượng** (box + class), trong khi Slide 5–6 thiên về **hồi quy / phân loại tabular**. Không cố nhét Linear/Logistic Regression làm model phát hiện tờ tiền chính. Thay vào đó:
- Dùng **YOLO (hoặc Faster R-CNN)** làm pipeline detection hiện đại.
- Dùng kiến thức Slide 2–6 để: **EDA đặc trưng**, **baseline cổ điển phụ**, **chọn ngưỡng**, **metric**, **so sánh trung thực train/val/test**.

---

## 1. Chốt bài toán theo Slide 1 (đầu vào – đầu ra – đánh giá)

### 1.1. Định nghĩa formal
- **Pattern (mẫu):** mỗi tờ tiền trong ảnh (có thể 0–n tờ / ảnh).
- **Feature / biểu diễn:**
  - Deep: feature map CNN trong YOLO (học tự động từ pixel).
  - Cổ điển (để gắn slide 3–4–6): vector màu/texture/HOG trên crop bbox.
- **Label:** class mệnh giá + tọa độ box.
- **Đầu ra hệ thống:** danh sách `(x1,y1,x2,y2, class, confidence)`.
- **Generalization:** tốt trên `test` (ảnh chưa thấy), không chỉ train.

### 1.2. Nhóm bài toán
- Chính: **phân loại có định vị** (object detection) — mở rộng của classification trong Slide 1.
- Phụ (nếu làm baseline cổ điển trên crop): **phân loại 9 lớp**.
- PCA/LDA (Slide 4): **giảm chiều / trực quan**, không phải mục tiêu cuối.

### 1.3. Metric quyết định “mô hình tốt”
Bám Slide 6 (P/R/F1) nhưng dùng bản detection:
- **mAP50**, **mAP50-95**
- Precision / Recall (overall + per-class)
- Confusion matrix giữa các mệnh giá (sau khi match box theo IoU)
- Phân tích theo độ khó ảnh: 1 tờ vs nhiều tờ, box nhỏ/lớn

**Quy tắc Slide 5–6:** mọi lựa chọn (model size, epoch, threshold, augment) chỉ dùng **train/val**. **Test mở 1 lần** khi đã khóa cấu hình.

---

## 2. EDA & chất lượng dữ liệu (Slide 1 + 3) — đang làm / mở rộng

### 2.1. Đã có
Notebook `01_EDA.ipynb`: split, class balance, objects/ảnh, bbox vs polygon, kích thước box, visualize nhãn.

### 2.2. Bổ sung EDA để “đúng tinh thần Slide 3”
Làm thêm trong notebook EDA hoặc `02_Feature_EDA.ipynb`:

1. **Đơn vị quan sát rõ ràng**
   - Mức ảnh: có bao nhiêu tờ?
   - Mức object: mỗi box là 1 mẫu phân loại mệnh giá.
2. **Chất lượng nhãn / bias nguồn**
   - Nguồn Roboflow + ảnh tự chụp: góc máy, nền, ánh sáng có lệch không?
   - Letterbox trắng sau resize 640 (Fit white edges) — ghi nhận ảnh hưởng.
3. **Outlier hình học**
   - Box cực nhỏ / cực lớn / aspect ratio bất thường → xem có lỗi gán nhãn không.
4. **Thống nhất annotation**
   - Convert toàn bộ polygon → bbox trước khi train detection.
5. **Thiếu class `500`**
   - Quyết định sớm: thu thập thêm **hoặc** thu hẹp đề bài còn 9 mệnh giá (nêu rõ trong báo cáo).
6. **Không leakage**
   - Không dùng ảnh test để chọn augment / early stop / threshold.
   - Không train lại trên test rồi báo metric test.

### 2.3. Feature cổ điển trên crop (gắn Slide 3 “ảnh → pixel/màu/texture”)
Với mỗi box (train/val):
- Histogram màu (HSV/RGB)
- Thống kê texture đơn giản (std, edge density) hoặc HOG
- Kích thước / aspect ratio box

→ Histogram/boxplot theo class (Slide 3), phục vụ PCA/LDA và baseline cổ điển.

---

## 3. Chiến lược model: Baseline → Cải tiến (đúng yêu cầu môn)

Nguyên tắc Slide 5: **luôn có baseline làm mốc**, mọi model mới phải thắng baseline trên cùng protocol đánh giá.

### Nhánh A — Deep detection (trục chính, khuyến nghị)

| Giai đoạn | Model | Mục đích | Gắn slide |
|-----------|-------|----------|-----------|
| **A0. Baseline** | YOLOv8n fine-tune, setting mặc định tối giản | Mốc so sánh nhanh | Slide 5 “baseline” |
| **A1. Cải tiến dữ liệu** | Cùng YOLOv8n + augment mạnh hơn + convert polygon→bbox + (tuỳ) cân class | Cải tiến từ phía dữ liệu/feature (Slide 3) | Slide 1 bước “phân tích lỗi & cải tiến” |
| **A2. Cải tiến model/capacity** | YOLOv8s (hoặc m) / train lâu hơn / tune lr | Tăng sức biểu diễn, kiểm soát overfit bằng val | Slide 5 overfit/CV tinh thần |
| **A3. Cải tiến quyết định** | Sweep `conf` (và IoU NMS) trên **val** theo cost | Bayes decision / threshold theo loss (Slide 2 + 6) | Rất “ra bài” với thầy |

**Đề xuất tối thiểu để đủ báo cáo:** A0 → (A1 hoặc A2) → A3.  
**Đề xuất đẹp:** A0 + A1 + A3, nếu còn thời gian thêm A2.

### Nhánh B — Baseline cổ điển theo đúng slide (điểm cộng rõ ràng)

Áp dụng khi muốn chứng minh nhóm dùng PCA/LDA/Logistic đúng bài giảng:

1. Lấy crop theo GT box (hoặc theo detector baseline).
2. Rút feature màu/texture/HOG → chuẩn hóa **fit trên train** (Slide 3 pipeline).
3. **PCA** trực quan 2D/3D theo class + scree plot (Slide 4).
4. **LDA** chiếu có giám sát; so sánh tách lớp với PCA.
5. Phân loại: **Logistic Regression** (± class weight) hoặc LDA classifier (Slide 6 + 4).
6. Đánh giá Precision/Recall/F1/confusion trên crop validation (Slide 6).

> Nhánh B **không thay** detection end-to-end, nhưng là “mô hình cơ sở mang tính học phần” rất thuyết phục khi giải thích: deep YOLO học φ(x) tự động, còn nhóm cũng làm φ(x) thủ công như giáo trình.

### Nhánh C (tuỳ chọn, nếu dư thời gian)
- Faster R-CNN làm model thay thế family khác (so với YOLO).
- Two-stage: YOLO localize → Logistic/LDA verify class trên crop (kết hợp deep + Bayes/logistic).

---

## 4. Áp dụng chi tiết từng slide (checklist “dùng được gì”)

### Slide 1 — Tổng quan
- [x] Viết rõ pattern / feature / label / model / generalization trong báo cáo.
- [ ] Tuân pipeline 7 bước thầy dạy: bài toán → hiểu dữ liệu → biểu diễn/tiền xử lý → chia dữ liệu → train → đánh giá → phân tích lỗi & lặp lại.
- [ ] Demo suy luận trên ảnh mới (giai đoạn inference ≠ training).

### Slide 2 — Bayes / loss / ngưỡng
**Áp dụng được (nên làm):**
- [ ] Coi `confidence` của YOLO như tín hiệu xếp hạng gần với “score/posterior”.
- [ ] Xây **loss matrix nghiệp vụ** đơn giản, ví dụ:
  - FN (bỏ sót tờ tiền): chi phí cao nếu mục tiêu đếm/kiểm quỹ.
  - FP class sai (nhầm 50k ↔ 500k): chi phí rất cao.
  - FP nền (hallucination box): chi phí trung bình.
- [ ] Trên **validation**, sweep `conf ∈ [0.05..0.7]`, chọn τ tối thiểu hóa `Cost = c_FP·FP + c_FN·FN` (đúng Slide 2 & 6).
- [ ] Khóa τ rồi mới đánh giá test.
- [ ] Trong báo cáo: giải thích vùng chồng lấn đặc trưng giữa mệnh giá gần màu/họa tiết (bất định Slide 2).

**Áp dụng hạn chế:**
- Không bắt buộc ước lượng Gaussian Bayes classifier end-to-end trên ảnh thô (không thực tế với detection).
- Có thể minh họa MLE/MAP trên **phân loại crop** nhánh B.

### Slide 3 — Dữ liệu & đặc trưng
**Áp dụng mạnh:**
- [x]/[ ] EDA đầy đủ (histogram class, box size, outliers).
- [ ] Checklist leakage: split cố định Roboflow; không fit gì trên test.
- [ ] Convert nhãn polygon → bbox.
- [ ] Feature engineering cổ điển trên crop (màu/texture).
- [ ] Ghi nhận selection bias: ảnh studio/Roboflow vs ảnh thực tế điện thoại.

### Slide 4 — PCA / LDA
**Áp dụng nên làm (notebook riêng):**
- [ ] PCA trên feature crop → scatter theo class, scree plot.
- [ ] LDA → xem class nào tách tốt / chồng (dự báo class dễ nhầm).
- [ ] Optional: classifier PCA+Logistic hoặc LDA làm **baseline cổ điển**.
- [ ] Nhận xét curse of dimensionality nếu dùng HOG cao chiều + ít mẫu/class.

**Không nên:** thay YOLO bằng PCA thuần để detect vị trí tờ tiền.

### Slide 5 — Tinh thần hồi quy (quy trình so sánh)
Dù không làm regression:
- [ ] Luôn có **baseline A0**.
- [ ] Chọn cấu hình bằng val (tương đương CV/val trong slide).
- [ ] Theo dõi overfit: train metric ≫ val metric → giảm augment mạnh/model lớn hoặc early stop.
- [ ] Bảng so sánh cuối kiểu slide 5:

| Model | Chọn từ | Val mAP50 | Test mAP50 | Test P | Test R | Ghi chú |
|-------|---------|-----------|------------|--------|--------|---------|
| Baseline YOLOv8n | — | | | | | mốc |
| +Augment / +s | Val | | | | | |
| +Threshold tối ưu cost | Val | | | | | |

- [ ] Không chọn model theo điểm test.

### Slide 6 — Logistic / metric phân loại
**Áp dụng trực tiếp vào detection & nhánh B:**
- [ ] Confusion matrix per-class (mệnh giá).
- [ ] Precision / Recall / F1 per-class + macro.
- [ ] PR curve / AP per-class (tương đương tinh thần PR-AUC/AP trong slide).
- [ ] Baseline ngây thơ (ví dụ: luôn đoán class phổ biến trên crop; hoặc conf rất thấp) để chứng minh model thật sự vượt prior.
- [ ] Nhấn mạnh: Accuracy tổng dễ đánh lừa khi đa class / đa object — ưu tiên P/R/mAP.

---

## 5. Lộ trình thực thi theo tuần (actionable)

### Giai đoạn 1 — Dữ liệu & EDA (đang làm)
1. Chạy xong `01_EDA.ipynb`, xuất CSV thống kê.
2. Quyết định phạm vi 9 vs 10 class.
3. Script convert polygon → YOLO bbox thuần.
4. (Khuyến nghị) `02_Feature_EDA_PCA_LDA.ipynb`: feature crop + PCA/LDA.

**Deliverable:** hình EDA + nhận xét bias/nhãn cho báo cáo.

### Giai đoạn 2 — Baseline deep
1. Cài `ultralytics`, sửa `data.yaml` path cho đúng máy.
2. Train **YOLOv8n** (A0), cố định seed, imgsz=640, epochs vừa đủ (vd 50–100).
3. Lưu `best.pt`, curve train/val, metric val.

**Deliverable:** bảng metric baseline + vài ảnh inference lỗi/đúng.

### Giai đoạn 3 — Cải tiến (chọn 1–2 hướng, đừng làm hết)
Ưu tiên theo “dễ giải thích bằng slide”:
1. **Dữ liệu:** augment + làm sạch nhãn (Slide 3).
2. **Quyết định:** tối ưu threshold theo cost trên val (Slide 2+6).
3. **Model:** YOLOv8s nếu còn GPU/thời gian (Slide 5 capacity vs overfit).
4. **Cổ điển:** PCA/LDA + Logistic trên crop (Slide 4+6) như baseline phụ.

**Deliverable:** ≥1 model cải tiến thắng baseline trên val; giải thích *vì sao* thắng.

### Giai đoạn 4 — So sánh công bằng & phân tích lỗi
1. Khóa cấu hình → chạy **một lần** trên test.
2. Bảng so sánh Baseline vs Improved (mAP, P, R, per-class AP).
3. Error analysis (Slide 1 bước 7):
   - Nhầm class nào với class nào? (kỳ vọng: 20k↔50k, 100k↔500k…)
   - Ảnh nhiều tờ / chồng / box nhỏ có recall thấp?
   - False positive trên nền / letterbox?
4. Liên hệ Slide 2: đây là vùng chồng lấn đặc trưng + bất định đo lường.

**Deliverable:** section “Phân tích lỗi & hạn chế” cho báo cáo.

### Giai đoạn 5 — Demo + báo cáo + slide nhóm
1. Demo upload ảnh → vẽ box + mệnh giá.
2. Báo cáo trích dẫn slide/Bishop + nguồn Roboflow.
3. Slide thuyết trình đi đúng câu chuyện: bài toán → EDA → baseline → cải tiến (kiến thức môn) → kết quả → hạn chế.

---

## 6. Protocol so sánh model (bắt buộc thống nhất)

Để tránh tranh cãi “model nào tốt hơn”:

1. **Cùng data split** Roboflow (train/valid/test).
2. **Cùng input size** (640) trừ khi ablation riêng.
3. **Cùng quy tắc NMS** khi so backbone; chỉ đổi khi ablation threshold.
4. Chọn checkpoint theo **val mAP50** (hoặc val cost), không theo test.
5. Báo cáo test kèm khoảng tin cậy định tính: số ảnh test = 218 — nêu hạn chế mẫu.
6. Mỗi lần cải tiến chỉ đổi **một nhóm yếu tố** chính (data / model / decision) để quy nhân được.

### Công thức cost gợi ý (A3)
Trên validation, sau matching IoU ≥ 0.5:
\[
\mathrm{Cost}(\tau)= c_{FP}\cdot FP(\tau) + c_{FN}\cdot FN(\tau)
\]
Ví dụ bắt đầu: \(c_{FP}=1\), \(c_{FN}=3\) hoặc \(5\) (bỏ sót đắt hơn cảnh báo nhầm).  
Đổi bộ chi phí → ngưỡng tối ưu đổi (đúng thông điệp Slide 2).

---

## 7. Đề xuất “combo vàng” cho nhóm (thực tế + đúng bài giảng)

Làm theo thứ tự này nếu thời gian eo hẹp:

1. **EDA hoàn chỉnh** (Slide 1+3) — đã có nền `01_EDA.ipynb`
2. **Baseline YOLOv8n** (Slide 5 tinh thần baseline)
3. **Cải tiến 1:** augment + chuẩn hóa nhãn bbox (Slide 3)
4. **Cải tiến 2:** tối ưu confidence threshold theo cost trên val (Slide 2+6)
5. **Notebook phụ:** PCA/LDA trên crop + confusion/P/R (Slide 4+6) — chứng minh dùng kiến thức thầy
6. **So sánh bảng + phân tích lỗi + demo**

Combo này đủ: có model cơ sở, có cải tiến, có metric chuẩn môn, và có đoạn lý thuyết Bayes/PCA/LDA không bị “chỉ fine-tune YOLO”.

---

## 8. Việc không nên làm (tránh lạc đề / mất thời gian)

- Dùng Linear Regression dự đoán mệnh giá như số thực (5000, 10000…) — sai bài toán, khó gắn detection.
- Train Logistic trực tiếp trên ảnh 640×640 flatten — chiều quá lớn, đúng “curse of dimensionality” nhưng kết quả kém, dễ phản tác dụng nếu không PCA.
- Tối ưu threshold trên **test**.
- Đổi quá nhiều thứ cùng lúc rồi không giải thích được cải tiến đến từ đâu.
- Hứa 10 class trong README trong khi data chỉ có 9 — phải chỉnh cho khớp.

---

## 9. Phân công gợi ý (map với README nhóm)

| Việc | Phụ trách gợi ý |
|------|-----------------|
| EDA + PCA/LDA notebook + hình báo cáo | Dũng / Lâm |
| Convert nhãn, train baseline & improved YOLO | Dũng / Lâm |
| Sweep threshold + bảng cost/P/R | Dũng |
| Thu thập thêm ảnh class thiếu / QA nhãn lệch | Minh |
| Viết báo cáo gắn từng slide + slide thuyết trình | Đức Anh (+ cả nhóm góp số liệu) |
| Demo inference | Lâm / Dũng |

---

## 10. Tiêu chí “xong” (Definition of Done)

- [ ] EDA có số liệu + nhận xét (split, class, box, hạn chế).
- [ ] Có **Baseline** và **≥1 Improved**, cùng protocol.
- [ ] Có bảng so sánh val/test (mAP, P, R; ideally per-class).
- [ ] Có tối thiểu **một** kỹ thuật “đúng slide” được phân tích rõ: threshold-theo-cost **hoặc** PCA/LDA+Logistic trên crop.
- [ ] Có phân tích lỗi + hạn chế.
- [ ] Có demo chạy được trên ảnh mới.
- [ ] Báo cáo/slide trích dẫn nguồn dữ liệu và liên hệ lý thuyết môn học.

---

## 11. File notebook / code dự kiến tiếp theo

```text
notebooks/
  01_EDA.ipynb                  # đã có
  plan.md                       # file này
  02_Feature_PCA_LDA.ipynb      # feature crop + PCA/LDA + logistic baseline
  03_Train_YOLO_Baseline.ipynb  # train A0 + log metric
  04_Improve_and_Compare.ipynb  # A1/A2 + threshold sweep + bảng so sánh
src/   (tuỳ chọn khi chuyển khỏi notebook)
  convert_polygon_to_bbox.py
  train.py / evaluate.py / inference.py
```

---

**Kết luận đề xuất:**  
Giữ **YOLO làm xương sống detection**, nhưng **đóng gói toàn bộ quy trình theo Slide 1–6**: EDA/feature (3), baseline & chống overfit (5), metric P/R/F1/mAP (6), quyết định theo cost/ngưỡng (2), và PCA/LDA như nhánh giải thích/baseline cổ điển (4). Như vậy vừa ra sản phẩm thực tế, vừa “ăn điểm” đúng kiến thức thầy dạy.
