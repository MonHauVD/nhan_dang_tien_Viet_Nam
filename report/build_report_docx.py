#!/usr/bin/env python3
"""Sinh file Word báo cáo BTL Nhận dạng mẫu — Nhận dạng tiền Việt Nam."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Bao_cao_Nhan_dang_tien_Viet_Nam.docx"
FIG = ROOT / "figures"
SAMPLE = Path("/home/dungtv/Desktop/Note/Hoc_Tap/Nhan_dang_mau/Sample_report.docx")


def set_run_font(run, name="Times New Roman", size=13, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_para(doc, text, *, size=13, bold=False, italic=False, align="justify",
             space_before=0, space_after=6, first_line=True, center=False):
    p = doc.add_paragraph()
    if center or align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line and align == "justify":
        pf.first_line_indent = Cm(1.0)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size=14 if level == 1 else 13, bold=True)
        run.font.color.rgb = RGBColor(0, 0, 0)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    return h


def add_page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def set_cell_text(cell, text, *, bold=False, center=True, size=11):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)


def shade_cell(cell, fill="D9E2F3"):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")
    tc.append(shd)


def add_bullet(doc, text, size=13):
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    run = p.add_run(text)
    set_run_font(run, size=size)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    return p


def _set_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    if tbl.tblPr is None:
        tbl.insert(0, tblPr)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    try:
        table.style = "Table Grid"
    except KeyError:
        _set_table_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[j], h, bold=True, size=11)
        shade_cell(table.rows[0].cells[j], "D6DCE4")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell_text(table.rows[i + 1].cells[j], str(val), bold=False, size=11,
                          center=True)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph()
    return table


def add_caption(doc, text):
    p = add_para(doc, text, size=11, italic=True, align="center", first_line=False, space_after=10)
    return p


def add_toc_field(doc):
    """Chèn trường Mục lục — người dùng cập nhật trong Word (F9 / chuột phải)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r' TOC \o "1-3" \h \z \u '
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_text = OxmlElement("w:t")
    fld_text.text = "Nhấn chuột phải → Cập nhật trường để hiện mục lục."
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_text)
    run._r.append(fld_end)


def try_add_image(doc, path: Path, width_cm=14.0, caption=None):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(path), width=Cm(width_cm))
        if caption:
            add_caption(doc, caption)
        return True
    return False


def build():
    # Tạo document mới (khổ A4, lề chuẩn báo cáo); bố cục bìa tham khảo Sample_report.
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)

    # ========== BÌA ==========
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc, "HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG", size=14, bold=True,
             align="center", first_line=False, space_after=0)
    add_para(doc, "KHOA CÔNG NGHỆ THÔNG TIN 1", size=13, bold=True,
             align="center", first_line=False, space_after=18)

    for _ in range(2):
        doc.add_paragraph()

    add_para(doc, "BÁO CÁO BÀI TẬP LỚN", size=16, bold=True,
             align="center", first_line=False, space_after=4)
    add_para(doc, "MÔN HỌC: NHẬN DẠNG MẪU", size=14, bold=True,
             align="center", first_line=False, space_after=18)

    add_para(doc,
             "Đề tài: Phát hiện và nhận dạng mệnh giá tiền giấy/polymer Việt Nam "
             "bằng mô hình học sâu (YOLOv8)",
             size=13, bold=True, align="center", first_line=False, space_after=24)

    members = [
        ("1", "B26CHKH008", "Hoàng Minh Đức Anh", "—", "Báo cáo, slide thuyết trình"),
        ("2", "B26CHKH030", "Trần Việt Dũng", "tranvietdung121@gmail.com",
         "Pipeline, huấn luyện, báo cáo, QA & Test"),
        ("3", "B26CHKH059", "Vũ Hoàng Lâm", "lamhoangvu204@gmail.com",
         "Pipeline, huấn luyện, gán nhãn, QA & Test"),
        ("4", "B26CHKH072", "Lê Khánh Minh", "—", "Thu thập ảnh, gán nhãn metadata"),
    ]
    add_para(doc, "Nhóm sinh viên thực hiện: Nhóm 08 — Lớp M26CQKH02-B", size=12, bold=True,
             align="center", first_line=False, space_after=8)
    add_table(
        doc,
        ["TT", "Mã SV", "Họ và tên", "Email", "Nhiệm vụ chính"],
        members,
        col_widths=[1.2, 2.8, 4.0, 4.2, 4.5],
    )

    add_para(doc, "Giảng viên phụ trách: (điền khi nộp)", size=12,
             align="center", first_line=False, space_before=12)
    for _ in range(3):
        doc.add_paragraph()
    add_para(doc, "Hà Nội — 2026", size=13, bold=True, align="center", first_line=False)

    # ========== NHẬN XÉT ==========
    add_page_break(doc)
    add_para(doc, "NHẬN XÉT CỦA GIẢNG VIÊN", size=14, bold=True, align="center",
             first_line=False, space_after=12)
    add_para(doc, "Giảng viên hướng dẫn: ........................................",
             size=12, first_line=False)
    add_para(doc, "Nhóm sinh viên thực hiện: 08", size=12, first_line=False)
    add_para(doc,
             "Tên đề tài: Phát hiện và nhận dạng mệnh giá tiền giấy/polymer Việt Nam",
             size=12, first_line=False, space_after=12)
    add_para(doc, "NỘI DUNG NHẬN XÉT:", size=12, bold=True, first_line=False)
    for _ in range(14):
        add_para(doc, "." * 90, size=12, first_line=False, space_after=2)
    add_para(doc, "Hà Nội, ngày .... tháng .... năm 2026", size=12,
             align="right", first_line=False, space_before=18)
    add_para(doc, "GIẢNG VIÊN HƯỚNG DẪN", size=12, bold=True,
             align="right", first_line=False)

    # ========== LỜI MỞ ĐẦU ==========
    add_page_break(doc)
    add_heading_custom(doc, "LỜI MỞ ĐẦU", level=1)
    add_para(
        doc,
        "Bài toán nhận dạng mẫu trên dữ liệu ảnh thực tế đòi hỏi kết hợp phân tích dữ liệu, "
        "lựa chọn mô hình phù hợp và đánh giá khách quan theo quy trình học máy chuẩn. "
        "Trong khuôn khổ học phần Nhận dạng mẫu, nhóm chọn bài toán phát hiện và nhận dạng "
        "mệnh giá tiền Việt Nam: từ một ảnh đầu vào, hệ thống cần định vị từng tờ tiền bằng "
        "hộp bao (bounding box) và gán nhãn mệnh giá tương ứng."
    )
    add_para(
        doc,
        "Đề tài gắn với ứng dụng thực tiễn (hỗ trợ kiểm đếm, hỗ trợ người khiếm thị, kiểm tra "
        "tự động trong quy trình nghiệp vụ), đồng thời đủ phức tạp để thể hiện các bước: "
        "khảo sát dữ liệu, xây dựng mô hình cơ sở (baseline), cải tiến mô hình và dữ liệu, "
        "so sánh chỉ số và phân tích hạn chế."
    )
    add_para(
        doc,
        "Báo cáo trình bày lần lượt bài toán và mục tiêu, nguồn dữ liệu cùng phân tích sơ bộ, "
        "phương pháp giải quyết với mô hình cơ sở và mô hình cải tiến, kết quả thực nghiệm, "
        "phân tích lỗi và giới hạn, kết luận cùng hướng phát triển. Phụ lục và các notebook "
        "kèm theo cung cấp chi tiết EDA, quá trình huấn luyện và bảng so sánh định lượng."
    )

    # ========== MỤC LỤC ==========
    add_page_break(doc)
    add_heading_custom(doc, "MỤC LỤC", level=1)
    add_toc_field(doc)
    add_para(
        doc,
        "(Trong Microsoft Word: chọn toàn bộ mục lục → chuột phải → Cập nhật trường → "
        "Cập nhật toàn bộ mục lục.)",
        size=11, italic=True, first_line=False, space_before=8,
    )

    # Mục lục tĩnh dự phòng (để đọc được ngay cả khi chưa cập nhật field)
    add_para(doc, "Dàn ý chính (tham chiếu nhanh):", size=12, bold=True,
             first_line=False, space_before=12)
    toc_lines = [
        "Lời mở đầu",
        "Chương 1. Giới thiệu bài toán và mục tiêu",
        "Chương 2. Dữ liệu và phân tích sơ bộ (EDA)",
        "Chương 3. Phương pháp giải quyết",
        "    3.1. Bài toán phát hiện đối tượng và chỉ số đánh giá",
        "    3.2. Mô hình cơ sở (Baseline) — YOLOv8n",
        "    3.3. Mô hình cải tiến — YOLOv8s và các vòng tinh chỉnh",
        "    3.4. Chọn ngưỡng tin cậy theo chi phí lỗi",
        "Chương 4. Thực nghiệm và kết quả",
        "Chương 5. Phân tích lỗi, hạn chế và thảo luận",
        "Chương 6. Kết luận và hướng phát triển",
        "Phụ lục A. Phân công nhiệm vụ và tự đánh giá đóng góp",
        "Phụ lục B. Hướng dẫn chạy mã nguồn và notebook báo cáo",
        "Tài liệu tham khảo",
    ]
    for line in toc_lines:
        add_para(doc, line, size=12, first_line=False, space_after=2, align="left")

    # ========== CHƯƠNG 1 ==========
    add_page_break(doc)
    add_heading_custom(doc, "CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN VÀ MỤC TIÊU", level=1)

    add_heading_custom(doc, "1.1. Phát biểu bài toán", level=2)
    add_para(
        doc,
        "Cho một ảnh chứa không, một hoặc nhiều tờ tiền Việt Nam, hệ thống cần trả về danh sách "
        "các đối tượng được phát hiện. Mỗi đối tượng gồm tọa độ hộp bao và nhãn mệnh giá. "
        "Đây là bài toán phát hiện đối tượng (object detection) đa lớp, mở rộng của phân loại "
        "ảnh khi cần đồng thời định vị và phân lớp."
    )
    add_para(
        doc,
        "Không gian nhãn thực nghiệm gồm chín mệnh giá phổ biến trên bộ dữ liệu Roboflow đã "
        "chọn: 1.000, 2.000, 5.000, 10.000, 20.000, 50.000, 100.000, 200.000 và 500.000 đồng. "
        "Mệnh giá 500 đồng chưa có đủ mẫu có nhãn trong nguồn chính nên không đưa vào tập "
        "huấn luyện; phạm vi bài toán được nêu rõ để tránh hiểu nhầm với bộ mười mệnh giá "
        "đầy đủ trên thực tế lưu thông."
    )

    add_heading_custom(doc, "1.2. Mục tiêu", level=2)
    add_para(doc, "Các mục tiêu cụ thể gồm:", first_line=False)
    goals = [
        "Xây dựng pipeline từ dữ liệu có nguồn rõ ràng đến mô hình phát hiện mệnh giá có thể tái lập.",
        "Phân tích sơ bộ dữ liệu (số mẫu, phân bố lớp, hình học hộp bao, chất lượng nhãn) bằng thống kê và trực quan hóa.",
        "Huấn luyện mô hình cơ sở làm mốc so sánh, sau đó xây dựng ít nhất một mô hình cải tiến (kiến trúc, tăng cường dữ liệu, bổ sung ảnh thực tế).",
        "Đánh giá công bằng trên cùng tập kiểm tra, kèm phân tích lỗi và thảo luận hạn chế của giải pháp.",
        "Cung cấp demo tương tác (camera) để minh họa khả năng hoạt động ngoài ảnh studio.",
    ]
    for g in goals:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(g)
        set_run_font(run, size=13)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    add_heading_custom(doc, "1.3. Phạm vi và giả định", level=2)
    add_para(
        doc,
        "Phạm vi tập trung vào tờ tiền giấy/polymer Việt Nam trong ảnh RGB. Hệ thống không "
        "nhận diện tiền giả theo đặc trưng bảo an chuyên sâu, không OCR số sê-ri làm mục tiêu "
        "chính, và giả định nhãn được gán đúng về mệnh giá. Điều kiện ánh sáng kém, chuyển "
        "động mạnh hoặc che khuất lớn được xem là tình huống khó và được thảo luận trong "
        "phần hạn chế."
    )

    # ========== CHƯƠNG 2 ==========
    add_heading_custom(doc, "CHƯƠNG 2. DỮ LIỆU VÀ PHÂN TÍCH SƠ BỘ (EDA)", level=1)

    add_heading_custom(doc, "2.1. Nguồn dữ liệu", level=2)
    add_para(
        doc,
        "Nguồn chính dùng để huấn luyện và đánh giá là bộ Vietnamese Currency trên Roboflow "
        "Universe (phiên bản xuất định dạng YOLOv8), gồm khoảng 2.194 ảnh và hơn 5.400 đối "
        "tượng đã gán nhãn, chia sẵn train / valid / test. Nguồn này có nguồn gốc công khai, "
        "phù hợp yêu cầu dùng dữ liệu thực với nguồn trích dẫn rõ."
    )
    add_para(doc, "Các nguồn tham khảo / bổ trợ trong quá trình xây dựng đề tài:", first_line=False)
    sources = [
        "Roboflow — Vietnamese Currency (first-qkz37): https://universe.roboflow.com/first-qkz37/vietnamese-currency-lgi9i",
        "GitHub — tien-giay-vn (vietanhdev): https://github.com/vietanhdev/tien-giay-vn",
        "Kaggle — Vietnamese Money (huyphmnguyenthanh): https://www.kaggle.com/datasets/huyphmnguyenthanh/vietnamese-money",
        "Roboflow — Vietnamese Banknote (bao-vu): https://universe.roboflow.com/bao-vu/vietnamese-banknote",
        "Ảnh tự chụp bằng camera thực tế (front/back, nhiều mệnh giá): thư mục real_image / datasets/real_cam_yolo — dùng để thu hẹp khoảng cách miền (domain gap) khi triển khai demo.",
    ]
    for s in sources:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(s)
        set_run_font(run, size=12)

    add_heading_custom(doc, "2.2. Quy mô và ý nghĩa đặc trưng / nhãn", level=2)
    add_table(
        doc,
        ["Tập", "Số ảnh (xấp xỉ)", "Vai trò"],
        [
            ["Train (Roboflow)", "1.537", "Huấn luyện trọng số"],
            ["Valid (Roboflow)", "439", "Theo dõi học, chọn cấu hình, quét ngưỡng"],
            ["Test (Roboflow)", "218", "Đánh giá khóa — chỉ mở một lần theo protocol"],
            ["Ảnh camera thật", "18 gốc (+ tăng cường)", "Fine-tune thích nghi miền thực tế"],
        ],
    )
    add_caption(doc, "Bảng 2.1. Quy mô các tập dữ liệu sử dụng trong thực nghiệm")

    add_para(
        doc,
        "Mỗi mẫu quan sát ở mức đối tượng là một tờ tiền trong ảnh. Đặc trưng đầu vào của mô "
        "hình học sâu là tensor ảnh (sau resize/letterbox về kích thước huấn luyện, thường "
        "640×640). Nhãn gồm chỉ số lớp mệnh giá và hộp bao chuẩn hóa theo định dạng YOLO "
        "(cx, cy, w, h). Một phần nhãn gốc dạng đa giác (polygon) được chuyển về hộp bao trước "
        "khi huấn luyện để thống nhất pipeline phát hiện."
    )

    add_heading_custom(doc, "2.3. Phân tích sơ bộ", level=2)
    add_para(
        doc,
        "Phân tích sơ bộ được thực hiện trong notebook EDA (xem report/notebooks). Các nội dung "
        "chính gồm: phân bố số đối tượng trên mỗi ảnh; phân bố số lượng theo từng mệnh giá; "
        "tỷ lệ nhãn hộp bao so với đa giác; thống kê kích thước và tỷ lệ khung hình của hộp; "
        "kiểm tra lệch miền giữa ảnh studio/Roboflow và ảnh camera; kiểm tra rò rỉ dữ liệu "
        "(không dùng test để chọn siêu tham số)."
    )
    add_para(
        doc,
        "Quan sát nổi bật: bộ Roboflow đã được tiền xử lý khá đồng nhất (letterbox nền trắng), "
        "nên chỉ số trên valid/test dễ đạt mức cao ngay từ mô hình nhỏ. Ngược lại, ảnh camera "
        "thật có chuyển động, mờ và nền rối — đây là miền khó hơn và là động lực chính cho các "
        "vòng cải tiến dữ liệu. Một số cặp mệnh giá có màu sắc gần nhau (ví dụ 5.000 và 20.000) "
        "dễ gây nhầm lớp khi điều kiện quan sát kém. Class 500 đồng vắng mặt trong nguồn chính, "
        "do đó báo cáo giới hạn phạm vi chín lớp."
    )
    add_para(
        doc,
        "Chi tiết số liệu, biểu đồ và kiểm tra chất lượng nhãn được trình bày đầy đủ trong "
        "file 01_EDA_Bao_cao.ipynb kèm theo thư mục báo cáo."
    )

    # ========== CHƯƠNG 3 ==========
    add_heading_custom(doc, "CHƯƠNG 3. PHƯƠNG PHÁP GIẢI QUYẾT", level=1)

    add_heading_custom(doc, "3.1. Bài toán phát hiện và chỉ số đánh giá", level=2)
    add_para(
        doc,
        "Mô hình đầu ra là tập dự đoán (hộp bao, lớp, điểm tin cậy). Việc khớp dự đoán với "
        "nhãn thật dựa trên ngưỡng giao nhau trên hợp (IoU). Các chỉ số chính:"
    )
    metrics = [
        "mAP50: trung bình độ chính xác trung bình theo lớp tại IoU = 0,5 — phản ánh khả năng định vị và phân lớp ở mức chấp nhận thông dụng.",
        "mAP50-95: trung bình trên nhiều ngưỡng IoU (0,5–0,95) — nhạy hơn với độ khớp hộp bao.",
        "Precision / Recall / F1 ở mức phát hiện: cân bằng giữa báo nhầm và bỏ sót.",
        "AP50 theo từng lớp: phát hiện lớp yếu để định hướng tăng cường dữ liệu hoặc điều chỉnh mất mát.",
    ]
    for m in metrics:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(m)
        set_run_font(run, size=13)

    add_para(
        doc,
        "Quy trình lựa chọn cấu hình tuân thủ nguyên tắc: mọi quyết định (kiến trúc, số epoch, "
        "mức tăng cường, ngưỡng tin cậy) chỉ dựa trên train/valid; tập test chỉ đánh giá một "
        "lần sau khi khóa cấu hình để tránh lạc quan hóa kết quả."
    )

    add_heading_custom(doc, "3.2. Mô hình cơ sở (Baseline) — YOLOv8n", level=2)
    add_para(
        doc,
        "Mô hình cơ sở dùng YOLOv8n (nano) khởi tạo từ trọng số COCO, fine-tune trên tập "
        "vn_currency_yolo_bbox đã chuyển polygon về hộp bao. Cấu hình tiêu biểu: imgsz = 640, "
        "khoảng 50 epoch, batch phù hợp GPU, seed cố định để tái lập. Mục tiêu của baseline "
        "không phải đạt hiệu năng tối đa mà tạo mốc so sánh trung thực cho mọi cải tiến sau."
    )
    add_para(
        doc,
        "Trên tập valid Roboflow, baseline đã đạt mAP50 rất cao (khoảng 0,995) và mAP50-95 "
        "khoảng 0,93. Kết quả này cho thấy miền dữ liệu đánh giá khá “dễ” so với camera thực, "
        "đồng thời đặt yêu cầu so sánh cải tiến phải nhìn thêm hành vi trên miền khó và phân "
        "tích theo lớp / theo chi phí lỗi — không chỉ nhìn một con số mAP tổng."
    )

    add_heading_custom(doc, "3.3. Mô hình cải tiến — YOLOv8s và các vòng tinh chỉnh", level=2)
    add_para(
        doc,
        "Mô hình cải tiến đi theo hướng tăng dung lượng mạng, tăng cường dữ liệu mạnh hơn và "
        "bổ sung miền camera thật, thực hiện qua nhiều vòng:"
    )
    improve = [
        "Vòng v2: chuyển sang YOLOv8s; tăng cường scale/mosaic/mixup/copy-paste/erasing; nhấn mạnh mất mát phân lớp; huấn luyện dài hơn baseline.",
        "Vòng v3: fine-tune từ v2; tăng imgsz (ví dụ 800); xoay 90° và oversample các lớp dễ nhầm (như 5.000, 20.000, 200.000) nhằm giảm nhầm cặp màu gần.",
        "Vòng v4 (mô hình cuối): fine-tune từ v3 trên tập có thêm ảnh camera thật đã gán nhãn hộp bao thủ công; giữ các lớp trọng tâm còn lỗi trên demo; kết hợp cơ chế giữ nhãn theo thời gian trên demo (~1 giây) để giảm nhấp nháy giữa các khung hình.",
    ]
    for it in improve:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(it)
        set_run_font(run, size=13)

    add_para(
        doc,
        "Như vậy, so với baseline, mô hình cuối khác biệt ở ba trục: kiến trúc (n → s), "
        "chiến lược tăng cường và lấy mẫu lại lớp khó, và thích nghi miền bằng dữ liệu camera. "
        "Quá trình thêm/bớt cấu hình và căn chỉnh được ghi lại trong notebook "
        "02_Qua_trinh_Train_va_Cai_tien.ipynb."
    )

    add_heading_custom(doc, "3.4. Chọn ngưỡng tin cậy theo chi phí lỗi", level=2)
    add_para(
        doc,
        "Điểm tin cậy (confidence) của YOLO ảnh hưởng trực tiếp tới Precision và Recall. "
        "Trên tập valid, thực hiện quét ngưỡng và chọn giá trị làm giảm một hàm chi phí "
        "proxy kết hợp tỷ lệ dương tính giả và âm tính giả (ví dụ phạt bỏ sót nặng hơn "
        "báo nhầm trong ngữ cảnh kiểm đếm). Ngưỡng tối ưu trên valid được ghi nhận riêng "
        "cho baseline và cho mô hình cuối; tập test vẫn báo cáo thêm ở ngưỡng cố định "
        "chung (0,25) để so sánh công bằng theo cùng protocol."
    )

    # ========== CHƯƠNG 4 ==========
    add_heading_custom(doc, "CHƯƠNG 4. THỰC NGHIỆM VÀ KẾT QUẢ", level=1)

    add_heading_custom(doc, "4.1. Thiết lập thực nghiệm", level=2)
    add_para(
        doc,
        "Huấn luyện trên máy có GPU NVIDIA (RTX 4090), môi trường Python với Ultralytics "
        "YOLOv8. Protocol so sánh công bằng khóa imgsz = 640 và conf = 0,25 khi đánh giá "
        "cặp baseline–final trên cùng data.yaml gốc (Roboflow). Các vòng cải tiến trung gian "
        "có thể dùng imgsz khác khi huấn luyện, nhưng bảng so sánh chính báo cáo lại ở "
        "imgsz 640 để không thiên vị độ phân giải."
    )

    add_heading_custom(doc, "4.2. Kết quả định lượng baseline và mô hình cuối", level=2)
    add_table(
        doc,
        ["Mô hình", "Split", "mAP50", "mAP50-95", "Precision", "Recall", "F1"],
        [
            ["YOLOv8n baseline", "valid", "0,9946", "0,9310", "0,9909", "0,9955", "0,9932"],
            ["YOLOv8s v4 final", "valid", "0,9949", "0,8941", "0,9960", "0,9972", "0,9966"],
            ["YOLOv8n baseline", "test", "0,9919", "0,9193", "0,9947", "0,9953", "0,9950"],
            ["YOLOv8s v4 final", "test", "0,9918", "0,8823", "0,9910", "0,9965", "0,9938"],
        ],
    )
    add_caption(doc, "Bảng 4.1. So sánh công bằng baseline và mô hình cuối (imgsz=640, conf=0,25)")

    add_para(
        doc,
        "Trên test Roboflow, mAP50 của hai mô hình gần tương đương (khoảng 0,992). mAP50-95 "
        "của mô hình cuối thấp hơn baseline trên miền này, có thể liên quan tới việc tối ưu "
        "hướng tới độ bền trên ảnh khó/camera và tăng cường mạnh làm thay đổi phân bố hộp. "
        "Trên valid, Precision/Recall/F1 của mô hình cuối nhỉnh hơn. Điểm then chốt khi "
        "diễn giải: miền Roboflow đã bão hòa từ baseline, nên phần lớn giá trị cải tiến thể "
        "hiện ở demo camera và giảm lỗi nhầm lớp trong điều kiện thực — không kỳ vọng delta "
        "mAP lớn trên test studio."
    )

    try_add_image(
        doc, FIG / "fig_val_compare_overview.png", 14.0,
        "Hình 4.1. Tổng quan chỉ số trên tập valid (baseline vs final)"
    )
    try_add_image(
        doc, FIG / "fig_test_compare_overview.png", 14.0,
        "Hình 4.2. Tổng quan chỉ số trên tập test (baseline vs final)"
    )
    try_add_image(
        doc, FIG / "fig_val_AP50_per_class.png", 14.0,
        "Hình 4.3. AP50 theo từng mệnh giá trên valid"
    )
    try_add_image(
        doc, FIG / "fig_conf_sweep_val.png", 14.0,
        "Hình 4.4. Quét ngưỡng tin cậy trên valid"
    )
    try_add_image(
        doc, FIG / "fig_qualitative_real_cam.png", 14.0,
        "Hình 4.5. Minh họa định tính trên ảnh camera thật"
    )

    add_heading_custom(doc, "4.3. Ngưỡng tin cậy chọn trên valid", level=2)
    add_table(
        doc,
        ["Mô hình", "conf*", "Precision", "Recall", "F1", "Ghi chú"],
        [
            ["Baseline", "0,30", "≈0,989", "≈0,998", "≈0,994", "Tối thiểu hóa cost proxy trên valid"],
            ["Final v4", "0,50", "≈0,997", "≈0,997", "≈0,997", "Ngưỡng cao hơn, Precision tăng"],
        ],
    )
    add_caption(doc, "Bảng 4.2. Ngưỡng tin cậy chọn theo quét trên valid")

    add_heading_custom(doc, "4.4. Nhận xét kết quả", level=2)
    add_para(
        doc,
        "Baseline đã đủ mạnh trên dữ liệu Roboflow, xác nhận tính khả thi của hướng YOLO cho "
        "bài toán. Các vòng cải tiến không nhằm “phá kỷ lục” mAP trên test studio mà hướng "
        "tới giảm lỗi thực dụng: nhầm cặp mệnh giá gần màu, mất dấu khi camera rung/mờ, và "
        "nhấp nháy nhãn trên luồng video. Protocol đánh giá tách bạch train/valid/test giúp "
        "tránh kết luận thiên lệch. Chi tiết bảng CSV và mã tái lập nằm trong "
        "03_So_sanh_Baseline_vs_Final.ipynb."
    )

    # ========== CHƯƠNG 5 ==========
    add_heading_custom(doc, "CHƯƠNG 5. PHÂN TÍCH LỖI, HẠN CHẾ VÀ THẢO LUẬN", level=1)

    add_heading_custom(doc, "5.1. Phân tích lỗi", level=2)
    errors = [
        "Nhầm lớp giữa các mệnh giá có gam màu gần nhau khi ánh sáng lệch trắng hoặc bão hòa màu thấp.",
        "Bỏ sót hoặc hộp không ổn định khi tờ tiền chuyển động nhanh, ảnh mờ mạnh, hoặc chỉ lộ một phần tờ.",
        "Đa tờ chồng lấn / góc nghiêng lớn làm giảm IoU dù lớp đúng, ảnh hưởng mAP50-95.",
        "Letterbox nền trắng trên dữ liệu Roboflow tạo thiên lệch miền so với nền bàn, tay người, vật dụng xung quanh trên camera.",
    ]
    for e in errors:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(e)
        set_run_font(run, size=13)

    add_heading_custom(doc, "5.2. Hạn chế của giải pháp", level=2)
    limits = [
        "Tập test Roboflow quy mô vừa phải (~218 ảnh) và phân bố gần train/valid, nên sức phân biệt giữa hai mô hình trên chỉ số tổng là hạn chế.",
        "Ảnh camera thật đã gán nhãn còn ít (khoảng 18 ảnh gốc), dù đã tăng cường và fine-tune.",
        "Thiếu lớp 500 đồng so với đầy đủ mệnh giá lưu thông.",
        "Chưa tích hợp mô-đun phát hiện tiền giả hay đọc số sê-ri.",
        "Demo phụ thuộc điều kiện webcam, khoảng cách và ánh sáng của môi trường trình bày.",
    ]
    for e in limits:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(e)
        set_run_font(run, size=13)

    add_heading_custom(doc, "5.3. Thảo luận", level=2)
    add_para(
        doc,
        "Trong nhận dạng mẫu, việc chỉ số trên một miền bão hòa gần như trần không có nghĩa "
        "là bài toán đã giải xong. Khoảng cách miền và chi phí lỗi theo ngữ cảnh ứng dụng mới "
        "là thước đo bổ sung cần thiết. Chiến lược baseline → cải tiến lặp → đánh giá khóa "
        "trên test phản ánh đúng tinh thần kiểm soát quá khớp và khái quát hóa. Hướng cải "
        "thiện tiếp theo hợp lý là mở rộng dữ liệu camera có kiểm soát nhãn, cân bằng lớp, "
        "và có thể thêm nhánh kiểm chứng lớp trên vùng cắt (crop) bằng đặc trưng màu/kết cấu "
        "khi cần giải thích hoặc hậu xử lý."
    )

    # ========== CHƯƠNG 6 ==========
    add_heading_custom(doc, "CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", level=1)
    add_para(
        doc,
        "Nhóm đã xây dựng hoàn chỉnh một pipeline nhận dạng mẫu cho bài toán phát hiện mệnh "
        "giá tiền Việt Nam: dữ liệu có nguồn trích dẫn, EDA, mô hình cơ sở YOLOv8n, mô hình "
        "cải tiến YOLOv8s qua nhiều vòng tinh chỉnh kèm ảnh thực tế, đánh giá định lượng "
        "công bằng và demo tương tác. Kết quả trên miền Roboflow đạt mức cao; giá trị cải "
        "tiến tập trung vào độ bền khi triển khai thực tế và giảm một số lỗi nhầm lớp."
    )
    add_para(
        doc,
        "Hướng phát triển: bổ sung lớp 500 đồng; mở rộng tập ảnh camera đa điều kiện; thử "
        "kiến trúc thay thế hoặc hậu xử lý phân lớp trên crop; nghiên cứu ngưỡng thích nghi "
        "theo ngữ cảnh; tích hợp OCR sê-ri nếu có nhu cầu định danh tờ."
    )

    # ========== PHỤ LỤC A ==========
    add_page_break(doc)
    add_heading_custom(doc, "PHỤ LỤC A. PHÂN CÔNG NHIỆM VỤ VÀ TỰ ĐÁNH GIÁ ĐÓNG GÓP", level=1)
    add_table(
        doc,
        ["TT", "Mã SV", "Họ và tên", "Nhiệm vụ chính", "Tự đánh giá đóng góp (%)"],
        [
            ["1", "B26CHKH008", "Hoàng Minh Đức Anh", "Báo cáo, slide thuyết trình", "25"],
            ["2", "B26CHKH030", "Trần Việt Dũng",
             "Pipeline, huấn luyện model, báo cáo & slide, QA & Test", "25"],
            ["3", "B26CHKH059", "Vũ Hoàng Lâm",
             "Pipeline, huấn luyện model, gán nhãn metadata, QA & Test", "25"],
            ["4", "B26CHKH072", "Lê Khánh Minh",
             "Thu thập nguồn ảnh, gán nhãn metadata thủ công", "25"],
        ],
    )
    add_caption(doc, "Bảng A.1. Phân công và tỷ lệ đóng góp (tự đánh giá nhóm)")
    add_para(
        doc,
        "Github thành viên: Vũ Hoàng Lâm — https://github.com/Hglamvu ; "
        "Lê Khánh Minh — https://github.com/KhnhMinh .",
        first_line=False,
    )

    # ========== PHỤ LỤC B ==========
    add_heading_custom(doc, "PHỤ LỤC B. HƯỚNG DẪN CHẠY MÃ NGUỒN VÀ NOTEBOOK BÁO CÁO", level=1)
    add_para(doc, "Cấu trúc thư mục báo cáo:", first_line=False)
    add_para(
        doc,
        "report/\n"
        "  Bao_cao_Nhan_dang_tien_Viet_Nam.docx\n"
        "  notebooks/\n"
        "    01_EDA_Bao_cao.ipynb\n"
        "    02_Qua_trinh_Train_va_Cai_tien.ipynb\n"
        "    03_So_sanh_Baseline_vs_Final.ipynb\n"
        "  figures/   # biểu đồ so sánh\n"
        "  exports/   # CSV/JSON chỉ số",
        size=11, first_line=False, align="left",
    )
    add_para(
        doc,
        "Mã nguồn huấn luyện và demo nằm ở thư mục gốc dự án: train_baseline.py, train_v2.py, "
        "train_v3.py, train_v4.py, demo_camera.py. Cài đặt: pip install -r requirements "
        "(hoặc notebooks/requirements.txt và requirements_demo.txt). Chạy notebook theo thứ "
        "tự 01 → 02 → 03. Demo camera: python demo_camera.py --weights "
        "<đường_dẫn_best.pt> --hold-seconds 1.0 ."
    )

    # ========== TÀI LIỆU THAM KHẢO ==========
    add_heading_custom(doc, "TÀI LIỆU THAM KHẢO", level=1)
    refs = [
        "[1] Ultralytics, YOLOv8 Documentation, https://docs.ultralytics.com/",
        "[2] Roboflow Universe, Vietnamese Currency dataset (first-qkz37), "
        "https://universe.roboflow.com/first-qkz37/vietnamese-currency-lgi9i",
        "[3] Jocher, G. et al., Ultralytics YOLO, GitHub repository.",
        "[4] Vietanhdev, tien-giay-vn, https://github.com/vietanhdev/tien-giay-vn",
        "[5] Huy Pham Nguyen Thanh, Vietnamese Money, Kaggle Datasets.",
        "[6] Bao Vu, Vietnamese Banknote, Roboflow Universe.",
        "[7] Everingham, M. et al., The Pascal Visual Object Classes (VOC) Challenge, IJCV, 2010 "
        "(nền tảng metric AP/mAP).",
        "[8] Tài liệu học phần Nhận dạng mẫu — Khoa CNTT1, Học viện Công nghệ Bưu chính Viễn thông.",
    ]
    for r in refs:
        add_para(doc, r, size=12, first_line=False, space_after=4, align="left")

    doc.save(str(OUT))
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
