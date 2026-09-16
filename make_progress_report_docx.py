#!/usr/bin/env python3
"""Generate the supervisor-facing WeedMap progress report as a DOCX file."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "weedmap_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
SOURCE_MD = REPORTS_DIR / "weedmap_progress_report.md"
OUTPUT_DOCX = REPORTS_DIR / "weedmap_progress_report.docx"

NAVY = "17365D"
BLUE = "2F5597"
PALE_BLUE = "EAF1F8"
PALE_GRAY = "F5F7FA"
MID_GRAY = "D9E1E8"
TEXT_GRAY = "4B5563"
WHITE = "FFFFFF"
BLACK = "000000"
GREEN = "2F855A"

COPY_ASSETS = {
    "real_weedmap_sample_frame0070.png": "real_weedmap_sample_frame0070.png",
    "real_weedmap_prediction_multispectral_weighted_ce.png": "real_prediction_3epochs.png",
    "real_weedmap_prediction_multispectral_weighted_ce_10epochs.png": "real_prediction_10epochs.png",
    "synthetic_prediction.png": "synthetic_prediction.png",
    "synthetic_history_weighted.png": "synthetic_history_weighted.png",
    "synthetic_history_multispectral_weighted_ce.png": "synthetic_history_multispectral_weighted_ce.png",
    "synthetic_history_ce.png": "synthetic_history_ce.png",
    "synthetic_history_focal.png": "synthetic_history_focal.png",
    "synthetic_history_dice.png": "synthetic_history_dice.png",
}


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=MID_GRAY, size="5") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    element = OxmlElement("w:cantSplit")
    tr_pr.append(element)


def keep_paragraph(paragraph, keep_next=False, page_break_before=False) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    if keep_next:
        node = OxmlElement("w:keepNext")
        p_pr.append(node)
    if page_break_before:
        node = OxmlElement("w:pageBreakBefore")
        p_pr.append(node)


def set_run_font(run, latin="Arial", east_asia="Arial Unicode MS", size=None, bold=None, color=None) -> None:
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_style_font(style, latin, east_asia, size, bold=False, color=BLACK) -> None:
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor.from_string(color)
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, "Arial", "Arial Unicode MS", 10.5, color=BLACK)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.widow_control = True

    title = styles["Title"]
    set_style_font(title, "Arial", "Hiragino Sans GB", 24, bold=True, color=BLACK)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(16)
    title_ppr = title._element.get_or_add_pPr()
    border = title_ppr.find(qn("w:pBdr"))
    if border is not None:
        title_ppr.remove(border)

    for name, size, before, after in (
        ("Heading 1", 16, 16, 8),
        ("Heading 2", 12.5, 12, 5),
        ("Heading 3", 11, 9, 4),
    ):
        style = styles[name]
        set_style_font(style, "Arial", "Hiragino Sans GB", size, bold=True, color=BLACK)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    if "Figure Caption" not in styles:
        cap = styles.add_style("Figure Caption", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Figure Caption"]
    set_style_font(cap, "Arial", "Arial Unicode MS", 9, color=TEXT_GRAY)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(9)
    cap.paragraph_format.keep_with_next = False

    if "Small Note" not in styles:
        note = styles.add_style("Small Note", WD_STYLE_TYPE.PARAGRAPH)
    else:
        note = styles["Small Note"]
    set_style_font(note, "Arial", "Arial Unicode MS", 9.5, color=TEXT_GRAY)
    note.paragraph_format.line_spacing = 1.25
    note.paragraph_format.space_after = Pt(5)


def setup_page(section) -> None:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    set_run_font(run, size=8.5, color=TEXT_GRAY)


def configure_header_footer(section) -> None:
    header = section.header
    p = header.paragraphs[0]
    p.text = "cv-weedmap-segmentation  实验进度报告"
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in p.runs:
        set_run_font(run, size=8.5, color=TEXT_GRAY)
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(p)


def add_paragraph(doc, text="", *, bold_lead=None, style=None, align=None, indent=True):
    p = doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)
    if indent and style not in ("Small Note", "Figure Caption"):
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align is not None:
        p.alignment = align
    return p


def add_bullets(doc, items, level=0, numbered=False):
    for index, item in enumerate(items, 1):
        p = doc.add_paragraph(style="Normal" if numbered else "List Bullet")
        p.paragraph_format.left_indent = Cm(0.65 + level * 0.45)
        p.paragraph_format.first_line_indent = Cm(-0.38 if numbered else -0.25)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"{index}.  {item}" if numbered else item)
        set_run_font(r)


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        set_run_font(run, latin="Arial", east_asia="Hiragino Sans GB", bold=True, color=BLACK)
    keep_paragraph(p, keep_next=True)
    return p


def add_table(doc, headers, rows, widths=None, numeric_from=1, font_size=8.5, first_col_left=True):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    keep_row_together(table.rows[0])
    if widths:
        for col_idx, col_width in enumerate(widths):
            table.columns[col_idx].width = Cm(col_width)
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = str(header)
        set_cell_shading(cell, NAVY)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell)
        if widths:
            cell.width = Cm(widths[idx])
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            for run in p.runs:
                set_run_font(run, east_asia="Hiragino Sans GB", size=font_size, bold=True, color=WHITE)
    for row_idx, values in enumerate(rows):
        row = table.add_row()
        keep_row_together(row)
        cells = row.cells
        for col_idx, value in enumerate(values):
            cell = cells[col_idx]
            cell.text = str(value)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if widths:
                cell.width = Cm(widths[col_idx])
            if row_idx % 2:
                set_cell_shading(cell, PALE_GRAY)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if first_col_left and col_idx < numeric_from else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    set_run_font(run, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_figure(doc, path: Path, caption: str, number: int, max_width_cm=16.2):
    if not path.exists():
        return False
    with Image.open(path) as image:
        px_w, px_h = image.size
    width_cm = max_width_cm
    if px_h / max(px_w, 1) > 0.95:
        width_cm = min(max_width_cm, 14.5)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(0)
    keep_paragraph(p, keep_next=True)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = doc.add_paragraph(f"图 {number}  {caption}", style="Figure Caption")
    keep_paragraph(cap)
    return True


def add_key_value_table(doc, pairs):
    return add_table(doc, ["项目", "内容"], pairs, widths=[4.2, 11.6], numeric_from=2, font_size=9)


def copy_assets() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = ROOT / "outputs"
    for source_name, target_name in COPY_ASSETS.items():
        source = outputs / source_name
        target = ASSETS_DIR / target_name
        if source.exists():
            shutil.copy2(source, target)


def generate_training_curve() -> bool:
    csv_path = ROOT / "outputs" / "real_weedmap_history_multispectral_weighted_ce_10epochs.csv"
    output = ASSETS_DIR / "real_weedmap_10epoch_curves.png"
    if not csv_path.exists():
        return False
    records = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        return False
    epochs = [int(float(row["epoch"])) for row in records]
    loss = [float(row["train_loss"]) for row in records]
    metrics = {
        "Val mean IoU": [float(row["mean_iou"]) * 100 for row in records],
        "Background IoU": [float(row["background_iou"]) * 100 for row in records],
        "Crop IoU": [float(row["crop_iou"]) * 100 for row in records],
        "Weed IoU": [float(row["weed_iou"]) * 100 for row in records],
    }
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=180)
    axes[0].plot(epochs, loss, marker="o", color="#2F5597", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(epochs)
    axes[0].grid(alpha=0.25)
    colors = ["#2F5597", "#4C78A8", "#59A14F", "#E15759"]
    for (label, values), color in zip(metrics.items(), colors):
        axes[1].plot(epochs, values, marker="o", label=label, linewidth=2, color=color)
    axes[1].set_title("Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU (%)")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Real WeedMap multispectral U-Net 10-epoch history", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def add_cover(doc: Document) -> None:
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(style="Title")
    r = p.add_run("基于无人机多光谱影像的\n作物、杂草、土壤/背景区分实验进度报告")
    set_run_font(r, latin="Arial", east_asia="Hiragino Sans GB", size=24, bold=True, color=BLACK)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.25

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(30)
    r = p.add_run("cv-weedmap-segmentation 项目阶段性汇报")
    set_run_font(r, latin="Arial", east_asia="Hiragino Sans GB", size=14, color=TEXT_GRAY)

    cover_rows = [
        ("研究方向", "无人机农业遥感 / 多光谱图像语义分割"),
        ("当前阶段", "Synthetic 模拟实验完成，真实 WeedMap 数据读取、标签验证、Dataset 构建、U-Net 初步训练完成"),
        ("汇报内容", "项目构思、实验流程、模型方法、关键变量、实验结果、可视化分析、下一步计划"),
    ]
    table = add_table(doc, ["基本信息", "阶段说明"], cover_rows, widths=[3.2, 12.6], numeric_from=2, font_size=9.5)
    table.rows[0].cells[0].merge(table.rows[0].cells[1])
    for p in table.rows[0].cells[0].paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(16)
    r = p.add_run("阶段结论  真实多光谱 U-Net 已跑通，10 epochs 的 weed IoU 提升至 45.76%")
    set_run_font(r, east_asia="Hiragino Sans GB", size=11, bold=True, color=BLUE)
    doc.add_page_break()


def add_toc(doc: Document) -> None:
    add_heading(doc, "目录", level=1)
    entries = [
        "项目背景与研究目标", "整体实验路线", "光谱指数模拟实验", "光谱阈值 baseline",
        "Synthetic 语义分割实验", "Class weights 与 loss function 对比",
        "Synthetic RGB vs multispectral 对比", "真实 WeedMap 数据读取与结构检查",
        "真实标签映射验证", "WeedMap PyTorch Dataset", "真实 WeedMap U-Net 训练",
        "10 epochs 稳定性实验", "真实预测可视化与误差分析", "关键指标和变量解释",
        "当前结论", "下一步计划", "给导师汇报时可以说的话",
    ]
    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, entry in enumerate(entries, 1):
        cells = table.add_row().cells
        cells[0].width = Cm(1.3)
        cells[1].width = Cm(14.5)
        cells[0].text = f"{i:02d}"
        cells[1].text = entry
        for col, cell in enumerate(cells):
            set_cell_margins(cell, top=70, bottom=70)
            if i % 2 == 0:
                set_cell_shading(cell, PALE_GRAY)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col == 0 else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    set_run_font(run, east_asia="Hiragino Sans GB" if col == 0 else "Arial Unicode MS", size=10, bold=(col == 0), color=BLUE if col == 0 else BLACK)
    doc.add_page_break()


def add_route_table(doc: Document) -> None:
    steps = [
        "环境准备", "光谱指数\n模拟实验", "Synthetic\n语义分割", "真实数据读取\n与结构检查", "标签映射\n验证",
        "Dataset\n构建", "真实 U-Net\n训练", "预测可视化\n与误差分析", "RGB 与多光谱\n对比及改进",
    ]
    rows = []
    for idx, step in enumerate(steps, 1):
        rows.append((f"{idx:02d}", step, "已完成" if idx <= 8 else "下一阶段"))
    add_table(doc, ["步骤", "实验环节", "进度"], rows, widths=[1.7, 10.2, 3.9], numeric_from=2, font_size=9.5)


def build_document(curve_generated: bool) -> None:
    if not SOURCE_MD.exists():
        raise FileNotFoundError(f"主要内容参考文件不存在: {SOURCE_MD}")
    SOURCE_MD.read_text(encoding="utf-8")

    doc = Document()
    setup_page(doc.sections[0])
    configure_styles(doc)
    configure_header_footer(doc.sections[0])
    doc.core_properties.title = "基于无人机多光谱影像的作物 杂草 土壤背景区分实验进度报告"
    doc.core_properties.subject = "cv-weedmap-segmentation 项目阶段性汇报"
    doc.core_properties.keywords = "无人机农业遥感, 多光谱, 语义分割, WeedMap, U-Net"

    add_cover(doc)
    add_toc(doc)

    add_heading(doc, "1 项目背景与研究目标")
    add_paragraph(doc, "本项目面向无人机农业遥感场景，目标是利用无人机影像中的空间纹理、多光谱通道和植被指数，对实验田中的作物、杂草、土壤或其他背景进行像素级区分。当前阶段使用 WeedMap 公共数据验证完整方法流程，后续可迁移至水稻田或柑橘田，并根据实际传感器波段和标注体系进行适配。")
    add_paragraph(doc, "任务定义：本研究属于语义分割，不是整张图分类。输入是无人机 RGB 图像或多光谱通道，输出是每个像素的类别，类别包括 background、crop 和 weed。该任务与导师课题中的作物、杂草、土壤区分高度相关。", bold_lead="任务定义：")
    add_bullets(doc, [
        "输入：无人机 RGB 图像或多光谱通道。",
        "输出：与输入空间位置对应的像素级类别图。",
        "目标类别：background、crop、weed。",
        "阶段目标：验证数据读取、标签映射、训练、评价与可视化流程，并形成可迁移的实验基线。",
    ])

    add_heading(doc, "2 整体实验路线")
    add_paragraph(doc, "实验采用由可控模拟到真实数据的递进路线。先通过光谱指数与 synthetic 数据验证原理和代码，再进入真实 WeedMap 数据，避免数据格式、标签映射、类别不平衡与模型训练问题同时出现。")
    add_route_table(doc)
    add_paragraph(doc, "当前已完成真实预测可视化与误差分析。下一阶段重点比较真实 RGB 与 multispectral 输入，并继续调整损失函数、类别权重和训练轮数。", bold_lead="当前进度：")

    add_heading(doc, "3 光谱指数模拟实验")
    add_paragraph(doc, "为理解多光谱数据的作用，首先模拟三类地物的典型反射率，并计算 NDVI 与 NDRE。数值用于验证光谱差异的方向和阈值方法，不代表具体田块的实测反射率。")
    add_table(doc, ["class", "Green", "Red", "RedEdge", "NIR", "NDVI", "NDRE"], [
        ("crop", "0.1200", "0.0501", "0.2398", "0.5500", "0.8332", "0.3928"),
        ("weed", "0.1401", "0.0701", "0.2800", "0.5099", "0.7587", "0.2912"),
        ("soil", "0.2000", "0.2301", "0.2600", "0.2899", "0.1151", "0.0544"),
    ], widths=[2.2, 2.0, 2.0, 2.4, 2.0, 2.1, 2.1], font_size=8.5)
    add_key_value_table(doc, [
        ("Green", "绿光波段反射率"), ("Red", "红光波段反射率"),
        ("RedEdge", "红边波段，常用于植被状态分析"), ("NIR", "近红外波段，健康植被通常反射较强"),
        ("NDVI", "基于 Red 和 NIR 的植被指数"), ("NDRE", "基于 RedEdge 和 NIR 的植被指数"),
    ])
    add_paragraph(doc, "结果表明，soil 的 NDVI 和 NDRE 明显偏低，因此土壤与植被相对容易区分。crop 与 weed 都属于植被，NDVI 均较高，二者更难仅靠单一指数分开，需要进一步结合多光谱信息、形状和空间纹理。", bold_lead="实验结论：")

    add_heading(doc, "4 光谱阈值 baseline")
    add_paragraph(doc, "阈值 baseline 使用以下固定规则，作为后续复杂模型的可解释对照：")
    add_bullets(doc, ["NDVI < 0.4 → soil", "NDVI ≥ 0.4 且 NDRE ≥ 0.35 → crop", "NDVI ≥ 0.4 且 NDRE < 0.35 → weed"])
    add_table(doc, ["指标", "结果"], [("Overall accuracy", "99.84%"), ("Crop accuracy", "99.66%"), ("Weed accuracy", "100.00%"), ("Soil accuracy", "100.00%")], widths=[8.0, 7.8], numeric_from=1, font_size=9.5)
    add_paragraph(doc, "该结果来自具有明显光谱差异的模拟分布，不能直接代表真实田间效果。baseline 的价值是提供简单、可解释、可复现的参照，用于判断后续复杂模型是否带来真实增益。")

    add_heading(doc, "5 Synthetic 语义分割实验")
    add_heading(doc, "5.1 数据与模型设计", level=2)
    add_bullets(doc, [
        "background 模拟土壤背景；crop 模拟规则排列的作物行；weed 模拟随机分布的小目标杂草。",
        "加入噪声以模拟无人机航拍和光照变化，并支持 RGB 与 multispectral 两种输入。",
        "U-Net encoder 提取高级语义特征，decoder 恢复空间分辨率，skip connection 保留浅层边界与小目标细节。",
        "模型输出 3 类 logits，分别对应 background、crop、weed。",
    ])
    fig_no = 1
    if add_figure(doc, ASSETS_DIR / "synthetic_prediction.png", "Synthetic crop weed background 分割预测示例", fig_no):
        fig_no += 1
    add_heading(doc, "5.2 Synthetic baseline 结果", level=2)
    add_table(doc, ["Setting", "Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("CE, 5 epochs", "93.16%", "59.31%", "98.09%", "79.59%", "0.26%"),
    ], widths=[3.5, 2.3, 2.3, 2.8, 2.3, 2.3], font_size=8.2)
    add_paragraph(doc, "Pixel accuracy 达到 93.16%，但 weed IoU 仅为 0.26%。模型主要学会了占比较大的 background 和更明显的 crop，几乎没有学会少数类 weed，因此不能只用总体准确率判断模型质量。")

    add_heading(doc, "6 Class weights 与 loss function 对比")
    add_heading(doc, "6.1 Class weights 改进", level=2)
    add_paragraph(doc, "Class weights 提高 weed 错分时的损失，使优化过程更重视少数类。加权交叉熵在 5 epochs 已明显改善 weed，延长至 10 epochs 后进一步稳定。")
    add_table(doc, ["Setting", "Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("Weighted CE, 5 epochs", "97.05%", "88.57%", "96.55%", "93.60%", "75.54%"),
        ("Weighted CE, 10 epochs", "98.85%", "94.99%", "98.79%", "96.98%", "89.19%"),
    ], widths=[4.1, 2.1, 2.1, 2.7, 2.2, 2.2], font_size=8.0)
    add_paragraph(doc, "与 CE 5 epochs 相比，Weighted CE 5 epochs 将 weed IoU 从 0.26% 提升到 75.54%，说明类别权重能明显缓解当前 synthetic 数据中的类别不平衡。")
    if add_figure(doc, ASSETS_DIR / "synthetic_history_weighted.png", "Synthetic weighted CE 训练曲线", fig_no):
        fig_no += 1
    add_heading(doc, "6.2 Loss function 对比", level=2)
    add_table(doc, ["Loss", "Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("CE", "98.04%", "89.71%", "98.66%", "94.34%", "76.14%"),
        ("Weighted CE", "98.85%", "94.99%", "98.79%", "96.98%", "89.19%"),
        ("Dice", "97.23%", "85.51%", "98.64%", "91.62%", "66.27%"),
        ("Focal", "98.00%", "91.16%", "98.18%", "94.00%", "81.30%"),
    ], widths=[3.2, 2.3, 2.3, 2.8, 2.3, 2.3], font_size=8.2)
    add_bullets(doc, [
        "CE：普通交叉熵，进行逐像素多分类。",
        "Weighted CE：在 CE 中加入类别权重，提高少数类错误的代价。",
        "Dice Loss：关注预测区域与真实区域的重叠。",
        "Focal Loss：降低容易分类样本的贡献，更关注难分类样本。",
    ])
    add_paragraph(doc, "当前 synthetic 设置下 Weighted CE 整体最好，Focal Loss 次之。具体 loss 的效果依赖数据分布，仍需在真实 WeedMap 数据上复验。", bold_lead="结果判断：")

    add_heading(doc, "7 Synthetic RGB vs multispectral 对比")
    add_table(doc, ["Input", "Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("RGB + Weighted CE", "98.85%", "94.99%", "98.79%", "96.98%", "89.19%"),
        ("Multispectral + Weighted CE", "98.97%", "94.97%", "98.95%", "97.90%", "88.07%"),
    ], widths=[4.5, 2.1, 2.1, 2.7, 2.2, 2.2], font_size=8.0)
    add_paragraph(doc, "在 synthetic 数据中，多光谱输入没有在 mean IoU 和 weed IoU 上明显超过 RGB。可能原因是模拟 RGB 的颜色差异已经足够明显，额外波段的信息增益有限。这一结果不代表真实数据中多光谱无效，仍需开展真实数据对照实验。")
    if add_figure(doc, ASSETS_DIR / "synthetic_history_multispectral_weighted_ce.png", "Synthetic multispectral weighted CE 训练曲线", fig_no):
        fig_no += 1

    add_heading(doc, "8 真实 WeedMap 数据读取与结构检查")
    add_bullets(doc, [
        "已下载并整理 WeedMap Tiles，提取 RedEdge_000 至 RedEdge_004，以及 Sequoia_005 至 Sequoia_007。",
        "每张 tile 的大小为 360 × 480，标签与影像能够正确读取。",
        "主要目录包括 tile/RGB、tile/CIR、tile/B、tile/G、tile/R、tile/RE、tile/NIR、tile/NDVI、groundtruth 和 mask。",
    ])
    if add_figure(doc, ASSETS_DIR / "real_weedmap_sample_frame0070.png", "真实 WeedMap RedEdge_004 样本可视化，包括 RGB、NDVI、NIR、RedEdge、GroundTruth、mask 和 overlay，说明真实 UAV 多光谱图像与标签能够对齐读取", fig_no, 16.4):
        fig_no += 1

    add_heading(doc, "9 真实标签映射验证")
    add_paragraph(doc, "本地像素级核验确认了彩色真值、整数标签和有效区 mask 的含义。")
    add_table(doc, ["数据源", "数值或颜色", "类别含义"], [
        ("GroundTruth_color", "black", "background"), ("GroundTruth_color", "green", "crop"),
        ("GroundTruth_color", "red", "weed"), ("GroundTruth_iMap", "0", "background"),
        ("GroundTruth_iMap", "10000", "crop"), ("GroundTruth_iMap", "2", "weed"),
        ("mask", "0", "valid area"), ("mask", "255", "invalid / no-data area"),
    ], widths=[5.2, 4.2, 6.4], numeric_from=3, font_size=9)
    add_paragraph(doc, "不能简单把 iMap 中的 10000 当成 ignore。后续训练优先从 GroundTruth_color 生成 0/1/2 标签，再将 mask=255 的区域设为 ignore_index=255。标签语义与无效区域必须分开处理，否则会把 crop 错误忽略。", bold_lead="关键处理：")

    add_heading(doc, "10 WeedMap PyTorch Dataset")
    add_paragraph(doc, "项目已实现 weedmap_dataset.py。RGB 模式读取 tile/RGB 并输出 3 通道；multispectral 模式读取 tile/G、tile/R、tile/RE、tile/NIR 和 tile/NDVI，输出 5 通道。")
    add_key_value_table(doc, [
        ("image", "torch.float32，shape=[C,H,W]，归一化到 0～1"),
        ("label", "torch.long，shape=[H,W]"),
        ("label values", "0=background，1=crop，2=weed，255=ignore"),
        ("过滤规则", "跳过缺失文件、全黑图像、label 全为 ignore、有效像素过少或没有前景的样本"),
    ])
    add_table(doc, ["Dataset", "Loaded Samples", "Skipped Missing", "Skipped Empty/Invalid"], [
        ("RGB", "454", "700", "516"), ("Multispectral", "884", "0", "786"),
    ], widths=[4.2, 3.7, 3.7, 4.2], font_size=9)
    add_paragraph(doc, "第一个有效样本检查结果为 image max > 0，label unique values = [0, 1, 2, 255]；background=104834、crop=4568、weed=2756、ignore=60642。该统计也说明 background 明显多于 crop 和 weed。")

    add_heading(doc, "11 真实 WeedMap U-Net 训练")
    add_paragraph(doc, "项目已实现 train_real_weedmap_unet.py，并完成真实多光谱输入的首轮训练。")
    add_key_value_table(doc, [
        ("input_type", "multispectral"), ("loss", "weighted_ce"), ("epochs", "3"),
        ("batch_size", "2"), ("learning rate", "0.001"), ("optimizer", "Adam"),
        ("train / val samples", "707 / 177"), ("ignore_index", "255"),
        ("class weights", "background=1.0，crop=4.0，weed=8.0"),
    ])
    add_bullets(doc, [
        "epochs 表示完整遍历训练集的轮数；batch_size 表示每次送入模型的样本数；lr 控制参数更新步长。",
        "weighted_ce 是带类别权重的交叉熵；ignore_index=255 使无效区域不参与 loss 和评价。",
        "class weights 提高 crop 和 weed，尤其是 weed 错分时的损失权重。",
    ])
    add_table(doc, ["Epoch", "Train Loss"], [("1", "0.4497"), ("2", "0.2724"), ("3", "0.2532")], widths=[7.9, 7.9], font_size=9.5)
    add_table(doc, ["Val Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("95.61%", "56.37%", "97.15%", "56.28%", "15.68%"),
    ], widths=[3.2, 3.0, 3.4, 3.0, 3.2], numeric_from=0, font_size=8.7, first_col_left=False)
    add_paragraph(doc, "Train loss 连续下降，说明模型能够正常学习。background 最容易识别，crop 已有初步效果，weed IoU 较低，说明真实杂草比模拟数据更难识别。")

    add_heading(doc, "12 10 epochs 稳定性实验")
    epoch_rows = [
        ("1", "0.4497", "95.38%", "52.28%", "96.99%", "47.65%", "12.18%"),
        ("2", "0.2724", "95.25%", "52.53%", "96.84%", "49.52%", "11.24%"),
        ("3", "0.2532", "95.61%", "56.37%", "97.15%", "56.28%", "15.68%"),
        ("4", "0.2455", "95.12%", "56.41%", "96.76%", "50.87%", "21.59%"),
        ("5", "0.2364", "95.52%", "54.94%", "96.83%", "55.68%", "12.31%"),
        ("6", "0.2365", "95.50%", "54.14%", "96.81%", "53.91%", "11.70%"),
        ("7", "0.2257", "95.07%", "57.62%", "96.18%", "57.81%", "18.86%"),
        ("8", "0.2099", "96.40%", "62.65%", "97.17%", "62.13%", "28.66%"),
        ("9", "0.2019", "96.29%", "63.80%", "97.14%", "60.38%", "33.90%"),
        ("10", "0.1781", "97.22%", "69.21%", "97.69%", "64.17%", "45.76%"),
    ]
    add_table(doc, ["Epoch", "Train Loss", "Val Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], epoch_rows, widths=[1.4, 2.2, 2.6, 2.2, 2.8, 2.3, 2.3], numeric_from=0, font_size=7.2, first_col_left=False)
    curve_path = ASSETS_DIR / "real_weedmap_10epoch_curves.png"
    if curve_generated and add_figure(doc, curve_path, "真实 WeedMap 多光谱 U-Net 的 train loss 与验证集 mean background crop weed IoU", fig_no, 16.4):
        fig_no += 1
    else:
        add_paragraph(doc, "训练曲线图暂未生成。", style="Small Note", indent=False)
    add_table(doc, ["Setting", "Val Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("3 epochs", "95.61%", "56.37%", "97.15%", "56.28%", "15.68%"),
        ("10 epochs", "97.22%", "69.21%", "97.69%", "64.17%", "45.76%"),
    ], widths=[3.3, 2.5, 2.3, 2.8, 2.3, 2.3], font_size=8.0)
    add_paragraph(doc, "Weed IoU 从 15.68% 提升到 45.76%，mean IoU 从 56.37% 提升到 69.21%。增加训练轮数对真实 WeedMap 的 weed 类非常有效；中间轮次仍有波动，后续应通过更长训练和多 seed 实验判断稳定性。", bold_lead="主要结果：")

    add_heading(doc, "13 真实预测可视化与误差分析")
    three_epoch = ASSETS_DIR / "real_prediction_3epochs.png"
    if add_figure(doc, three_epoch, "真实 WeedMap 3 epochs 预测结果，包含输入、真值、预测与 error map", fig_no, 16.4):
        fig_no += 1
    if add_figure(doc, ASSETS_DIR / "real_prediction_10epochs.png", "真实 WeedMap 10 epochs 预测结果，weed 响应较 3 epochs 明显增加", fig_no, 16.4):
        fig_no += 1
    add_table(doc, ["Pixel Accuracy", "Background IoU", "Crop IoU", "Weed IoU", "Mean IoU"], [
        ("95.49%", "96.24%", "56.40%", "37.34%", "63.33%"),
    ], widths=[3.2, 3.4, 3.0, 3.0, 3.2], numeric_from=0, font_size=8.7, first_col_left=False)
    add_paragraph(doc, "上述指标对应 10 epochs 的 sample index=0。单样本指标低于整体验证集最终指标，说明不同样本难度存在差异。")
    add_bullets(doc, [
        "模型能识别出部分 crop 行状结构，10 epochs 后 weed 预测明显增多。",
        "大部分 background 预测正确。错误主要集中在作物与杂草边界，以及零散小目标 weed 周围。",
        "Weed 仍存在漏检；其面积小、分布零散且外观与作物相近，是当前真实农业遥感中最难的类别。",
    ])

    add_heading(doc, "14 关键指标和变量解释")
    add_table(doc, ["指标或变量", "含义与使用注意"], [
        ("Pixel Accuracy", "所有有效像素中预测正确的比例。背景占比大时，该值可能很高，但 weed 仍可能预测较差。"),
        ("IoU", "Intersection over Union，衡量某一类别预测区域与真实区域的交集占并集的比例。"),
        ("Mean IoU", "background、crop、weed 三类 IoU 的平均值，比 pixel accuracy 更能反映各类分割质量。"),
        ("Weed IoU", "本项目最重要的指标之一。杂草通常小且零散，容易与作物或土壤混淆。"),
        ("ignore_index=255", "无效区域不参与训练 loss 和指标计算，避免模型学习黑边或 no-data 区域。"),
        ("epochs", "完整遍历训练集的轮数。真实 weed 学习较慢，需观察更多轮次。"),
        ("class weights", "提高少数类错误对 loss 的贡献，缓解 background 占比过高造成的偏置。"),
    ], widths=[4.0, 11.8], numeric_from=2, font_size=9)

    add_heading(doc, "15 当前结论")
    conclusions = [
        "已经从 synthetic 实验推进到真实 UAV 多光谱语义分割数据。",
        "光谱指数实验说明土壤与植被较容易区分，但 crop 和 weed 都属于植被，二者更难区分。",
        "Synthetic 实验表明类别不平衡会导致模型忽视 weed。",
        "Class weights 能显著提升 weed IoU。",
        "真实 WeedMap 数据必须正确处理标签映射和 invalid/no-data mask。",
        "真实 U-Net 训练已经跑通，loss 正常下降。",
        "真实数据中 background 最容易，crop 次之，weed 最难。",
        "10 epochs 相比 3 epochs 明显提升 weed IoU。",
        "当前模型已具备初步区分作物、杂草、土壤或背景的能力，但 weed 仍有漏检与边界误差。",
        "后续需要比较真实 RGB 与 multispectral，调整 loss 与 class weights，增加训练轮数并开展多 seed 实验。",
    ]
    add_bullets(doc, conclusions, numbered=True)
    add_heading(doc, "15.1 当前项目完成内容", level=2)
    add_table(doc, ["阶段", "内容", "状态"], [
        ("环境搭建", "Python、PyTorch、MPS、VS Code、Codex、GitHub", "完成"),
        ("光谱基础", "NDVI/NDRE 模拟 crop/weed/soil", "完成"),
        ("阈值 baseline", "NDVI/NDRE 规则分类", "完成"),
        ("Synthetic segmentation", "crop/weed/background U-Net", "完成"),
        ("Class imbalance", "CE vs weighted CE", "完成"),
        ("Loss 对比", "CE / weighted CE / Dice / Focal", "完成"),
        ("输入对比", "RGB vs multispectral synthetic", "完成"),
        ("真实数据读取", "WeedMap Tiles 解压和结构检查", "完成"),
        ("标签验证", "color / iMap / mask 映射验证", "完成"),
        ("Dataset", "WeedMapDataset", "完成"),
        ("真实训练", "multispectral weighted CE 3/10 epochs", "完成"),
        ("预测可视化", "真实样本预测图和 error map", "完成"),
    ], widths=[4.2, 9.4, 2.2], numeric_from=2, font_size=8.5)

    add_heading(doc, "16 下一步计划")
    plans = [
        "真实 RGB vs multispectral 对比：保持划分、训练轮数和评价方式一致，比较 crop IoU、weed IoU 与 mean IoU。",
        "Loss 与 class weights 调参：比较 weighted CE、Focal、Dice，并测试 weed 权重 8、12、16。",
        "训练轮数与稳定性实验：比较 10 epochs 与 20 epochs，观察 weed IoU 是否继续提升及是否过拟合。",
        "多 seed 重复实验：降低单次数据划分与初始化的偶然性，使用 mean ± std 汇报。",
        "数据增强：加入 random flip、rotation、brightness/noise，提高对视角、光照与成像噪声的适应性。",
        "迁移到导师实验田：面向水稻田和柑橘田的作物、杂草、土壤区分，按实际传感器与标签调整输入和预处理。",
    ]
    add_bullets(doc, plans, numbered=True)

    add_heading(doc, "17 给导师汇报时可以说的话")
    speech = (
        "老师，目前我已经完成了从模拟数据到真实 WeedMap 多光谱无人机数据的完整语义分割流程。前期我先用 NDVI、NDRE 做了作物、杂草、土壤的光谱差异模拟，之后用 synthetic 数据训练 U-Net，发现普通 CE 会忽视 weed，加入 class weights 后 weed IoU 大幅提升。接着我下载并整理了真实 WeedMap Tiles 数据，完成了标签映射验证，发现本地数据中 color 标签最可靠，因此 Dataset 中从 GroundTruth_color 生成 0/1/2 标签，并把 mask=255 作为 ignore 区域。现在真实多光谱 U-Net 已经能训练，3 epochs 时 weed IoU 是 15.68%，10 epochs 提升到 45.76%，说明模型已经开始学习杂草区域，但 weed 仍然是最难类别。下一步我准备做 RGB 和 multispectral 对比，并继续调整 loss 和 class weights，提高 weed 分割效果。"
    )
    p = add_paragraph(doc, speech, indent=False)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.right_indent = Cm(0.8)
    p.paragraph_format.line_spacing = 1.5

    doc.save(OUTPUT_DOCX)


def main() -> None:
    copy_assets()
    curve_generated = generate_training_curve()
    build_document(curve_generated)
    print(f"Generated: {OUTPUT_DOCX}")
    print(f"Assets: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
