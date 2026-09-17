#!/usr/bin/env python3
"""Regenerate the existing supervisor-facing WeedMap progress report."""

from pathlib import Path
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "weedmap_progress_report.docx"
INK, NAVY, PALE, GRID = "17212B", "213B58", "F2F6FA", "D9D9D9"


def set_font(style, size, bold=False):
    style.font.name = "Arial"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor.from_string(INK)
    rpr = style._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for name in ("ascii", "hAnsi"):
        fonts.set(qn(f"w:{name}"), "Arial")
    fonts.set(qn("w:eastAsia"), "PingFang SC")


def setup(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.1), Cm(1.9)
    sec.left_margin, sec.right_margin = Cm(2), Cm(2)
    styles = doc.styles
    set_font(styles["Normal"], 10)
    styles["Normal"].paragraph_format.line_spacing = 1.25
    styles["Normal"].paragraph_format.space_after = Pt(5)
    set_font(styles["Title"], 20, True)
    styles["Title"].paragraph_format.space_after = Pt(12)
    ppr = styles["Title"]._element.get_or_add_pPr()
    border = ppr.find(qn("w:pBdr"))
    if border is not None:
        ppr.remove(border)
    for name, size, before, after in (("Heading 1", 13, 15, 6), ("Heading 2", 10.5, 9, 4)):
        set_font(styles[name], size, True)
        styles[name].paragraph_format.space_before = Pt(before)
        styles[name].paragraph_format.space_after = Pt(after)
        styles[name].paragraph_format.keep_with_next = True
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("WeedMap 实验进度报告")


def paragraph(doc, text):
    return doc.add_paragraph(text)


def section(doc, number, title):
    doc.add_heading(f"{number} {title}", level=1)


def shade(cell, color):
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(node)


def table(doc, headers, rows, widths, size=8):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, width in enumerate(widths):
        t.columns[i].width = Cm(width)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "5")
        edge.set(qn("w:color"), GRID)
        borders.append(edge)
    t._tbl.tblPr.append(borders)
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row_index, values in enumerate([headers, *rows]):
        row = t.rows[0] if row_index == 0 else t.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.width = Cm(widths[i])
            cell.text = str(value)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index == 0:
                shade(cell, NAVY)
            elif row_index % 2 == 0:
                shade(cell, PALE)
            mar = OxmlElement("w:tcMar")
            for side, amount in (("top", 95), ("bottom", 95), ("start", 90), ("end", 90)):
                item = OxmlElement(f"w:{side}")
                item.set(qn("w:w"), str(amount))
                item.set(qn("w:type"), "dxa")
                mar.append(item)
            cell._tc.get_or_add_tcPr().append(mar)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.08
                for run in p.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(size)
                    run.font.bold = row_index == 0
                    run.font.color.rgb = RGBColor.from_string("FFFFFF" if row_index == 0 else INK)
                    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "PingFang SC")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def metrics(doc, rows, size=7.5):
    table(doc, ["模型 / 损失", "Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"],
          rows, [4.2, 2.35, 2.35, 2.8, 2.5, 2.8], size)


def build():
    doc = Document()
    setup(doc)
    doc.core_properties.title = "WeedMap 多光谱语义分割实验进度报告"
    doc.add_paragraph("WeedMap 多光谱语义分割实验进度报告", style="Title")
    paragraph(doc, "面向导师的阶段性汇报。报告总结真实 WeedMap 数据处理、统一样本对照、三 seed 模型实验，以及基于植被覆盖率的后续模型路线。当前最佳设置为 MobileNetV2ShallowUNet 方案 A + boundary weighted CE r5_w4：验证集 Mean IoU 76.71% ± 0.55%，Weed IoU 59.82% ± 0.64%。")

    section(doc, 1, "项目目标与数据集")
    paragraph(doc, "项目目标是利用无人机 RGB 与多光谱影像，将每个有效像素分为 background、crop、weed，为田间杂草识别和精准除草提供区域信息。前期通过 NDVI、NDRE 模拟光谱差异，并用 synthetic 数据跑通分割流程；报告中的主要结论依据真实 WeedMap Tiles 实验。")
    paragraph(doc, "真实 tile 尺寸为 360 × 480。原始数据检查得到 1670 个候选样本；后续严格对照使用 RGB 与多光谱均可用的 WeedMap common split，共 454 张，固定划分为训练 363 张、验证 91 张。多光谱输入为 G、R、RedEdge、NIR、NDVI 五通道。")

    section(doc, 2, "真实 WeedMap 数据处理与标签映射")
    paragraph(doc, "已检查影像、GroundTruth 与有效区 mask 的空间对应关系。本地数据中 GroundTruth_color 的类别语义最明确，Dataset 从 color 标签生成 0/1/2 类别；mask=255 标记无效区域，在训练和评价中作为 ignore_index。")
    table(doc, ["数据", "原值", "训练含义"], [
        ("GroundTruth_color", "black / green / red", "background=0 / crop=1 / weed=2"),
        ("GroundTruth_iMap", "0 / 10000 / 2", "对应 background / crop / weed；10000 不是 ignore"),
        ("mask", "0 / 255", "有效区域 / 无效区域"),
    ], [4, 4.7, 8.3], 8.8)

    section(doc, 3, "RGB 与 multispectral 公平对比")
    paragraph(doc, "两组使用同一 454 张样本、相同 363/91 划分、SmallUNet、Weighted CE、类别权重 1/4/8、10 epochs 和 batch size 2。下表是验证集最后一轮结果；这是输入通道对照，不能与后文 20 epochs 的 best checkpoint 直接混为同一设置。")
    metrics(doc, [
        ("RGB", "94.28%", "65.27%", "95.49%", "65.75%", "34.56%"),
        ("Multispectral", "94.38%", "67.76%", "94.94%", "61.62%", "46.73%"),
    ], 8)
    paragraph(doc, "多光谱 Mean IoU 高 2.49 个百分点，Weed IoU 高 12.17 个百分点；RGB Crop IoU 高 4.13 个百分点。该结果支持后续以多光谱作为主输入，并保留对 crop 类差异的关注。")

    section(doc, 4, "SmallUNet baseline")
    paragraph(doc, "真实数据主 baseline：WeedMap common split，多光谱输入，SmallUNet，Weighted CE，类别权重 background/crop/weed=1/4/8，20 epochs，batch size 2，seed 0/1/2。每个 seed 按验证集 Mean IoU 选择 best checkpoint；下表的 ± 为三 seed 样本标准差。")
    metrics(doc, [("SmallUNet + Weighted CE", "96.38% ± 0.36%", "74.78% ± 0.82%", "96.70% ± 0.43%", "70.76% ± 1.11%", "56.89% ± 2.06%")])
    paragraph(doc, "三个 seed 的 best epoch 均为 15。早期 synthetic 普通 CE 的 Weed IoU 仅 0.26%，加入 class weights 后 5 epochs 达到 75.54%；这一模拟实验说明少数类权重的重要性，但数值不代表真实田间性能。")

    section(doc, 5, "Loss 对比：Weighted CE / Focal Loss / Dice + CE")
    paragraph(doc, "固定 common split、多光谱 SmallUNet、20 epochs、batch size 2、seed 0/1/2 和 best checkpoint 选择规则，仅比较损失函数。")
    metrics(doc, [
        ("Weighted CE", "96.38% ± 0.36%", "74.78% ± 0.82%", "96.70% ± 0.43%", "70.76% ± 1.11%", "56.89% ± 2.06%"),
        ("Focal Loss", "96.69% ± 0.26%", "74.00% ± 1.25%", "97.19% ± 0.11%", "69.27% ± 3.00%", "55.55% ± 1.72%"),
        ("Dice + CE", "96.73% ± 0.24%", "74.45% ± 3.37%", "97.21% ± 0.05%", "69.54% ± 3.82%", "56.61% ± 6.31%"),
    ])
    paragraph(doc, "Weighted CE 的平均 Mean IoU 和 Weed IoU 均最高，作为稳定 baseline。Dice + CE 的 seed 0/1 表现较强，但 seed 2 明显下降，跨 seed 波动较大；Focal Loss 的平均 Weed IoU 略低。")

    section(doc, 6, "Boundary error analysis")
    paragraph(doc, "对 SmallUNet + Weighted CE 的 20 epochs best checkpoint 分析 sample index=0，排除 label=255 后共 112158 个有效像素、5812 个错误像素。5px 边界区域含 38199 个有效像素，占有效像素 34.06%，却包含 5532 个错误，占全部错误 95.18%。")
    table(doc, ["区域", "错误像素", "区域错误率"], [
        ("5px 边界内", "5532", "14.48%"),
        ("5px 边界外", "280", "0.38%"),
    ], [7, 4.5, 5.5], 8.8)
    paragraph(doc, "该单张样本表明错误集中在边界，因而尝试对边界像素增加损失权重；它不能单独代表整体验证集。")

    section(doc, 7, "SmallUNet + boundary weighted CE r5_w4")
    paragraph(doc, "在 SmallUNet baseline 上使用 boundary_weighted_ce，边界半径 5px、边界权重 4.0，类别权重仍为 1/4/8。common split、多光谱、20 epochs、batch size 2、三 seed 与 best checkpoint 规则保持一致。")
    metrics(doc, [("SmallUNet + boundary CE r5_w4", "96.79% ± 0.16%", "75.17% ± 1.02%", "97.15% ± 0.11%", "70.07% ± 2.41%", "58.29% ± 0.92%")])
    paragraph(doc, "相对 SmallUNet + Weighted CE，Mean IoU 提高 0.39 个百分点，Weed IoU 提高 1.40 个百分点；Crop IoU 均值下降 0.69 个百分点。sample index=0 的 5px 边界错误率由 14.48% 降至 9.50%，总错误像素由 5812 降至 3746。")

    section(doc, 8, "MobileNetV2ShallowUNet 方案 A")
    paragraph(doc, "方案 A 是 MobileNetV2-style shallow encoder 改造实验，不是论文完整 MobileNetV2-U-Net 复现。它保留当前 SmallUNet 的 C1/C2、两级 Decoder、skip connection 和 segmentation head，以浅层 MobileNetV2-style inverted residual blocks B1-B6 替换原 enc2 与 bottleneck。完整 B1-B17 多尺度结构留作方案 B。")
    paragraph(doc, "参数量由 SmallUNet 的 117363 降至 107155，减少约 8.7%。在相同 common split、多光谱、Weighted CE、20 epochs、batch size 2、三 seed 条件下：")
    metrics(doc, [("方案 A + Weighted CE", "96.53% ± 0.47%", "75.86% ± 1.22%", "96.80% ± 0.44%", "71.74% ± 2.76%", "59.03% ± 1.74%")])
    paragraph(doc, "与原 SmallUNet + Weighted CE 比较，Mean IoU 高 1.08 个百分点，Weed IoU 高 2.14 个百分点。")

    section(doc, 9, "MobileNetV2ShallowUNet + boundary weighted CE r5_w4")
    paragraph(doc, "当前最强语义分割设置：WeedMap common split；multispectral；MobileNetV2ShallowUNet 方案 A；boundary_weighted_ce；boundary radius=5、boundary weight=4.0；类别权重 background=1.0、crop=4.0、weed=8.0；20 epochs、batch size=2、seed=0/1/2；每个 seed 按验证集 Mean IoU 选择 best checkpoint。")
    table(doc, ["Seed", "Best epoch", "Pixel Acc", "Mean IoU", "Background IoU", "Crop IoU", "Weed IoU"], [
        ("0", "20", "96.91%", "76.97%", "97.20%", "73.41%", "60.30%"),
        ("1", "12", "96.78%", "76.08%", "97.15%", "71.02%", "60.07%"),
        ("2", "18", "97.25%", "77.07%", "97.44%", "74.69%", "59.10%"),
        ("均值 ± 标准差", "—", "96.98% ± 0.24%", "76.71% ± 0.55%", "97.26% ± 0.16%", "73.04% ± 1.86%", "59.82% ± 0.64%"),
    ], [2.5, 2, 2.35, 2.35, 2.8, 2.5, 2.5], 7.4)
    doc.add_heading("与前序主线对比", level=2)
    metrics(doc, [
        ("SmallUNet + Weighted CE", "96.38% ± 0.36%", "74.78% ± 0.82%", "96.70% ± 0.43%", "70.76% ± 1.11%", "56.89% ± 2.06%"),
        ("SmallUNet + boundary CE r5_w4", "96.79% ± 0.16%", "75.17% ± 1.02%", "97.15% ± 0.11%", "70.07% ± 2.41%", "58.29% ± 0.92%"),
        ("方案 A + Weighted CE", "96.53% ± 0.47%", "75.86% ± 1.22%", "96.80% ± 0.44%", "71.74% ± 2.76%", "59.03% ± 1.74%"),
        ("方案 A + boundary CE r5_w4", "96.98% ± 0.24%", "76.71% ± 0.55%", "97.26% ± 0.16%", "73.04% ± 1.86%", "59.82% ± 0.64%"),
    ], 7.4)
    paragraph(doc, "相对原 SmallUNet + Weighted CE baseline 的均值提升（百分点）：Pixel Accuracy +0.60、Mean IoU +1.93、Background IoU +0.56、Crop IoU +2.28、Weed IoU +2.93。结果表明方案 A 的浅层 encoder 改造与 boundary weighted CE 在当前实验中可以叠加提升。")

    section(doc, 10, "VCR 植被覆盖率评判指标")
    paragraph(doc, "VCR = vegetation pixels / valid pixels。其中 vegetation pixels 指 NDVI > 0.2 且 label != 255 的像素；valid pixels 指 label != 255 的像素。VCR 用于场景划分，不参与分割模型训练或 IoU 计算。")
    table(doc, ["场景", "初始 VCR 阈值", "样本数", "比例"], [
        ("sparse", "VCR < 0.20", "13 / 454", "≈ 2.86%"),
        ("transition", "0.20 ≤ VCR ≤ 0.30", "9 / 454", "≈ 1.98%"),
        ("dense", "VCR > 0.30", "432 / 454", "≈ 95.15%"),
    ], [3.5, 6, 3.5, 4], 8.8)
    paragraph(doc, "WeedMap common split 的 454 张样本中，VCR 均值 0.7928、中位数 0.9116、最小值 0.0000、最大值 0.9999。绝大多数样本为高植被覆盖场景。0.20 和 0.30 是初始经验阈值，后续需结合人工样本检查与实际除草需求调整。")

    section(doc, 11, "YOLO / U-Net 路线选择标准")
    paragraph(doc, "无人机多光谱图像 → NDVI / 植物-土壤区分指标 → 计算 VCR → 判断 sparse / transition / dense → 选择定位或区域分割路线。")
    table(doc, ["VCR 场景", "候选路线", "除草用途"], [
        ("sparse", "YOLO", "单株或单簇定位，点状精准除草"),
        ("transition", "同时测试 YOLO 与 U-Net，或人工确认", "根据目标形态与作业要求选择"),
        ("dense", "U-Net / MobileNetV2ShallowUNet", "输出区域 mask，支持区域除草"),
    ], [3, 6.8, 7.2], 8.6)
    paragraph(doc, "common split 中 dense 占约 95.15%，因此当前阶段以 U-Net / MobileNetV2ShallowUNet 语义分割为主路线有数据依据。YOLO 适合作为低覆盖稀疏场景的候选路线或后续扩展；目前仅完成 smoke test，尚无与 U-Net 在相同条件下的正式效果对比。")

    section(doc, 12, "给老师汇报用总结")
    paragraph(doc, "老师，目前我已经完成从模拟数据到真实 WeedMap 多光谱无人机数据的语义分割流程。前期用 NDVI、NDRE 模拟作物、杂草和土壤的光谱差异，并在 synthetic 数据上训练 U-Net。普通 CE 容易忽视 weed，加入 class weights 后 Weed IoU 大幅提升。随后我整理真实 WeedMap Tiles，验证标签映射；本地 GroundTruth_color 的语义最清楚，因此 Dataset 由它生成 0/1/2 标签，mask=255 作为 ignore 区域。严格公平的 RGB 与多光谱对比使用相同的 454 个样本，其中训练 363 个、验证 91 个。多光谱的 Mean IoU 和 Weed IoU 均高于 RGB，说明额外光谱通道对杂草识别有帮助。")
    paragraph(doc, "真实数据中，Weighted CE 是稳定的基础 loss。分析 sample index=0 后发现，5px 边界区域包含 Weighted CE 的 95.18% 错误，因此加入 boundary weighted CE。SmallUNet 的 r5_w4 三 seed 平均 Mean IoU 达到 75.17%，Weed IoU 达到 58.29%。接着实现 MobileNetV2ShallowUNet 方案 A：这是浅层 MobileNetV2-style encoder 改造，保留当前两级 U-Net decoder，并非论文完整 MobileNetV2-U-Net 复现。方案 A 单用 Weighted CE 已优于原 SmallUNet；与 boundary weighted CE r5_w4 结合后，三 seed 平均 Mean IoU 为 76.71% ± 0.55%，Weed IoU 为 59.82% ± 0.64%，是目前最好的语义分割结果。")
    paragraph(doc, "我还新增了植被覆盖率 VCR 作为模型路线选择指标，通过 NDVI 区分植物与土壤并计算每张图的覆盖率。WeedMap common split 中 dense 样本占 95.15%，说明当前数据以高植被覆盖场景为主，继续优化 U-Net / MobileNetV2ShallowUNet 语义分割是合理的；YOLO 更适合作为低覆盖稀疏场景的补充路线，用于单株或单簇定位与点状精准除草。VCR 阈值仍需结合人工检查和实际作业需求校准。")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Updated {OUTPUT}")


if __name__ == "__main__":
    build()
