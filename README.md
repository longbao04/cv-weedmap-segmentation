# WeedMap 语义分割入门项目

本项目使用 WeedMap 公共数据集，学习无人机多光谱农田影像中的作物（crop）、杂草（weed）和背景（background）语义分割。

语义分割是**像素级分类**：为图像中的每个像素分配类别，不是给整张图片贴一个分类标签。输入 image 保存影像信息，mask 保存每个像素的类别，overlay 将标签颜色叠加到影像上，便于检查位置是否对应。

本项目采用以下演示类别编号；真实数据的标签编码需要在读取后核实，不能直接假设相同：

| 演示编号 | 类别 | 含义 |
| --- | --- | --- |
| 0 | background | 可近似理解为土壤/非目标背景 |
| 1 | crop | 作物 |
| 2 | weed | 杂草 |

WeedMap 的 sugar beet field（甜菜田）场景与导师提出的水稻/柑橘实验田任务具有共同目标：从无人机影像中区分作物、杂草和土壤/背景。可以先学习数据读取、多光谱通道、植被指数、标签可视化和分割评估，再迁移到实验田。不同作物、种植布局、传感器和拍摄条件存在差异，迁移时需要检查当地数据与标注，不能假设模型直接通用。

项目已用模拟数据跑通 U-Net 分割流程，并提供使用本地真实 WeedMap 数据训练的脚本。项目不包含或下载真实数据。数据理解摘要见 `dataset_notes.md`，后续安排见 `project_plan.md`，实验记录见 `synthetic_experiment_notes.md` 和 `real_experiment_notes.md`。

## 当前真实 WeedMap 结果与路线

以下训练结果均基于 WeedMap common split 的多光谱输入，三 seed（0/1/2）、20 epochs、batch size 2，并按验证集 mean IoU 选择各 seed 的 best checkpoint。表中为三 seed 均值 ± 样本标准差；提升均指百分点。VCR 是独立的场景评判指标，不属于模型训练结果。

### 1. SmallUNet baseline

`SmallUNet + weighted CE` 使用 background/crop/weed 类别权重 1/4/8，Mean IoU 为 74.78% ± 0.82%，Weed IoU 为 56.89% ± 2.06%。

### 2. Loss 对比：Weighted CE / Focal Loss / Dice + CE

| Loss（SmallUNet） | Mean IoU | Weed IoU |
| --- | ---: | ---: |
| Weighted CE | 74.78% ± 0.82% | 56.89% ± 2.06% |
| Focal Loss | 74.00% ± 1.25% | 55.55% ± 1.72% |
| Dice + CE | 74.45% ± 3.37% | 56.61% ± 6.31% |

### 3. Boundary error analysis

在 `SmallUNet + weighted CE` 的 sample index=0 上，距真实边界 5px 内的有效像素占 34.06%，却包含 95.18% 的错误。此项为单张样本分析，不能代替整体验证集指标。

### 4. SmallUNet + boundary weighted CE r5_w4

边界半径 5、边界权重 4.0 时，三 seed Mean IoU 为 75.17% ± 1.02%，Weed IoU 为 58.29% ± 0.92%。

### 5. MobileNetV2ShallowUNet 方案 A

方案 A 是 MobileNetV2-style shallow encoder 改造实验：保留当前 `SmallUNet` 的 C1/C2、两级 Decoder、skip connection 和 segmentation head，以浅层 inverted residual blocks B1～B6 替换原 `enc2` 和 bottleneck。它不是论文完整 MobileNetV2-U-Net 复现；完整 B1～B17 多尺度版本留作方案 B。搭配 weighted CE 时，三 seed Mean IoU 为 75.86% ± 1.22%，Weed IoU 为 59.03% ± 1.74%。

### 6. MobileNetV2ShallowUNet + boundary weighted CE r5_w4

设置：WeedMap common split、multispectral 输入、`model=mobilenetv2_shallow_unet`（方案 A）、`loss=boundary_weighted_ce`、`boundary_radius=5`、`boundary_weight=4.0`，类别权重 background 1.0、crop 4.0、weed 8.0；训练 20 epochs、batch size 2、seeds 0/1/2，各 seed 按验证集 mean IoU 选择 best checkpoint。

| Seed | Best epoch | Pixel accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 96.91% | 76.97% | 97.20% | 73.41% | 60.30% |
| 1 | 12 | 96.78% | 76.08% | 97.15% | 71.02% | 60.07% |
| 2 | 18 | 97.25% | 77.07% | 97.44% | 74.69% | 59.10% |

| 模型 / 损失 | Pixel accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmallUNet / weighted CE | 96.38% ± 0.36% | 74.78% ± 0.82% | 96.70% ± 0.43% | 70.76% ± 1.11% | 56.89% ± 2.06% |
| SmallUNet / boundary weighted CE r5_w4 | 96.79% ± 0.16% | 75.17% ± 1.02% | 97.15% ± 0.11% | 70.07% ± 2.41% | 58.29% ± 0.92% |
| MobileNetV2ShallowUNet 方案 A / weighted CE | 96.53% ± 0.47% | 75.86% ± 1.22% | 96.80% ± 0.44% | 71.74% ± 2.76% | 59.03% ± 1.74% |
| MobileNetV2ShallowUNet 方案 A / boundary weighted CE r5_w4 | **96.98% ± 0.24%** | **76.71% ± 0.55%** | **97.26% ± 0.16%** | **73.04% ± 1.86%** | **59.82% ± 0.64%** |
| 相对 SmallUNet / weighted CE 提升（百分点） | +0.60 | +1.93 | +0.56 | +2.28 | +2.93 |

目前最强语义分割结果来自 MobileNetV2ShallowUNet + boundary weighted CE r5_w4，在三 seed 上达到 Mean IoU 76.71% ± 0.55%、Weed IoU 59.82% ± 0.64%。这说明浅层 MobileNetV2-style encoder 与 boundary-aware loss 可以叠加提升，尤其对 weed 类识别更有帮助。

### 7. VCR 植被覆盖率评判指标

`VCR = vegetation pixels / valid pixels`，其中 vegetation pixels 满足 `NDVI > 0.2` 且 `label != 255`，valid pixels 满足 `label != 255`。VCR 用于场景划分，不参与模型训练指标计算。

| 场景 | 初始阈值 | 样本数 | 占 454 张比例 |
| --- | --- | ---: | ---: |
| sparse | VCR < 0.20 | 13 | 13 / 454 ≈ 2.86% |
| transition | 0.20 ≤ VCR ≤ 0.30 | 9 | 9 / 454 ≈ 1.98% |
| dense | VCR > 0.30 | 432 | 432 / 454 ≈ 95.15% |

common split 共 454 张，VCR 均值 0.7928、中位数 0.9116、最小值 0.0000、最大值 0.9999。阈值 0.20 / 0.30 是初始经验阈值，后续需结合人工样本检查和实际除草需求调整。

### 8. YOLO / U-Net 路线选择标准

无人机多光谱图像 → NDVI / 植物-土壤区分指标 → 计算 VCR → 判断 sparse / transition / dense：

| 场景 | 候选路线 | 用途 |
| --- | --- | --- |
| sparse | YOLO | 单株/单簇定位和点状精准除草 |
| dense | U-Net / MobileNetV2ShallowUNet | 区域 mask 和区域除草 |
| transition | 同时测试 YOLO 与 U-Net，或人工确认 | 根据目标形态选择 |

VCR 统计显示 WeedMap common split 以密集植被覆盖场景为主，因此当前阶段继续优化语义分割模型是合理的；YOLO 更适合作为低覆盖稀疏场景下的补充模型路线。YOLO 已完成 smoke test、5 epochs bbox 过滤对比和 weed-only vs crop+weed 对照，尚无与语义分割模型的正式同条件对比。

### YOLO baseline lightweight metrics

以下数值来自 `outputs/yolo_baseline_summary.csv`：检测指标取 5 epochs 训练结果最后一轮的 all-class 值，Params、GFLOPs（imgsz=480）和 Model Size 取各 run 的 `best.pt`。crop+weed 与 weed-only 的类别口径不同，不宜直接据此比较检测效果。

| Run | 检测类别 | Precision | Recall | mAP50 | mAP50-95 | Params | GFLOPs | Model Size (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r020_5epochs` | crop+weed | 0.49736 | 0.59299 | 0.52611 | 0.29074 | 3,011,238 | 4.608432 | 6.21425 |
| `r010_5epochs` | crop+weed | 0.50539 | 0.58950 | 0.53062 | 0.29557 | 3,011,238 | 4.608432 | 6.21425 |
| `weed_only_r010_5epochs` | weed-only | 0.43149 | 0.48721 | 0.42087 | 0.18213 | 3,011,043 | 4.6078272 | 6.213866 |

当前最佳 YOLO baseline 是 crop+weed `r010_5epochs`：mAP50 为 0.53062，mAP50-95 为 0.29557，模型参数量约 3.01M、GFLOPs 约 4.61、模型大小约 6.21 MB。

这些数值作为 MobileNetV3-YOLOv8n 轻量化改造的对照基线。该结构已完成输入归一化修正、5 epochs backbone freeze warm-up 和 20 epochs 全模型 fine-tuning，并在一致验证条件下比较 detection metrics（Precision、Recall、mAP50、mAP50-95）和 lightweight metrics（Params、GFLOPs、Model Size、inference speed）。当前项目主线仍是 MobileNetV2ShallowUNet + boundary CE r5_w4 语义分割。

## Next-stage experiment plan

### 1. Current completed stage

**A. Semantic segmentation main line.** 当前最强设置是 `MobileNetV2ShallowUNet + boundary weighted CE r5_w4`：Mean IoU = 76.71% ± 0.55%，Weed IoU = 59.82% ± 0.64%，Pixel accuracy = 96.98% ± 0.24%。这是目前 WeedMap common split 上最强的语义分割设置；相比 `SmallUNet + weighted CE`，Mean IoU 和 Weed IoU 均有提升。

**B. YOLO detection auxiliary line.** 当前最佳 YOLO baseline 是 `YOLOv8n crop+weed r010`：Precision = 0.50539，Recall = 0.58950，mAP50 = 0.53062，mAP50-95 = 0.29557，Params ≈ 3.01M，GFLOPs ≈ 4.61，Model size ≈ 6.21 MB。检测支线已完成 bbox statistics、r020/r015/r010 对比、weed-only 对照实验、prediction visualization、完整验证集 PR/F1 threshold sweep 和 lightweight baseline summary。

### 2. Short-term next experiments

**A. Segmentation robustness check.** 后续增加 random seeds，检查更多 validation samples 的预测图，分析错误是否仍主要集中在 crop/weed 边界区域，并统计最强分割模型的参数量、模型大小和推理速度。这些是后续验证计划，当前不启动训练。

**B. YOLO post-processing analysis.** 已完成 `YOLOv8n crop+weed r010` 的框数统计、完整验证集 PR/F1 sweep、confusion matrix 和 CPU inference speed 检查。平均 F1 最佳起点为 `conf≈0.16`；`conf=0.40` 仅作为更干净的展示候选。部署前仍需根据实际误喷与漏喷成本及目标硬件重新校准。

**C. MobileNetV3-YOLOv8n design.** 已实现 MobileNetV3-Small backbone + YOLOv8 PAN-FPN + anchor-free Detect head，并完成 ImageNet 输入归一化适配、5 epochs backbone freeze warm-up 和 20 epochs 全模型 fine-tuning。统一验证得到 mAP50=0.4810、mAP50-95=0.2106；参数量和 GFLOPs 分别比 YOLOv8n 少约 20.6% 和 44.7%，但精度仍低且当前 CPU inference 更慢，因此保留为轻量化结构研究结果，尚不替代 r010 baseline。

### 3. Medium-term paper/report structure

1. Dataset and preprocessing
2. Multispectral semantic segmentation
3. Boundary-aware loss analysis
4. Lightweight MobileNetV2-style U-Net
5. Detection dataset construction from semantic masks
6. YOLOv8n detection baseline
7. Future lightweight YOLO design

### 4. Overall conclusion

当前项目主线仍是语义分割，因为 WeedMap common split 中 dense vegetation samples 占多数。YOLO 检测支线主要作为 sparse vegetation scenarios 的候选路线，以及后续 MobileNetV3-YOLOv8n 轻量化检测研究的基础。

## 实验报告

[`synthetic_segmentation_report.md`](synthetic_segmentation_report.md) 是 synthetic U-Net 分割实验总结报告，汇总了 baseline、class weights、训练稳定性、loss function 对比，以及 RGB 与 multispectral 输入对比结果。

## 项目进度报告

面向导师汇报的完整实验进度、真实 WeedMap 训练结果与后续计划见 [`reports/weedmap_progress_report.md`](reports/weedmap_progress_report.md)。

## 导师汇报版 Word 报告

生成命令：

```bash
python make_progress_report_docx.py
```

生成的 Word 文档位于 `reports/weedmap_progress_report.docx`，图片会复制到 `reports/assets/` 并直接嵌入文档。该文件可使用 Microsoft Word 或 macOS Pages 打开。

## 项目结构

```text
cv-weedmap-segmentation/
├── README.md
├── project_plan.md
├── dataset_notes.md
├── synthetic_experiment_notes.md
├── real_experiment_notes.md
├── synthetic_segmentation_report.md
├── inspect_dataset_structure.py
├── analyze_real_weedmap_labels.py
├── analyze_vegetation_coverage.py
├── build_common_sample_list.py
├── visualize_sample_placeholder.py
├── visualize_real_weedmap_sample.py
├── visualize_real_weedmap_prediction.py
├── visualize_yolo_detection_labels.py
├── visualize_yolo_predictions.py
├── weedmap_dataset.py
├── synthetic_dataset.py
├── unet.py
├── losses.py
├── metrics.py
├── train_synthetic_unet.py
├── train_real_weedmap_unet.py
├── plot_synthetic_history.py
├── plot_real_loss_comparison.py
├── visualize_synthetic_prediction.py
├── requirements.txt
└── .gitignore
```

以后将 WeedMap Tiles 数据放到 `data/weedmap/`；`data/`、`models/` 和 `outputs/` 已忽略，不提交到 Git。

## 本地运行

使用已安装相应依赖的 Python 环境。依赖列表保存在 `requirements.txt`；当前阶段不执行联网安装。

```bash
python inspect_dataset_structure.py --data-root data/weedmap
python visualize_sample_placeholder.py
python -m py_compile inspect_dataset_structure.py visualize_sample_placeholder.py
```

第一个脚本在数据缺失时会给出提示；第二个脚本使用 numpy 构造假数据，显示 image、mask 和 overlay 三张图，不读取 WeedMap 数据。

## 真实 WeedMap 样本可视化

分析 8 个本地真实 WeedMap Tiles 子集的标签分布、color → iMap 像素级映射、有效区域 mask，并将逐样本统计保存到 `outputs/real_weedmap_label_summary.csv`：

```bash
python analyze_real_weedmap_labels.py
```

默认数据根目录为 `data/weedmap`；脚本只读取本地数据，不训练模型。可通过 `--data-root`、`--output` 和 `--subsets` 调整输入与输出。

本地已有 `data/weedmap/RedEdge_004/004` 数据时，可读取默认的 `frame0070`，显示 RGB、NDVI、NIR、RedEdge、彩色真值、iMap、有效区域 mask，以及叠加在 RGB 上的 weed 区域：

```bash
python visualize_real_weedmap_sample.py
```

图片默认保存到 `outputs/real_weedmap_sample_frame0070.png`，终端同时打印该样本 GT_iMap 的 unique values 和 counts。可通过 `--data-root`、`--sample-id` 和 `--output` 指定其他数据目录、样本及输出路径。

## 真实 WeedMap Dataset

`WeedMapDataset` 从 `data/weedmap` 读取真实 WeedMap Tiles 数据，支持 RGB 和五通道多光谱输入，并将彩色真值转换为训练标签。运行内置检查：

```bash
python weedmap_dataset.py
```

脚本会分别测试 RGB 与 multispectral Dataset，并打印样本数以及首个样本的 image/label shape、dtype、数值范围、标签 unique values 和各类像素数。Dataset 默认使用 `filter_empty=True` 过滤全黑输入、全 ignore 标签、有效像素过少或没有 crop/weed 前景的空样本。缺少必要输入或标签的样本仍会自动跳过；初始化输出会分别说明 `skipped missing samples` 和 `skipped empty/invalid samples`，因部分子集缺少 RGB 而被跳过的样本也计入前者。

为 RGB 与 multispectral 公平对比构建共享样本列表：

```bash
python build_common_sample_list.py
```

脚本扫描 8 个真实数据子集，只保留 RGB、G/R/RE/NIR/NDVI、彩色标签和 mask 均存在，且两种输入非全黑、标签含有效 crop/weed 前景的样本。结果保存到自动创建的 `splits/real_weedmap_common_samples.csv`，字段为 `sensor,subset_id,sample_id`；终端打印总数、各子集数量及 crop/weed 像素统计。可用 `--data-root` 和 `--output` 指定路径。

统计 common split 中每张多光谱图像的有效区域 NDVI 植被覆盖率（VCR）：

```bash
python analyze_vegetation_coverage.py --ndvi-threshold 0.2
```

脚本按 seed=0 的 80/20 规则标记 train/val，输出 `outputs/weedmap_vegetation_coverage_summary.csv` 和 `outputs/weedmap_vegetation_coverage_histogram.png`，并打印总体统计。

## YOLO detect baseline 数据准备

YOLO 是稀疏场景的候选检测路线和辅助探索路线。数据集脚本默认使用 crop + weed 两类；当前 YOLO baseline 暂时保留 crop+weed detection，weed-only 作为对照实验。运行以下命令生成或重新生成 YOLO bbox 数据集：

```bash
python prepare_yolo_detection_dataset.py --overwrite
```

脚本读取同一份共享样本列表，按 U-Net 的 seed=0 和 80/20 规则划分（当前为 train 363 张、val 91 张），直接复制原始 RGB 图到 `data/yolo_weedmap_detect/images/{train,val}`，并在 `labels/{train,val}` 写出对应标签和空目标图片的空 `.txt`。默认格式为 `0=crop`、`1=weed`，不输出 background；使用八连通区域生成 bbox。脚本现有默认值会跳过 component 面积小于 20 像素、框宽或高小于 4 像素、框面积超过图像面积 25% 的框；这些只是 smoke test 使用的默认值，尚未经 bbox 分布统计与可视化验证，不能当作最终过滤规则。可通过 `--min-area`、`--min-box-width`、`--min-box-height`、`--max-box-area-ratio` 调整，并可用 `--skip-border-touching` 跳过接触图像边界的框。终端输出保留的 crop/weed 框数，以及 `skipped small boxes`、`skipped huge boxes`、`skipped border boxes` 数量。`data.yaml` 写在输出目录。还可用 `--data-root`、`--sample-list-csv`、`--output-dir`、`--seed` 调整；已有非空输出目录时需使用 `--overwrite`，否则请选择空目录。此步骤仅准备检测数据，不训练模型。

新增 `--target-classes weed_only` 版本 `data/yolo_weedmap_detect_weed_only_r010`，只导出 weed 框并映射为 class 0，用于测试只检测 weed 是否比 crop+weed detection 更适合精准除草。它仍来自 semantic mask connected components，并非人工 instance bbox；5 epochs 对照结果见下文。

### YOLO bbox statistics before optimization

运行 `python analyze_yolo_bbox_statistics.py` 分析当前检测数据集的标签质量，逐框结果保存到 `outputs/yolo_bbox_statistics.csv`，四张分布图保存到 `reports/assets/`。当前 YOLO 标签来自 semantic mask 的 connected components，并非人工 instance bbox。统计结果供后续决定 min-area、max-area-ratio、min-width、min-height 等过滤规则；此步骤不会自动修改过滤规则。

### YOLO 检测流程：数据集构建阶段与推理后处理阶段

**A. Detection Dataset 构建：** WeedMap semantic mask → target mask → connected components → component bbox → bbox statistics → visualization / statistical analysis → dataset bbox filtering rules → YOLO labels → Detection Dataset。

WeedMap 原始标签是 semantic segmentation mask，没有 instance identity。按目标类别生成 target mask 后，connected component 只能近似地自动生成 bbox，不能默认一个 component 对应一株独立 weed 或 crop。稀疏场景中，独立小型 component 更可能接近单株或单簇目标；密集或粘连区域的大面积 component 可能包含多株植株或片状杂草区域，不宜解释为单株。此阶段的 filtering 是**数据集构建阶段的 bbox 清洗**，目的是从 semantic mask 生成较合理的 YOLO 训练标签。确定小框过滤阈值前，应先统计 bbox width、height、area、aspect ratio、bbox/image area ratio 的分布，再结合框叠加可视化检查确定规则，避免误删真实的小 weed。当前脚本尚未完成这套阈值验证。

**B. YOLO 训练 / 推理：** Detection Dataset → YOLOv8n smoke test / training → network candidate predictions → confidence filtering → NMS → final bounding boxes。

YOLO 网络输出候选框、类别和置信度；confidence filtering 去掉低置信度预测框，NMS 去掉高度重叠的重复预测框。这两步是**模型推理阶段的预测框后处理**，不属于 backbone，也不属于 U-Net 或 YOLO 的特征提取网络；与 A 阶段的训练标签 bbox 清洗是两种不同操作。YOLOv8n 目前只用于验证 mask → connected components → bbox → YOLO dataset → training/inference 流程，不是论文 MobileNetV3-YOLOv3 的复现。common split 的 VCR 统计以 dense 场景为主，当前主线仍是 U-Net / MobileNetV2ShallowUNet 语义分割。

训练 YOLO 前，先检查转换后的检测框：

```bash
python visualize_yolo_detection_labels.py
```

默认在 `RedEdge_004_frame0070` 的 RGB 图上以绿色绘制 crop 框、红色绘制 weed 框，保存到 `outputs/yolo_label_visualization_RedEdge_004_frame0070.png`。可用 `--dataset-dir`、`--split`、`--sample-name` 和 `--output` 指定其他样本及输出路径。

### YOLO smoke test

使用已准备好的 YOLO 检测数据和本地 `yolov8n.pt` 权重，运行一轮 smoke test：

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/yolo_weedmap_detect/data.yaml \
  epochs=1 \
  imgsz=480 \
  batch=4 \
  device=mps \
  project=runs/yolo_weedmap \
  name=smoke_test
```

本次 smoke test 只验证 YOLO 数据格式和训练流程可用，不用于与 U-Net 正式比较。YOLO 标签由 segmentation mask 自动转换，并非人工 bbox 标注，后续检测结果需要谨慎解释。结果记录见 [`real_experiment_notes.md`](real_experiment_notes.md)。

### YOLO bbox filtering comparison: r020 vs r015 vs r010

两组均使用 YOLOv8n，epochs=5、imgsz=480、batch=4、device=mps、seed=0；train images=363、val images=91。仅调整数据集构建阶段的 `max-box-area-ratio`，其余 bbox 过滤参数相同：

| 设置 | max-box-area-ratio | min-area | min-box-width | min-box-height |
| --- | ---: | ---: | ---: | ---: |
| r020 baseline | 0.20 | 80 | 6 | 6 |
| r015 | 0.15 | 80 | 6 | 6 |
| r010 | 0.10 | 80 | 6 | 6 |

| 设置 | 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| r020 baseline | all | 0.498 | 0.594 | 0.526 | 0.291 |
| r020 baseline | crop | 0.525 | 0.650 | 0.595 | 0.370 |
| r020 baseline | weed | 0.470 | 0.538 | 0.457 | 0.212 |
| r015 | all | 0.508 | 0.587 | 0.530 | 0.297 |
| r015 | crop | 0.522 | 0.672 | 0.616 | 0.387 |
| r015 | weed | 0.492 | 0.505 | 0.445 | 0.206 |
| r010 | all | 0.505 | 0.591 | 0.531 | 0.296 |
| r010 | crop | 0.527 | 0.652 | 0.601 | 0.376 |
| r010 | weed | 0.482 | 0.530 | 0.460 | 0.215 |

r015 的 all mAP50（0.530）和 r010（0.531）几乎持平，mAP50-95 略高，但 weed recall、weed mAP50 和 weed mAP50-95 均低于 r010。综合整体与 weed 类指标，r010 继续作为当前最佳过滤设置；三组差异很小，不能声称过滤阈值带来显著提升。weed-only detection 对照见下节。YOLO 当前仍是 detection pipeline 和稀疏场景候选路线；WeedMap common split 仍以 dense 场景为主，语义分割主线仍是 MobileNetV2ShallowUNet + boundary CE r5_w4。

### YOLO weed-only vs crop+weed comparison

对比 dataset A（crop+weed r010）与 dataset B（weed-only r010）的 weed class 检测结果。两组均使用 YOLOv8n，epochs=5、imgsz=480、batch=4、device=mps、seed=0；bbox 标签均由 WeedMap semantic mask 自动生成。

| 数据集 / 检测设置 | Weed P | Weed R | Weed mAP50 | Weed mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| A：crop+weed r010 | 0.482 | 0.530 | 0.460 | 0.215 |
| B：weed-only r010 | 0.431 | 0.488 | 0.421 | 0.182 |

在当前数据构建方式、YOLOv8n、5 epochs 和 seed=0 的设置下，weed-only detection 没有超过 crop+weed detection：后者在 weed precision、recall、mAP50 和 mAP50-95 上均更高。这不构成对 weed-only 路线的最终否定。当前 YOLO baseline 暂时保留 crop+weed detection 为主要检测设置，weed-only 作为对照实验记录；YOLO 仍是稀疏场景候选路线，当前主线仍是 MobileNetV2ShallowUNet + boundary CE r5_w4。

### YOLO prediction visualization and confidence threshold observation

`visualize_yolo_predictions.py` 对同一个 validation sample（`--sample-index 0`）比较训练 5 epochs 的 crop+weed r010 与 weed-only r010。图中包含 RGB image、crop+weed r010 ground truth 与 prediction、weed-only r010 ground truth 与 prediction。可运行 `python visualize_yolo_predictions.py --sample-index 0 --conf 0.25 --iou 0.7`；结果保存为 `outputs/yolo_prediction_comparison_sample0.png`。

在 sample index=0 上，crop+weed r010 的预测框数量相对更克制，输出更干净，重复预测较少；weed-only r010 更倾向于产生密集的 weed 预测框，容易出现更多候选框或重复框。这一定性现象与验证集定量结果方向一致：

| 检测设置（weed class） | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| crop+weed r010，5 epochs | 0.482 | 0.530 | 0.460 | 0.215 |
| weed-only r010，5 epochs | 0.431 | 0.488 | 0.421 | 0.182 |

将 confidence threshold 从 0.25 提高到 0.40 后，低置信度预测框明显减少，可视化结果更干净。`--conf` 是 YOLO 推理阶段控制最终输出框数量的后处理参数。提高阈值可能减少误检和重复框，也可能增加漏检；目前不能认定 `conf=0.40` 最优，只将其作为可视化观察和后续推理后处理调参的候选设置。可用 `python visualize_yolo_predictions.py --sample-index 0 --conf 0.40 --iou 0.7` 查看；脚本使用同一输出路径，保存前需保留需要对比的旧图。

confidence filtering 和 NMS 属于 YOLO inference post-processing，不属于 backbone，也不属于 dataset bbox filtering；它们与前述 semantic mask → connected components → bbox 过程中的数据集过滤不同。当前 YOLO baseline 暂时保留 crop+weed r010 为主要检测设置，weed-only r010 作为对照实验记录。YOLO 仍定位为稀疏场景候选路线和 detection pipeline 探索；项目主线仍是 MobileNetV2ShallowUNet + boundary CE r5_w4 语义分割。这些框数量与观感仅是 sample index=0 的可视化观察，不能替代整个验证集指标；整体结论仍以 validation metrics 为主。

### YOLO post-processing threshold analysis plan

运行 `python analyze_yolo_postprocessing_thresholds.py --sample-limit 20`，对当前 crop+weed r010 最佳 YOLO baseline 在排序后的前 20 张 validation images 上比较候选 `conf=0.25/0.30/0.40/0.50` 与 NMS `iou=0.50/0.60/0.70`；逐组合统计预测框总数、crop/weed 框数和平均置信度，保存到 `outputs/yolo_postprocessing_threshold_summary.csv`。confidence filtering 和 NMS IoU threshold 是 YOLO 推理阶段后处理参数，不属于 backbone，也不同于 dataset bbox filtering。该分析用于观察不同设置对预测框数量、weed 框数量和潜在重复预测的影响；框数变化本身不能证明重复框或误检减少，后续仍需结合可视化和验证指标。当前仅作为推理阶段分析与候选设置，不改变训练结果，也不指定最终最优阈值。

### YOLO post-processing threshold analysis results

基于当前最佳 YOLO baseline `YOLOv8n crop+weed r010`，使用前 20 张 validation images 统计不同 confidence threshold 与 NMS IoU threshold 组合的推理结果；完整数据见 `outputs/yolo_postprocessing_threshold_summary.csv`。下表固定 NMS `iou=0.50`：

| Confidence threshold | Mean boxes/image | Crop boxes | Weed boxes | Mean confidence |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 34.25 | 263 | 422 | 0.4140 |
| 0.30 | 25.15 | 179 | 324 | 0.4644 |
| 0.40 | 14.40 | 95 | 193 | 0.5539 |
| 0.50 | 8.60 | 55 | 117 | 0.6243 |

confidence threshold 是当前比较中影响预测框数量的主要因素：从 `conf=0.25` 提高到 `conf=0.40`，每张图平均预测框数从 34.25 降到 14.40，weed 预测框数从 422 降到 193；可视化结果明显更干净。

固定 `conf=0.40` 时，NMS IoU threshold 的影响较小：

| NMS IoU threshold | Mean boxes/image | Weed boxes |
| ---: | ---: | ---: |
| 0.50 | 14.40 | 193 |
| 0.60 | 14.55 | 195 |
| 0.70 | 14.65 | 196 |

在这 20 张 validation images 上，调整 NMS IoU threshold 对预测框数量的影响较小，confidence threshold 更关键。`conf=0.40, iou=0.50` 可作为当前 YOLO prediction visualization 的候选设置：它比 `conf=0.25` 更干净，同时不像 `conf=0.50` 那样大幅减少预测框。

**限制：**本分析只统计预测框数量和平均置信度，没有直接计算 TP、FP、FN，因此不能说明 `conf=0.40` 是最终最优阈值。最终阈值仍需结合人工可视化、PR/F1 曲线或验证集 detection metrics 判断。

完整验证集 PR/F1 sweep 使用 `python analyze_yolo_validation_curves.py --plots`。r010 best checkpoint 的平均 F1 在 `conf≈0.164` 达到最高值 0.544；crop 与 weed 各自的最佳 F1 阈值分别约为 0.164 和 0.169。候选 `conf=0.25/0.30/0.40/0.50` 的平均 F1 分别为 0.509、0.466、0.364、0.247，`conf=0.40` 时平均 precision 约 0.747，但平均 recall 仅约 0.241。因此 `conf=0.40, iou=0.50` 只保留为干净展示候选；需要平衡 precision/recall 的验证或部署默认值应从 `conf≈0.16` 开始，并根据实际误喷与漏喷成本进一步校准。完整汇总保存到 `outputs/yolo_validation_curve_summary.csv`；当前 CPU 完整验证测得 inference 约 10.4 ms/image。

### YOLO baseline summary for future MobileNetV3-YOLOv8n comparison

运行 `python summarize_yolo_baselines.py` 汇总 crop+weed r020、r015、r010 和 weed-only r010 四个 5 epochs 实验，结果保存到 `outputs/yolo_baseline_summary.csv`。检测指标取各 run 的 `results.csv` 最后一轮 all-class 值；Params、GFLOPs（imgsz=480）和模型大小取各自的 `weights/best.pt`。两类检测的 all-class 指标与 weed-only 指标口径不同，跨设置比较 weed 表现时应参考上文的 weed class 结果。

这些结果是 MobileNetV3-YOLOv8n 轻量化改造的 baseline；已完成的两阶段训练结果见下节。后续结构调整仍需统一输入尺寸、设备和测速条件，同时比较 detection metrics（precision、recall、mAP50、mAP50-95）与 lightweight metrics（Params、FLOPs、model size、inference speed）。

### MobileNetV3-Small-YOLOv8 architecture experiment

`configs/mobilenetv3_small_yolov8.yaml` 使用 torchvision MobileNetV3-Small backbone，提取 P3/8、P4/16、P5/32 特征，保留 YOLOv8-style PAN-FPN 与 anchor-free Detect head。backbone 使用官方 ImageNet 预训练权重；模型入口显式完成 ImageNet mean/std 归一化，避免将 YOLO 的 0–1 输入直接送入预训练 backbone。`check_mobilenetv3_yolov8.py` 用于结构与前向检查，`train_mobilenetv3_yolov8.py` 用于两阶段训练，`compare_yolo_architectures.py` 在统一验证条件下生成 `outputs/yolo_architecture_comparison.csv`。

| 模型 | P | R | mAP50 | mAP50-95 | Params | GFLOPs@480 | Model size | CPU inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n r010, 5 epochs | 0.5049 | 0.5909 | 0.5306 | 0.2956 | 3,011,238 | 4.61 | 6.21 MB | 10.4 ms/image |
| MobileNetV3-Small-YOLOv8 r010, 初始 5 epochs（归一化修正前） | 0.2251 | 0.2594 | 0.1284 | 0.0325 | 2,390,742 | 2.55 | 5.04 MB | 25–41 ms/image |
| MobileNetV3-Small-YOLOv8 r010, 两阶段训练 | 0.4773 | 0.5269 | 0.4810 | 0.2106 | 2,390,748 | 2.55 | 5.04 MB | 40.8 ms/image |

两阶段训练先冻结 backbone 5 epochs 预热新 neck/head，再从最佳 warm-up 权重解冻训练 20 epochs。修正归一化并充分训练后，mAP50 从 0.1284 恢复到 0.4810，与 YOLOv8n 的绝对差距缩小到 0.0496；mAP50-95 仍低 0.0850，表明高 IoU 下的框定位精度仍是主要短板。新结构参数量减少约 20.6%、GFLOPs 减少约 44.7%、模型文件减少约 18.9%，但当前 Apple CPU 实测延迟约为 baseline 的 3.9 倍，说明较低理论计算量不等于当前后端上更快。现阶段不能宣称该结构总体优于 YOLOv8n；r010 YOLOv8n 继续作为检测基线，MobileNetV3 结果保留用于后续目标硬件部署测试与 neck/head 改进。

## 训练真实 WeedMap U-Net

使用本地真实 WeedMap 数据、现有 `WeedMapDataset` 和 `SmallUNet` 训练三分类模型：

```bash
python train_real_weedmap_unet.py --input-type multispectral --loss weighted_ce --epochs 3
python train_real_weedmap_unet.py --input-type multispectral --loss weighted_ce --epochs 10 --batch-size 2 --save-path models/real_weedmap_multispectral_weighted_ce_10epochs.pth
python train_real_weedmap_unet.py --input-type multispectral --loss weighted_ce --weed-weight 12 --sample-list-csv splits/real_weedmap_common_samples.csv
python train_real_weedmap_unet.py --input-type multispectral --loss focal --epochs 20 --sample-list-csv splits/real_weedmap_common_samples.csv
python train_real_weedmap_unet.py --input-type multispectral --loss dice_ce --epochs 20 --sample-list-csv splits/real_weedmap_common_samples.csv
python train_real_weedmap_unet.py --input-type multispectral --loss boundary_weighted_ce --boundary-radius 5 --boundary-weight 3.0 --epochs 20 --sample-list-csv splits/real_weedmap_common_samples.csv
```

默认读取 `data/weedmap`，按固定随机种子划分 80% 训练集和 20% 验证集。多光谱输入使用 G、R、RedEdge、NIR、NDVI 五个通道；RGB 输入使用三个通道。标签 255 作为 `ignore_index`，不参与 loss 或验证指标，也不会被当作第 4 类。支持普通 CE（`ce`）、加权 CE（`weighted_ce`）、Focal Loss（`focal`）、Dice+CE（`dice_ce`）和边界加权 CE（`boundary_weighted_ce`）；默认 weighted CE 权重为 background=1.0、crop=4.0、weed=8.0。Focal Loss 使用 `gamma=2.0` 且不使用 alpha；Dice+CE 中的 CE 部分不使用 class weights。边界加权 CE 根据有效标签上下左右相邻类别变化确定边界，以欧氏半径膨胀；普通有效像素权重为 1，边界附近权重由 `--boundary-weight` 控制（默认 3.0），`--boundary-radius` 默认为 5 像素。

公平比较 RGB 与 multispectral 时，可让两次训练使用同一份共享样本列表：

```bash
python train_real_weedmap_unet.py --input-type rgb --sample-list-csv splits/real_weedmap_common_samples.csv
python train_real_weedmap_unet.py --input-type multispectral --sample-list-csv splits/real_weedmap_common_samples.csv
```

每个 epoch 输出 train loss、验证集 pixel accuracy、mean IoU 和三类 IoU。训练结束时保存最后一轮模型；验证集 mean IoU 创新高时同时保存 best checkpoint。默认最后一轮模型路径为 `models/real_weedmap_<input_type>_<loss>.pth`，best 模型在文件名后缀前加 `_best`；例如 `--save-path models/xxx.pth` 对应 `models/xxx_best.pth`。history 保存到 `outputs/real_weedmap_history_<input_type>_<loss>.csv`，其中 `is_best` 标记当时刷新最佳 mean IoU 的 epoch。`models/` 和 `outputs/` 会自动创建且已被 Git 忽略。可用 `--data-root`、`--batch-size`、`--lr`、`--seed` 和 `--num-workers` 调整训练参数。

### MobileNetV2ShallowUNet（方案 A）

`--model` 默认为 `small_unet`，保持以上实验的模型和默认文件名。选择 `--model mobilenetv2_shallow_unet` 时使用方案 A：保留当前 `SmallUNet` 的 C1/C2、Decoder、skip connection 和 segmentation head，用浅层 MobileNetV2-style inverted residual blocks B1～B6 替换原来的 `enc2` 与 bottleneck。这是 MobileNetV2-style shallow encoder 改造实验，**不是论文完整 MobileNetV2-U-Net 复现**；完整 B1～B17 多尺度版本留作方案 B。新模型的默认权重和 history 文件名会加入 `mobilenetv2_shallow_unet`，避免覆盖 `SmallUNet` 实验。可运行 `python check_mobilenetv2_shallow_unet_shapes.py` 检查 5 通道、360×480 输入的各级 shape；预测可视化脚本也支持同名 `--model` 参数。

方案 A 的张量路径（高×宽×通道）：`输入 360×480×5 → e1 360×480×16 → B3 180×240×24 → e2 adapter 24→32 → e2 180×240×32 → B6 90×120×32 → center adapter 32→64 → center 90×120×64`。Decoder 保持原结构：`center 90×120×64 → up2 180×240×32 → concat e2 180×240×64 → dec2 180×240×32 → up1 360×480×16 → concat e1 360×480×32 → dec1 360×480×16 → head 360×480×3`。

`SmallUNet` 共 117,363 个参数，`MobileNetV2ShallowUNet` 共 107,155 个参数，减少 10,208 个，约 8.7%。由于原 `SmallUNet` 已经很小，这属于**轻微减少参数量**，不能称为大幅轻量化。

正式实验使用 WeedMap common split、multispectral 输入、`weighted_ce`（background=1.0、crop=4.0、weed=8.0）、20 epochs、batch size 2、seed 0/1/2；按验证集 mean IoU 选择每个 seed 的 best checkpoint。

| 模型 / seed | Best epoch | Pixel accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 方案 A / 0 | 15 | 96.04% | 74.86% | 96.31% | 71.21% | 57.04% |
| 方案 A / 1 | 14 | 96.57% | 75.51% | 96.92% | 69.29% | 60.30% |
| 方案 A / 2 | 20 | 96.98% | 77.22% | 97.17% | 74.73% | 59.75% |

| 三 seed 统计 | Pixel accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmallUNet weighted CE | 96.38% ± 0.36% | 74.78% ± 0.82% | 96.70% ± 0.43% | 70.76% ± 1.11% | 56.89% ± 2.06% |
| 方案 A weighted CE | 96.53% ± 0.47% | 75.86% ± 1.22% | 96.80% ± 0.44% | 71.74% ± 2.76% | 59.03% ± 1.74% |
| 方案 A 相对提升（百分点） | +0.15 | +1.08 | +0.10 | +0.98 | +2.14 |

seed2 best checkpoint 的 sample0 单张可视化结果：pixel accuracy 96.43%、background IoU 96.70%、crop IoU 68.74%、weed IoU 45.66%、mean IoU 70.36%。该单张结果与上表整体验证集结果的统计范围不同。

方案 A 在三个随机种子上整体优于原 `SmallUNet` weighted CE baseline，尤其提升 weed IoU，说明浅层 MobileNetV2-style encoder 对当前 WeedMap 多光谱语义分割任务有效。方案 B 后续可考虑完整 B1～B17 和更深的多尺度 Decoder；详细记录见 [`real_experiment_notes.md`](real_experiment_notes.md)。

## 真实 WeedMap 预测可视化

使用已训练的真实 WeedMap U-Net 模型可视化单张样本的输入、NDVI、真值、预测、错误区域和预测叠加图：

```bash
python visualize_real_weedmap_prediction.py
```

脚本默认加载 `models/real_weedmap_multispectral_weighted_ce.pth`，读取过滤后的第 0 个 multispectral 样本，并将图片保存到 `outputs/real_weedmap_prediction_multispectral_weighted_ce.png`。终端会打印忽略标签 255 后的 pixel accuracy、background/crop/weed IoU 和 mean IoU。可通过 `--data-root`、`--input-type`、`--loss`、`--model-path`、`--sample-index`、`--sample-list-csv` 和 `--output` 调整运行参数；使用 RGB 输入时需要指定与三通道模型对应的权重文件。

同一样本对比主要模型的预测效果：`python compare_best_model_predictions.py --sample-index 0`。脚本使用 WeedMap common split、固定 multispectral 输入及 seed 0 验证子集，输出 `outputs/best_model_prediction_comparison_sample0.png` 和各模型单样本指标，辅助展示 boundary loss 与 MobileNetV2ShallowUNet 的改进；缺失的 checkpoint 会在图中标注并在终端提示。

### 主要模型预测结果可视化对比

在同一个 WeedMap common validation sample（seed 0，sample index=0）上，对比 RGB visualization、NDVI、Ground Truth、四个主要语义分割模型的 prediction 和对应 error map。下表是这张样本的指标，非整体验证集结果。

| 模型 / 损失 | Pixel accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmallUNet + weighted CE | 97.95% | 71.66% | 98.36% | 70.37% | 46.24% |
| SmallUNet + boundary CE r5_w4 | 97.80% | 68.20% | 98.25% | 56.38% | 49.96% |
| MobileNetV2ShallowUNet + weighted CE | 98.24% | 74.20% | 98.62% | 72.50% | 51.49% |
| MobileNetV2ShallowUNet + boundary CE r5_w4 | 98.21% | 74.45% | 98.38% | 63.04% | **61.92%** |

![主要模型在 sample index=0 上的预测与错误图对比](reports/assets/best_model_prediction_comparison_sample0.png)

在 sample index=0 上，MobileNetV2ShallowUNet + boundary CE r5_w4 获得最高的 weed IoU，为 61.92%，说明该组合在该样本上对 weed 类识别更有帮助。MobileNetV2ShallowUNet + weighted CE 的 mean IoU 为 74.20%，MobileNetV2ShallowUNet + boundary CE r5_w4 的 mean IoU 为 74.45%，二者接近，但 boundary loss 明显提高了该样本上的 weed IoU。

这是单个 validation sample 的可视化对比，只用于定性展示和辅助解释；最终整体结论仍以三 seed 验证集平均结果为主。目前主结果为 MobileNetV2ShallowUNet + boundary weighted CE r5_w4：Mean IoU 76.71% ± 0.55%，Weed IoU 59.82% ± 0.64%。

分析真实预测错误与 GroundTruth 边界的距离：

```bash
python analyze_boundary_errors.py
```

脚本默认使用 20 epochs best checkpoint、共同样本列表中的第 0 张 multispectral 样本，统计忽略标签 255 后的总错误率及距边界 1、3、5 像素内的错误数，并将六宫格可视化保存为 `outputs/boundary_error_analysis_sample0.png`。

指定模型权重和可视化输出路径：

```bash
python analyze_boundary_errors.py \
  --model-path models/real_weedmap_common_multispectral_boundary_weighted_ce_r5_w3_20epochs_seed0_best.pth \
  --output outputs/boundary_error_analysis_boundary_weighted_ce_sample0.png
```

## 真实 WeedMap loss 对比曲线

使用已有的三份 multispectral、seed=0、20 epochs history CSV 绘制验证集 mean IoU 和 weed IoU 曲线：

```bash
python plot_real_loss_comparison.py
```

图片分别保存到 `outputs/real_weedmap_loss_comparison_seed0.png` 和 `reports/assets/real_weedmap_loss_comparison_seed0.png`；缺少任何输入 CSV 时会报出对应路径。

## 训练过程记录与曲线

训练时每个 epoch 结束后保存 history，`outputs/` 不存在时自动创建，训练结束后打印 CSV 路径：

```bash
python train_synthetic_unet.py --epochs 5
python train_synthetic_unet.py --epochs 5 --use-class-weights
```

普通训练保存到 `outputs/synthetic_history_rgb_ce.csv`，加权训练保存到 `outputs/synthetic_history_rgb_weighted_ce.csv`。记录包含 epoch、average_train_loss、pixel_accuracy、mean_iou、background_iou、crop_iou 和 weed_iou；accuracy 与 IoU 使用 0～1 的数值。同一种模式重新训练会覆盖对应 history，模型也按 loss 名称分别保存。

训练后绘制对应曲线：

```bash
python plot_synthetic_history.py
python plot_synthetic_history.py --use-class-weights
```

脚本显示六个指标的子图，并分别保存到 `outputs/synthetic_history_rgb_ce.png` 或 `outputs/synthetic_history_rgb_weighted_ce.png`。CSV 不存在时会提示对应训练命令。训练曲线帮助观察 loss 是否下降、mIoU 是否提升，尤其是 weed IoU 随 epoch 的变化，便于比较普通与加权训练的效果。

## Synthetic segmentation baseline

`SyntheticWeedDataset` 用 NumPy 生成 128×128 的 RGB 农田图像：棕色土壤背景、规则排列的作物、随机小块杂草，并加入光照变化和噪声。图像 tensor 为 `[3,128,128]`、数值范围为 0～1；mask 为 `[128,128]` 的 long tensor，类别编号为 0/1/2。

小型 U-Net 通过 encoder 提取特征、decoder 恢复分辨率，skip connection 保留空间细节。输出为 `[B,3,128,128]` 类别分数，使用 CrossEntropyLoss 和 Adam 训练；预测时沿类别维取 argmax 得到 mask。

在已具备 `requirements.txt` 中依赖的本地 Python 环境中运行，不需要下载数据：

```bash
python train_synthetic_unet.py --epochs 5
python visualize_synthetic_prediction.py
```

训练参数还包括 `--batch-size`（默认 16）与 `--lr`（默认 0.001）。程序自动选择 MPS 或 CPU，使用 256 张训练图像和 64 张独立种子的测试图像，每个 epoch 输出平均训练 loss 和整份测试集的指标。权重保存为 `models/synthetic_unet_rgb_ce.pth`。

默认不使用 class weights。加入 `--use-class-weights` 后使用 weighted CrossEntropyLoss，类别权重为 background=1.0、crop=2.0、weed=6.0。对少数类 weed 给予更高错误惩罚，缓解类别不平衡：

```bash
python train_synthetic_unet.py --epochs 5 --use-class-weights
python visualize_synthetic_prediction.py --use-class-weights
```

加权模型保存为 `models/synthetic_unet_rgb_weighted_ce.pth`，可视化通过同一开关加载对应模型；模型文件不存在时会提示文件路径和训练命令。普通与加权模型的可视化分别保存到 `outputs/synthetic_prediction_rgb_<loss>.png`。

pixel accuracy 是像素级准确率，即预测正确像素占总像素的比例。IoU 是语义分割常用指标，表示某类别预测区域与真实区域的交集除以并集；mIoU 是所有类别 IoU 的平均值。若某类在预测和真值中均不存在，该类 IoU 记为 NaN，不参与平均；背景占比高时应结合 crop、weed IoU 判断效果。

可视化使用三个新的模拟样本，显示 RGB image、true mask、predicted mask 和 error map（红色表示错误），并保存到 `outputs/synthetic_prediction_rgb_<loss>.png`。模拟数据只用于理解 image → mask → 模型 → 指标 → 可视化流程，指标不能代表真实农田表现。

## Loss Function 对比实验

当前比较 CrossEntropyLoss（`ce`）、Weighted CrossEntropyLoss（`weighted_ce`）、Dice Loss（`dice`）和 Focal Loss（`focal`），重点观察少数类 weed IoU，并结合 mean IoU 判断整体分割效果。保持数据种子、epochs、batch-size 和 lr 一致，不修改 U-Net 主体结构。

- `ce`：普通交叉熵，默认选项，进行逐像素分类。
- `weighted_ce`：类别权重为 background=1.0、crop=2.0、weed=6.0，提高少数类 weed 分错时的惩罚。
- `dice`：更关注预测区域和真实区域的重叠。本实现先按类别计算 soft Dice，再对包括 background 的三个类别平均，平滑项为 1e-6。
- `focal`：更关注难分类样本，通过降低容易分类像素的贡献帮助优化，默认 gamma=2，未额外叠加类别权重。

Dice Loss 和 Focal Loss 常用于类别不平衡或小目标分割任务，是否改善 weed 识别需要通过实验确认。`losses.py` 接收原始类别分数 `[B,3,H,W]` 和类别 mask `[B,H,W]`，默认 `num_classes=3`、`ignore_index=None`；设置忽略标签时，对应像素不参与损失。

训练命令：

```bash
python train_synthetic_unet.py --epochs 10 --loss ce
python train_synthetic_unet.py --epochs 10 --loss weighted_ce
python train_synthetic_unet.py --epochs 10 --loss dice
python train_synthetic_unet.py --epochs 10 --loss focal
```

模型和历史记录分别保存如下；重新训练同一种 input type + loss 组合会覆盖其模型和 CSV：

| loss | 模型 | 训练历史 CSV |
| --- | --- | --- |
| ce | `models/synthetic_unet_rgb_ce.pth` | `outputs/synthetic_history_rgb_ce.csv` |
| weighted_ce | `models/synthetic_unet_rgb_weighted_ce.pth` | `outputs/synthetic_history_rgb_weighted_ce.csv` |
| dice | `models/synthetic_unet_rgb_dice.pth` | `outputs/synthetic_history_rgb_dice.csv` |
| focal | `models/synthetic_unet_rgb_focal.pth` | `outputs/synthetic_history_rgb_focal.csv` |

每个 epoch 继续打印 average train loss、pixel accuracy、mean IoU、background IoU、crop IoU 和 weed IoU。不同损失的数值尺度不同，应通过相同评估指标比较效果，不直接比较 loss 大小。

画曲线命令（图片保存为 `outputs/synthetic_history_rgb_<loss>.png`）：

```bash
python plot_synthetic_history.py --loss dice
python plot_synthetic_history.py --loss focal
```

可视化预测命令：

```bash
python visualize_synthetic_prediction.py --loss dice
python visualize_synthetic_prediction.py --loss focal
```

可视化按输入类型与 loss 保存图片，避免不同实验互相覆盖。模型或 CSV 不存在时，脚本会提示对应文件路径和训练命令。

三个脚本均保留 `--use-class-weights` 兼容旧命令，推荐使用 `--loss weighted_ce` 作为新方式；如果同时传入两种参数，旧开关优先，最终使用 `weighted_ce`。原先的 `synthetic_unet.pth` / `synthetic_unet_weighted.pth` 和 `synthetic_history_baseline.csv` / `synthetic_history_weighted.csv` 不会自动加载或迁移，新实验使用上表中的文件名；已有实验指标继续保留在实验记录中。


## RGB 输入 vs 多光谱输入实验

导师课题强调利用作物、杂草、土壤的光谱差异进行区分。本阶段通过 `--input-type rgb/multispectral` 比较输入形式，默认 `rgb`，继续支持 `--loss ce/weighted_ce/dice/focal` 和旧开关 `--use-class-weights`。目标是从 RGB 输入过渡到更接近无人机多光谱影像的输入形式。

- RGB：输入为 `[3,128,128]`，依次为 Red、Green、Blue，范围为 0～1。
- 多光谱：输入为 `[6,128,128]`，依次为 Green、Red、RedEdge、NIR、NDVI、NDRE。前四个通道范围为 0～1，两个指数保留约 -1～1 的原始范围。
- `NDVI = (NIR - Red) / (NIR + Red + eps)`，`NDRE = (NIR - RedEdge) / (NIR + RedEdge + eps)`，`eps=1e-6` 防止除零。

多光谱输入比 RGB 包含更多植被光谱信息。Green、Red 与同种子的 RGB 样本共享，RedEdge、NIR 使用类别相关的模拟反射率并添加光照变化和噪声，两个指数由生成的波段计算。相同 seed/index 下两种输入的 mask 一致，仍为 0=background、1=crop、2=weed。U-Net 输入通道分别为 3 和 6，输出始终为三个类别。

保持 epochs=10、batch-size=16、lr=0.001、数据种子和评估方式一致，先以 weighted_ce 比较 weed IoU 和 mean IoU：

```bash
python train_synthetic_unet.py --epochs 10 --loss weighted_ce --input-type rgb
python train_synthetic_unet.py --epochs 10 --loss weighted_ce --input-type multispectral
python visualize_synthetic_prediction.py --loss weighted_ce --input-type multispectral
python plot_synthetic_history.py --loss weighted_ce --input-type multispectral
```

| 输入 | 模型 | 训练历史 CSV |
| --- | --- | --- |
| rgb | `models/synthetic_unet_rgb_weighted_ce.pth` | `outputs/synthetic_history_rgb_weighted_ce.csv` |
| multispectral | `models/synthetic_unet_multispectral_weighted_ce.pth` | `outputs/synthetic_history_multispectral_weighted_ce.csv` |

所有实验按 `synthetic_unet_<input_type>_<loss>.pth` 和 `synthetic_history_<input_type>_<loss>.csv` 保存。预测图为 `outputs/synthetic_prediction_<input_type>_<loss>.png`，曲线图为 `outputs/synthetic_history_<input_type>_<loss>.png`。同一组合重新训练会覆盖其模型和 CSV。此前仅包含 loss 的旧文件不会自动迁移或加载；缺少对应模型或 CSV 时，脚本会提示包含输入类型的训练命令。

多光谱预测可视化将 Red/Green/NIR 映射到显示用的 R/G/B，形成近似 RGB 合成图；由于 NIR 替代蓝光，它不是真彩色照片。仍显示 true mask、predicted mask 和 error map。模拟反射率不是实测光谱，实验结果只能用于比较当前模拟条件，不能证明真实田间的多光谱优势。上述 synthetic 实验不联网、不下载或训练真实 WeedMap 数据，结果记录在 `synthetic_experiment_notes.md`。

## Activation function ablation plan

当前模型默认使用 ReLU；为保持已有结果可复现，MobileNetV2 inverted residual block 中的默认激活保留原有 ReLU6。后续将比较 ReLU、LeakyReLU 和 GELU 在 MobileNetV2ShallowUNet + boundary weighted CE r5_w4 上的影响。该实验只替换 encoder/decoder 和 inverted residual block 中间层激活函数，不改变最后 segmentation head，因为 CrossEntropyLoss 需要 raw logits。

- Model: MobileNetV2ShallowUNet
- Input: multispectral
- Loss: boundary_weighted_ce
- radius = 5
- boundary weight = 4
- Epochs = 20
- First test seed = 0
- Activations: relu, leaky_relu, gelu

后续可视化应优先选择 dense vegetation validation samples，例如 VCR > 0.8 或 validation split 中 VCR 最高的样本，以更符合 WeedMap common split 的主体分布，而不再只看零散样本。当前仅准备代码和计划，尚未运行激活函数消融训练。
