# WeedMap multispectral weed recognition stage summary

## 1. Research objective

本项目基于 WeedMap 无人机低空遥感数据，探索多光谱图像下的作物与杂草识别方法。研究包括面向 dense vegetation 场景的语义分割主线，以及面向 sparse vegetation 场景的 YOLO 检测支线。

## 2. Dataset and preprocessing

当前实验使用 WeedMap common split，共 454 个样本，其中训练集 363 个、验证集 91 个。输入图像尺寸为 360 × 480；多光谱通道为 G、R、RE、NIR、NDVI。语义标签包含 background、crop、weed 三类，无效 mask 区域使用 `ignore_index=255`，不参与损失和指标计算。

按植被覆盖率（VCR）划分场景的结果如下：

| 指标 | 结果 |
| --- | ---: |
| Total samples | 454 |
| Sparse | 13 |
| Transition | 9 |
| Dense | 432 |
| Mean VCR | 0.7928 |
| Median VCR | 0.9116 |
| Dense ratio | 432 / 454 = 95.15% |

当前 common split 以 dense vegetation samples 为主，因此语义分割是当前研究主线。

## 3. Semantic segmentation main line

已完成 RGB 与 multispectral 输入对比，并在 SmallUNet 和 MobileNetV2ShallowUNet 上开展 weighted CE、focal loss、dice+CE、boundary weighted CE 等实验。

当前最强语义分割设置为 **MobileNetV2ShallowUNet + boundary weighted CE r5_w4**。在 WeedMap common split 上，其三 seed 验证结果（均值 ± 标准差）为：

| 指标 | 结果 |
| --- | ---: |
| Pixel accuracy | 96.98% ± 0.24% |
| Mean IoU | 76.71% ± 0.55% |
| Background IoU | 97.26% ± 0.16% |
| Crop IoU | 73.04% ± 1.86% |
| Weed IoU | 59.82% ± 0.64% |

该模型是当前 WeedMap common split 上最强的语义分割设置。实验表明，浅层 MobileNetV2-style encoder 与 boundary-aware loss 可以叠加提升 weed 类表现。这里的 MobileNetV2ShallowUNet 是当前项目中的浅层 MobileNetV2-style encoder 改造，并非论文完整 MobileNetV2-U-Net 复现。

## 4. YOLO detection auxiliary line

YOLO 支线的标签由 **semantic mask → connected components → bbox → YOLO labels** 自动生成。这些 bbox 是从 semantic masks 得到的 pseudo boxes，并非人工 instance annotations；一个 connected component 不一定对应一株植物。

已完成 bbox statistics、r020/r010 `max-box-area-ratio` 对比、weed-only 对照、prediction visualization、confidence threshold 的单样本观察，以及 lightweight baseline summary。

当前最佳 YOLO 基线为 **YOLOv8n crop+weed r010**。其 5 epochs 实验结果如下：

| 指标 | 结果 |
| --- | ---: |
| Precision | 0.50539 |
| Recall | 0.58950 |
| mAP50 | 0.53062 |
| mAP50-95 | 0.29557 |
| Params | 3,011,238 |
| GFLOPs | 4.608432 |
| Model size | 6.21425 MB |

在当前相同数据构建方式与 5 epochs 设置下，crop+weed r010 的 weed 类检测指标优于 weed-only r010。这提示保留 crop 类可能有助于模型学习作物与杂草的区分及作物行结构；这一解释仍需进一步验证。YOLO 检测支线目前仍作为 sparse vegetation scenarios 的候选路线。

## 5. Segmentation vs detection positioning

- **Dense vegetation：**植株或植被区域容易粘连，边界复杂，使用区域级 mask 表达作物和杂草更自然，因此更适合 semantic segmentation。
- **Sparse vegetation：**单株或单簇目标相对独立，更容易转为 bbox；YOLO detection 可用于目标定位和点喷。
- **当前路线：**WeedMap common split 中 dense 样本占 432 / 454（95.15%），所以当前主线是 segmentation；YOLO 是面向 sparse 场景的辅助路线，不替代分割主线。

## 6. Future work

### A. Segmentation robustness

- 增加 random seeds，检查结果稳定性。
- 扩充 validation visualizations，并在更多样本上分析边界错误。
- 测量 inference speed 和 model size。

### B. YOLO post-processing

- 系统评估 confidence threshold 与 NMS IoU threshold 对预测结果的影响。
- 检查 PR/F1 curves，并测量 inference speed。

### C. MobileNetV3-YOLOv8n planned experiment

- 尝试 MobileNetV3-style lightweight backbone，保留 YOLOv8 neck、detect head 与 anchor-free framework。
- 在统一条件下比较 mAP、Params、GFLOPs、model size 和 inference speed。
- 该实验目前仅为计划，尚未实现。
