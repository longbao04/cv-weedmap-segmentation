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

已完成 bbox statistics、r020/r015/r010 `max-box-area-ratio` 对比、weed-only 对照、prediction visualization、confidence threshold 的完整验证集 PR/F1 sweep，以及 lightweight baseline summary。

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

## 6. VCR regression attention router

**VCR Regression Attention Router** 已完成首轮正式实验。它位于 weed recognition pipeline 前端，从原始 `G/R/RE/NIR` 四个波段预测连续 VCR，再按 `predicted VCR < 0.20 → sparse → YOLO`、`0.20 ≤ predicted VCR ≤ 0.30 → transition → 双模型或人工确认`、`predicted VCR > 0.30 → dense → segmentation` 路由。不输入 NDVI，因为监督标签由 NDVI threshold 生成，直接输入 NDVI 可能只是复现标签规则；NDVI-VCR 规则保留为零训练 baseline。

采用回归而不是三分类，是因为数据极度不平衡：sparse 13、transition 9、dense 432。直接三分类容易全部预测 dense 并获得虚高 accuracy；连续回归能够保留覆盖率信息，并支持后续调整阈值。

`outputs/vcr_regression_samples.csv` 已由现有 VCR summary 生成并完成 454 条样本校验，字段包含四波段路径、VCR、scene type、subset 和 split。Tiny CNN、SE-Tiny CNN、CBAM-Tiny CNN 均完成 30 epochs × 3 seeds 正式实验。当前代码和正式实验实际使用 Huber loss（`delta=0.1`）。

CBAM-Tiny CNN 的平均 MAE 最低，为 **0.0923 ± 0.0133**，说明 attention 可能有助于学习植被覆盖程度；但 sparse recall 只有 **0.1111 ± 0.1925**，尚不能稳定识别 sparse 场景。non-dense 样本数量太少，validation sparse 仅 3 张，每错一张都会产生很大波动；少数样本还可能集中在特定 subset，相邻帧相似带来空间数据泄漏风险。因此当前结果属于探索性实验，不能作为稳定部署路由器。

VCR regression attention router 已经作为 scene density routing 的初步实验完成。CBAM-Tiny CNN 在 VCR 回归误差上表现最好，但 sparse recall 仍然不足，因此当前不能替代基于 NDVI-VCR 的零训练规则，也不能作为最终部署模块。它现阶段更适合作为扩展数据、改进采样策略和研究 attention-based scene routing 的基础实验。项目主线仍是 dense vegetation 场景下的 semantic segmentation；VCR router 的作用是未来连接 segmentation 与 detection 两条分支，而不是替代它们。

## 7. Future work

### A. Segmentation robustness

- 增加 random seeds，检查结果稳定性。
- 扩充 validation visualizations，并在更多样本上分析边界错误。
- 测量 inference speed 和 model size。

### B. YOLO post-processing

- 根据实际误喷与漏喷成本，在当前 F1 最佳起点 `conf≈0.16` 周围做部署校准。
- 当前 CPU 完整验证 inference 约 10.4 ms/image；部署前需在目标硬件上重新测速。

### C. MobileNetV3-YOLOv8n architecture experiment

- 已实现 MobileNetV3-Small backbone + YOLOv8-style PAN-FPN + anchor-free Detect head。
- 已修正 ImageNet 预训练 backbone 的输入归一化，并完成 5 epochs backbone freeze warm-up + 20 epochs 全模型 fine-tuning。
- 统一验证结果为 P=0.4773、R=0.5269、mAP50=0.4810、mAP50-95=0.2106；相较修正前初始实验的 mAP50=0.1284 已显著恢复。
- 2.39M Params、2.55 GFLOPs、5.04 MB，分别比 YOLOv8n 减少约 20.6%、44.7%、18.9%。
- mAP50 和 mAP50-95 仍分别低 0.0496 和 0.0850；当前 CPU inference 为 40.8 ms/image，慢于 baseline 的 10.4 ms/image，因此暂不替换 YOLOv8n r010。

### D. VCR regression scene router

- 增加 sparse / transition 样本。
- 采用 repeated stratified cross-validation。
- 采用 subset-level holdout 测试泛化能力，并重点检查相邻帧造成的空间数据泄漏。
- 优先报告 MAE、RMSE、balanced accuracy、macro F1、MCC、sparse recall 和 confusion matrix；accuracy 仅作为辅助指标。
- 后续可继续比较 Tiny CNN、SE、CBAM，但需要更多 non-dense 样本支撑结论。
