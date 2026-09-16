# Synthetic Crop/Weed/Background 语义分割实验报告

## 1. 项目背景

本项目面向无人机农业遥感场景中的 crop / weed / background 语义分割任务，对应导师提出的作物、杂草、土壤/背景区分问题。当前阶段先使用 synthetic 模拟数据跑通从数据生成、模型训练到指标评估和结果可视化的完整流程，后续再迁移到真实 WeedMap 数据。

## 2. 任务定义

- 输入：模拟农田图像；
- 输出：像素级 segmentation mask；
- 类别：
  - `0 = background`
  - `1 = crop`
  - `2 = weed`

本项目属于语义分割任务，需要为图像中的每个像素预测类别，而不是对整张图像进行分类。

## 3. Synthetic 数据设计

Synthetic 数据使用不同的视觉模式模拟农田中的三类区域：background 模拟土壤，crop 模拟规则排列的作物区域，weed 模拟随机分布的小目标杂草区域。同时加入噪声，以模拟真实航拍影像中的光照、传感器和环境波动。

在基础 RGB 数据之外，后续还扩展了 multispectral 输入，用于初步比较 RGB 信息与模拟多光谱信息对分割结果的影响。

## 4. 模型方法

实验使用小型 U-Net 完成像素级预测：

- encoder 逐步提取图像特征；
- decoder 恢复特征图的空间分辨率；
- skip connection 将浅层空间细节传递到解码端，帮助保留目标边界和小目标信息；
- loss 包括 `ce`、`weighted_ce`、`dice` 和 `focal`；
- 评估指标包括 pixel accuracy、per-class IoU 和 mean IoU。

其中，pixel accuracy 衡量所有像素中预测正确的比例；per-class IoU 分别衡量 background、crop 和 weed 的区域重叠质量；mean IoU 是各类别 IoU 的平均值。

## 5. Baseline 实验

Baseline 使用普通 `CrossEntropyLoss`，训练 5 epochs，结果如下：

| 指标 | 结果 |
| --- | ---: |
| Pixel Accuracy | 93.16% |
| Mean IoU | 59.31% |
| Background IoU | 98.09% |
| Crop IoU | 79.59% |
| Weed IoU | 0.26% |

Baseline 的 pixel accuracy 看起来不低，但 weed IoU 几乎为 0，说明模型主要依靠占比更高的 background 和 crop 获得较高的整体准确率，并没有真正学会识别少数类 weed。

## 6. Class Weights 实验

本实验使用 weighted `CrossEntropyLoss`，class weights 设置为 background=1.0、crop=2.0、weed=6.0，训练 5 epochs，结果如下：

| 指标 | 结果 |
| --- | ---: |
| Pixel Accuracy | 97.05% |
| Mean IoU | 88.57% |
| Background IoU | 96.55% |
| Crop IoU | 93.60% |
| Weed IoU | 75.54% |

加入 class weights 后，weed IoU 从 0.26% 显著提升到 75.54%，同时 mean IoU 也明显提高。这说明提高少数类 weed 的损失权重能够缓解类别不平衡问题，使模型更加重视杂草区域。

## 7. 10 Epochs 稳定性实验

将 weighted_ce 的训练轮数增加到 10 epochs，结果如下：

| 指标 | 结果 |
| --- | ---: |
| Average Train Loss | 0.0501 |
| Pixel Accuracy | 98.85% |
| Mean IoU | 94.99% |
| Background IoU | 98.79% |
| Crop IoU | 96.98% |
| Weed IoU | 89.19% |

训练过程中，weed IoU 在前期波动较大，但从 Epoch 6 到 Epoch 10 逐渐稳定并持续提升。与 5 epochs 的结果相比，增加训练轮数进一步改善了少数类的学习效果和整体分割性能。

## 8. Loss Function 对比实验

在相同实验设置下，对四种 loss function 的结果进行比较：

| Loss Function | Pixel Accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| ce | 98.04% | 89.71% | 98.66% | 94.34% | 76.14% |
| weighted_ce | **98.85%** | **94.99%** | **98.79%** | **96.98%** | **89.19%** |
| dice | 97.23% | 85.51% | 98.64% | 91.62% | 66.27% |
| focal | 98.00% | 91.16% | 98.18% | 94.00% | 81.30% |

`weighted_ce` 在当前实验中效果最好，取得最高的 mean IoU 和 weed IoU。`focal` 对 weed 识别也有帮助，表现优于普通 `ce`；`dice` 在本实验设置下不如 `weighted_ce`。结果说明，不同 loss function 对少数类 weed 的学习效果有明显影响。

## 9. RGB vs Multispectral 输入对比

两种输入均使用 weighted_ce 并训练 10 epochs，结果如下：

| 输入 | Average Train Loss | Pixel Accuracy | Mean IoU | Background IoU | Crop IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGB | — | 98.85% | **94.99%** | 98.79% | 96.98% | **89.19%** |
| Multispectral | 0.0360 | **98.97%** | 94.97% | **98.95%** | **97.90%** | 88.07% |

在当前 synthetic 设置下，多光谱输入的 pixel accuracy、background IoU 和 crop IoU 略高，但 mean IoU 和 weed IoU 没有明显超过 RGB。两者整体表现非常接近，因此当前结果不能说明多光谱具有明确优势，也不能说明多光谱在真实无人机数据中无效。模拟波段与真实传感器数据之间仍有明显差异，多光谱价值需要在真实数据上进一步验证。

## 10. 主要结论

- Pixel accuracy 不能充分评价语义分割效果，尤其容易受到大面积 background 类别的影响；
- 对本任务而言，weed IoU 和 mean IoU 是更重要的指标；
- 类别不平衡会导致模型忽视少数类 weed；
- class weights 是缓解类别不平衡、提升 weed 分割效果的有效方法；
- 增加训练轮数可以让少数类学习更加稳定；
- 不同 loss function 对 weed IoU 有明显影响，当前实验中 weighted_ce 表现最好；
- synthetic 数据可以帮助跑通完整流程，但真实 WeedMap 数据中的背景、光照、目标形态和光谱变化会更加复杂。

## 11. 与导师课题的关系

本项目对应导师提出的无人机实验田任务：利用无人机影像以及光谱和空间信息，区分作物、杂草、土壤/背景。当前项目已经完成了 synthetic 数据上的语义分割基础流程，包括数据构造、U-Net 训练、多种 loss 比较、分类别指标评估，以及 RGB 与多光谱输入对比。

下一步可以将这一流程迁移到真实 WeedMap 数据，并进一步应用于水稻或柑橘实验田数据，以验证模型在真实农业遥感场景中的适用性。

## 12. 后续计划

- 下载并整理真实 WeedMap 数据；
- 编写真实数据 Dataset；
- 可视化真实 image/mask；
- 建立 NDVI baseline；
- 训练真实 WeedMap U-Net；
- 比较 RGB、多光谱、植被指数输入；
- 后续迁移到水稻/柑橘实验田。
