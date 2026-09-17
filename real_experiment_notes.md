# 真实 WeedMap 实验记录

## 真实数据训练目标

本实验使用真实 WeedMap tiles 训练现有的小型 U-Net，完成 background、crop 和 weed 三类像素级语义分割。训练过程记录验证集 pixel accuracy、mean IoU 及三个类别各自的 IoU，为后续输入形式和损失函数对比提供统一基线。

## 为什么使用 `ignore_index=255`

真实数据的有效区域 mask 会把无数据、边缘或不应参与监督的像素标记为 255。模型只有 0=background、1=crop、2=weed 三个输出类别，255 不是第 4 类。训练和评估都必须忽略这些像素，否则会错误惩罚模型、污染混淆矩阵，并使指标不能代表有效标注区域的表现。

## 为什么先使用 multispectral

多光谱输入包含 G、R、RedEdge、NIR 和 NDVI 五个通道。RedEdge、NIR 和植被指数能补充 RGB 中不明显的植被光谱信息，因此先用 multispectral 建立更贴近 WeedMap 数据特点的真实数据基线。

## 为什么使用 weighted_ce

真实农田图像通常存在明显类别不平衡，background 像素较多，weed 像素较少。默认使用 background=1.0、crop=4.0、weed=8.0 的 weighted cross entropy，提高作物和杂草误分类的代价，减少训练被背景类别主导的风险。最终效果仍需结合各类别 IoU 判断。

## 第一次真实训练结果

- `input_type=multispectral`
- `loss=weighted_ce`
- `epochs=3`
- train loss：`0.4497 -> 0.2724 -> 0.2532`
- Epoch 3 val pixel accuracy：`95.61%`
- val mean IoU：`56.37%`
- background IoU：`97.15%`
- crop IoU：`56.28%`
- weed IoU：`15.68%`

整体像素准确率较高，但 weed IoU 明显低于 background 和 crop。下一步将通过真实预测可视化检查 weed 的漏检、误检、边界混淆及小目标表现，分析 weed IoU 较低的原因。

## 真实 WeedMap 10 epochs 稳定性实验

### 实验设置

- `input_type=multispectral`
- `loss=weighted_ce`
- `epochs=10`
- `batch_size=2`
- class weights：`background=1.0`、`crop=4.0`、`weed=8.0`
- `ignore_index=255`
- train samples：`707`
- val samples：`177`

### 训练结果

| Epoch | Train loss | Val pixel accuracy | Val mean IoU | Background IoU | Crop IoU | Weed IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/10 | 0.4497 | 95.38% | 52.28% | 96.99% | 47.65% | 12.18% |
| 2/10 | 0.2724 | 95.25% | 52.53% | 96.84% | 49.52% | 11.24% |
| 3/10 | 0.2532 | 95.61% | 56.37% | 97.15% | 56.28% | 15.68% |
| 4/10 | 0.2455 | 95.12% | 56.41% | 96.76% | 50.87% | 21.59% |
| 5/10 | 0.2364 | 95.52% | 54.94% | 96.83% | 55.68% | 12.31% |
| 6/10 | 0.2365 | 95.50% | 54.14% | 96.81% | 53.91% | 11.70% |
| 7/10 | 0.2257 | 95.07% | 57.62% | 96.18% | 57.81% | 18.86% |
| 8/10 | 0.2099 | 96.40% | 62.65% | 97.17% | 62.13% | 28.66% |
| 9/10 | 0.2019 | 96.29% | 63.80% | 97.14% | 60.38% | 33.90% |
| 10/10 | 0.1781 | 97.22% | 69.21% | 97.69% | 64.17% | 45.76% |

### 3 epochs 与 10 epochs 对比

| 训练轮数 | Val mean IoU | Weed IoU |
| ---: | ---: | ---: |
| 3 epochs | 56.37% | 15.68% |
| 10 epochs | 69.21% | 45.76% |

weed IoU 从 `15.68%` 提升到 `45.76%`，说明增加训练轮数对真实 WeedMap 的 weed 类非常有效。

### 10 epochs 预测可视化结果

- sample index：`0`
- pixel accuracy：`95.49%`
- background IoU：`96.24%`
- crop IoU：`56.40%`
- weed IoU：`37.34%`
- mean IoU：`63.33%`

### 实验现象总结

- background IoU 一直较高，因为背景像素占比大且更容易识别。
- crop IoU 稳定提升。
- weed IoU 前期波动较大，但 Epoch 8 到 Epoch 10 明显上升。
- 真实 WeedMap 中 weed 是最难类别，训练轮数不足时容易漏检。
- 不能只看 pixel accuracy，必须重点看 weed IoU 和 mean IoU。

## 严格公平 RGB vs Multispectral 对比实验

### 实验设置

- `sample_list_csv=splits/real_weedmap_common_samples.csv`
- RGB 和 Multispectral 使用完全相同的 454 个样本
- train samples：`363`
- val samples：`91`
- `input_type` 分别为 `rgb` 和 `multispectral`
- `loss=weighted_ce`
- `epochs=10`
- `batch_size=2`
- class weights：`background=1.0`、`crop=4.0`、`weed=8.0`
- `ignore_index=255`

RGB 和 Multispectral 使用同一个 sample list、相同样本数和相同训练/验证划分，因此这次对比比之前样本集合不同的实验更公平，可以更可靠地比较输入通道带来的差异。

### 实验结果

| Input | Train Loss | Val Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 0.2956 | 94.28% | 65.27% | 95.49% | 65.75% | 34.56% |
| Multispectral | 0.2166 | 94.38% | 67.76% | 94.94% | 61.62% | 46.73% |

### 结果解释

1. Multispectral 的 mean IoU 为 67.76%，高于 RGB 的 65.27%，说明整体分割效果略好。
2. Multispectral 的 weed IoU 为 46.73%，比 RGB 的 34.56% 高 12.17 个百分点，说明多光谱通道对杂草识别更有帮助。
3. RGB 的 crop IoU 为 65.75%，比 Multispectral 的 61.62% 高 4.13 个百分点，说明 RGB 对作物行状结构也有一定优势。
4. 对导师课题来说，weed 是关键难点，因此多光谱方向更值得继续深入。

### 最佳轮次与训练波动

Multispectral 在 Epoch 8 达到更好结果：mean IoU 为 70.38%，weed IoU 为 50.43%；Epoch 10 的 mean IoU 为 67.76%，weed IoU 为 46.73%。真实 WeedMap 训练后期存在波动，后续实验应优先报告验证集 mean IoU 最优的 best checkpoint 指标，而不仅仅是最后一轮指标。训练脚本现已同时保存 best checkpoint 和最后一轮模型。

## Best checkpoint 验证结果

在严格公平 multispectral 实验中，训练脚本已经支持根据 val mean IoU 保存 best checkpoint。

### 整体验证集结果

- best epoch = 8
- best val mean IoU = 70.38%
- best background IoU = 96.09%
- best crop IoU = 64.61%
- best weed IoU = 50.43%
- best model path = `models/real_weedmap_common_multispectral_weighted_ce_10epochs_best.pth`

与最后一轮 Epoch 10 对比：

- Epoch 10 val mean IoU = 67.76%
- Epoch 10 weed IoU = 46.73%
- best Epoch 8 val mean IoU = 70.38%
- best Epoch 8 weed IoU = 50.43%

这组结果说明：

1. 真实 WeedMap 训练后期存在波动。
2. 最后一轮模型不一定是最优模型。
3. 保存 best checkpoint 可以避免因为后期波动导致最终模型性能下降。
4. 后续实验应优先报告 best checkpoint 的结果。

### 单张 sample index=0 预测可视化结果

| Model | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|
| Epoch 10 final | 92.63% | 57.12% | 93.41% | 48.42% | 29.53% |
| Best checkpoint | 94.19% | 60.77% | 94.96% | 53.49% | 33.85% |

best checkpoint 在验证集整体指标和单张预测可视化指标上都优于最后一轮模型，因此后续真实 WeedMap 实验应保存并使用 best checkpoint。

## Weed class weight 调参实验

### 实验设置

- `sample_list_csv=splits/real_weedmap_common_samples.csv`
- `input_type=multispectral`
- RGB 和 multispectral 公平对比后，继续在 multispectral 上调参
- `loss=weighted_ce`
- `epochs=10`
- `batch_size=2`
- train samples：`363`
- val samples：`91`
- background weight：`1.0`
- crop weight：`4.0`
- weed weight：`8`、`12`、`16`
- `ignore_index=255`
- 使用 best checkpoint，根据 val mean IoU 保存最佳模型

### 实验结果

| Weed Weight | Best Epoch | Best Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 70.38% | 96.09% | 64.61% | 50.43% |
| 12 | 8 | 69.64% | 95.25% | 64.63% | 49.04% |
| 16 | 8 | 67.98% | 94.72% | 62.88% | 46.36% |

### 结果解释

1. `weed weight=8` 当前效果最好，best mean IoU 和 weed IoU 都最高。
2. `weed weight=12` 与 8 接近，但两项指标略低。
3. `weed weight=16` 时，mean IoU 和 weed IoU 均下降。
4. weed 权重不是越大越好；过大的 weed 权重可能破坏 background、crop、weed 三类之间的平衡。
5. 当前后续实验可以继续以 `background=1.0`、`crop=4.0`、`weed=8.0` 作为默认 weighted CE 设置。
6. 后续若继续调参，可以尝试更细粒度的 `weed weight=10`，或者改用 Focal Loss / Dice + CE。

## 20 epochs 训练实验

### 实验设置

- `sample_list_csv=splits/real_weedmap_common_samples.csv`
- `input_type=multispectral`
- `loss=weighted_ce`
- class weights：`background=1.0`、`crop=4.0`、`weed=8.0`
- `epochs=20`
- `batch_size=2`
- train samples：`363`
- val samples：`91`
- `ignore_index=255`
- 使用 best checkpoint，根据 val mean IoU 保存最佳模型

### 验证集整体结果

| Setting | Best Epoch | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|---:|
| 10 epochs best | 8 | 95.47% | 70.38% | 96.09% | 64.61% | 50.43% |
| 20 epochs best | 15 | 95.97% | 73.88% | 96.20% | 70.89% | 54.54% |

20 epochs best checkpoint 明显优于 10 epochs best checkpoint：mean IoU 从 70.38% 提升到 73.88%，weed IoU 从 50.43% 提升到 54.54%，crop IoU 从 64.61% 提升到 70.89%。这说明在真实 WeedMap 上继续训练到 20 epochs 是有效的。

但 Epoch 15 之后指标仍有波动。例如，Epoch 20 的 mean IoU 降到 69.52%，weed IoU 降到 46.82%，因此 best checkpoint 机制仍然必要。

### 单张 sample index=0 可视化结果

| Model | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|
| 10 epochs best | 94.19% | 60.77% | 94.96% | 53.49% | 33.85% |
| 20 epochs best | 94.82% | 63.90% | 95.32% | 59.39% | 36.99% |

单张预测可视化上，20 epochs best 同样优于 10 epochs best。错误仍主要集中在 crop/weed 边界、小目标 weed 以及植物混杂区域。后续可以继续尝试更长训练，例如 30 epochs，但必须使用 best checkpoint。20 epochs 的多 seed 重复结果见下节。

## 20 epochs 多 seed 重复实验

### 实验设置

- `sample_list_csv=splits/real_weedmap_common_samples.csv`
- `input_type=multispectral`
- `loss=weighted_ce`
- class weights：`background=1.0`、`crop=4.0`、`weed=8.0`
- `epochs=20`
- `batch_size=2`
- train samples：`363`
- val samples：`91`
- `ignore_index=255`
- seeds：`0`、`1`、`2`
- 使用 best checkpoint，根据 val mean IoU 保存最佳模型

### 各 seed 的 best checkpoint 验证集结果

| Seed | Best Epoch | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 95.97% | 73.88% | 96.20% | 70.89% | 54.54% |
| 1 | 15 | 96.51% | 74.99% | 96.96% | 69.59% | 58.44% |
| 2 | 15 | 96.65% | 75.47% | 96.94% | 71.80% | 57.68% |

### 多 seed 均值与样本标准差

| Metric | Mean ± Std |
|---|---:|
| Pixel Accuracy | 96.38% ± 0.36% |
| Mean IoU | 74.78% ± 0.82% |
| Background IoU | 96.70% ± 0.43% |
| Crop IoU | 70.76% ± 1.11% |
| Weed IoU | 56.89% ± 2.06% |

三个 seed 的 best epoch 均为 Epoch 15，说明当前设置在 20 epochs 内的最佳轮次比较稳定。三个 seed 的 mean IoU 位于 73.88%～75.47%，波动较小；weed IoU 均超过 54%，明显高于 10 epochs best 的 50.43%。多 seed 平均 weed IoU 为 56.89% ± 2.06%，支持 20 epochs 的提升不是偶然的单次结果。当前可将 `multispectral + weighted CE + class weights 1/4/8 + 20 epochs + best checkpoint` 作为后续 baseline。

## Loss 多 seed 对比实验

### 实验设置

- `sample_list_csv=splits/real_weedmap_common_samples.csv`
- `input_type=multispectral`
- `epochs=20`，`batch_size=2`
- train samples：`363`；val samples：`91`
- `ignore_index=255`
- seeds：`0`、`1`、`2`
- 使用 best checkpoint，根据 val mean IoU 保存最佳模型
- 对比 Weighted CE（`background=1.0`、`crop=4.0`、`weed=8.0`）、Focal Loss、Dice + CE。Weighted CE 各 seed 明细见上一节。

### Focal Loss：各 seed 的 best checkpoint 验证集结果

| Seed | Best Epoch | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 96.59% | 74.72% | 97.09% | 69.55% | 57.52% |
| 1 | 19 | 96.51% | 72.56% | 97.17% | 66.14% | 54.36% |
| 2 | 19 | 96.99% | 74.73% | 97.30% | 72.12% | 54.77% |

### Dice + CE：各 seed 的 best checkpoint 验证集结果

| Seed | Best Epoch | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 96.81% | 76.35% | 97.17% | 72.07% | 59.82% |
| 1 | 19 | 96.92% | 76.44% | 97.26% | 71.39% | 60.67% |
| 2 | 16 | 96.46% | 70.56% | 97.20% | 65.14% | 49.35% |

### 三种 loss 的均值与样本标准差

| Loss | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|
| Weighted CE | 96.38% ± 0.36% | 74.78% ± 0.82% | 96.70% ± 0.43% | 70.76% ± 1.11% | 56.89% ± 2.06% |
| Focal Loss | 96.69% ± 0.26% | 74.00% ± 1.25% | 97.19% ± 0.11% | 69.27% ± 3.00% | 55.55% ± 1.72% |
| Dice + CE | 96.73% ± 0.24% | 74.45% ± 3.37% | 97.21% ± 0.05% | 69.54% ± 3.82% | 56.61% ± 6.31% |

Weighted CE 的平均 mean IoU 最高，为 74.78% ± 0.82%；平均 weed IoU 也最高，为 56.89% ± 2.06%。Focal Loss 的 background IoU 较高，weed IoU 的跨 seed 波动也较小，但平均 weed IoU 略低于 Weighted CE。Dice + CE 在 seed=0 和 seed=1 上表现很强，seed=2 的 mean IoU 和 weed IoU 明显下降，因此样本标准差较大。当前阶段最稳的主 baseline 仍是 `multispectral + weighted CE + class weights 1/4/8 + 20 epochs + best checkpoint`。Dice + CE 有后续探索价值，需要进一步调参或增加 seed 验证稳定性。

## Boundary error analysis 与 boundary weighted CE

### 问题观察与结构检查

预测可视化显示，错误主要集中在 crop/weed 轮廓附近，Error map 中有许多白色边界圈。检查当前 `unet.py`：模型已有 encoder-decoder skip connection，使用 `torch.cat(..., dim=1)` 进行 concat，而非 add。因此问题并非缺少 skip connection，而是当前普通 U-Net 的边界精细化能力仍然不足。

### Boundary error analysis：Weighted CE 20 epochs best，sample index=0

| 统计项 | 结果 |
|---|---:|
| Total valid pixels | 112158 |
| Total error pixels | 5812 |
| Overall error rate | 5.18% |
| Errors within 1px boundary | 4504（占错误 77.49%） |
| Errors within 3px boundary | 5372（占错误 92.43%） |
| Errors within 5px boundary | 5532（占错误 95.18%） |
| Errors outside 5px boundary | 280（占错误 4.82%） |
| Valid pixels within 5px boundary | 38199（占有效像素 34.06%） |
| Error rate within 5px boundary | 14.48% |
| Error rate outside 5px boundary | 0.38% |

5px 边界区域仅占有效像素的 34.06%，却包含 95.18% 的错误像素；该区域错误率为 14.48%，远高于非边界区域的 0.38%。当前模型的主要瓶颈集中在边界区域，而非大面积内部区域。

### Boundary weighted CE 实验设置与三 seed 结果（r5_w3）

- `loss=boundary_weighted_ce`，`boundary radius=5`，`boundary weight=3.0`；
- `input_type=multispectral`，`epochs=20`，`batch_size=2`；
- `sample_list_csv=splits/real_weedmap_common_samples.csv`；
- 使用 best checkpoint。

| Seed | Best Epoch | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 96.71% | 75.95% | 97.10% | 71.66% | 59.09% |
| 1 | 16 | 96.73% | 73.84% | 97.15% | 68.42% | 55.97% |
| 2 | 18 | 97.04% | 75.43% | 97.33% | 71.06% | 57.92% |

三 seed 均值 ± 样本标准差：

| Metric | Mean ± Std |
|---|---:|
| Pixel Accuracy | 96.83% ± 0.19% |
| Mean IoU | 75.08% ± 1.10% |
| Background IoU | 97.19% ± 0.12% |
| Crop IoU | 70.38% ± 1.73% |
| Weed IoU | 57.66% ± 1.58% |

与原 Weighted CE 的三 seed 结果对比（均值 ± 样本标准差）：

| Loss | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|
| Weighted CE | 96.38% ± 0.36% | 74.78% ± 0.82% | 96.70% ± 0.43% | 70.76% ± 1.11% | 56.89% ± 2.06% |
| Boundary weighted CE（r5_w3） | 96.83% ± 0.19% | 75.08% ± 1.10% | 97.19% ± 0.12% | 70.38% ± 1.73% | 57.66% ± 1.58% |

### Boundary weight=4.0 三 seed 调参结果（r5_w4）

保持 `input_type=multispectral`、`loss=boundary_weighted_ce`、`boundary_radius=5`、`epochs=20`，按验证集 mean IoU 选择 best checkpoint；将 `boundary_weight` 调至 4.0。

| Seed | Best Epoch | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 96.67% | 75.74% | 97.04% | 70.89% | 59.28% |
| 1 | 16 | 96.73% | 73.99% | 97.14% | 67.37% | 57.45% |
| 2 | 16 | 96.97% | 75.78% | 97.26% | 71.97% | 58.13% |

三 seed 均值 ± 样本标准差：

| Metric | Mean ± Std |
|---|---:|
| Pixel Accuracy | 96.79% ± 0.16% |
| Mean IoU | 75.17% ± 1.02% |
| Background IoU | 97.15% ± 0.11% |
| Crop IoU | 70.07% ± 2.41% |
| Weed IoU | 58.29% ± 0.92% |

与 r5_w3 对比（三 seed 均值 ± 样本标准差）：

| 实验 | Mean IoU | Weed IoU |
|---|---:|---:|
| r5_w3（boundary weight=3.0） | 75.08% ± 1.10% | 57.66% ± 1.58% |
| r5_w4（boundary weight=4.0） | 75.17% ± 1.02% | 58.29% ± 0.92% |

r5_w4 的 mean IoU 平均提高 0.09 个百分点，weed IoU 平均提高 0.63 个百分点，且 weed IoU 的跨 seed 样本标准差从 1.58 降至 0.92 个百分点。提升幅度较小，主要体现在 weed IoU 更高、更稳定。当前最优候选更新为 `multispectral + boundary_weighted_ce + boundary_radius=5 + boundary_weight=4.0 + 20 epochs + best checkpoint`。

### Sample index=0 的预测与边界错误对比（r5_w3）

| Model | Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|
| Weighted CE 20 epochs best | 94.82% | 63.90% | 95.32% | 59.39% | 36.99% |
| Boundary weighted CE | 96.66% | 68.86% | 97.34% | 66.02% | 43.22% |

Boundary weighted CE 后的 sample index=0 边界错误统计：

| 统计项 | 结果 |
|---|---:|
| Total valid pixels | 112158 |
| Total error pixels | 3746 |
| Overall error rate | 3.34% |
| Errors within 1px boundary | 3155（占错误 84.22%） |
| Errors within 3px boundary | 3580（占错误 95.57%） |
| Errors within 5px boundary | 3630（占错误 96.90%） |
| Errors outside 5px boundary | 116（占错误 3.10%） |
| Valid pixels within 5px boundary | 38199（占有效像素 34.06%） |
| Error rate within 5px boundary | 9.50% |
| Error rate outside 5px boundary | 0.16% |

Sample index=0 的 total error pixels 从 5812 降至 3746，overall error rate 从 5.18% 降至 3.34%；5px boundary error rate 从 14.48% 降至 9.50%，outside 5px error rate 从 0.38% 降至 0.16%。绝对错误数表明 boundary-aware loss 减少了边界附近错误；剩余错误的边界占比仍高，说明边界仍是后续优化重点。

阶段性结论：boundary weighted CE 相比原 Weighted CE 有小幅提升；在 boundary weight 调参中，r5_w4 相比 r5_w3 的 weed IoU 更高、更稳定，可作为当前候选主结果。继续保留 Weighted CE 作为稳定 baseline。

## 阶段性结论

目前可将 r5_w4（`multispectral + boundary_weighted_ce + boundary_radius=5 + boundary_weight=4.0 + 20 epochs + best checkpoint`）作为候选主结果。三 seed 的 Pixel Accuracy = 96.79% ± 0.16%，Mean IoU = 75.17% ± 1.02%，Weed IoU = 58.29% ± 0.92%。相比 r5_w3 提升幅度较小；Weighted CE 仍是稳定 baseline。

## 下一步计划

- 可继续测试 `boundary_weighted_ce` 的其他 boundary weight，例如 2.0。
- 生成 boundary weighted CE 的预测对比图。
- 后续可用 `python plot_real_loss_comparison.py` 生成真实 WeedMap loss 对比曲线图。
- 可尝试调节 Dice + CE 的权重系数，例如 `CE + 0.5 Dice` 或 `CE + 2 Dice`，并增加 seed 验证稳定性。
- 可尝试 30 epochs，但必须继续根据 val mean IoU 使用 best checkpoint；
- 后续更新 Word 报告给导师。

## 语义分割与目标检测路线选择分析

参考《基于深度学习的无人机低空遥感水稻杂草实时识别研究》的低空遥感实时识别思路及[相关水稻杂草语义分割研究](https://www.mdpi.com/2072-4292/13/21/4370)，模型路线应随田间生长状态和输出需求调整，语义分割与目标检测没有绝对优劣。这是对下一步实验设计的分析，不是当前 WeedMap 数据上两条路线的实测结论。

- **分蘖期**：作物和杂草密集、交叠且呈片状分布，难以稳定区分单株或单簇；优先考虑语义分割，获取像素级杂草区域并估计施药范围。
- **苗期**：作物间距较大，杂草较稀疏、点状目标明显；可优先验证 YOLO 系列目标检测，用于稀疏目标的实时定位和计数。

当前 U-Net 已完成 background/crop/weed 三类分割；预测错误主要出现在 crop/weed 边界、小目标 weed 和植物混杂区域，boundary weighted CE 后边界错误有所减少。这说明**普通 U-Net 在稀疏、小目标、边界破碎场景下存在局限**，不能简单推断 U-Net 不适合本任务。

下一步保留 **U-Net / boundary-aware U-Net** 像素级分割路线，用于杂草区域和施药区域估计；增加 **YOLO 系列**作为稀疏作物/杂草目标定位、实时识别和计数的对照路线。YOLO 目前只完成一轮 smoke test，尚未进行正式的同数据对比，因此不能据此否定 U-Net 或声称 YOLO 更优。

参考上述水稻杂草识别论文的思路，密集、片状杂草优先考虑语义分割，稀疏、点状杂草可验证目标检测。因此新增 YOLO detect baseline 的数据准备脚本 `prepare_yolo_detection_dataset.py`：使用共享样本列表及 seed=0 的 U-Net 划分，将有效 crop/weed 像素的连通区域转为检测框，供后续对照实验使用。转换时按连通区域面积、框宽高、框面积占比过滤，也可选择跳过接触图像边界的框。当前 YOLO 标签来自 segmentation mask 自动转换，不是人工 bbox 标注，因此后续结果需要谨慎解释。

在训练 YOLO 前，需要先可视化由 mask 转换得到的 bbox 标签，确认目标框是否合理。可运行 `python visualize_yolo_detection_labels.py` 查看默认样本的 crop/weed 框。

重新生成 YOLO bbox 数据集时，需要使用 `--overwrite` 避免新旧标签混合。

## YOLO smoke test 结果

- ultralytics version：`8.4.154`
- model：`yolov8n.pt`
- epochs：`1`
- train images：`363`
- val images：`91`
- val instances：`3989`
- inference speed：MPS 上约 `2.8 ms/image`

| 类别 | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| all | 0.179 | 0.0616 |
| crop | 0.244 | — |
| weed | 0.114 | — |

这次 smoke test 仅用于验证 YOLO 数据格式和训练流程可用，不用于与 U-Net 正式比较。当前 YOLO 标签由 segmentation mask 自动转换，不是人工 bbox 标注；后续检测结果需要谨慎解释。
