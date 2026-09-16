# Synthetic segmentation 实验记录

本阶段仅使用模拟数据，未下载或训练真实 WeedMap 数据。保持 U-Net 主体结构不变，已记录四种 loss 的对比结果，当前进入 RGB 输入 vs 多光谱输入对比阶段。

## 训练曲线分析

训练曲线用于观察每个 epoch 的平均训练 loss、pixel accuracy、mIoU 和各类 IoU。重点检查 loss 是否下降、mIoU 和 weed IoU 是否持续提升或趋于稳定，并在相同训练设置下比较 baseline 与 weighted 的变化。整体准确率较高时，仍需单独关注 weed IoU。

此前训练记录使用 `outputs/synthetic_history_baseline.csv` 和 `outputs/synthetic_history_weighted.csv`。新增 loss 选择后，普通与加权实验分别使用 `outputs/synthetic_history_ce.csv` 和 `outputs/synthetic_history_weighted_ce.csv`；旧记录不会自动迁移。运行以下命令绘制对应曲线：

```bash
python plot_synthetic_history.py
python plot_synthetic_history.py --use-class-weights
```

根据已观察到的训练曲线，baseline 的 weed IoU 基本接近 0；虽然 pixel accuracy 看起来不低，但模型几乎没有学会分割 weed。加入 class weights 后，weed IoU 有明显提升，说明加权损失让模型开始关注 weed，但其训练曲线波动较大。

weed 是少数类、小目标，而且与 crop 都属于植被，类别区分更困难，因此 weighted 的 weed IoU 比 background IoU、crop IoU 更不稳定。这表明最终 weed IoU 的提升与训练过程中学习不稳定的现象可以同时存在。以下记录已观察到的现象，未提供的曲线趋势标记为“待补充”。

| 观察项目 | Baseline | Weighted |
| --- | --- | --- |
| loss 的变化 | 待补充 | 待补充 |
| pixel accuracy 的变化 | 看起来不低，但几乎没有学会 weed；具体变化趋势待补充 | 待补充 |
| mIoU 的变化 | 待补充 | 待补充 |
| background IoU 的变化 | 待补充 | 相比 weed IoU 更稳定；具体变化趋势待补充 |
| crop IoU 的变化 | 待补充 | 相比 weed IoU 更稳定；具体变化趋势待补充 |
| weed IoU 的变化 | 基本接近 0 | 明显提升，但训练过程中波动较大 |
| 曲线分析结论 | 整体像素准确率掩盖了 weed 类别的识别不足 | class weights 让模型开始关注 weed，但少数类、小目标的学习仍不稳定 |

## Baseline：普通 CrossEntropyLoss

训练命令：

```bash
python train_synthetic_unet.py --epochs 5
```

| 项目 | 结果 |
| --- | --- |
| epochs | 5 |
| pixel accuracy | 93.16% |
| mean IoU | 59.31% |
| background IoU | 98.09% |
| crop IoU | 79.59% |
| weed IoU | 0.26% |

Baseline 的问题是 pixel accuracy 高，但 weed IoU 很低，说明模型几乎没有学会分割小目标 weed。background/crop 像素多、weed 像素少，类别不平衡可能让模型更偏向多数类；因此不能只看整体像素准确率。

## Class weights 实验：weighted CrossEntropyLoss

设置 background=1.0、crop=2.0、weed=6.0。weed 像素少，所以给更高权重，对分错 weed 给予更高错误惩罚，缓解类别不平衡。以下通过各类 IoU 和 mean IoU 比较分割效果。

```bash
python train_synthetic_unet.py --epochs 5 --use-class-weights
python visualize_synthetic_prediction.py --use-class-weights
```

此前模型保存为 `models/synthetic_unet_weighted.pth`；新脚本使用 `models/synthetic_unet_weighted_ce.pth`。以下记录已提供的实验结果；比较时应保持 epochs、batch-size、lr 和模拟数据种子一致。

| 项目 | 结果 |
| --- | --- |
| loss function | Weighted CrossEntropyLoss |
| class weights | background=1.0, crop=2.0, weed=6.0 |
| epochs | 5 |
| average train loss | 0.1719 |
| pixel accuracy | 97.05% |
| mean IoU | 88.57% |
| background IoU | 96.55% |
| crop IoU | 93.60% |
| weed IoU | 75.54% |

### 观察与结论

1. Baseline 的 pixel accuracy 为 93.16%，看起来较高，但 weed IoU 只有 0.26%，说明模型几乎没有学会分割 weed。
2. 加入 class weights 后，weed IoU 从 0.26% 提升到 75.54%，提升非常明显。
3. mean IoU 从 59.31% 提升到 88.57%，说明整体分割质量明显改善。
4. crop IoU 从 79.59% 提升到 93.60%，说明 crop 的分割效果也得到改善。
5. background IoU 从 98.09% 降到 96.55%，结合 crop 和 weed IoU 的提升，说明模型不再过度偏向 background。
6. Baseline 的 weed IoU 接近 0，说明普通 CrossEntropyLoss 容易被 background/crop 这类大面积类别主导。pixel accuracy 高并不代表模型真的识别好了所有类别。
7. class weights 能显著提高 weed IoU，说明它对类别不平衡问题有效，是本次模拟数据实验中处理小目标/少数类识别困难的有效 baseline 改进方法。
8. Weighted 的 weed IoU 在训练过程中波动较大，说明少数类 weed 的学习更不稳定。weed 面积小、属于小目标，且与 crop 都属于植被，因此其 IoU 比 background IoU、crop IoU 更不稳定。
9. 在 crop/weed/background 分割任务中，应该重点关注 per-class IoU，尤其是 weed IoU，并结合 mean IoU 判断整体分割质量，而不是只看 pixel accuracy。
10. 这个现象和真实农业遥感任务很接近，因为真实田间杂草通常面积小、分布零散、和作物光谱相近。本次结果来自模拟数据，尚未验证真实田间数据上的效果。

## 10 epochs + class weights 稳定性实验

在模拟数据上，将 weighted CrossEntropyLoss 的训练轮数扩展到 10 epochs，观察少数类 weed 的学习稳定性。5 epochs weighted 与 10 epochs weighted 的结果对比如下：

| 指标 | 5 epochs weighted | 10 epochs weighted |
| --- | --- | --- |
| average train loss | 0.1719 | 0.0501 |
| pixel accuracy | 97.05% | 98.85% |
| mean IoU | 88.57% | 94.99% |
| background IoU | 96.55% | 98.79% |
| crop IoU | 93.60% | 96.98% |
| weed IoU | 75.54% | 89.19% |

训练过程中各 epoch 的 weed IoU：

| Epoch | weed IoU |
| --- | --- |
| 1 | 0.00% |
| 2 | 41.21% |
| 3 | 0.20% |
| 4 | 73.92% |
| 5 | 75.54% |
| 6 | 82.64% |
| 7 | 84.88% |
| 8 | 84.37% |
| 9 | 86.99% |
| 10 | 89.19% |

### 观察与结论

1. weed IoU 从 5 epochs weighted 的 75.54% 提升到 10 epochs weighted 的 89.19%，提高了 13.65 个百分点。
2. mean IoU 从 88.57% 提升到 94.99%，提高了 6.42 个百分点，整体分割质量进一步改善。
3. pixel accuracy 从 97.05% 提升到 98.85%，提高了 1.80 个百分点。
4. 前 5 个 epoch 的 weed IoU 波动较大，尤其 Epoch 3 从 Epoch 2 的 41.21% 降到 0.20%，说明训练初期对少数类 weed 的学习尚不稳定。
5. Epoch 6 到 Epoch 10 的 weed IoU 均保持在 82% 以上；虽然 Epoch 8 有小幅回落，但后期整体更稳定，并在 Epoch 10 达到 89.19%。
6. 结合此前 baseline 与 weighted 的对比，class weights 对少数类 weed 有明显帮助，但需要更多训练轮数让模型稳定学习。本次 10 epochs 的后期表现支持这一判断，尚不能据此认定更长训练或其他数据上也一定稳定。
7. 在真实 crop/weed/background 农业遥感分割任务中，少数类 weed 需要重点关注。不能只看 pixel accuracy，应结合 mean IoU 和各类 IoU，尤其是 weed IoU，评估模型是否学会识别杂草。本次结果来自模拟数据，真实数据上的效果仍需验证。

## Loss Function 对比实验

实验目的：比较不同 loss function 对 weed IoU 和 mean IoU 的影响，继续改进 synthetic crop/weed/background 分割中的少数类 weed 识别。

比较 CrossEntropyLoss（ce）、Weighted CrossEntropyLoss（weighted_ce）、Dice Loss（dice）和 Focal Loss（focal）。Dice Loss 更关注预测区域和真实区域的重叠；Focal Loss 更关注难分类样本，两者常用于类别不平衡或小目标分割。weighted_ce 的类别权重保持为 background=1.0、crop=2.0、weed=6.0；focal 默认 gamma=2，不额外使用类别权重。

保持模拟数据种子、batch-size、lr 和 epochs 一致，先统一运行 10 epochs。除损失函数外，保持 U-Net 主体结构和评估方式不变。不同 loss 的数值尺度不同，主要用 weed IoU、mean IoU 和各类 IoU 比较效果。

| loss | epochs | average train loss | pixel accuracy | mean IoU | background IoU | crop IoU | weed IoU | 观察 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ce | 10 | 0.0527 | 98.04% | 89.71% | 98.66% | 94.34% | 76.14% | weed IoU 比 5 epochs baseline 的 0.26% 明显提升 | 增加训练轮数本身也有帮助，但 weed IoU 仍低于 focal 和 weighted_ce |
| weighted_ce | 10 | 0.0501 | 98.85% | 94.99% | 98.79% | 96.98% | 89.19% | pixel accuracy、mean IoU 和各类 IoU 均为本组最高；后 5 个 epoch 的 weed IoU 均超过 82% | 整体效果最好，是本组实验中最有效的少数类 weed 改进方法 |
| dice | 10 | 0.0954 | 97.23% | 85.51% | 98.64% | 91.62% | 66.27% | weed IoU 和 mean IoU 均低于 ce、focal 和 weighted_ce | 本实验中 Dice Loss 的 weed 分割效果较弱 |
| focal | 10 | 0.0122 | 98.00% | 91.16% | 98.18% | 94.00% | 81.30% | weed IoU 比 ce 高 5.16 个百分点，mean IoU 也更高 | 对难分类样本和少数类有帮助，但整体效果仍低于 weighted_ce |

本表记录已提供的 10 epochs 结果，未重新训练。weighted_ce 的 average train loss（0.0501）引用前文已有的 10 epochs class weights 稳定性实验记录。前文的 5 epochs baseline 保留为历史记录，仅用于观察增加训练轮数的影响，不作为本表同轮数 loss 对比的结果。

### 观察与结论

1. weighted_ce 的整体效果最好，mean IoU 为 94.99%，weed IoU 为 89.19%，两项指标均为本组实验最高。
2. focal loss 的 weed IoU 为 81.30%，明显高于普通 ce 的 76.14%，提高了 5.16 个百分点，说明 focal loss 对难分类样本和少数类有帮助。
3. dice loss 的 weed IoU 为 66.27%，在本实验中低于 ce、focal 和 weighted_ce。
4. 普通 ce 在训练 10 epochs 后 weed IoU 达到 76.14%，比 5 epochs baseline 的 0.26% 明显更好，说明增加训练轮数本身也有帮助。
5. 不同 loss 的 loss 数值不能直接比较大小，因为计算公式不同；应主要比较 pixel accuracy、mean IoU 和 per-class IoU，不能根据 average train loss 的大小判断哪种 loss 的分割效果更好。
6. 在当前 synthetic crop/weed/background 分割实验中，weighted_ce 是本次比较的四种 loss 中最有效的少数类 weed 改进方法。
7. 真实农业遥感任务中，也应该重点关注 weed IoU 和 mean IoU，而不是只看 pixel accuracy，以判断模型是否有效识别少数类杂草。


## RGB 输入 vs 多光谱输入实验

实验目的：比较 RGB 输入和 Green/Red/RedEdge/NIR/NDVI/NDRE 多光谱输入对 weed IoU、mean IoU 的影响，探索利用作物、杂草、土壤的光谱差异进行分割。

输入分别为 3 通道 RGB 和 6 通道模拟多光谱。多光谱中的 NDVI、NDRE 由 NIR/Red/RedEdge 按归一化差值公式计算。相同种子下两种输入共享 mask 和 Green/Red 波段，额外模拟红边、近红外反射率；多光谱输入比 RGB 包含更多植被光谱信息，但这些反射率不是实测数据。

对比时保持 weighted_ce（background=1.0、crop=2.0、weed=6.0）、epochs=10、batch-size=16、lr=0.001、训练 seed=42、测试 seed=2026 和评估方式一致，仅调整输入及 U-Net 第一层输入通道数。RGB 已有结果引用此前记录，多光谱结果根据已提供的实验指标补充，本次未重新训练。multispectral 的 average train loss 为 0.0360。

| input type | loss | epochs | pixel accuracy | mean IoU | background IoU | crop IoU | weed IoU | 观察 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgb | weighted_ce | 10 | 98.85% | 94.99% | 98.79% | 96.98% | 89.19% | weed IoU 仍低于 background/crop IoU，但高于 multispectral | 作为输入对比的参考基线，本次 weed IoU 高于多光谱输入 |
| multispectral | weighted_ce | 10 | 98.97% | 94.97% | 98.95% | 97.90% | 88.07% | pixel accuracy、background IoU 和 crop IoU 略高于 RGB，mean IoU 基本持平，weed IoU 下降 | 当前 synthetic 数据设置下，多光谱输入没有明显提升 weed 类别分割效果 |

```bash
python train_synthetic_unet.py --epochs 10 --loss weighted_ce --input-type rgb
python train_synthetic_unet.py --epochs 10 --loss weighted_ce --input-type multispectral
python visualize_synthetic_prediction.py --loss weighted_ce --input-type multispectral
python plot_synthetic_history.py --loss weighted_ce --input-type multispectral
```

新模型与 history CSV 按 input type 和 loss 命名，旧文件不自动迁移。以上命令保留为实验复现参考，本次仅更新已提供的结果，未执行训练、联网或下载文件。后续可结合预测错误图与曲线判断收益和稳定性。

### 观察与结论

1. multispectral 输入的 pixel accuracy 为 98.97%，略高于 RGB 的 98.85%。
2. multispectral 的 background IoU（98.95%）和 crop IoU（97.90%）略高于 RGB 的 98.79% 和 96.98%。
3. mean IoU 基本持平，RGB 为 94.99%，multispectral 为 94.97%。
4. weed IoU 从 RGB 的 89.19% 下降到 multispectral 的 88.07%，下降了 1.12 个百分点。
5. 在当前 synthetic 数据设置下，多光谱输入没有明显提升 weed 类别分割效果。
6. 这不能说明多光谱无效，因为当前数据是模拟数据，RGB 已经包含较明显的类别差异。
7. 在真实无人机多光谱数据中，NIR、RedEdge、NDVI、NDRE 仍可能对区分作物、杂草和土壤有重要价值。
8. 更严谨的实验需要多 seed 重复运行，并在真实 WeedMap 数据上验证。
