# Synthetic segmentation 实验记录

本阶段仅使用模拟数据，未下载或训练真实 WeedMap 数据。保持 U-Net 主体结构不变，比较普通 CrossEntropyLoss 和 weighted CrossEntropyLoss。

## 训练曲线分析

训练曲线用于观察每个 epoch 的平均训练 loss、pixel accuracy、mIoU 和各类 IoU。重点检查 loss 是否下降、mIoU 和 weed IoU 是否持续提升或趋于稳定，并在相同训练设置下比较 baseline 与 weighted 的变化。整体准确率较高时，仍需单独关注 weed IoU。

训练记录分别保存为 `outputs/synthetic_history_baseline.csv` 和 `outputs/synthetic_history_weighted.csv`。运行以下命令绘制对应曲线：

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

模型保存为 `models/synthetic_unet_weighted.pth`。以下记录已提供的实验结果；比较时应保持 epochs、batch-size、lr 和模拟数据种子一致。

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
