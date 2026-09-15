# Synthetic segmentation 实验记录

本阶段仅使用模拟数据，未下载或训练真实 WeedMap 数据。保持 U-Net 主体结构不变，比较普通 CrossEntropyLoss 和 weighted CrossEntropyLoss。

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
6. 这个实验说明，在 crop/weed/background 这类类别不平衡任务中，只看 pixel accuracy 不够，weed IoU 和 mean IoU 更重要。
7. 在本次模拟数据实验中，class weights 是处理小目标/少数类识别困难的有效 baseline 改进方法。
