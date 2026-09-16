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

## 真实 RGB vs Multispectral 输入对比实验

真实 WeedMap 数据上，RGB 和 multispectral 输入均使用 `weighted_ce` 训练 10 epochs，验证集结果如下：

| Input | Train Loss | Val Pixel Acc | Mean IoU | Background IoU | Crop IoU | Weed IoU |
|---|---:|---:|---:|---:|---:|---:|
| RGB | 0.2956 | 94.28% | 65.27% | 95.49% | 65.75% | 34.56% |
| Multispectral | 0.1781 | 97.22% | 69.21% | 97.69% | 64.17% | 45.76% |

Multispectral 的 mean IoU 和 weed IoU 更高，初步表明多光谱通道对杂草识别更有帮助。RGB 的 crop IoU 略高，但整体 mean IoU 和 weed IoU 不如 multispectral。对作物、杂草、土壤/背景区分任务来说，weed 是关键难点，因此 multispectral 当前更值得继续深入。

**对比局限：**RGB Dataset 有 454 个有效样本，multispectral Dataset 有 884 个有效样本，样本集合不完全一致，因此当前结果仍是初步对比，不能单独归因于输入通道。后续需要让 RGB 和 multispectral 使用同一批样本，再比较指标。

运行 `python build_common_sample_list.py` 可生成 `splits/real_weedmap_common_samples.csv`。列表只包含 RGB 与五通道 multispectral 输入、彩色标签及 mask 都齐全，且过滤空输入和无前景标签后的样本。构建这份共享样本列表，是为了让 RGB 和 multispectral 在完全相同的样本上使用相同训练/验证划分，公平比较输入通道带来的差异；现有对比结果尚未使用此列表。

为了严格比较 RGB 和 multispectral，需要让二者使用同一个 `sample_list_csv`（训练脚本参数 `--sample-list-csv`），避免样本数量不同导致比较不公平。

## 下一步计划

- 基于 RGB 和 multispectral 共享样本列表，在相同样本与数据划分上做严格公平对比；
- 尝试 weed class weight = 12 或 16；
- 继续做 20 epochs 和多 seed 实验。
