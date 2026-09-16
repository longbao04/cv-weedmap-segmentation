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

## 实验报告

[`synthetic_segmentation_report.md`](synthetic_segmentation_report.md) 是 synthetic U-Net 分割实验总结报告，汇总了 baseline、class weights、训练稳定性、loss function 对比，以及 RGB 与 multispectral 输入对比结果。

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
├── visualize_sample_placeholder.py
├── visualize_real_weedmap_sample.py
├── weedmap_dataset.py
├── synthetic_dataset.py
├── unet.py
├── losses.py
├── metrics.py
├── train_synthetic_unet.py
├── train_real_weedmap_unet.py
├── plot_synthetic_history.py
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

## 训练真实 WeedMap U-Net

使用本地真实 WeedMap 数据、现有 `WeedMapDataset` 和 `SmallUNet` 训练三分类模型：

```bash
python train_real_weedmap_unet.py --input-type multispectral --loss weighted_ce --epochs 3
```

默认读取 `data/weedmap`，按固定随机种子划分 80% 训练集和 20% 验证集。多光谱输入使用 G、R、RedEdge、NIR、NDVI 五个通道；RGB 输入使用三个通道。标签 255 作为 `ignore_index`，不参与 loss 或验证指标，也不会被当作第 4 类。默认 weighted CE 权重为 background=1.0、crop=4.0、weed=8.0。

每个 epoch 输出 train loss、验证集 pixel accuracy、mean IoU 和三类 IoU。默认模型保存到 `models/real_weedmap_<input_type>_<loss>.pth`，可用 `--save-path` 修改；history 保存到 `outputs/real_weedmap_history_<input_type>_<loss>.csv`。`models/` 和 `outputs/` 会自动创建且已被 Git 忽略。可用 `--data-root`、`--batch-size`、`--lr`、`--seed` 和 `--num-workers` 调整训练参数。

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
