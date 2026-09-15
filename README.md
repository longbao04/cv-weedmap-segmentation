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

项目已进入 synthetic U-Net baseline 阶段，在真实 WeedMap 数据之前用模拟数据跑通分割流程。项目不包含或下载真实数据，不训练真实 WeedMap。数据理解摘要见 `dataset_notes.md`，后续安排见 `project_plan.md`。

## 项目结构

```text
cv-weedmap-segmentation/
├── README.md
├── project_plan.md
├── dataset_notes.md
├── inspect_dataset_structure.py
├── visualize_sample_placeholder.py
├── synthetic_dataset.py
├── unet.py
├── metrics.py
├── train_synthetic_unet.py
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

## Synthetic segmentation baseline

`SyntheticWeedDataset` 用 NumPy 生成 128×128 的 RGB 农田图像：棕色土壤背景、规则排列的作物、随机小块杂草，并加入光照变化和噪声。图像 tensor 为 `[3,128,128]`、数值范围为 0～1；mask 为 `[128,128]` 的 long tensor，类别编号为 0/1/2。

小型 U-Net 通过 encoder 提取特征、decoder 恢复分辨率，skip connection 保留空间细节。输出为 `[B,3,128,128]` 类别分数，使用 CrossEntropyLoss 和 Adam 训练；预测时沿类别维取 argmax 得到 mask。

在已具备 `requirements.txt` 中依赖的本地 Python 环境中运行，不需要下载数据：

```bash
python train_synthetic_unet.py --epochs 5
python visualize_synthetic_prediction.py
```

训练参数还包括 `--batch-size`（默认 16）与 `--lr`（默认 0.001）。程序自动选择 MPS 或 CPU，使用 256 张训练图像和 64 张独立种子的测试图像，每个 epoch 输出平均训练 loss 和整份测试集的指标。权重保存为 `models/synthetic_unet.pth`。

pixel accuracy 是像素级准确率，即预测正确像素占总像素的比例。IoU 是语义分割常用指标，表示某类别预测区域与真实区域的交集除以并集；mIoU 是所有类别 IoU 的平均值。若某类在预测和真值中均不存在，该类 IoU 记为 NaN，不参与平均；背景占比高时应结合 crop、weed IoU 判断效果。

可视化使用三个新的模拟样本，显示 RGB image、true mask、predicted mask 和 error map（红色表示错误），并保存到 `outputs/synthetic_prediction.png`。模拟数据只用于理解 image → mask → 模型 → 指标 → 可视化流程，指标不能代表真实农田表现。
