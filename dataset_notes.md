# WeedMap 数据理解笔记

以下内容结合项目背景与当前本地 RedEdge_004 数据整理。未联网核实的数据集定义仍应以 WeedMap 官方说明为准。

- **数据来源：** 无人机多光谱影像。
- **场景：** sugar beet field（甜菜田）。
- **目标类别：** background、crop、weed。background 可以近似理解为土壤/非目标背景，但具体范围要以标注定义为准。
- **RedEdge-M 数据可能包含：** B（蓝）、G（绿）、R（红）、RE（红边）、NIR（近红外）、NDVI、RGB、CIR 等通道或影像产品。
- **Sequoia 数据可能包含：** G、R、RE、NIR、NDVI、CIR 等通道或影像产品。

RGB 是可见光组合，CIR 是彩色红外组合，NDVI 是由红光与近红外计算的植被指数；这些产品与单独的原始波段不同。不能仅根据名称推断实际波段顺序、位深、尺度或配准状态。

mask / annotation 是语义分割标签，用于表示每个像素所属的类别，不是整张图的类别标签。读取真实数据后，应检查标签是否为单通道整数、颜色编码或其他形式，核实类别映射、忽略值以及与影像的尺寸和空间对应关系。演示脚本中的 0/1/2 仅为本项目假数据约定。

后续可以使用 confusion matrix（混淆矩阵）分析预测结果：约定**行是真实标签，列是预测标签**，按照 background、crop、weed 排列。对角线表示正确分类，非对角线表示混淆；例如 crop 行、weed 列表示真实作物被预测为杂草。评估库可能采用不同约定，使用前应核实。

真实数据中，crop 和 weed 比 soil/background 更难区分：作物和杂草都属于植被，光谱与外观可能相似，还受到生长阶段、遮挡、阴影和空间分辨率影响。植被指数可以帮助区分植被与非植被，但不能简单等同于作物/杂草分类。

## 真实 WeedMap 标签映射验证

对当前本地 8 个 WeedMap Tiles 子集的 `GroundTruth_color`、`GroundTruth_iMap` 和 `mask` 配对文件进行逐像素验证后，标签映射为：

- color 标签中 black、green、red 分别对应 background、crop、weed；
- 本地像素级验证发现 iMap 中 green crop 对应 `10000`；
- red weed 对应 `2`；
- black background 对应 `0`；
- `mask=0` 是有效区域，`mask=255` 是无效/no-data 区域。

因此，后续训练建议优先从 `GroundTruth_color` 生成 `0/1/2` 训练 mask，或者将 iMap 重新映射为 `0=background, 1=crop, 2=weed`。逐样本及逐子集验证可运行 `python analyze_real_weedmap_labels.py`；生成的 CSV 位于 `outputs/real_weedmap_label_summary.csv`，不提交到 Git。

## Dataset 读取规则

- `tile/RGB` 用于 `rgb` 输入。
- `tile/G`、`tile/R`、`tile/RE`、`tile/NIR`、`tile/NDVI` 用于 `multispectral` 输入，通道按此顺序堆叠。
- `GroundTruth_color` 转换为 `0/1/2` 训练标签：black 为 background，green 为 crop，red 为 weed。
- 有效区域文件中 `mask=255` 的无效/no-data 区域转为 `ignore_index=255`。
- 数据中存在输入全黑或标签全为 `ignore_index` 的空样本。`WeedMapDataset` 默认使用 `filter_empty=True` 跳过这些样本，并要求至少 1000 个有效像素和 1 个 crop/weed 前景像素。
- 后续训练不应使用 label 全为 `ignore_index` 的样本。
- 后续训练使用 `CrossEntropyLoss` 时应设置 `ignore_index=255`。

## 当前 RedEdge_004 数据结构与统计

当前本地子集位于 `data/weedmap/RedEdge_004/004`，包含以下目录：

```text
004/
├── tile/
│   ├── RGB/
│   ├── CIR/
│   ├── B/
│   ├── G/
│   ├── R/
│   ├── RE/
│   ├── NIR/
│   └── NDVI/
├── groundtruth/
└── mask/
```

当前统计发现：

- RedEdge_004 有 117 个样本。
- 图像大小为 360×480（高×宽）。
- 当前子集的 GT_iMap 出现 background=0、weed=2、crop=10000。
- 当前统计中值 1 没有出现；本地逐像素验证表明值 10000 是 crop，不能将其当作 ignore。
- 因此读取真实数据时需要先将 iMap 的 10000 重映射为训练类别 1，或直接从彩色标签生成训练 mask。
