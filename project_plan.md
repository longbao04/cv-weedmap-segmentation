# 项目计划

**当前阶段：synthetic U-Net baseline 的类别不平衡处理。** 已用 NumPy 模拟数据跑通 image / mask / 模型 / 指标 / 可视化流程。Baseline 训练 5 个 epochs 后 pixel accuracy 为 93.16%，但 weed IoU 仅为 0.26%，需要关注少数类 weed 的分割效果。本阶段不下载真实数据，不训练真实 WeedMap，不修改 U-Net 主体结构。

接下来比较普通 CrossEntropyLoss 与 weighted CrossEntropyLoss：保持模拟数据种子、epochs、batch-size 和 lr 一致，通过 `--use-class-weights` 启用 background=1.0、crop=2.0、weed=6.0 的权重。分别保存普通与加权模型，比较 pixel accuracy、mIoU、各类 IoU 和预测可视化，重点观察 weed IoU 是否改善。Baseline 已记录在 `synthetic_experiment_notes.md`，加权实验结果待运行后填写。

以下是真实数据阶段的后续学习路线；synthetic baseline 是进入这些阶段前的练习，不能替代真实数据验证。

1. **阶段 1：理解 WeedMap 数据结构。** 检查 Tiles 目录、文件后缀、通道命名、image 与 annotation 的对应关系，核实标签值和类别映射。使用 `inspect_dataset_structure.py` 做初步统计。
2. **阶段 2：读取 RGB / NIR / RedEdge / NDVI 等通道。** 核实传感器、波段顺序、尺寸、数据类型和数值范围；检查通道是否配准。若计算 NDVI，使用 `(NIR - R) / (NIR + R)`，处理零分母，并先确认波段数值是否适合计算。
3. **阶段 3：可视化 image 和 mask。** 显示单波段、RGB、标签颜色图和 overlay，检查空间对齐与标签编号。先通过假数据演示理解，再读取真实样本。
4. **阶段 4：建立传统 baseline，例如 NDVI 阈值。** 先区分植被与非植被，分析阈值变化的效果。NDVI 阈值通常不能单独可靠地区分 crop 和 weed，需明确这一限制。
5. **阶段 5：训练 U-Net baseline。** 数据理解与检查完成后，建立小规模训练流程；按独立地块或采集区域划分训练、验证和测试数据，减少相邻图块泄漏。记录每类 IoU、mIoU 与 confusion matrix。
6. **阶段 6：比较 RGB 输入、多光谱输入、植被指数输入。** 在一致的数据划分、训练设置和评估标准下比较，关注 crop 与 weed 的混淆以及增加通道的实际收益。
7. **阶段 7：迁移到水稻/柑橘实验田数据。** 核实目标传感器、通道、种植布局与标注，建立本地测试集，再评估直接迁移或微调的效果。
