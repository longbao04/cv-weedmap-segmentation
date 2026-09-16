# 真实 WeedMap 实验记录

## 真实数据训练目标

本实验使用真实 WeedMap tiles 训练现有的小型 U-Net，完成 background、crop 和 weed 三类像素级语义分割。训练过程记录验证集 pixel accuracy、mean IoU 及三个类别各自的 IoU，为后续输入形式和损失函数对比提供统一基线。

## 为什么使用 `ignore_index=255`

真实数据的有效区域 mask 会把无数据、边缘或不应参与监督的像素标记为 255。模型只有 0=background、1=crop、2=weed 三个输出类别，255 不是第 4 类。训练和评估都必须忽略这些像素，否则会错误惩罚模型、污染混淆矩阵，并使指标不能代表有效标注区域的表现。

## 为什么先使用 multispectral

多光谱输入包含 G、R、RedEdge、NIR 和 NDVI 五个通道。RedEdge、NIR 和植被指数能补充 RGB 中不明显的植被光谱信息，因此先用 multispectral 建立更贴近 WeedMap 数据特点的真实数据基线。

## 为什么使用 weighted_ce

真实农田图像通常存在明显类别不平衡，background 像素较多，weed 像素较少。默认使用 background=1.0、crop=4.0、weed=8.0 的 weighted cross entropy，提高作物和杂草误分类的代价，减少训练被背景类别主导的风险。最终效果仍需结合各类别 IoU 判断。

## 后续比较

后续将在相同数据划分、随机种子、训练轮数和学习率下比较 RGB 与 multispectral 输入，重点观察 mean IoU、crop IoU 和 weed IoU，检验额外光谱通道是否在真实数据上带来稳定收益。
