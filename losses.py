"""多分类语义分割损失：适用于类别不平衡或小目标分割任务。"""

import torch
from torch import nn
from torch.nn import functional as F


def _prepare_targets(logits, targets, num_classes, ignore_index):
    # logits 为 [B,C,H,W] 原始分数，targets 为 [B,H,W] 类别编号。
    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError("logits 必须为 [B,num_classes,H,W]")
    if targets.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("targets 必须为与 logits 对应的 [B,H,W]")
    valid = (torch.ones_like(targets, dtype=torch.bool) if ignore_index is None
             else targets != ignore_index)
    # 忽略标签可能为 255 或 -100，先替换为合法编号，避免索引越界。
    safe_targets = targets.masked_fill(~valid, 0).long()
    return safe_targets, valid


class DiceLoss(nn.Module):
    """按类别计算 soft Dice，再平均三个类别（包括 background）。"""

    def __init__(self, num_classes=3, ignore_index=None, smooth=1e-6):
        super().__init__()
        if num_classes < 1 or not 0 < smooth < float("inf"):
            raise ValueError("num_classes 必须为正整数，smooth 必须为有限正数")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        safe_targets, valid = _prepare_targets(
            logits, targets, self.num_classes, self.ignore_index)
        probabilities = logits.softmax(dim=1)
        # one-hot 将类别编号转为各类别的真实区域，形状为 [B,C,H,W]。
        truth = F.one_hot(safe_targets, self.num_classes).permute(0, 3, 1, 2)
        mask = valid.unsqueeze(1).to(probabilities.dtype)
        probabilities = probabilities * mask
        truth = truth.to(probabilities.dtype) * mask
        # Dice Loss 更关注预测区域和真实区域的重叠。
        # 各类别分别计算并平均，帮助少数类、小目标参与优化。
        dimensions = (0, 2, 3)
        intersection = (probabilities * truth).sum(dim=dimensions)
        total = probabilities.sum(dim=dimensions) + truth.sum(dim=dimensions)
        dice = (2 * intersection + self.smooth) / (total + self.smooth)
        # 全部像素被忽略时结果为可反向传播的零损失。
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """多分类 Focal Loss：默认 gamma=2，对有效像素取平均。"""

    def __init__(self, num_classes=3, ignore_index=None, gamma=2.0):
        super().__init__()
        if num_classes < 1 or not 0 <= gamma < float("inf"):
            raise ValueError("num_classes 必须为正整数，gamma 必须为有限非负数")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits, targets):
        safe_targets, valid = _prepare_targets(
            logits, targets, self.num_classes, self.ignore_index)
        # log_softmax 比先 softmax 再取对数更稳定。
        log_pt = logits.log_softmax(dim=1).gather(
            1, safe_targets.unsqueeze(1)).squeeze(1)
        # Focal Loss 更关注难分类样本：降低容易分类像素的贡献。
        # 常用于类别不平衡或小目标分割；gamma=0 时退化为普通交叉熵。
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        return (loss * valid).sum() / valid.sum().clamp_min(1)
