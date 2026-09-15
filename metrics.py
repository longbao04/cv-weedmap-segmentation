"""分割指标：输入预测类别和真实类别，均为 [B,H,W] 或 [H,W]。"""

import torch

CLASS_NAMES = ("background", "crop", "weed")


def confusion_matrix(prediction, target, num_classes=3):
    # 行是真实类别，列是预测类别；累加整份测试集后再计算指标。
    prediction = prediction.detach().reshape(-1).long()
    target = target.detach().reshape(-1).long()
    if prediction.shape != target.shape:
        raise ValueError("预测与标签的像素数量必须相同")
    encoded = target * num_classes + prediction
    return torch.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def pixel_accuracy(prediction=None, target=None, *, matrix=None):
    # 像素级准确率 = 预测正确的像素数 / 总像素数。
    if matrix is None:
        matrix = confusion_matrix(prediction, target)
    matrix = matrix.double()
    return (matrix.diag().sum() / matrix.sum().clamp_min(1)).item()


def per_class_iou(prediction=None, target=None, num_classes=3, *, matrix=None):
    # IoU = 预测与真值的交集 / 并集，分别计算每一个类别。
    if matrix is None:
        matrix = confusion_matrix(prediction, target, num_classes)
    matrix = matrix.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    # 真值和预测都没有该类时 IoU 未定义，记为 NaN。
    return torch.where(union > 0, intersection / union,
                       torch.full_like(union, float("nan")))


def mean_iou(prediction=None, target=None, num_classes=3, *, matrix=None):
    # mIoU 为各类 IoU 的平均值；跳过并集为零的未出现类别。
    values = per_class_iou(prediction, target, num_classes, matrix=matrix)
    return torch.nanmean(values).item()
