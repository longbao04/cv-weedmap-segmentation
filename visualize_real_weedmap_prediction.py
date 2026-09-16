"""Visualize one prediction from a U-Net trained on real WeedMap data."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from unet import SmallUNet
from weedmap_dataset import WeedMapDataset


NUM_CLASSES = 3
IGNORE_INDEX = 255
CLASS_NAMES = ("background", "crop", "weed")
INPUT_CHOICES = ("rgb", "multispectral")
LOSS_CHOICES = ("ce", "weighted_ce")
MASK_COLORS = np.array(
    [
        (0.0, 0.0, 0.0),  # background: black
        (0.0, 1.0, 0.0),  # crop: green
        (1.0, 0.0, 0.0),  # weed: red
    ],
    dtype=np.float32,
)
IGNORE_COLOR = np.array((0.5, 0.5, 0.5), dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/weedmap")
    parser.add_argument(
        "--input-type", choices=INPUT_CHOICES, default="multispectral"
    )
    parser.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default="weighted_ce",
        help="训练模型使用的 loss，仅用于图标题和实验标识",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/real_weedmap_multispectral_weighted_ce.pth"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/real_weedmap_prediction_multispectral_weighted_ce.png"
        ),
    )
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_display_image(image, input_type):
    """Return a displayable RGB image and an honest description of its bands."""
    if input_type == "rgb":
        return image[:3].permute(1, 2, 0).numpy(), "RGB image"

    # WeedMapDataset returns multispectral channels as G, R, RE, NIR, NDVI.
    # Blue is unavailable, so NIR is used in its place for visualization only.
    composite = image[[1, 0, 3]].permute(1, 2, 0).numpy()
    return composite, "RGB image (R/G/NIR composite)"


def colorize_mask(mask, include_ignore):
    color = np.zeros((*mask.shape, 3), dtype=np.float32)
    for class_id, class_color in enumerate(MASK_COLORS):
        color[mask == class_id] = class_color
    if include_ignore:
        color[mask == IGNORE_INDEX] = IGNORE_COLOR
    return color


def calculate_metrics(prediction, label):
    """Calculate accuracy and IoU using only pixels whose label is not 255."""
    valid = label != IGNORE_INDEX
    if not np.any(valid):
        raise ValueError("所选样本没有可用于指标计算的有效标签像素。")

    valid_prediction = prediction[valid]
    valid_label = label[valid]
    pixel_accuracy = float(np.mean(valid_prediction == valid_label))
    ious = []
    for class_id in range(NUM_CLASSES):
        intersection = np.count_nonzero(
            (valid_prediction == class_id) & (valid_label == class_id)
        )
        union = np.count_nonzero(
            (valid_prediction == class_id) | (valid_label == class_id)
        )
        ious.append(intersection / union if union else math.nan)

    present_ious = [iou for iou in ious if not math.isnan(iou)]
    mean_iou = float(np.mean(present_ious)) if present_ious else math.nan
    return pixel_accuracy, ious, mean_iou


def make_error_map(prediction, label):
    error_map = np.zeros((*label.shape, 3), dtype=np.float32)
    valid = label != IGNORE_INDEX
    error_map[valid & (prediction != label)] = (1.0, 1.0, 1.0)
    error_map[~valid] = IGNORE_COLOR
    return error_map


def make_overlay(display_image, prediction, label, alpha=0.45):
    overlay = display_image.copy()
    valid = label != IGNORE_INDEX
    for class_id in (1, 2):
        selected = valid & (prediction == class_id)
        overlay[selected] = (
            (1.0 - alpha) * overlay[selected] + alpha * MASK_COLORS[class_id]
        )
    return np.clip(overlay, 0.0, 1.0)


def main():
    args = parse_args()
    device = select_device()

    if not args.model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{args.model_path}")

    dataset = WeedMapDataset(
        data_root=args.data_root,
        input_type=args.input_type,
        filter_empty=True,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"--sample-index 必须在 0 到 {len(dataset) - 1} 之间，"
            f"当前值为 {args.sample_index}。"
        )

    image, label = dataset[args.sample_index]
    in_channels = 3 if args.input_type == "rgb" else 5
    model = SmallUNet(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()

    label_array = label.numpy()
    display_image, image_title = make_display_image(image, args.input_type)
    ground_truth_color = colorize_mask(label_array, include_ignore=True)
    prediction_color = colorize_mask(prediction, include_ignore=False)
    error_map = make_error_map(prediction, label_array)
    overlay = make_overlay(display_image, prediction, label_array)
    pixel_accuracy, class_ious, mean_iou = calculate_metrics(
        prediction, label_array
    )

    if args.input_type == "multispectral":
        ndvi_image = image[4].numpy()
        ndvi_title = "NDVI (input channel 5)"
        ndvi_kwargs = {"cmap": "RdYlGn", "vmin": 0.0, "vmax": 1.0}
    else:
        ndvi_image = np.dot(display_image[..., :3], (0.299, 0.587, 0.114))
        ndvi_title = "No NDVI (RGB grayscale)"
        ndvi_kwargs = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    panels = (
        (display_image, image_title, {}),
        (ndvi_image, ndvi_title, ndvi_kwargs),
        (ground_truth_color, "GroundTruth mask", {}),
        (prediction_color, "Prediction mask", {}),
        (error_map, "Error map (white = wrong)", {}),
        (overlay, "Prediction overlay", {}),
    )
    for axis, (panel, title, kwargs) in zip(axes.flat, panels):
        axis.imshow(panel, interpolation="nearest", **kwargs)
        axis.set_title(title)
        axis.axis("off")

    legend_handles = [
        Patch(color=MASK_COLORS[index], label=name)
        for index, name in enumerate(CLASS_NAMES)
    ]
    legend_handles.append(Patch(color=IGNORE_COLOR, label="ignore"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=4)
    fig.suptitle(
        f"Real WeedMap prediction | sample={args.sample_index} | "
        f"input={args.input_type} | loss={args.loss}"
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    fig.subplots_adjust(hspace=0.18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"device: {device}")
    print(f"sample index: {args.sample_index}")
    print(f"pixel accuracy: {pixel_accuracy:.2%}")
    for name, iou in zip(CLASS_NAMES, class_ious):
        print(f"{name} IoU: {iou:.2%}")
    print(f"mean IoU: {mean_iou:.2%}")
    print(f"预测可视化已保存到 {args.output}")


if __name__ == "__main__":
    main()
