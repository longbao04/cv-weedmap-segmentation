"""Compare four best checkpoints on one seed-0 WeedMap validation sample."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import random_split

from unet import MobileNetV2ShallowUNet, SmallUNet
from visualize_real_weedmap_prediction import (
    CLASS_NAMES,
    IGNORE_COLOR,
    MASK_COLORS,
    calculate_metrics,
    colorize_mask,
    make_error_map,
    select_device,
)
from weedmap_dataset import WeedMapDataset


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_LIST = PROJECT_ROOT / "splits/real_weedmap_common_samples.csv"
MODELS = (
    (
        "SmallUNet\nweighted CE",
        SmallUNet,
        "real_weedmap_common_multispectral_weighted_ce_20epochs_best.pth",
    ),
    (
        "SmallUNet\nboundary CE r5_w4",
        SmallUNet,
        "real_weedmap_common_multispectral_boundary_weighted_ce_r5_w4_20epochs_seed0_best.pth",
    ),
    (
        "MobileNetV2ShallowUNet\nweighted CE",
        MobileNetV2ShallowUNet,
        "real_weedmap_common_mobilenetv2_shallow_weighted_ce_20epochs_seed0_best.pth",
    ),
    (
        "MobileNetV2ShallowUNet\nboundary CE r5_w4",
        MobileNetV2ShallowUNet,
        "real_weedmap_common_mobilenetv2_shallow_boundary_r5_w4_20epochs_seed0_best.pth",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index within the seed-0 validation subset (default: 0)")
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT_ROOT / "data/weedmap")
    return parser.parse_args()


def display_rgb(dataset, dataset_index, image):
    """Use the real RGB tile; fall back to an explicitly labeled composite."""
    input_paths, _, _ = dataset.samples[dataset_index]
    rgb_path = input_paths[0].parent.parent / "RGB" / input_paths[0].name
    if rgb_path.is_file():
        return WeedMapDataset._load_rgb(rgb_path), "RGB visualization"
    return (image[[1, 0, 3]].permute(1, 2, 0).numpy(),
            "R/G/NIR composite (RGB unavailable)")


def unavailable_panel(axis, title, reason):
    axis.set_facecolor("#eeeeee")
    axis.text(0.5, 0.5, reason, ha="center", va="center",
              transform=axis.transAxes, fontsize=11, color="#555555")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def main():
    args = parse_args()
    dataset = WeedMapDataset(
        data_root=args.data_root,
        input_type="multispectral",
        filter_empty=True,
        sample_list_csv=SAMPLE_LIST,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("Common split needs at least two valid samples")
    _, validation = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    if not 0 <= args.sample_index < len(validation):
        raise ValueError(
            f"--sample-index must be 0 to {len(validation) - 1}; "
            f"received {args.sample_index}"
        )

    image, label_tensor = validation[args.sample_index]
    dataset_index = validation.indices[args.sample_index]
    label = label_tensor.numpy()
    rgb, rgb_title = display_rgb(dataset, dataset_index, image)
    device = select_device()
    print(f"common split: {SAMPLE_LIST}")
    print(f"validation split: seed=0, sample-index={args.sample_index}, "
          f"dataset-index={dataset_index}")
    print(f"device: {device}")

    fig, axes = plt.subplots(2, 6, figsize=(24, 9), squeeze=False)
    axes[0, 0].imshow(rgb, interpolation="nearest")
    axes[0, 0].set_title(rgb_title)
    axes[0, 1].imshow(colorize_mask(label, include_ignore=True),
                      interpolation="nearest")
    axes[0, 1].set_title("Ground Truth")
    axes[1, 0].imshow(image[4].numpy(), cmap="RdYlGn", vmin=0, vmax=1,
                      interpolation="nearest")
    axes[1, 0].set_title("NDVI (input channel 5)")
    overlay = rgb.copy()
    valid = label != 255
    for class_id in (1, 2):
        selected = valid & (label == class_id)
        overlay[selected] = 0.55 * rgb[selected] + 0.45 * MASK_COLORS[class_id]
    axes[1, 1].imshow(np.clip(overlay, 0, 1), interpolation="nearest")
    axes[1, 1].set_title("Ground Truth overlay")

    for column, (title, model_class, filename) in enumerate(MODELS, start=2):
        model_path = PROJECT_ROOT / "models" / filename
        if not model_path.is_file():
            print(f"{title.replace(chr(10), ' ')}: model file missing: {model_path}")
            unavailable_panel(axes[0, column], title, "Model file missing")
            unavailable_panel(axes[1, column], "Error map", "Model file missing")
            continue

        model = model_class(in_channels=5, num_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
        model.eval()
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0]
        prediction = prediction.cpu().numpy()
        accuracy, class_ious, mean_iou = calculate_metrics(prediction, label)
        print(f"{title.replace(chr(10), ' ')}:")
        print(f"  pixel accuracy: {accuracy:.2%}")
        print(f"  mean IoU: {mean_iou:.2%}")
        for name, iou in zip(CLASS_NAMES, class_ious):
            print(f"  {name} IoU: {iou:.2%}")

        axes[0, column].imshow(colorize_mask(prediction, include_ignore=False),
                               interpolation="nearest")
        axes[0, column].set_title(title)
        axes[1, column].imshow(make_error_map(prediction, label),
                               interpolation="nearest")
        axes[1, column].set_title("Error map (white = wrong)")

    for axis in axes.flat:
        axis.axis("off")
    legend = [Patch(color=MASK_COLORS[i], label=name)
              for i, name in enumerate(CLASS_NAMES)]
    legend.append(Patch(color=IGNORE_COLOR, label="ignore"))
    fig.legend(handles=legend, loc="lower center", ncol=4)
    fig.suptitle("WeedMap common validation comparison | multispectral | "
                 f"seed 0 | sample {args.sample_index}")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output = (PROJECT_ROOT / "outputs" /
              f"best_model_prediction_comparison_sample{args.sample_index}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"comparison saved to {output}")


if __name__ == "__main__":
    main()
