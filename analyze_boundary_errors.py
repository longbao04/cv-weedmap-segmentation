"""Analyze whether real WeedMap prediction errors fall near label boundaries."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unet import SmallUNet
from weedmap_dataset import WeedMapDataset


IGNORE_INDEX = 255
NUM_CLASSES = 3
MASK_COLORS = np.array(
    [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/weedmap")
    parser.add_argument(
        "--sample-list-csv",
        default="splits/real_weedmap_common_samples.csv",
    )
    parser.add_argument(
        "--input-type", choices=("rgb", "multispectral"), default="multispectral"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "models/real_weedmap_common_multispectral_weighted_ce_20epochs_best.pth"
        ),
        help="Path to the model checkpoint (default: %(default)s)",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/boundary_error_analysis_sample0.png"),
        help="Path for the saved visualization (default: %(default)s)",
    )
    return parser.parse_args()


def ground_truth_boundary(label):
    """Mark both sides of a class change between valid four-neighbors."""
    valid = label != IGNORE_INDEX
    boundary = np.zeros(label.shape, dtype=bool)

    horizontal = valid[:, 1:] & valid[:, :-1] & (label[:, 1:] != label[:, :-1])
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    vertical = valid[1:, :] & valid[:-1, :] & (label[1:, :] != label[:-1, :])
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def dilate_within_radius(boundary, radius):
    """Dilate by a Euclidean pixel radius using NumPy; no SciPy required."""
    height, width = boundary.shape
    padded = np.pad(boundary, radius, mode="constant")
    nearby = np.zeros_like(boundary)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                nearby |= padded[
                    radius + dy : radius + dy + height,
                    radius + dx : radius + dx + width,
                ]
    return nearby


def colorize_mask(mask, show_ignore=False):
    colored = np.zeros((*mask.shape, 3), dtype=np.float32)
    for class_id, color in enumerate(MASK_COLORS):
        colored[mask == class_id] = color
    if show_ignore:
        colored[mask == IGNORE_INDEX] = (0.5, 0.5, 0.5)
    return colored


def load_display_rgb(dataset, sample_index, image):
    if dataset.input_type == "rgb":
        return image[:3].permute(1, 2, 0).numpy(), "RGB image"

    input_paths, _, _ = dataset.samples[sample_index]
    rgb_path = input_paths[0].parent.parent / "RGB" / input_paths[0].name
    if rgb_path.is_file():
        return WeedMapDataset._load_rgb(rgb_path), "RGB image"

    # Other sample lists may contain multispectral tiles without a blue band.
    composite = image[[1, 0, 3]].permute(1, 2, 0).numpy()
    return composite, "RGB image (R/G/NIR composite)"


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")

    dataset = WeedMapDataset(
        data_root=args.data_root,
        input_type=args.input_type,
        filter_empty=True,
        sample_list_csv=args.sample_list_csv,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"--sample-index must be between 0 and {len(dataset) - 1}; "
            f"received {args.sample_index}"
        )

    image, label = dataset[args.sample_index]
    device = select_device()
    in_channels = 3 if args.input_type == "rgb" else 5
    model = SmallUNet(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0]
    prediction = prediction.cpu().numpy()
    label = label.numpy()

    valid = label != IGNORE_INDEX
    total_valid = int(np.count_nonzero(valid))
    if total_valid == 0:
        raise ValueError("The selected sample has no valid label pixels")
    errors = valid & (prediction != label)
    total_errors = int(np.count_nonzero(errors))
    boundary = ground_truth_boundary(label)
    nearby = {radius: dilate_within_radius(boundary, radius) & valid for radius in (1, 3, 5)}
    near_errors = {radius: int(np.count_nonzero(errors & nearby[radius])) for radius in nearby}
    outside_errors = total_errors - near_errors[5]

    print(f"sample index: {args.sample_index}")
    print(f"total valid pixels: {total_valid}")
    print(f"total error pixels: {total_errors}")
    print(f"overall error rate: {total_errors / total_valid:.2%}")
    for radius in (1, 3, 5):
        share = near_errors[radius] / total_errors if total_errors else 0.0
        print(f"errors within {radius}px boundary: {near_errors[radius]} ({share:.2%} of errors)")
    outside_share = outside_errors / total_errors if total_errors else 0.0
    print(f"errors outside 5px boundary: {outside_errors} ({outside_share:.2%} of errors)")
    near_valid = int(np.count_nonzero(nearby[5]))
    outside_valid = total_valid - near_valid
    print(f"valid pixels within 5px boundary: {near_valid} ({near_valid / total_valid:.2%})")
    print(f"error rate within 5px boundary: {near_errors[5] / near_valid:.2%}" if near_valid else "error rate within 5px boundary: n/a")
    print(f"error rate outside 5px boundary: {outside_errors / outside_valid:.2%}" if outside_valid else "error rate outside 5px boundary: n/a")

    rgb, rgb_title = load_display_rgb(dataset, args.sample_index, image)
    panels = (
        (rgb, rgb_title, {}),
        (colorize_mask(label, show_ignore=True), "GroundTruth mask", {}),
        (colorize_mask(prediction), "Prediction mask", {}),
        (errors, "Error map (white = wrong)", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (boundary, "Boundary map", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (errors & nearby[5], "Error near boundary (5px)", {"cmap": "gray", "vmin": 0, "vmax": 1}),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (panel, title, kwargs) in zip(axes.flat, panels):
        axis.imshow(panel, interpolation="nearest", **kwargs)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Real WeedMap boundary errors | sample={args.sample_index}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"visualization saved to {args.output}")


if __name__ == "__main__":
    main()
