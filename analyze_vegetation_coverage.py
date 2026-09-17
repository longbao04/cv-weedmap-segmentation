"""Measure NDVI vegetation coverage on the WeedMap common samples."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from weedmap_dataset import WeedMapDataset


DEFAULT_SAMPLE_LIST = Path("splits/real_weedmap_common_samples.csv")
DEFAULT_CSV = Path("outputs/weedmap_vegetation_coverage_summary.csv")
DEFAULT_HISTOGRAM = Path("outputs/weedmap_vegetation_coverage_histogram.png")
CSV_FIELDS = (
    "sample_id",
    "split",
    "valid_pixels",
    "vegetation_pixels",
    "vegetation_coverage_ratio",
    "scene_type",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/weedmap"))
    parser.add_argument("--sample-list-csv", type=Path, default=DEFAULT_SAMPLE_LIST)
    parser.add_argument("--ndvi-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the 80/20 train/val split")
    args = parser.parse_args()
    if not math.isfinite(args.ndvi_threshold):
        parser.error("--ndvi-threshold must be finite")
    return args


def scene_type(vcr):
    if vcr < 0.20:
        return "sparse"
    if vcr > 0.30:
        return "dense"
    return "transition"


def main():
    args = parse_args()
    dataset = WeedMapDataset(
        data_root=args.data_root,
        input_type="multispectral",
        ignore_index=255,
        filter_empty=False,
        sample_list_csv=args.sample_list_csv,
    )
    with args.sample_list_csv.open(newline="", encoding="utf-8-sig") as file:
        listed_count = sum(1 for _ in csv.DictReader(file))
    if not dataset or len(dataset) != listed_count:
        raise RuntimeError(
            f"Expected {listed_count} common samples, but loaded {len(dataset)}; "
            "check missing image or label files"
        )

    train_size = int(0.8 * len(dataset))
    shuffled = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    train_indices = set(shuffled[:train_size])

    rows = []
    counts = {"sparse": 0, "transition": 0, "dense": 0}
    for index in range(len(dataset)):
        image, label = dataset[index]
        valid = label != 255
        valid_pixels = int(valid.sum().item())
        if valid_pixels == 0:
            raise ValueError(f"Sample at index {index} has no valid pixels")
        # Multispectral channel order: G, R, RE, NIR, NDVI.
        vegetation_pixels = int(((image[4] > args.ndvi_threshold) & valid).sum().item())
        vcr = vegetation_pixels / valid_pixels
        category = scene_type(vcr)
        counts[category] += 1

        input_paths, _, _ = dataset.samples[index]
        subset = input_paths[0].relative_to(args.data_root).parts[0]
        sample_id = f"{subset}_{input_paths[0].stem}"
        rows.append(
            {
                "sample_id": sample_id,
                "split": "train" if index in train_indices else "val",
                "valid_pixels": valid_pixels,
                "vegetation_pixels": vegetation_pixels,
                "vegetation_coverage_ratio": f"{vcr:.8f}",
                "scene_type": category,
            }
        )

    DEFAULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ratios = [float(row["vegetation_coverage_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=20, range=(0, 1), edgecolor="white")
    ax.axvline(0.20, color="orange", linestyle="--", label="sparse / transition")
    ax.axvline(0.30, color="green", linestyle="--", label="transition / dense")
    ax.set(xlabel="Vegetation coverage ratio", ylabel="Samples", title="WeedMap vegetation coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DEFAULT_HISTOGRAM, dpi=150)
    plt.close(fig)

    print(f"total samples: {len(rows)}")
    for category in ("sparse", "transition", "dense"):
        print(f"{category} count: {counts[category]}")
    for name, value in (
        ("mean VCR", mean(ratios)),
        ("median VCR", median(ratios)),
        ("min VCR", min(ratios)),
        ("max VCR", max(ratios)),
    ):
        print(f"{name}: {value:.4f}")
    print(f"CSV saved to: {DEFAULT_CSV}")
    print(f"Histogram saved to: {DEFAULT_HISTOGRAM}")


if __name__ == "__main__":
    main()
