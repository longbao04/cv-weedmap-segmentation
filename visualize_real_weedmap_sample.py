"""Visualize one real WeedMap RedEdge sample without training a model."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image, UnidentifiedImageError


DEFAULT_DATA_ROOT = Path("data/weedmap/RedEdge_004/004")
DEFAULT_SAMPLE_ID = "frame0070"
DEFAULT_OUTPUT = Path("outputs/real_weedmap_sample_frame0070.png")


def build_paths(data_root, sample_id):
    """Return all files required for a sample, using WeedMap's naming rules."""
    tile_name = f"{sample_id}.png"
    sequence_id = data_root.name
    return {
        "RGB": data_root / "tile" / "RGB" / tile_name,
        "NDVI": data_root / "tile" / "NDVI" / tile_name,
        "NIR": data_root / "tile" / "NIR" / tile_name,
        "RedEdge": data_root / "tile" / "RE" / tile_name,
        "R": data_root / "tile" / "R" / tile_name,
        "G": data_root / "tile" / "G" / tile_name,
        "B": data_root / "tile" / "B" / tile_name,
        "valid mask": data_root / "mask" / tile_name,
        "GroundTruth color": data_root / "groundtruth" / f"{sequence_id}_{sample_id}_GroundTruth_color.png",
        "GroundTruth iMap": data_root / "groundtruth" / f"{sequence_id}_{sample_id}_GroundTruth_iMap.png",
    }


def load_images(paths):
    """Load images with Pillow so 16-bit integer class labels stay unchanged."""
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path.resolve()}" for path in missing)
        raise FileNotFoundError(f"缺少以下 WeedMap 样本文件：\n{formatted}")

    images = {}
    for name, path in paths.items():
        try:
            with Image.open(path) as image:
                images[name] = np.asarray(image).copy()
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"无法读取图像 {path.resolve()}：{error}") from error
    return images


def validate_shapes(images):
    """Ensure all bands, labels, and masks are spatially aligned."""
    expected_shape = images["RGB"].shape[:2]
    mismatched = {
        name: image.shape[:2]
        for name, image in images.items()
        if image.shape[:2] != expected_shape
    }
    if mismatched:
        details = ", ".join(f"{name}={shape}" for name, shape in mismatched.items())
        raise ValueError(f"图像尺寸不一致；RGB={expected_shape}，不匹配的文件：{details}")
    if images["RGB"].ndim != 3 or images["RGB"].shape[2] < 3:
        raise ValueError(f"RGB 图像应至少有 3 个通道，实际形状为 {images['RGB'].shape}")
    if images["GroundTruth iMap"].ndim != 2:
        raise ValueError(
            f"GroundTruth iMap 应为二维单通道标签图，实际形状为 {images['GroundTruth iMap'].shape}"
        )


def as_display_rgb(image):
    """Convert an integer or floating-point RGB image to the [0, 1] range."""
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.integer):
        return rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
    return np.clip(rgb.astype(np.float32), 0.0, 1.0)


def print_imap_counts(imap):
    values, counts = np.unique(imap, return_counts=True)
    class_names = {0: "background", 2: "weed", 10000: "crop"}
    print("GT_iMap unique values and counts:")
    for value, count in zip(values, counts):
        integer_value = int(value)
        label = class_names.get(integer_value, "unknown")
        print(f"  {integer_value} ({label}): {int(count)}")


def print_mask_values(mask):
    values = np.unique(mask)
    print(f"Mask unique values: {values.tolist()}")
    print("  mask=0: valid area")
    print("  mask=255: invalid/no-data area")


def make_figure(images, sample_id):
    rgb = as_display_rgb(images["RGB"])
    imap = images["GroundTruth iMap"]
    mask = images["valid mask"]
    valid_mask = mask == 0

    weed = (imap == 2) & valid_mask
    overlay = rgb.copy()
    weed_color = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    overlay[weed] = 0.55 * rgb[weed] + 0.45 * weed_color

    # Map WeedMap's sparse iMap values to compact display indices.
    imap_display = np.full(imap.shape, 3, dtype=np.uint8)
    imap_display[imap == 0] = 0
    imap_display[imap == 10000] = 1
    imap_display[imap == 2] = 2
    label_colors = ("#70543a", "#28b946", "#ef3b32", "#8e44ad")
    label_cmap = ListedColormap(label_colors)
    label_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), label_cmap.N)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    panels = (
        (rgb, "RGB", {}),
        (images["NDVI"], "NDVI", {"cmap": "viridis"}),
        (images["NIR"], "NIR", {"cmap": "gray"}),
        (images["RedEdge"], "RedEdge", {"cmap": "gray"}),
        (images["GroundTruth color"], "GroundTruth color", {}),
        (imap_display, "GroundTruth iMap", {"cmap": label_cmap, "norm": label_norm}),
        (valid_mask, "valid area mask", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (overlay, "overlay (weed in red)", {}),
    )
    for axis, (image, title, options) in zip(axes.flat, panels):
        axis.imshow(image, interpolation="nearest", **options)
        axis.set_title(title)
        axis.axis("off")

    legend_items = (
        (label_colors[0], "0: background"),
        (label_colors[1], "10000: crop"),
        (label_colors[2], "2: weed"),
        (label_colors[3], "other value"),
    )
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in legend_items],
        loc="lower center",
        ncol=5,
    )
    fig.suptitle(f"Real WeedMap RedEdge_004 sample: {sample_id}", fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="可视化真实 WeedMap RedEdge 数据样本。")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"样本数据根目录（默认：{DEFAULT_DATA_ROOT}）")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID,
                        help=f"样本编号，不含扩展名（默认：{DEFAULT_SAMPLE_ID}）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"输出图片路径（默认：{DEFAULT_OUTPUT}）")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.expanduser()
    output = args.output.expanduser()
    try:
        images = load_images(build_paths(data_root, args.sample_id))
        validate_shapes(images)
        print_imap_counts(images["GroundTruth iMap"])
        print_mask_values(images["valid mask"])
        figure = make_figure(images, args.sample_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(figure)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"错误：{error}") from error
    print(f"可视化已保存：{output.resolve()}")


if __name__ == "__main__":
    main()
