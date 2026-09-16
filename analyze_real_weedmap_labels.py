"""Analyze label distributions and mappings in local WeedMap Tiles data."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_DATA_ROOT = Path("data/weedmap")
DEFAULT_OUTPUT = Path("outputs/real_weedmap_label_summary.csv")
DEFAULT_SUBSETS = (
    "RedEdge_000",
    "RedEdge_001",
    "RedEdge_002",
    "RedEdge_003",
    "RedEdge_004",
    "Sequoia_005",
    "Sequoia_006",
    "Sequoia_007",
)
COLOR_CLASSES = {
    "black": ((0, 0, 0), 0),
    "green": ((0, 255, 0), 10000),
    "red": ((255, 0, 0), 2),
}
CSV_FIELDS = (
    "sensor",
    "subset_id",
    "sample_id",
    "background_pixels",
    "crop_pixels",
    "weed_pixels",
    "invalid_pixels",
    "total_pixels",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析本地真实 WeedMap Tiles 的标签分布和像素级标签映射。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"WeedMap 数据根目录（默认：{DEFAULT_DATA_ROOT}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"逐样本 CSV 输出路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="要分析的子集目录名，默认分析 8 个真实 Tiles 子集。",
    )
    return parser.parse_args()


def sample_id_from_color(path, subset_id):
    prefix = f"{subset_id}_"
    suffix = "_GroundTruth_color"
    if not path.stem.startswith(prefix) or not path.stem.endswith(suffix):
        raise ValueError(f"无法从彩色标签文件名解析 sample_id：{path.name}")
    return path.stem[len(prefix):-len(suffix)]


def discover_samples(subset_root, subset_id):
    groundtruth_dir = subset_root / "groundtruth"
    mask_dir = subset_root / "mask"
    if not groundtruth_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"子集缺少 groundtruth 或 mask 目录：{subset_root.resolve()}"
        )

    samples = []
    for color_path in sorted(groundtruth_dir.glob("*_GroundTruth_color.png")):
        sample_id = sample_id_from_color(color_path, subset_id)
        imap_path = groundtruth_dir / f"{subset_id}_{sample_id}_GroundTruth_iMap.png"
        mask_path = mask_dir / f"{sample_id}.png"
        missing = [path for path in (imap_path, mask_path) if not path.is_file()]
        if missing:
            names = ", ".join(str(path.resolve()) for path in missing)
            raise FileNotFoundError(f"样本 {sample_id} 缺少配对文件：{names}")
        samples.append((sample_id, color_path, imap_path, mask_path))

    if not samples:
        raise FileNotFoundError(f"未找到彩色标签：{groundtruth_dir.resolve()}")

    color_ids = {sample[0] for sample in samples}
    imap_ids = {
        path.stem[len(f"{subset_id}_"):-len("_GroundTruth_iMap")]
        for path in groundtruth_dir.glob("*_GroundTruth_iMap.png")
        if path.stem.startswith(f"{subset_id}_")
    }
    mask_ids = {path.stem for path in mask_dir.glob("*.png")}
    if color_ids != imap_ids or color_ids != mask_ids:
        raise ValueError(
            f"{subset_root.name} 的 color/iMap/mask 样本集合不一致："
            f"color={len(color_ids)}, iMap={len(imap_ids)}, mask={len(mask_ids)}"
        )
    return samples


def load_label_images(color_path, imap_path, mask_path):
    try:
        with Image.open(color_path) as image:
            color = np.asarray(image.convert("RGB"))
        with Image.open(imap_path) as image:
            imap = np.asarray(image).copy()
        with Image.open(mask_path) as image:
            mask = np.asarray(image).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise RuntimeError(f"无法读取标签图像：{error}") from error

    if imap.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"iMap 和 mask 应为二维单通道图，实际形状：iMap={imap.shape}, mask={mask.shape}"
        )
    if color.shape[:2] != imap.shape or color.shape[:2] != mask.shape:
        raise ValueError(
            f"配对图像尺寸不一致：color={color.shape[:2]}, "
            f"iMap={imap.shape}, mask={mask.shape}"
        )
    return color, imap, mask


def count_selected_values(array, selected_values):
    values, counts = np.unique(array, return_counts=True)
    all_counts = {int(value): int(count) for value, count in zip(values, counts)}
    selected = {value: all_counts.get(value, 0) for value in selected_values}
    selected["other"] = int(array.size - sum(selected.values()))
    return selected


def color_masks_and_counts(color):
    masks = {}
    for name, (rgb, _) in COLOR_CLASSES.items():
        masks[name] = np.all(color == np.asarray(rgb, dtype=color.dtype), axis=2)
    known = masks["black"] | masks["green"] | masks["red"]
    counts = {name: int(pixel_mask.sum()) for name, pixel_mask in masks.items()}
    counts["other"] = int(color.shape[0] * color.shape[1] - known.sum())
    return masks, counts


def analyze_subset(data_root, subset_name):
    try:
        sensor, subset_id = subset_name.rsplit("_", 1)
    except ValueError as error:
        raise ValueError(f"子集名称应采用 Sensor_ID 格式：{subset_name}") from error
    subset_root = data_root / subset_name / subset_id
    if not subset_root.is_dir():
        raise FileNotFoundError(f"子集路径不存在：{subset_root.resolve()}")

    samples = discover_samples(subset_root, subset_id)
    totals = {
        "color": Counter(),
        "imap": Counter(),
        "mask": Counter(),
    }
    dimensions = Counter()
    mapping_counts = {name: Counter() for name in COLOR_CLASSES}
    mapping_checks = {
        name: {"passed": 0, "failed": [], "absent": 0}
        for name in COLOR_CLASSES
    }
    rows = []

    for sample_id, color_path, imap_path, mask_path in samples:
        color, imap, mask = load_label_images(color_path, imap_path, mask_path)
        height, width = color.shape[:2]
        dimensions[(height, width)] += 1

        color_masks, color_counts = color_masks_and_counts(color)
        imap_counts = count_selected_values(imap, (0, 2, 10000))
        mask_counts = count_selected_values(mask, (0, 255))
        totals["color"].update(color_counts)
        totals["imap"].update(imap_counts)
        totals["mask"].update(mask_counts)

        for color_name, pixel_mask in color_masks.items():
            values, counts = np.unique(imap[pixel_mask], return_counts=True)
            sample_mapping = {
                int(value): int(count) for value, count in zip(values, counts)
            }
            mapping_counts[color_name].update(sample_mapping)
            color_total = color_counts[color_name]
            expected_value = COLOR_CLASSES[color_name][1]
            if color_total == 0:
                mapping_checks[color_name]["absent"] += 1
            elif sample_mapping.get(expected_value, 0) > color_total / 2:
                mapping_checks[color_name]["passed"] += 1
            else:
                mapping_checks[color_name]["failed"].append(sample_id)

        rows.append({
            "sensor": sensor,
            "subset_id": subset_id,
            "sample_id": sample_id,
            "background_pixels": color_counts["black"],
            "crop_pixels": color_counts["green"],
            "weed_pixels": color_counts["red"],
            "invalid_pixels": mask_counts[255],
            "total_pixels": height * width,
        })

    return rows, totals, dimensions, mapping_counts, mapping_checks


def format_dimensions(dimensions):
    return ", ".join(
        f"{height}x{width}（{count} 个）"
        for (height, width), count in sorted(dimensions.items())
    )


def format_mapping(counter):
    total = sum(counter.values())
    if total == 0:
        return "无该颜色像素"
    preferred = (0, 10000, 2)
    parts = [f"{value}: {counter.get(value, 0)}" for value in preferred]
    other = total - sum(counter.get(value, 0) for value in preferred)
    parts.append(f"其他值: {other}")
    return ", ".join(parts)


def print_subset_summary(
    subset_name, rows, totals, dimensions, mapping_counts, mapping_checks
):
    print(f"\n=== {subset_name} ===")
    print(f"样本数量：{len(rows)}")
    print(f"图像尺寸（高x宽）：{format_dimensions(dimensions)}")
    print("GroundTruth_color 像素统计：")
    print(f"  black pixels: {totals['color']['black']}")
    print(f"  green crop pixels: {totals['color']['green']}")
    print(f"  red weed pixels: {totals['color']['red']}")
    print(f"  other pixels: {totals['color']['other']}")
    print("GroundTruth_iMap 像素统计：")
    print(f"  value=0: {totals['imap'][0]}")
    print(f"  value=2: {totals['imap'][2]}")
    print(f"  value=10000: {totals['imap'][10000]}")
    print(f"  其他值: {totals['imap']['other']}")
    print("mask 像素统计：")
    print(f"  value=0: {totals['mask'][0]}")
    print(f"  value=255: {totals['mask'][255]}")
    if totals["mask"]["other"]:
        print(f"  其他值: {totals['mask']['other']}")

    print("像素级 color -> iMap 验证：")
    print(f"  已检查全部 {len(rows)} 个 color/iMap 配对文件。")
    for color_name, (_, expected_value) in COLOR_CLASSES.items():
        counter = mapping_counts[color_name]
        checks = mapping_checks[color_name]
        total = sum(counter.values())
        matches = counter.get(expected_value, 0)
        percentage = 100.0 * matches / total if total else 0.0
        status = "通过" if total and matches > total / 2 else "无像素" if not total else "警告"
        print(
            f"  {color_name} -> {expected_value}: {matches}/{total} "
            f"({percentage:.4f}%, {status}); 分布 [{format_mapping(counter)}]"
        )
        print(
            f"    逐文件：通过={checks['passed']}，失败={len(checks['failed'])}，"
            f"无 {color_name} 像素={checks['absent']}"
        )
        if checks["failed"]:
            print(f"    失败样本（前 10 个）：{', '.join(checks['failed'][:10])}")

    crop_max = max(rows, key=lambda row: row["crop_pixels"])
    weed_max = max(rows, key=lambda row: row["weed_pixels"])
    both = [row for row in rows if row["crop_pixels"] > 0 and row["weed_pixels"] > 0]
    print(
        f"crop 像素最多的样本：{crop_max['sample_id']} "
        f"({crop_max['crop_pixels']} pixels)"
    )
    print(
        f"weed 像素最多的样本：{weed_max['sample_id']} "
        f"({weed_max['weed_pixels']} pixels)"
    )
    print(f"同时包含 crop 和 weed 的样本数量：{len(both)}")
    print("同时包含 crop 和 weed 的前 10 个样本：")
    if both:
        for row in both[:10]:
            print(
                f"  {row['sample_id']} "
                f"(crop={row['crop_pixels']}, weed={row['weed_pixels']})"
            )
    else:
        print("  无")


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_verified_mapping():
    print("\n=== 本地验证得到的标签映射 ===")
    print("GroundTruth_color:")
    print("black = background, green = crop, red = weed")
    print("\nGroundTruth_iMap in current Tiles:")
    print("0 = background, 10000 = crop, 2 = weed")
    print("\nmask:")
    print("0 = valid area, 255 = invalid/no-data area")


def main():
    args = parse_args()
    data_root = args.data_root.expanduser()
    output = args.output.expanduser()
    if not data_root.is_dir():
        raise SystemExit(f"错误：数据根目录不存在：{data_root.resolve()}")

    all_rows = []
    try:
        for subset_name in args.subsets:
            rows, totals, dimensions, mapping_counts, mapping_checks = analyze_subset(
                data_root, subset_name
            )
            print_subset_summary(
                subset_name, rows, totals, dimensions, mapping_counts, mapping_checks
            )
            all_rows.extend(rows)
        write_csv(all_rows, output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"错误：{error}") from error

    print_verified_mapping()
    print(f"\nCSV 已保存：{output.resolve()}（{len(all_rows)} 个样本）")


if __name__ == "__main__":
    main()
