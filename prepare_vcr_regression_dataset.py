"""Build the four-band sample manifest for VCR regression experiments."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs/weedmap_vegetation_coverage_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs/vcr_regression_samples.csv"
DEFAULT_DATA_ROOT = ROOT / "data/weedmap"
BANDS = ("G", "R", "RE", "NIR")
OUTPUT_FIELDS = (
    "sample_id",
    "split",
    "subset",
    "frame_id",
    "G_path",
    "R_path",
    "RE_path",
    "NIR_path",
    "VCR",
    "scene_type",
)
REQUIRED_INPUT_FIELDS = {
    "sample_id",
    "split",
    "vegetation_coverage_ratio",
    "scene_type",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def expected_scene_type(vcr):
    if vcr < 0.20:
        return "sparse"
    if vcr > 0.30:
        return "dense"
    return "transition"


def parse_sample_id(sample_id):
    if "_frame" not in sample_id:
        raise ValueError(f"Invalid sample_id (missing _frame): {sample_id!r}")
    subset, frame_suffix = sample_id.rsplit("_frame", 1)
    frame_id = f"frame{frame_suffix}"
    if not subset or not frame_suffix or "_" not in subset:
        raise ValueError(f"Invalid sample_id: {sample_id!r}")
    subset_id = subset.rsplit("_", 1)[-1]
    if not subset_id.isdigit():
        raise ValueError(f"Invalid subset in sample_id: {sample_id!r}")
    return subset, subset_id, frame_id


def manifest_path(path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def prepare_rows(input_path, data_root):
    if not input_path.is_file():
        raise FileNotFoundError(f"VCR summary not found: {input_path}")

    output_rows = []
    seen = set()
    missing_paths = []
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not REQUIRED_INPUT_FIELDS.issubset(
            reader.fieldnames
        ):
            missing_fields = sorted(
                REQUIRED_INPUT_FIELDS.difference(reader.fieldnames or ())
            )
            raise ValueError(
                "VCR summary is missing required fields: "
                + ", ".join(missing_fields)
            )

        for line_number, source in enumerate(reader, start=2):
            sample_id = (source.get("sample_id") or "").strip()
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id on line {line_number}: {sample_id}")
            seen.add(sample_id)

            split = (source.get("split") or "").strip()
            if split not in {"train", "val"}:
                raise ValueError(
                    f"Invalid split on line {line_number}: {split!r}"
                )

            try:
                vcr = float(source["vegetation_coverage_ratio"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid VCR on line {line_number}") from error
            if not math.isfinite(vcr) or not 0.0 <= vcr <= 1.0:
                raise ValueError(f"VCR outside [0, 1] on line {line_number}: {vcr}")

            scene_type = (source.get("scene_type") or "").strip()
            expected = expected_scene_type(vcr)
            if scene_type != expected:
                raise ValueError(
                    f"Scene/VCR mismatch on line {line_number}: "
                    f"scene_type={scene_type!r}, expected={expected!r}"
                )

            subset, subset_id, frame_id = parse_sample_id(sample_id)
            tile_root = data_root / subset / subset_id / "tile"
            band_paths = {
                band: tile_root / band / f"{frame_id}.png" for band in BANDS
            }
            missing_paths.extend(path for path in band_paths.values() if not path.is_file())

            output_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "subset": subset,
                    "frame_id": frame_id,
                    **{
                        f"{band}_path": manifest_path(path)
                        for band, path in band_paths.items()
                    },
                    "VCR": f"{vcr:.8f}",
                    "scene_type": scene_type,
                }
            )

    if not output_rows:
        raise ValueError(f"VCR summary contains no samples: {input_path}")
    if missing_paths:
        preview = "\n".join(f"- {path}" for path in missing_paths[:10])
        suffix = "" if len(missing_paths) <= 10 else f"\n... and {len(missing_paths) - 10} more"
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} required four-band files:\n{preview}{suffix}"
        )
    return output_rows


def main():
    args = parse_args()
    rows = prepare_rows(args.input.resolve(), args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(row["split"] for row in rows)
    scene_counts = Counter(row["scene_type"] for row in rows)
    subset_counts = Counter(row["subset"] for row in rows)
    print(f"samples: {len(rows)}")
    print(
        "splits: "
        + ", ".join(f"{name}={split_counts[name]}" for name in ("train", "val"))
    )
    print(
        "scenes: "
        + ", ".join(
            f"{name}={scene_counts[name]}"
            for name in ("sparse", "transition", "dense")
        )
    )
    print(f"subsets: {len(subset_counts)}")
    print("input bands: G, R, RE, NIR (NDVI excluded)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
