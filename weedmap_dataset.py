"""PyTorch Dataset for the local WeedMap Tiles data."""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - only used in environments without torch
    raise ImportError(
        "weedmap_dataset.py requires PyTorch. Please install torch before using "
        "WeedMapDataset."
    ) from exc


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
MULTISPECTRAL_CHANNELS = ("G", "R", "RE", "NIR", "NDVI")


class WeedMapDataset(Dataset):
    """Read RGB or five-channel multispectral WeedMap tiles and labels.

    Invalid/no-data pixels are returned as ``ignore_index`` in the label.
    Samples missing any required input, color ground truth, or valid-area mask
    are omitted while the dataset is indexed. By default, all-black images,
    all-ignore labels, and samples below the valid/foreground pixel thresholds
    are also omitted.
    """

    def __init__(
        self,
        data_root="data/weedmap",
        subsets=None,
        input_type="rgb",
        ignore_index=255,
        filter_empty=True,
        min_valid_pixels=1000,
        min_foreground_pixels=1,
    ):
        if input_type not in ("rgb", "multispectral"):
            raise ValueError("input_type must be 'rgb' or 'multispectral'")
        if not isinstance(ignore_index, int):
            raise TypeError("ignore_index must be an integer")
        if not isinstance(filter_empty, bool):
            raise TypeError("filter_empty must be a boolean")
        for name, value in (
            ("min_valid_pixels", min_valid_pixels),
            ("min_foreground_pixels", min_foreground_pixels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        self.data_root = Path(data_root)
        self.subsets = tuple(DEFAULT_SUBSETS if subsets is None else subsets)
        self.input_type = input_type
        self.ignore_index = ignore_index
        self.filter_empty = filter_empty
        self.min_valid_pixels = min_valid_pixels
        self.min_foreground_pixels = min_foreground_pixels
        self.samples = []
        skipped_missing = 0
        skipped_empty = 0

        for subset_name in self.subsets:
            subset_id = subset_name.rsplit("_", 1)[-1]
            subset_root = self.data_root / subset_name / subset_id
            groundtruth_dir = subset_root / "groundtruth"
            mask_dir = subset_root / "mask"
            tile_dir = subset_root / "tile"

            sample_ids = self._discover_sample_ids(
                subset_id, groundtruth_dir, mask_dir, tile_dir
            )
            for sample_id in sorted(sample_ids):
                color_path = (
                    groundtruth_dir
                    / f"{subset_id}_{sample_id}_GroundTruth_color.png"
                )
                mask_path = mask_dir / f"{sample_id}.png"
                if input_type == "rgb":
                    input_paths = (tile_dir / "RGB" / f"{sample_id}.png",)
                else:
                    input_paths = tuple(
                        tile_dir / channel / f"{sample_id}.png"
                        for channel in MULTISPECTRAL_CHANNELS
                    )

                required_paths = (*input_paths, color_path, mask_path)
                if not all(path.is_file() for path in required_paths):
                    skipped_missing += 1
                    continue

                sample = (input_paths, color_path, mask_path)
                if filter_empty and not self._is_valid_sample(sample):
                    skipped_empty += 1
                    continue
                self.samples.append(sample)

        print(
            f"WeedMapDataset: input_type={input_type}, "
            f"loaded samples={len(self.samples)}, "
            f"skipped missing samples={skipped_missing}, "
            f"skipped empty/invalid samples={skipped_empty}"
        )

    @staticmethod
    def _discover_sample_ids(subset_id, groundtruth_dir, mask_dir, tile_dir):
        """Return the union of IDs so missing labels are counted as skipped."""
        sample_ids = set()
        color_prefix = f"{subset_id}_"
        color_suffix = "_GroundTruth_color"

        if groundtruth_dir.is_dir():
            for path in groundtruth_dir.glob("*_GroundTruth_color.png"):
                stem = path.stem
                if stem.startswith(color_prefix) and stem.endswith(color_suffix):
                    sample_ids.add(stem[len(color_prefix) : -len(color_suffix)])
        if mask_dir.is_dir():
            sample_ids.update(path.stem for path in mask_dir.glob("frame*.png"))
        if tile_dir.is_dir():
            for channel_dir in tile_dir.iterdir():
                if channel_dir.is_dir():
                    sample_ids.update(
                        path.stem for path in channel_dir.glob("frame*.png")
                    )
        return sample_ids

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _normalize_image(array):
        if np.issubdtype(array.dtype, np.integer):
            if np.issubdtype(array.dtype, np.unsignedinteger):
                scale = np.iinfo(array.dtype).max
            else:
                maximum = int(array.max()) if array.size else 0
                if maximum <= 255:
                    scale = 255
                elif maximum <= 65535:
                    scale = 65535
                else:
                    scale = maximum
            return array.astype(np.float32) / max(scale, 1)

        array = array.astype(np.float32)
        maximum = float(array.max()) if array.size else 0.0
        if maximum > 1.0:
            array /= maximum
        return np.clip(array, 0.0, 1.0)

    @classmethod
    def _load_rgb(cls, path):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        return cls._normalize_image(array)

    @classmethod
    def _load_band(cls, path):
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Expected a single-channel image: {path}")
        return cls._normalize_image(array)

    def _load_image(self, input_paths):
        if self.input_type == "rgb":
            return self._load_rgb(input_paths[0]).transpose(2, 0, 1)

        bands = [self._load_band(path) for path in input_paths]
        shapes = {band.shape for band in bands}
        if len(shapes) != 1:
            raise ValueError(
                "Multispectral band dimensions do not match: "
                + ", ".join(str(path) for path in input_paths)
            )
        return np.stack(bands, axis=0)

    def _load_label(self, color_path, mask_path, image_shape):
        with Image.open(color_path) as color_image:
            color = np.asarray(color_image.convert("RGB"))
        with Image.open(mask_path) as valid_image:
            valid_mask = np.asarray(valid_image)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[..., 0]

        height, width = image_shape
        if color.shape[:2] != (height, width) or valid_mask.shape != (height, width):
            raise ValueError(
                "Input and label dimensions do not match for "
                f"{color_path}: image={(height, width)}, "
                f"color={color.shape[:2]}, mask={valid_mask.shape}"
            )

        label = np.full((height, width), self.ignore_index, dtype=np.int64)
        label[np.all(color == (0, 0, 0), axis=-1)] = 0
        label[np.all(color == (0, 255, 0), axis=-1)] = 1
        label[np.all(color == (255, 0, 0), axis=-1)] = 2
        label[valid_mask == 255] = self.ignore_index
        return label

    def _is_valid_sample(self, sample):
        input_paths, color_path, mask_path = sample
        try:
            image = self._load_image(input_paths)
            label = self._load_label(color_path, mask_path, image.shape[1:])
        except (OSError, ValueError):
            return False

        valid_pixels = np.count_nonzero(label != self.ignore_index)
        foreground_pixels = np.count_nonzero((label == 1) | (label == 2))
        return (
            image.size > 0
            and float(image.max()) > 0.0
            and valid_pixels > 0
            and valid_pixels >= self.min_valid_pixels
            and foreground_pixels >= self.min_foreground_pixels
        )

    def __getitem__(self, index):
        input_paths, color_path, mask_path = self.samples[index]
        image = self._load_image(input_paths)
        label = self._load_label(color_path, mask_path, image.shape[1:])

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        label_tensor = torch.from_numpy(label).long()
        return image_tensor, label_tensor


def _print_dataset_test(input_type):
    print(f"\n=== {input_type} dataset test ===")
    dataset = WeedMapDataset(input_type=input_type)
    print(f"Dataset samples: {len(dataset)}")
    if len(dataset) == 0:
        print("No complete samples were found.")
        return

    image, label = dataset[0]
    print(
        "First image: "
        f"shape={tuple(image.shape)}, dtype={image.dtype}, "
        f"min={image.min().item():.6f}, max={image.max().item():.6f}"
    )
    print(
        "First label: "
        f"shape={tuple(label.shape)}, dtype={label.dtype}, "
        f"unique values={torch.unique(label).tolist()}"
    )
    pixel_counts = {
        "background": int((label == 0).sum().item()),
        "crop": int((label == 1).sum().item()),
        "weed": int((label == 2).sum().item()),
        "ignore": int((label == dataset.ignore_index).sum().item()),
    }
    print(
        "First label pixel counts: "
        + ", ".join(f"{name}={count}" for name, count in pixel_counts.items())
    )
    if image.max().item() <= 0:
        raise RuntimeError("All-black image sample was not filtered")
    if torch.all(label == dataset.ignore_index):
        raise RuntimeError("All-ignore label sample was not filtered")
    print(f"{input_type} dataset test: PASS")


if __name__ == "__main__":
    _print_dataset_test("rgb")
    _print_dataset_test("multispectral")
