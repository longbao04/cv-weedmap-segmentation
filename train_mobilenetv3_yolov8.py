"""Train the MobileNetV3-Small YOLOv8 detector on the WeedMap r010 dataset."""

import argparse
from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def default_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=ROOT / "configs/mobilenetv3_small_yolov8.yaml",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/yolo_weedmap_detect_r010/data.yaml",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=ROOT / "models/mobilenetv3_small_yolov8_imagenet_unit_range_init.pt",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--freeze",
        type=int,
        default=1,
        help="Freeze the first N top-level layers; use 5 for normalizer+backbone, 1 for normalizer only.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=ROOT / "runs/yolo_mobilenetv3_compare",
    )
    parser.add_argument("--name", default="mobilenetv3_small_5epochs")
    parser.add_argument(
        "--imagenet-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--rebuild-init",
        action="store_true",
        help="Rebuild the initialization checkpoint even when it already exists.",
    )
    args = parser.parse_args()
    if (args.epochs < 1 or args.imgsz < 32 or args.batch < 1
            or args.workers < 0 or args.freeze < 1):
        parser.error("epochs、imgsz、batch 必须为正数，workers/freeze 不能小于 0")
    return args


def create_initial_checkpoint(args):
    model = YOLO(str(args.model_config))
    if args.imagenet_pretrained:
        source = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        ).features
        configure_input_normalizer(model.model.model[0])
        target = model.model.model[1].m
        target.load_state_dict(source.state_dict(), strict=True)
        verify_input_adaptation(source, model.model.model[0], target)
        print("Loaded ImageNet weights with an explicit frozen input normalizer")
    else:
        print("Using random MobileNetV3-Small backbone initialization")
    args.init_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.init_checkpoint)
    print(f"Initial checkpoint saved to {args.init_checkpoint}")


def configure_input_normalizer(normalizer):
    """Configure a grouped 1x1 convolution as ImageNet normalization."""
    mean = torch.tensor((0.485, 0.456, 0.406))
    std = torch.tensor((0.229, 0.224, 0.225))
    with torch.no_grad():
        normalizer.weight.copy_((1.0 / std).view(3, 1, 1, 1))
        normalizer.bias.copy_(-mean / std)


def verify_input_adaptation(source, normalizer, target):
    """Check that the model normalizer matches torchvision preprocessing."""
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    source.eval()
    target.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = source((image - mean) / std)
        actual = target(normalizer(image))
    max_error = (expected - actual).abs().max().item()
    if max_error > 1e-4:
        raise RuntimeError(f"Input-normalization check failed: {max_error}")
    print(f"Input-normalization max error: {max_error:.2e}")


def main():
    args = parse_args()
    if args.rebuild_init or not args.init_checkpoint.is_file():
        create_initial_checkpoint(args)
    else:
        print(f"Reusing initial checkpoint: {args.init_checkpoint}")

    model = YOLO(str(args.init_checkpoint))
    train_args = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        plots=True,
    )
    train_args["freeze"] = args.freeze
    model.train(**train_args)


if __name__ == "__main__":
    main()
