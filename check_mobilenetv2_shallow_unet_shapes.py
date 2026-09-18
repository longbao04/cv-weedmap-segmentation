"""Check the shallow MobileNetV2-style U-Net tensor shapes without training."""

import argparse

import torch

from unet import MobileNetV2ShallowUNet


EXPECTED_SHAPES = {
    "input": (1, 5, 360, 480),
    "e1": (1, 16, 360, 480),
    "B3": (1, 24, 180, 240),
    "e2": (1, 32, 180, 240),
    "B6": (1, 32, 90, 120),
    "center": (1, 64, 90, 120),
    "up2": (1, 32, 180, 240),
    "concat2": (1, 64, 180, 240),
    "d2": (1, 32, 180, 240),
    "up1": (1, 16, 360, 480),
    "concat1": (1, 32, 360, 480),
    "d1": (1, 16, 360, 480),
    "logits": (1, 3, 360, 480),
}
EXPECTED_RESIDUALS = {
    "B1": False,
    "B2": False,
    "B3": True,
    "B4": False,
    "B5": True,
    "B6": True,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation", choices=("relu", "leaky_relu", "gelu"), default="relu"
    )
    args = parser.parse_args()
    model = MobileNetV2ShallowUNet(
        in_channels=5, num_classes=3, activation=args.activation
    ).eval()
    print(f"activation: {args.activation}")
    for name, expected in EXPECTED_RESIDUALS.items():
        actual = getattr(model, name.lower()).use_residual
        print(f"{name} residual add: {actual}")
        if actual != expected:
            raise AssertionError(
                f"{name} residual add: expected {expected}, got {actual}"
            )

    dummy = torch.randn(1, 5, 360, 480)
    observed = {"input": tuple(dummy.shape)}
    tensor_ids = {}
    hooks = []

    def record_output(name):
        def hook(_module, _inputs, output):
            observed[name] = tuple(output.shape)
            if name == "B3":
                tensor_ids["B3"] = id(output)

        return hook

    def record_input(name):
        def hook(_module, inputs):
            observed[name] = tuple(inputs[0].shape)
            if name == "B4_input":
                tensor_ids["B4_input"] = id(inputs[0])

        return hook

    for name, module in (
        ("e1", model.c2),
        ("B3", model.b3),
        ("e2", model.e2_adapter),
        ("B6", model.b6),
        ("center", model.center_adapter),
        ("up2", model.up2),
        ("d2", model.dec2),
        ("up1", model.up1),
        ("d1", model.dec1),
        ("logits", model.head),
    ):
        hooks.append(module.register_forward_hook(record_output(name)))
    hooks.append(model.dec2.register_forward_pre_hook(record_input("concat2")))
    hooks.append(model.dec1.register_forward_pre_hook(record_input("concat1")))
    hooks.append(model.b4.register_forward_pre_hook(record_input("B4_input")))

    try:
        with torch.inference_mode():
            logits = model(dummy)
    finally:
        for hook in hooks:
            hook.remove()

    for name, expected in EXPECTED_SHAPES.items():
        actual = observed.get(name)
        print(f"{name}: {actual}")
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")

    if observed.get("B4_input") != observed["B3"]:
        raise AssertionError(
            f"B4 input shape: expected {observed['B3']}, "
            f"got {observed.get('B4_input')}"
        )
    if tensor_ids.get("B4_input") != tensor_ids.get("B3"):
        raise AssertionError("B4 must receive the original B3 output tensor")
    print("B4 input: original B3 output tensor")

    if tuple(logits.shape) != (1, 3, 360, 480):
        raise AssertionError(f"Unexpected final output shape: {tuple(logits.shape)}")
    print(f"Final output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
