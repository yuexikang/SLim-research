import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


_VIRIDIS_ANCHORS = np.asarray(
    [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ],
    dtype=np.float32,
)


def _robust_unit(array, lower=0.01, upper=0.99):
    array = np.nan_to_num(np.asarray(array, dtype=np.float32))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    low, high = np.quantile(finite, [lower, upper])
    if high <= low + 1e-8:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _colorize(unit):
    unit = np.clip(unit, 0.0, 1.0)
    position = unit * (_VIRIDIS_ANCHORS.shape[0] - 1)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, _VIRIDIS_ANCHORS.shape[0] - 1)
    fraction = (position - left)[..., None]
    rgb = _VIRIDIS_ANCHORS[left] * (1.0 - fraction) + _VIRIDIS_ANCHORS[right] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _heatmap(tensor):
    array = tensor.detach().float().cpu().numpy()
    return _colorize(_robust_unit(array))


def _grayscale(tensor):
    array = tensor.detach().float().cpu().numpy()
    unit = _robust_unit(array, lower=0.0, upper=1.0)
    gray = np.clip(unit * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _feature_activity(feature):
    return feature.detach().float().std(dim=0, unbiased=False)


def _feature_norm(feature):
    return torch.linalg.vector_norm(feature.detach().float(), dim=0)


def _orientation_rgb(orientation, reliability):
    orientation = orientation.detach().float().cpu().numpy()
    reliability = reliability.detach().float().cpu().numpy()
    x, y = orientation[0], orientation[1]
    red = 0.5 + 0.5 * x
    green = 0.5 + 0.5 * (-0.5 * x + math.sqrt(3.0) * 0.5 * y)
    blue = 0.5 + 0.5 * (-0.5 * x - math.sqrt(3.0) * 0.5 * y)
    brightness = 0.2 + 0.8 * np.clip(reliability, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1) * brightness[..., None]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _panel(rgb, label, size=256, nearest=True):
    image = Image.fromarray(rgb, mode="RGB").resize(
        (size, size), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    )
    panel = Image.new("RGB", (size, size + 22), "white")
    panel.paste(image, (0, 22))
    ImageDraw.Draw(panel).text((5, 5), label, fill="black")
    return panel


def _grid(panels, columns):
    rows = math.ceil(len(panels) / columns)
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (columns * width, rows * height), (235, 235, 235))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * width, (index // columns) * height))
    return canvas


def _atomic_save_image(image, path):
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def _atomic_save_json(payload, path):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _first_value(batch, key):
    value = batch.get(key)
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


@torch.no_grad()
def write_latest_feature_maps(
    output_dir,
    batch,
    base0,
    base1,
    output0,
    output1,
    epoch,
    global_step,
    batch_idx,
    losses=None,
    gabor_parameters=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sides = (("A", output0), ("B", output1))

    input_panels = [
        _panel(
            _grayscale(batch["image0"][0, 0]),
            "image A (model input 512)",
            size=512,
            nearest=False,
        ),
        _panel(
            _grayscale(batch["image1"][0, 0]),
            "image B (model input 512)",
            size=512,
            nearest=False,
        ),
    ]
    _atomic_save_image(_grid(input_panels, columns=2), output_dir / "inputs.png")

    branch_panels = []
    scale_names = ("512", "256", "128")
    for side_name, output in sides:
        scale_count = len(output["unary"]) // 2
        for scale_index, scale_name in enumerate(scale_names[:scale_count]):
            odd = output["unary"][2 * scale_index][0]
            even = output["unary"][2 * scale_index + 1][0]
            branch_panels.extend(
                [
                    _panel(
                        _heatmap(_feature_activity(odd)),
                        f"{side_name} S{scale_name} Odd",
                    ),
                    _panel(
                        _heatmap(_feature_activity(even)),
                        f"{side_name} S{scale_name} Even",
                    ),
                ]
            )
    _atomic_save_image(
        _grid(branch_panels, columns=6), output_dir / "odd_even_features.png"
    )

    gate_panels = []
    for side_name, output in sides:
        reliability = output["reliability"][0, 0]
        gate_panels.append(
            _panel(_heatmap(reliability), f"{side_name} reliability")
        )
        gate_panels.append(
            _panel(
                _orientation_rgb(output["orientation"][0], reliability),
                f"{side_name} orientation",
            )
        )
        for scale_index, scale_name in enumerate(scale_names):
            gate_panels.append(
                _panel(
                    _heatmap(output["scale_weights"][0, scale_index]),
                    f"{side_name} scale {scale_name}",
                )
            )
        for scale_index, scale_name in enumerate(scale_names):
            selector = output["oe_selector"][0, scale_index].detach().float().cpu().numpy()
            selector_rgb = np.repeat((selector * 255).astype(np.uint8)[..., None], 3, axis=-1)
            gate_panels.append(
                _panel(
                    selector_rgb,
                    f"{side_name} OE {scale_name}",
                    nearest=True,
                )
            )
    _atomic_save_image(_grid(gate_panels, columns=8), output_dir / "physical_gates.png")

    descriptor_panels = []
    for side_name, base, output in (("A", base0, output0), ("B", base1, output1)):
        descriptor_panels.extend(
            [
                _panel(
                    _heatmap(_feature_activity(output["physical"][0])),
                    f"{side_name} physical",
                ),
                _panel(
                    _heatmap(_feature_norm(output["delta"][0])),
                    f"{side_name} delta norm",
                ),
                _panel(
                    _heatmap(_feature_norm(base[0])),
                    f"{side_name} SLiM norm",
                ),
                _panel(
                    _heatmap(_feature_norm(output["enhanced"][0])),
                    f"{side_name} enhanced norm",
                ),
            ]
        )
    _atomic_save_image(
        _grid(descriptor_panels, columns=4), output_dir / "descriptor_features.png"
    )

    payload = {
        "version": "Physical Encoder V2.1.2",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "batch_idx": int(batch_idx),
        "remote_id": str(_first_value(batch, "remote_id")),
        "variant": str(_first_value(batch, "remote_aug_variant")),
        "model_input_shape": list(batch["image0"].shape[-2:]),
        "coarse_feature_shape": list(output0["physical"].shape[-2:]),
        "display": {
            "input": "native 512x512 model tensor",
            "features": "native 64x64 coarse grid enlarged with nearest-neighbor",
        },
        "files": [
            "inputs.png",
            "odd_even_features.png",
            "physical_gates.png",
            "descriptor_features.png",
        ],
        "losses": {
            name: float(value.detach().float().cpu()) for name, value in (losses or {}).items()
        },
        "gabor": gabor_parameters or {},
    }
    _atomic_save_json(payload, output_dir / "latest_step.json")


__all__ = ["write_latest_feature_maps"]
