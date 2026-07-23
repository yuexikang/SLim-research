import math

import numpy as np
import torch

from src.physical.v3_himo import FrozenHIMO, HIMOConfig, cooccurrence_filter
from src.physical.v3_polarp import (
    PolarPConfig,
    describe_keypoints,
    match_code_exact_descriptors,
)


def test_frozen_himo_has_no_parameters_and_finite_outputs():
    generator = torch.Generator().manual_seed(66)
    image = torch.rand((1, 1, 64, 64), generator=generator) * 255
    model = FrozenHIMO(
        HIMOConfig(
            cooccurrence=False,
            patch_size=24,
            spatial_bins=12,
        )
    )
    output = model(image)
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    assert output["odd_response"].shape == (1, 2, 64, 64)
    assert output["even_response"].shape == (1, 2, 64, 64)
    assert output["orientation"].shape == (1, 1, 64, 64)
    assert torch.isfinite(output["orientation"]).all()
    assert torch.all(output["orientation"] >= 0)
    assert torch.all(output["orientation"] < math.pi)
    assert set(torch.unique(output["weight"]).tolist()).issubset({0.0, 1.0})


def test_cooccurrence_filter_is_deterministic():
    image = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) % 64
    first = cooccurrence_filter(image, max_levels=128)
    second = cooccurrence_filter(image, max_levels=128)
    assert first.shape == image.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_polarp_code_descriptor_is_480_dimensional_and_deterministic():
    height = width = 80
    yy, xx = np.mgrid[:height, :width]
    magnitude = np.ones((height, width), dtype=np.float32)
    orientation = np.mod(np.arctan2(yy - 40, xx - 40), math.pi).astype(np.float32)
    keypoints = np.array([[40.0, 40.0]], dtype=np.float32)
    config = PolarPConfig(
        patch_size=48,
        spatial_bins=12,
        orientation_bins=12,
        rotation_invariant=True,
        multiple_orientations=False,
    )
    points0, descriptors0, diagnostics0 = describe_keypoints(
        magnitude, orientation, keypoints, config
    )
    points1, descriptors1, diagnostics1 = describe_keypoints(
        magnitude, orientation, keypoints, config
    )
    assert points0.shape == (1, 2)
    assert descriptors0.shape == (1, 480)
    assert len(diagnostics0) == 1
    np.testing.assert_array_equal(points0, points1)
    np.testing.assert_array_equal(descriptors0, descriptors1)
    assert diagnostics0 == diagnostics1


def test_code_exact_matcher_keeps_unique_targets():
    descriptor0 = np.zeros((3, 480), dtype=np.float32)
    descriptor1 = np.zeros((2, 480), dtype=np.float32)
    descriptor0[0, 0] = 1
    descriptor0[1, 1] = 1
    descriptor0[2, 0] = 0.9
    descriptor1[0, 0] = 1
    descriptor1[1, 1] = 1
    source, target, distance = match_code_exact_descriptors(
        descriptor0, descriptor1
    )
    assert len(target) == len(np.unique(target))
    assert len(source) == len(distance)
    assert set(target.tolist()) == {0, 1}
