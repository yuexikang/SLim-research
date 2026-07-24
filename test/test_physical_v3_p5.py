import math

import torch

from src.physical.v3_p5_models import (
    PointwiseP5Head,
    RectConvP5Head,
    build_p5_state,
    sample_coarse_centers,
)


def synthetic_himo(batch=2, size=64):
    generator = torch.Generator().manual_seed(66)
    odd = torch.rand((batch, 1, size, size), generator=generator) * 10
    even = torch.rand((batch, 1, size, size), generator=generator) * 10
    orientation = torch.rand(
        (batch, 1, size, size),
        generator=generator,
    ) * math.pi
    weight = torch.ones((batch, 1, size, size))
    return {
        "magnitude_odd": odd,
        "magnitude_even": even,
        "orientation": orientation,
        "weight": weight,
    }


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def test_p5_state_shape_range_and_coarse_sampling():
    state = build_p5_state(synthetic_himo())
    assert state.shape == (2, 5, 64, 64)
    assert torch.isfinite(state).all()
    assert torch.all(state[:, 4] >= 0)
    assert torch.all(state[:, 4] <= 1)
    coarse = sample_coarse_centers(state, coarse_scale=8)
    assert coarse.shape == (2, 5, 8, 8)


def test_pointwise_and_rectconv_outputs_match_and_are_normalized():
    state = sample_coarse_centers(build_p5_state(synthetic_himo()))
    pointwise = PointwiseP5Head()
    rectconv = RectConvP5Head()
    pointwise_output = pointwise(state)
    rectconv_output = rectconv(state)
    assert pointwise_output.shape == rectconv_output.shape == (2, 128, 8, 8)
    torch.testing.assert_close(
        torch.linalg.vector_norm(pointwise_output, dim=1),
        torch.ones((2, 8, 8)),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(rectconv_output, dim=1),
        torch.ones((2, 8, 8)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pointwise_and_rectconv_parameter_counts_are_matched():
    pointwise = parameter_count(PointwiseP5Head())
    rectconv = parameter_count(RectConvP5Head())
    relative_difference = abs(pointwise - rectconv) / pointwise
    assert relative_difference < 0.05
