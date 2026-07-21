import math

import pytest
import torch
from torch.nn import functional as F

from src.physical.data import manifest_indices_from_selected_rows
from src.physical.v1_lightning import PhysicalV1Module
from src.physical.v1_losses import orientation_equivariance_loss
from src.physical.v1_models import (
    ContinuousOrientationCanonicalization,
    OrientationAlignedNeighborhood,
    ParametricSteerableGaborBank,
    PhysicalOrientationEstimator,
    build_physical_v1_encoder,
    count_trainable_parameters,
)


def test_parametric_bank_constraints_and_quadrature():
    bank = ParametricSteerableGaborBank()
    with torch.no_grad():
        bank.delta_lambda.copy_(torch.tensor([-20.0, 0.0, 20.0]))
        bank.delta_sigma.copy_(torch.tensor([20.0, 0.0, -20.0]))
        bank.gamma_logits.copy_(torch.tensor([-20.0, 0.0, 20.0]))
    wavelength, sigma, gamma = bank.constrained_parameters()
    assert torch.all(wavelength >= bank.base_wavelengths * math.exp(-0.25))
    assert torch.all(wavelength <= bank.base_wavelengths * math.exp(0.25))
    assert torch.all(sigma >= 0.56 * wavelength * math.exp(-0.20))
    assert torch.all(sigma <= 0.56 * wavelength * math.exp(0.20))
    assert torch.all((gamma >= 0.3) & (gamma <= 0.8))
    for frequency in range(3):
        even, odd = bank.kernels_for_frequency(frequency)
        assert even.shape[0] == odd.shape[0] == 8
        assert torch.allclose(even.mean(dim=(-2, -1)), torch.zeros(8, 1), atol=1e-6)
        assert torch.allclose(odd.mean(dim=(-2, -1)), torch.zeros(8, 1), atol=1e-6)
        assert torch.allclose(
            torch.linalg.vector_norm(even.flatten(1), dim=1), torch.ones(8), atol=1e-5
        )
        assert torch.allclose(
            torch.linalg.vector_norm(odd.flatten(1), dim=1), torch.ones(8), atol=1e-5
        )
        inner_product = (even * odd).flatten(1).sum(dim=1)
        assert torch.all(inner_product.abs() < 1e-5)
    assert sum(parameter.numel() for parameter in bank.parameters()) == 9


def test_phase_agreement_range_and_no_nan():
    estimator = PhysicalOrientationEstimator()
    even = torch.randn(2, 3, 8, 16, 16)
    odd = torch.randn_like(even)
    amplitude = torch.sqrt(even.square() + odd.square() + 1e-6)
    phase, theta, orientation, confidence = estimator(even, odd, amplitude)
    assert torch.isfinite(phase).all()
    assert torch.isfinite(theta).all()
    assert torch.isfinite(orientation).all()
    assert torch.isfinite(confidence).all()
    assert phase.min() >= 0 and phase.max() <= 1
    assert confidence.min() >= 0 and confidence.max() <= 1


def test_orientation_estimator_zero_response_has_finite_gradient():
    estimator = PhysicalOrientationEstimator()
    even = torch.zeros(1, 3, 8, 4, 4, requires_grad=True)
    odd = torch.zeros_like(even, requires_grad=True)
    amplitude = torch.sqrt(even.square() + odd.square() + 1e-6)
    phase, theta, orientation, confidence = estimator(even, odd, amplitude)
    loss = phase.sum() + theta.sum() + orientation.sum() + confidence.sum()
    loss.backward()
    assert torch.isfinite(even.grad).all()
    assert torch.isfinite(odd.grad).all()
    assert torch.equal(confidence, torch.zeros_like(confidence))


def test_canonicalization_confidence_bypass_and_full_gate():
    canonicalizer = ContinuousOrientationCanonicalization()
    response = torch.randn(1, 3, 8, 5, 5)
    theta = torch.full((1, 5, 5), math.pi / 8)
    zero = torch.zeros(1, 1, 5, 5)
    one = torch.ones_like(zero)
    expected = canonicalizer.circular_sample(response, torch.ones(1, 5, 5))
    assert torch.equal(canonicalizer(response, theta, zero), response)
    assert torch.allclose(canonicalizer(response, theta, one), expected)


def test_oan_symmetry_bypass_and_pi_equivalence():
    torch.manual_seed(4)
    oan = OrientationAlignedNeighborhood(channels=3)
    feature = torch.randn(1, 3, 9, 9)
    theta = torch.randn(1, 9, 9)
    symmetric = oan.symmetric_weight
    assert torch.allclose(symmetric, torch.flip(symmetric, dims=(-2, -1)))
    aligned = oan.aligned_only(feature, theta)
    aligned_pi = oan.aligned_only(feature, theta + math.pi)
    assert torch.allclose(aligned, aligned_pi, rtol=1e-4, atol=1e-5)
    zero = torch.zeros(1, 1, 9, 9)
    one = torch.ones_like(zero)
    assert torch.equal(oan(feature, theta, zero), feature)
    assert torch.allclose(oan(feature, theta, one), aligned)


@pytest.mark.parametrize(
    "model_name",
    [
        "physical_v1_core",
        "physical_v1_no_oan",
        "physical_v1_energy_only",
        "physical_v1_no_confidence_gate",
        "physical_v1_simple_scale",
        "physical_v1_fixed_bank",
    ],
)
def test_v1_output_contract_and_norm(model_name):
    model = build_physical_v1_encoder(model_name).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 1, 64, 64))
    expected = {
        "fused": (1, 128, 8, 8),
        "edge": (1, 32, 8, 8),
        "contour": (1, 32, 8, 8),
        "stable": (1, 32, 8, 8),
        "orientation": (1, 2, 8, 8),
        "confidence": (1, 1, 8, 8),
        "scale_weights": (1, 3, 8, 8),
        "expert_weights": (1, 3, 8, 8),
    }
    assert {name: tuple(value.shape) for name, value in output.items()} == expected
    for name in ("fused", "edge", "contour", "stable", "orientation"):
        norm = torch.linalg.vector_norm(output[name], dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    assert torch.allclose(output["scale_weights"].sum(dim=1), torch.ones(1, 8, 8))
    assert torch.allclose(output["expert_weights"].sum(dim=1), torch.ones(1, 8, 8))


def test_core_parameter_count_matches_design_and_fixed_bank_freezes_filters():
    core = build_physical_v1_encoder("physical_v1_core")
    fixed = build_physical_v1_encoder("physical_v1_fixed_bank")
    assert count_trainable_parameters(core) == 34927
    assert count_trainable_parameters(fixed) == 34918
    assert all(parameter.requires_grad for parameter in core.gabor.physical_parameters())
    assert not any(parameter.requires_grad for parameter in fixed.gabor.physical_parameters())


def test_orientation_loss_uses_homography_jacobian():
    source_orientation = torch.tensor([1.0, 0.0]).reshape(1, 2, 1, 1)
    target_orientation = torch.tensor([-1.0, 0.0]).reshape(1, 2, 1, 1)
    confidence = torch.ones(1, 1, 1, 1)
    output0 = {"orientation": source_orientation, "confidence": confidence}
    output1 = {"orientation": target_orientation, "confidence": confidence}
    homography = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    batch = {"H_0to1": homography}
    correspondences = (
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
    )
    loss = orientation_equivariance_loss(
        output0, output1, batch, correspondences, coarse_scale=8
    )
    assert loss < 1e-6


def test_all_v1_core_losses_backpropagate_without_unused_parameters():
    torch.manual_seed(8)
    module = PhysicalV1Module(
        model_name="physical_v1_core", similarity_mode="chunked", chunk_size=32
    )
    image0 = torch.rand(1, 1, 64, 64)
    image1 = image0.clone()
    batch = {
        "image0": image0,
        "image1": image1,
        "H_0to1": torch.eye(3)[None],
        "remote_aug_variant": ["translation"],
    }
    output0, output1, correspondences = module._forward_pair(batch)
    total, fused, orientation, branch, branch_losses = module._losses(
        batch, output0, output1, correspondences, deterministic=True
    )
    assert torch.isfinite(torch.stack([total, fused, orientation, branch])).all()
    assert set(branch_losses) == {"edge", "contour", "stable"}
    total.backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_selected_rows_are_mapped_independently_of_file_order():
    rows = [
        {"id": "a", "dataset": "x"},
        {"id": "b", "dataset": "x"},
        {"id": "c", "dataset": "y"},
    ]
    selected = [rows[2], rows[0]]
    assert manifest_indices_from_selected_rows(rows, selected) == [0, 2]
    with pytest.raises(ValueError, match="absent"):
        manifest_indices_from_selected_rows(rows, [{"id": "missing"}])
