import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.physical.v2_losses import (
    positive_dual_softmax,
    recovery_and_preservation_loss,
)
from src.physical.matching import PPMatchingLoss
from src.physical.v2_models import (
    DensePolarDescriptor,
    PhysicalEncoderV2,
    SharedMASW,
    build_physical_v2_encoder,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_manifest_is_exact_v1_selection():
    source = ROOT / "logs/tb_logs/physical_v0/tiny_cnn_ratio30_gpu3_bs8_seed66/selected_train_rows.jsonl"
    target = ROOT / "data/remote_archive/manifests/train_physical_v2_optical_single_ratio30_seed66.jsonl"
    summary_path = ROOT / "data/remote_archive/manifests/train_physical_v2_optical_single_ratio30_seed66_summary.json"
    assert source.read_bytes() == target.read_bytes()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(rows) == summary["rows"] == 16337
    assert hashlib.sha256(target.read_bytes()).hexdigest() == summary["sha256"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["image"] for row in rows}) == len(rows)


def test_shared_masw_returns_finite_unit_axial_direction():
    torch.manual_seed(3)
    masw = SharedMASW()
    gx = torch.randn(2, 1, 24, 24)
    gy = torch.randn(2, 1, 24, 24)
    a, b = masw(gx, gy)
    magnitude, orientation = masw.fields(a, b)
    assert magnitude.shape == (2, 1, 24, 24)
    assert orientation.shape == (2, 2, 24, 24)
    assert torch.isfinite(magnitude).all()
    assert torch.isfinite(orientation).all()
    norm = torch.linalg.vector_norm(orientation, dim=1)
    valid = magnitude[:, 0] > 2e-2
    assert torch.allclose(norm[valid], torch.ones_like(norm[valid]), atol=2e-3, rtol=2e-3)


def test_phase_agreement_is_bounded_without_saturation():
    encoder = PhysicalEncoderV2(
        enable_pair_transformer=False, enable_polar=False
    ).eval()
    image = torch.rand(1, 1, 64, 64)
    fields = encoder._physical_scale(image, scale_index=2)
    reliability = fields["reliability"]
    assert torch.isfinite(reliability).all()
    assert reliability.min() >= 0
    assert reliability.max() <= 1
    assert (reliability < 0.999).float().mean() > 0.95


def test_hard_odd_even_coupling_uses_analytic_magnitude_selector():
    encoder = PhysicalEncoderV2(
        enable_pair_transformer=False, enable_polar=False
    )
    odd = torch.ones(1, 96, 4, 4)
    even = torch.full_like(odd, 2.0)
    magnitude_odd = torch.zeros(1, 1, 4, 4)
    magnitude_even = torch.ones_like(magnitude_odd)
    magnitude_odd[:, :, :2] = 2.0
    fields = {
        "magnitude_odd": magnitude_odd,
        "magnitude_even": magnitude_even,
        "orientation_odd": F.normalize(torch.randn(1, 2, 4, 4), dim=1),
        "orientation_even": F.normalize(torch.randn(1, 2, 4, 4), dim=1),
    }
    feature, _, _, selector = encoder._couple(odd, even, fields)
    assert torch.equal(selector[:, :, :2], torch.ones_like(selector[:, :, :2]))
    assert torch.equal(selector[:, :, 2:], torch.zeros_like(selector[:, :, 2:]))
    assert torch.equal(feature[:, :, :2], odd[:, :, :2])
    assert torch.equal(feature[:, :, 2:], even[:, :, 2:])


def test_polar_reversal_fusion_is_strictly_invariant():
    polar = DensePolarDescriptor(channels=16, heads=4, chunk_size=8)
    descriptor0 = torch.randn(2, 7, 16)
    descriptor_pi = torch.randn(2, 7, 16)
    forward = polar.reversal_fuse(descriptor0, descriptor_pi)
    reversed_order = polar.reversal_fuse(descriptor_pi, descriptor0)
    assert torch.equal(forward, reversed_order)


def test_v2_pair_exchange_and_output_contract():
    torch.manual_seed(4)
    encoder = build_physical_v2_encoder("physical_v2_core", polar_chunk_size=32).eval()
    image0 = torch.rand(1, 1, 64, 64)
    image1 = torch.rand(1, 1, 64, 64)
    output0, output1 = encoder.forward_pair(image0, image1)
    swapped1, swapped0 = encoder.forward_pair(image1, image0)
    for key in ("physical", "delta", "orientation", "reliability", "scale_weights", "oe_selector"):
        assert torch.allclose(output0[key], swapped0[key], atol=1e-5, rtol=1e-5)
        assert torch.allclose(output1[key], swapped1[key], atol=1e-5, rtol=1e-5)
    assert output0["physical"].shape == (1, 96, 8, 8)
    assert output0["delta"].shape == (1, 192, 8, 8)
    assert output0["scale_weights"].shape == (1, 3, 8, 8)
    assert output0["oe_selector"].shape == (1, 3, 8, 8)
    norm = torch.linalg.vector_norm(output0["physical"], dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(output0["delta"]) == 0


def test_positive_dual_softmax_and_recovery_loss_backpropagate():
    torch.manual_seed(8)
    base0 = torch.randn(2, 8, 4, 4)
    base1 = torch.randn(2, 8, 4, 4)
    delta0 = torch.randn_like(base0, requires_grad=True)
    delta1 = torch.randn_like(base1, requires_grad=True)
    b_ids = torch.tensor([0, 0, 1, 1])
    i_ids = torch.tensor([1, 5, 2, 7])
    j_ids = torch.tensor([1, 5, 2, 7])
    correspondences = (b_ids, i_ids, j_ids)
    probability = positive_dual_softmax(
        base0 + delta0,
        base1 + delta1,
        correspondences,
        temperature=torch.tensor(0.05),
        chunk_size=2,
    )
    assert probability.shape == (4,)
    assert ((probability >= 0) & (probability <= 1)).all()
    recover, keep, diagnostics = recovery_and_preservation_loss(
        base0,
        base1,
        base0 + delta0,
        base1 + delta1,
        correspondences,
        temperature=torch.tensor(0.05),
        chunk_size=2,
    )
    (recover + keep).backward()
    assert torch.isfinite(delta0.grad).all()
    assert torch.isfinite(delta1.grad).all()
    assert set(diagnostics) == {
        "base_positive_confidence",
        "enhanced_positive_confidence",
        "recovery_weight",
    }


def test_stable_pp_loss_backpropagates_below_legacy_probability_floor():
    torch.manual_seed(9)
    descriptor0 = F.normalize(torch.randn(1, 8, 64, 64), dim=1).requires_grad_()
    descriptor1 = F.normalize(torch.randn(1, 8, 64, 64), dim=1).requires_grad_()
    correspondences = (
        torch.tensor([0]),
        torch.tensor([123]),
        torch.tensor([2345]),
    )
    loss_fn = PPMatchingLoss(
        positive_percent=1.0,
        temperature=0.05,
        chunk_size=1,
        stable_log=True,
    )
    loss = loss_fn(descriptor0, descriptor1, correspondences, mode="chunked")
    loss.backward()
    assert loss > -torch.log(torch.tensor(1e-6))
    assert descriptor0.grad.abs().sum() > 0
    assert descriptor1.grad.abs().sum() > 0
