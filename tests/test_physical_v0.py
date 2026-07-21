import json

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.datasets.remote_sensing import RemoteSensingHomographyDataset
from src.physical.data import PhysicalV0DataModule, stratified_manifest_indices
from src.physical.matching import PPMatchingLoss
from src.physical.models import (
    FixedQuadratureOrientationBank,
    SoftOrientationCanonicalization,
    build_physical_v0_encoder,
    count_trainable_parameters,
)


def test_gabor_bank_is_fixed_and_quadrature():
    bank = FixedQuadratureOrientationBank()
    assert list(bank.parameters()) == []
    assert bank.kernels.shape == (16, 1, 9, 9)
    even, odd = bank.kernels[:8], bank.kernels[8:]
    inner_product = (even * odd).flatten(1).sum(dim=1)
    assert torch.all(inner_product.abs() < 1e-5)


def test_integer_orientation_shift_is_circular():
    canonicalizer = SoftOrientationCanonicalization()
    energy = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
    shifted = canonicalizer.circular_sample(energy, torch.ones(1, 1, 1))
    expected = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0], dtype=torch.float32).reshape(1, 8, 1, 1)
    assert torch.equal(shifted, expected)


@pytest.mark.parametrize(
    "model_name",
    ["physical_full", "physical_no_canon", "physical_single_scale", "tiny_cnn"],
)
def test_encoder_output_shape_and_norm(model_name):
    model = build_physical_v0_encoder(model_name).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 1, 64, 64))
    assert output.shape == (1, 128, 8, 8)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=1), torch.ones(1, 8, 8), atol=1e-5)


def test_tiny_cnn_parameter_count_is_within_five_percent():
    physical = count_trainable_parameters(build_physical_v0_encoder("physical_full"))
    tiny = count_trainable_parameters(build_physical_v0_encoder("tiny_cnn"))
    assert abs(physical - tiny) / physical <= 0.05


def test_stratified_selection_is_deterministic_and_balanced():
    rows = [
        {"dataset": "a", "subset": "x", "id": str(index)} for index in range(80)
    ] + [
        {"dataset": "b", "subset": "y", "id": str(index)} for index in range(20)
    ]
    first = stratified_manifest_indices(rows, ratio=0.3, seed=66)
    second = stratified_manifest_indices(rows, ratio=0.3, seed=66)
    assert first == second
    assert len(first) == 30
    selected_a = sum(rows[index]["dataset"] == "a" for index in first)
    assert selected_a == 24


def test_deterministic_train_homography_depends_on_epoch(tmp_path):
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.arange(64 * 64, dtype=np.uint8).reshape(64, 64))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "dataset": "test",
                "subset": "test",
                "split": "train",
                "mode": "single_synth",
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = RemoteSensingHomographyDataset(
        manifest,
        image_size=64,
        mode="train",
        aug_variants=["roll"],
        seed=66,
        deterministic_train=True,
    )
    first = dataset[0]["H_0to1"]
    repeated = dataset[0]["H_0to1"]
    dataset.set_epoch(1)
    next_epoch = dataset[0]["H_0to1"]
    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_epoch)


def test_one_variant_per_row_is_deterministic_and_changes_across_epochs(tmp_path):
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.arange(64 * 64, dtype=np.uint8).reshape(64, 64))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "dataset": "test",
                "subset": "test",
                "split": "train",
                "mode": "single_synth",
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = RemoteSensingHomographyDataset(
        manifest,
        image_size=64,
        mode="train",
        seed=66,
        deterministic_train=True,
        one_variant_per_row=True,
    )
    assert len(dataset) == 1
    first = dataset[0]
    repeated = dataset[0]
    assert first["remote_aug_variant"] == repeated["remote_aug_variant"]
    assert torch.equal(first["H_0to1"], repeated["H_0to1"])

    variants = set()
    homographies = []
    for epoch in range(10):
        dataset.set_epoch(epoch)
        item = dataset[0]
        variants.add(item["remote_aug_variant"])
        homographies.append(item["H_0to1"])
    assert len(variants) > 1
    assert any(not torch.equal(homographies[0], value) for value in homographies[1:])


def test_validation_one_variant_is_fixed_and_matches_full_validation(tmp_path):
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.arange(64 * 64, dtype=np.uint8).reshape(64, 64))
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "id": f"sample-{index}",
            "dataset": "test",
            "subset": "test",
            "split": "val",
            "mode": "single_synth",
            "image": str(image_path),
        }
        for index in range(4)
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    fast = RemoteSensingHomographyDataset(
        manifest,
        image_size=64,
        mode="val",
        seed=66,
        one_variant_per_row=True,
    )
    full = RemoteSensingHomographyDataset(
        manifest,
        image_size=64,
        mode="val",
        seed=66,
    )
    assert len(fast) == len(rows)
    assert len(full) == len(rows) * len(full.aug_variants)

    first_pass = [fast[index] for index in range(len(fast))]
    fast.set_epoch(9)
    second_pass = [fast[index] for index in range(len(fast))]
    for row_index, (first, second) in enumerate(zip(first_pass, second_pass)):
        assert first["remote_aug_variant"] == second["remote_aug_variant"]
        assert torch.equal(first["H_0to1"], second["H_0to1"])
        variant_index = full.aug_variants.index(first["remote_aug_variant"])
        full_item = full[row_index * len(full.aug_variants) + variant_index]
        assert torch.equal(first["image0"], full_item["image0"])
        assert torch.equal(first["image1"], full_item["image1"])
        assert torch.equal(first["H_0to1"], full_item["H_0to1"])


def test_data_module_switches_from_fast_to_full_validation(tmp_path):
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64), dtype=np.uint8))
    train_manifest = tmp_path / "train.jsonl"
    val_manifest = tmp_path / "val.jsonl"
    train_manifest.write_text(
        json.dumps(
            {
                "id": "train-0",
                "dataset": "test",
                "subset": "train",
                "split": "train",
                "mode": "single_synth",
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    val_rows = [
        {
            "id": f"val-{index}",
            "dataset": "test",
            "subset": "val",
            "split": "val",
            "mode": "single_synth",
            "image": str(image_path),
        }
        for index in range(3)
    ]
    val_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in val_rows), encoding="utf-8"
    )
    data_module = PhysicalV0DataModule(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        experiment_dir=tmp_path / "experiment",
        image_size=64,
        batch_size=1,
        val_batch_size=1,
        num_workers=0,
        train_data_ratio=1.0,
        val_one_variant_per_row=True,
    )
    data_module.setup("fit")
    assert len(data_module.val_dataset) == len(val_rows)
    assert len(data_module.full_val_dataset) == len(val_rows) * 5
    assert len(data_module.val_dataloader().dataset) == len(val_rows)
    data_module.enable_full_validation()
    assert len(data_module.val_dataloader().dataset) == len(val_rows) * 5


def test_full_and_chunked_loss_and_gradients_match():
    torch.manual_seed(7)
    raw0 = torch.randn(2, 16, 8, 8)
    raw1 = torch.randn(2, 16, 8, 8)
    descriptor0_full = F.normalize(raw0, dim=1).requires_grad_(True)
    descriptor1_full = F.normalize(raw1, dim=1).requires_grad_(True)
    descriptor0_chunk = descriptor0_full.detach().clone().requires_grad_(True)
    descriptor1_chunk = descriptor1_full.detach().clone().requires_grad_(True)
    indices = torch.arange(64)
    correspondences = (
        torch.cat([torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)]),
        torch.cat([indices, indices]),
        torch.cat([indices.roll(1), indices.roll(2)]),
    )
    selected = torch.arange(100)
    full = PPMatchingLoss(positive_percent=1.0, chunk_size=17)
    chunked = PPMatchingLoss(positive_percent=1.0, chunk_size=17)
    chunked.load_state_dict(full.state_dict())

    full_loss = full(descriptor0_full, descriptor1_full, correspondences, "full", selected)
    chunked_loss = chunked(descriptor0_chunk, descriptor1_chunk, correspondences, "chunked", selected)
    full_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(full_loss, chunked_loss, rtol=1e-5, atol=1e-6)
    assert torch.allclose(descriptor0_full.grad, descriptor0_chunk.grad, rtol=1e-4, atol=1e-6)
    assert torch.allclose(descriptor1_full.grad, descriptor1_chunk.grad, rtol=1e-4, atol=1e-6)
    assert torch.allclose(full.log_temperature.grad, chunked.log_temperature.grad, rtol=1e-4, atol=1e-6)
