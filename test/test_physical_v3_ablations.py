import math

import numpy as np

from src.physical.v3_ablation import (
    descriptor_oracle_diagnostics,
    physical_branch_arrays,
    physical_state_diagnostics,
    unique_target_matches,
)
from src.physical.v3_polarp import PolarPConfig, describe_keypoints


def synthetic_state(size=32):
    yy, xx = np.mgrid[:size, :size]
    odd = (xx + 1).astype(np.float32)
    even = (yy + 2).astype(np.float32)
    orientation_odd = np.mod(xx / size * math.pi, math.pi).astype(np.float32)
    orientation_even = np.mod(yy / size * math.pi, math.pi).astype(np.float32)
    choose_odd = odd >= even
    return {
        "magnitude_odd": odd,
        "magnitude_even": even,
        "orientation_odd": orientation_odd,
        "orientation_even": orientation_even,
        "orientation": np.where(
            choose_odd, orientation_odd, orientation_even
        ).astype(np.float32),
        "weight": np.ones((size, size), dtype=np.float32),
    }


def test_all_fixed_physical_branches_are_finite():
    state = synthetic_state()
    outputs = {}
    for branch in ("pvalid", "odd", "even", "hard_magnitude", "soft_magnitude"):
        magnitude, orientation = physical_branch_arrays(state, branch)
        assert magnitude.shape == (32, 32)
        assert orientation.shape == (32, 32)
        assert np.isfinite(magnitude).all()
        assert np.isfinite(orientation).all()
        assert np.all(orientation >= 0)
        assert np.all(orientation < math.pi)
        outputs[branch] = magnitude
    assert not np.array_equal(outputs["odd"], outputs["even"])


def test_no_psd_and_symmetric_halfturn_keep_480_dimensions():
    size = 80
    yy, xx = np.mgrid[:size, :size]
    magnitude = np.hypot(xx - 40, yy - 40).astype(np.float32) + 1
    orientation = np.mod(
        np.arctan2(yy - 40, xx - 40),
        math.pi,
    ).astype(np.float32)
    point = np.array([[40.0, 40.0]], dtype=np.float32)
    for mode in ("none", "symmetric"):
        config = PolarPConfig(
            patch_size=48,
            multiple_orientations=False,
            descriptor_normalization="l2",
            half_turn_mode=mode,
        )
        _, descriptor, diagnostics = describe_keypoints(
            magnitude,
            orientation,
            point,
            config,
        )
        assert descriptor.shape == (1, 480)
        assert np.isfinite(descriptor).all()
        np.testing.assert_allclose(np.linalg.norm(descriptor, axis=1), 1, atol=1e-6)
        assert diagnostics[0]["symmetric_half_turn"] == float(mode == "symmetric")


def test_descriptor_oracle_is_perfect_for_identity_correspondences():
    points = np.array([[4, 4], [8, 8], [12, 12]], dtype=np.float32)
    descriptors = np.zeros((3, 480), dtype=np.float32)
    descriptors[np.arange(3), np.arange(3)] = 1
    metrics = descriptor_oracle_diagnostics(
        points,
        descriptors,
        points,
        descriptors,
        np.eye(3),
        correct_threshold=1.0,
    )
    assert metrics["oracle_eligible"] == 3
    assert metrics["oracle_r1"] == 1
    assert metrics["oracle_r5"] == 1
    assert metrics["positive_distance"] == 0


def test_unique_target_matcher_reports_hubness():
    descriptors0 = np.zeros((4, 480), dtype=np.float32)
    descriptors1 = np.zeros((2, 480), dtype=np.float32)
    descriptors0[:, 0] = [0, 0.1, 0.2, 4]
    descriptors1[:, 0] = [0, 4]
    source, target, _, metrics = unique_target_matches(
        descriptors0,
        descriptors1,
    )
    assert len(source) == len(target) == 2
    assert metrics["unique_target_ratio"] == 0.5
    assert metrics["max_target_fan_in"] == 3


def test_physical_state_identity_has_zero_orientation_error():
    state = synthetic_state()
    points = np.array([[5, 5], [10, 10], [20, 20]], dtype=np.float32)
    metrics = physical_state_diagnostics(
        state,
        state,
        points,
        np.eye(3),
    )
    assert metrics["physical_samples"] == 3
    assert metrics["orientation_median_error_deg"] < 1e-5
    assert metrics["odd_correlation"] > 0.999
    assert metrics["even_correlation"] > 0.999
    assert metrics["vimo_agreement"] == 1
    assert metrics["vimo_iou"] == 1
