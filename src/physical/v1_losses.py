import torch
from torch.nn import functional as F


def _flatten_field(field):
    return field.flatten(2).transpose(1, 2).contiguous()


def doubled_angle_to_direction(orientation):
    theta = 0.5 * torch.atan2(orientation[..., 1], orientation[..., 0])
    return torch.stack([theta.cos(), theta.sin()], dim=-1)


def homography_jacobian(homography, points, eps=1e-6):
    homography = homography.to(dtype=points.dtype)
    x = points[..., 0]
    y = points[..., 1]
    h11, h12, h13 = homography[:, 0, 0], homography[:, 0, 1], homography[:, 0, 2]
    h21, h22, h23 = homography[:, 1, 0], homography[:, 1, 1], homography[:, 1, 2]
    h31, h32, h33 = homography[:, 2, 0], homography[:, 2, 1], homography[:, 2, 2]
    nx = h11 * x + h12 * y + h13
    ny = h21 * x + h22 * y + h23
    denominator = h31 * x + h32 * y + h33
    safe = torch.where(
        denominator.abs() > eps,
        denominator,
        denominator.sign().masked_fill(denominator == 0, 1.0) * eps,
    )
    denominator_squared = safe.square()
    row0 = torch.stack(
        [
            (h11 * safe - h31 * nx) / denominator_squared,
            (h12 * safe - h32 * nx) / denominator_squared,
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            (h21 * safe - h31 * ny) / denominator_squared,
            (h22 * safe - h32 * ny) / denominator_squared,
        ],
        dim=-1,
    )
    jacobian = torch.stack([row0, row1], dim=-2)
    valid = denominator.abs() > eps
    valid = valid & torch.isfinite(jacobian).flatten(-2).all(dim=-1)
    return jacobian, valid


def orientation_equivariance_loss(
    output0,
    output1,
    batch,
    correspondences,
    coarse_scale=8,
    eps=1e-6,
):
    b_ids, i_ids, j_ids = correspondences
    if b_ids.numel() == 0:
        return output0["orientation"].sum() * 0.0

    orientation0 = _flatten_field(output0["orientation"])[b_ids, i_ids].float()
    orientation1 = _flatten_field(output1["orientation"])[b_ids, j_ids].float()
    confidence0 = _flatten_field(output0["confidence"])[b_ids, i_ids, 0].float()
    confidence1 = _flatten_field(output1["confidence"])[b_ids, j_ids, 0].float()

    width0 = output0["orientation"].shape[-1]
    x = (i_ids.remainder(width0).float() + 0.5) * float(coarse_scale)
    y = (torch.div(i_ids, width0, rounding_mode="floor").float() + 0.5) * float(
        coarse_scale
    )
    points = torch.stack([x, y], dim=-1)
    homography = batch["H_0to1"].to(device=points.device, dtype=points.dtype)[b_ids]
    jacobian, valid = homography_jacobian(homography, points, eps=eps)

    direction0 = doubled_angle_to_direction(orientation0)
    direction1 = doubled_angle_to_direction(orientation1)
    expected1 = torch.einsum("nij,nj->ni", jacobian, direction0)
    expected_norm = torch.linalg.vector_norm(expected1, dim=-1)
    valid = valid & torch.isfinite(expected1).all(dim=-1) & (expected_norm > eps)
    expected1 = F.normalize(expected1, p=2, dim=-1, eps=eps)

    error = 1.0 - (direction1 * expected1).sum(dim=-1).abs().clamp(0.0, 1.0)
    weight = torch.minimum(confidence0, confidence1).detach() * valid.float()
    return (weight * error).sum() / weight.sum().clamp_min(eps)


__all__ = [
    "doubled_angle_to_direction",
    "homography_jacobian",
    "orientation_equivariance_loss",
]
