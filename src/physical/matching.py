import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@torch.no_grad()
def coarse_homography_correspondences(batch, coarse_scale=8):
    image0 = batch["image0"]
    image1 = batch["image1"]
    device = image0.device
    batch_size, _, height0, width0 = image0.shape
    _, _, height1, width1 = image1.shape
    h0, w0 = height0 // coarse_scale, width0 // coarse_scale
    h1, w1 = height1 // coarse_scale, width1 // coarse_scale

    y0, x0 = torch.meshgrid(
        torch.arange(h0, device=device, dtype=image0.dtype),
        torch.arange(w0, device=device, dtype=image0.dtype),
        indexing="ij",
    )
    y1, x1 = torch.meshgrid(
        torch.arange(h1, device=device, dtype=image1.dtype),
        torch.arange(w1, device=device, dtype=image1.dtype),
        indexing="ij",
    )
    grid0 = torch.stack([x0, y0], dim=-1).reshape(1, -1, 2) + 0.5
    grid1 = torch.stack([x1, y1], dim=-1).reshape(1, -1, 2) + 0.5
    grid0 = grid0.repeat(batch_size, 1, 1) * coarse_scale
    grid1 = grid1.repeat(batch_size, 1, 1) * coarse_scale

    def warp(points, homography):
        ones = torch.ones(*points.shape[:-1], 1, device=device, dtype=points.dtype)
        homogeneous = torch.cat([points, ones], dim=-1)
        warped = torch.einsum("bij,bnj->bni", homography.to(points.dtype), homogeneous)
        denominator = warped[..., 2:3]
        safe = torch.where(
            denominator.abs() > 1e-8,
            denominator,
            torch.full_like(denominator, 1e-8),
        )
        return warped[..., :2] / safe

    homography = batch["H_0to1"].to(device=device, dtype=image0.dtype)
    warped0 = warp(grid0, homography) / coarse_scale - 0.5
    warped1 = warp(grid1, torch.linalg.inv(homography)) / coarse_scale - 0.5
    rounded0 = warped0.round().long()
    rounded1 = warped1.round().long()
    index1 = rounded0[..., 0] + rounded0[..., 1] * w1
    index0 = rounded1[..., 0] + rounded1[..., 1] * w0

    invalid0 = (
        (rounded0[..., 0] < 0)
        | (rounded0[..., 0] >= w1)
        | (rounded0[..., 1] < 0)
        | (rounded0[..., 1] >= h1)
    )
    invalid1 = (
        (rounded1[..., 0] < 0)
        | (rounded1[..., 0] >= w0)
        | (rounded1[..., 1] < 0)
        | (rounded1[..., 1] >= h0)
    )
    index1[invalid0] = 0
    index0[invalid1] = 0
    loop_back = torch.gather(index0, 1, index1.clamp(0, h1 * w1 - 1))
    source_indices = torch.arange(h0 * w0, device=device)[None]
    valid = (loop_back == source_indices) & ~invalid0
    valid[:, 0] = False
    b_ids, i_ids = torch.where(valid)
    j_ids = index1[b_ids, i_ids]
    return b_ids, i_ids, j_ids


def flatten_descriptors(descriptors):
    return descriptors.flatten(2).transpose(1, 2).contiguous()


def full_similarity(descriptor0, descriptor1, temperature):
    flat0 = flatten_descriptors(descriptor0)
    flat1 = flatten_descriptors(descriptor1)
    return torch.einsum("bic,bjc->bij", flat0, flat1) / temperature.clamp_min(1e-4)


class PPMatchingLoss(nn.Module):
    def __init__(
        self,
        gamma=2.0,
        positive_percent=0.9,
        temperature=0.05,
        chunk_size=256,
        stable_log=False,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.positive_percent = float(positive_percent)
        self.chunk_size = int(chunk_size)
        self.stable_log = bool(stable_log)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(1e-3, 1.0)

    def select_positives(self, correspondences, device, selected=None):
        b_ids, i_ids, j_ids = correspondences
        if b_ids.numel() == 0:
            raise RuntimeError("No valid coarse homography correspondences in this batch.")
        if selected is not None:
            return tuple(value[selected] for value in (b_ids, i_ids, j_ids))
        count = max(1, int(b_ids.numel() * self.positive_percent))
        indices = torch.randperm(b_ids.numel(), device=device)[:count]
        return b_ids[indices], i_ids[indices], j_ids[indices]

    def _focal(self, probability):
        return -(1.0 - probability).pow(self.gamma) * probability.clamp_min(1e-6).log()

    def _focal_from_log_probability(self, log_probability):
        probability = log_probability.exp()
        return -(1.0 - probability).pow(self.gamma) * log_probability

    def _full_loss(self, flat0, flat1, positives):
        b_ids, i_ids, j_ids = positives
        similarity = torch.einsum("bic,bjc->bij", flat0, flat1) / self.temperature
        # Focal/log operations are promoted to FP32 by autocast; accumulate in
        # FP32 as well to avoid lossy half-precision reductions.
        losses = torch.zeros(b_ids.numel(), device=flat0.device, dtype=torch.float32)
        for batch_index in b_ids.unique(sorted=True):
            mask = b_ids == batch_index
            current_i = i_ids[mask]
            current_j = j_ids[mask]
            unique_i, inverse_i = current_i.unique(sorted=True, return_inverse=True)
            unique_j, inverse_j = current_j.unique(sorted=True, return_inverse=True)
            if self.stable_log:
                row_log = F.log_softmax(
                    similarity[batch_index, unique_i].float(), dim=1
                )[inverse_i, current_j]
                column_log = F.log_softmax(
                    similarity[batch_index, :, unique_j].float(), dim=0
                )[current_i, inverse_j]
                losses[mask] = self._focal_from_log_probability(row_log + column_log)
            else:
                row_probability = F.softmax(similarity[batch_index, unique_i], dim=1)
                column_probability = F.softmax(similarity[batch_index, :, unique_j], dim=0)
                probability = (
                    row_probability[inverse_i, current_j]
                    * column_probability[current_i, inverse_j]
                )
                losses[mask] = self._focal(probability)
        return losses.mean()

    def _chunk_loss(self, flat0, flat1, positives):
        b_ids, i_ids, j_ids = positives
        total = flat0.new_zeros(())
        count = 0
        for batch_index in b_ids.unique(sorted=True):
            mask = b_ids == batch_index
            current_i = i_ids[mask]
            current_j = j_ids[mask]
            for start in range(0, current_i.numel(), self.chunk_size):
                chunk_i = current_i[start : start + self.chunk_size]
                chunk_j = current_j[start : start + self.chunk_size]

                def compute(source, target, temperature, source_indices, target_indices):
                    row_logits = source[source_indices] @ target.transpose(0, 1) / temperature
                    column_logits = source @ target[target_indices].transpose(0, 1) / temperature
                    indices = torch.arange(source_indices.numel(), device=source.device)
                    if self.stable_log:
                        row_log = F.log_softmax(row_logits.float(), dim=1)[
                            indices, target_indices
                        ]
                        column_log = F.log_softmax(column_logits.float(), dim=0)[
                            source_indices, indices
                        ]
                        return self._focal_from_log_probability(row_log + column_log).sum()
                    row_probability = F.softmax(row_logits, dim=1)[indices, target_indices]
                    column_probability = F.softmax(column_logits, dim=0)[
                        source_indices, indices
                    ]
                    return self._focal(row_probability * column_probability).sum()

                source = flat0[batch_index]
                target = flat1[batch_index]
                if torch.is_grad_enabled() and (source.requires_grad or target.requires_grad):
                    chunk_sum = checkpoint(
                        compute,
                        source,
                        target,
                        self.temperature,
                        chunk_i,
                        chunk_j,
                        use_reentrant=False,
                    )
                else:
                    chunk_sum = compute(
                        source, target, self.temperature, chunk_i, chunk_j
                    )
                total = total + chunk_sum
                count += chunk_i.numel()
        return total / max(count, 1)

    def forward(self, descriptor0, descriptor1, correspondences, mode="full", selected=None):
        flat0 = flatten_descriptors(descriptor0)
        flat1 = flatten_descriptors(descriptor1)
        positives = self.select_positives(
            correspondences,
            device=flat0.device,
            selected=selected,
        )
        if mode == "full":
            return self._full_loss(flat0, flat1, positives)
        if mode == "chunked":
            return self._chunk_loss(flat0, flat1, positives)
        raise ValueError(f"Unknown similarity mode: {mode}")
