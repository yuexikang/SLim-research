from collections import defaultdict

import torch
from torch.nn import functional as F

from .matching import flatten_descriptors


@torch.no_grad()
def descriptor_statistics(
    descriptor0,
    descriptor1,
    correspondences,
    temperature,
    variants=None,
    negative_radius=1,
    chunk_size=256,
    scale_by_sqrt_dim=False,
):
    flat0 = flatten_descriptors(descriptor0)
    flat1 = flatten_descriptors(descriptor1)
    if scale_by_sqrt_dim:
        scale = flat0.shape[-1] ** 0.5
        flat0 = flat0 / scale
        flat1 = flat1 / scale
    b_ids, i_ids, j_ids = correspondences
    width = descriptor1.shape[-1]
    height = descriptor1.shape[-2]
    results = defaultdict(
        lambda: {
            "count": 0.0,
            "correct0": 0.0,
            "correct1": 0.0,
            "positive": 0.0,
            "hard_negative": 0.0,
            "margin": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
        }
    )

    for batch_index in range(flat0.shape[0]):
        mask = b_ids == batch_index
        source_ids = i_ids[mask]
        target_ids = j_ids[mask]
        variant = str(variants[batch_index]) if variants is not None else "all"
        if source_ids.numel() == 0:
            continue
        for start in range(0, source_ids.numel(), chunk_size):
            current_i = source_ids[start : start + chunk_size]
            current_j = target_ids[start : start + chunk_size]
            cosine = flat0[batch_index, current_i] @ flat1[batch_index].transpose(0, 1)
            predicted = cosine.argmax(dim=1)
            pred_x, pred_y = predicted % width, predicted // width
            gt_x, gt_y = current_j % width, current_j // width
            distance = torch.maximum((pred_x - gt_x).abs(), (pred_y - gt_y).abs())
            positive = cosine[
                torch.arange(current_i.numel(), device=cosine.device), current_j
            ]

            negative = cosine.clone()
            row = torch.arange(current_i.numel(), device=cosine.device)
            for dy in range(-negative_radius, negative_radius + 1):
                for dx in range(-negative_radius, negative_radius + 1):
                    x = gt_x + dx
                    y = gt_y + dy
                    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                    negative[row[valid], y[valid] * width + x[valid]] = -torch.inf
            hard_negative = negative.max(dim=1).values
            log_probability = F.log_softmax(cosine / temperature.clamp_min(1e-4), dim=1)
            entropy = -(log_probability.exp() * log_probability).sum(dim=1)
            normalized_entropy = entropy / torch.log(
                torch.tensor(cosine.shape[1], device=cosine.device, dtype=cosine.dtype)
            )

            for key in ("all", variant):
                result = results[key]
                result["count"] += float(current_i.numel())
                result["correct0"] += float((distance == 0).sum())
                result["correct1"] += float((distance <= 1).sum())
                result["positive"] += float(positive.sum())
                result["hard_negative"] += float(hard_negative.sum())
                result["margin"] += float((positive - hard_negative).sum())
                result["entropy"] += float(entropy.sum())
                result["normalized_entropy"] += float(normalized_entropy.sum())
    return dict(results)


@torch.no_grad()
def nearest_neighbors(descriptor0, descriptor1, chunk_size=256):
    flat0 = flatten_descriptors(descriptor0)
    flat1 = flatten_descriptors(descriptor1)
    if flat0.shape[0] != 1:
        raise ValueError("nearest_neighbors currently expects batch size 1.")

    def directional(source, target):
        indices = []
        scores = []
        for start in range(0, source.shape[0], chunk_size):
            similarity = source[start : start + chunk_size] @ target.transpose(0, 1)
            value, index = similarity.max(dim=1)
            indices.append(index)
            scores.append(value)
        return torch.cat(indices), torch.cat(scores)

    index01, score01 = directional(flat0[0], flat1[0])
    index10, _ = directional(flat1[0], flat0[0])
    source = torch.arange(index01.numel(), device=index01.device)
    mutual = index10[index01] == source
    return source[mutual], index01[mutual], score01[mutual]
