import torch
from torch.nn import functional as F


def flatten_feature(feature):
    return feature.flatten(2).transpose(1, 2).contiguous()


def slim_scaled_features(feature):
    flat = flatten_feature(feature)
    return flat / (flat.shape[-1] ** 0.5)


def positive_dual_softmax(
    feature0,
    feature1,
    correspondences,
    temperature,
    chunk_size=256,
    slim_scaling=True,
    return_log=False,
):
    flat0 = slim_scaled_features(feature0) if slim_scaling else flatten_feature(feature0)
    flat1 = slim_scaled_features(feature1) if slim_scaling else flatten_feature(feature1)
    b_ids, i_ids, j_ids = correspondences
    probabilities = torch.empty(
        b_ids.numel(), device=flat0.device, dtype=torch.float32
    )
    log_probabilities = torch.empty_like(probabilities)
    temperature = torch.as_tensor(temperature, device=flat0.device).clamp_min(1e-4)
    for batch_index in b_ids.unique(sorted=True):
        batch_mask = b_ids == batch_index
        output_indices = torch.where(batch_mask)[0]
        current_i = i_ids[batch_mask]
        current_j = j_ids[batch_mask]
        source = flat0[batch_index]
        target = flat1[batch_index]
        for start in range(0, current_i.numel(), int(chunk_size)):
            end = start + int(chunk_size)
            chunk_i = current_i[start:end]
            chunk_j = current_j[start:end]
            row_logits = source[chunk_i] @ target.transpose(0, 1) / temperature
            column_logits = source @ target[chunk_j].transpose(0, 1) / temperature
            indices = torch.arange(chunk_i.numel(), device=flat0.device)
            row_log = F.log_softmax(row_logits.float(), dim=1)[indices, chunk_j]
            column_log = F.log_softmax(column_logits.float(), dim=0)[chunk_i, indices]
            current_log = row_log + column_log
            log_probabilities[output_indices[start:end]] = current_log
            probabilities[output_indices[start:end]] = current_log.exp()
    if return_log:
        return probabilities, log_probabilities
    return probabilities


def focal_positive_loss(probability, gamma=2.0, weight=None, log_probability=None):
    if log_probability is None:
        log_probability = probability.clamp_min(1e-6).log()
    loss = -(1.0 - probability).pow(float(gamma)) * log_probability
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def recovery_and_preservation_loss(
    base0,
    base1,
    enhanced0,
    enhanced1,
    correspondences,
    temperature,
    chunk_size=256,
    recovery_weighting=True,
    alpha=2.0,
    recovery_gamma=2.0,
    keep_quantile=0.30,
    keep_margin=0.02,
):
    with torch.no_grad():
        base_probability = positive_dual_softmax(
            base0,
            base1,
            correspondences,
            temperature,
            chunk_size=chunk_size,
            slim_scaling=True,
        )
    enhanced_probability, enhanced_log_probability = positive_dual_softmax(
        enhanced0,
        enhanced1,
        correspondences,
        temperature,
        chunk_size=chunk_size,
        slim_scaling=True,
        return_log=True,
    )
    if recovery_weighting:
        weights = 1.0 + float(alpha) * (1.0 - base_probability).pow(
            float(recovery_gamma)
        )
        weights = weights / weights.mean().clamp_min(1e-6)
    else:
        weights = torch.ones_like(base_probability)
    recover = focal_positive_loss(
        enhanced_probability,
        gamma=2.0,
        weight=weights,
        log_probability=enhanced_log_probability,
    )

    b_ids = correspondences[0]
    keep_losses = []
    for batch_index in b_ids.unique(sorted=True):
        mask = b_ids == batch_index
        count = max(1, int(round(mask.sum().item() * float(keep_quantile))))
        local_base = base_probability[mask]
        local_enhanced = enhanced_probability[mask]
        keep = torch.topk(local_base, k=min(count, local_base.numel())).indices
        keep_losses.append(
            F.relu(local_base[keep] - local_enhanced[keep] - float(keep_margin)).mean()
        )
    keep = torch.stack(keep_losses).mean()
    diagnostics = {
        "base_positive_confidence": base_probability.mean(),
        "enhanced_positive_confidence": enhanced_probability.mean(),
        "recovery_weight": weights.mean(),
    }
    return recover, keep, diagnostics


__all__ = [
    "focal_positive_loss",
    "flatten_feature",
    "positive_dual_softmax",
    "recovery_and_preservation_loss",
    "slim_scaled_features",
]
