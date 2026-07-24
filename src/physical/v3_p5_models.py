"""Trainable P5 controls for the Physical V3 local-encoder ablation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .v3_himo import FrozenHIMO, HIMOConfig


IMPLEMENTATION_VERSION = "V3.0.3"


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, value):
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


def _masked_median(value, mask):
    medians = []
    for batch_index in range(value.shape[0]):
        selected = value[batch_index][mask[batch_index]]
        if selected.numel() == 0:
            selected = value[batch_index].flatten()
        medians.append(selected.float().median())
    return torch.stack(medians).to(value.dtype)[:, None, None, None]


def build_p5_state(himo_output, clip_max=5.0):
    odd = himo_output["magnitude_odd"].float().clamp_min(0).pow(0.25)
    even = himo_output["magnitude_even"].float().clamp_min(0).pow(0.25)
    valid = himo_output["weight"].float() > 0.5
    odd = (odd / _masked_median(odd, valid).clamp_min(1e-6)).clamp(0, clip_max)
    even = (even / _masked_median(even, valid).clamp_min(1e-6)).clamp(0, clip_max)
    orientation = himo_output["orientation"].float()
    cosine = torch.cos(2.0 * orientation)
    sine = torch.sin(2.0 * orientation)
    cosine = torch.where(valid, cosine, torch.ones_like(cosine))
    sine = torch.where(valid, sine, torch.zeros_like(sine))
    relative = (odd - even).abs() / (odd + even).clamp_min(1e-6)
    state = torch.cat([cosine, sine, odd, even, relative], dim=1)
    return state.detach().clone()


def sample_coarse_centers(state, coarse_scale=8):
    offset = int(coarse_scale) // 2
    sampled = state[
        :,
        :,
        offset::coarse_scale,
        offset::coarse_scale,
    ]
    expected_height = state.shape[-2] // coarse_scale
    expected_width = state.shape[-1] // coarse_scale
    return sampled[:, :, :expected_height, :expected_width].contiguous()


class PointwiseP5Head(nn.Module):
    def __init__(self, output_dim=128, hidden_dim=48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(5, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
            LayerNorm2d(output_dim),
        )

    def forward(self, state):
        return F.normalize(self.network(state), p=2, dim=1, eps=1e-6)


class RectConvP5Head(nn.Module):
    def __init__(self, output_dim=128, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(5, hidden_dim, kernel_size=5),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
            LayerNorm2d(output_dim),
        )

    def forward(self, state):
        return F.normalize(self.network(state), p=2, dim=1, eps=1e-6)


class PhysicalV3P5Encoder(nn.Module):
    MODEL_NAMES = ("pointwise_p5", "rectconv_p5")

    def __init__(
        self,
        model_name,
        output_dim=128,
        coarse_scale=8,
        max_cooccurrence_levels=8192,
    ):
        super().__init__()
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Unknown P5 model: {model_name}")
        self.model_name = model_name
        self.coarse_scale = int(coarse_scale)
        self.himo = FrozenHIMO(
            HIMOConfig(max_cooccurrence_levels=max_cooccurrence_levels)
        )
        self.head = (
            PointwiseP5Head(output_dim=output_dim)
            if model_name == "pointwise_p5"
            else RectConvP5Head(output_dim=output_dim)
        )

    def train(self, mode=True):
        super().train(mode)
        self.himo.eval()
        return self

    def forward(self, image):
        self.himo.eval()
        with torch.autocast(device_type=image.device.type, enabled=False):
            himo_output = self.himo(image.float())
            state = build_p5_state(himo_output)
        state = sample_coarse_centers(state, self.coarse_scale)
        return self.head(state)

    @property
    def trainable_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_physical_v3_p5_encoder(model_name, **kwargs):
    return PhysicalV3P5Encoder(model_name=model_name, **kwargs)


__all__ = [
    "IMPLEMENTATION_VERSION",
    "PhysicalV3P5Encoder",
    "PointwiseP5Head",
    "RectConvP5Head",
    "build_p5_state",
    "build_physical_v3_p5_encoder",
    "sample_coarse_centers",
]
