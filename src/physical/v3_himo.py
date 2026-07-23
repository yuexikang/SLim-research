"""Frozen, source-traceable HIMO core for the Physical Encoder V3 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


HIMO_SOURCE_COMMIT = "884297aa36dfb9c89d9a7d5bf66c142bf8707a77"
HIMO_IMPLEMENTATION_VERSION = "3.0.0"


@dataclass(frozen=True)
class HIMOConfig:
    cooccurrence: bool = True
    sigma_s: float = 5.0
    sigma_oc: float = 1.6
    log_gabor_scales: int = 4
    log_gabor_orientations: int = 6
    min_wavelength: float = 3.0
    wavelength_multiplier: float = 1.6
    sigma_on_frequency: float = 0.75
    patch_size: int = 72
    spatial_bins: int = 12
    masw_scales: int = 4
    intensity_difference: bool = True
    validness_threshold: float = 0.1
    max_cooccurrence_levels: int = 2048


def _gaussian_kernel(
    size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    center = (size - 1) / 2.0
    coordinate = torch.arange(size, device=device, dtype=dtype) - center
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def _filter_replicate(image: Tensor, kernel: Tensor) -> Tensor:
    radius_y = kernel.shape[-2] // 2
    radius_x = kernel.shape[-1] // 2
    channels = image.shape[1]
    weight = kernel[None, None].repeat(channels, 1, 1, 1)
    padded = F.pad(image, (radius_x, radius_x, radius_y, radius_y), mode="replicate")
    return F.conv2d(padded, weight, groups=channels)


def _shift_with_zeros(image: Tensor, dy: int, dx: int) -> Tensor:
    height, width = image.shape
    result = torch.zeros_like(image)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    target_y0 = max(0, dy)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    target_x0 = max(0, dx)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)
    if source_y1 > source_y0 and source_x1 > source_x0:
        result[target_y0:target_y1, target_x0:target_x1] = image[
            source_y0:source_y1, source_x0:source_x1
        ]
    return result


def _cooccurrence_matrix(image: Tensor, sigma_oc: float, max_levels: int) -> Tensor:
    integer = torch.floor(image.clamp_min(0) + 0.5).to(torch.long)
    levels = int(integer.max().item()) + 1
    if levels > max_levels:
        raise ValueError(
            f"HIMO CoF needs {levels} gray levels, above the safety limit "
            f"{max_levels}. Check input normalization or raise max_cooccurrence_levels."
        )
    counts = torch.zeros(levels * levels, device=image.device, dtype=torch.float32)
    height, width = integer.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            y0 = max(0, -dy)
            y1 = min(height, height - dy)
            x0 = max(0, -dx)
            x1 = min(width, width - dx)
            neighbor_y0 = y0 + dy
            neighbor_y1 = y1 + dy
            neighbor_x0 = x0 + dx
            neighbor_x1 = x1 + dx
            source = integer[y0:y1, x0:x1]
            target = integer[neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1]
            index = source.reshape(-1) * levels + target.reshape(-1)
            counts += torch.bincount(index, minlength=levels * levels).float()

    matrix = counts.reshape(levels, levels)
    matrix = matrix / matrix.sum().clamp_min(torch.finfo(matrix.dtype).eps)
    if sigma_oc > 0:
        size = 2 * math.ceil(3 * sigma_oc) + 1
        kernel = _gaussian_kernel(
            size,
            sigma_oc,
            device=image.device,
            dtype=matrix.dtype,
        )
        matrix = _filter_replicate(matrix[None, None], kernel)[0, 0]
        matrix = matrix / matrix.sum().clamp_min(torch.finfo(matrix.dtype).eps)
    return matrix


def cooccurrence_filter(
    image: Tensor,
    sigma_s: float = 5.0,
    sigma_oc: float = 1.6,
    max_levels: int = 2048,
) -> Tensor:
    """Reproduce the inspectable CoOcurFilter.m path for one grayscale image."""
    if image.ndim != 2:
        raise ValueError(f"Expected [H,W], got {tuple(image.shape)}")
    shifted = image - image.min().clamp_max(0)
    integer = torch.floor(shifted + 0.5).to(torch.long)
    matrix = _cooccurrence_matrix(shifted, sigma_oc, max_levels)
    levels = matrix.shape[0]
    window_size = int(3 * sigma_s)
    radius = math.ceil(window_size / 2)
    coordinate = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    spatial = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma_s * sigma_s))

    numerator = torch.zeros_like(shifted, dtype=torch.float32)
    denominator = torch.zeros_like(shifted, dtype=torch.float32)
    center = integer.clamp(0, levels - 1)
    for row, dy in enumerate(range(-radius, radius + 1)):
        for column, dx in enumerate(range(-radius, radius + 1)):
            neighbor = _shift_with_zeros(integer, -dy, -dx).clamp(0, levels - 1)
            cooccurrence_weight = matrix[neighbor, center]
            weight = cooccurrence_weight * spatial[row, column]
            numerator += neighbor.float() * weight
            denominator += weight
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def _frequency_axis(length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if length % 2:
        return torch.arange(
            -(length - 1) // 2,
            (length - 1) // 2 + 1,
            device=device,
            dtype=dtype,
        ) / (length - 1)
    return torch.arange(
        -length // 2,
        length // 2,
        device=device,
        dtype=dtype,
    ) / length


def _log_gabor_filters(
    height: int,
    width: int,
    config: HIMOConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    x_range = _frequency_axis(width, device, dtype)
    y_range = _frequency_axis(height, device, dtype)
    y, x = torch.meshgrid(y_range, x_range, indexing="ij")
    radius = torch.fft.ifftshift(torch.sqrt(x.square() + y.square()))
    theta = torch.fft.ifftshift(torch.atan2(-y, x))
    radius = radius.clone()
    radius[0, 0] = 1.0
    lowpass = 1.0 / (1.0 + (radius / 0.45).pow(30))

    radial_filters = []
    for scale in range(config.log_gabor_scales):
        wavelength = config.min_wavelength * config.wavelength_multiplier**scale
        center_frequency = 1.0 / wavelength
        radial = torch.exp(
            -torch.log(radius / center_frequency).square()
            / (2.0 * math.log(config.sigma_on_frequency) ** 2)
        )
        radial = radial * lowpass
        radial[0, 0] = 0.0
        radial_filters.append(radial)

    orientation_filters = []
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    for orientation in range(config.log_gabor_orientations):
        angle = orientation * math.pi / config.log_gabor_orientations
        ds = sin_theta * math.cos(angle) - cos_theta * math.sin(angle)
        dc = cos_theta * math.cos(angle) + sin_theta * math.sin(angle)
        delta = torch.atan2(ds, dc).abs()
        delta = torch.minimum(
            delta * config.log_gabor_orientations / 2.0,
            torch.tensor(math.pi, device=device, dtype=dtype),
        )
        orientation_filters.append((torch.cos(delta) + 1.0) / 2.0)
    return torch.stack(radial_filters), torch.stack(orientation_filters)


def _sobel_responses(image: Tensor) -> Tuple[Tensor, Tensor]:
    odd = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    even = image.new_tensor([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) * (2.0 / 3.0)
    odd_x = _filter_replicate(image, odd)
    odd_y = _filter_replicate(image, odd.transpose(0, 1))
    sign_x = torch.where(odd_x >= 0, 1.0, -1.0)
    sign_y = torch.where(odd_y >= 0, 1.0, -1.0)
    even_x = _filter_replicate(image, even) * sign_x
    even_y = _filter_replicate(image, even.transpose(0, 1)) * sign_y
    return torch.complex(even_x, odd_x), torch.complex(even_y, odd_y)


def _deep_shallow_responses(image: Tensor, config: HIMOConfig) -> Tuple[Tensor, Tensor]:
    complex_x, complex_y = _sobel_responses(image)
    height, width = image.shape[-2:]
    radial, directional = _log_gabor_filters(
        height,
        width,
        config,
        device=image.device,
        dtype=image.dtype,
    )
    image_fft = torch.fft.fft2(image[:, 0])
    for orientation in range(config.log_gabor_orientations):
        angle = orientation * math.pi / config.log_gabor_orientations
        for scale in range(config.log_gabor_scales):
            response = torch.fft.ifft2(
                image_fft * radial[scale] * directional[orientation]
            )[:, None]
            direction = torch.where(response.imag >= 0, 1.0, -1.0)
            alienated = torch.complex(response.real * direction, response.imag)
            scale_weight = config.log_gabor_scales - scale
            complex_x = complex_x - alienated * math.cos(angle) * scale_weight
            complex_y = complex_y + alienated * math.sin(angle) * scale_weight
    odd = torch.cat([complex_x.imag, complex_y.imag], dim=1)
    even = torch.cat([complex_x.real, complex_y.real], dim=1)
    return odd, even


def _masw(response: Tensor, config: HIMOConfig) -> Tuple[Tensor, Tensor]:
    window_radius = math.floor(
        math.sqrt(
            (config.patch_size // 2) ** 2 / (2 * config.spatial_bins + 1)
        )
    )
    patch_size = 2 * window_radius + 1
    coordinate = torch.arange(
        -window_radius,
        window_radius + 1,
        device=response.device,
        dtype=response.dtype,
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    circle = (xx.square() + yy.square() < (window_radius + 1) ** 2).to(response.dtype)
    radius_1 = 1.0
    radius_2 = math.sqrt(
        (config.patch_size // 2) ** 2 / (2 * config.spatial_bins + 1)
    )
    step = (radius_2 - radius_1) / max(config.masw_scales - 1, 1)
    kernel = response.new_zeros((patch_size, patch_size))
    for index in range(config.masw_scales):
        sigma = (radius_1 + step * index) / 6.0
        kernel += _gaussian_kernel(
            patch_size,
            sigma,
            device=response.device,
            dtype=response.dtype,
        )
    kernel *= circle
    x, y = response[:, :1], response[:, 1:2]
    xx_stat = _filter_replicate(x.square(), kernel)
    yy_stat = _filter_replicate(y.square(), kernel)
    xy_stat = _filter_replicate(x * y, kernel)
    axis_x = xx_stat - yy_stat
    axis_y = 2.0 * xy_stat
    orientation = torch.atan2(axis_y, axis_x) / 2.0 + math.pi / 2.0
    magnitude = axis_x.square() + axis_y.square()
    return magnitude, orientation


def _validness(magnitude_odd: Tensor, magnitude_even: Tensor, config: HIMOConfig) -> Tensor:
    scale = 6
    radius = scale // 2
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=magnitude_odd.device,
        dtype=magnitude_odd.dtype,
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    circle = (xx.square() + yy.square() < (radius + 1) ** 2).to(magnitude_odd.dtype)
    kernel = _gaussian_kernel(
        scale + 1,
        scale / 6.0,
        device=magnitude_odd.device,
        dtype=magnitude_odd.dtype,
    )
    kernel *= circle
    cross = _filter_replicate(magnitude_odd * magnitude_even, kernel)
    odd_square = _filter_replicate(magnitude_odd.square(), kernel)
    even_square = _filter_replicate(magnitude_even.square(), kernel)
    score = (odd_square * even_square - cross.square()) / (
        odd_square + even_square + torch.finfo(magnitude_odd.dtype).eps
    )
    return (score > config.validness_threshold).to(magnitude_odd.dtype)


class FrozenHIMO(nn.Module):
    """Single-scale inspectable HIMO core used by the V3.0.0 no-training baseline."""

    def __init__(self, config: HIMOConfig | None = None):
        super().__init__()
        self.config = config or HIMOConfig()

    @torch.inference_mode()
    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(image.shape)}")
        image = image.float()
        normalized = []
        cofiltered = []
        for sample in image:
            plane = sample[0]
            nonzero = plane[plane != 0]
            median = (
                torch.quantile(nonzero, 0.5)
                if nonzero.numel()
                else plane.new_tensor(1.0)
            )
            scaled = plane * 255.0 / median.clamp_min(1e-6) / 2.0
            normalized.append(scaled)
            if self.config.cooccurrence:
                filtered = cooccurrence_filter(
                    scaled,
                    sigma_s=self.config.sigma_s,
                    sigma_oc=self.config.sigma_oc,
                    max_levels=self.config.max_cooccurrence_levels,
                )
                scaled = filtered * 0.25 + scaled * 0.75
            cofiltered.append(scaled)
        normalized_tensor = torch.stack(normalized)[:, None]
        filtered_tensor = torch.stack(cofiltered)[:, None]

        odd_response, even_response = _deep_shallow_responses(
            filtered_tensor, self.config
        )
        magnitude_odd, orientation_odd = _masw(odd_response, self.config)
        magnitude_even, orientation_even = _masw(even_response, self.config)
        choose_odd = magnitude_odd >= magnitude_even
        orientation = torch.where(choose_odd, orientation_odd, orientation_even)
        orientation = torch.remainder(orientation, math.pi)
        if self.config.intensity_difference:
            weight = _validness(magnitude_odd, magnitude_even, self.config)
        else:
            weight = torch.where(choose_odd, magnitude_odd, magnitude_even).pow(0.25)
        return {
            "normalized_image": normalized_tensor,
            "cofiltered_image": filtered_tensor,
            "odd_response": odd_response,
            "even_response": even_response,
            "magnitude_odd": magnitude_odd,
            "magnitude_even": magnitude_even,
            "orientation_odd": orientation_odd,
            "orientation_even": orientation_even,
            "orientation": orientation,
            "weight": weight,
            "choose_odd": choose_odd,
        }
