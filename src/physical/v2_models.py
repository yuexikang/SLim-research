import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from default_config import get_config
from src.slim import SLiM
from src.utils.misc import LayerNorm2d

from .v1_models import (
    ContinuousOrientationCanonicalization,
    FixedBlurPool,
    FixedLocalDivisiveNormalization,
    ParametricSteerableGaborBank,
)


def _autocast_disabled(tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def safe_axial_angle(orientation, eps=1e-3):
    """Convert doubled-angle vectors to an unsigned angle without atan2(0, 0)."""
    x = orientation[:, 0].float()
    y = orientation[:, 1].float()
    valid = (x.square() + y.square()) >= float(eps) ** 2
    safe_x = torch.where(valid, x, torch.ones_like(x))
    safe_y = torch.where(valid, y, torch.zeros_like(y))
    return 0.5 * torch.atan2(safe_y, safe_x)


class FrozenSLiMCoarseExtractor(nn.Module):
    """Frozen official SLiM backbone restricted to the 1/8 coarse feature."""

    def __init__(self, checkpoint_path="ckpt/megadepth_19epochs.ckpt"):
        super().__init__()
        config = get_config("outdoor_test")
        self.slim = SLiM(config.MODEL)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.slim.load_state_dict(checkpoint["state_dict"], strict=True)
        self.coarse_scale_idx = int(config.MODEL.COARSE_SCALE_IDX)
        self.checkpoint_path = str(checkpoint_path)
        self._initialized = False
        self.requires_grad_(False)
        self.slim.eval()

    @property
    def temperature(self):
        return self.slim.coarse_temperature.detach()

    def train(self, mode=True):
        super().train(False)
        self.slim.eval()
        return self

    @torch.no_grad()
    def forward(self, image0, image1):
        self.slim.eval()
        if not self._initialized:
            self.slim.initial_forward()
            self._initialized = True
        features0, features1 = self.slim.feature_backbone(image0, image1)
        return (
            features0[self.coarse_scale_idx].detach(),
            features1[self.coarse_scale_idx].detach(),
        )


class SharedMASW(nn.Module):
    """Fixed multi-area structure window in doubled-angle space."""

    def __init__(self, sigmas=(1.0, 2.0, 4.0), eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        for index, sigma in enumerate(sigmas):
            radius = int(math.ceil(3.0 * float(sigma)))
            coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-coordinates.square() / (2.0 * float(sigma) ** 2))
            kernel = torch.outer(kernel, kernel)
            kernel = kernel / kernel.sum()
            self.register_buffer(f"kernel_{index}", kernel[None, None], persistent=True)
        self.num_kernels = len(sigmas)

    def _smooth(self, field, kernel):
        radius = kernel.shape[-1] // 2
        return F.conv2d(F.pad(field, (radius,) * 4, mode="reflect"), kernel)

    def forward(self, gx, gy):
        with _autocast_disabled(gx):
            gx = gx.float()
            gy = gy.float()
            x = gx.square() - gy.square()
            y = 2.0 * gx * gy
            a = 0.0
            b = 0.0
            for index in range(self.num_kernels):
                kernel = getattr(self, f"kernel_{index}").to(gx.device)
                a = a + self._smooth(x, kernel)
                b = b + self._smooth(y, kernel)
            a = a / self.num_kernels
            b = b / self.num_kernels
        return a, b

    def fields(self, a, b):
        magnitude = torch.sqrt(a.square() + b.square() + self.eps)
        orientation = torch.cat([a, b], dim=1) / magnitude.clamp_min(self.eps)
        return magnitude, orientation


def fixed_2d_sincos(height, width, channels, device, dtype):
    if channels % 4:
        raise ValueError("Position encoding channels must be divisible by four.")
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    frequencies = torch.exp(
        torch.arange(channels // 4, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(channels // 4 - 1, 1))
    )
    position = torch.cat(
        [
            torch.sin(x[..., None] / frequencies),
            torch.cos(x[..., None] / frequencies),
            torch.sin(y[..., None] / frequencies),
            torch.cos(y[..., None] / frequencies),
        ],
        dim=-1,
    )
    return position.reshape(1, height * width, channels).to(dtype=dtype)


class ReliabilityLinearAttention(nn.Module):
    def __init__(self, channels=96, eps=1e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.norm_query = nn.LayerNorm(channels)
        self.norm_source = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.update_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    @staticmethod
    def feature_map(feature):
        return F.elu(feature) + 1.0

    def forward(self, query, source, source_reliability, query_pe, source_pe):
        output_dtype = query.dtype
        with _autocast_disabled(query):
            query_fp32 = query.float()
            source_fp32 = source.float()
            query_pe = query_pe.float()
            source_pe = source_pe.float()
            normalized_query = self.norm_query(query_fp32)
            normalized_source = self.norm_source(source_fp32)
            q = self.feature_map(self.q_proj(normalized_query + query_pe))
            k = self.feature_map(self.k_proj(normalized_source + source_pe))
            value = self.v_proj(normalized_source)
            reliability = source_reliability.detach().float().clamp(0.0, 1.0)
            weighted_key = k * reliability
            key_value = torch.einsum("blc,bld->bcd", weighted_key, value)
            key_sum = weighted_key.sum(dim=1)
            denominator = torch.einsum("blc,bc->bl", q, key_sum).clamp_min(self.eps)
            attended = torch.einsum("blc,bcd->bld", q, key_value)
            attended = attended / denominator[..., None]
            update = self.out_proj(attended)
            update = update + self.mlp(self.update_norm(update))
            output = query_fp32 + update
        return output.to(output_dtype)


class PairInteractionRound(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.self_layer = ReliabilityLinearAttention(channels)
        self.cross_layer = ReliabilityLinearAttention(channels)

    def forward(self, feature0, feature1, reliability0, reliability1, pe):
        old0, old1 = feature0, feature1
        self0 = self.self_layer(old0, old0, reliability0, pe, pe)
        self1 = self.self_layer(old1, old1, reliability1, pe, pe)
        old0, old1 = self0, self1
        cross0 = self.cross_layer(old0, old1, reliability1, pe, pe)
        cross1 = self.cross_layer(old1, old0, reliability0, pe, pe)
        return cross0, cross1


class LinearPairTransformer(nn.Module):
    def __init__(self, channels=96, rounds=2):
        super().__init__()
        self.rounds = nn.ModuleList(
            [PairInteractionRound(channels) for _ in range(int(rounds))]
        )

    def forward(self, feature0, feature1, reliability0, reliability1):
        height, width = feature0.shape[-2:]
        flat0 = feature0.flatten(2).transpose(1, 2).contiguous()
        flat1 = feature1.flatten(2).transpose(1, 2).contiguous()
        rel0 = reliability0.flatten(2).transpose(1, 2).contiguous()
        rel1 = reliability1.flatten(2).transpose(1, 2).contiguous()
        pe = fixed_2d_sincos(
            height, width, flat0.shape[-1], flat0.device, flat0.dtype
        )
        for interaction in self.rounds:
            flat0, flat1 = interaction(flat0, flat1, rel0, rel1, pe)
        output0 = flat0.transpose(1, 2).reshape_as(feature0)
        output1 = flat1.transpose(1, 2).reshape_as(feature1)
        return output0, output1


class DensePolarDescriptor(nn.Module):
    def __init__(self, channels=96, heads=4, chunk_size=1024):
        super().__init__()
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.ring_embedding = nn.Embedding(3, channels)
        self.sector_embedding = nn.Embedding(9, channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fusion = nn.Linear(channels * 2, channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        nn.init.normal_(self.cls_token, std=0.02)
        ring_ids = [0] + [1] * 8 + [2] * 8
        sector_ids = [8] + list(range(8)) + list(range(8))
        self.register_buffer("ring_ids", torch.tensor(ring_ids), persistent=True)
        self.register_buffer("sector_ids", torch.tensor(sector_ids), persistent=True)

    def _sample_state(self, feature, phi, state_offset):
        batch, channels, height, width = feature.shape
        total = height * width
        flat_phi = phi.reshape(batch, total)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=torch.float32),
            torch.arange(width, device=feature.device, dtype=torch.float32),
            indexing="ij",
        )
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        sector = torch.arange(8, device=feature.device, dtype=torch.float32) * (
            2.0 * math.pi / 8.0
        )
        outputs = []
        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            current = end - start
            anchor_x = flat_x[start:end][None, :, None].expand(batch, -1, -1)
            anchor_y = flat_y[start:end][None, :, None].expand(batch, -1, -1)
            angle = flat_phi[:, start:end, None] + float(state_offset) + sector
            sample_x = [anchor_x]
            sample_y = [anchor_y]
            for radius in (2.0, 4.0):
                sample_x.append(anchor_x + radius * angle.cos())
                sample_y.append(anchor_y + radius * angle.sin())
            sample_x = torch.cat(sample_x, dim=2)
            sample_y = torch.cat(sample_y, dim=2)
            grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
            grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).to(feature.dtype)
            sampled = F.grid_sample(
                feature,
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            tokens = sampled.permute(0, 2, 3, 1).reshape(batch * current, 17, channels)
            positional = self.ring_embedding(self.ring_ids) + self.sector_embedding(
                self.sector_ids
            )
            tokens = tokens + positional[None].to(tokens.dtype)
            cls = self.cls_token.to(tokens.dtype).expand(tokens.shape[0], -1, -1)
            encoded = self.transformer(torch.cat([cls, tokens], dim=1))[:, 0]
            outputs.append(encoded.reshape(batch, current, channels))
        return torch.cat(outputs, dim=1)

    def reversal_fuse(self, descriptor0, descriptor_pi):
        symmetric = descriptor0 + descriptor_pi
        antisymmetric = (descriptor0 - descriptor_pi).abs()
        return self.norm(self.fusion(torch.cat([symmetric, antisymmetric], dim=-1)))

    def forward(self, feature, orientation):
        phi = safe_axial_angle(orientation)
        descriptor0 = self._sample_state(feature, phi, 0.0)
        descriptor_pi = self._sample_state(feature, phi, math.pi)
        fused = self.reversal_fuse(descriptor0, descriptor_pi)
        batch, _, height, width = feature.shape
        fused = fused.transpose(1, 2).reshape(batch, self.channels, height, width)
        return F.normalize(fused, p=2, dim=1, eps=1e-8)


class PhysicalEncoderV2(nn.Module):
    MODEL_CONFIGS = {
        "physical_v2_core": {},
        "physical_v2_no_recovery_weight": {"recovery_weighting": False},
        "physical_v2_no_pair_transformer": {"enable_pair_transformer": False},
        "physical_v2_no_polar": {"enable_polar": False},
        "physical_v2_soft_oe": {"oe_mode": "soft"},
        "physical_v2_single_scale_512": {"single_scale": True},
    }

    def __init__(
        self,
        channels=96,
        recovery_weighting=True,
        enable_pair_transformer=True,
        enable_polar=True,
        oe_mode="hard",
        single_scale=False,
        polar_chunk_size=1024,
    ):
        super().__init__()
        if oe_mode not in {"hard", "soft"}:
            raise ValueError(f"Unknown Odd/Even coupling mode: {oe_mode}")
        self.channels = int(channels)
        self.recovery_weighting = bool(recovery_weighting)
        self.enable_pair_transformer = bool(enable_pair_transformer)
        self.enable_polar = bool(enable_polar)
        self.oe_mode = str(oe_mode)
        self.single_scale = bool(single_scale)
        self.ldn = FixedLocalDivisiveNormalization()
        self.gabor = ParametricSteerableGaborBank(response_fp32=True)
        self.masw = SharedMASW()
        self.blur_pool = FixedBlurPool(1)
        self.canonicalizer = ContinuousOrientationCanonicalization()
        self.token_projection = nn.Sequential(
            nn.Conv2d(11, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.odd_transformer = (
            LinearPairTransformer(channels, rounds=2) if self.enable_pair_transformer else None
        )
        self.even_transformer = (
            LinearPairTransformer(channels, rounds=2) if self.enable_pair_transformer else None
        )
        self.polar = (
            DensePolarDescriptor(channels, heads=4, chunk_size=polar_chunk_size)
            if self.enable_polar
            else None
        )
        self.no_polar_head = (
            None
            if self.enable_polar
            else nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                LayerNorm2d(channels),
            )
        )
        self.adapter_norm = LayerNorm2d(channels)
        self.adapter = nn.Conv2d(channels, 192, 1, bias=False)
        nn.init.zeros_(self.adapter.weight)
        theta = torch.arange(8, dtype=torch.float32) * (math.pi / 8.0)
        self.register_buffer("cos_theta", theta.cos()[None, None, :, None, None])
        self.register_buffer("sin_theta", theta.sin()[None, None, :, None, None])

    def physical_filter_parameters(self):
        return [parameter for parameter in self.gabor.physical_parameters() if parameter.requires_grad]

    def _downsample(self, field, steps):
        for _ in range(int(steps)):
            field = self.blur_pool(field)
        return field

    def _image_pyramid(self, image):
        half = self.blur_pool(image)
        quarter = self.blur_pool(half)
        return (image, half, quarter)

    def _canonical_spectrum(self, spectrum, orientation):
        theta = safe_axial_angle(orientation)
        spectrum = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(1e-6)
        canonical = self.canonicalizer(
            spectrum[:, None], theta, torch.ones_like(theta[:, None])
        )
        return canonical[:, 0]

    def _physical_scale(self, image, scale_index):
        normalized = self.ldn(image)
        even, odd, amplitude = self.gabor(normalized)
        sign = torch.where(odd.detach() >= 0, torch.ones_like(odd), -torch.ones_like(odd))
        corrected_even = even * sign
        gx_odd = (odd * self.cos_theta.to(odd.dtype)).sum(dim=(1, 2), keepdim=False)[:, None]
        gy_odd = (odd * self.sin_theta.to(odd.dtype)).sum(dim=(1, 2), keepdim=False)[:, None]
        gx_even = (
            corrected_even * self.cos_theta.to(corrected_even.dtype)
        ).sum(dim=(1, 2), keepdim=False)[:, None]
        gy_even = (
            corrected_even * self.sin_theta.to(corrected_even.dtype)
        ).sum(dim=(1, 2), keepdim=False)[:, None]
        a_odd, b_odd = self.masw(gx_odd, gy_odd)
        a_even, b_even = self.masw(gx_even, gy_even)
        spectrum_odd = odd.abs().sum(dim=1)
        spectrum_even = corrected_even.abs().sum(dim=1)
        coherent = (
            torch.sqrt(
                even.float().sum(dim=1).square()
                + odd.float().sum(dim=1).square()
                + 1e-6
            )
            - math.sqrt(1e-6)
        ).clamp_min(0.0)
        amplitude_sum = amplitude.float().sum(dim=1).clamp_min(1e-6)
        # Amplitude-weighted mean of per-orientation phase coherence:
        # sum_k ((coherent_k / amplitude_k) * amplitude_k) / sum_k amplitude_k.
        phase_numerator = coherent.sum(dim=1, keepdim=True)
        phase_denominator = amplitude_sum.sum(dim=1, keepdim=True)
        steps = 3 - int(scale_index)
        fields = [
            a_odd,
            b_odd,
            a_even,
            b_even,
            spectrum_odd,
            spectrum_even,
            phase_numerator,
            phase_denominator,
        ]
        fields = [self._downsample(field, steps) for field in fields]
        (
            a_odd,
            b_odd,
            a_even,
            b_even,
            spectrum_odd,
            spectrum_even,
            phase_numerator,
            phase_denominator,
        ) = fields
        magnitude_odd, orientation_odd = self.masw.fields(a_odd, b_odd)
        magnitude_even, orientation_even = self.masw.fields(a_even, b_even)
        spectrum_odd = self._canonical_spectrum(spectrum_odd, orientation_odd)
        spectrum_even = self._canonical_spectrum(spectrum_even, orientation_even)
        total = magnitude_odd + magnitude_even
        local = F.avg_pool2d(total, 9, stride=1, padding=4)
        normalized_strength = (total / local.clamp_min(1e-6)).clamp(0.0, 10.0)
        strength = normalized_strength / (1.0 + normalized_strength)
        phase = (phase_numerator / phase_denominator.clamp_min(1e-6)).clamp(0.0, 1.0)
        concentration_odd = magnitude_odd / (
            magnitude_odd + F.avg_pool2d(magnitude_odd, 9, stride=1, padding=4)
        ).clamp_min(1e-6)
        concentration_even = magnitude_even / (
            magnitude_even + F.avg_pool2d(magnitude_even, 9, stride=1, padding=4)
        ).clamp_min(1e-6)
        concentration = torch.maximum(concentration_odd, concentration_even).clamp(0.0, 1.0)
        reliability = (phase * concentration * strength).clamp(0.0, 1.0)
        ratio_odd = magnitude_odd / total.clamp_min(1e-6)
        ratio_even = magnitude_even / total.clamp_min(1e-6)
        odd_input = torch.cat([ratio_odd, strength, reliability, spectrum_odd], dim=1)
        even_input = torch.cat([ratio_even, strength, reliability, spectrum_even], dim=1)
        return {
            "odd": self.token_projection(odd_input.contiguous()),
            "even": self.token_projection(even_input.contiguous()),
            "magnitude_odd": magnitude_odd,
            "magnitude_even": magnitude_even,
            "orientation_odd": orientation_odd,
            "orientation_even": orientation_even,
            "reliability": reliability,
        }

    def _physical_fields(self, image):
        pyramid = self._image_pyramid(image)
        count = 1 if self.single_scale else 3
        return [self._physical_scale(pyramid[index], index) for index in range(count)]

    def _couple(self, odd, even, fields):
        denominator = (fields["magnitude_odd"] + fields["magnitude_even"]).clamp_min(1e-6)
        soft = fields["magnitude_odd"] / denominator
        selector = (fields["magnitude_odd"] >= fields["magnitude_even"]).to(odd.dtype)
        weight = selector if self.oe_mode == "hard" else soft.to(odd.dtype)
        feature = weight * odd + (1.0 - weight) * even
        orientation = weight.float() * fields["orientation_odd"] + (
            1.0 - weight.float()
        ) * fields["orientation_even"]
        orientation = F.normalize(orientation, p=2, dim=1, eps=1e-6)
        magnitude = weight.float() * fields["magnitude_odd"] + (
            1.0 - weight.float()
        ) * fields["magnitude_even"]
        return feature, orientation, magnitude, selector

    def _fuse_scales(self, scales):
        qualities = []
        for scale in scales:
            local = F.avg_pool2d(scale["magnitude"], 9, stride=1, padding=4)
            normalized = (scale["magnitude"] / local.clamp_min(1e-6)).clamp(0.0, 10.0)
            bounded = normalized / (1.0 + normalized)
            qualities.append((bounded * scale["reliability"]).detach())
        quality = torch.cat(qualities, dim=1)
        weights = (quality + 1e-6) / (quality + 1e-6).sum(dim=1, keepdim=True)
        feature = sum(weights[:, i : i + 1] * scale["feature"] for i, scale in enumerate(scales))
        orientation = sum(
            weights[:, i : i + 1].float() * scale["orientation"]
            for i, scale in enumerate(scales)
        )
        orientation = F.normalize(orientation, p=2, dim=1, eps=1e-6)
        reliability = sum(
            weights[:, i : i + 1] * scale["reliability"]
            for i, scale in enumerate(scales)
        )
        if len(scales) == 1:
            full_weights = torch.zeros(
                weights.shape[0], 3, *weights.shape[-2:], device=weights.device, dtype=weights.dtype
            )
            full_weights[:, 0] = 1.0
        else:
            full_weights = weights
        selectors = torch.cat([scale["selector"] for scale in scales], dim=1)
        if len(scales) == 1:
            selectors = selectors.expand(-1, 3, -1, -1).contiguous()
        return feature, orientation, reliability, full_weights, selectors

    def forward_pair(self, image0, image1):
        fields0 = self._physical_fields(image0)
        fields1 = self._physical_fields(image1)
        scales0 = []
        scales1 = []
        unary0 = []
        unary1 = []
        for current0, current1 in zip(fields0, fields1):
            unary0.extend([F.normalize(current0["odd"], p=2, dim=1), F.normalize(current0["even"], p=2, dim=1)])
            unary1.extend([F.normalize(current1["odd"], p=2, dim=1), F.normalize(current1["even"], p=2, dim=1)])
            if self.enable_pair_transformer:
                odd0, odd1 = self.odd_transformer(
                    current0["odd"], current1["odd"], current0["reliability"], current1["reliability"]
                )
                even0, even1 = self.even_transformer(
                    current0["even"], current1["even"], current0["reliability"], current1["reliability"]
                )
            else:
                odd0, odd1 = current0["odd"], current1["odd"]
                even0, even1 = current0["even"], current1["even"]
            for target, odd, even, fields in (
                (scales0, odd0, even0, current0),
                (scales1, odd1, even1, current1),
            ):
                feature, orientation, magnitude, selector = self._couple(odd, even, fields)
                target.append(
                    {
                        "feature": feature,
                        "orientation": orientation,
                        "magnitude": magnitude,
                        "reliability": fields["reliability"],
                        "selector": selector,
                    }
                )
        fused0 = self._fuse_scales(scales0)
        fused1 = self._fuse_scales(scales1)
        output = []
        for fused, unary in ((fused0, unary0), (fused1, unary1)):
            feature, orientation, reliability, scale_weights, selectors = fused
            physical = (
                self.polar(feature, orientation)
                if self.enable_polar
                else F.normalize(self.no_polar_head(feature.contiguous()), p=2, dim=1, eps=1e-8)
            )
            delta = self.adapter(self.adapter_norm(physical).contiguous())
            output.append(
                {
                    "physical": physical,
                    "delta": delta,
                    "orientation": orientation,
                    "confidence": reliability,
                    "reliability": reliability,
                    "scale_weights": scale_weights,
                    "oe_selector": selectors,
                    "unary": unary,
                }
            )
        return output[0], output[1]


def build_physical_v2_encoder(model_name="physical_v2_core", **overrides):
    model_name = str(model_name).lower()
    if model_name not in PhysicalEncoderV2.MODEL_CONFIGS:
        choices = ", ".join(PhysicalEncoderV2.MODEL_CONFIGS)
        raise ValueError(f"Unknown Physical V2 model: {model_name}. Choices: {choices}")
    config = {**PhysicalEncoderV2.MODEL_CONFIGS[model_name], **overrides}
    return PhysicalEncoderV2(**config)


__all__ = [
    "DensePolarDescriptor",
    "FrozenSLiMCoarseExtractor",
    "LinearPairTransformer",
    "PhysicalEncoderV2",
    "SharedMASW",
    "build_physical_v2_encoder",
    "fixed_2d_sincos",
]
