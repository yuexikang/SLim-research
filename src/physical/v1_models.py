import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d

from src.utils.misc import LayerNorm2d

from .models import count_trainable_parameters


def _autocast_disabled(tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


class FixedLocalDivisiveNormalization(nn.Module):
    def __init__(self, kernel_size=9, sigma=2.0, eps=1e-4, clip=5.0):
        super().__init__()
        radius = int(kernel_size) // 2
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-coordinates.square() / (2.0 * float(sigma) ** 2))
        kernel = torch.outer(kernel, kernel)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None], persistent=True)
        self.padding = radius
        self.eps = float(eps)
        self.clip = float(clip)

    def forward(self, image):
        padded = F.pad(image, (self.padding,) * 4, mode="reflect")
        kernel = self.kernel.to(device=image.device, dtype=image.dtype)
        mean = F.conv2d(padded, kernel)
        centered = image - mean
        variance = F.conv2d(F.pad(centered.square(), (self.padding,) * 4, mode="reflect"), kernel)
        normalized = centered / torch.sqrt(variance + self.eps)
        return normalized.clamp(-self.clip, self.clip)


class ParametricSteerableGaborBank(nn.Module):
    def __init__(self, orientations=8, bank_mode="parameterized", eps=1e-6):
        super().__init__()
        if int(orientations) != 8:
            raise ValueError("Physical V1 currently requires exactly eight orientations.")
        if bank_mode not in {"parameterized", "fixed"}:
            raise ValueError(f"Unknown bank mode: {bank_mode}")
        self.orientations = int(orientations)
        self.bank_mode = str(bank_mode)
        self.eps = float(eps)
        self.register_buffer(
            "base_wavelengths", torch.tensor([3.0, 6.0, 12.0]), persistent=True
        )
        self.register_buffer(
            "kernel_sizes", torch.tensor([9, 15, 25], dtype=torch.long), persistent=True
        )
        self.delta_lambda = nn.Parameter(torch.zeros(3))
        self.delta_sigma = nn.Parameter(torch.zeros(3))
        gamma_initial = math.log(0.4 / 0.6)
        self.gamma_logits = nn.Parameter(torch.full((3,), gamma_initial))
        if self.bank_mode == "fixed":
            for parameter in self.physical_parameters():
                parameter.requires_grad_(False)

    def physical_parameters(self):
        return [self.delta_lambda, self.delta_sigma, self.gamma_logits]

    def constrained_parameters(self):
        wavelength = self.base_wavelengths * torch.exp(0.25 * torch.tanh(self.delta_lambda))
        sigma = 0.56 * wavelength * torch.exp(0.20 * torch.tanh(self.delta_sigma))
        gamma = 0.3 + 0.5 * torch.sigmoid(self.gamma_logits)
        return wavelength, sigma, gamma

    def kernels_for_frequency(self, frequency):
        frequency = int(frequency)
        wavelength, sigma, gamma = self.constrained_parameters()
        kernel_size = int(self.kernel_sizes[frequency])
        radius = kernel_size // 2
        coordinates = torch.arange(
            -radius,
            radius + 1,
            device=wavelength.device,
            dtype=wavelength.dtype,
        )
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        theta = torch.arange(
            self.orientations, device=wavelength.device, dtype=wavelength.dtype
        ) * (math.pi / self.orientations)
        cosine = theta.cos()[:, None, None]
        sine = theta.sin()[:, None, None]
        x_theta = xx[None] * cosine + yy[None] * sine
        y_theta = -xx[None] * sine + yy[None] * cosine
        envelope = torch.exp(
            -(
                x_theta.square()
                + gamma[frequency].square() * y_theta.square()
            )
            / (2.0 * sigma[frequency].square())
        )
        phase = 2.0 * math.pi * x_theta / wavelength[frequency]
        even = envelope * phase.cos()
        odd = envelope * phase.sin()

        def normalize(kernels):
            kernels = kernels - kernels.mean(dim=(-2, -1), keepdim=True)
            norm = kernels.square().sum(dim=(-2, -1), keepdim=True).sqrt()
            return kernels / norm.clamp_min(self.eps)

        return normalize(even)[:, None], normalize(odd)[:, None]

    def forward(self, image):
        even_responses = []
        odd_responses = []
        for frequency in range(3):
            with _autocast_disabled(image):
                even_kernel, odd_kernel = self.kernels_for_frequency(frequency)
            kernels = torch.cat([even_kernel, odd_kernel], dim=0).to(
                device=image.device, dtype=image.dtype
            )
            padding = int(self.kernel_sizes[frequency]) // 2
            response = F.conv2d(F.pad(image, (padding,) * 4, mode="reflect"), kernels)
            even, odd = response.split(self.orientations, dim=1)
            even_responses.append(even)
            odd_responses.append(odd)
        even = torch.stack(even_responses, dim=1)
        odd = torch.stack(odd_responses, dim=1)
        amplitude = torch.sqrt(even.square() + odd.square() + self.eps)
        return even, odd, amplitude


class PhysicalOrientationEstimator(nn.Module):
    def __init__(self, orientations=8, temperature=0.1, eps=1e-6):
        super().__init__()
        self.orientations = int(orientations)
        self.temperature = float(temperature)
        self.eps = float(eps)
        theta = torch.arange(self.orientations, dtype=torch.float32) * (
            math.pi / self.orientations
        )
        self.register_buffer("cos2theta", torch.cos(2.0 * theta)[None, :, None, None])
        self.register_buffer("sin2theta", torch.sin(2.0 * theta)[None, :, None, None])

    def phase_agreement(self, even, odd, amplitude):
        real = even.float().sum(dim=1)
        imaginary = odd.float().sum(dim=1)
        coherent = torch.sqrt(real.square() + imaginary.square() + self.eps) - math.sqrt(
            self.eps
        )
        denominator = amplitude.float().sum(dim=1).clamp_min(self.eps)
        return (coherent / denominator).clamp(0.0, 1.0)

    def forward(self, even, odd, amplitude):
        with _autocast_disabled(even):
            phase = self.phase_agreement(even, odd, amplitude)
            energy = phase * amplitude.float().sum(dim=1)
            normalized = energy / energy.mean(dim=1, keepdim=True).clamp_min(self.eps)
            weights = F.softmax(normalized / self.temperature, dim=1)
            vx = (weights * self.cos2theta).sum(dim=1)
            vy = (weights * self.sin2theta).sum(dim=1)
            norm_squared = vx.square() + vy.square()
            confidence = (
                torch.sqrt(norm_squared + self.eps) - math.sqrt(self.eps)
            ).clamp(0.0, 1.0)
            safe_vx = vx + (norm_squared < self.eps).to(vx.dtype) * math.sqrt(self.eps)
            theta = 0.5 * torch.atan2(vy, safe_vx)
            orientation = torch.stack([(2.0 * theta).cos(), (2.0 * theta).sin()], dim=1)
        return phase.to(even.dtype), theta, orientation, confidence[:, None]


class ContinuousOrientationCanonicalization(nn.Module):
    def __init__(self, orientations=8):
        super().__init__()
        self.orientations = int(orientations)

    def circular_sample(self, responses, channel_shift):
        if responses.ndim != 5 or responses.shape[2] != self.orientations:
            raise ValueError(
                f"Expected [B,F,{self.orientations},H,W], got {tuple(responses.shape)}"
            )
        batch, frequencies, orientations, height, width = responses.shape
        base = torch.arange(
            orientations, device=responses.device, dtype=responses.dtype
        )
        positions = base[None, None, :, None, None] + channel_shift[:, None, None]
        lower = torch.floor(positions)
        upper_weight = positions - lower
        lower_index = lower.long().remainder(orientations).expand(
            batch, frequencies, orientations, height, width
        )
        upper_index = (lower.long() + 1).remainder(orientations).expand_as(lower_index)
        lower_value = torch.gather(responses, 2, lower_index)
        upper_value = torch.gather(responses, 2, upper_index)
        return lower_value * (1.0 - upper_weight) + upper_value * upper_weight

    def forward(self, responses, theta, confidence):
        shift = self.orientations * theta.to(responses.dtype) / math.pi
        canonical = self.circular_sample(responses, shift)
        gate = confidence.to(responses.dtype)[:, :, None]
        return responses * (1.0 - gate) + canonical * gate


class PointwiseExpert(nn.Module):
    def __init__(self, in_channels, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feature):
        return self.net(feature.contiguous())


class Contiguous(nn.Module):
    def forward(self, feature):
        return feature.contiguous()


class OrientationAlignedNeighborhood(nn.Module):
    def __init__(self, channels=32, kernel_size=5):
        super().__init__()
        if int(kernel_size) != 5:
            raise ValueError("Physical V1 OAN is defined with a 5x5 neighborhood.")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        radius = self.kernel_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        base = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        self.register_buffer("base_offsets", base, persistent=True)
        self.raw_weight = nn.Parameter(
            torch.empty(self.channels, 1, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        self.pointwise = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.norm = LayerNorm2d(self.channels)
        self.activation = nn.GELU()

    @property
    def symmetric_weight(self):
        return 0.5 * (self.raw_weight + torch.flip(self.raw_weight, dims=(-2, -1)))

    def offsets_from_theta(self, theta):
        cosine = theta.float().cos()[:, None]
        sine = theta.float().sin()[:, None]
        x = self.base_offsets[:, 0][None, :, None, None]
        y = self.base_offsets[:, 1][None, :, None, None]
        rotated_x = cosine * x - sine * y
        rotated_y = sine * x + cosine * y
        delta_x = rotated_x - x
        delta_y = rotated_y - y
        return torch.stack([delta_y, delta_x], dim=2).flatten(1, 2)

    def aligned_only(self, feature, theta):
        with _autocast_disabled(feature):
            feature_fp32 = feature.float()
            offsets = self.offsets_from_theta(theta).float()
            padded = F.pad(feature_fp32, (2, 2, 2, 2), mode="reflect")
            aligned = deform_conv2d(
                padded,
                offsets,
                self.symmetric_weight.float(),
                bias=None,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                mask=None,
            )
            aligned = self.activation(self.norm(self.pointwise(aligned.contiguous())))
        return aligned.to(feature.dtype)

    def forward(self, feature, theta, confidence):
        aligned = self.aligned_only(feature, theta)
        gate = confidence.to(feature.dtype)
        return feature * (1.0 - gate) + aligned * gate


class FixedBlurPool(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        one_dimensional = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = torch.outer(one_dimensional, one_dimensional) / 256.0
        self.register_buffer("kernel", kernel[None, None], persistent=True)
        self.channels = int(channels)

    def forward(self, feature):
        channels = feature.shape[1]
        kernel = self.kernel.to(device=feature.device, dtype=feature.dtype).expand(
            channels, 1, 5, 5
        )
        return F.conv2d(
            F.pad(feature, (2, 2, 2, 2), mode="reflect"),
            kernel,
            stride=2,
            groups=channels,
        )


class SharedPointwiseRefinement(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feature):
        return self.net(feature.contiguous())


class PhysicalEncoderV1(nn.Module):
    MODEL_CONFIGS = {
        "physical_v1_core": {},
        "physical_v1_no_oan": {"enable_oan": False},
        "physical_v1_energy_only": {"response_mode": "energy_only"},
        "physical_v1_no_confidence_gate": {"confidence_gate": False},
        "physical_v1_simple_scale": {"scale_fusion": "simple"},
        "physical_v1_fixed_bank": {"bank_mode": "fixed"},
    }

    def __init__(
        self,
        enable_oan=True,
        response_mode="full",
        confidence_gate=True,
        scale_fusion="stability",
        bank_mode="parameterized",
        channels=32,
    ):
        super().__init__()
        if response_mode not in {"full", "energy_only"}:
            raise ValueError(f"Unknown response mode: {response_mode}")
        if scale_fusion not in {"stability", "simple"}:
            raise ValueError(f"Unknown scale fusion: {scale_fusion}")
        self.enable_oan = bool(enable_oan)
        self.response_mode = str(response_mode)
        self.confidence_gate = bool(confidence_gate)
        self.scale_fusion = str(scale_fusion)
        self.bank_mode = str(bank_mode)
        self.channels = int(channels)

        self.ldn = FixedLocalDivisiveNormalization()
        self.gabor = ParametricSteerableGaborBank(bank_mode=bank_mode)
        self.orientation_estimator = PhysicalOrientationEstimator()
        self.canonicalizer = ContinuousOrientationCanonicalization()
        self.stable_encoder = PointwiseExpert(33, channels)
        if self.response_mode == "full":
            self.edge_encoder = PointwiseExpert(48, channels)
            self.contour_encoder = PointwiseExpert(48, channels)
            self.expert_names = ("edge", "contour", "stable")
        else:
            self.edge_encoder = None
            self.contour_encoder = None
            self.expert_names = ("stable",)

        if self.enable_oan:
            self.oan = nn.ModuleDict(
                {
                    name: OrientationAlignedNeighborhood(channels)
                    for name in self.expert_names
                }
            )
        else:
            self.oan = None

        self.blur_pool = FixedBlurPool(channels)
        self.downsample_refinement = SharedPointwiseRefinement(channels)
        self.scale_proxy = PointwiseExpert(channels * 3, channels)
        scale_gate_channels = channels * 6 + 6 if self.scale_fusion == "stability" else channels * 3
        self.scale_gate = nn.Sequential(
            Contiguous(),
            nn.Conv2d(scale_gate_channels, 64, 1, bias=False),
            LayerNorm2d(64),
            nn.GELU(),
            Contiguous(),
            nn.Conv2d(64, 3, 1, bias=True),
        )
        if self.response_mode == "full":
            self.expert_gate = nn.Sequential(
                Contiguous(),
                nn.Conv2d(channels * 3 + 4, channels, 1, bias=False),
                LayerNorm2d(channels),
                nn.GELU(),
                Contiguous(),
                nn.Conv2d(channels, 3, 1, bias=True),
            )
        else:
            self.expert_gate = None
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(channels, 128, 1, bias=False),
            LayerNorm2d(128),
        )

    def physical_filter_parameters(self):
        return [parameter for parameter in self.gabor.physical_parameters() if parameter.requires_grad]

    @staticmethod
    def _normalize(feature):
        return F.normalize(feature, p=2, dim=1, eps=1e-8)

    @staticmethod
    def _image_pyramid(image):
        return [
            image,
            F.interpolate(
                image, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True
            ),
            F.interpolate(
                image, scale_factor=0.25, mode="bilinear", align_corners=False, antialias=True
            ),
        ]

    def _downsample(self, feature, steps):
        for _ in range(int(steps)):
            feature = self.downsample_refinement(self.blur_pool(feature))
        return feature

    def _downsample_field(self, feature, steps):
        for _ in range(int(steps)):
            feature = self.blur_pool(feature)
        return feature

    def _scale_features(self, image):
        all_experts = []
        orientations = []
        confidences = []
        for scale_index, scale_image in enumerate(self._image_pyramid(image)):
            normalized = self.ldn(scale_image)
            even, odd, amplitude = self.gabor(normalized)
            phase, theta, orientation, confidence = self.orientation_estimator(
                even, odd, amplitude
            )
            gate = confidence if self.confidence_gate else torch.ones_like(confidence)
            even = self.canonicalizer(even, theta, gate)
            odd = self.canonicalizer(odd, theta, gate)
            amplitude = self.canonicalizer(amplitude, theta, gate)

            stable_input = torch.cat(
                [amplitude.flatten(1, 2), phase, confidence], dim=1
            )
            experts = {"stable": self.stable_encoder(stable_input)}
            if self.response_mode == "full":
                odd_flat = odd.flatten(1, 2)
                even_flat = even.flatten(1, 2)
                experts["edge"] = self.edge_encoder(
                    torch.cat([odd_flat, odd_flat.abs()], dim=1)
                )
                experts["contour"] = self.contour_encoder(
                    torch.cat([even_flat, even_flat.abs()], dim=1)
                )

            if self.enable_oan:
                experts = {
                    name: self.oan[name](feature, theta, gate)
                    for name, feature in experts.items()
                }

            steps = 3 - scale_index
            experts = {
                name: self._downsample(feature, steps)
                for name, feature in experts.items()
            }
            if self.response_mode == "energy_only":
                experts = {
                    "edge": experts["stable"],
                    "contour": experts["stable"],
                    "stable": experts["stable"],
                }
            all_experts.append(experts)
            orientation = self._downsample_field(orientation, steps)
            orientations.append(self._normalize(orientation))
            confidences.append(self._downsample_field(confidence, steps).clamp(0.0, 1.0))
        return all_experts, orientations, confidences

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"PhysicalEncoderV1 expects [B,1,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError("Input height and width must be divisible by 8.")
        if min(image.shape[-2:]) < 64:
            raise ValueError("PhysicalEncoderV1 requires input dimensions of at least 64 pixels.")

        scales, orientations, confidences = self._scale_features(image)
        proxies = [
            self._normalize(
                self.scale_proxy(
                    torch.cat(
                        [scale["edge"], scale["contour"], scale["stable"]], dim=1
                    )
                )
            )
            for scale in scales
        ]
        agreements = [
            (proxies[0] * proxies[1]).sum(dim=1, keepdim=True),
            (proxies[1] * proxies[2]).sum(dim=1, keepdim=True),
            (proxies[0] * proxies[2]).sum(dim=1, keepdim=True),
        ]
        if self.scale_fusion == "stability":
            differences = [
                (proxies[0] - proxies[1]).abs(),
                (proxies[1] - proxies[2]).abs(),
                (proxies[0] - proxies[2]).abs(),
            ]
            scale_gate_input = torch.cat(
                [*proxies, *differences, *agreements, *confidences], dim=1
            )
        else:
            scale_gate_input = torch.cat(proxies, dim=1)
        scale_weights = F.softmax(self.scale_gate(scale_gate_input), dim=1)

        branches = {}
        for name in ("edge", "contour", "stable"):
            branches[name] = self._normalize(
                sum(
                    scale_weights[:, index : index + 1] * scales[index][name]
                    for index in range(3)
                )
            )

        orientation = self._normalize(
            sum(
                scale_weights[:, index : index + 1] * orientations[index]
                for index in range(3)
            )
        )
        confidence = sum(
            scale_weights[:, index : index + 1] * confidences[index]
            for index in range(3)
        ).clamp(0.0, 1.0)

        if self.response_mode == "full":
            expert_gate_input = torch.cat(
                [
                    branches["edge"],
                    branches["contour"],
                    branches["stable"],
                    *agreements,
                    confidence,
                ],
                dim=1,
            )
            expert_weights = F.softmax(self.expert_gate(expert_gate_input), dim=1)
            structure = sum(
                expert_weights[:, index : index + 1] * branches[name]
                for index, name in enumerate(("edge", "contour", "stable"))
            )
        else:
            batch, _, height, width = branches["stable"].shape
            expert_weights = branches["stable"].new_zeros(batch, 3, height, width)
            expert_weights[:, 2] = 1.0
            structure = branches["stable"]
        fused = self._normalize(self.descriptor_head(structure.contiguous()))
        return {
            "fused": fused,
            "edge": branches["edge"],
            "contour": branches["contour"],
            "stable": branches["stable"],
            "orientation": orientation,
            "confidence": confidence,
            "scale_weights": scale_weights,
            "expert_weights": expert_weights,
        }


def build_physical_v1_encoder(model_name="physical_v1_core"):
    model_name = str(model_name).lower()
    if model_name not in PhysicalEncoderV1.MODEL_CONFIGS:
        choices = ", ".join(PhysicalEncoderV1.MODEL_CONFIGS)
        raise ValueError(f"Unknown Physical V1 model: {model_name}. Choices: {choices}")
    return PhysicalEncoderV1(**PhysicalEncoderV1.MODEL_CONFIGS[model_name])


__all__ = [
    "ContinuousOrientationCanonicalization",
    "FixedBlurPool",
    "FixedLocalDivisiveNormalization",
    "OrientationAlignedNeighborhood",
    "ParametricSteerableGaborBank",
    "PhysicalEncoderV1",
    "PhysicalOrientationEstimator",
    "build_physical_v1_encoder",
    "count_trainable_parameters",
]
