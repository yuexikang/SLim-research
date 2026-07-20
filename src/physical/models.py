import math

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.misc import LayerNorm2d


def count_trainable_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _make_gabor_quadrature_bank(
    orientations=8,
    kernel_size=9,
    sigma=2.5,
    wavelength=4.0,
    gamma=0.5,
):
    if kernel_size % 2 != 1:
        raise ValueError("Gabor kernel_size must be odd.")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    even_kernels = []
    odd_kernels = []
    for index in range(orientations):
        theta = index * math.pi / orientations
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        envelope = torch.exp(
            -(x_theta.square() + gamma**2 * y_theta.square()) / (2.0 * sigma**2)
        )
        phase = 2.0 * math.pi * x_theta / wavelength
        even = envelope * torch.cos(phase)
        odd = envelope * torch.sin(phase)
        even = even - even.mean()
        odd = odd - odd.mean()
        even_kernels.append(even / even.square().sum().sqrt().clamp_min(1e-8))
        odd_kernels.append(odd / odd.square().sum().sqrt().clamp_min(1e-8))
    even = torch.stack(even_kernels)[:, None]
    odd = torch.stack(odd_kernels)[:, None]
    return torch.cat([even, odd], dim=0)


class FixedQuadratureOrientationBank(nn.Module):
    def __init__(
        self,
        orientations=8,
        kernel_size=9,
        sigma=2.5,
        wavelength=4.0,
        gamma=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.orientations = int(orientations)
        self.padding = int(kernel_size) // 2
        self.eps = float(eps)
        kernels = _make_gabor_quadrature_bank(
            orientations=self.orientations,
            kernel_size=kernel_size,
            sigma=sigma,
            wavelength=wavelength,
            gamma=gamma,
        )
        self.register_buffer("kernels", kernels, persistent=True)

    def forward(self, image):
        padded = F.pad(image, (self.padding,) * 4, mode="reflect")
        responses = F.conv2d(padded, self.kernels.to(dtype=image.dtype))
        even, odd = responses.split(self.orientations, dim=1)
        return torch.sqrt(even.square() + odd.square() + self.eps)


class SoftOrientationCanonicalization(nn.Module):
    def __init__(self, orientations=8, temperature=0.1):
        super().__init__()
        self.orientations = int(orientations)
        self.temperature = float(temperature)
        theta = torch.arange(self.orientations, dtype=torch.float32) * (
            math.pi / self.orientations
        )
        self.register_buffer("cos2theta", torch.cos(2.0 * theta)[None, :, None, None])
        self.register_buffer("sin2theta", torch.sin(2.0 * theta)[None, :, None, None])

    def major_orientation(self, energy):
        # Keep angle estimation in FP32 under AMP; small orientation-weight
        # differences are exactly what drive the continuous channel shift.
        weights = F.softmax(energy.float() / self.temperature, dim=1)
        vx = (weights * self.cos2theta).sum(dim=1)
        vy = (weights * self.sin2theta).sum(dim=1)
        return 0.5 * torch.atan2(vy, vx)

    def circular_sample(self, energy, channel_shift):
        batch, channels, height, width = energy.shape
        if channels != self.orientations:
            raise ValueError(
                f"Expected {self.orientations} orientation channels, got {channels}."
            )
        base = torch.arange(channels, device=energy.device, dtype=energy.dtype)
        positions = base[None, :, None, None] + channel_shift[:, None]
        lower = torch.floor(positions)
        upper_weight = positions - lower
        lower_index = lower.long().remainder(channels).expand(batch, channels, height, width)
        upper_index = (lower.long() + 1).remainder(channels).expand_as(lower_index)
        lower_value = torch.gather(energy, 1, lower_index)
        upper_value = torch.gather(energy, 1, upper_index)
        return lower_value * (1.0 - upper_weight) + upper_value * upper_weight

    def forward(self, energy):
        orientation = self.major_orientation(energy)
        shift = self.orientations * orientation / math.pi
        return self.circular_sample(energy, shift), orientation


class StructureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, 1, bias=False),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class PhysicalDownsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class DescriptorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
            LayerNorm2d(128),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1, eps=1e-8)


class PhysicalEncoderV0(nn.Module):
    def __init__(self, canonicalize=True, multiscale=True):
        super().__init__()
        self.canonicalize = bool(canonicalize)
        self.multiscale = bool(multiscale)
        self.quadrature = FixedQuadratureOrientationBank()
        self.orientation = SoftOrientationCanonicalization()
        self.structure_encoder = StructureEncoder()
        path_depths = (3, 2, 1) if self.multiscale else (3,)
        self.downsample_paths = nn.ModuleList(
            [
                nn.Sequential(*[PhysicalDownsampleBlock() for _ in range(depth)])
                for depth in path_depths
            ]
        )
        self.scale_fusion = (
            nn.Conv2d(64 * len(path_depths), len(path_depths), 1)
            if len(path_depths) > 1
            else None
        )
        self.descriptor_head = DescriptorHead()

    def _image_pyramid(self, image):
        if not self.multiscale:
            return [image]
        return [
            image,
            F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True),
            F.interpolate(image, scale_factor=0.25, mode="bilinear", align_corners=False, antialias=True),
        ]

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"PhysicalEncoderV0 expects [B,1,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError("Input height and width must be divisible by 8.")

        aligned = []
        for scale_image, downsample in zip(self._image_pyramid(image), self.downsample_paths):
            energy = self.quadrature(scale_image)
            if self.canonicalize:
                energy, _ = self.orientation(energy)
            structure = self.structure_encoder(energy)
            aligned.append(downsample(structure))

        if len(aligned) == 1:
            fused = aligned[0]
        else:
            concatenated = torch.cat(aligned, dim=1)
            weights = F.softmax(self.scale_fusion(concatenated), dim=1)
            fused = sum(weights[:, index : index + 1] * feature for index, feature in enumerate(aligned))
        return self.descriptor_head(fused)


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=False):
        super().__init__()
        self.residual = bool(residual and in_channels == out_channels and stride == 1)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        output = self.net(x)
        return output + x if self.residual else output


class TinyCNNEncoder(nn.Module):
    """Parameter-matched generic CNN baseline without physical priors."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(16),
            nn.GELU(),
        )
        self.stage16 = DSConvBlock(16, 16, residual=True)
        self.down32 = DSConvBlock(16, 32, stride=2)
        self.stage32 = DSConvBlock(32, 32, residual=True)
        self.down64 = DSConvBlock(32, 64, stride=2)
        self.stage64 = nn.Sequential(*[DSConvBlock(64, 64, residual=True) for _ in range(6)])
        self.descriptor_head = DescriptorHead()

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"TinyCNNEncoder expects [B,1,H,W], got {tuple(image.shape)}")
        x = self.stem(image)
        x = self.stage16(x)
        x = self.down32(x)
        x = self.stage32(x)
        x = self.down64(x)
        x = self.stage64(x)
        return self.descriptor_head(x)


def build_physical_v0_encoder(model_name):
    model_name = str(model_name).lower()
    if model_name == "physical_full":
        return PhysicalEncoderV0(canonicalize=True, multiscale=True)
    if model_name == "physical_no_canon":
        return PhysicalEncoderV0(canonicalize=False, multiscale=True)
    if model_name == "physical_single_scale":
        return PhysicalEncoderV0(canonicalize=True, multiscale=False)
    if model_name == "tiny_cnn":
        return TinyCNNEncoder()
    raise ValueError(f"Unknown Physical V0 model: {model_name}")
