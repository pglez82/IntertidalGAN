"""
TideGAN: Conditional image-to-image translation for tide-aware coastal imagery.

Architecture:
  Generator: U-Net with AdaIN condition injection
    - Input: reference image (3 ch) + condition vector (broadcast as spatial maps)
    - Output: synthetic image (3 ch)

The condition (normalized tide level + site embedding) is injected via
AdaIN-style feature modulation at every normalization layer in the generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Conditional injection helpers ──────────────────────────────────────────
class ConditionEmbed(nn.Module):
    """
    Embeds a scalar tide level + site one-hot into a channel vector,
    broadcast to spatial maps for injection into the generator.
    """
    def __init__(self, n_sites=3, cond_dim=32):
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(1 + n_sites, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, cond_dim),
        )

    def forward(self, condition):
        """condition: (batch, 1 + n_sites) or (1 + n_sites,)"""
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        return self.mlp(condition)  # (batch, cond_dim)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization – scales and shifts features by condition embedding."""
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.scale_shift = nn.Sequential(
            nn.Linear(cond_dim, in_channels * 2),
        )

    def forward(self, x, cond_embed):
        """x: (B, C, H, W), cond_embed: (B, cond_dim)"""
        stats = self.norm(x)  # (B, C, H, W)
        params = self.scale_shift(cond_embed).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        return stats * (1 + scale) + shift


# ── Generator blocks ─────────────────────────────────────────────────────
class CondEncoder(nn.Module):
    """Downsampling block: Conv→AdaIN→LeakyReLU."""
    def __init__(self, in_ch, out_ch, cond_dim=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = AdaIN(out_ch, cond_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, cond_embed):
        x = self.conv(x)
        x = self.norm(x, cond_embed)
        return self.activation(x)


class CondDecoder(nn.Module):
    """Upsampling block: ConvTranspose→AdaIN→ReLU."""
    def __init__(self, in_ch, out_ch, cond_dim=16):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = AdaIN(out_ch, cond_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, cond_embed):
        x = self.deconv(x)
        x = self.norm(x, cond_embed)
        return self.activation(x)


class CondConvBlock(nn.Module):
    """2× Conv block with AdaIN for decoder (after skip concat)."""
    def __init__(self, in_ch, out_ch, cond_dim=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = AdaIN(out_ch, cond_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = AdaIN(out_ch, cond_dim)

    def forward(self, x, cond_embed):
        x = self.conv1(x)
        x = self.norm1(x, cond_embed)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, cond_embed)
        return self.relu1(x)


class ResBlock(nn.Module):
    """Residual block with AdaIN condition injection."""
    def __init__(self, channels, cond_dim=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = AdaIN(channels, cond_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = AdaIN(channels, cond_dim)

    def forward(self, x, cond_embed):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out, cond_embed)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out, cond_embed)
        return out + identity


class Generator(nn.Module):
    """
    U-Net generator for conditional image-to-image translation.

    Architecture:
      Encoder: 4 down-sampling blocks (64→128→256→512 channels)
        Input: 256×256 → 128×128 → 64×64 → 32×32 → 16×16
      Bottleneck: 3 ResNet blocks at 512ch, 16×16
      Decoder: 4 up-sampling blocks with skip connections
        16×16 → 32×32 → 64×64 → 128×128 → 256×256
    """

    def __init__(self, n_sites=3, cond_dim=32):
        super().__init__()
        self.condition_embed = ConditionEmbed(n_sites, cond_dim)

        # ── Encoder (halves spatial dims each block) ────────────────────
        # Input: 256×256
        self.enc1 = CondEncoder(3, 64, cond_dim)       # 256→128
        self.enc2 = CondEncoder(64, 128, cond_dim)     # 128→64
        self.enc3 = CondEncoder(128, 256, cond_dim)    # 64→32
        self.enc4 = CondEncoder(256, 512, cond_dim)    # 32→16

        # ── Bottleneck ──────────────────────────────────────────────────
        self.bn1 = ResBlock(512, cond_dim)
        self.bn2 = ResBlock(512, cond_dim)
        self.bn3 = ResBlock(512, cond_dim)

        # ── Decoder ─────────────────────────────────────────────────────
        # Each decoder: upsample + concat with skip → ConvBlock
        self.dec4 = CondDecoder(512, 256, cond_dim)    # 16→32
        self.dec4_block = CondConvBlock(256 + 256, 256, cond_dim)  # skip=e3(256ch, 32×32)

        self.dec3 = CondDecoder(256, 128, cond_dim)    # 32→64
        self.dec3_block = CondConvBlock(128 + 128, 128, cond_dim)  # skip=e2(128ch, 64×64)

        self.dec2 = CondDecoder(128, 64, cond_dim)     # 64→128
        self.dec2_block = CondConvBlock(64 + 64, 64, cond_dim)     # skip=e1(64ch, 128×128)

        self.dec1 = CondDecoder(64, 64, cond_dim)      # 128→256
        self.dec1_block = CondConvBlock(64 + 3, 64, cond_dim)        # skip=ref_image(3ch, 256×256)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, ref_image, condition):
        """
        Args:
            ref_image:  (B, 3, H, W)  reference image in [-1, 1]
            condition:  (B, 1 + n_sites)  normalized tide + site one-hot
        Returns:
            generated:  (B, 3, H, W)  in [-1, 1]
        """
        cond_embed = self.condition_embed(condition)

        # ── Encoder ─────────────────────────────────────────────────────
        e1 = self.enc1(ref_image, cond_embed)          # 256×256 → 128×128
        e2 = self.enc2(e1, cond_embed)                 # 128×128 → 64×64
        e3 = self.enc3(e2, cond_embed)                 # 64×64 → 32×32
        e4 = self.enc4(e3, cond_embed)                 # 32×32 → 16×16

        # ── Bottleneck ──────────────────────────────────────────────────
        b = self.bn1(e4, cond_embed)
        b = self.bn2(b, cond_embed)
        b = self.bn3(b, cond_embed)

        # ── Decoder (U-Net: upsample → concat with skip → conv block) ──
        # Stage 4: upsample 16→32, concat with e3 (32×32, 256ch)
        d4 = self.dec4(b, cond_embed)                  # 16×16 → 32×32, 256ch
        d4 = self.dec4_block(torch.cat([d4, e3], dim=1), cond_embed)  # 32×32, 256ch

        # Stage 3: upsample 32→64, concat with e2 (64×64, 128ch)
        d3 = self.dec3(d4, cond_embed)                 # 32×32 → 64×64, 128ch
        d3 = self.dec3_block(torch.cat([d3, e2], dim=1), cond_embed)  # 64×64, 128ch

        # Stage 2: upsample 64→128, concat with e1 (128×128, 64ch)
        d2 = self.dec2(d3, cond_embed)                 # 64×64 → 128×128, 64ch
        d2 = self.dec2_block(torch.cat([d2, e1], dim=1), cond_embed)  # 128×128, 64ch

        # Stage 1: upsample 128→256, concat with ref_image (256×256, 3ch)
        d1 = self.dec1(d2, cond_embed)                 # 128×128 → 256×256, 64ch
        d1 = self.dec1_block(torch.cat([d1, ref_image], dim=1), cond_embed)  # 256×256, 64ch

        # ── Output ──────────────────────────────────────────────────────
        out = self.final(d1)
        out = torch.tanh(out)  # clip to [-1, 1]
        return out


# ── Utility: count parameters ─────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_sites = 3
    cond_dim = 32
    patch_size = 256
    batch_size = 2

    # Test Generator
    G = Generator(n_sites, cond_dim).to(device)
    print(f"Generator parameters: {count_parameters(G):,}")

    ref = torch.randn(batch_size, 3, patch_size, patch_size).to(device)
    cond = torch.randn(batch_size, 1 + n_sites).to(device)
    with torch.no_grad():
        gen = G(ref, cond)
    print(f"Generator output: {gen.shape}  (range: {gen.min():.3f}, {gen.max():.3f})")


