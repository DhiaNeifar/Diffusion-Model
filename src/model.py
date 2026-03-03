import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / denom)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleEpsModel(nn.Module):
    def __init__(self, num_classes: int, time_dim: int = 128, label_dim: int = 32, base_ch: int = 64):
        # Small U-Net for epsilon_theta(x_t, t, y) with global receptive field.
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.label_emb = nn.Embedding(num_classes, label_dim)
        cond_dim = time_dim + label_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, cond_dim)
        self.downsample1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.downsample2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)

        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, cond_dim)

        self.upsample2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(base_ch * 4, base_ch * 2, cond_dim)
        self.upsample1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.up1 = ResBlock(base_ch * 2, base_ch, cond_dim)

        self.out_norm = nn.GroupNorm(_num_groups(base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        te = self.time_mlp(self.time_emb(t))
        ye = self.label_emb(y)
        cond = self.cond_mlp(torch.cat([te, ye], dim=1))

        h0 = self.in_conv(x)
        h1 = self.down1(h0, cond)
        h2 = self.down2(self.downsample1(h1), cond)
        h_mid = self.downsample2(h2)
        h_mid = self.mid1(h_mid, cond)
        h_mid = self.mid2(h_mid, cond)

        h = self.upsample2(h_mid)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h1], dim=1), cond)

        return self.out_conv(F.silu(self.out_norm(h)))
