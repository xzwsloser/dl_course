import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.reshape(B, C, H, W)
        return x + self.proj(out)


class SmallUNet(nn.Module):
    """Compact U-Net for 28x28 MNIST. Params ~1.7M (well under 5M limit)."""

    def __init__(self, in_ch=1, base_ch=32, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.enc1 = ResidualBlock(in_ch, ch1, time_dim)
        self.down1 = nn.Conv2d(ch1, ch1, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(ch1, ch2, time_dim)
        self.down2 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)
        self.enc3 = ResidualBlock(ch2, ch3, time_dim)

        self.mid1 = ResidualBlock(ch3, ch3, time_dim)
        self.attn = SelfAttention(ch3)
        self.mid2 = ResidualBlock(ch3, ch3, time_dim)

        self.up3 = nn.ConvTranspose2d(ch3, ch3, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(ch3 + ch2, ch2, time_dim)
        self.up2 = nn.ConvTranspose2d(ch2, ch2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(ch2 + ch1, ch1, time_dim)
        self.dec1 = ResidualBlock(ch1, ch1, time_dim)
        self.final = nn.Conv2d(ch1, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h3 = self.enc3(self.down2(h2), t_emb)
        mid = self.mid2(self.attn(self.mid1(h3, t_emb)), t_emb)
        d3 = self.dec3(torch.cat([self.up3(mid), h2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), h1], dim=1), t_emb)
        return self.final(self.dec1(d2, t_emb))
