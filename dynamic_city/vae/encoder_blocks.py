import torch
import torch.nn as nn
from timm.layers import Mlp

from dynamic_city.utils.attention_utils import Attention


class VoxelEncoder(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, in_channels=None, down_xyz=(0, 0, 0)):
        super().__init__()
        if in_channels is None:
            in_channels = channels
        self.conv0 = nn.Conv3d(in_channels, channels, kernel_size, padding=padding, padding_mode='replicate')
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, padding=padding, padding_mode='replicate'),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(channels, channels, kernel_size, padding=padding, padding_mode='replicate'),
            nn.InstanceNorm3d(channels)
        )
        self.down = nn.ModuleList()
        for i in range(max(down_xyz)):
            down_kernel = down_stride = tuple([2 if ratio > i else 1 for ratio in down_xyz])
            self.down.append(
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=down_kernel, stride=down_stride),
                    nn.InstanceNorm3d(channels),
                )
            )
            self.down.append(
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size, padding=padding, padding_mode='replicate'),
                    nn.InstanceNorm3d(channels),
                    nn.LeakyReLU(1e-1, True),
                    nn.Conv3d(channels, channels, kernel_size, padding=padding, padding_mode='replicate'),
                    nn.InstanceNorm3d(channels)
                )
            )

    def forward(self, x):
        x = self.conv0(x)  # B * T, C, X, Y, Z
        x = x + self.conv1(x)  # B * T, C, X, Y, Z

        for i, module in enumerate(self.down):
            if i % 2 == 0:
                x = module(x)
            else:
                x = x + module(x)
        return x


class PlaneDownsampler(nn.Module):
    def __init__(self, channels, down_xy=(1, 1)):
        super().__init__()
        self.down = nn.ModuleList([nn.Identity()])
        for i in range(max(down_xy)):
            down_kernel = down_stride = tuple([2 if ratio > i else 1 for ratio in down_xy])
            leaky_relu = nn.LeakyReLU(1e-1, True) if i != max(down_xy) - 1 else nn.Identity()
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=down_kernel, stride=down_stride),
                    leaky_relu,
                )
            )

    def forward(self, x):
        for module in self.down:
            x = module(x)
        return x


class Transformer(nn.Module):
    """
    Simple Transformer block with flash attention.
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleList()
            layer.append(nn.Sequential(nn.LayerNorm(dim), Attention(dim, heads, attn_drop=dropout)))
            layer.append(nn.Sequential(nn.LayerNorm(dim), Mlp(dim, mlp_dim, drop=dropout)))
            self.layers.append(nn.ModuleList(layer))

    def batch_forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def forward(self, x):  # B, N, C
        # flash-attn limits batch size to 32768
        # split batch if too large
        batch_size = x.shape[0]
        max_batch_size = 32768

        if batch_size > max_batch_size:
            x_split = torch.split(x, max_batch_size)
            processed_splits = list()
            for x_batch in x_split:
                processed_x_batch = self.batch_forward(x_batch)
                processed_splits.append(processed_x_batch)
            x = torch.cat(processed_splits, dim=0)
        else:
            x = self.batch_forward(x)

        return x
