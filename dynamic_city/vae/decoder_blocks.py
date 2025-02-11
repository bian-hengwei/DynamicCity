import torch.nn as nn
from einops import rearrange

from dynamic_city.utils.vae_train_utils import (
    add_positional_encoding, compose_hexplane_channelwise, decompose_hexplane_channelwise, zero_module,
)


class HexplaneConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_layers_h = nn.Conv2d(in_channels * 3, out_channels * 3, groups=3, kernel_size=3, stride=1, padding=1)
        self.in_layers_t = nn.Conv2d(in_channels * 3, out_channels * 3, groups=3, kernel_size=3, stride=1, padding=1)
        self.norms = nn.ModuleList([nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True) for _ in range(6)])
        self.out_layers_h = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Conv2d(out_channels * 3, out_channels * 3, groups=3, kernel_size=3, stride=1, padding=1))
        )
        self.out_layers_t = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Conv2d(out_channels * 3, out_channels * 3, groups=3, kernel_size=3, stride=1, padding=1))
        )
        self.shortcut_h = nn.Conv2d(in_channels * 3, out_channels * 3, groups=3, kernel_size=1, stride=1, padding=0)
        self.shortcut_t = nn.Conv2d(in_channels * 3, out_channels * 3, groups=3, kernel_size=1, stride=1, padding=0)

    def forward(self, feat_maps):
        h_original, t_original, sizes_xyz, sizes_t = compose_hexplane_channelwise(feat_maps)
        h = self.in_layers_h(h_original)
        t = self.in_layers_t(t_original)

        feat_maps = decompose_hexplane_channelwise(h, t, sizes_xyz, sizes_t)
        feat_maps = [self.norms[i](feat_maps[i]) for i in range(6)]
        h, t, _, _ = compose_hexplane_channelwise(feat_maps)

        h = self.out_layers_h(h)
        t = self.out_layers_t(t)
        h = h + self.shortcut_h(h_original)
        t = t + self.shortcut_t(t_original)

        return decompose_hexplane_channelwise(h, t, sizes_xyz, sizes_t)


class PlaneUpsampler(nn.Module):
    def __init__(self, channels, up_xy=(1, 1)):
        super().__init__()
        kernel_size = stride = tuple([2 ** up for up in up_xy])
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(1e-1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(1e-1, inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


class VoxelDecoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels_high_res, num_classes, down_xyz, pos_num_freq):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_channels_high_res = hidden_channels_high_res
        self.num_classes = num_classes
        self.down_x, self.down_y, self.down_z = down_xyz
        self.pos_num_freq = pos_num_freq

        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(1e-1, inplace=True),
        )
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(self.hidden_channels),
                nn.LeakyReLU(1e-1, inplace=True),
                nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(self.hidden_channels),
                nn.LeakyReLU(1e-1, inplace=True),
            )
        )
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=1),
                nn.InstanceNorm3d(self.hidden_channels),
                nn.LeakyReLU(1e-1, inplace=True),
                nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=1),
                nn.InstanceNorm3d(self.hidden_channels),
                nn.LeakyReLU(1e-1, inplace=True),
            )
        )
        self.transpose_conv = nn.ConvTranspose3d(
            self.hidden_channels, self.hidden_channels_high_res,
            kernel_size=(2 ** self.down_x, 2 ** self.down_y, 2 ** self.down_z),
            stride=(2 ** self.down_x, 2 ** self.down_y, 2 ** self.down_z)
        )
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                self.hidden_channels_high_res + 4 * 2 * self.pos_num_freq,
                self.num_classes, kernel_size=1
            ),
            nn.InstanceNorm3d(self.num_classes),
            nn.LeakyReLU(1e-1, inplace=True),
            nn.Conv3d(self.num_classes, self.num_classes, kernel_size=1),
        )

    def forward(self, x):
        b, t = x.shape[:2]
        x = rearrange(x, 'b t x y z c -> (b t) c x y z')
        x = self.conv_in(x)
        for conv_layer in self.conv_layers:
            x = x + conv_layer(x)
        x = self.transpose_conv(x)
        x = rearrange(x, '(b t) c x y z -> b t x y z c', b=b, t=t)
        x = add_positional_encoding(x, pos_num_freq=self.pos_num_freq)
        x = rearrange(x, 'b t x y z c -> (b t) c x y z')
        x = self.final_conv(x)
        x = rearrange(x, '(b t) c x y z -> b t x y z c', b=b, t=t)
        return x
