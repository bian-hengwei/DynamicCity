from abc import ABC, abstractmethod
from functools import reduce

import torch.nn as nn

from dynamic_city.vae.decoder_blocks import HexplaneConvBlock, PlaneUpsampler, VoxelDecoderBlock


class DecoderBase(ABC, nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        hadarard_op = lambda lst: reduce(lambda x, y: x * y, lst)
        summation_op = lambda lst: sum(lst)
        self.op = hadarard_op if conf.model.hadamard else summation_op

        self.conv_block = HexplaneConvBlock(conf.model.latent_channels, conf.model.query_channels)
        self.hex_upsample_block = None  # subclass

    def forward(self, hexplane):
        hexplane = self.conv_block(hexplane)  # increase number of channels
        hexplane = [block(plane) for block, plane in zip(self.hex_upsample_block, hexplane)]
        coords = [[2, 5], [2, 4], [2, 3], [4, 5], [3, 5], [3, 4]]
        hexplane = [plane.unsqueeze(dim1).unsqueeze(dim2).permute(0, 2, 3, 4, 5, 1) for (dim1, dim2), plane in
            zip(coords, hexplane)]  # unsqueeze
        voxel = self.op(hexplane)  # broadcast and combine
        out = self.forward_voxel(voxel)
        return out

    @abstractmethod
    def forward_voxel(self, voxel):
        pass


class ConvDecoder(DecoderBase):
    def __init__(self, conf):
        super().__init__(conf)
        upsample_configs = [
            (self.conf.model.hex_down_x, self.conf.model.hex_down_y),
            (self.conf.model.hex_down_x, self.conf.model.hex_down_z),
            (self.conf.model.hex_down_y, self.conf.model.hex_down_z),
            (self.conf.model.hex_down_t, self.conf.model.hex_down_x),
            (self.conf.model.hex_down_t, self.conf.model.hex_down_y),
            (self.conf.model.hex_down_t, self.conf.model.hex_down_z)
        ]
        self.hex_upsample_block = nn.ModuleList(
            [PlaneUpsampler(self.conf.model.query_channels, up_xy=(down1, down2)) for down1, down2 in upsample_configs]
        )
        self.voxel_decoder = VoxelDecoderBlock(
            self.conf.model.query_channels,
            self.conf.model.conv_hidden_low_res,
            self.conf.model.conv_hidden_high_res,
            self.conf.dataset.num_classes,
            (self.conf.model.down_x, self.conf.model.down_y, self.conf.model.down_z),
            self.conf.model.pe_freq
        )

    def forward_voxel(self, voxel):
        out = self.voxel_decoder(voxel)
        return out
