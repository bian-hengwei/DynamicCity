import functools
import operator
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops import rearrange, repeat

from dynamic_city.vae.encoder_blocks import PlaneDownsampler, Transformer, VoxelEncoder


class EncoderBase(ABC, nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.t = conf.dataset.sequence_length
        self.latent_channels = latent_channels = conf.model.latent_channels
        self.separate_t_encoder = conf.model.separate_t_encoder
        self.down_xyz = conf.model.down_x, conf.model.down_y, conf.model.down_z

        self.embedding = nn.Embedding(conf.dataset.num_classes, conf.model.latent_channels)

        self.voxel_encoder = VoxelEncoder(latent_channels, in_channels=latent_channels, down_xyz=self.down_xyz)
        if self.separate_t_encoder:
            t_in_channels = latent_channels + (self.conf.dataset.sequence_length if conf.model.one_hot_time else 0)
            self.t_encoder = VoxelEncoder(latent_channels, in_channels=t_in_channels, down_xyz=self.down_xyz)

        self.norm = nn.InstanceNorm2d(latent_channels)

        down_ratios = [
            (conf.model.hex_down_x, conf.model.hex_down_y),
            (conf.model.hex_down_x, conf.model.hex_down_z),
            (conf.model.hex_down_y, conf.model.hex_down_z),
            (conf.model.hex_down_t, conf.model.hex_down_x),
            (conf.model.hex_down_t, conf.model.hex_down_y),
            (conf.model.hex_down_t, conf.model.hex_down_z),
        ]
        self.downsamplers = nn.ModuleList(
            [PlaneDownsampler(latent_channels, down_xy=ratio) for ratio in down_ratios]
        )

    def forward(self, x):
        x = x.detach().clone()  # B, T, X, Y, Z
        x = self.embedding(x)  # B, T, X, Y, Z, C
        t_vox = None
        if self.separate_t_encoder:
            t_vox = self.vox_convs(x, self.t_encoder, one_hot=self.conf.model.one_hot_time)
        x = self.vox_convs(x, self.voxel_encoder)
        x = self.vox_to_planes(x, t_vox)
        for i, downsampler in enumerate(self.downsamplers):
            x[i] = self.downsamplers[i](x[i])
        return x

    def vox_convs(self, x, encoder, one_hot=False):
        B, T, X, Y, Z, C = x.shape
        if one_hot:
            x = rearrange(x, 'b t x y z c -> b t c x y z')
            one_hot_tensor = torch.eye(T, device=x.device)[None, :, :, None, None, None].expand(B, T, T, X, Y, Z)
            x = torch.cat([x, one_hot_tensor], dim=2)
            x = rearrange(x, 'b t c x y z -> (b t) c x y z')
        else:
            x = rearrange(x, 'b t x y z c -> (b t) c x y z')
        x = encoder(x)
        x = rearrange(x, '(b t) c x y z -> b c t x y z', b=B, t=T)
        return x

    @abstractmethod
    def vox_to_planes(self, x, t_vox=None):
        pass


class TrEncoder(EncoderBase):
    def __init__(self, conf):
        super().__init__(conf)

        T = self.t  # t downsampling is performed later
        X, Y, Z = self.conf.dataset.spatial_size
        X, Y, Z = X // (2 ** conf.model.down_x), Y // (2 ** conf.model.down_y), Z // (2 ** conf.model.down_z)
        shape_dict = dict(t=T, x=X, y=Y, z=Z)

        plane_names = ['xy', 'xz', 'yz', 'tx', 'ty', 'tz']
        plane_names_input = {'xy': 'txyz', 'xz': 'txyz', 'yz': 'txyz', 'tx': 'txyz', 'ty': 'txyz', 'tz': 'txyz'}

        self.separate_t_reducer = self.conf.model.separate_t_reducer
        if self.separate_t_reducer:
            plane_names = ['xyz'] + plane_names
            plane_names_input = {
                'xyz': 'txyz', 'xy': 'xyz', 'xz': 'xyz', 'yz': 'xyz', 'tx': 'txyz', 'ty': 'txyz', 'tz': 'txyz'
            }

        channels = self.conf.model.latent_channels

        self.tokens = nn.ParameterDict()
        self.pos_encodings = nn.ParameterDict()
        self.transformers = nn.ModuleDict()
        self.rearrange1 = dict()
        self.rearrange2 = dict()
        self.convs = nn.ModuleDict()

        for plane in plane_names:  # xy
            reduce_dims = ''.join([dim for dim in plane_names_input[plane] if dim not in plane])  # tz

            self.tokens[plane] = nn.Parameter(torch.randn(1, 1, channels))

            pe_dim1 = functools.reduce(operator.mul, [shape_dict[key] for key in reduce_dims], 1)
            self.pos_encodings[plane] = nn.Parameter(torch.randn(1, pe_dim1 + 1, channels))  # 1, tz+1, c

            self.transformers[plane] = Transformer(
                channels,
                self.conf.model.transformer_depth,
                self.conf.model.transformer_heads,
                self.conf.model.transformer_dim_mlp,
                dropout=self.conf.model.dropout
            )

            self.rearrange1[plane] = (f'b c {" ".join(plane_names_input[plane])} -> '
                                      f'(b {" ".join(plane)}) ({" ".join(reduce_dims)}) c')  # bctxyz->(bxy)(tzc)
            self.rearrange2[plane] = (f'(b {" ".join(plane)}) c -> b c {" ".join(plane)}',  # (bxy)c -> bcxy
            {key: shape_dict[key] for key in plane})

            convnd = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
            self.convs[plane] = nn.Sequential(
                convnd[len(plane)](channels, channels, 1),
                nn.Tanh(),
                convnd[len(plane)](channels, channels, 1),
            )

    def vox_to_planes(self, x, t_vox=None):  # B, C, T, X, Y, Z
        t_vox = x if t_vox is None else t_vox
        if self.separate_t_reducer:
            x = self.process_plane(x, 'xyz')
        xy = self.process_plane(x, 'xy')
        xz = self.process_plane(x, 'xz')
        yz = self.process_plane(x, 'yz')
        tx = self.process_plane(t_vox, 'tx')
        ty = self.process_plane(t_vox, 'ty')
        tz = self.process_plane(t_vox, 'tz')

        features = [xy, xz, yz, tx, ty, tz]
        return features

    def process_plane(self, x, name):
        plane = rearrange(x, self.rearrange1[name])  # B, C, T, X, Y, Z -> B*X*Y, T*Z, C
        token = repeat(self.tokens[name], '1 1 c -> b 1 c', b=plane.shape[0])
        plane = torch.cat([plane, token], dim=1)
        plane = plane + self.pos_encodings[name]
        plane = self.transformers[name](plane)[:, 0]  # B*X*Y, 1, C
        plane = rearrange(plane, self.rearrange2[name][0], **self.rearrange2[name][1])
        plane = self.convs[name](plane)
        return plane
