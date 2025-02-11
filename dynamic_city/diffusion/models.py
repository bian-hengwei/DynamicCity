# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, PatchEmbed

from dynamic_city.diffusion.embedders import (
    CmdCondEmbedder, HexCondEmbedder, LayoutCondEmbedder, TimestepEmbedder, TrajCondEmbedder,
)
from dynamic_city.utils.attention_utils import Attention, CrossAttention
from dynamic_city.utils.data_utils import Command
from dynamic_city.utils.dit_utils import get_2d_sincos_pos_embed, modulate


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, x_attn=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.use_x_attn = x_attn
        if self.use_x_attn:
            self.normx = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.x_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, y):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_x_attn:
            x = x + self.x_attn(self.normx(x), y)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        # config args
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,

        # main args
        in_channels=16,
        x_attn=False,
        txyz=(8, 64, 64, 8),
        seq_len=16,
        patch_mask=None,

        # cond args
        hex_cond=False,
        layout_cond=False,
        traj_cond=False,
        cmd_cond=False,
        traj_size=48,

        # other args
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.txyz = txyz
        self.dim_t, self.dim_x, self.dim_y, self.dim_z = txyz
        self.traj_size = traj_size
        self.patch_mask = patch_mask

        im_size = self.dim_x + self.dim_t + self.dim_z
        self.hex_emb = self.layout_emb = self.traj_emb = self.cmd_emb = None

        self.x_embedder = PatchEmbed(im_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if hex_cond:
            self.hex_emb = HexCondEmbedder(im_size, patch_size, in_channels, hidden_size, class_dropout_prob)

        if layout_cond:
            self.layout_emb = LayoutCondEmbedder(im_size, patch_size, seq_len, 1, self.dim_x, class_dropout_prob)

        if traj_cond:
            self.traj_emb = TrajCondEmbedder(class_dropout_prob, traj_size, hidden_size, num_layers=3)

        if cmd_cond:
            self.cmd_emb = CmdCondEmbedder(class_dropout_prob, len(Command), hidden_size)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, x_attn) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear
        if self.hex_emb is not None:
            w = self.hex_emb.embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.hex_emb.embedder.proj.bias, 0)

        if self.cmd_emb is not None:
            nn.init.normal_(self.cmd_emb.embedder.weight, std=0.02)

        # Initialize patch_embed like nn.Linear
        if self.layout_emb is not None:
            w = self.layout_emb.embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.layout_emb.embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        if self.patch_mask is not None:
            bs, _, dim = x.shape
            n = self.patch_mask.shape[1]
            x_padded = torch.zeros(bs, n, dim, device=x.device, requires_grad=True).half()
            indices = self.patch_mask.squeeze(-1).bool().nonzero(as_tuple=True)
            x_padded[indices] = x.view(-1, dim)
            x = x_padded

        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        p_x = p_y = int(x.shape[1] ** 0.5)
        assert p_x * p_y == x.shape[1]

        x = x.reshape(shape=(x.shape[0], p_x, p_y, p, p, c))
        x = torch.einsum('nxypqc->ncxpyq', x)
        imgs = x.reshape(shape=(x.shape[0], c, p_x * p, p_y * p))
        return imgs

    def forward(
        self, x, t, hexplane, layout, cmd, traj,
        inference=False, drop_hex=False, drop_layout=False, drop_traj=False, drop_cmd=False
    ):
        x = self.x_embedder(x) + self.pos_embed  # B, N, D; N = X * Y / patch_size ** 2
        t = self.t_embedder(t)  # B, N

        if self.hex_emb is not None:
            hexplane = self.hex_emb(hexplane, self.training, inference, drop_hex) + self.pos_embed  # B, N, D
        else:
            hexplane = torch.zeros_like(x)

        if self.layout_emb is not None:
            layout = self.layout_emb(layout, self.training, inference, drop_layout) + self.pos_embed  # B, N, D
        else:
            layout = torch.zeros_like(x)

        if self.traj_emb is not None:
            traj = traj.reshape(-1, self.traj_size)
            trajectory = self.traj_emb(traj, self.training, inference, drop_traj)  # B, N
        else:
            trajectory = torch.zeros_like(t)

        if self.cmd_emb is not None:
            command = self.cmd_emb(cmd, self.training, inference, drop_cmd)  # B, N
        else:
            command = torch.zeros_like(t)

        if self.patch_mask is not None:
            bs, _, dim = x.shape
            x = x * self.patch_mask
            x = x[self.patch_mask.squeeze(-1).bool()].view(bs, -1, dim)
            hexplane = hexplane * self.patch_mask
            hexplane = hexplane[self.patch_mask.squeeze(-1).bool()].view(bs, -1, dim)
            layout = layout * self.patch_mask
            layout = layout[self.patch_mask.squeeze(-1).bool()].view(bs, -1, dim)

        c = t.unsqueeze(1) + trajectory.unsqueeze(1) + command.unsqueeze(1) + hexplane + layout  # B, N, D
        for block in self.blocks:
            x = block(x, c, hexplane + layout)  # B, N, D
        x = self.final_layer(x, c)  # B, N, patch_size ** 2 * out_channels
        x = self.unpatchify(x)  # B, out_channels, X, Y

        return x

    def forward_with_cfg(self, x, t, cfg_scale, *args, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, *args, **kwargs)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_DC_1(**kwargs):
    return DiT(depth=18, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_DC_2(**kwargs):
    return DiT(depth=16, hidden_size=384, patch_size=2, num_heads=8, **kwargs)


DiT_models = {
    'DiT-DC/1': DiT_DC_1,
    'DiT-DC/2': DiT_DC_2,
}
