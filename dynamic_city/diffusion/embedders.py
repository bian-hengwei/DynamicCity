import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Embedder(nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.embedder = nn.Identity()

    def drop(self, x):
        if isinstance(x, torch.Tensor):
            shape = x.shape
            device = x.device
        elif isinstance(x, list):
            shape = x[0].shape
            device = x[0].device
        else:
            raise ValueError

        drop_ids = torch.rand(shape[0], device=device) < self.dropout_prob
        drop_ids = drop_ids.unsqueeze(-1)
        if len(shape) == 3:
            drop_ids = drop_ids.unsqueeze(-1)

        if isinstance(x, torch.Tensor):
            x = torch.where(drop_ids, 0, x)
        elif isinstance(x, list):
            x = [torch.where(drop_ids, 0, p) for p in x]
        return x

    def forward(self, x, train, drop_sample=False, drop_all=False):
        x = self.embedder(x)

        if train and self.dropout_prob > 0:
            x = self.drop(x)

        if drop_sample:
            if isinstance(x, torch.Tensor):
                assert x.shape[0] == 2
                x[1] = 0
            elif isinstance(x, list):
                assert x[0].shape[0] == 2
                for p in x:
                    p[1] = 0
            else:
                raise ValueError

        if drop_all:
            if isinstance(x, torch.Tensor):
                x = torch.zeros_like(x)
            elif isinstance(x, list):
                assert x[0].shape[0] == 2
                for p in x:
                    p[:] = 0
            else:
                raise ValueError

        return x


class HexCondEmbedder(Embedder):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, dropout_prob):
        super().__init__(dropout_prob)
        self.embedder = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


class LayoutCondEmbedder(Embedder):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, layout_size, dropout_prob):
        super().__init__(dropout_prob)
        self.img_size = img_size
        self.layout_size = layout_size
        self.embedder = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

    def forward(self, x, *args, **kwargs):
        x = F.interpolate(x, size=(self.layout_size, self.layout_size), mode='nearest')
        pad = self.img_size - self.layout_size
        x = F.pad(x, (0, pad, 0, pad), mode='constant', value=0)
        return super().forward(x, *args, **kwargs)


class TrajCondEmbedder(Embedder):
    def __init__(self, dropout_prob, in_channels, out_channels, num_layers):
        super().__init__(dropout_prob)
        pose_encoder = list()
        pose_encoder.append(nn.Linear(in_channels, out_channels))
        for _ in range(num_layers - 1):
            pose_encoder.append(nn.ReLU())
            pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.embedder = nn.Sequential(*pose_encoder)


class CmdCondEmbedder(Embedder):
    def __init__(self, dropout_prob, num_classes, hidden_size):
        super().__init__(dropout_prob)
        self.embedder = nn.Embedding(num_classes, hidden_size)
