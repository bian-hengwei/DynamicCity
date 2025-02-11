import torch
import torch.nn as nn
from einops import rearrange

from dynamic_city.vae.decoder import ConvDecoder
from dynamic_city.vae.encoder import TrEncoder


class DynamicCityAE(nn.Module):
    """
    Main class for the DynamicCity autoencoder.
    Enables easy switching between different encoders and decoders and optional VAE.
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        # encoder
        encoder_type = {
            'tr': TrEncoder,
        }[conf.model.encoder_type]
        self.encoder = encoder_type(conf)

        # vae
        if conf.model.vae:
            self.vae = HexPlaneVAE(conf)

        # decoder
        decoder_type = {
            'conv': ConvDecoder,
        }[conf.model.decoder_type]
        self.decoder = decoder_type(conf)

    def forward(self, voxels):
        hexplane = self.encoder(voxels)

        mus, logvars = None, None
        if self.conf.model.vae:
            hexplane, mus, logvars = self.vae(hexplane)

        pred = self.decoder(hexplane)

        return dict(
            hexplane=hexplane,
            mus=mus,
            logvars=logvars,
            pred=pred,
        )


class HexPlaneVAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.fc_mu_logvar = nn.ModuleList(
            [nn.Linear(conf.model.latent_channels, conf.model.latent_channels * 2) for _ in range(6)]
        )

    def forward(self, x):
        mus, logvars = list(), list()
        for i, plane in enumerate(x):  # B, C, X, Y * 6
            b, c, dim1, dim2 = plane.shape
            plane = rearrange(plane, 'b c d1 d2 -> b (d1 d2) c')

            mu_logvar = self.fc_mu_logvar[i](plane)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
            mus.append(rearrange(mu, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))
            logvars.append(rearrange(logvar, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))

        x = [self.latent_sample(mus[i], logvars[i]) for i in range(len(x))]
        return x, mus, logvars

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
