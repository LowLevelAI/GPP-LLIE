import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager


from vae.encoder_decoder import Encoder, Decoder
from vae.util_vae import DiagonalGaussianDistribution



class CondEncoder(pl.LightningModule):
    def __init__(self,
                 double_z=True,
                 z_channels=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1, 2, 4],
                 num_res_blocks=2,
                 attn_resolutions=[],
                 dropout=0.0,
                 embed_dim=3,
                 image_key="image",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                 attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=True, in_channels=in_channels,
                 resolution=resolution, z_channels=z_channels, double_z=double_z)

        self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)

    def forward(self, x, mid_feat=False):
        if mid_feat:
            h, enc_feat = self.encoder(x, mid_feat=True)
        else:
            h = self.encoder(x)
        
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if mid_feat:
            return posterior.sample(), enc_feat
        else:
            return posterior.sample()


