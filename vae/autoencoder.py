import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager


from vae.encoder_decoder import Encoder, Decoder
from vae.util_vae import DiagonalGaussianDistribution



class AutoencoderKL(pl.LightningModule):
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
                 ignore_keys=[],
                 image_key="image",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                 attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=True, in_channels=in_channels,
                 resolution=resolution, z_channels=z_channels, double_z=double_z)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                 attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=True, in_channels=in_channels,
                 resolution=resolution, z_channels=z_channels)

        self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample()

    def decode(self, z, mid_feat=False):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, mid_feat=mid_feat)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec


    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    

    def configure_optimizers(self, lr):
        lr = lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        return opt_ae

