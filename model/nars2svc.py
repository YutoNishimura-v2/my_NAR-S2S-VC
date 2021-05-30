import os
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
from conformer.Models import Encoder, Decoder
from conformer.Layers import PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class NARS2SVC(nn.Module):
    """ NAR-S2S-VC """

    def __init__(self, preprocess_config, model_config):
        super().__init__()

        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear_1 = nn.Linear(
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            model_config["conformer"]["encoder_hidden"],
        )
        self.mel_linear_2 = nn.Linear(
            model_config["conformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

    def forward(
        self,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        s_pitches,
        s_energies,
        t_mels=None,
        t_mel_lens=None,
        max_t_mel_len=None,
        t_pitches=None,
        t_energies=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        s_mel_masks = get_mask_from_lengths(s_mel_lens, max_s_mel_len)
        # PAD前の, 元データが入っている部分がTrueになっているmaskの取得
        t_mel_masks = (
            get_mask_from_lengths(t_mel_lens, max_t_mel_len)
            if t_mel_lens is not None
            else None
        )

        output = self.mel_linear_1(s_mels)

        output = self.encoder(s_mels, s_mel_masks)

        # (
        #     output,
        #     p_predictions,
        #     e_predictions,
        #     log_d_predictions,
        #     d_rounded,
        #     mel_lens,
        #     mel_masks,
        # ) = self.variance_adaptor(
        #     output,
        #     s_mel_masks,
        #     t_mel_masks,
        #     max_t_mel_len,
        #     t_pitches,
        #     t_energies,
        #     p_control,
        #     e_control,
        #     d_control,
        # )

        # ここまでのoutputは, (batch, mel_len+pad, hidden)となっている.
        # masksはtargetのもの.なければmel_lensから作成.
        output, t_mel_masks = self.decoder(output, t_mel_masks)
        # ここは, hiddenの次元をmelのchannel数にあわせる. ここでは256→80
        output = self.mel_linear_2(output)

        # postnetできれいにするのかな？で完成.
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )