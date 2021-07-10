import os
import sys

import torch
import torch.nn as nn
from conformer.Layers import PostNet
from conformer.Models import Decoder, Encoder
from utils.tools import get_mask_from_lengths

sys.path.append('.')
from .modules import VarianceAdaptor


class NARS2SVC(nn.Module):
    """ NAR-S2S-VC """

    def __init__(self, preprocess_config, model_config):
        super().__init__()

        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
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

        n_speaker = 2

        if os.path.exists(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.txt")):
            n_speaker = 0
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.txt"), "r",
                      encoding="utf-8") as f:
                for _ in f.readlines():
                    n_speaker += 1

        self.speaker_emb = nn.Embedding(
            n_speaker,
            model_config["conformer"]["encoder_hidden"],
        )
        assert model_config["conformer"]["encoder_hidden"] == model_config["conformer"]["decoder_hidden"]

    def forward(
        self,
        s_sp_ids,
        t_sp_ids,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        s_pitches,
        s_energies,
        s_durations=None,
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

        output = output + self.speaker_emb(s_sp_ids).unsqueeze(1).expand(-1, max_s_mel_len, -1)

        output = self.encoder(output, s_mel_masks)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            t_mel_lens,
            t_mel_masks,
        ) = self.variance_adaptor(
            output,
            s_mel_masks,
            s_pitches,
            s_energies,
            s_durations,
            t_mel_masks,
            max_t_mel_len,
            t_pitches,
            t_energies,
            p_control,
            e_control,
            d_control,
        )

        output = output + self.speaker_emb(t_sp_ids).unsqueeze(1).expand(-1, torch.max(t_mel_lens), -1)

        # ここまでのoutputは, (batch, mel_len+pad, hidden)となっている.
        # masksはtargetのもの.なければmel_lensから作成.
        output = self.decoder(output, t_mel_masks)
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
            s_mel_masks,
            t_mel_masks,
            s_mel_lens,
            t_mel_lens,
        )
