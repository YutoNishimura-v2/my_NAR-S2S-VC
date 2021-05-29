"""Conformerを利用したEncoder, Decoder

おそらく, transformerで行っている, position_encは必要ないはず. 内部で行っているので.

基本的に, transformer/Models.pyを使う.
"""
from conformer.Layers import ConformerBlock
import sys

import torch.nn as nn

sys.path.append('.')


class Encoder(nn.Module):
    def __init__(self, config):
        """
        """
        super().__init__()

        d_model = config["conformer"]["encoder_hidden"]
        n_head = config["conformer"]["encoder_head"]
        ff_expansion = config["conformer"]["ff_expansion_factor"]  # paper: 4
        conv_expansion = config["conformer"]["conv_expansion_factor"]  # paper: 2
        ff_dropout = config["conformer"]["ff_dropout"]  # FFT: 0.2
        attention_dropout = config["conformer"]["attention_dropout"]  # FFT: 0.2
        conv_dropout = config["conformer"]["conv_dropout"]  # FFT: 0.2
        kernel_size = config["conformer"]["conv_kernel_size"]
        n_layers = config["conformer"]["encoder_layer"]  # paper: 4

        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, n_head, ff_expansion, conv_expansion, ff_dropout,
                    attention_dropout, conv_dropout, kernel_size
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_output, mask, return_attns=False):
        enc_slf_attn_lst = []
        _, max_len, _ = enc_output.size()
        # ここでもとめられているmaskは2次元にしたmask.
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=slf_attn_mask
            )
            if return_attns is True:
                enc_slf_attn_lst += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    def __init__(self, config):
        """
        """
        super().__init__()

        d_model = config["conformer"]["decoder_hidden"]
        n_head = config["conformer"]["decoder_head"]
        ff_expansion = config["conformer"]["ff_expansion_factor"]  # paper: 4
        conv_expansion = config["conformer"]["conv_expansion_factor"]  # paper: 2
        ff_dropout = config["conformer"]["ff_dropout"]  # FFT: 0.2
        attention_dropout = config["conformer"]["attention_dropout"]  # FFT: 0.2
        conv_dropout = config["conformer"]["conv_dropout"]  # FFT: 0.2
        kernel_size = config["conformer"]["conv_kernel_size"]
        n_layers = config["conformer"]["decoder_layer"]  # paper: 4

        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, n_head, ff_expansion, conv_expansion, ff_dropout,
                    attention_dropout, conv_dropout, kernel_size
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, dec_output, mask, return_attns=False):
        dec_slf_attn_lst = []
        _, max_len, _ = dec_output.size()
        # ここでもとめられているmaskは2次元にしたmask.
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=slf_attn_mask
            )
            if return_attns is True:
                dec_slf_attn_lst += [dec_slf_attn]

        return dec_output
