import sys
from typing import Tuple

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

sys.path.append('.')

from conformer.Modules import (
    ResidualConnectionModule,
    LayerNorm,
)
from conformer.SubLayers import (
    FeedForwardModule,
    MultiHeadedSelfAttentionModule,
    ConformerConvModule,
)


class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
          まぁつまるところ, hiddenだよね.
        num_attention_heads (int, optional): Number of attention heads
          これは単なるattention heads. paperでは2としている.
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
          feed_forward時に次元を増やして減らしてをやるが, その比率. paperでは4.
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
          同上. paperでは2.
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
          dropout. 特に記載はなかったので, defaultでよいかも. FastSpeech2では0.2.
        attention_dropout_p (float, optional): Probability of attention module dropout
          同上.
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
          同上.
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
          これは, paperでは7となっていた.
        half_step_residual (bool): Flag indication whether to use half step residual or not
          記載なし. defaultで良さそう.

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(
            self,
            encoder_dim: int = 512,
            attention_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.FF_1 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )
        self.attention = ResidualConnectionModule(
            module=MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                d_attention=attention_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
            ),
            attention=True
        )
        self.conv = ResidualConnectionModule(
            module=ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            ),
        )
        self.FF_2 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )
        self.layer_norm = LayerNorm(encoder_dim)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        output = self.FF_1(input)
        output, attn = self.attention(output, mask)
        output = self.conv(output)
        output = self.FF_2(output)
        output = self.layer_norm(output)

        return output, attn


class ConvNorm(torch.nn.Module):
    # PostNetで利用.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5

    いたってシンプルなconv層. 分かれてるのは, 入出力のchannelを合わせているだけ.
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
