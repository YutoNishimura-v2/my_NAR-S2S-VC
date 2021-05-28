"""Conformerを利用したEncoder, Decoder

おそらく, transformerで行っている, position_encは必要ないはず. 内部で行っているので.

基本的に, transformer/Models.pyを使う.
"""
import sys

import torch.nn as nn

sys.path.append('.')
from conformer.Layers import ConformerBlock

class Encoder(nn.Module):
    def __init__(self, config):
        """
            ↓conformer blockのinput

            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            device: str = 'cuda',
        """
        super().__init__()

        n_layers = config["conformer"]["encoder_layer"]  # paper: 4

        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    # d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )