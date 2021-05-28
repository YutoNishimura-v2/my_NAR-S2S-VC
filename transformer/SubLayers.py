import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        # ↓BatchNormの別バージョン. 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Examples:
          FFTBlockから流れてくるのは, enc_input, enc_input, enc_input, mask=slf_attn_mask
          こちらの方々.
          また, 
          d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
          )
          n_head = config["transformer"]["encoder_head"]
          ここら辺の値. decoderも同様.
          enc_inputのshapeは, src_word_embを見ればよくて, こいつは, idをembにするので, d_word_vec,
          つまり,d_word_vec = config["transformer"]["encoder_hidden"]へと次元を拡大するので,
          texts.size() = (batch, max_seq_len)だったものを,
          (batch, max_seq_len, d_word_vec)にする. これに対して, pos_embをbatch方向にexpandしたものを
          足していることに注意.

          結局, enc_inputのshapeは, (batch, max_seq_len, encoder_hidden)である.
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # 各q,k,vを, Linerで, encoder_hiddenのサイズに.
        """
        nn.Linear: 実はこいつ, n次元にも適用出来て, その場合は最終次元にだけ影響する.
        Examples:
          今回の場合だと, d_model, n_head * d_kとあるが, どっちもencoder_hiddenではある.

        なので, 
        self.w_qs(q).size() = (batch, max_seq_len(len_qとか), encoder_hidden(n_head*d_k))
        であって下のようなviewが可能.

        [contiguousについて](https://qiita.com/kenta1984/items/d68b72214ce92beebbe2)
        viewは, メモリが並んでないと使えないので, contiguousで並べる.
        こういう時にしか不都合は起きないので, 普段は気にしなくてよさそう.
        """
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # repeatは, 何倍するかみたいなお話. なので, expandではkeepを-1としたが, ここでは1
        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        # maskはあくまでmasked_fillにしか使わないので, expandとかしてもなんの問題もなさそう.
        # 正直怖くてあまり使いたくないけどね.
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)  # ここでlayer_norm

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module 
    
    まじで, ただの2層構造だった.
    """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,  # 計算式を見ればわかるが, このpaddingを行えば大きさ不変
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)  # またlayer_norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # ここでもresidualか.
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
