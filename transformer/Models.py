import sys

import torch
import torch.nn as nn
import numpy as np

sys.path.append('.')
from text.symbols import symbols
from transformer.Layers import FFTBlock
import transformer.Constants as Constants

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        # 以下, モデルたちの定義に使用.
        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        # 以下, forwardで使用.
        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        # これはそのまんまembedding.
        # Args:
        #   n_src_vocab: vocab数ですね.
        #   d_word_vec: encoder_hidden. ここに接続するんでしょうね.
        #   padding_idx: paddingに与えるidx. ちゃんとConstantsというファイルで管理. 偉すぎる.
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        # [positional_encoding](https://qiita.com/halhorn/items/c91497522be27bde17ce)
        # 場所をわからせするのに必要みたいですね. 詳細は省.
        # Args:
        #   n_position: ポジション数. 最大文章長が利用される.
        #   d_word_vec: embeddingと同じ. これと足し合わせることになるのでそれはそう.
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,  # 勾配計算を行わない.
        )

        # FFTBlockをとにかく積み重ねる. このblockの詳細はその定義のところで.
        # Args:
        #   d_model: またencoder_hidden.
        #   n_head: head数. attentionの話かな.
        #   d_k = d_v: hidden//head. なにに使うんだ?
        #   d_inner: conv_filter_sizeらしい.
        #   dropout: その通り.
        #   n_layers: そのまんま.
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):
        """
        Args:
          src_seq: textデータ.
          mask: textデータの, 非padding成分を取り出したもの.
        """

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        """
        Examples
          mask.size() = (batch, 149(max_seq_len))
          mask.unsqueeze(1).size() = (batch, 1, 149(max_seq_len))
          # ↓expandは, 増やしたい部分だけリピートして増やしてくれるイメージ.
          # 今回なら, x,z平面上の数字を, y軸方向にリピート.
          # 注意として, memory共有が行われている点に注意. このmaskを使うのは, MultiHeadAttention内.
          mask.unsqueeze(1).expand(-1, max_len, -1) = (batch, 149(max_seq_len), 149(max_seq_len))
        """
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # trainではない(self.trainingはnn.Moduleの変数. model.train()でtrueになる. 便利やん.)
        # 推論の時かつ, 訓練におけるtextの最大長を超えたときにどうするか↓
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            # max_seq_lenは1000程度, max_lenは100ちょいなので基本引っかからない気はする.
            # その場合, train時ではmax_seq_lenでpositionをencodingしていたが,
            # ここではちゃんとmax_lenでencoding.
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            # そうでないときは, position_enc()でmaxlen分の変数の型を用意.
            # 正直上と下で分けているところは, Parameterにしてるか否かの違いしかなさそう.
            # position_encodingではmax_seq_len分positionを作るが, max_lenで切る.
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            # あとは流しに行く.
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        # encodingのコピペやんけ!
        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        # ここも同じ.
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        # ここも同じ.
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        """
        Args:
          enc_seq: variance_adaptorのoutputが入ってくる.
                   shape: (batch, max_seq(durationしたので700とか), hidden)
          mask: mel_masksが入ってくる. mel_lensがNoneでないとき.
          mel_lensは普通にNoneではないみたい.
        """

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        # 条件分岐も同じですね.
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
            # inferenceでmax_seq_lenを超えた場合は, ↓みたいにちょん切りはしない.
        else:
            """
            max_len: enc_seqのshapeのこと.
            trainの場合, max_lenをtargetとしてもらっている.
            そのmax_lenを使って, 出力をpaddingしているため, max_lenに統一されている.
            max_lenは, batchごとのmelの最大値, かな.
            なので, これとmax_seq_lenとの比較は, その名の通りmax_seq_lenまでにするか否か.
            melの長さがmax_seq_lenを超えたら困ることがあるということなのだろうか.
            要検証. vocodar周りとかに関係していそう.


            Examples:
              print(mask.size())
              >>> torch.Size([8, 911])
              print(max_len)
              >>> 911
              print(self.max_seq_len)
              >>> 1000
              print(slf_attn_mask.size())
              >>> torch.Size([8, 911, 911])  # maskをmax_len分expandしただけ.
            """
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

            # 以下がencoderと違う.
            # max_lenでちぎることはencodingでは行っていなかった.
            # どちらも同じFFTBlockなのに, ここがどう影響するんだろうか.
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        # 以下も一緒.
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask


if __name__ == "__main__":
    # Test
    import sys
    import yaml
    from torch.utils.data import DataLoader

    sys.path.append('.')
    from utils.tools import to_device, get_mask_from_lengths
    from dataset import Dataset

    # JSUTをちゃんと読み込みましょう!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/JSUT/preprocess.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/JSUT/train.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("./config/JSUT/model.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    encoder = Encoder(config=model_config).to(device)
    encoder.train()
    for batchs in train_loader:
        for batch in batchs:
            batch = to_device(batch, device)  # これで受け取らないとダメでしょ!
            src_masks = get_mask_from_lengths(batch[4], batch[5])
            output = encoder(batch[3], src_masks)
            break  # testなので一回でやめておきます.
        break

    """
    Examples:
      batchの入力に対するencoderの出力.
      encoderはpos足してmask計算して, FFTにn_layer回入れるだけ.
      FFTではattantion+2_layerかけるだけで、shapeも不変
      意外とわかりやすい.
      ↓実際に, 各FFT後のサイズ. 不変.
      (batch, max_seq_len, encoder_hidden)ですね.
      >>torch.Size([8, 149, 256])
      >>torch.Size([8, 149, 256])
      >>torch.Size([8, 149, 256])
      >>torch.Size([8, 149, 256])

      # decoderのcheck. encoderとの違いがどう影響しているのかを調査.
    """
