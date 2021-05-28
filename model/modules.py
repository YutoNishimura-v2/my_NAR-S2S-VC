import os
import json
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

sys.path.append('.')
from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor 

    論文にあったように, 特徴を計算する際に, train, validの時は
    targetの値をちゃんと挙げていることに注意.
    """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        # duration, pitch, energyで共通なのね.
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        # default: pitch: feature: "phoneme_level"
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        # default: variance_embedding: pitch_quantization: "linear"
        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        # default: n_bins: 256
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        # default: path: preprocessed_path: "./preprocessed_data/JSUT"
        with open(
            os.path.join(preprocess_config["path"]
                         ["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        # logなんだったら, logとってからexpにかけろ!
        # pitch_bins: pitrchの最小値最大値までの値をn_binsで分割して配列に.
        # pitch_min,maxはデータセット全体で見たなので共通.
        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min),
                                   np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min),
                                   np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        # pitchとenergyに関してはなんとembedding!
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):
        """
        Examples:
          print(self.pitch_bins)  # mean=0 とstd=1二はしているが, だからと言って, max+min=0になるわけではないよ.
          >>> [-3.2176e+00, -3.1793e+00, -3.1410e+00, -3.1027e+00, -3.0644e+00,
          ... ,6.3553e+00,  6.3936e+00,  6.4319e+00,  6.4702e+00,  6.5085e+00],
          print(self.pitch_bins.size())
          >>> torch.Size([255])
          print(target)  # マイナスもあるよ.
          >>> [ 1.7435e-01,  1.7435e-01,  1.7435e-01,  1.7435e-01,  1.7435e-01,
          1.7435e-01,  4.7183e-01,  1.1367e+00,  1.8390e+00,  1.6053e+00,
          print(target.size())
          >>> torch.Size([8, 106])
          print(torch.bucketize(target, self.pitch_bins))
          >>> [ 89,  89,  89,  89,  89,  89,  97, 114, 133, 126,  94,  56,  57,  57,
          53,  52,  55,  58,  62,  68,  86, 117, 115, 113,  94,  63,  52,  41,
          print(torch.bucketize(target, self.pitch_bins).size())
          >>> torch.Size([8, 106])
        """
        # まずは予測を行う.
        prediction = self.pitch_predictor(x, mask)
        if target is not None:  # つまりtrain時.
            # bucketizeで, その値がどのself.pitch_binsの間に入っているかを調べて, そのindexを返す.
            # 例: target = 2, pitch_bins = 1,3,5
            # なら, returnは, 1となる.
            # 要するに, pitchの大きさに対する特徴を学習させようとしている.
            # embeddingで次元を膨らませている感じ(既に, targetだけでpitchの値はあるにはあるので)
            embedding = self.pitch_embedding(
                torch.bucketize(target, self.pitch_bins))
        else:  # inference時.
            # 同様に, こっちはpredictionに対して. それはそうだが.
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        # pitchとやっていることは同じ.
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(
                torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        """
        inputに対して, 
        - durationならlogとして扱うので, expしてから, inputとともにleguratorへ.
        - pitch, energyは, predictして, それを直接足すのではなく, embeddingしてから足す.
            - 確かにそのほうが細かい違いよりもう少し大きい特徴をとらえてくれそう.
            - あとは単に次元を合わせる目的もあるのかな.

        全てに共通して, ちゃんとtrainではtargetの値を用いて補正していることに注意.

        pitch, energyは, normalize値の予測なのでマイナスおっけー.
        durationは, もちろん全て正の数を予測するので, logとして出力を解釈してexpする.

        Examples:
          x: torch.Size([8, 124, 256])
          src_mask: torch.Size([8, 124])  # 音韻ごとに分けたもの.

          log_duration_prediction: [ 6.8136e-01,  6.4592e-01, -1.4724e+00, -5.2702e-01,  1.6697e-01,
         -1.0734e+00,  1.0511e-01, -9.8899e-01, -1.2230e+00, -5.7782e-01,
          log_duration_prediction: torch.Size([8, 124])  # 音韻ごとに, どれだけdurationがあるかという.
          # durationは, その音韻がどれくらいの長さのmelに対応しているかを表す.
          duration_target: [ 2,  3,  8,  6,  3,  3,  6,  8,  4,  5, 10,  3,  3, 13, 25,  4,  5,  2,
         11,  6,  5, 10,  3,  6,  8,  9,  9,  4,  9,  6,  6,  3,  9,  3,  8,  6,
          7,  7,  5,  6, 16,  4,  3, 11,  3,  3, 14, 34,  4,  6, 11,  6,  3, 14,
          5,  5,  8,  7,  5,  7,  6,  2,  5,  6,  9,  3,  5,  8,  3,  8,  9, 11,
          8, 12,  4,  9,  3,  4,  7,  7,  4,  6, 11,  6,  6,  7,  7,  3,  7,  3,
          5,  6,  6,  5,  5,  4,  8,  5,  5,  6,  5,  7,  3, 12, 11,  4],

          pitch_prediction: [ 2.0979e-01,  5.6083e-01,  1.5123e+00,  2.2033e-01,  5.5890e-01,
         -9.8557e-01,  4.2298e-01, -4.1248e-01, -9.0365e-01, -1.5142e-01,
          pitch_prediction:  torch.Size([8, 124])

          pitch_embedding: [-0.8416, -0.4116, -1.0908,  ...,  1.6614,  0.5056, -0.6048],
         [-0.8416, -0.4116, -1.0908,  ...,  1.6614,  0.5056, -0.6048],
          pitch_embedding:  torch.Size([8, 124, 256])
        """
        # まずは, durationを計算する.
        log_duration_prediction = self.duration_predictor(x, src_mask)

        # pitchがphoneme, つまりframe_levelではない場合
        if self.pitch_feature_level == "phoneme_level":
            # get_pitch_embeddingはこのクラスの関数.
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding

        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        # durationの正解データがあるのであれば, targetとともにreguratorへ.
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            # そうでないなら, predictionを利用.
            duration_rounded = torch.clamp(  # 最小値を0にする. マイナスは許さない.
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            # そして, predictで作ったduration_roundedを使ってregulatorへ.
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            # inferenceではmel_maskもないので, Noneとしてくる.
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            # frame_levelなら, 一気に見るので, src_maskではなく, mel_maskを見てもらう.
            # mel_maskじゃないと次元も合わないよね.
            # 違いはそこだけ.
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            # embbeddingのほうを足しておく.
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator 
    
    Examples:
      print(output.size())
      >>> torch.Size([8, 987, 256])
      print(torch.LongTensor(mel_len).to(device).size())
      >>> torch.Size([8]) # batchごとに. padding前のもの.
      print(mel_len)
      >>> [714, 670, 626, 626, 711, 563, 589, 595]
    """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            # expandedのlenがまさに出力したいmelの幅になる.
            mel_len.append(expanded.shape[0])

        # ここでは, まだoutputは長さバラバラのlistであることに注意. 
        # 長さを揃えなきゃ.
        if max_len is not None:
            # max_lenがあるなら, それでpad.
            output = pad(output, max_len)
        else:
            # targetがないならmax_lenもないですね.
            # その場合は自動で一番長い部分を探してくれる.
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        """
        音韻を表すデータx(ここではbatch)を, 実際にduration分引き延ばしてあげる関数. 
        各音韻をforで愚直にみる.
        x[i] = (hidden)
        これに対して, duration分expand
        x[i] = (duration, hidden)
        そして最後に合体
        out = (sum duration, hidden)

        Args:
          batch: 1dataのinput(batchはLRのほうでとっている).(max_seq_len, hidden)
          predicted: 1dataのpredict.

        Examples:
          print(predicted.size())
          >>> torch.Size([102])
          print(batch.size())
          >>> torch.Size([102, 256])
          print(vec)
          >>> tensor([-0.4082,  2.0394,  2.8549, -0.4010, -0.7433, -0.7538,  0.1471, -1.9889,
          print(predicted[i])
          >>> tensor(3, device='cuda:0')
          print(out.size())
          >>> torch.Size([911, 256])
        """
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)  # listをtorchの2次元へ, かな.

        return out

    def forward(self, x, duration, max_len):
        # durationがpredictか, targetのdurationです.
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor

    transformerでもない, ふっつーの変換層. 確かに大層な変換は必要なさそうではあるが.
    durationに関してはもっといい奴が必要そうな気はする.
    """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        # ↓ここはhiddenと一致していないとね.
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(  # なんでわざわざ? 命名のためかな?
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,  # same sizeに.
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


if __name__ == "__main__":
    # Test
    import sys
    import torch
    import yaml
    from torch.utils.data import DataLoader

    sys.path.append('.')
    from utils.tools import to_device
    from dataset import Dataset
    from model.fastspeech2 import FastSpeech2

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

    model = FastSpeech2(preprocess_config, model_config)
    model.train()
    model = model.to(device)
    for batchs in train_loader:
        for batch in batchs:
            batch = to_device(batch, device)  # これで受け取らないとダメでしょ!
            output = model(*(batch[2:]))
