import sys
import torch
import torch.nn as nn

sys.path.append('.')
from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor

    論文にあったように, 特徴を計算する際に, train, validの時は
    targetの値をちゃんと挙げていることに注意.
    """

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()
        # duration, pitch, energyで共通なのね.
        self.duration_predictor = VariancePredictor(model_config, "duration")
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config, "pitch")
        self.energy_predictor = VariancePredictor(model_config, "energy")
        self.pitch_conv1d_1 = Conv(1, model_config["conformer"]["encoder_hidden"])  # inputをhiddenに.
        self.pitch_conv1d_2 = Conv(1, model_config["variance_predictor"]["filter_size"])  # predictをhiddenに.
        self.energy_conv1d_1 = Conv(1, model_config["conformer"]["encoder_hidden"])
        self.energy_conv1d_2 = Conv(1, model_config["variance_predictor"]["filter_size"])

    def forward(
        self,
        x,
        src_mask,
        src_pitch,
        src_energy,
        src_duration=None,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        # まずは, durationを計算する.
        log_duration_prediction = self.duration_predictor(x, src_mask)

        # convして, 次元を合わせる
        pitch_conv = self.pitch_conv1d_1(src_pitch)
        energy_conv = self.energy_conv1d_1(src_energy)

        # durationの正解データがあるのであれば, targetとともにreguratorへ.
        if src_duration is not None:
            x, mel_len = self.length_regulator(x, src_duration, max_len)
            pitch, _ = self.length_regulator(pitch_conv, src_duration, max_len)
            energy, _ = self.length_regulator(energy_conv, src_duration, max_len)
            duration_rounded = src_duration
        else:
            # そうでないなら, predictionを利用.
            duration_rounded = torch.clamp(  # 最小値を0にする. マイナスは許さない.
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            # そして, predictで作ったduration_roundedを使ってregulatorへ.
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            pitch, _ = self.length_regulator(pitch_conv, duration_rounded, max_len)
            energy, _ = self.length_regulator(energy_conv, duration_rounded, max_len)
            # inferenceではmel_maskもないので, Noneとしてくる.
            mel_mask = get_mask_from_lengths(mel_len)

        pitch += x
        energy += x

        # pitch, energyを計算
        pitch_prediction = self.pitch_predictor(pitch, mel_mask) * p_control
        energy_prediction = self.energy_predictor(energy, mel_mask) * e_control

        # pitchを, また次元増やしてhiddenに足す.
        if pitch_target is not None:
            # 正解データがある場合はそちらを利用してあげる.
            pitch = self.pitch_conv1d_2(pitch_target)
            energy = self.energy_conv1d_2(energy_target)
        else:
            pitch = self.pitch_conv1d_2(pitch_prediction)
            energy = self.energy_conv1d_2(energy_prediction)

        x = x + pitch + energy

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

    ふっつーの変換層. 確かに大層な変換は必要なさそうではあるが.
    durationに関してはもっといい奴が必要そうな気はする.
    """

    def __init__(self, model_config, mode="duration"):
        super(VariancePredictor, self).__init__()

        assert mode in ["duration", "pitch", "energy"]
        self.input_size = model_config["conformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"][mode]["kernel_size"]
        self.layer_num = model_config["variance_predictor"][mode]["layer_num"]
        # ↓ここはhiddenと一致していないとね.
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        conv_layers = []

        for i in range(self.layer_num):
            if i == 0:
                conv = Conv(
                    self.input_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            else:
                conv = Conv(
                    self.filter_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2,  # same sizeに.
                )
            conv_layers += [conv, nn.ReLU(), nn.LayerNorm(self.filter_size),
                            nn.Dropout(self.dropout)]

        self.conv_layer = nn.Sequential(*conv_layers)

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
        if len(x.size()) == 2:
            x = x.contiguous().view(x.size()[0], x.size()[1], 1)
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

    from utils.tools import to_device
    from dataset import TrainDataset
    from model.nars2svc import NARS2SVC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/N2C/preprocess.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/N2C/train.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("./config/N2C/model.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )

    train_dataset = TrainDataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    model = NARS2SVC(preprocess_config, model_config)
    model.train()
    model = model.to(device)
    for batchs in train_loader:
        for batch in batchs:
            batch = to_device(batch, device)  # これで受け取らないとダメでしょ!
            output = model(*(batch[2:]))
