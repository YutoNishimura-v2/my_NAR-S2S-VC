import os
import json
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_configs(args, train_config):
    if os.path.exists(os.path.join(train_config["path"]["log_path"], os.path.basename(args.preprocess_config))):
        print(os.path.join(train_config["path"]["log_path"],
              os.path.basename(args.preprocess_config)), " is already exists")
    else:
        shutil.copyfile(args.preprocess_config, os.path.join(train_config["path"]["log_path"],
                                                             os.path.basename(args.preprocess_config)))
    if os.path.exists(os.path.join(train_config["path"]["log_path"], os.path.basename(args.model_config))):
        print(os.path.join(train_config["path"]["log_path"], os.path.basename(args.model_config)), " is already exists")
    else:
        shutil.copyfile(args.model_config, os.path.join(train_config["path"]["log_path"],
                                                        os.path.basename(args.model_config)))
    if os.path.exists(os.path.join(train_config["path"]["log_path"], os.path.basename(args.train_config))):
        print(os.path.join(train_config["path"]["log_path"], os.path.basename(args.train_config)), " is already exists")
    else:
        shutil.copyfile(args.train_config, os.path.join(train_config["path"]["log_path"],
                                                        os.path.basename(args.train_config)))


def to_device(data, device):
    if len(data) == 15:  # train用.
        (
            s_ids,
            t_ids,
            s_sp_ids,
            t_sp_ids,
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            s_pitches,
            s_energies,
            s_durations,
            t_mels,
            t_mel_lens,
            max_t_mel_len,
            t_pitches,
            t_energies,
        ) = data

        s_sp_ids = torch.from_numpy(s_sp_ids).long().to(device)
        t_sp_ids = torch.from_numpy(t_sp_ids).long().to(device)
        s_mels = torch.from_numpy(s_mels).float().to(device)
        s_mel_lens = torch.from_numpy(s_mel_lens).to(device)
        s_pitches = torch.from_numpy(s_pitches).float().to(device)
        s_energies = torch.from_numpy(s_energies).to(device)
        s_durations = torch.from_numpy(s_durations).to(device)
        t_mels = torch.from_numpy(t_mels).float().to(device)
        t_mel_lens = torch.from_numpy(t_mel_lens).to(device)
        t_pitches = torch.from_numpy(t_pitches).float().to(device)
        t_energies = torch.from_numpy(t_energies).to(device)

        return (
            s_ids,
            t_ids,
            s_sp_ids,
            t_sp_ids,
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            s_pitches,
            s_energies,
            s_durations,
            t_mels,
            t_mel_lens,
            max_t_mel_len,
            t_pitches,
            t_energies,
        )

    if len(data) == 8:  # infe用.
        (ids, s_sp_ids, t_sp_ids, s_mels, s_mel_lens, s_mel_max_len, s_pitches, s_energies) = data

        s_sp_ids = torch.from_numpy(s_sp_ids).long().to(device)
        t_sp_ids = torch.from_numpy(t_sp_ids).long().to(device)
        s_mels = torch.from_numpy(s_mels).float().to(device)
        s_mel_lens = torch.from_numpy(s_mel_lens).to(device)
        s_pitches = torch.from_numpy(s_pitches).float().to(device)
        s_energies = torch.from_numpy(s_energies).to(device)

        return (ids, s_sp_ids, t_sp_ids, s_mels, s_mel_lens, s_mel_max_len, s_pitches, s_energies)

    if len(data) == 11:  # infe, duration_forcing用.
        (ids, s_sp_ids, t_sp_ids, s_mels, s_mel_lens, s_mel_max_len, s_pitches, s_energies,
         s_durations, t_mel_lens, max_t_mel_len) = data

        s_sp_ids = torch.from_numpy(s_sp_ids).long().to(device)
        t_sp_ids = torch.from_numpy(t_sp_ids).long().to(device)
        s_mels = torch.from_numpy(s_mels).float().to(device)
        s_mel_lens = torch.from_numpy(s_mel_lens).to(device)
        s_pitches = torch.from_numpy(s_pitches).float().to(device)
        s_energies = torch.from_numpy(s_energies).to(device)
        s_durations = torch.from_numpy(s_durations).to(device)
        t_mel_lens = torch.from_numpy(t_mel_lens).to(device)

        return (ids, s_sp_ids, t_sp_ids, s_mels, s_mel_lens, s_mel_max_len, s_pitches, s_energies,
                s_durations, None, t_mel_lens, max_t_mel_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        # log_step時にlog.txtにも追加する情報.
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        # synth_stepで記述.
        logger.add_figure(tag, fig, step)

    if audio is not None:
        # synth_stepで記述.
        # 正規化して記録.
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    """
    Examples:
      lengths = [2, 4, 6, 1, 9, ...]のように, 各テキストの長さがbatch_size分格納されている.
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    # max_len分の0, 1, 2, 3...のidxを取得.
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    # Examples:
    #   lengths = [[3,3,3,3,3], [2,2,2,2,2], ...] # max_len分, そのテキスト値が引き延ばされる.
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    # ここで得られるマスクは, 要するにpaddingされる前の, 0ではない部分のidxがTrueとなったmask.
    # フツーに, PADを特別な値にすればいい気はしなくもないが, それを共有する方が大変か.

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def mel_denormalize(mels: torch.Tensor, preprocess_config):
    # melの正規化を元に戻す.

    mel_dim = mels.dim()
    if mel_dim == 2:
        # (dim, time)
        mels = mels.unsqueeze(0)

    assert mels.size()[1] == preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "target", "stats.json")
    ) as f:
        stats = json.load(f)
        means = stats["mel_means"]
        stds = stats["mel_stds"]

    for idx, (mean, std) in enumerate(zip(means, stds)):
        mels[:, idx, :] = mels[:, idx, :] * std + mean

    return mels[0] if mel_dim == 2 else mels


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):
    # さすがにbatch一つ目のデータを利用.
    # batch[0][0]で, 後者がバッチのうちの0番目を指定している事に注意.
    # 基本targetのものを利用している.
    basename = targets[1][0]
    mel_len = predictions[9][0].item()
    # melたちは, (time, dim)のように, 最後がmel_channel数.
    mel_target = targets[10][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    if model_config["variance_predictor"]["teacher_forcing"] is False:
        pitch_pre = predictions[2][0, :mel_len].detach().cpu().numpy()
        energy_pre = predictions[3][0, :mel_len].detach().cpu().numpy()
    else:
        pitch_pre = targets[13][0, :mel_len].detach().cpu().numpy()
        energy_pre = targets[14][0, :mel_len].detach().cpu().numpy()
    pitch = targets[13][0, :mel_len].detach().cpu().numpy()
    energy = targets[14][0, :mel_len].detach().cpu().numpy()

    mel_target = mel_denormalize(mel_target, preprocess_config)
    mel_prediction = mel_denormalize(mel_prediction, preprocess_config)

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "target", "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    # pltとして, figを用意.
    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch_pre, energy_pre),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    # 普通にvocoderでinferして渡してあげている.
    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        mel_prediction = mel_denormalize(mel_prediction, preprocess_config)

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "target", "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(  # NOQA
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    # これで, (batch, dim, time)になっている.
    # mel_predictions = mel_denormalize(mel_predictions, preprocess_config)
    # mel_denormalizeは一度やったら保存されるので不要.
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
