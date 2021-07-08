import os
import random
import json

import librosa
import numpy as np
import pyworld as pw
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.source_in_dir = config["path"]["source_prevoice_path"]
        self.target_in_dir = config["path"]["target_prevoice_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        """
        主な変更点
        - speakerの廃止
        - TextGrid関連の廃止
            - それに伴って, 音声区間を切り取る操作や, durationなどが消えてしまったことに注意.
            - さらに, phoneme averageが不可能となったのでこれも廃止.
        """
        os.makedirs((os.path.join(self.out_dir, "source", "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "target", "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "source", "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "target", "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "source", "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "target", "energy")), exist_ok=True)

        print("Processing Data ...")

        # Compute pitch, energy, duration, and mel-spectrogram
        # one-to-oneなので, speakerという概念は不要そう.
        outs = []
        for i, input_dir in enumerate([self.source_in_dir, self.target_in_dir]):
            out = list()
            n_frames = 0
            pitch_scaler = StandardScaler()
            energy_scaler = StandardScaler()
            mel_scalers = [StandardScaler() for _ in range(self.n_mel_channels)]
            source_or_target = ["source", "target"][i]

            for wav_name in tqdm(os.listdir(input_dir)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                # melとかenergyをここで計算.
                ret = process_utterance(input_dir, os.path.join(self.out_dir, source_or_target), basename,
                                        self.sampling_rate, self.hop_length, self.STFT)
                if ret is None:
                    continue
                else:
                    info, pitch, energy, mel = ret
                out.append(info)
                # mel: (80, time)

                if len(pitch) > 0:
                    # partial_fitでオンライン学習. meanとstdを.
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                if len(mel) > 0:
                    for idx, scaler in enumerate(mel_scalers):
                        scaler.partial_fit(mel[idx, :].reshape((-1, 1)))

                n_frames += mel.shape[1]

            print("Computing statistic quantities ...")
            # Perform normalization if necessary
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]

            # melのも.
            mel_means = []
            mel_stds = []

            for scaler in mel_scalers:
                mel_means.append(scaler.mean_[0])
                mel_stds.append(scaler.scale_[0])

            pitch_min, pitch_max = normalize(
                os.path.join(self.out_dir, source_or_target, "pitch"), pitch_mean, pitch_std
            )
            energy_min, energy_max = normalize(
                os.path.join(self.out_dir, source_or_target, "energy"), energy_mean, energy_std
            )
            if source_or_target == "source":
                # targetの時はnormalizeを行わない. 保存はすでに行われていることに注意.
                mel_normalize(
                    os.path.join(self.out_dir, source_or_target, "mel"), mel_means, mel_stds
                )

            # Save files

            with open(os.path.join(self.out_dir, source_or_target, "stats.json"), "w") as f:
                # 結局, 例えばnormalize=Trueだと, min,maxと, normalizeに用いた,
                # originalのmeanとstdが入ってくる.
                stats = {
                    "pitch": [
                        float(pitch_min),
                        float(pitch_max),
                        float(pitch_mean),
                        float(pitch_std),
                    ],
                    "energy": [
                        float(energy_min),
                        float(energy_max),
                        float(energy_mean),
                        float(energy_std),
                    ],
                    "mel_means": mel_means,
                    "mel_stds": mel_stds
                }
                f.write(json.dumps(stats))

            print(
                "{} Data Total time: {} hours {} minutes".format(
                    source_or_target,
                    n_frames * self.hop_length / self.sampling_rate // 3600,
                    n_frames * self.hop_length / self.sampling_rate % 3600 / 60
                )
            )

            random.shuffle(out)
            out = [r for r in out if r is not None]
            out = np.sort(out)
            outs.append(out)

        assert len(outs[0]) == len(outs[1]), "target と sourceのデータ数が合いません."

        outs = np.array(outs)
        index_ = np.random.permutation(len(outs[0]))
        train_outs = outs[:, index_[self.val_size:]]
        valid_outs = outs[:, index_[: self.val_size]]

        for i, source_or_target in enumerate(["source", "target"]):
            # Write metadata
            with open(os.path.join(self.out_dir, source_or_target, "train.txt"), "w", encoding="utf-8") as f:
                for m in train_outs[i]:
                    f.write(m + "\n")
            with open(os.path.join(self.out_dir, source_or_target, "val.txt"), "w", encoding="utf-8") as f:
                for m in valid_outs[i]:
                    f.write(m + "\n")

        print("""
        targetとsourceのtrain.txt, val.txtを確認し, 対応関係が成り立っているか確認してください.
        成り立っていない場合, ファイル名に一貫性がありません.
        np.sortをした際にtargetとsourceが想定通りの対応関係になるような対称的な命名にしましょう.
        """)


def process_utterance(input_dir, out_dir, basename,
                      sampling_rate, hop_length, STFT):
    wav_path = os.path.join(input_dir, "{}.wav".format(basename))

    # Read and trim wav files
    # TextGridがないせいで, 最初と最後切り取りが出来ていないことにも注意.
    wav, _ = librosa.load(wav_path, sr=sampling_rate)

    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        frame_period=hop_length / sampling_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)

    if np.sum(pitch != 0) <= 1:
        return None

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)

    # energyとpitchはここでlogをとる.
    pitch = np.log(pitch+1e-6)
    energy = np.log(energy+1e-6)

    # Save files
    pitch_filename = "pitch-{}.npy".format(basename)
    np.save(os.path.join(out_dir, "pitch", pitch_filename), pitch)

    energy_filename = "energy-{}.npy".format(basename)
    np.save(os.path.join(out_dir, "energy", energy_filename), energy)

    mel_filename = "mel-{}.npy".format(basename)
    np.save(
        os.path.join(out_dir, "mel", mel_filename),
        mel_spectrogram.T,
    )

    return (
        basename,
        remove_outlier(pitch),
        remove_outlier(energy),
        mel_spectrogram,
    )


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


def normalize(in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value


def mel_normalize(in_dir, mel_means, mel_stds):
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        mel = np.load(filename)
        mel = mel.T  # 転置して保存していたので.
        for idx, (mean, std) in enumerate(zip(mel_means, mel_stds)):
            mel[idx, :] = (mel[idx, :] - mean) / std
        np.save(filename, mel.T)
