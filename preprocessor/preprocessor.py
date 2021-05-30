import os
import random
import json
import sys

import librosa
import numpy as np
from numpy.lib.index_tricks import _ix__dispatcher
from numpy.lib.utils import source
import pyworld as pw
from sklearn.preprocessing import StandardScaler

sys.path.append('.')
import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.source_in_dir = config["path"]["source_raw_path"]
        self.target_in_dir = config["path"]["target_raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

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
            mel_scaler = StandardScaler()
            source_or_target = ["source", "target"][i]

            for wav_name in os.listdir(input_dir):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                ret = self.process_utterance(source_or_target, input_dir, basename)
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

                n_frames += mel.shape[1]

            print("Computing statistic quantities ...")
            # Perform normalization if necessary
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                energy_mean = 0
                energy_std = 1

            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, source_or_target, "pitch"), pitch_mean, pitch_std
            )
            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, source_or_target, "energy"), energy_mean, energy_std
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

    def process_utterance(self, source_or_target, input_dir, basename):
        """
        Args:
          idx: source, targetの識別子. idx==0ならsource.
        """
        wav_path = os.path.join(input_dir, "{}.wav".format(basename))

        # Read and trim wav files
        # TextGridがないせいで, 最初と最後切り取りが出来ていないことにも注意.
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

        # Save files
        pitch_filename = "pitch-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, source_or_target, "pitch", pitch_filename), pitch)

        energy_filename = "energy-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, source_or_target, "energy", energy_filename), energy)

        mel_filename = "mel-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, source_or_target, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            basename,
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram,
        )

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

