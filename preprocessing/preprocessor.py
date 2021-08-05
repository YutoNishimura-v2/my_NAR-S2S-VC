import os
import random
import json

import librosa
import numpy as np
import pandas as pd
import pyworld as pw
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from glob import glob

import audio as Audio


class Preprocessor:
    def __init__(self, config, finetuning=False):
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

        self.finetuning = finetuning
        self.multi_speaker = config["preprocessing"]["multi_speaker"]
        self.is_continuous_pitch = config["preprocessing"]["continuous_pitch"]

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
        speakers = []
        none_list = []
        for i, input_dir in enumerate([self.source_in_dir, self.target_in_dir]):
            out = list()
            n_frames = 0
            if self.finetuning is not True:
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
                                        self.sampling_rate, self.hop_length, self.STFT, self.is_continuous_pitch)
                if ret is None:
                    none_list.append(wav_name)
                    continue
                else:
                    info, pitch, energy, mel = ret
                out.append(info)

                if self.multi_speaker is True:
                    # wavのファイル名の先頭に話者名が記載されていることを仮定.
                    # 例: 「JSUT_JSSS_....」なら, JSUT_to_JSSSで, sourceがJSUTであることを示している.
                    speakers.append(basename.split('_')[0])
                    speakers.append(basename.split('_')[1])

                if self.finetuning is not True:
                    if len(pitch) > 0:
                        # partial_fitでオンライン学習. meanとstdを.
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))
                    if len(mel) > 0:
                        for idx, scaler in enumerate(mel_scalers):
                            scaler.partial_fit(mel[idx, :].reshape((-1, 1)))

                n_frames += mel.shape[1]

            if self.finetuning is not True:
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

            else:
                if source_or_target == "source":
                    print("reading from: ", self.config["path"]["source_stats_path"])
                    with open(self.config["path"]["source_stats_path"]) as f:
                        stats = json.load(f)
                        pitch_mean, pitch_std = stats["pitch"][2:]
                        energy_mean, energy_std = stats["energy"][2:]
                        mel_means = stats["mel_means"]
                        mel_stds = stats["mel_stds"]
                else:
                    print("reading from: ", self.config["path"]["source_stats_path"])
                    with open(self.config["path"]["target_stats_path"]) as f:
                        stats = json.load(f)
                        pitch_mean, pitch_std = stats["pitch"][2:]
                        energy_mean, energy_std = stats["energy"][2:]
                        mel_means = stats["mel_means"]
                        mel_stds = stats["mel_stds"]

            pitch_min, pitch_max = normalize(
                os.path.join(self.out_dir, source_or_target, "pitch"), pitch_mean, pitch_std
            )
            energy_min, energy_max = normalize(
                os.path.join(self.out_dir, source_or_target, "energy"), energy_mean, energy_std
            )
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

        # 片方にのみ生き残ったデータなどもあるので注意.
        out_1 = set(outs[0])
        out_2 = set(outs[1])

        out = out_1 & out_2

        if (len(out) < len(out_1)) or (len(out) < len(out_2)):
            print("おそらくprocess_utteranceで一部除かれてしまいました.")
            print("len(out): ", len(out))
            print("len(out_1): ", len(out_1))
            print("len(out_2): ", len(out_2))
            print("out-out_1: ", out-out_1)
            print("out-out_2: ", out-out_2)
            print("out_1-out: ", out_1-out)
            print("out_2-out: ", out_2-out)

        out = np.array(list(out))
        index_ = np.random.permutation(out.shape[0])
        train_outs = out[index_[self.val_size:]]
        valid_outs = out[index_[: self.val_size]]

        for i, source_or_target in enumerate(["source", "target"]):
            # Write metadata
            with open(os.path.join(self.out_dir, source_or_target, "train.txt"), "w", encoding="utf-8") as f:
                for m in train_outs:
                    f.write(m + "\n")
            with open(os.path.join(self.out_dir, source_or_target, "val.txt"), "w", encoding="utf-8") as f:
                for m in valid_outs:
                    f.write(m + "\n")

        print("process_utteranceで弾かれたwavたちです: ", none_list)

        if self.multi_speaker is True:
            speakers = set(speakers)
            with open(os.path.join(self.out_dir, "speakers.txt"), "w", encoding="utf-8") as f:
                for speaker in speakers:
                    f.write(speaker + '\n')
            print("正しく想定したspeakersが記録されたか確認してください.")


def process_utterance(input_dir, out_dir, basename,
                      sampling_rate, hop_length, STFT, is_continuous_pitch):
    wav_path = os.path.join(input_dir, "{}.wav".format(basename))

    # Read and trim wav files
    # TextGridがないせいで, 最初と最後切り取りが出来ていないことにも注意.
    wav, _ = librosa.load(wav_path, sr=sampling_rate)

    # Compute fundamental frequency
    try:
        pitch, t = pw.dio(
            wav.astype(np.float64),
            sampling_rate,
            frame_period=hop_length / sampling_rate * 1000,
        )
    except IndexError:
        print("skipped: ", input_dir, basename)
        return None
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)

    if np.sum(pitch != 0) <= 1:
        return None

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)

    if (mel_spectrogram.shape[1] != energy.shape[0]) or (mel_spectrogram.shape[1] != pitch.shape[0]):
        # ここで一致していないと, 後でエラーになりますので.
        return None

    energy = np.log(energy+1e-6)

    if is_continuous_pitch is True:
        no_voice_indexes = np.where(energy < -5.0)
        pitch[no_voice_indexes] = np.min(pitch)
        pitch = continuous_pitch(pitch)

    pitch = np.log(pitch+1e-6)

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
    for filename in tqdm(os.listdir(in_dir)):
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)

        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))

    return min_value, max_value


def mel_normalize(in_dir, mel_means, mel_stds):
    for filename in tqdm(os.listdir(in_dir)):
        filename = os.path.join(in_dir, filename)
        mel = np.load(filename)
        mel = mel.T  # 転置して保存していたので.
        for idx, (mean, std) in enumerate(zip(mel_means, mel_stds)):
            mel[idx, :] = (mel[idx, :] - mean) / std
        np.save(filename, mel.T)


def continuous_pitch(pitch: np.ndarray) -> np.ndarray:
    # 0の値をとったらnan扱いとして, 線形補完を行ってみる.
    pitch = np.where(pitch < 1e-6, np.nan, pitch)

    df = pd.Series(pitch)
    df = df.interpolate()

    first_value = pitch[df.isnull().values.tolist().index(False)]
    df = df.fillna(first_value)

    pitch = df.values

    return pitch


def wav_path_matching(source_wav_path, target_wav_path):
    source_wavs = set([os.path.basename(p) for p in glob(os.path.join(source_wav_path, "*.wav"))])
    target_wavs = set([os.path.basename(p) for p in glob(os.path.join(target_wav_path, "*.wav"))])

    union = source_wavs & target_wavs

    if len(union) < len(source_wavs):
        print("こちらに対応するwavがtargetにないようです: ", source_wavs-union)
        exit(0)

    if len(union) < len(target_wavs):
        print("こちらに対応するwavがsourceにないようです: ", target_wavs-union)
        exit(0)

    print("sourceとtargetのファイル名に問題はありませんでした")
