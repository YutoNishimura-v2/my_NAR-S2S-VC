"""
inference専用のpreprocesing

input_pathとoutput_pathを指定したらあとは勝手にやってほしい. preprocessのconfigもやね.
"""
import os
import json

import audio as Audio
from preprocessor.n2c_voiceprocess import load_and_save, delete_novoice_from_path
from preprocessor.preprocessor import process_utterance, normalize, mel_normalize


def inference_preprocess(input_path, output_path, preprocess_config):
    """
    想定: input_pathには, Cubaseから落とした, ノイズ処理だけした生のデータがためられている.
    """
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    STFT = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    print("Processing Data ...")

    os.makedirs(output_path, exist_ok=True)

    # まず, 音声に対する前処理.

    # サンプリングレートの変更.
    load_and_save(input_path, output_path, sr)

    # 無音区間の削除
    delete_novoice_from_path(output_path, output_path)

    # ここから, 入力に用いるmelやpitchなどを作りに行く.

    os.makedirs((os.path.join(output_path, "mel")), exist_ok=True)
    os.makedirs((os.path.join(output_path, "pitch")), exist_ok=True)
    os.makedirs((os.path.join(output_path, "energy")), exist_ok=True)

    # Compute pitch, energy, duration, and mel-spectrogram
    # one-to-oneなので, speakerという概念は不要そう.
    out = []
    n_frames = 0

    for wav_name in os.listdir(output_path):  # output_pathに処理済みのwavがいることに注意.
        if ".wav" not in wav_name:
            continue

        basename = wav_name.split(".")[0]
        # melとかenergyをここで計算.
        ret = process_utterance(input_path, output_path, basename,
                                sr, hop_length, STFT)
        if ret is None:
            continue
        else:
            info_, _, _, mel = ret
        # mel: (80, time)
        n_frames += mel.shape[1]
        out.append(info_)

    print("Computing statistic quantities ...")
    # Perform normalization if necessary

    # 訓練時の正規化に用いた値を利用.
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "source", "stats.json")
    ) as f:
        stats = json.load(f)
        pitch_mean, pitch_std = stats["pitch"][2:]
        energy_mean, energy_std = stats["energy"][2:]
        mel_means = stats["mel_means"]
        mel_stds = stats["mel_stds"]

    normalize(os.path.join(output_path, "pitch"), pitch_mean, pitch_std)
    normalize(os.path.join(output_path, "energy"), energy_mean, energy_std)
    mel_normalize(os.path.join(output_path, "mel"), mel_means, mel_stds)

    print(
        "Data Total time: {} hours {} minutes".format(
            n_frames * hop_length / sr // 3600,
            n_frames * hop_length / sr % 3600 / 60
        )
    )

    with open(os.path.join(output_path, "inference.txt"), "w", encoding="utf-8") as f:
        for m in out:
            f.write(m + "\n")
