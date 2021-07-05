# wavが大量に入ったフォルダに対して, mel化の処理を施す.
# もちろん, configはNARS2Sのものを利用する.
import os
import sys
from glob import glob

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append('.')
from hifigan.meldataset import load_wav, mel_spectrogram
from preprocessing.n2c_voiceprocess import load_and_save


def main(args, preprocess_config):
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    os.makedirs(args.pre_voice_path, exist_ok=True)

    # sampling rateを変更する.
    # そのために, 一時フォルダを用意.
    print("\nchanging sampling rate")
    load_and_save(args.input_path, args.pre_voice_path, sr)

    # melにする.
    wav_paths = glob(os.path.join(args.pre_voice_path, '*.wav'))

    mels_dir = os.path.join(args.output_path, 'mels')
    os.makedirs(mels_dir, exist_ok=True)

    print("\nmel save...")

    n_fft = preprocess_config["preprocessing"]["stft"]["filter_length"]
    num_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
    hop_size = preprocess_config["preprocessing"]["stft"]["hop_length"]
    win_size = preprocess_config["preprocessing"]["stft"]["win_length"]
    fmin = preprocess_config["preprocessing"]["mel"]["mel_fmin"]
    fmax = preprocess_config["preprocessing"]["mel"]["mel_fmax"]

    for wav_path in tqdm(wav_paths):
        audio, sampling_rate = load_wav(wav_path, sr)
        assert sampling_rate == sr
        audio = audio / 32768.0
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        mel = mel_spectrogram(audio, n_fft, num_mels, sr,
                              hop_size, win_size, fmin, fmax)

        file_name = os.path.basename(wav_path).replace('.wav', '')
        np.save(os.path.join(mels_dir, file_name), mel)

    # train, valに分ける.
    indexes = np.random.permutation(len(wav_paths))
    train_path = np.array(wav_paths)[indexes[args.val_num:]]
    val_path = np.array(wav_paths)[indexes[:args.val_num]]

    with open(os.path.join(args.output_path, 'train.txt'), 'w', encoding='utf-8') as f:
        for path_ in train_path:
            file_name = os.path.basename(path_).replace('.wav', '')
            f.write(file_name+"\n")

    with open(os.path.join(args.output_path, 'val.txt'), 'w', encoding='utf-8') as f:
        for path_ in val_path:
            file_name = os.path.basename(path_).replace('.wav', '')
            f.write(file_name+"\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str
    )
    parser.add_argument(
        '--pre_voice_path',
        type=str,
    )
    parser.add_argument(
        '--output_path',
        type=str,
    )
    parser.add_argument(
        '-p',
        '--preprocess_config',
        type=str
    )
    parser.add_argument(
        '--val_num',
        type=int,
        default=1000
    )

    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding='utf-8'), Loader=yaml.FullLoader
    )

    main(args, preprocess_config)
