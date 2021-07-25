import os
import sys
from glob import glob

import numpy as np
# import torch
import yaml
# from tqdm import tqdm

sys.path.append('.')
# from hifigan.meldataset import load_wav, mel_spectrogram
from preprocessing.n2c_voiceprocess import load_and_save


def main(args, preprocess_config):
    """
    まったく新しいデータを, trainデータとして使いたい場合.
    sampling rateを変更して, train, valでtextファイル作るだけ.
    """
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    os.makedirs(args.output_path, exist_ok=True)

    # sampling rateを変更する.
    # そのために, 一時フォルダを用意.
    print("\nchanging sampling rate")
    load_and_save(args.input_path, args.output_path, sr)

    wav_paths = glob(os.path.join(args.output_path, '*.wav'))

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
        default=2000
    )

    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding='utf-8'), Loader=yaml.FullLoader
    )

    main(args, preprocess_config)
