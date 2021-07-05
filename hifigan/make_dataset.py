# wavが大量に入ったフォルダに対して, mel化の処理を施す.
# もちろん, configはNARS2Sのものを利用する.
import os
from glob import glob

import yaml
import numpy as np

from preprocessing.n2c_voiceprocess import load_and_save
from utils.utils import get_mels


def main(args, preprocess_config):
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    tmp_dir = os.path.join(args.output_path, 'tmp_dir')
    os.makedirs(tmp_dir, exist_ok=True)

    # sampling rateを変更する.
    # そのために, 一時フォルダを用意.
    load_and_save(args.input_path, tmp_dir, sr)

    # melにする.
    wav_paths = glob(os.path.join(tmp_dir, '*.wav'))
    mels = get_mels(wav_paths, 80, preprocess_config)

    mels_dir = os.path.join(args.output_path, 'mels')
    os.makedirs(mels_dir, exist_ok=True)

    print("\nmel save...")
    for mel, wav_path in zip(mels, wav_paths):
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
        default=300
    )

    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding='utf-8'), Loader=yaml.FullLoader
    )

    main(args, preprocess_config)
