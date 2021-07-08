import argparse
import os
import shutil

import torch
import yaml
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import SourceDataset
from preprocessing.inference_preprocessor import inference_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize(model, configs, vocoder, loader, control_values, output_path):
    preprocess_config, model_config, _ = configs
    pitch_control, energy_control, duration_control = control_values

    for batchs in tqdm(loader):
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(
                    *(batch[1:]),
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control
                )
                synth_samples(
                    batch,
                    output,
                    vocoder,
                    model_config,
                    preprocess_config,
                    output_path
                )


def inference_mel(model, loader, control_values, output_path):
    pitch_control, energy_control, duration_control = control_values

    for batchs in tqdm(loader):
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(
                    *(batch[1:]),
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control
                )
                mel_predictions = output[1].transpose(1, 2)  # (batch, dim, time)へ.
                mel_lens = output[9].cpu().numpy()
                basenames = batch[0]
                for i, mel in enumerate(mel_predictions):
                    # mel = mel_denormalize(mel, preprocess_config)
                    mel = mel.cpu().numpy()
                    mel = mel[:, :mel_lens[i]]
                    # vocoderとしてのinputは, dim, timeが想定されているみたい.
                    np.save(os.path.join(output_path, "mels", basenames[i]), mel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_step",
        type=int,
        required=True,
        help="何step目の訓練済み重みを用いて推論したいかです. データセット名は, configで指定するのでここでは数字だけを入れます."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="音声の入ったフォルダへのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="吐き出したい場所へのパス",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--get_mel_for_hifigan",
        action='store_true'
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    if args.get_mel_for_hifigan is not True:
        # Load vocoder
        vocoder = get_vocoder(device)
        # preprocess
        inference_preprocess(args.input_path, args.output_path, preprocess_config)
        # Get dataset
        dataset = SourceDataset("inference.txt", args.output_path, train_config)
    else:
        # for hifi-gan
        # input_path: 例: ./preprocessed_data/JSUT_JSSS/JSSS

        # make inference.txt
        with open(os.path.join(args.input_path, "inference.txt"), 'w', encoding='utf-8') as f_i:
            with open(os.path.join(args.input_path, "train.txt"), 'r', encoding='utf-8') as f_t:
                for line in f_t.readlines():
                    n = line.strip("\n")
                    f_i.write(n+'\n')
            with open(os.path.join(args.input_path, "val.txt"), 'r', encoding='utf-8') as f_v:
                for line in f_v.readlines():
                    n = line.strip("\n")
                    f_i.write(n+'\n')
        # copy train, val texts
        shutil.copy(os.path.join(args.input_path, "train.txt"), args.output_path)
        shutil.copy(os.path.join(args.input_path, "val.txt"), args.output_path)

        dataset = SourceDataset("inference.txt", args.input_path, train_config, duration_force=True)

        os.makedirs(os.path.join(args.output_path, "mels"), exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
    )

    control_values = args.pitch_control, args.energy_control, args.duration_control

    if args.get_mel_for_hifigan is not True:
        synthesize(model, configs, vocoder, dataloader, control_values, args.output_path)
    else:
        inference_mel(model, dataloader, control_values, args.output_path)
