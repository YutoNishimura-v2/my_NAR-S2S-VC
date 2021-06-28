import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import SourceDataset
from preprocessor.inference_preprocessor import inference_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize(model, configs, vocoder, batchs, control_values, output_path):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

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
        "--output",
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
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # preprocess
    inference_preprocess(args.input_path, args.output_path, preprocess_config)

    # Get dataset
    dataset = SourceDataset("inference.txt", args.output_path)
    batchs = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
    )

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, configs, vocoder, batchs, control_values, args.output_path)
