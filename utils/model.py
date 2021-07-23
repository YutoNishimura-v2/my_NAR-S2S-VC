import os
import json

import torch

import hifigan
from model.nars2svc import NARS2SVC
from model.optimizer import ScheduledOptim


def my_load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                      f"required shape: {model_state_dict[k].shape}, "
                      f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
        else:
            state_dict.pop(k)
            print(f"Dropping parameter {k}")
    model.load_state_dict(state_dict)


def get_model(args, configs, device, train=False):
    """
    train.pyなどで使用.

    args.restore_step: 途中で止めたなら, そこから再開できるように, clpt番号を指定可能.
                       exp_nameに対応していないのは残念かも.
    trainなら, optimもスケジュールで. そして, こちらもあるならoptimを保存していたものを利用.
    """
    (preprocess_config, model_config, train_config) = configs

    model = NARS2SVC(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        my_load_state_dict(model, ckpt["model"])

    if train is True:
        if train_config["optimizer"]["reset_optim"] is True:
            scheduled_optim = ScheduledOptim(
                model, train_config, model_config, 0
            )
            model.train()
            return model, scheduled_optim
        else:
            scheduled_optim = ScheduledOptim(
                model, train_config, model_config, args.restore_step
            )
            if args.restore_step:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
            model.train()
            return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(device, train_config):
    """
    vocoderを用意.

    config: model_configのこと.
    speaker: support  'LJSpeech', 'universal'

    MelGANとHiFi-GANに対応しているのね.
    """
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)

    weight_path = "hifigan/generator_universal.pth.tar"
    if train_config["path"]["vocoder_weight"] is not None:
        weight_path = train_config["path"]["vocoder_weight"]
    ckpt = torch.load(weight_path, map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, preprocess_config, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
