"""
Examples:
>>> python preprocess.py config/JSUT_JSSS/preprocess.yaml
"""

import argparse
import shutil
import os

import yaml

from preprocessing.preprocessor import Preprocessor, wav_path_matching
from preprocessing.n2c_voiceprocess import voice_preprocess
from preprocessing.calc_duration import get_duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess_config", type=str, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, help="path to preprocess.yaml")
    parser.add_argument("--finetuning", action='store_true')
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    # 入力ファイルの対応チェック
    wav_path_matching(preprocess_config["path"]["source_raw_path"], preprocess_config["path"]["target_raw_path"])

    # 音声に対する前処理
    # ちょっとでもpre_voiceを間違って作ってしまうと, no_voice処理のせいで, durationとずれる.
    assert os.path.exists(preprocess_config["path"]["source_prevoice_path"]) is False, \
        "すでに存在しているpre_voiceです. 再度実行すると事故が起こるので, 注意してください."
    voice_preprocess(preprocess_config)

    # melの用意とか
    assert os.path.exists(preprocess_config["path"]["preprocessed_path"]) is False, \
        "すでに存在しているpreprocessed_dataです. 再度実行すると事故が起こるので, 注意してください."
    preprocessor = Preprocessor(preprocess_config, finetuning=args.finetuning)
    preprocessor.build_from_path()

    # durationの用意
    get_duration(preprocess_config, model_config)

    # ちゃんとconfigを残しておく.
    pre_voice_path = os.path.dirname(preprocess_config["path"]["source_prevoice_path"])
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    shutil.copy(args.preprocess_config, pre_voice_path)
    shutil.copy(args.preprocess_config, preprocessed_path)
    shutil.copy(args.model_config, pre_voice_path)
    shutil.copy(args.model_config, preprocessed_path)
