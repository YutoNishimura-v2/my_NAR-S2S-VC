"""
Examples:
>>> python preprocess.py config/JSUT_JSSS/preprocess.yaml

"""

import argparse

import yaml

from preprocessing.preprocessor import Preprocessor
from preprocessing.n2c_voiceprocess import voice_preprocess
from preprocessing.calc_duration import get_duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    # 音声に対する前処理
    voice_preprocess(config)

    # melの用意とか
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()

    # durationの用意
    get_duration(config)
