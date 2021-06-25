"""
Examples:
>>> python preprocess.py config/N2C/preprocess.yaml
"""

import argparse

import yaml

from preprocessor.preprocessor import Preprocessor
from preprocessor.n2c_voiceprocess import voice_preprocess


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
