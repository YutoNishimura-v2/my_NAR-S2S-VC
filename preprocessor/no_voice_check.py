# 簡単なチェック用コード

import os
import os.path as opth
import shutil
import sys
from glob import glob

import numpy as np
import yaml

sys.path.append('.')

from utils.utils import get_mels, plot_mels
from preprocessor.n2c_voiceprocess import delete_novoice_from_path


# ここだけ指定してください
########################
config_base_path = "./config/JSUT_JSSS"
shuffle = False
########################

preprocess_config = yaml.load(open(os.path.join(config_base_path, "preprocess.yaml"),
                              "r", encoding='utf-8'), Loader=yaml.FullLoader)

# 使うデータ
source_input_paths = np.array(glob(os.path.join(preprocess_config["path"]["source_raw_path"], '*.wav')))
target_input_paths = np.array(glob(os.path.join(preprocess_config["path"]["target_raw_path"], '*.wav')))

assert len(source_input_paths) == len(target_input_paths)

if shuffle is True:
    indexes = np.random.permutation(len(source_input_paths))[:5]
    source_input_paths = source_input_paths[indexes]
    target_input_paths = target_input_paths[indexes]
else:
    source_input_paths = source_input_paths[:5]
    target_input_paths = target_input_paths[:5]

tmp_dir = "./tmp_for_novoicecheck"
os.makedirs(os.path.join(tmp_dir, "source"), exist_ok=True)
os.makedirs(os.path.join(tmp_dir, "target"), exist_ok=True)

s_wav_paths = []
t_wav_paths = []

for source_path, target_path in zip(source_input_paths, target_input_paths):
    output_path = opth.join(tmp_dir, "source")
    s_wav_paths.append(os.path.join(output_path, os.path.basename(source_path)))
    delete_novoice_from_path(source_path, output_path, preprocess_config)

    output_path = opth.join(tmp_dir, "target")
    t_wav_paths.append(os.path.join(output_path, os.path.basename(target_path)))
    delete_novoice_from_path(target_path, output_path, preprocess_config)


s_mels = get_mels(source_input_paths.tolist(), 80, preprocess_config)
s_mels_cut = get_mels(s_wav_paths, 80, preprocess_config)
t_mels = get_mels(target_input_paths.tolist(), 80, preprocess_config)
t_mels_cut = get_mels(t_wav_paths, 80, preprocess_config)

for i in range(len(s_mels)):
    plot_mels([s_mels[i], s_mels_cut[i]], [source_input_paths[i], s_wav_paths[i]],
              preprocess_config["preprocessing"]["audio"]["sampling_rate"])

for i in range(len(t_mels)):
    plot_mels([t_mels[i], t_mels_cut[i]], [target_input_paths[i], t_wav_paths[i]],
              preprocess_config["preprocessing"]["audio"]["sampling_rate"])

shutil.rmtree(tmp_dir)
