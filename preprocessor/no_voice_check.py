# 簡単なチェック用コード

import os
import os.path as opth
import shutil
import sys
from glob import glob

import numpy as np
import yaml
from pydub import AudioSegment
from pydub.silence import split_on_silence

sys.path.append('.')

from utils.utils import get_mels, plot_mels


# ここだけ指定してください
########################
config_base_path = "./config/JSUT_JSSS"
########################

preprocess_config = yaml.load(open(os.path.join(config_base_path, "preprocess.yaml"),
                              "r", encoding='utf-8'), Loader=yaml.FullLoader)

# 使うデータ
source_input_paths = glob(os.path.join(preprocess_config["path"]["source_raw_path"], '*.wav'))
target_input_paths = glob(os.path.join(preprocess_config["path"]["target_raw_path"], '*.wav'))

assert len(source_input_paths) == len(target_input_paths)

indexes = np.random.permutation(len(source_input_paths))[:5]

source_input_paths = np.array(source_input_paths)[indexes]
target_input_paths = np.array(target_input_paths)[indexes]

tmp_dir = "./tmp_for_novoicecheck"
os.makedirs(tmp_dir, exist_ok=True)

s_wav_paths = []
t_wav_paths = []

for i, (source_path, target_path) in enumerate(zip(source_input_paths, target_input_paths)):
    s_audio = AudioSegment.from_wav(source_path)
    t_audio = AudioSegment.from_wav(target_path)

    for j, audio_ in enumerate([s_audio, t_audio]):
        chunks = split_on_silence(audio_,
                                  min_silence_len=preprocess_config["preprocessing"]["audio"]["min_silence_len"],
                                  silence_thresh=preprocess_config["preprocessing"]["audio"]["silence_thresh"],
                                  keep_silence=preprocess_config["preprocessing"]["audio"]["keep_silence"])
        audio_cut = AudioSegment.empty()
        for chunk in chunks:
            audio_cut += chunk

        if j == 0:
            # source
            output_path = opth.join(tmp_dir, "source", opth.basename(source_path))
            s_wav_paths.append(output_path)
        else:
            output_path = opth.join(tmp_dir, "source", opth.basename(target_path))
            t_wav_paths.append(output_path)

        audio_cut.export(output_path, format="wav")

s_mels = get_mels(s_wav_paths, 80, preprocess_config)
t_mels = get_mels(s_wav_paths, 80, preprocess_config)

for i in range(len(s_mels)):
    plot_mels([s_mels[i], t_mels[i]], [s_wav_paths[i], t_wav_paths[i]],
              preprocess_config["preprocessing"]["audio"]["sampling_rate"])

shutil.rmtree(tmp_dir)
