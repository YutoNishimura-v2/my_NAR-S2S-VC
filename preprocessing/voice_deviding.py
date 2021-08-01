import os
import os.path as opth
from glob import glob

import numpy as np
from pydub import AudioSegment, silence
from tqdm import tqdm

from preprocessing.calc_duration import get_duration_for_deviding


def add_tail_before_silence(audio, s_time, silence_thresh, chunk_size=10):
    """音声に対して, threshを下回るまでの時間を返す.
    """
    trim_ms = s_time  # ms
    while audio[trim_ms:trim_ms+chunk_size].dBFS < silence_thresh and trim_ms < len(audio):
        trim_ms += chunk_size

    return trim_ms


def devide_voice_from_path(s_wav_path, t_wav_path, preprocess_config):
    silence_thresh = preprocess_config["preprocessing"]["audio"]["silence_thresh"]
    min_silence_len = preprocess_config["preprocessing"]["audio"]["min_silence_len"]

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    s_audio = AudioSegment.from_wav(s_wav_path)
    t_audio = AudioSegment.from_wav(t_wav_path)
    s_duration = get_duration_for_deviding(s_wav_path, t_wav_path, preprocess_config)

    t_silences = silence.detect_silence(t_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    t_times = s_duration * hop_length * 1000 / sr
    s_times = np.ones_like(t_times)
    s_times = s_times * hop_length * 1000 / sr
    t_times = t_times.cumsum()
    s_times = s_times.cumsum()

    time_s = 0  # target用time
    time_s_ = 0  # source用time
    for i, (silence_s, silence_e) in enumerate(t_silences):
        silence_s_index = np.argmax(t_times > silence_s)
        silence_e_index = np.argmax(t_times > silence_e)
        silence_s_ = s_times[silence_s_index] + (t_times[silence_s_index] - silence_s)
        silence_s_ = round(silence_s_)
        # silence_s_ = add_tail_before_silence(s_audio, round(silence_s_), silence_thresh)
        silence_e_ = s_times[silence_e_index] + (t_times[silence_e_index] - silence_e)
        silence_e_ = round(silence_e_)

        t_audio_cut = t_audio[time_s:silence_s]
        time_s = silence_e
        s_audio_cut = s_audio[time_s_:silence_s_]
        time_s_ = silence_e_

        t_audio_cut.export(t_wav_path.replace(".wav", f"_{i}.wav"), format="wav")
        s_audio_cut.export(s_wav_path.replace(".wav", f"_{i}.wav"), format="wav")

        if i == len(t_silences) - 1:  # 最後のループ
            t_audio_cut = t_audio[time_s:]
            s_audio_cut = s_audio[time_s_:]
            t_audio_cut.export(t_wav_path.replace(".wav", f"_{i+1}.wav"), format="wav")
            s_audio_cut.export(s_wav_path.replace(".wav", f"_{i+1}.wav"), format="wav")

    os.remove(s_wav_path)
    os.remove(t_wav_path)


def voice_deviding(preprocess_config):
    """一定の長さ以上でvoiceを切りに行く. sourceとtargetで同じ内容で切れるように,
    durationを考慮しながら切断する.
    """
    source_prevoice_path = preprocess_config["path"]["source_prevoice_path"]
    target_prevoice_path = preprocess_config["path"]["target_prevoice_path"]

    s_wav_paths = glob(os.path.join(source_prevoice_path, '*.wav'))

    print("deviding voices....")
    for s_wav_path in tqdm(s_wav_paths):
        basename = opth.basename(s_wav_path)
        t_wav_path = opth.join(target_prevoice_path, basename)

        devide_voice_from_path(s_wav_path, t_wav_path, preprocess_config)
