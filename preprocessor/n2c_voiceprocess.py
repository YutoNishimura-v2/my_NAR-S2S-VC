"""
ここでは, duration_prepareで行ったように, 音声に対する前処理を行う.
やること

1. 音声のサンプリング数を24000へ変更.
2. 無音区間の削除, 結合
"""
from glob import glob
import os.path as opth
import os

import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence


def load_and_save(input_path, output_path, sr):
    """
    sr以外, つまり, monoと16bitなのは固定.
    """
    for wav_path in glob(opth.join(input_path, "*.wav")):
        y, sr = librosa.core.load(wav_path, sr=sr, mono=True)  # 22050Hz、モノラルで読み込み
        sf.write(opth.join(output_path, opth.basename(wav_path)), y, sr, subtype="PCM_16")


def change_sr(config):
    sr = config["preprocessing"]["audio"]["sampling_rate"]
    source_raw_path = config["path"]["source_raw_path"]
    source_prevoice_path = config["path"]["source_prevoice_path"]
    target_raw_path = config["path"]["target_raw_path"]
    target_prevoice_path = config["path"]["target_prevoice_path"]

    load_and_save(source_raw_path, source_prevoice_path, sr)
    load_and_save(target_raw_path, target_prevoice_path, sr)


def delete_novoice_from_path(input_path, output_path, preprocess_config):
    for input_path in glob(opth.join(input_path, "*.wav")):
        audio = AudioSegment.from_wav(input_path)
        chunks = split_on_silence(audio, min_silence_len=preprocess_config["preprocessing"]["audio"]["min_silence_len"],
                                  silence_thresh=preprocess_config["preprocessing"]["audio"]["silence_thresh"],
                                  keep_silence=preprocess_config["preprocessing"]["audio"]["keep_silence"])
        audio_cut = AudioSegment.empty()
        for chunk in chunks:
            audio_cut += chunk
        audio_cut.export(opth.join(output_path, opth.basename(input_path)), format="wav")


def delete_novoice(config):
    # 無音区間の削除
    source_prevoice_path = config["path"]["source_prevoice_path"]
    target_prevoice_path = config["path"]["target_prevoice_path"]

    delete_novoice_from_path(source_prevoice_path, source_prevoice_path, config)
    delete_novoice_from_path(target_prevoice_path, target_prevoice_path, config)


def voice_preprocess(config):
    os.makedirs(config["path"]["source_prevoice_path"], exist_ok=True)
    os.makedirs(config["path"]["target_prevoice_path"], exist_ok=True)
    # まずはsrを変更して, prevoice_pathに保存.
    print("changing sampling_rate....")
    change_sr(config)
    # そして無音区間を削除
    print("delete no voice....")
    delete_novoice(config)
