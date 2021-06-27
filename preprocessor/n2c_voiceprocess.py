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


def change_sr(config):
    sr = config["preprocessing"]["audio"]["sampling_rate"]
    source_raw_path = config["path"]["source_raw_path"]
    source_prevoice_path = config["path"]["source_prevoice_path"]
    target_raw_path = config["path"]["target_raw_path"]
    target_prevoice_path = config["path"]["target_prevoice_path"]

    for sourse_wav_path in glob(opth.join(source_raw_path, "*.wav")):
        y, sr = librosa.core.load(sourse_wav_path, sr=sr, mono=True)  # 22050Hz、モノラルで読み込み
        sf.write(opth.join(source_prevoice_path, opth.basename(sourse_wav_path)), y, sr, subtype="PCM_16")

    for target_wav_path in glob(opth.join(target_raw_path, "*.wav")):
        y, sr = librosa.core.load(target_wav_path, sr=sr, mono=True)  # 22050Hz、モノラルで読み込み
        sf.write(opth.join(target_prevoice_path, opth.basename(target_wav_path)), y, sr, subtype="PCM_16")


def delete_novoice(config):
    # 無音区間の削除
    source_prevoice_path = config["path"]["source_prevoice_path"]
    target_prevoice_path = config["path"]["target_prevoice_path"]

    for sourse_wav_path in glob(opth.join(source_prevoice_path, "*.wav")):
        audio = AudioSegment.from_wav(sourse_wav_path)
        chunks = split_on_silence(audio, min_silence_len=50, silence_thresh=-100, keep_silence=10)
        audio_cut = AudioSegment.empty()
        for chunk in chunks:
            audio_cut += chunk
        audio_cut.export(sourse_wav_path, format="wav")

    for target_wav_path in glob(opth.join(target_prevoice_path, "*.wav")):
        audio = AudioSegment.from_wav(target_wav_path)
        chunks = split_on_silence(audio, min_silence_len=50, silence_thresh=-100, keep_silence=10)
        audio_cut = AudioSegment.empty()
        for chunk in chunks:
            audio_cut += chunk
        audio_cut.export(target_wav_path, format="wav")


def voice_preprocess(config):
    os.makedirs(config["path"]["source_prevoice_path"], exist_ok=True)
    os.makedirs(config["path"]["target_prevoice_path"], exist_ok=True)
    # まずはsrを変更して, prevoice_pathに保存.
    print("changing sampling_rate....")
    change_sr(config)
    # そして無音区間を削除
    print("delete no voice....")
    delete_novoice(config)
