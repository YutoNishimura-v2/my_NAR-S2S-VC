"""
ここでは, duration_prepareで行ったように, 音声に対する前処理を行う.
やること

1. 音声のサンプリング数を24000へ変更.
2. 無音区間の削除, 結合
"""
import os
import os.path as opth
from glob import glob

import librosa
import soundfile as sf
from pydub import AudioSegment, silence
from tqdm import tqdm


def load_and_save(input_path, output_path, sr):
    """
    sr以外, つまり, monoと16bitなのは固定.
    wav以外もlibrosaは読み込めるらしいが, 今のところは需要ないのでwav限定の運用で.
    """
    for wav_path in tqdm(glob(opth.join(input_path, "*.wav"))):
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


def make_novoice_to_zero(audio: AudioSegment, silence_thresh: float) -> AudioSegment:
    """無音判定をくらった部分を, 0にしてしまう.
    """
    silences = silence.detect_silence(audio, min_silence_len=50, silence_thresh=silence_thresh)

    audio_new = AudioSegment.empty()
    s_index = 0
    for silence_ in silences:
        audio_new += audio[s_index:silence_[0]]
        audio_new += AudioSegment.silent(duration=silence_[1]-silence_[0])
        s_index = silence_[1]

    audio_new += audio[s_index:]

    return audio_new


def delete_novoice_from_path(input_path, output_path, preprocess_config, chunk_size=50):
    """無音区間を先頭と末尾から削除します.
    Args:
      input_path: wavファイルへのpath
      output_path: wavファイルを貯めたい場所. ファイル名はinput_pathのbasenameからとる.
      chunk_size: 削除に用いる音声の最小単位. 基本defaultのままで良さそう.
    """
    silence_thresh = preprocess_config["preprocessing"]["audio"]["silence_thresh"]
    silence_thresh_h = preprocess_config["preprocessing"]["audio"]["silence_thresh_head"]

    if (silence_thresh_h is not None) and (silence_thresh_h < silence_thresh):
        print("Warning: 基本的に, silene_thresh_hの方が, silence_threshよりも大きいべきです.")

    audio = AudioSegment.from_wav(input_path)

    # 参考: https://stackoverflow.com/questions/29547218/
    # remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
        trim_ms = 0  # ms

        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    if silence_thresh_h is None:
        silence_thresh_h = silence_thresh

    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh_h, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio.reverse(),
                                      silence_threshold=silence_thresh, chunk_size=chunk_size)

    duration = len(audio)
    audio_cut = audio[start_trim:duration-end_trim]

    audio_cut = make_novoice_to_zero(audio_cut, silence_thresh)

    audio_cut.export(opth.join(output_path, opth.basename(input_path)), format="wav")


def delete_novoice(config):
    # 無音区間の削除
    source_prevoice_path = config["path"]["source_prevoice_path"]
    target_prevoice_path = config["path"]["target_prevoice_path"]

    for s_wav_path in tqdm(glob(os.path.join(source_prevoice_path, '*.wav'))):
        delete_novoice_from_path(s_wav_path, source_prevoice_path, config)
    for t_wav_path in tqdm(glob(os.path.join(target_prevoice_path, '*.wav'))):
        delete_novoice_from_path(t_wav_path, target_prevoice_path, config)


def voice_preprocess(config):
    os.makedirs(config["path"]["source_prevoice_path"], exist_ok=True)
    os.makedirs(config["path"]["target_prevoice_path"], exist_ok=True)
    # まずはsrを変更して, prevoice_pathに保存.
    print("changing sampling_rate....")
    change_sr(config)

    # そして無音区間を削除
    print("delete no voice....")
    delete_novoice(config)
