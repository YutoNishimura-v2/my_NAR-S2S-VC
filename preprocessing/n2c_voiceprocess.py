"""
ここでは, duration_prepareで行ったように, 音声に対する前処理を行う.
やること

1. 音声のサンプリング数を24000へ変更.
2. 無音区間の削除, 結合
"""
from glob import glob
import os.path as opth
import os

from tqdm import tqdm
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence


def load_and_save(input_path, output_path, sr):
    """
    sr以外, つまり, monoと16bitなのは固定.
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


def delete_novoice_from_path(input_path, output_path, preprocess_config):
    """
    Args:
      input_path: wavファイルへのpath
      output_path: wavファイルを貯めたい場所. ファイル名はinput_pathのbasenameからとる.
    """
    min_silence_len = preprocess_config["preprocessing"]["audio"]["min_silence_len"]
    silence_thresh = preprocess_config["preprocessing"]["audio"]["silence_thresh"]
    keep_silence = preprocess_config["preprocessing"]["audio"]["keep_silence"]
    head_tail_only = preprocess_config["preprocessing"]["audio"]["head_tail_only"]

    audio = AudioSegment.from_wav(input_path)

    if head_tail_only is True:
        # 参考: https://stackoverflow.com/questions/29547218/
        # remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
        def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
            trim_ms = 0  # ms

            assert chunk_size > 0  # to avoid infinite loop
            while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
                trim_ms += chunk_size

            return trim_ms
        start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh, chunk_size=min_silence_len)
        end_trim = detect_leading_silence(audio.reverse(),
                                          silence_threshold=silence_thresh, chunk_size=min_silence_len)

        duration = len(audio)
        audio_cut = audio[start_trim:duration-end_trim]

    else:
        chunks = split_on_silence(audio, min_silence_len=min_silence_len,
                                  silence_thresh=silence_thresh,
                                  keep_silence=keep_silence)
        audio_cut = AudioSegment.empty()
        for chunk in chunks:
            audio_cut += chunk

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
