# duration prepareで作った, 簡易版.

from typing import List, Optional, Union

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import audio as Audio


def plot_mels(mels: List[np.ndarray], wav_paths: List[str], sr: int, sharex: bool = True) -> None:
    fig, ax = plt.subplots(len(mels), 1, sharex=sharex, figsize=(15, 10))

    if len(mels) == 1:
        ax = [ax]

    for i, mel_spectrogram in enumerate(mels):
        img = librosa.display.specshow(mel_spectrogram, sr=sr,
                                       x_axis='time', y_axis='mel', ax=ax[i])
        ax[i].set(title=f"{wav_paths[i].split('/')[-1]}")
        fig.colorbar(img, format='%+2.0f dB', ax=ax[i])
    plt.tight_layout()
    plt.show()


def get_mels(wav_paths: Union[str, List[str]], mel_num: Optional[int], config: dict
             ) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Examples:
      wav_paths = ["./input/c_beyond_26.wav", "./input/n_beyond_26.wav"]
    """
    mels = []

    mel_num = mel_num if mel_num is not None else config["preprocessing"]["mel"]["n_mel_channels"]

    STFT = Audio.stft.TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        mel_num,
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    if type(wav_paths) == str:
        wav, _ = librosa.load(
            wav_paths, sr=config["preprocessing"]["audio"]["sampling_rate"])
        mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, STFT)
        return mel_spectrogram
    else:
        for wav_path in tqdm(wav_paths):
            wav, _ = librosa.load(
                wav_path, sr=config["preprocessing"]["audio"]["sampling_rate"])
            mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, STFT)
            mels.append(mel_spectrogram)

    return mels
