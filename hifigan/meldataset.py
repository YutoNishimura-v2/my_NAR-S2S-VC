import math
import os
import random

import librosa
import numpy as np
import torch
from torch._C import Value
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

from audio.stft import TacotronSTFT

MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sr):
    # sampling_rate, data = read(full_path)
    wav, _ = librosa.load(
        full_path, sr=sr)
    return wav, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
                    center=False, max_audio_len=None):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    if max_audio_len is not None:
        y = y[:, :max_audio_len]
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def mel_spectrogram_nars2s(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, max_audio_len=None):
    # y: audio
    # valは9000で切り取らないので, melにするとズレうる. なのでここで調整.
    def get_mel_from_wav(audio, _stft, max_audio_len):
        if max_audio_len is not None:
            audio = audio[:, :max_audio_len]  # batchで来る想定.
        audio = torch.clip(audio, -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec, _ = _stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    STFT = TacotronSTFT(
        n_fft,
        hop_size,
        win_size,
        num_mels,
        sampling_rate,
        fmin,
        fmax
    )

    mel = get_mel_from_wav(y, STFT, max_audio_len).to(y.device)

    return mel.unsqueeze(0) if mel.dim() == 2 else mel


def get_dataset_filelist(a):
    with open(os.path.join(a.input_mel_path, "train.txt"), 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wav_path, x + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(os.path.join(a.input_mel_path, "val.txt"), 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wav_path, x + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files  # wav_path
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:  # もはやここでしかfinetuning使わない.
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        mel = np.load(
            os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
        mel = torch.from_numpy(mel)

        if len(mel.shape) < 3:
            mel = mel.unsqueeze(0)

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)
            if audio.size(1) > self.segment_size:
                if mel.size(2) - frames_per_seg - 1 == 0:
                    # 例えば, audio_len = 9120なら, melはhop_size=300で30となり, 0になってしまう.
                    mel_start = 0
                else:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
            else:
                mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # loss用のmel??
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
