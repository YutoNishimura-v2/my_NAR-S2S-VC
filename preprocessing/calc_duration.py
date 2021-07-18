# durationはここで計算して完結する.
from typing import List
import os
import os.path as opth
from glob import glob

import numpy as np
from scipy.spatial.distance import cityblock
from fastdtw import fastdtw
import librosa
from tqdm import tqdm

import audio as Audio


def calc_duration(ts_src: List[np.ndarray], target_path: str) -> np.ndarray:
    """
    Args:
      ts_src: アライメントさせたい対象.
        その中身は, (d, time)の時系列のリスト.
        最初のがtarget, 次にsorceが入っている.

    source : target = 多 : 1 のケース
        - 該当するsourceの最初を1, それ以外は0として削除してしまう.
        - 理由; -1, -2 などとして, meanをとることは可能だが, -1, -2のように連続値が出力される必要があり. その制約を課すのは難しいと感じた.
        - 理由: また, 削除は本質的な情報欠落には当たらないと思われる. targetにはない情報なのでつまり不要なので.

    source: target = 1 : 多のケース
        - これは従来通り, sourceに多を割り当てることで対応.
    """
    t_src, s_src = ts_src
    duration = np.ones(s_src.shape[1])

    # alignment開始.
    _, path = fastdtw(t_src.T, s_src.T, dist=cityblock)

    # xのpathを取り出して, 長さで正規化する.
    patht = np.array(list(map(lambda l: l[0], path)))
    paths = np.array(list(map(lambda l: l[1], path)))

    b_p_t, b_p_s = 0, 0  # 初期化.
    count = 0
    for p_t, p_s in zip(patht[1:], paths[1:]):
        if b_p_t == p_t:
            # もし, targetの方が連続しているなら, s:t=多:1なので削る.
            duration[p_s] = 0  # 消したいのは, p_tに対応しているp_sであることに注意.
        if b_p_s == p_s:
            # sourceが連続しているなら, s:t=1:多なので増やす方.
            count += 1
        elif count > 0:
            # count > 0で, 一致をしなくなったなら, それは連続が終了したとき.
            duration[b_p_s] += count
            count = 0

        b_p_t = p_t
        b_p_s = p_s

    duration[b_p_s] += count if count > 0 else 0

    assert np.sum(duration) == t_src.shape[1], f"""{target_path}にてdurationの不一致がおきました\n
    duration: {duration}\n
    np.sum(duration): {np.sum(duration)}\n
    len(t_src): {len(t_src)}"""

    return duration


def get_duration(p_config, m_config):
    print("calc duration...")

    source_in_dir = p_config["path"]["source_prevoice_path"]
    target_in_dir = p_config["path"]["target_prevoice_path"]
    out_dir = p_config["path"]["preprocessed_path"]

    STFT = Audio.stft.TacotronSTFT(
        p_config["preprocessing"]["stft"]["filter_length"],
        p_config["preprocessing"]["stft"]["hop_length"],
        p_config["preprocessing"]["stft"]["win_length"],
        20,  # config["preprocessing"]["mel"]["n_mel_channels"]のところ.
        p_config["preprocessing"]["audio"]["sampling_rate"],
        p_config["preprocessing"]["mel"]["mel_fmin"],
        p_config["preprocessing"]["mel"]["mel_fmax"],
    )

    reduction_factor = m_config["reduction_factor"]

    os.makedirs((os.path.join(out_dir, "source", "duration")), exist_ok=True)

    # sortして, 対応関係が保たれるという仮定を立てている.
    source_wav_paths = np.sort(glob(opth.join(source_in_dir, "*.wav")))
    target_wav_paths = np.sort(glob(opth.join(target_in_dir, "*.wav")))

    for source_path, target_path in tqdm(zip(source_wav_paths, target_wav_paths)):
        assert opth.basename(source_path) == opth.basename(target_path), "対応関係が壊れています."

        source_wav, _ = librosa.load(
            source_path, sr=p_config["preprocessing"]["audio"]["sampling_rate"])
        target_wav, _ = librosa.load(
            target_path, sr=p_config["preprocessing"]["audio"]["sampling_rate"])
        source_mel, _ = Audio.tools.get_mel_from_wav(source_wav, STFT)
        target_mel, _ = Audio.tools.get_mel_from_wav(target_wav, STFT)

        source_mel = reduction(source_mel, reduction_factor, m_config["reduction_mean"])
        target_mel = reduction(target_mel, reduction_factor, m_config["reduction_mean"])

        duration = calc_duration([target_mel, source_mel], target_path)
        duration_filename = f"duration-{opth.basename(source_path).replace('.wav', '')}.npy"
        np.save(os.path.join(out_dir, "source", "duration", duration_filename), duration)


def reduction(x: np.ndarray, reduction_factor: int, mean_: bool = False) -> np.ndarray:
    """1D or 2Dに対応.

    2Dの場合: (*, time) を想定.
    """
    n_dim = len(x.shape)

    if n_dim > 2:
        raise ValueError("次元が2以上のarrayは想定されていません.")

    if mean_ is True:
        if n_dim == 1:
            x = np.pad(x, (0, (reduction_factor-x.shape[0] % reduction_factor) % reduction_factor), mode='edge')
            x = x.reshape(x.shape[0]//reduction_factor, reduction_factor)

        else:
            x = np.pad(x, [(0, 0), (0, (reduction_factor-x.shape[1] % reduction_factor) % reduction_factor)],
                       mode='edge')
            x = x.reshape(x.shape[0], x.shape[1]//reduction_factor, reduction_factor)

        x = x.mean(-1)

    else:
        slice = np.arange(0, x.shape[n_dim-1], reduction_factor)
        if n_dim == 1:
            x = x[slice]
        else:
            x = x[:, slice]

    return x


def mel_reshape(x: np.ndarray, reduction_factor: int):
    """
    (time, mel_num)→(time/reduction_factor, mel_num*reduction_factor)
    """
    if len(x.shape) != 2:
        raise ValueError("未対応です")

    x = np.pad(x, [(0, (reduction_factor-x.shape[0] % reduction_factor) % reduction_factor), (0, 0)],
               mode='constant')

    x = x.reshape(x.shape[0]//reduction_factor, x.shape[1]*reduction_factor)

    return x
