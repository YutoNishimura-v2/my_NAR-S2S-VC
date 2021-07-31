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


def calc_duration(ts_src: List[np.ndarray], target_path: str, diagonal_index: np.ndarray = None) -> np.ndarray:
    """
    Args:
      ts_src: アライメントさせたい対象.
        その中身は, (d, time)の時系列のリスト.
        最初のがtarget, 次にsorceが入っている.
      diagonal_index: 対角化させたいindex. False, Trueの値が入っている.
        対角化とは, sourceとtargetのtime indexをx, y軸とでもしたときに, 斜めになるようにすること.

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

    if diagonal_index is not None:
        assert s_src.shape[1] == len(
            diagonal_index), f"s_src.shape: {s_src.shape}, len(diagonal_index): {len(diagonal_index)}"
        index_list = get_s_e_index_from_bools(diagonal_index)
        for index_s_t in index_list:
            duration_part = duration[index_s_t[0]:index_s_t[1]]
            if np.sum(duration_part) > len(duration_part):
                # targetの無音区間の方が長いケース.
                delta = int(np.sum(duration_part)-len(duration_part))
                duration_part[:delta] = 2
                duration_part[delta:] = 1
            else:
                # sourceの方が長いケース.
                s_index = int(np.sum(duration_part))
                duration_part[:s_index] = 1
                duration_part[s_index:] = 0

            duration[index_s_t[0]:index_s_t[1]] = duration_part

    assert np.sum(duration) == t_src.shape[1], f"""{target_path}にてdurationの不一致がおきました\n
    duration: {duration}\n
    np.sum(duration): {np.sum(duration)}\n
    t_src.shape: {t_src.shape}"""

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
        source_mel, energy = Audio.tools.get_mel_from_wav(source_wav, STFT)
        target_mel, _ = Audio.tools.get_mel_from_wav(target_wav, STFT)

        source_mel = reduction(source_mel, reduction_factor)
        target_mel = reduction(target_mel, reduction_factor)

        duration = calc_duration([target_mel, source_mel], target_path, np.log(energy+1e-6) < -5.0)
        duration_filename = f"duration-{opth.basename(source_path).replace('.wav', '')}.npy"
        np.save(os.path.join(out_dir, "source", "duration", duration_filename), duration)


def reduction(x: np.ndarray, reduction_factor: int) -> np.ndarray:
    """1D or 2Dに対応.
    2Dの場合: (*, time) を想定.
    """
    n_dim = len(x.shape)

    if n_dim > 2:
        raise ValueError("次元が2以上のarrayは想定されていません.")

    if n_dim == 1:
        x = x[:(x.shape[0]//reduction_factor)*reduction_factor]
        x = x.reshape(x.shape[0]//reduction_factor, reduction_factor)

    else:
        x = x[:, :(x.shape[1]//reduction_factor)*reduction_factor]
        x = x.reshape(x.shape[0], x.shape[1]//reduction_factor, reduction_factor)

    x = x.mean(-1)

    return x


def get_s_e_index_from_bools(array):
    """True, Falseのbool配列から, Trueがあるindexの最初と最後を見つけてindexの
    配列として返す.
    """
    index_list = []
    flg = 0
    s_ind = 0
    for i, v in enumerate(array):
        if (flg == 0) and v:
            s_ind = i
            flg = 1
        elif (flg == 1) and (not v):
            index_list.append([s_ind, i])
            flg = 0

    if v:
        # 最後がTrueで終わっていた場合.
        index_list.append([s_ind, i+1])

    return index_list
