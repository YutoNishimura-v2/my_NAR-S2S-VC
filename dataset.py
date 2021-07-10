import os

import numpy as np
from torch.utils.data import Dataset

from utils.tools import pad_1D, pad_2D


class TrainDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]  # out_pathのこと.
        self.batch_size = train_config["optimizer"]["batch_size"]

        # processed_path直下にtrain.txtが作られているはず.
        # process_metaではsptitしているだけ.
        self.basenames = self.process_meta(
            filename
        )
        self.sort = sort
        self.drop_last = drop_last

        self.speakers = {}
        if os.path.exists(os.path.join(self.preprocessed_path, "speakers.txt")):
            with open(os.path.join(self.preprocessed_path, "speakers.txt"), "r", encoding="utf-8") as f:
                for i, line in enumerate(f.readlines()):
                    n = line.strip("\n")
                    self.speakers[n] = i

    def __len__(self):
        return len(self.basenames[0])

    def __getitem__(self, idx):
        # 基本的には, あらかじめ計算しておいた, pitch, energy, duration, melをtextとともに用いる.
        # textは, symbol化済みなので, それをidに変換する.
        basenames = []
        speakers = []
        mels = []
        pitchs = []
        energys = []
        for i, source_or_target in enumerate(["source", "target"]):
            # i==0がsource.
            basename = self.basenames[i][idx]

            mel_path = os.path.join(
                self.preprocessed_path,
                source_or_target,
                "mel",
                "mel-{}.npy".format(basename),
            )
            mel = np.load(mel_path)

            pitch_path = os.path.join(
                self.preprocessed_path,
                source_or_target,
                "pitch",
                "pitch-{}.npy".format(basename),
            )
            pitch = np.load(pitch_path)

            energy_path = os.path.join(
                self.preprocessed_path,
                source_or_target,
                "energy",
                "energy-{}.npy".format(basename),
            )
            energy = np.load(energy_path)

            if source_or_target == "source":
                duration_path = os.path.join(
                    self.preprocessed_path,
                    source_or_target,
                    "duration",
                    "duration-{}.npy".format(basename),
                )
                duration = np.load(duration_path)

            basenames.append(basename)
            if len(self.speakers) > 0:
                speakers.append(self.speakers[basename.split('_')[0]])
            else:
                # multi_speakerをoffにしていたら, ここに来る. この時は, source: 0, target: 1
                # として強制的にspeakerを持たせる.
                speakers.append(i)
            mels.append(mel)
            pitchs.append(pitch)
            energys.append(energy)

        sample = {
            "s_id": basenames[0],
            "t_id": basenames[1],
            "s_speaker_id": speakers[0],
            "t_speaker_id": speakers[1],
            "s_mel": mels[0],
            "s_pitch": pitchs[0],
            "s_energy": energys[0],
            "s_duration": duration,
            "t_mel": mels[1],
            "t_pitch": pitchs[1],
            "t_energy": energys[1],
        }

        return sample

    def process_meta(self, filename):
        names = []
        for source_or_target in ["source", "target"]:
            with open(
                os.path.join(self.preprocessed_path, source_or_target, filename), "r", encoding="utf-8"
            ) as f:
                name = []
                for line in f.readlines():
                    n = line.strip("\n")
                    name.append(n)
                names.append(name)
        return names

    def reprocess(self, data, idxs):
        """reporocess. ここではpaddingを行う.
        """
        # もしソートされていたら, テキストサイズが大きい順でbatch_size個
        # でまとめられたidxsが入ってくる.
        # まずはそいつらのデータを取得.
        s_ids = [data[idx]["s_id"] for idx in idxs]
        t_ids = [data[idx]["t_id"] for idx in idxs]
        s_sp_ids = [data[idx]["s_speaker_id"] for idx in idxs]
        t_sp_ids = [data[idx]["t_speaker_id"] for idx in idxs]
        s_mels = [data[idx]["s_mel"] for idx in idxs]
        s_pitches = [data[idx]["s_pitch"] for idx in idxs]
        s_energies = [data[idx]["s_energy"] for idx in idxs]
        s_durations = [data[idx]["s_duration"] for idx in idxs]
        t_mels = [data[idx]["t_mel"] for idx in idxs]
        t_pitches = [data[idx]["t_pitch"] for idx in idxs]
        t_energies = [data[idx]["t_energy"] for idx in idxs]

        # textとmelのlenを取得.
        s_mel_lens = np.array([s_mel.shape[0] for s_mel in s_mels])
        t_mel_lens = np.array([t_mel.shape[0] for t_mel in t_mels])

        # padding. tools.pyにあり.
        # 与えられたtext内からmax_sizeを探し出して, padしてくれる.
        s_mels = pad_2D(s_mels)
        s_pitches = pad_1D(s_pitches)
        s_energies = pad_1D(s_energies)
        s_durations = pad_1D(s_durations)
        t_mels = pad_2D(t_mels)
        t_pitches = pad_1D(t_pitches)
        t_energies = pad_1D(t_energies)

        # ついでにmaxの値も返す.
        return (
            s_ids,
            t_ids,
            s_sp_ids,
            t_sp_ids,
            s_mels,
            s_mel_lens,
            max(s_mel_lens),
            s_pitches,
            s_energies,
            s_durations,
            t_mels,
            t_mel_lens,
            max(t_mel_lens),
            t_pitches,
            t_energies,
        )

    def collate_fn(self, data):
        """reporocessに渡すために, batch*group分のbatchをdataloaderではあえてとって,
        その中の大きい順を獲得し, groupごとにreprocessする.

        予想: sortを行って, 大きい奴らでまとめてからpaddingすることで,
        めちゃくちゃ大きいtext基準に, 小さいtextがたくさんpaddingされるみたいな無駄をなくしていると予想される.
        """
        # collate_fn自体は, 自分でバッチを作りたいときにやるやつ.
        # dataでbatchを受け取ることに注意.
        data_size = len(data)
        # 注意. ここで, dataloaderに渡す値は, self.batch_size * groupsizeであることに注意.
        # train.pyでは, groupsizeは変数だが, ここのtestでは4に固定されていることに注意.
        # なので, ここで届くdata_sizeは, self.batch_size * 4

        if self.sort:
            len_arr = np.array([d["s_mel"].shape[0] for d in data])
            # ↑textの長さたち. そりゃ, 1つ1つ長さは異なる.
            idx_arr = np.argsort(-len_arr)
            # 長い順に取り出す. 昇順なので, マイナス.
            # そのidxであることに注意.
        else:
            idx_arr = np.arange(data_size)
            # ソートしないなら, batch_size分の配列を. そのまま読むって感じだね.

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        # ↑ self.batch_size = 8だとして, idx_arrのlenが22だとする.
        # この時, tailは, 22-6 = 16以降.
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # こっちは切りのいい部分.
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        # batch_sizeの倍数になっているので, batch_sizeでreshapeは可能.
        if not self.drop_last and len(tail) > 0:
            # drop_lastは, ここで切り捨てるか否かの判断ですね.
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            # reprocessのためにgroup化したようなもの. 何をやっているかはreprocessで実際に.
            output.append(self.reprocess(data, idx))

        return output


class SourceDataset(Dataset):
    def __init__(self, filename, filepath, train_config, sort=True, drop_last=False, duration_force=False, t_speaker=1):
        """
        Args:
          filepath: 処理したい音声の直上フォルダを指定.
        ここに, mel, energy, pitchのフォルダがある.
        また, basenameの名前も保管されている.
        """
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.preprocessed_path = filepath
        self.basename = self.process_meta(filename)
        self.sort = sort
        self.drop_last = drop_last
        self.duration_force = duration_force

        self.speakers = {}
        if os.path.exists(os.path.join(self.preprocessed_path, "speakers.txt")):
            with open(os.path.join(self.preprocessed_path, "speakers.txt"), "r", encoding="utf-8") as f:
                for i, line in enumerate(f.readlines()):
                    n = line.strip("\n")
                    self.speakers[n] = i

        self.target_speaker = t_speaker  # default: 1

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "mel-{}.npy".format(basename),
        )
        mel = np.load(mel_path)

        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "pitch-{}.npy".format(basename),
        )
        pitch = np.load(pitch_path)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "energy-{}.npy".format(basename),
        )
        energy = np.load(energy_path)

        if len(self.speakers) > 0:
            speaker = self.speakers[basename.split('_')[0]]
        else:
            # multi_speakerをoffにしていたら, ここに来る. この時は, source: 0, target: 1
            # として強制的にspeakerを持たせる.
            speaker = 0

        if self.duration_force is True:
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "duration-{}.npy".format(basename),
            )
            duration = np.load(duration_path)
            sample = {
                "id": basename,
                "s_speaker_id": speaker,
                "t_speaker_id": self.target_speaker,
                "s_mel": mel,
                "s_pitch": pitch,
                "s_energy": energy,
                "s_duration": duration
            }
        else:
            sample = {
                "id": basename,
                "s_speaker_id": speaker,
                "t_speaker_id": self.target_speaker,
                "s_mel": mel,
                "s_pitch": pitch,
                "s_energy": energy,
            }

        return sample

    def process_meta(self, filename):
        name = []
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                n = line.strip("\n")
                name.append(n)
        return name

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        s_sp_ids = [data[idx]["s_speaker_id"] for idx in idxs]
        t_sp_ids = [data[idx]["t_speaker_id"] for idx in idxs]
        s_mels = [data[idx]["s_mel"] for idx in idxs]
        s_pitches = [data[idx]["s_pitch"] for idx in idxs]
        s_energies = [data[idx]["s_energy"] for idx in idxs]
        if self.duration_force is True:
            s_durations = [data[idx]["s_duration"] for idx in idxs]
            t_mel_lens = np.array([np.sum(s_duration) for s_duration in s_durations], dtype=np.int32)
            s_durations = pad_1D(s_durations)

        # textとmelのlenを取得.
        s_mel_lens = np.array([s_mel.shape[0] for s_mel in s_mels])

        # padding. tools.pyにあり.
        # 与えられたtext内からmax_sizeを探し出して, padしてくれる.
        s_mels = pad_2D(s_mels)
        s_pitches = pad_1D(s_pitches)
        s_energies = pad_1D(s_energies)

        # ついでにmaxの値も返す.

        if self.duration_force is True:
            return (
                ids,
                s_sp_ids,
                t_sp_ids,
                s_mels,
                s_mel_lens,
                max(s_mel_lens),
                s_pitches,
                s_energies,
                s_durations,
                t_mel_lens,
                max(t_mel_lens)
            )
        else:
            return (
                ids,
                s_sp_ids,
                t_sp_ids,
                s_mels,
                s_mel_lens,
                max(s_mel_lens),
                s_pitches,
                s_energies,
            )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["s_mel"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


if __name__ == "__main__":
    # Test
    import sys
    import torch
    import yaml
    from torch.utils.data import DataLoader

    sys.path.append('.')
    from utils.tools import to_device

    # JSUTをちゃんと読み込みましょう!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/N2C/preprocess.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/N2C/train.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader
    )

    train_dataset = TrainDataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=False
    )
    val_dataset = TrainDataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0

    max_ = -1
    min_ = 22000

    for batchs in train_loader:
        for batch in batchs:
            print("source_mel_lens: ", batch[3])
            print("duration sum: ", np.sum(batch[7], axis=1))
            print("target_mel_lens: ", batch[9])
            to_device(batch, device)
            max_ = max(max_, np.max(batch[9]))
            min_ = min(min_, np.min(batch[9]))
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            max_ = max(max_, np.max(batch[9]))
            min_ = min(min_, np.min(batch[9]))
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
    print('max_mel_length: ', max_)
    print('min_mel_length: ', min_)
