# 読み解いていく.
- 気になる点
    - `FastSpeech 2 - PyTorch Implementation`
        - F0値をpitch featureとして使っていて, これは一番古いバージョンの話らしい
        - 最新版はpitch spectrogram. ここは要変更か.
    - `Implementation Issues`
        - Postnetをdecoderで使ったとあったが, これはNARのやつでもそうとあった. `2.1 overview`
        - pitchとかenergy predictionはphoneme-levelでやったとあった: 元論文では特に言及がなかったように思える.

## フォルダ構造
- audio: 音声の変換のutilたちって感じ
    - tools.py
        - get_mel_from_wav
            - 実際にmelを吐き出す関数
            - 但し, stftで作ったオブジェクトが必要
    - stft.py
        - TacotronSTFT
            - preprocessで使われているのはこっちのstft.
            - こいつにconfigを渡して作成.
- config: datasetごとにyaml形式で書かれたconfig. どう扱われているかは見ていきたい.
- demo: demoのwavが入っている.
- hifigan: HiFi-GANというvocoder
- img: git用
- lexicon: 辞書っぽい. 日本語のはないので使わなさそう.
- model: モデルが入ってる.
    - 
- preprocessed_data: 前処理が終わったデータたちがデータセットごとにはいっている.
- preprocessor: 前処理用
    - preprocessor.py
        - build_from_path
            - pitch, energy, duration, mel-sepectを計算する.
            - mel自体は, audioのtoolを使って計算されている.
            - 全てをnp形式で保存.
    - 他の.pyはどこで使うのか?
- text: text関係の, 処理utilかな?
    - 
- transformer: こちらにもlayerなどがあり. Decoderとか.
    - 
- utils: その名の通り.
    - 


- ↑適宜, 必要なものを埋めていく.


## 使い方
### train
1. pipenv --python 3
2. pipenv install
3. unzip hifigan/generator_universal.pth.tar.zip
    - ただ, pthが出てきただけでした.
4. unzip preprocessed_data/JSUT/TextGrid.zip
    - TextGrid形式のファイルが出てきた. 各音素のinterval, 区切れが記載されていた. すごい.
5. mkdir -p raw_data/JSUT/JSUT
6. cp path/to/JSUT/*/wavs/*.wav raw_data/JSUT/JSUT
7. python retrieve_transcripts.py
    - ディクテーションされたテキストファイルからテキストを抜き出して, labファイルを作成.
8. python preprocess.py config/JSUT/preprocess.yaml #this may take some time. python3となっていたが, pythonなのでpythonに変更した.
    - preprocessorフォルダのpreprocessorを用いて, configを受け取りつつ, preprocessをする.
    - yamlファイルは, 万能の辞書みたいなもん?
        - 何重にも辞書にできるので, 階層構造が実現. 見やすいかも.
9. python train.py -p config/JSUT/preprocess.yaml -m config/JSUT/model.yaml -t config/JSUT/train.yaml
    - 全てのyamlを渡してtrain.

## ちょっとしたコメント、メモ
- scanssdではconfigはoptionsとかいう.pyのargparseで記録していたが,
普通にyamlのほうが見やすいしかしこい.
mainでargparseして3行くらいでyaml受け取るだけ. いい話.


## モデル構造解読
まずは, trainループの構成要素↓
- model
    - FastSpeech2: model/fastspeech2.py
- optim
    - ScheduledOptim: model/optimizer.py
- loss
    - FastSpeech2Loss: model/losses.py

軽そうなものから読んでいく.
- model/losses.py
    - フツーに計算してるだけっぽい.
    - [masked_select](https://pytorch.org/docs/stable/generated/torch.masked_select.html)
        - Trueのとこだけ取り出せる. 便利.
    - postnetのlossも計算してるの面白いな.
    - melはL1で, その他はmseで評価.
- model/optimizer.py
    - ただのschedularですね.
- model/fastspeech2.py
    - encoder: transformerへ.
    - variance_adaptor: modules.pyへ.
    - decoder: transformerへ.
    - mel_linear: 普通のlinerっぽいね. 間に挟んでいる.
    - postnet: transformerへ.

    - done.
    - encoder, decoderは基本的には普通のTransformer.
    - variance_adaptorがやっぱり曲者.
        - durationでは, 予測結果で, 音韻を引き延ばすことを行っている. 無理やり.
        - pitch, energyは割と単純に数層で予測して, embeddingで, その大きさらしさを学習させている.
    - postnetはただの最終調整. ふつーのcnn

- train.py
    - ちゃんと上から読んでいくことにする.
    - log
        - [Pytorch対応の, tensorboadを用いている.](https://qiita.com/nj_ryoo0/items/f3aac1c0e92b3295c101)
        - リアルタイムで更新してくれて, しかもaudioを保存. 学習途中の音声を聞けるのすごすぎる.

### todo
- vocoderでは関係なさそう...なぜ...?
    - lossでも, targetをmaskのshapeで切り取るということを行っている.
    - durationの違いはあくまでもdurationでのみ評価して, melではmelの違いだけを見るということか?
    - とりあえず, 今見ているmelの最長サイズは987なので, まぁ問題ないが,
    それ以上の長さの音声は完全に無駄になってしまうということに気を付けようか.

    - 12s以内にしておけば問題なさそう.