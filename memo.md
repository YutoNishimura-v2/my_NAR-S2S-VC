# my_NAR-S2S-VC: 20210528~
FastSpeech2のコードの一部を変更する形で, VCを実装していく.

- todo
    - synth_samplesで, mel_denormalizeをする.
        - transposeをしているので, どっちがどっちか分からないので、実行時確認.
    - inference_melでも, transposeが必要か不必要か, hifi-ganに入れてみてから確認していく.

## 必要なもの
- pyworld用
    - [C++のビルドツール](https://self-development.info/%E3%80%8Cmicrosoft-visual-c-14-0-or-greater-is-required-%E3%80%8D%E3%81%8C%E5%87%BA%E3%81%9F%E5%A0%B4%E5%90%88%E3%81%AE%E5%AF%BE%E5%87%A6%E6%96%B9%E6%B3%95/)
    - その他pyworldのrequirements
- pydub用
    - [ffmpeg](https://opty-life.com/apps/ffmpeg-windows-install/)

- すぐにいろんな環境で使えるように(あとはバージョン管理がめんどくさい)ので, pipenvを使いこなしたい...。
    - まずはtorchのinstallをするが, 一筋縄ではいかない模様.
    - torch: `https://download.pytorch.org/whl/cu102/torch-1.9.0%2Bcu102-cp37-cp37m-win_amd64.whl` 
        - おそらくこれは, `cuda10.2`の, `torch==1.9.0`で, `python 3.7`という意味っぽい. 最後のpythonが分からなかったが, cp39でダメだったのがこれだと行けたのでたぶんそう
    - torchvision: `https://download.pytorch.org/whl/cu102/torchvision-0.10.0%2Bcu102-cp37-cp37m-win_amd64.whl`
        - 間違ってずっと%2Bcuのついていないものをinstallしてた.
        - **一度↑こうやってミスってしまったら, キャッシュを消しに行こう!!**
            - `%LOCALAPPDATA%\pipenv\pipenv\Cache`: キャッシュの場所

    - pipenvは問題あり&installに時間かかるのでやっぱりなしで.
    - condaでもrequirementsを書くことが出来るらしいので, それでやってみる.
    - [参考](https://qiita.com/yubessy/items/2dd43551aa8308dc7eca)
        - この人みたいに, frozenを利用した方が良さそう.


    - RTX3090で使うには, CUDA10.2だとだめらしい.

    - tensorboardを使うには, 自動installされる[torch-tb-profilerを消さなきゃダメ](https://stackoverflow.com/questions/68049028/tensorboard-loading-forever-not-loading-in-vs-code-nor-in-terminal)
        - そんでもって, vscodeの機能では開かないので, cmdに`tensorboard --logdir $path`とする必要あり.

## 前処理の流れメモ
- preprocess.pyを実行するだけで勝手にやってくれるようにする.
    - このコードは, yamlを読み込んで, Preprocessorに渡して, build_from_pathを行う.
    - (追加)その前に, 無音部分消去などの音声に対する前処理を行う関数を追加.

- n2c_voiceprocess.py
    - 詳細はそこに書いたが, サンプリング周波数を変更したりするやーつ.
- preprocessor.py
    - build_from_path
        - melの正規化を行っている. 80次元あると思うが, 80個standardscalerを用意する. アホ.
        - pitch, energyは元論文に依って, logをとってから正規化を行う.
- calc_duration.py
    - duration_prepareのノートブックでやったことを書き直しただけ.

- 本来は, 句読点まで一致した読み合わせのペアを, JSUTとJSSSから選ぶ必要があるが, 
ここでの目的は, 僕の汚いズレた発音も含めてtrainingできるようにすることなので, あえて句読点の違う汚いデータも入れてみる.

## 変更点
- Encoder, DecoderのTransformerをConformerへ.: 20210528~20210529
    - espnet2のFastSpeech2はConformerを採用しているらしいので利用させてもらう.
    - 例のごとく読みづらいったりゃありゃしないので, [こちら](https://github.com/sooftware/conformer/blob/main/conformer/model.py)からいただく.
    - text embeddingなどは不要なことに注意.

    - paperには, `The attention dimension was set to 384` とあるが, 
    Varianceのhiddenは256だし, このConformerの実装にはattentionだけ次元を変える
    といった機能が付いていないみたい. 必要なら実装するが, そもそもattention部分
    のみをさしているかも正確には怪しい(他のところでも, module名は特定して言っているので, おそらくここでもattentionのことをまさに言っているはずではある)
    - ちなみに, transformerでは. `conv_filter_size`というパラメタがあり,
    FFTBlockの`PositionwiseFeedForward`にて, 1024までchannelを広げて戻してをやっている.
    - これに相当するものは一応あるにはあるが, conformerの実装では対応していないみたい. また, やっぱりattentionと明言しているので別の部分では?でもattentionって次元を自分でかえる部分ないけどな...。
    - 20210530: とりあえず放置.

    - 一応done.
    - position encoding抜いたり, まぁ一部Conformerの中身も弄ったくらいで, とりあえず形は揃えたという感じ. 形だけはあっているので, 大きな問題は起きなさそうである.


    - 20210628~20210628
    - attantionのdimentionを変える点
        - 先生とかに聞いたところ, 理由はなぞ.
        - ただ, attentionだけ次元変えるのが良さそう(Encoderまとめてでもいいが...。)
        - attentionだけ次元を変えられるように改造します.

    - 20210629~
    - pitrchからencoderへのgradient flowをなくしたい.
        - やっぱり過学習したので.
        - [lossで一部のrequires_gradをfalseにすれば良さそう](https://stackoverflow.com/questions/57945356/how-can-i-disable-gradient-updates-for-some-modules-in-autograd-backpropagation)
        - この手法だと, まず, requires_gradは計算前に設定しないと, もう意味がない.
        - つまり, energyの計算とそれ以外で別に行う必要があるが, それは計算無駄すぎる気がする.

        - ほかの手法として, やはりdetachしてしまうというのがある. pitchに足すencoderからの入力を
        detachしてしまえば, そこから後ろへはながれないはず.
        - [detachだけだと, メモリは共有されるらしいが](https://teratail.com/questions/234693)、
        むしろ, というよりも今回はあくまで勾配フローだけなくしたいので、逆にcloneしたら困りそう.

- Vocoderについて, PGANの追加: 最後: 20210628~
    - hifi-GANを試したが, 高音部分がダメそう.
        - さすがにfine-tuningしたら大丈夫なのか? いまいちわからない.
    - vocoderがうんちだともはや何も意味がないので, 先にやりたい.

    - vocoderはfinetuningをしたほうがよさそう. 精度を見て思いました.
    - finetuningに使用する入力は, 十分学習させた後の, FastSpeech2の出力.
        - それが一番最適だよね.
        - その代わり, 真のmelをreconstructする能力は失うことに注意.
    - finetuningさえすれば, 別にPWGANでなくても, hifi-GANで問題ない.

- duration教師データ用意について: 20210604~20210624
    - 先生に聞いたところ, 以下の2方針がよさそう.
    - melの次元を20に下げる
    - 前後の無声区間を削除する.

- data preparing の方式変更: 20210529~20210626
    - 改造部分
        - preprocessor.py: done
        - dataset.py: 20210530~done: Inference用のdatasetは未改修.


    - input, outputともにmelが必要ということを前提に用意していく.

    - mel, 正規化は未実装. 先生に聞いてみる.
        - 80次元あるが, しょうがない, 次元ごとに正規化が正しい.

- nars2svc.pyの作成: 20210530~20210626
    - now: Decoderの, mel_mask=Noneの場合の動作確認中


- Variance Predictorを, Converterへ.: ~20210626
    - duration について
        - paperでは, AR-modelを別に用意してattentionを利用...とやっているが, それこそが精度低下の
        原因だったはず.
        - なので, 何とかして, fastspeech2のやり方にのっとりたい.
        - 今回, textは完全に同じというところを利用する.
        - sourceとtargetで, durationのtargetが2つ手に入るはず.
        - それを用いて, 各音素ごとに, melの特徴量をexpandしてつじつまを合わせる.
        - 手法としては, 画像の圧縮拡大のものを利用しよう.

        - ↑これのせいで, sourceとtargetどちらのdurationもpredictする必要があることに注意.
        なので, 一部重みを共有した, 2つ出力可能なduration predictorを作成する.

        - また, inference時は不要そうだがどうだろう. onにしてももちろんいいけど,
        速度向上のためにはoffでいいんじゃないか? 要検討.

        - 20210624追記
            - これに関して, durationは結局0未満を許さないようにした. なので従来通りのコードで特に問題はなさそう. 当然動作確認は必須だが.

    - 特筆すべき変更点
        - phoneme levelの廃止
        - 今回は, すべてframeレベルに注意.
        - ここでやっていることは, maskをどちらでとるか.
            - また, maskは, melをbatchにするためにpaddingが必要なので,
            - そのためのmask.
  
- 全重みがinitで初期化されるようにする: pytorchでは, linearなどと作った時点でHeで初期化されている.

- (reduction factor): よくわからない. melのフレームを一回で数フレーム一気に出す話らしい. クオリティ出したいしいらないかも...?
    - [tactron2では, 用いていないらしい](https://paperswithcode.com/method/tacotron-2) のでいったん不要か. 実行時間減少に寄与.

- inference関連の整備: 20210626~20210628
    - inference.pyを動くようにしないと.
    - 結構変えているので注意.

    - higi-ganをtrainする用に, 訓練済みモデルを使用して出力したmelを用意したい.
        - ただ, 無音区間の削除なども行ってしまっているため,単純に未知データ用のinferenceを適用してしまうと, ズレる.
        - なので, 想定としては, 
        - input_path: 既にNARS2Sをtrainするのに使ったmel, energy等々があると思うので、そのpathを指定する.
        - output_pathは適当.
        - configもちゃんと実行したいmodelのconfigを指定してあげる.
        - ファイル名はmelと音声で一致していると, hifiに渡すときスムーズ.
            - JSSSとJSUTではそうなっているが, N2Cではそうなっていない.
            - それはまぁ適当なスクリプトでも入れてあげればいい.


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