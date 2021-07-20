# 今までのtrainの履歴を残しておく.

- NARS2S_1回目
    - date: 20210629
    - output_folder_name: JSUT_JSSS
    - dataset: JSUT_JSSS
    - options: default

    - memo
        - 初めての実行. ほぼ動作確認.
        - pitchのみ過学習. gradient flowを止めてみる.
        - また, configを次回以降はlogのoutput_folder_name直下にcpするようにする.

- NARS2S_2回目
    - date: 20210629
    - output_folder_name: JSUT_2_JSSS
    - dataset: JSUT_JSSS
    - options
        - pitch: stop_gradient_flow = True
    
    - memo
        - 二回目の実行.
        - pitchのgradient_flowを止めたら, 見事に過学習せず.
        - 40,000iterできれいな音声に!
        - 一方で, durationとenergyは過学習気味.
        - この2つに関しても, 層を深くしてgradient_flowをstopしてみる.

- NARS2S_3回目
    - date: 20210630
    - output_folder_name: JSUT_2_JSSS_2
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
        - energy
            - stop_gradient_flow = True
            - layer = 5
        - duration
            - stop_gradient_flow = True
            - layer = 5
    
    - memo
        - duration
            - val精度変わらず.
            - 過学習すらしていない.
            - durationに関しては, これ以上の改善は難しそうである.
                - データの問題説がありそう. val=trainだし...。
        - energy
            - 悪化.
            - eenergyにはgradient_flowが必要だった.
        - pitch
            - ほぼ変わらず. 若干過学習が抑えられた? 
            - valは一定.
        - mel2つ
            - 変わらず.
    
    - next
        - duration
            - 恐らく, valはこれ以上下がらない.
            - 如何にして過学習を抑えるか. 
                - その観点でいえば, gradient_flow=Falseで良さそう.
            - 次は, 恐らく最悪ではあるが,
            layer=5, gradient_flowありでやってみる.

        - energy
            - まず, gradient_flowは必須.
            - なので, これもgradient_flowとの兼ね合いでどれほど過学習するかを見る.
        
        - pitch
            - pitchはこれでよさそう.
            - 実験的に, layer=8にしてみる.

- NARS2S_4回目
    - date: 20210630
    - output_folder_name: JSUT_2_JSSS_3
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
            - layer = 8
        - energy
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - stop_gradient_flow = False
            - layer = 5

    - memo
        - pitch
            - ほぼ変わらず. ほんとにほんと少しだけvalもtrainも小さい.
            - スピード無視ならこれでいいかな?
        
        - energy
            - layer2, flowありと, train一緒, val低いという最高の結果. いいね.
        
        - duration
            - layer2, flowありと, train若干まし, val同じという少しマシな結果.
            - layer2, flowなしなら同じval, trainもvalと同じ
            - まぁ過学習しない方が良さそう.
        
        - mel
            - まあ何もいじってないからそりゃ変わらない.

    - next
        - 最終学習. 決定版でいいかな. 今までのbestで行く.
        - 現段階のbest
            - pitch
                - stop_gradient_flow = True
                - layer = 8
            - energy
                - stop_gradient_flow = False
                - layer = 5
            - duration
                - stop_gradient_flow = True
                - layer = 5
        - ほかに気になるパラメタ
            - pitch
                - gradient_flow = True, layer = 2ならどうだろうか
            - energy
                - gradient_flow = False, layer = 8 としたら?
            - duration
                - gradient_flow = False, layer = 2なら?

            - あとは, attention_dimを落とした場合, encoder全体を384にした場合など気になる.

- NARS2S_5回目
    - date: 20210701
    - output_folder_name: JSUT_2_JSSS_4
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = False
            - layer = 2
        - energy
            - stop_gradient_flow = False
            - layer = 8
        - duration
            - stop_gradient_flow = True
            - layer = 2

    - memo
        - pitch
            - がっつり過学習. ダメみたいですね
        - energy
            - (一個前と比べて)変化なし.
        - duration
            - 同様にtrain_val張り付きくん.
        - mel
            - 変わらず.

    - next
        - すべてbestパラメタで, encoderとattention dimをいじりたい.
        - pitch
            - 誤差ではあるが, JSUT_2_JSSS
            - stop_gradient_flow = True
            - layer = 5
        - energy
            - JSUT_2_JSSS_3
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - JSUT_2_JSSS_2
            - stop_gradient_flow = True
            - layer = 5
        
        - attention_dim = 256 
            - 次は, encoder=attention=384

- NARS2S_6回目
    - date: 20210701
    - output_folder_name: JSUT_2_JSSS_5
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
            - layer = 5
        - energy
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - stop_gradient_flow = True
            - layer = 5
        - attention_dim = 256 

    - memo
        - それぞれ, best時のものと比べる
        - pitch
            - びっくりするほど一致.
            - 変化なし.
        - energy
            - 同じく.
        - duration
            - 同じく.
        - mel
            - 変わらん....

        - attentionの次元, 増やすだけ無駄だったかもしれん.
            - データ量が増えるとまた違うかも.
            - 増えた時用のために, 大きめにしておくのはあり.

    - next
        - pitchたちはそのまま
        - encoderも384にしてみる.

- NARS2S_7回目
    - date: 20210701
    - output_folder_name: JSUT_2_JSSS_6
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
            - layer = 5
        - energy
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - stop_gradient_flow = True
            - layer = 5

        - encoder_hidden = 384
        - decoder_hidden = 384
        - attention_dim = 384 
        - filter_size: 384
    
    - memo
        - mel以外: 変わらず
        - mel: 思い切り過学習
    
    - next
        - これで終わり. どちらもbestで行く.

- NARS2S_8回目
    - date: 20210701
    - output_folder_name: JSUT_2_JSSS_7
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
            - layer = 5
        - energy
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - stop_gradient_flow = True
            - layer = 5

        - annotation = 384

    - memo
        - ok.
        - 40,000 stepが一番よさそう.
        - さっそく, melを作成しに行く.

- inference
    - `python inference.py --restore_step 40000 --input_path ./preprocessed_data/JSUT_JSSS/source --output_path ./output/mel_for_hifi-gan/JSUT_2_JSSS -p ./output/log/JSUT_2_JSSS_7/preprocess.yaml -m ./output/log/JSUT_2_JSSS_7/model.yaml -t ./output/log/JSUT_2_JSSS_7/train.yaml --get_mel_for_hifigan`

- Hifi-gan_1回目
    - date: 20210703
    - output_folder_name: JSUT_2_JSSS_1
    - dataset: JSUT_JSSSで, source: JSUT
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - 設定が違うので, trainingからする必要あり...。
        - `python train.py --input_mel_path ../output/mel_for_hifi-gan/JSUT_2_JSSS --input_wav_path ../pre_voice/JSUT_JSSS/JSSS --checkpoint_path ./output/JSUT_2_JSSS_1 --config ./configs/config_JSUT_JSSS.json`

        - hop_length　= 300
            - これにすると, 倍率から変える必要あり.
            - dilationは基本, upsampleでやっているみたい.
                - resblockのdilationはpadなのでkernelと対応.
                - 4回のupsampleをすべてかけてたらhop_sizeになるようにする.

    - inference2
        - 長さ合わない問題のために, durationをforcingすることに.
        - `python inference.py --restore_step 40000 --input_path ./preprocessed_data/JSUT_JSSS/source --output_path ./output/mel_for_hifi-gan/JSUT_2_JSSS_2 -p ./output/log/JSUT_2_JSSS_7/preprocess.yaml -m ./output/log/JSUT_2_JSSS_7/model.yaml -t ./output/log/JSUT_2_JSSS_7/train.yaml --get_mel_for_hifigan`

- Hifi-gan_2回目
    - date: 20210703
    - output_folder_name: JSUT_2_JSSS_2
    - dataset: JSUT_JSSSで, source: JSUT
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - 設定が違うので, trainingからする必要あり...。
        - `python ./hifigan/train.py --input_mel_path ./output/mel_for_hifi-gan/JSUT_2_JSSS_2 --input_wav_path ./pre_voice/JSUT_JSSS/JSSS --checkpoint_path ./hifigan/output/JSUT_2_JSSS_2 --config ./hifigan/configs/config_JSUT_JSSS.json`

        - hop_length　= 300
            - これにすると, 倍率から変える必要あり.
            - dilationは基本, upsampleでやっているみたい.
                - resblockのdilationはpadなのでkernelと対応.
                - 4回のupsampleをすべてかけてたらhop_sizeになるようにする.

        - なぜか, melの計算が合わない問題...。
            - 0002で実験.
            - durationでも380が正解.
            - source, targetのmelはともに380になっている.
            - なぜか, generatorの出力をmel化したものだけ381になってしまう...。
            - 超凡ミスなコードミスでした.
        
            - melをNARS2Sのものから戻したら超高速化出来た。おそらくGPU対応していなかったんだろう...。

        - そして普通にdurationのミスが発覚...。

- NARS2S: データ用意
    - options
        - source_raw_path: "./raw_data/JSUT_JSSS/JSUT"
        - source_prevoice_path: "./pre_voice/JSUT_JSSS_2/JSUT"
        - target_raw_path: "./raw_data/JSUT_JSSS/JSSS"
        - target_prevoice_path: "./pre_voice/JSUT_JSSS_2/JSSS"
        - preprocessed_path: "./preprocessed_data/JSUT_JSSS_2"

- NARS2S_9回目
    - date: 20210703
    - output_folder_name: JSUT_2_JSSS_8
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
            - layer = 5
        - energy
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - stop_gradient_flow = True
            - layer = 5

        - annotation = 384
    
    - memo
        - durationを間違えていたので, やり直し.
        - ついでに, ちゃんと無音区間を消すことにした.

        - durationはよくなったがそれ以外が軒並み超絶悪化
        - 先生の言葉を見返したら, 冒頭と末尾の無音を消せというお話だった.
            - 途中は消さないほうが良さそう...。
            - 先頭と最後尾のみを削除するコードに変更して再実行してみる.

- NARS2S: データ用意
    - options
        - source_raw_path: "./raw_data/JSUT_JSSS/JSUT"
        - source_prevoice_path: "./pre_voice/JSUT_JSSS_3/JSUT"
        - target_raw_path: "./raw_data/JSUT_JSSS/JSSS"
        - target_prevoice_path: "./pre_voice/JSUT_JSSS_3/JSSS"
        - preprocessed_path: "./preprocessed_data/JSUT_JSSS_3"

- NARS2S_10回目
    - date: 20210703
    - output_folder_name: JSUT_2_JSSS_9
    - dataset: JSUT_JSSS
    - options
        - pitch
            - stop_gradient_flow = True
            - layer = 5
        - energy
            - stop_gradient_flow = False
            - layer = 5
        - duration
            - stop_gradient_flow = True
            - layer = 5

        - annotation = 384
    
    - memo
        - durationを間違えていたので, やり直し.
        - ついでに, ちゃんと無音区間を消すことにした.

        - durationはよくなったがそれ以外が軒並み超絶悪化
        - 先生の言葉を見返したら, 冒頭と末尾の無音を消せというお話だった.
            - 途中は消さないほうが良さそう...。
            - 先頭と最後尾のみを削除するコードに変更して再実行してみる.

        - 途中を消さないので再実行したら何とかなった。
            - 精度はちょっと落ちたか？
            - 正直、無音部分のほうが推定は簡単なので、そこが減った分lossが上がったと推察される.

- inference
    - `python inference.py --restore_step 50000 --input_path ./preprocessed_data/JSUT_JSSS_3/source --output_path ./output/mel_for_hifi-gan/JSUT_2_JSSS_3 -p ./output/log/JSUT_2_JSSS_9/preprocess.yaml -m ./output/log/JSUT_2_JSSS_9/model.yaml -t ./output/log/JSUT_2_JSSS_9/train.yaml --get_mel_for_hifigan`

- Hifi-gan_3回目
    - date: 20210704
    - output_folder_name: JSUT_2_JSSS_3
    - dataset: JSUT_JSSSで, source: JSUT
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - 設定が違うので, trainingからする必要あり...。
        - `python ./hifigan/train.py --input_mel_path ./output/mel_for_hifi-gan/JSUT_2_JSSS_3 --input_wav_path ./pre_voice/JSUT_JSSS_3/JSSS --checkpoint_path ./hifigan/output/JSUT_2_JSSS_3 --config ./hifigan/configs/config_JSUT_JSSS.json`

        - 全然だめ. 圧倒的データ不足...。
        - ちゃんと, 沢山データ集めてそれで学習します.

- get_out_of_wavs
    - まずは, datasetからwavのみ取り出す.

- make_dataset
    - そして, melとtrain.txt, val.txtを作成する.
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/Universal --pre_voice_path ./pre_voice/Universal --output_path ./preprocessed_data/Universal -p ./config/JSUT_JSSS/preprocess.yaml`
    - ここでは, configのaudio情報しか利用しないことに注意.

    - Universalとは,
        - JSUT
        - JVS
        - LJSpeech
        - VCTK
    - この4つを混ぜたもの.
    - これでも足りなければ, 500h分もあるデータセットを追加予定.
    - pre_voiceにあるのは, sr飲みかえたもの. 無音区間削除等はしていないことに注意.

- Hifi-gan_4回目
    - date: 20210705
    - output_folder_name: Universal_1
    - dataset: いろんなデータセット詰め合わせ
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - config自体は, JSUTと変えない、というか変えたら意味ない.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal --input_wav_path ./pre_voice/Universal --checkpoint_path ./hifigan/output/Universal --config ./hifigan/configs/config_Universal.json`

        - 途中からまったく学習が進まない事態に...。
        - 考えられる原因は以下
            - パラメタが無理
                - 変えた部分
                - upsamplingの倍率
                    - hop_lengthの因数分解しか許されないのでああやるしかない
                - fmaxの上限がnull
                    - これも, 高周波成分をちゃんと変換したいので残したいが...。
            - 最初から別ドメイン混ぜまくりは無理
                - これなら別々に学習させるだけなのでありがたい
                - その一方で, これが原因だとすると, JSUT_JSSSで失敗した理由が説明できない
                - データ数が少ないといっても, 一応3000データはあったので...。
                - これを試して無理だったら、もうパラメタを諦めるか, hifiganを諦めるかする必要がありそう..

        - とりあえず対策
            - 一部の重みを再利用して転移学習してみる.
            - upsample部分は合わないが, resblockの重みはハマるはずなのでそこを利用.

- Hifi-gan_5回目
    - date: 20210705
    - output_folder_name: Universal_2
    - dataset: いろんなデータセット詰め合わせ
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - Universalのfinetuningとして行ってみる.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal --input_wav_path ./pre_voice/Universal --checkpoint_path ./hifigan/output/Universal_2 --config ./hifigan/configs/config_Universal.json --load_model_only`

        - ダメでした.
        - melが変なのが気になる
            - melについて調査しよう.
            - 結果, audio_clipをすることによって変なことになっていることが判明.
                - 公式ではこれをdatasetとして使っているが, 学習が進まない以上これでやってみるしかない.

- make_dataset
    - 作り直し.
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/Universal --pre_voice_path ./pre_voice/Universal --output_path ./preprocessed_data/Universal_2 -p ./config/JSUT_JSSS/preprocess.yaml`
    - ここでは, configのaudio情報しか利用しないことに注意.

- Hifi-gan_6回目
    - date: 20210705
    - output_folder_name: Universal_3
    - dataset: いろんなデータセット詰め合わせ
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - Universalのfinetuningとして行ってみる.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal_2 --input_wav_path ./pre_voice/Universal --checkpoint_path ./hifigan/output/Universal_3 --config ./hifigan/configs/config_Universal.json --load_model_only`

        - 大分下がったが, 0.4止まりみたい. 継続する価値はあるが, その前に他のことを試したい.

- Hifi-gan_7回目
    - date: 20210706
    - output_folder_name: Universal_4
    - dataset: いろんなデータセット詰め合わせ
    - options
        - training
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - 1から学習してみる
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal_2 --input_wav_path ./pre_voice/Universal --checkpoint_path ./hifigan/output/Universal_4 --config ./hifigan/configs/config_Universal.json`

        - 結局, finetuningとほぼ変わらず.
        - いまだにデータ量なのかデータの質(ばらつき)なのか測りかねているので, 2つ実験をする.
            1. さらにデータを追加. この傾向だと, 0.2まで行ってくれるのではないか.
            2. パラメタの原因かどうかを探る. そのために, まずはLJSpeechのみで訓練をしてみる. それでうまくいけば, 一気にいろんなドメインで学習するのがあほだし, うまくいかなければパラメタが原因の可能性が高い. 正直, upsamplingよりも, fmax=nullがだいぶ怪しそう....。


- make_dataset
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/LJSpeech --pre_voice_path ./pre_voice/LJSpeech --output_path ./preprocessed_data/LJSpeech -p ./config/JSUT_JSSS/preprocess.yaml`
    - ここでは, configのaudio情報しか利用しないことに注意.

- preprocess
    - 次は, JSSS_2_JSUTを行う.
    - そのために, まずはpreprocessed_dataを作成する.
    - といっても, melとかは作り直す必要がないので, source, targetだけひっくり返して, durationのみ作成する.

- NARS2S_finetuning_1回目
    - date: 20210706
    - output_folder_name: JSSS_2_JSUT
    - dataset: JSSS_2_JSUT
    - options
        - finetuning
            - JSUT_2_JSSS_9 の 50000からスタート
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo 
        - `python train.py --restore_step 50000 -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`

        - 途中で学習が止まった. pitchとかは特に. melはゆるやかな減少.
        - optimは読み込まないモードを再び.

- make_dataset
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/LibriTTS --pre_voice_path ./pre_voice/LibriTTS --output_path ./preprocessed_data/LibriTTS -p ./config/JSUT_JSSS/preprocess.yaml`
    - ここでは, configのaudio情報しか利用しないことに注意.

- NARS2S_finetuning_2回目
    - date: 20210706
    - output_folder_name: JSSS_2_JSUT_2
    - dataset: JSSS_2_JSUT
    - options
        - finetuning
            - JSUT_2_JSSS_9 の 50000からスタート
            - optimizerは初期化.
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo 
        - `python train.py --restore_step 50000 -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`

        - 途中で学習が止まった. pitchとかは特に. melはゆるやかな減少.
        - optimは読み込まないモードを再び.

        - melはめちゃ過学習...。
        - それ以外はうまくいっているが...。 普通に, 1回目をもう少しまわすべきだった. 次はそうする.

- Hifi-gan_8回目
    - date: 20210706
    - output_folder_name: LJSpeech_1
    - dataset: LJSpeech
    - options
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo
        - 1から学習してみる.
        - これでうまく行けば, 異なるドメインを一気に学習があほだとわかり, うまくいかなければ, パラメタが怪しい.
        - 苦労したけど, colabでは出来なかった...。
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/LJSpeech --input_wav_path ./pre_voice/LJSpeech --checkpoint_path ./hifigan/output/LJSpeech_1 --config ./hifigan/configs/config_LJSpeech.json`

        - 恐らく, `異なるドメインを一気に学習がアホ`が確かめられた.
            - 本来データが山ほどあって有利なはずのUniversalよりも, LJSpeech単体でよい成績を残している.
            - なので, 別々に学習するべき.
        
        - 以降, これを使ってfinetuningしていく.

- Hifi-gan_9回目
    - date: 20210706
    - output_folder_name: LibriTTS_1
    - dataset: LibriTTS
    - options
        - LJSpeechの続き.
        - finetuning
        - batch_size = 12: 謎のエラーとして出るから注意.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/LibriTTS --input_wav_path ./pre_voice/LibriTTS --checkpoint_path ./hifigan/output/LibriTTS_1 --config ./hifigan/configs/config_LJSpeech.json`

        - まったく学習がうまく進まず...。
        - データの質が悪い説があるので, 他のデータでfinetuningする.

- NARS2S_finetuning_3回目
    - date: 20210706
    - output_folder_name: JSSS_2_JSUT
    - dataset: JSSS_2_JSUT
    - options
        - finetuning
            - JSSS_2_JSUT の 52500からスタート
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo 
        - `python train.py --restore_step 52500 -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`

        - さっきよりは大分ましだが, energyが悲惨なことになっているし, melもfinetuning以前より落ちる気配がない.
        - まず, energy, durationは指定通りlayer = 2にしたほうがよさそう.
        - そんで, mel対策としては, finetuningをスタートするのを30000くらいのもっと早めに始める.

        - なので, permを変えて訓練しなおし.

make_dataset
    - Universalではなく, 1つ1つ単体のデータセットを用意しておく.
    - mel_check.ipynbで, Universalから計算済みのmelやpre_voiceをcopyして抜き出すことで対応する.
        - なので, やることはtrain.txt, val.txt作るだけ.
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/LJSpeech --pre_voice_path ./pre_voice/jsut_ver1.1 --output_path ./preprocessed_data/jsut_ver1.1 -p ./config/JSUT_JSSS/preprocess.yaml --val_num 500`
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/LJSpeech --pre_voice_path ./pre_voice/jvs_ver1 --output_path ./preprocessed_data/jvs_ver1 -p ./config/JSUT_JSSS/preprocess.yaml --val_num 300`
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/LJSpeech --pre_voice_path ./pre_voice/VCTK-Corpus --output_path ./preprocessed_data/VCTK-Corpus -p ./config/JSUT_JSSS/preprocess.yaml`

- Hifi-gan_10回目
    - date: 20210706
    - output_folder_name: VCTK-Corpus_1
    - dataset: VCTK-Corpus
    - options
        - LJSpeechの続き.
        - finetuning
        - batch_size = 12: 謎のエラーとして出るから注意.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/VCTK-Corpus --input_wav_path ./pre_voice/VCTK-Corpus --checkpoint_path ./hifigan/output/VCTK-Corpus_1 --config ./hifigan/configs/config_LJSpeech.json`

        
    - memo
        - 全くうまくいかず...。
        - さすがにfinetuningについて色々考え直した方が良さそう.


- issueを読んで分かったこと.
    - [44kHzの人](https://github.com/jik876/hifi-gan/issues/7)
        - "discriminator_periods": [3, 5, 7, 11, 17, 23, 37],
        - これにするとよかったらしい.
    - [48kの人](https://github.com/jik876/hifi-gan/issues/11)
        - upsample数をいじっている人もいる.
    - [split_sizeについて](https://github.com/jik876/hifi-gan/issues/38)
        - segment_size / hop_size = 32になると良さそうらしい.
    - [Universalはどうやって作られた?](https://github.com/jik876/hifi-gan/issues/46)
        - LibriTTS, VCTK, LJSpeechらしい.
        - ちゃんと, finetuningの設定でやってみるか.    
    - [44100はどんなパラメタがよい?](https://github.com/jik876/hifi-gan/issues/58)
        - upsampling系や, kernel, pediodsをいじると良さそうらしい.

- これを踏まえて以下の変更を加えて再実行したい
    - descriminator_preriodsの変更
    - upsample数の増加
    - segment_size / hop_size = 32
    - generatorのkernel sizeの増加
    - Libriも含めたUniversalデータの作成.


- pre_voice: Universal_2の作成
- preprocessed_data: Universal_3の作成.
    - どちらも, Libiriも加えたすべてのデータにする.
    - valは3000.

    - jsut_ver1.1: 7696
    - jvs_ver1: 3110
    - LibriTTS: 205044
    - LJSpeech: 13100
    - VCTK-Corpus: 44257
    - sum: 273207
- makedataset
    - `python ./hifigan/make_dataset.py --input_path ./raw_data/LJSpeech --pre_voice_path ./pre_voice/Universal_2 --output_path ./preprocessed_data/Universal_3 -p ./config/JSUT_JSSS/preprocess.yaml --val_num 3000`

- NARS2S_new_1回目
    - date: 20210707
    - output_folder_name: JSUT_2_JSSS_10
    - dataset: JSUT_JSSS_3
    - options
        - train
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`
        - energyとdurationの層を浅くして, そのうえで30000くらいでiterは止めてfinetuningに備える.



- Hifi-gan_11回目
    - date: 20210708
    - output_folder_name: Universal_5
    - dataset: Universal_3
    - options
        - training.
        - batch_size = 12: 謎のエラーとして出るから注意.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal_3 --input_wav_path ./pre_voice/Universal_2 --checkpoint_path ./hifigan/output/Universal_5 --config ./hifigan/configs/config_Universal_2.json`

    - memo
        - 結局, modelの容量を増やしたりしてみたが, 大局は変わらないので, finetuning周りで元の実装に戻してみた.

        - 結果, LJSpeech単体と変わらず....どうして....

- NARS2S_new_finetuning_1回目
    - date: 20210708
    - output_folder_name: JSSS_2_JSUT_3
    - dataset: JSSS_2_JSUT
    - options
        - train
        - batch_size = 12: 謎のエラーとして出るから注意.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 20000`
        - energyとdurationの層を浅くして, そのうえで30000くらいでiterは止めてfinetuningに備える.

        - memo
            - だめ。energyも過学習してるし, melも下がらんし, 正直前と変わらん....。
            - 研究室でお話を聞く

        
        - 質問回答メモ
            - まぁ普通に考えて, trainはX→YとY→X同時にやるべき
            - targetの逆正規化はtargetのものでやるべき←これ, 未実装.
                - それなら, 推論時はtargetのものはないので, 不可能
                - というよりそもそもtargetのmelのnormalize不要な気がする
                - 論文にも, sourceはそうするが, targetについての記載はまったくなかったので.
                - なくします.
            - finetuningの際, optimizerはリセット. 一方で, スタートのlrは小さくしたほうがよさそう
                - リセットはわかる. 確かにリセットしたほうが良さそう.
            
            - vocoderが悪いだけな気がする. つまり, 確かにみかけlossは下がっていないが, 実は既にうまくいってる説
                - 僕が気にしていたノイズも, 普通にmelが下がったもので乗っていた. なので確かにその説が濃厚.
            
            - energy過学習は気にしなくてよさそう?
                - ablation studyを見るとenergyは主観にほぼ影響しないので.
            
            - 残る疑問は, 同発話で簡単っぽいタスクなのにlossが下がらないのはなぜ??
                - これは先生も未経験.
                - とりあえずは, 出来ること(optimizerの話)をやってみるべき.


- targetのmelは正規化しないように変更.
    - それに伴い, datasetを作り直し.
    - `python preprocess.py config/JSUT_JSSS/preprocess.yaml`
    - prevoiceはJSUT_JSSS_3を引き続き利用.
    - preprocessedとしてJSUT_JSSS_4に.
    - durationはmel_num=20で作り直しているので, 影響受けない.
    - なのでmelのとこだけ計算しなおす.

    - durationは同じだから...と思ってJSUT_JSSS_3のものをコピペしたらエラーでまくった. printしようとするとエラーが出なくなる意味不明なエラー. しょうがないのでちゃんと作り直し.
    - そのdebugをしている最中に, 音声がぶつ切りに合っていることを発見してしまった. ちゃんと音声も確認して閾値は決めよう.

    - pre_voiceから, JSUT_JSSS_4として作り直し

- NARS2S_11回目
    - date: 20210708
    - output_folder_name: JSUT_2_JSSS_11
    - dataset: JSUT_JSSS_4
    - options
        - energyを元に戻した
        - targetのmelをnormalizeするのやめた
        - preprocessで, tailの閾値を低くしてぶつ切りをなくした
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`

- makedataset
    - 一方だけmelを正規化していないので, ちゃんと作り直す.
    - pre_voice: JSUT_JSSS_4
    - preprocessed_Data: JSSS_2_JSUT_2

- NARS2S_new_finetuning_2回目
    - date: 20210708
    - output_folder_name: JSSS_2_JSUT_4
    - dataset: JSSS_2_JSUT_2
    - options
        - lrはreset.
        - JSUT_2_JSSS_11の35000からスタート.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 35000`

        - 結果: うまくいかない
        - targetのmelを正規化する前よりもうまくいっていない...。
        - やはり, finetuning時は, input, outputどちらもtrain時のstatsを使うべきなきがする.
        - それでやってみます.

- targetのmelを正規化するように直す.
    - 但し, finetuning時は, statsを配置するように変更.
    - pretrain modelは, JSUT_2_JSSS_10
        - mel正規化されているはず. ただのパラメタ変えただけのものなので.
        - pre_voice: JSUT_JSSS_3
        - preprocessed: JSUT_JSSS_3
    - なので, このpreprocessedで作られたstatsを用いて正規化をしよう!!
    - `python preprocess.py --config ./config/JSUT_JSSS/preprocess.yaml --finetuning`

- NARS2S_new_finetuning_3回目
    - date: 20210709
    - output_folder_name: JSSS_2_JSUT_5
    - dataset: JSSS_2_JSUT_3
    - options
        - lrはreset.
        - JSUT_2_JSSS_10の20000からスタート.
            - energyのlayerが2ですね
        - melの正規化はやり直し.
        - statsに関して, train時のモノで正規化をする事にしてみる.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 20000`

        - あとはinit_lrも変更したい(今1)
        - あと, 正規化の数字を補正するのもあり(finetuningのデータのmeanとstdも使って補正かけるということ).

        - いい感じ. 
        - 欲を言えばもう少し下がってほしいので, lrの件をやってみることにする.
        - そっちで失敗したらこれを訓練続行する.

- NARS2S_new_finetuning_4回目
    - date: 20210709
    - output_folder_name: JSSS_2_JSUT_6
    - dataset: JSSS_2_JSUT_3
    - options
        - lrはreset. init_lr = 0.1
        - JSUT_2_JSSS_10の20000からスタート.
            - energyのlayerが2ですね
        - statsに関して, train時のモノで正規化をする事にしてみる.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 20000`

        - validationは低いが, trainがなかなか下がらん, 失敗.
        - 一個前の続きを行うことにする.

- NARS2S_new_finetuning_3回目
    - date: 20210709
    - output_folder_name: JSSS_2_JSUT_5
    - dataset: JSSS_2_JSUT_3
    - options
        - lrはreset.
        - JSUT_2_JSSS_10の20000からスタート.
            - energyのlayerが2ですね
        - melの正規化はやり直し.
        - statsに関して, train時のモノで正規化をする事にしてみる.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 22500`

- NARS2S_new_finetuning_5回目
    - date: 20210709
    - output_folder_name: JSSS_2_JSUT_7
    - dataset: JSSS_2_JSUT_3
    - options
        - lrはreset. 10倍スタートしてみる.
        - JSUT_2_JSSS_10の20000からスタート.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 20000`

        - 普通にダメでした.

        - ここまでやった感じ, 他にできることはもうないので, 次にやることは同時訓練をしておいてN2Cに備えるくらいですね.
        
        - 実験からわかったことをまとめると
            - mel lossは0.3未満なら普通に大丈夫そう
            - source, targetともにfinetuning時はpre_trainの正規化特徴量で正規化しよう.
            - lrはresetしたほうが良いが, その時のlrは初期値でよさそう. 絶妙な値だった.
        

    - next
        - X→Y, Y→Xを同時に学習.
        - そのためには, prevoiceから用意が必要. ファイル名も変える必要があるので...。
        - とりあえずhifiganを訓練しておいて時間あるときにやりましょう.

- Hifi-gan_11回目
    - date: 20210710
    - output_folder_name: Universal_5
    - dataset: Universal_3
    - options
        - training.
            - Universal_5の続き.
        - batch_size = 12: 謎のエラーとして出るから注意.
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal_3 --input_wav_path ./pre_voice/Universal_2 --checkpoint_path ./hifigan/output/Universal_5 --config ./hifigan/configs/config_Universal_2.json --checkpoint_interval 25000 --summary_interval 500 --validation_interval 5000`
            - hifiganでtrainingを続ける場合は勝手に重み探して継続してくれるからいいね.
            - intervalを5倍にした. 1Mやる前提なので, そんなに頻繁に出されてもこまるので.

    - memo
        - やはりうまくいかない。。。。。
        - もはやデータ量の問題でもない気がする(一応LJSpeechは改良できるみたい)
        - 一応, LibriTTSを全部使ってなかったみたいなので, あと400hは増やせるが、本質的でない気もする...。
        - 全てが終わってGPUをもてあましたらやるくらい。

        - とりあえず今は, 22050に合わせることにしてみる.
            - 22050でも悪くない気がする.
            - fmax8000の問題も, finetuningすれば行けると信じる.

- makedataset for NARS2S
    - hifiganのconfigに合うように訓練してみる.
    - sr = 22050
    - n_fft = 1024
    - hop_size = 256
    - win_size = 1024
    - また, X→YとY→Xは同時に行う.

    - `python preprocess.py --config ./config/JSUT_to_from_JSSS/preprocess.yaml`

- NARS2S_new_1回目
    - date: 20210710
    - output_folder_name: JSUT_to_from_JSSS
    - dataset: JSUT_to_from_JSSS
    - options
        - 初のhifiganに合わせたパラメタ設定. どうなることやら.
    
    - memo 
        - `python train.py -p ./config/JSUT_to_from_JSSS/preprocess.yaml -t ./config/JSUT_to_from_JSSS/train.yaml -m ./config/JSUT_to_from_JSSS/model.yaml`

        - 学習が進まない...妙だな....
            - よく考えたら, 話者が異なるんだから混ぜたら困るのは当然!!!!
            - 話者指定してあげないと無理だよ!!!!!
            - ということで, speaker情報を復活させます...。

            - それはよくない.
            - inputとoutput両方にspeakerのembeddingが必要だろう.
                - でもfinetuningする前提ならそっちのほうがうまくいったりするのか...?
                - とりあえず, 普通にfinetuningするほうで実験を進めつつ, 改造しよう.

- makedataset for NARS2S
    - hifiganのconfigに合うように訓練してみる.
    - sr = 22050
    - n_fft = 1024
    - hop_size = 256
    - win_size = 1024
    - X→Yは同時に行わず, 今まで通りまずはJSUT_JSSSのpre_voiceを作成する.

    - pre_voice: JSUT_JSSS_5
    - preprocessed: JSUT_JSSS_5
    
    - pre_voice: JSUT_JSSS_5
    - preprocessed: JSSS_2_JSUT_4
        - finetuning用に, JSUT_JSSS_5のstatsを利用することに注意.

    - `python preprocess.py --config ./config/JSUT_JSSS/preprocess.yaml`

- NARS2S_new_1回目
    - date: 20210710
    - output_folder_name: JSUT_2_JSSS_12
    - dataset: JSUT_JSSS_5
    - options
        - hifiganのパラメタに合わせた設定.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`


- NARS2S_new_finetuning_1回目
    - date: 20210711
    - output_folder_name: JSSS_2_JSUT_8
    - dataset: JSSS_2_JSUT_4
    - options
        - hifiganのパラメタに合わせた設定.
        - finetuningを始めるには少しだけ早いが, 寝るので...。
        - あと, 本命はmulti_speakerだしね.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml --restore_step 17500`


- makedataset for NARS2S
    - hifiganのconfigに合うように訓練してみる.
    - sr = 22050
    - n_fft = 1024
    - hop_size = 256
    - win_size = 1024
    - X→Yは同時. multi_speaker.

    - JSUT_to_from_JSSS
    - prevoice: JSUT_to_from_JSSS_2
    - preprocess: JSUT_to_from_JSSS_2

- NARS2S_new_1回目
    - date: 20210711
    - output_folder_name: JSUT_to_from_JSSS_2
    - dataset: JSUT_to_from_JSSS_2
    - options
        - hifiganのパラメタに合わせた設定.
        - multi_speaker = True
    
    - memo 
        - `python train.py -p ./config/JSUT_to_from_JSSS/preprocess.yaml -t ./config/JSUT_to_from_JSSS/train.yaml -m ./config/JSUT_to_from_JSSS/model.yaml`
    
        - lossはちょうどfinetuningとpretrainの中間くらい.
        - 一方で, lossが低いのに音質も悪いという謎の現象.
        - multiという難易度の高さに対してデータが少ないと判断し, jvsを追加することにした.

- makedataset for NARS2S
    - jsut_jsss_jvs
    - prevoice: jsut_jsss_jvs
    - preprocess: jsut_jsss_jvs

    - jvsを混ぜてみた. jvsは, 一話者につき4話者への変換をするようにデータを作成した.

    - その際に生じた問題がいくつか存在
        - まず, process_utterenceの際に, Noneを返されるようなデータがあり、そのせいでデータ数が合わない.
        - また, Noneでなくても, pitchとenergyとmelで時間方向に不一致も.
            - この場合もNoneを返すように改造.

- NARS2S_new_1回目
    - date: 20210713
    - output_folder_name: jsut_jsss_jvs
    - dataset: jsut_jsss_jvs
    - options
        - hifiganのパラメタに合わせた設定.
        - multi_speaker = True
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`
    
        - かなりlossが低い. 少なくとも一個前の二人のmultiより全然よい結果. すこし過学習を心配していたが, 問題なさそう.
        - 一方で, validationがtrainよりも低く, その傾向が続いている
            - 今度は, データに対してモデルの容量が小さそう.
            - モデルのパラメタを大きくしてみる.

- NARS2S_new_2回目
    - date: 20210714
    - output_folder_name: jsut_jsss_jvs_2
    - dataset: jsut_jsss_jvs
    - options
        - modelのパラメタを軒並み大きく。
        - hiddenは弄っていない. いじるとメモリに乗らない.
            - layer数だけ全て増やした感じ.
            - パラメタが1.5倍くらいになった.
        - batch_size = 4
            - こうじゃないと乗りません.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - めっちゃloss落ちる. 成功か???
        - とりあえずここまで訓練したモデルでfinetuningをN2Cでしてみる.


- make_dataset: N2C
    - melody+までの結果で一回やってみる.
    - jsut_jsss_jvsの重みからfinetuning.
    - なので, `python preprocess.py --config config/N2C/preprocess.yaml --finetuning`
    - finetuningとして, jsut_jsss_jvsのstatsを指定.
    - また, multi_speakerはoffにしてみる.

- N2C_finetuning_1回目
    - date: 20210715
    - output_folder_name: N2C
    - dataset: N2C
    - options
        - jsut_jsss_jvsのfinetuning
        - multi_speaker = False
        - batch_size = 8
        - これによって, 重みがembeddingのところは読み込めないので, hifiganで用いた読み込めるやつを利用する.
    
    - memo
        - `python train.py -p ./config/N2C/preprocess.yaml -t ./config/N2C/train.yaml -m ./config/N2C/model.yaml --restore_step 85000`

- inference一回目
    - だめだめ
    - traindataすらまともに動かず...。
        - 説1: energyとpitchとかが悪いから説.
        - 確かに, validationもteacherforcingで評価してるの正直良くないと思う.
        - validation時はtest同様のデータしか与えないこととする.
            - duration(+mask系)は上げないとlossが計算できないので仕方ないので上げる.
            - pitchとenergyだけはあげないということにしてみる.

        - おそらく, targetデータを頼らないような訓練もするべきな気がするが, ↓この実験次第ですね.

- NARS2S_new_3回目
    - date: 20210715
    - output_folder_name: jsut_jsss_jvs_3
    - dataset: jsut_jsss_jvs
    - options
        - validationの評価をinferenceと同じにした.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`
        
        - 予想通り? inferenceの結果通り? pitchとenergyをもらえないとmel系のvalは絶望的だった.
            - 考えられる理由は二つ
            - multi-speakerのせい
                - singleならちゃんとpitch, energyも行けるのかも.
                    - loss的にはsingleのものよりもmultiのが低くなってるけどね...。
                    - 音声変換において, lossと質は必ずしも比例しないので実験の価値はあり.
            - teacher_forcingのせい
                - これを頼り過ぎてしまったのかもしれないね.
                - ということで, これを行わない訓練をしてみる.

- NARS2S_new_4回目
    - date: 20210716
    - output_folder_name: jsut_jsss_jvs_4
    - dataset: jsut_jsss_jvs
    - options
        - validationの評価をinferenceと同じにした.
        - pitchとenergyのteacher_forcingをなしにしてみた.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - やはり, teacher_forcingをなしにしたことによって, validationを下げることには成功した.
        - 一方で, そもそも下がらなくなってしまった. せめて0.2前半は行きたい.

        - もしかしたら, 論文通りonetooneでないと難しいのかもしれないが, とりあえずもう一度チャレンジ
        - pitchとenergyをがっつり学習させに行く.
            - energyは5,5にして, pitchはgradient flowをTrueにしてみる↓
        
- NARS2S_new_5回目
    - date: 20210716
    - output_folder_name: jsut_jsss_jvs_5
    - dataset: jsut_jsss_jvs
    - options
        - validationの評価をinferenceと同じにした.
        - pitchとenergyのteacher_forcingをなしにしてみた.
        - そのうえでenergy, pitchを強化.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - 結果, 普通に4と同じ. むしろ過学習気味??
            - こうなってくると, あとできることはあと2つ
            - 途中まではteacher_forcingをTrueにして, 途中からFalseにして学習
            - ちゃんとmulti_speakerではなく, one-to-oneで学習
            - まずは, 前者であってほしいし前者を実験してみる.


- NARS2S_new_finetunig_1回目
    - date: 20210716
    - output_folder_name: jsut_jsss_jvs_6
    - dataset: jsut_jsss_jvs
    - options
        - jsut_jsss_jvsの85000からスタート.
        - teacher_forcingをFalseにしている. 果たしてうまく学習できるか.
        - optimもresetしておいた.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml --restore_step 85000`

        - jsut_jsss_jvs_4: 0からteacherforcingなしで学習したもの　と比べると、さすがにすぐにlossは追い抜かせたものの, 非常に緩やかな現象.
        - 一応はいまだ減少傾向にあるが、暫く続けないとダメそう&既に収束しそうなので, いったん保留. 他にやることがなくなったら学習を再開して確認してみたい.

        - それよりも, とりあえずは論文の再現, one-to-oneならちゃんとvalが低いのかを調べてみる.

- NARS2S_new_finetunig_1回目
    - date: 20210717
    - output_folder_name: jsut_jsss_jvs_6
    - dataset: jsut_jsss_jvs
    - options
        - 続き.
    
    - memo 
        - `python train.py -p ./output/log/jsut_jsss_jvs_6/preprocess.yaml -t ./output/log/jsut_jsss_jvs_6/train.yaml -m ./output/log/jsut_jsss_jvs_6/model.yaml --restore_step 97500`

        - jsut_jsss_jvs_4: 0からteacherforcingなしで学習したもの　と比べると、さすがにすぐにlossは追い抜かせたものの, 非常に緩やかな現象.
        - 一応はいまだ減少傾向にあるが、暫く続けないとダメそう&既に収束しそうなので, いったん保留. 他にやることがなくなったら学習を再開して確認してみたい.

        - それよりも, とりあえずは論文の再現, one-to-oneならちゃんとvalが低いのかを調べてみる.

- NARS2S_new_1回目
    - date: 20210717
    - output_folder_name: JSUT_2_JSSS_12
    - dataset: JSUT_JSSS_5
    - options
        - 評価方法を変えたやつ！！
        - hifiganのパラメタに合わせた設定.
        - vlaidationを, pitchとenergyを用いない評価方法で評価してみる.
        - ということで, 17500から再開してみた.
    
    - memo 
        - `python train.py -p ./output/log/JSUT_2_JSSS_12/preprocess.yaml -t ./output/log/JSUT_2_JSSS_12/train.yaml -m ./output/log/JSUT_2_JSSS_12/model.yaml --restore_step 17500`

        - 結果, だめでした. こいつも一切学習できていなかった...。
            - これはひとえに、pitchが悪いと思う.
            - 明らかに, pitchはうまく学習できていないし、逆にpitchさえちゃんとしていれば予測も何とかなるのはわかっているので。
            - そこで、pitchが悪い原因を考えてみると、やはりギザギザになっているのが良くないと思う。なので何とかして平坦化したい
            - それ、reduction factorの役目なのでは???
                - reduction factorを実装してみる.
                - 正直詳細はどこにも載っていなくてよくわからないが, 単純にレイヤー数を増減してしまえばよさそう?

- make_dataset
    - pre_voice: jsut_jsss_jvs
    - preprocessed_data: jsut_jsss_jvs_2

    - durationだけ, reduction factorを用いて計算しなおしたもの.
    - `python preprocess.py -p ./config/jsut_jsss_jvs/preprocess.yaml -m ./config/jsut_jsss_jvs/model.yaml`


- NARS2S_new_1回目
    - date: 20210717
    - output_folder_name: jsut_jsss_jvs_7
    - dataset: jsut_jsss_jvs_2
    - options
        - reduction_factorを初搭載.
        - teacher_forcing = True
        - batch_size = 32  # reduction factorすげぇ!
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - ただただ悪化してしまった....。
        - sliceはもったいないので, ちゃんと平均とった入力にしてもいいかもしれない.

- make_dataset
    - pre_voice: jsut_jsss_jvs
    - preprocessed_data: jsut_jsss_jvs_2

    - durationだけ, reduction factorのmeanを用いて再計算.
    - さすがに, 捨てるのはもったいないので, _oldとしてmeanでないものはとっておく
    - `python preprocess.py -p ./config/jsut_jsss_jvs/preprocess.yaml -m ./config/jsut_jsss_jvs/model.yaml`


- NARS2S_new_1回目
    - date: 20210717
    - output_folder_name: jsut_jsss_jvs_8
    - dataset: jsut_jsss_jvs_2
    - options
        - reduction_factor, meanにしてみる.
        - sliceモードも残しているけどね
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - meanにしても何の意味もなかった...。
        - ここで, ちゃんと元論文と元実装を見返す
        
        - Fastspeech2: pitchとenergyは, reductionしたやつ同士でlossをとっていた. melは元通りにしているが.
        - 元論文: pitch, energyに関してはそもそもreduction factorするとも書いていない. melに関してはするといっているが, lossはどうとるかも書いていない. targetにもreduction factorと言っているのが気になる...。
        - とりあえず, どちらに対してもreduction_factorしてみるか.

- NARS2S_new_1回目
    - date: 20210717
    - output_folder_name: jsut_jsss_jvs_9
    - dataset: jsut_jsss_jvs_2
    - options
        - lossの計算も全てreductionしたもので行ってみる. そのために, targetのもdatasetにてreductionをしてしまい, 図示の時だけ3倍に膨らませるという方針に.
        - melだけ特別扱いするのもよいと思う。とにかくそれよりもまずはpitchを落とす必要がある気はする.
            - jsut_jsss_jvs_8にて, teacher forcing Trueにて, melを特別扱い(melはreduction倍に増やしてからpostnetに入れる)を下にもかかわらず、下がらなかった.
            - これは, melだけはreductionをそもそもやらないほうが良い説がある
                - 一方で, それは元論文に反すること....うーん。
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - pitchが過去最高に学習できている. pitch, energyに関してもreductionするのは成功と言えそう.
        - 一方で, melは下がってくれないし、当然ながらtargetですら, ぶつぶつ音に聞こえる...。
            - 逆にmelだけreductionするべきではない気がする...。もしくは, せめてpost_netを通すか.

        - melぶつぶつ問題は, 元論文ではどうなっているんだろうか. targetにもreduction factorを使ったとあったけども...

        - 一番よさそうなのは, pitchとenergyだけreductionで, melは通常通りやることか??
        - とりあえずもう少し訓練させてから判断する.

        - 次の選択肢
            - melのみをreductionから外す
            - このままで, mel-postの過学習対策に, teacher-forcingをfalseにする.

            - このどちらもが元論文に反しているの、どうにかならないのか...。
            
            - あとは, pitchのgradient_flow = True にしてみたい.

        - reduction factorは, sourceのみに行わず、targetにも行うことで初めてうまく学習が進む
            - はじめ, sourceのmel, pitch, energyを1/3に圧縮し, modelから出るときに3倍にして出してloss計算していたのですが普通に精度が悪くなっただけでした.
              そこで, 先生からアドバイスいただいたとおりに、targetも圧縮してしまって圧縮した同士でloss計算すると, いままで学習できていなかったpitchもかなりlossが低下しました  
        - 一方で, mel は vocoderに入れる際に当然3倍に膨らませる必要があるのですが, それによってぶつぶつ音になってしまった
            - 元論文は、「target melにもreduction factor schemeを適用」し, 「vocoderをground truthのmelで訓練した」とあり、ここでのground truthはさすがに訓練に使っている, 1/3に圧縮し, 3倍に膨張させ戻したもののことを言ってるのでしょうか、そうでないとさすがにぶつぶつ音は直らないとおもうので....
        - また, mel lossに関して, validationは依然として下がらない
            - まだ学習途中なのでちゃんとしたことはある程度進んでからいうべきですが、ここまでのところを見るとすでに大分過学習しています...。確かに、pitchもlossが大分下がったとは言え、それでもまだ正解と見比べると差がすごいので、難しそうだなぁと思っています。なので、teacher forcing (学習時は正解のpitchを使ってmelを計算)をやめるべきかなとも考えていますが、こちらも論文にはちゃんと正解のpitchを使ったとあるのでなぞですね....

- NARS2S_new_1回目
    - date: 20210718
    - output_folder_name: jsut_jsss_jvs_10
    - dataset: jsut_jsss_jvs_2
    - options
        - jsut_jsss_jvs_9で, pitchが0.2より下がらないのと, mel-postが過学習してしまったので, とりあえずは 前者対策: gradient flow復活, 後者: teacher_forcing: false として, やってみる.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - 普通に想像していたよりもいい!
        - pitchもmelも下がる上に過学習していない. このまままわし続ければいい線行きそう?

        - いい線は行ったが, melがやはり不安. pitchも欲を言えばもう少し下がってほしい.
            - ちゃんとreductionを見たら, Tacotronでは, reshapeしてmel_numを増やしていた.
            - なので同じ方法でやってみる.
            - つまり, 以下のようにやる.
        
        - この方法だと, mel-targetに対してreduction factorがなにも影響しない
            - 元論文ではがっつりtargetにやると書いているので, 正直よくない...。
        - それでも, これは今までのいいとこどりを全てできている気がするので, 論文を無視するのも悪くないだろう.
        
```
input: (B, time, mel_num)
↓reshape
(B, time/3, mel_num*3)
↓linear
(B, time/3, 256)
↓encoder
(B, time/3, 256)
↓variance adaptor → pitch, energy, durationはtime/3で予測(正解は平均したものを利用)
(B, time/3, 256)
↓decoder
(B, time/3, 256)
↓linear
(B, time/3, mel_num*3)
↓reshape
(B, time, mel_num)
↓postnet
(B, time, mel_num)

↑
下の実験では変更した.
```

- NARS2S_new_1回目
    - date: 20210718
    - output_folder_name: jsut_jsss_jvs_11
    - dataset: jsut_jsss_jvs_2
    - options
        - reduction factor schemeを, ちゃんとTacotron準拠で行ってみた.
            - 説明
            - melはsource, targetともに, reshapeを行う.
                - 時間を第一次元にしてreshapeしないと狂うことに注意.
                - targetもreshapeする理由として, そうしないとt_mel_maskが本来の長さのtimeによって作成されてしまい,
                - それがpitchなどのvarianceに対してpadとして使われてしまうから.
                - 確かにその時だけ計算すればよいかもしれないが、
                - これで自然に論文通り「targetにもsame reduction scheme」することができたので, とても正しそう.
                - lossに関しても, targetは弄ってないので正しく学習はできるはず.
            
            - それ以外は, 今まで通り, reshape → meanという処理で単に時系列を1/3にしてしまう.

            - これらによって, 今までの課題である, 
                - 1/3にするとmelは難しくなるけどpitchは簡単になる...という矛盾をうまく回避できた. すごい!!!
        
        - ほかは, 今まで通り, pitchのgradient flow = True, teacher_forcing = False
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - うーんなんか微妙? スタートダッシュがあまりよろしくないので, ちゃんと論文のパラメタでやってみる.

- NARS2S_new_1回目
    - date: 20210718
    - output_folder_name: jsut_jsss_jvs_12
    - dataset: jsut_jsss_jvs_2
    - options
        - reduction factor schemeを, ちゃんとTacotron準拠で行ってみた.
        
        - pitchのgradient flow = False, teacher_forcing = True
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - 微妙...。
        - one-to-oneを一応試しておく...。

make_dataset
- durationだけ作り直し.
    - prevoice: JSUT_JSSS_5
    - preprocessed_data: JSUT_JSSS_6(JSUT_JSSS_5からコピー)
    - `python preprocess.py -p ./config/JSUT_JSSS/preprocess.yaml -m ./config/JSUT_JSSS/model.yaml`

- NARS2S_new_1回目
    - date: 20210718
    - output_folder_name: JSUT_2_JSSS_13
    - dataset: JSUT_JSSS_6
    - options
        - reduction factor schemeを, ちゃんとTacotron準拠で行ってみた.
        
        - pitchのgradient flow = False, teacher_forcing = True

        - multi_speaker = falseで, ほぼ論文通りのはず. さてどうなるか.
    
    - memo 
        - `python train.py -p ./config/JSUT_JSSS/preprocess.yaml -t ./config/JSUT_JSSS/train.yaml -m ./config/JSUT_JSSS/model.yaml`

        - 全然ダメダメ. 単なる悪化....。
        - 論文, 信用ならぬ......。
            - もしかしたら100kまわしたら下がりきるのかもしれんが...。初手で悪すぎる.
        
        - 唯一いい線を行っていた, ↓こいつの続きをやる.


- NARS2S_new_1回目
    - date: 20210718
    - output_folder_name: jsut_jsss_jvs_11
    - dataset: jsut_jsss_jvs_2
    - options
        - 続きをやる.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml --restore_step 2000`

        - 45kくらい学習させたが,　さすがに永遠にlossが下がるわけもなく.
            - まぁちょっとノイズが激しいくらい? になったので, 試しにhifiganに突っ込んで学習させてみて, どこまでいい音質で出せるかを聞いてみる.
            - それでよければ終了し, N2Cで訓練する.
                - これも, finetuningでやるか, 最初から混ぜてやるかは考えるべき.
                - 可能ならfinetuningでやりたい.
            - 悪ければ, モデルを見直す.


- make_mel_for_hifigan
    - `python inference.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml --restore_step 46000 --input_path ./preprocessed_data/jsut_jsss_jvs_2/source --output_path ./output/mel_for_hifi-gan/jsut_jsss_jvs --get_mel_for_hifigan --target_mel_path ./preprocessed_data/jsut_jsss_jvs_2/target/mel`

    - 推論だけしてmelを用意.

    - 注意: reduction_factorのせいで, target_melと, inference_melでは, 最後のいくつかのフレームがpadされてるされてない問題が生じてしまう...。
        - inference時は, その分削ってあげる必要がありそう...。
        - 元のtarget_melを読み込み, そのlenで切ればok.
        - 出力されたmelは一番長い長さでpaddingされているので, 単にassertでずれが3未満ならおｋとできない.
        - まぁもうここまで来たら信じるしかない.

- hifigan_finetuning_1回目
    - date: 20210719
    - output_folder_name: jsut_jsss_jvs
    - dataset: jsut_jsss_jvs
    - options
        - 推論結果のmelを使ってfinetuningしてみる.
        - optimはresetしてみる.
    
    - memo

        - finetuningやり方再掲
            - ↓のように, optionを指定する
                - mel_pathは, train, val.txtの入った一個上の階層をさすこと.
            - checkpoint_pathに, pretrainの重みを入れておくこと.
        

        - 謎に学習できない問題が発生したが, それはmax_wav_valueのせい.
        - librosaは最初から正規化してwavを出す(floatとして)ので割ると意味不明になる.
        - 一方, 元の実装ではscipyを利用していて, それを使うとintで読み込む.
            - なので, 割る必要があった. ただそれだけ.
            - vocoder inferでも, ちゃんと*max_wav_valueしたあと, int16に直している.
        
        - configは, ちゃんとuniversalと同じもので.
        - `python ./hifigan/train.py --input_mel_path ./output/mel_for_hifi-gan/jsut_jsss_jvs --input_wav_path ./pre_voice/jsut_jsss_jvs/target --checkpoint_path ./hifigan/output/jsut_jsss_jvs --config ./hifigan/config.json --fine_tuning --load_model_only`

- hifigan_finetuning_2回目
    - date: 20210719
    - output_folder_name: jsut_jsss_jvs_2
    - dataset: jsut_jsss_jvs_2
    - options
        - ちゃんとtargetから作られたきれいなmelでfinetuningしてみる.
        - これでできなかったらやばい.
    
    - memo

        - そのまま, preprocessed_dataのパスを指定することはできない. 先頭にmel-とかついてるし, フォルダ名もmelになっているので.
        - ~~めんどくさいが, renameしてコピーしたほうがよさそう.~~
            - 転置まで必要...。超めんどくさいね, 互換性なさすぎ.
        - `python ./hifigan/train.py --input_mel_path ./output/mel_for_hifi-gan/jsut_jsss_jvs_2 --input_wav_path ./pre_voice/jsut_jsss_jvs/target --checkpoint_path ./hifigan/output/jsut_jsss_jvs_2 --config ./hifigan/config.json --fine_tuning --load_model_only`

        - うーん, 下がりきらず, ノイズも取れず.
        - いっぽう, ボイチャコミュニティの人の結果を見ると, 250kもまわしてはいるけど, loss = 0.3でノイズのないきれいな結果になっている.
        - これは何か間違っている可能性がなくもない...lossが打ち止めになるのは変だよね
        
        - ここで, まず学習が進まない原因は?
            - 今回のものは, melの計算をhifigan製でないもので行っている.
                - 正直, mel計算に大きなロジックの差はないはずなので, 致命的な違いとは思っていない.
            - じゃあ, mel計算をhifigan製で行ったUniversalはなぜダメだった??
                - 今日発見した, max_value問題があった.
                - これは, wavの値を限りなく0に近づけるもの. melの計算はなぜかできるという特徴がある.
                - なので, これを修正した今はちゃんと学習できるのではないかという仮説.

            - ではなぜUniversalの重みによるpre_trainはダメだった?
                - 正直, ドメインが違うから, というのがありそう
                - ここまでわかった音声タスクの特徴として, lossが低い=クオリティが高いとは一概に言えないということ.
                - なので, loss自体はUniversalから始めたから低いが, ドメイン, ひいてはノイズの乗り方とかが違うため, クオリティは出なかったと思われる.

            - なので, 以上の仮説が正しければ, 
                - Universalの訓練がうまくいく
                - そして, lossは0.3程度でも, ノイズは少なくとも乗らないはず

            - これを確かめるために, 以前のUniversal_5の設定でそのままやり直してみる. 違うのは, max_valueのところ.


- Hifi-gan_12回目
    - date: 20210720
    - output_folder_name: Universal_6
    - dataset: Universal_3
    - options
        - max_wav_valueで割るのをやめたもの.
    
    - memo
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/Universal_3 --input_wav_path ./pre_voice/Universal_2 --checkpoint_path ./hifigan/output/Universal_6 --config ./hifigan/configs/config_Universal_2.json --checkpoint_interval 10000 --summary_interval 250 --validation_interval 2500`

        - ダメでした. まったく同じ.
        
        - max_value 問題は, trainの時はまったく問題なかった!!!!
            - なぜなら, normalizeをしていたから...まじか... targetの音が普通の時点で気づくべき。
        
        - ではなぜ学習がうまくいかないのか??
            - 仮説
            - 日本語と英語のデータセットは思っている以上にドメインが違う! 学習が困難!!
            - なので, 日本語だけならVCコミュニティの人のようにうまくいくはず!!



- Hifi-gan_13回目
    - date: 20210720
    - output_folder_name: Universal_7
    - dataset: jsut_jsss_jvs
    - options
        - 日本語のみのデータセットで学習させ空てみることにした。
    
    - memo
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/jsut_jsss_jvs/target --input_wav_path ./pre_voice/jsut_jsss_jvs/target --checkpoint_path ./hifigan/output/Universal_7 --config ./hifigan/config.json --checkpoint_interval 5000 --summary_interval 100 --validation_interval 1000`

        - うまくいった.
        - 完全に, 「fmax = null」が悪さをしていたと予想される.
            - ドメイン仮説は偽
                - LJSpeech単体の実験でも同じvalに落ち着いていたことを思い出したい.
            - 今回は, Universalと同じパラメタ, つまりfmax=8000で行っているので.
            - 他のstft系のパラメタも悪さをしていないとは言い切れないことに注意しよう.
        
        - これでもvalがすぐ打ち止めになった....下がりはしたけど。
        - ちょっと原因不明. ちゃんとLJSpeechで再現可能か見てみる.

- Hifi-gan_14回目
    - date: 20210720
    - output_folder_name: LJSpeech_2
    - dataset: LJSpeech
    - options
        - configはhifiganのuniversal. 再現実験ということですね.
    
    - memo
        - `python ./hifigan/train.py --input_mel_path ./preprocessed_data/LJSpeech --input_wav_path ./raw_data/LJSpeech --checkpoint_path ./hifigan/output/LJSpeech_2 --config ./hifigan/config.json --checkpoint_interval 5000 --summary_interval 100 --validation_interval 1000`


        - めっちゃうまくいった....。
        - 他にも無意識に勝手にいじっていたところを思い出した。
            - melを求めるところで, 実はreturn_complexを, future warningが出るからって勝手にfalseにしていた...あほすぎる.
            - おそらくここさえまた修正すれば問題なさそう.

        - 途中で止まった.
        - さらによく見たら, schedulerも弄ってしまっていた事に気づく....
        - それも直したらちゃんと再現できた. やったね.

        - 一応念のため, return_complex=Falseにして再度実験.
            - schedulerとどっちが致命的だったのかを調べるため.
        
        - 実験の結果, 無意味であることが判明. あとで挙動を調べよう.
            - なので, schedulerを弄ってしまっていたのが致命的だった...。

- makedataset
    - 懲りずに, 論文と同じパラメタでmelを作ってみる.
    - pre_voice: jsut_jsss_jvs_2
    - preprocessed_data: jsut_jsss_jvs_3

    - fmax: 8000. ここは悩んだが, 成功したときに, fmax: nullだとhifiganではどうしようもない.  ← nullはおそらく関係なかったかも...。
    - また, parallel wave ganも見ると, mel rangeは80-7600とあるので, おそらく8000としても問題ないはず.
    - さらに言えば, 今回の目的であるpitchには, 高周波成分はそこまで影響しなさそう. F0を予測するものなので.
    - `python preprocess.py -p ./config/jsut_jsss_jvs/preprocess.yaml -m ./config/jsut_jsss_jvs/model.yaml`


- NARS2S_new_1回目
    - date: 20210720
    - output_folder_name: jsut_jsss_jvs_13
    - dataset: jsut_jsss_jvs_3
    - options
        - パラメタを論文準拠で行ってみる. そもそもパラメタの影響が大きいと思ったのは, nullのせいだっけ. 勘違いだったけどね.
        - reduction_factorもちゃんとつけたので, 実験ということで.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - まさかのパラメタ無影響...まったく変わらず....パラメタ以外はjsut_jsss_jvs_12と同じなのでね.
        - 正直他にできることが...


- NARS2S_new_1回目
    - date: 20210720
    - output_folder_name: jsut_jsss_jvs_14
    - dataset: jsut_jsss_jvs_3
    - options
        - conv1dのところで, kernel=3とかにしてみた。微調整過ぎて...。
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml`

        - 当然こんなので変わるわけもなく. むしろ悪化してるんやが...

        - まだ一度も100kまでまわしていなかった気がする. なので, ちょっと信じてまわしてみることにする.

        - 一応, 100kレベルのスケールで見れば, pitchは全然収束していない.
        - teacher_forcingのvalも, pitchがいい感じになってきたら減っていくと信じる.

- NARS2S_new_1回目
    - date: 20210720
    - output_folder_name: jsut_jsss_jvs_12(続き)
    - dataset: jsut_jsss_jvs_2
    - options
        - reduction factor schemeを, ちゃんとTacotron準拠で行ってみた.
        
        - pitchのgradient flow = False, teacher_forcing = True

        - つまり, ほぼ論文通りのパラメタ. valが高いのを我慢して, 100k, やってみる.

        - pitchがうまく学習できれば, valも落ちてくれると信じる. それまで耐える.
            - 100k行ってみよう.
    
    - memo 
        - `python train.py -p ./config/jsut_jsss_jvs/preprocess.yaml -t ./config/jsut_jsss_jvs/train.yaml -m ./config/jsut_jsss_jvs/model.yaml --restore_step 3000`



- todo
    - hifiganのdataset周り変更
        - 現状, makedataset含め, melを作る前提になっているが, trainの時は, sr揃えるのだけやればよい.
        - なのに, train時もtrain, val.txtがmel_wav_pathにあること前提になってしまっている.
            - それはfinetuning時のみ覗く設定にする.
            - train時はtrain.txt, val.txtをmake_datasetで作れるように.

    - return_complexの挙動確認


# 処理系を勝手にいじくるな! まずは論文の再現をちゃんとしてから考察を始めよ！
    - よくわかってないくせに, load_wavをlibrosaのものに変えて, それなのに37000で割り算して超微小なaudioにしているの, あほすぎる.

    - 酷い. schedulerの位置も勝手に変えている...そんなことしちゃダメでしょ....
    - return_complexも勝手に変えてるし...