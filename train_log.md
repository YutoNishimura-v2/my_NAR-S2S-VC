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