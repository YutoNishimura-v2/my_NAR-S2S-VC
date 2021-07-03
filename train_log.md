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