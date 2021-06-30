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