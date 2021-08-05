# NARS2S 再現実装
- todo
  - 環境構築方法をrequirementsに変更.
## 環境構築(要修正)
**注意**  
現在, windowsのみで利用できるconda env fileを用意しています.  
非wiondowsの方は, お手数ですが手動で`NARS2S.yaml`にあるライブラリをconda, pip installしてください.

## train
### 事前準備
- 入力音声データ配置
  - sampling rateなどなどはいじる必要ありません.
  - 以下の構造になるように配置してください.
  - 音声データに関して, 以下の制約があります.
    - source, targetに配置する音声は, その名の通り, source: 変換元音声, target: 変換後音声(教師データ)になります.
    - **ペアになっている音声のファイル名は同一にしてください.**
    - multi-speakerを利用したい場合
      - 音声ファイル名の最初に, 以下の文字を追加してください.
      - `{source_speaker_name}_{target_speaker_name}_`
      - 先述した通り, これを加えたうえでsource, targetでファイル名をそろえてください.

```
MY_NAR-S2S-VC
|- raw_data
    |- (dataset名): 任意
        |- source
        |   |- *.wav
        |- target
            |- *.wav
```

- configの編集
  - 自由にいじってください.
  - 何に対応しているか, などはconfigに直接書き込んでいます.


### 手順
1. `python preprocess.py -p {preprocess.yamlへのpath} -m {model.yamlへのpath}`

2. `python train.py -p {preprocess.yamlへのpath} -t {train.yamlへのpath} -m {model.yamlへのpath}`

### 絡んでくるコード解説
実行される順に解説していきます.
1. preprocess.py
   - 実行する前処理をまとめたものです.
   1. wav_path_matching
       - raw_dataにあるsourceとtargetの音声のペアのファイル名が同一になっているかをチェックします. 

   2. voice_preprocess
      1. change_sr
         - sampling_rateを変更します.
      2. delete_novoice
         - 音声ファイルの先頭・末尾に存在する無音部分を削除します.
  
   3. voice_deviding(head_tail_onlyをFalseにした場合)
       - 指定した秒数以上無音が続いているところで音声を分割します.
  
   4. preprocessor.build_from_path
      - melなどの計算をしていきます.
      1. mel, energy, pitchを計算
         - その際に, energy, pitchはlogをとってpitchに関して連続pitchにします(trueなら)
      2. それらに関して正規化
      3. train, val.txtを作成
         - 単なるtrain-val_splitです.
      4. speakers.txtを作成(multi_speakerがtrueなら)
         - 登場するspeaker一覧を作成します.
  
   5. get_duration
       - durationの計算をします.
       1. wavから, mel_dim=20のmelを作成します.
       2. reductionします.
          - ここでは, 指定フレーム毎に纏め, meanをとって時間軸をreduction_factor分圧縮しています.
       3. fastdtwによってアライメントします.
          - その際に, 無音区間においてはなるべくsource, targetの時間軸をx,y軸としたときに対角に対応するように, durationを設定します.

2. train.py
   - ここは一般的なpytorchコードと同様なので, 省略します.


## finetuning

## inference
