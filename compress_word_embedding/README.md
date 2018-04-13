# 深層コード学習による単語分散表現の圧縮

* 本モデルは，各種分散表現の圧縮表現を学習するスクリプトである
* 正確には，以下のような(denoising)Auto-Encoder Modelを学習する
	* 入力：分散表現 $v_x \in R^{N_d}$
	* 中間層：圧縮表現 $\[1,N_k\]^{N_m}$
		* より正確には one-hot vector のsoftmax近似
	* 出力層：復元された分散表現 $v_x' \in R^{N_d}$
* 本モデルの詳細は，以下の文献を参照のこと

```
SHU, Raphael; NAKAYAMA, Hideki. Compressing Word Embeddings via Deep Compositional Code Learning. arXiv preprint arXiv:1711.01068, 2017.
朱中元, 中山英樹. 深層コード学習による単語分散表現の圧縮. In: 言語処理学会第24回年次大会(NLP2018), 2018
```

## 実行環境
* スクリプトの開発およびテストは Python 2.7.14 を使用した
* 必要なpackageについては  `./requirement.txt`  を参照のこと
* 入力データは [NumPy](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.save.html) `.npy` 形式に対応している

## 事前準備
* KerasおよびTensorFlowをインストール，動作確認を行うこと

## 設定
* 設定は 1)実行時引数として定義 2)スクリプト内部で定義 の2種類に大別される

### 実行時引数として定義
* `--input_embedding`：入力する分散表現．例：`gensim.models.Word2Vec` の `syn0.npy` ファイル
* `--output_dir`：学習済みモデル(keras.model形式)の保存先ディレクトリ．ファイル名は入力から自動決定
* `--N_k`：圧縮表現・コードワードベクトルの本数
* `--N_m`：圧縮表現・コードブックの数
* `--N_epoch`：エポック数

### スクリプト内部で定義
* `F_temperature`：Gumbel-Softmax Trick適用時のアニーリング温度
* `N_minibatch`：ミニバッチサイズ
* `optimizer_name`：最適化手法

## 実行例

```
compress_word_embedding.py \
--input_embedding PATH_TO_THE_NPY_FILE
--output_dir ../trained_model/
```

## 学習済みモデルの利用
* 学習済みモデルは 1)圧縮表現への変換 2)分散表現の復元 の2つに用いる
* 利用例については，同梱のJupyter Notebookを参照されたい

### 圧縮表現への変換
* 以下の手順に従う
	1. 圧縮表現を求めたい分散表現をひとつ用意する
	2. モデルに入力して，中間層の値を出力させる
		* 中間層のkeras.layer.nameは `gumbel_softmax` である．$N_m$個の$N_k$次元ベクトルが得られる
	3. 中間層の値を離散値に変換する．これが圧縮表現である
		* $m \in \[1,N_m\]$ ごとにargmaxを取る

### 分散表現の復元
* 以下の手順に従う
	1. モデルから，Decoder層のパラメータを抽出する
		* Decoder層のkeras.layer.nameは `decoder_[0-(N_m-1)]` である．$N_m$個の$(N_k, N_d)$次元行列が得られる
	2. 復元したい圧縮表現をひとつ用意する
		* 例：`[2,3,8,5,4,1,0,11]`
	3. Decoder層のパラメータからそれぞれ$k$行目を取得して足し合わせる．これが分散表現である
		* 例：`decoder_0:2行目 + decoder_1:3行目 + ...`

以上
