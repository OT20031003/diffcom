

# DiffCom-HARQ: Generative Communication with Latent Diffusion & Uncertainty-Aware Retransmission

このプロジェクトは、拡散モデルを用いた画像通信フレームワーク **DiffCom** を拡張し、実用的な無線通信環境における信頼性を向上させるためのシステムです。

主な拡張機能として、**DeepJSCCの潜在空間（Latent Space）で動作する条件付き拡散モデル**と、生成画像の不確実性に基づいた**部分再送制御（Spatially-Aligned Partial HARQ）**を実装しています。

## 🚀 主な機能 (Key Features)

1. **DiffCom (Pixel-Space Posterior Sampling)**:
* 受信信号をガイド（条件）として、拡散モデルにより高画質な画像を復元します。


2. **Latent Diffusion for Signal Denoising**:
* DeepJSCCのエンコーダ出力（シンボル）を学習した拡散モデル。
* **SNR条件付け (SNR Conditioning)** により、チャネル状態に応じて変化するDeepJSCCの信号分布に追従し、適切なデノイジングを行います。


3. **Uncertainty-Aware HARQ**:
* 拡散モデルの「幻覚（Hallucination）」と「通信エラー」を区別する高度な再送制御。
* **MSE-Difference Metric**: 決定論的デコーダ（DeepJSCC）と確率的デコーダ（DiffCom）の出力差分を利用し、ノイズによる破綻箇所のみをピンポイントで再送要求します。



---

## 📂 ディレクトリ構成

```text
diffcom/
├── _djscc/                # DeepJSCC (JSCC Encoder/Decoder)
│   └── ckpt/              # 学習済みモデル (ADJSCC_C=2.pth.tar 等)
├── channel/               # 通信路モデル (AWGN, Fading)
├── configs/               # 設定ファイル (diffcom.yaml)
├── data/                  # データローダー
├── guided_diffusion/      # 拡散モデルのコア実装 (UNet, GaussianDiffusion)
├── testsets/              # データセット
│   └── ffhq_train_70k/    # FFHQ 256x256 (70,000枚)
├── train_latent_diffusion_v2.py  # ★新規: 潜在空間拡散モデルの訓練スクリプト
├── main_diffcom.py        # 推論・HARQシミュレーション実行スクリプト
└── README.md              # 本ファイル

```

---

## 🛠️ セットアップ (Setup)

### 1. 環境構築

必要なライブラリをインストールします。

```bash
pip install torch torchvision numpy tqdm pyyaml matplotlib scipy
# その他、timm, lpips など (MYREADME.txt 参照)

```

### 2. データセットの準備 (FFHQ)

FFHQデータセット（256x256リサイズ版）をダウンロードし、以下の配置にします。

```text
diffcom/testsets/ffhq_train_70k/
├── 00000.png
├── 00001.png
...
└── 69999.png

```

### 3. 設定ファイルの確認

`configs/diffcom.yaml` でデータセットパスやDeepJSCCのモデル設定を確認してください。

```yaml
testset_name: ffhq_train_70k
operator_name: 'djscc'
djscc:
  channel_num: 2  # 使用するDeepJSCCの次元数 (C)
  jscc_model_path: '_djscc/ckpt/ADJSCC_C=2.pth.tar'

```

---

## 🏋️‍♂️ 訓練 (Training Latent Diffusion)

DeepJSCCの潜在空間分布を学習する拡散モデルを訓練します。
このモデルは、**SNR（0dB〜20dB）を条件として受け取り**、DeepJSCCのエンコーダ出力  の分布を正確に模倣します。

```bash
python train_latent_diffusion.py \
  --data_path ./testsets/ffhq_train_70k \
  --djscc_ckpt ./_djscc/ckpt/ADJSCC_C=2.pth.tar \
  --diffcom_config ./configs/diffcom.yaml \
  --batch_size 32 \
  --epochs 100 \
  --save_interval 10000 \
  --gpu_id 0

```
```
nohup python train_latent_diffusion.py \
  --data_path ./testsets/ffhq_train_70k \
  --djscc_ckpt ./_djscc/ckpt/ADJSCC_C=2.pth.tar \
  --diffcom_config ./configs/diffcom.yaml \
  --batch_size 32 \
  --epochs 100 \
  --gpu_id 0 \
  > training.log 2>&1 &
```
* **重要**: `train_latent_diffusion.py` は、学習ループ内でランダムなSNRを生成し、`class_cond=True` を利用して拡散モデルに注入します。

---

## 🧪 推論とHARQ (Inference with HARQ)

学習済みモデルを用いて、通信・復元・再送のシミュレーションを行います。

### HARQの動作原理

従来の「不確実性（分散）」だけではテクスチャの複雑さをノイズと誤認する問題がありました。本システムでは以下のロジックを採用しています。

1. **初回送信**: DeepJSCCで送信し、受信信号  を得る。
2. **不整合検知**:
* **MSE復元**: ノイズを除去・平滑化した画像  を生成。
* **DiffCom復元**: 受信信号をガイドに生成した画像 。
* **差分計算**: 。この値が大きい領域は、DiffComがノイズをテクスチャと誤認して「幻覚」を見ている可能性が高い領域です。


3. **部分再送**:  となる領域のみマスクして再送要求 (NACK) を送ります。

### 実行コマンド

```bash
python main_diffcom.py --opt ./configs/diffcom.yaml

```

(※ `main_diffcom.py` 内で `tau_harq=True` および上記のMSE差分ロジックが有効化されている必要があります)

---

## 📝 技術的背景 (Methodology)

### なぜ潜在空間で訓練するのか？ (Latent Space Denoising)

* **計算効率**: 画素空間（256x256）に比べ、潜在空間（64x64）は計算コストが大幅に低く、高速なデノイジングが可能です。
* **HARQへの応用**: 潜在空間での拡散モデルは、シンボルの尤度を直接評価できるため、より純粋な「通信路ノイズによる不確実性」を検知するのに適しています。

### なぜSNR条件付けが必要なのか？ (SNR Conditioning)

* DeepJSCC (ADJSCC) は、入力SNRに応じて特徴マップのスケールや分布を動的に変化させます（AFModule）。
* SNRを固定して拡散モデルを学習させると、テスト時のSNR変動に対応できず性能が劣化します。
* 本実装では、SNRをクラス条件として拡散モデルに入力することで、あらゆるチャネル環境に適応可能なデノイザーを実現しています。

---

## 📚 参考文献

* DiffCom: Channel Received Signal Is a Natural Condition to Guide Diffusion Posterior Sampling
* DeepJSCC: Deep Joint Source-Channel Coding for Wireless Image Transmission
* Guided Diffusion (OpenAI)