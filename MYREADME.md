# DiffComをGoogle Colab で動かす手順

## マウント方法
```
from google.colab import drive
import os

# Driveのマウント
drive.mount('/content/drive')

%cd /content/drive/MyDrive/diffcom
```


## 必要なライブラリのインストール
```
!pip install timm lpips DISTS_pytorch pytorch_msssim pyiqa pyyaml numpy matplotlib scipy compressai
```

## エラー修正 
### DISTSの重みファイルコピー
```
!cp _pdjscc/loss_utils/perceptual_similarity/dists_loss/weights.pt /usr/weights.pt
```
