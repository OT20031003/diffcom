# DiffComをGoogle Colab で動かす手順

## 毎回実行 sshの設定など
```
# ==========================================
# Google Colab 起動時用 初期化スクリプト
# ==========================================
from google.colab import drive
import os
import shutil

# 1. Google Driveのマウント
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# --- 設定項目（ここを確認してください） ---
GIT_USERNAME = "OT20031003"           # GitHubに表示される名前（適宜変更してください）
GIT_EMAIL = "ono1003takuma@gmail.com" # あなたのメールアドレス
KEY_FILENAME = "id_ed25520"           # 作成した鍵の名前
PROJECT_DIR = "/content/drive/MyDrive/diffcom" # 作業ディレクトリ
DRIVE_KEY_PATH = f"/content/drive/MyDrive/.ssh_keys/{KEY_FILENAME}"
# ----------------------------------------

# 2. Gitのユーザー設定（毎回リセットされるため再設定）
!git config --global user.name "{GIT_USERNAME}"
!git config --global user.email "{GIT_EMAIL}"

# 3. SSH環境の構築
ssh_dir = "/root/.ssh"
local_key_path = os.path.join(ssh_dir, KEY_FILENAME)
config_path = os.path.join(ssh_dir, "config")

# .sshディレクトリ作成
if not os.path.exists(ssh_dir):
    os.makedirs(ssh_dir)
    os.chmod(ssh_dir, 0o700)

# 鍵のコピーと権限設定
if os.path.exists(DRIVE_KEY_PATH):
    shutil.copy(DRIVE_KEY_PATH, local_key_path)
    os.chmod(local_key_path, 0o600) # 権限を厳しく設定（必須）
    print(f"✅ SSH鍵 ({KEY_FILENAME}) をセットアップしました。")
else:
    print(f"❌ エラー: Driveに鍵が見つかりません: {DRIVE_KEY_PATH}")

# 4. SSH Configファイルの作成
# (標準外の名前 id_ed25520 を使うために必須の設定)
ssh_config = f"""
Host github.com
    HostName github.com
    User git
    IdentityFile {local_key_path}
    StrictHostKeyChecking no
"""
with open(config_path, "w") as f:
    f.write(ssh_config)

# 5. known_hosts の更新（初回接続時の警告回避）
!ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null

# 6. 接続テストとディレクトリ移動
print("-" * 20)
print("接続テスト中...")
!ssh -T git@github.com

print("-" * 20)
if os.path.exists(PROJECT_DIR):
    %cd {PROJECT_DIR}
    print(f"📂 作業ディレクトリに移動しました: {PROJECT_DIR}")
else:
    print(f"⚠️ ディレクトリが見つかりません: {PROJECT_DIR}")
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
