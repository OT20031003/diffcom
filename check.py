import torch
import yaml
import os
import sys

# DeepJSCC & DiffCom modules
# パスが通っていない場合は適宜修正してください
from _djscc.network import ADJSCC
from _pdjscc.net.channel import Channel 
from utils.util import Config
from data.datasets import get_test_loader 

def check_scale():
    # 設定 (環境に合わせて変更してください)
    config_path = "./configs/diffcom.yaml"
    ckpt_path = "./_djscc/ckpt/ADJSCC_C=2.pth.tar"
    data_path = "./testsets/ffhq_demo_100" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checking scale on {device}...")

    # Config loading
    with open(config_path, 'r') as f:
        cfg = Config(yaml.safe_load(f))

    # 【修正】channel属性がない場合に初期化する
    if not hasattr(cfg, 'channel') or cfg.channel is None:
        cfg.channel = {}
    
    cfg.channel['type'] = 'awgn'
    cfg.channel['chan_param'] = 10 # SNR 10dB
    
    # logger属性のダミー作成
    if not hasattr(cfg, 'logger'):
        cfg.logger = None
    
    # モデル構築
    latent_channels = cfg.djscc.get('channel_num', 16)
    channel_module = Channel(cfg)
    model = ADJSCC(C=latent_channels, channel=channel_module, device=device)

    # 重みロード (キー名の修正付き)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('Encoder.'): new_k = k.replace('Encoder.', 'jscc_encoder.')
            elif k.startswith('Decoder.'): new_k = k.replace('Decoder.', 'jscc_decoder.')
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model.to(device)
    model.eval()

    # データローダー
    try:
        loader = get_test_loader(data_path, batch_size=16, shuffle=False)
        images = next(iter(loader))[0].to(device)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # スケール確認
    with torch.no_grad():
        # 1. 生のエンコーダ出力 (現在の学習コードが使っている値)
        # DeepJSCCの仕様では、ここは正規化されていない可能性があります
        z_raw = model.encode(images, given_SNR=10.0)
        
        # 2. パワー正規化後の出力 (テスト時/通信時の実際の値)
        # Channelクラス内部の complex_normalize を手動で呼び出して再現
        # 平均パワー(pwr)が 1.0 になるようにスケーリングされます
        z_normalized, pwr = channel_module.complex_normalize(z_raw, power=1)

        print("\n" + "=" * 60)
        print(" SCALE CHECK RESULT")
        print("=" * 60)
        
        raw_mean = z_raw.mean().item()
        raw_std = z_raw.std().item()
        raw_max = z_raw.abs().max().item()
        raw_pwr = (z_raw ** 2).mean().item() * 2 # 複素数換算のパワー概算

        norm_mean = z_normalized.mean().item()
        norm_std = z_normalized.std().item()
        norm_max = z_normalized.abs().max().item()
        norm_pwr_val = pwr.mean().item()

        print(f"{'Metric':<15} | {'Raw (Train Input)':<20} | {'Normalized (Test Input)':<20}")
        print("-" * 60)
        print(f"{'Mean':<15} | {raw_mean:.6f}             | {norm_mean:.6f}")
        print(f"{'Std':<15}  | {raw_std:.6f}              | {norm_std:.6f}")
        print(f"{'Max Abs':<15} | {raw_max:.6f}              | {norm_max:.6f}")
        print(f"{'Avg Power':<15} | {raw_pwr:.6f} (Approx)      | {norm_pwr_val:.6f} (Should be 1.0)")
        print("-" * 60)

        if abs(norm_pwr_val - 1.0) > 0.1:
            print("!! WARNING: Normalized power is not 1.0. Check channel code. !!")
        
        # 判定
        ratio = raw_std / (norm_std + 1e-9)
        if abs(ratio - 1.0) > 0.1:
            print(f"\n[CONCLUSION] Scale mismatch detected!")
            print(f"Training data is scaling by approx x{ratio:.2f} compared to Test data.")
            print("Action: You MUST normalize z in the training loop.")
        else:
            print("\n[CONCLUSION] Scale matches. Normalization might be happening inside encode() or is unnecessary.")

if __name__ == "__main__":
    check_scale()