import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np

# DeepJSCC & DiffCom modules
from _djscc.network import ADJSCC
from _pdjscc.net.channel import Channel 
from data.datasets import get_test_loader 
from guided_diffusion import script_util
from utils.util import Config

def train(args):
    # -------------------------------------------------------------------------
    # 1. Setup & Config
    # -------------------------------------------------------------------------
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DeepJSCC Config
    with open(args.diffcom_config, 'r') as f:
        diffcom_cfg = Config(yaml.safe_load(f))
    
    # DeepJSCC Channel Dimension (C)
    latent_channels = diffcom_cfg.djscc.get('channel_num', 16)
    print(f"DeepJSCC Latent Channels (C): {latent_channels}")

    # -------------------------------------------------------------------------
    # 2. Prepare DeepJSCC (Frozen Encoder)
    # -------------------------------------------------------------------------
    # 通信路モデルの初期化 (ADJSCCの構築に必要)
    channel_module = Channel(args.channel_type, config=diffcom_cfg) 
    
    djscc_model = ADJSCC(C=latent_channels, channel=channel_module, device=device)
    
    # Load Pretrained Weights
    if os.path.exists(args.djscc_ckpt):
        checkpoint = torch.load(args.djscc_ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        djscc_model.load_state_dict(state_dict)
        print(f"Loaded DeepJSCC checkpoint from {args.djscc_ckpt}")
    else:
        raise FileNotFoundError(f"DeepJSCC checkpoint not found: {args.djscc_ckpt}")

    djscc_model.to(device)
    djscc_model.eval()
    for param in djscc_model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # 3. Create Diffusion Model for Latent Space
    # -------------------------------------------------------------------------
    # DeepJSCC (Stride=4) なので 256x256 -> 64x64
    image_size = 64 
    
    # ★ポイント1: class_cond=True にして SNR を条件として受け取れるようにする
    model, diffusion = script_util.create_model_and_diffusion(
        image_size=image_size,
        class_cond=True,           # SNR条件付けを有効化
        learn_sigma=True,          # 分散も学習
        num_channels=128,          # モデルサイズ (適宜調整)
        num_res_blocks=2,
        channel_mult="",
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.1,               # 過学習防止
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    
    # ★ポイント2: 入力/出力チャンネル数を DeepJSCC の次元 (C) に合わせて修正
    # script_util.py を書き換えずに層を差し替える安全な方法
    
    # 入力層の修正 (3 -> C)
    old_conv_in = model.input_blocks[0][0]
    model.input_blocks[0][0] = nn.Conv2d(
        latent_channels, 
        old_conv_in.out_channels, 
        kernel_size=3, padding=1
    )
    
    # 出力層の修正 (3 or 6 -> C or C*2)
    old_conv_out = model.out[2]
    out_ch = latent_channels * 2 if args.learn_sigma else latent_channels
    model.out[2] = nn.Conv2d(
        old_conv_out.in_channels, 
        out_ch, 
        kernel_size=3, padding=1
    )
    
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -------------------------------------------------------------------------
    # 4. Dataset
    # -------------------------------------------------------------------------
    dataloader = get_test_loader(args.data_path, batch_size=args.batch_size, shuffle=True)

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print(f"Start Training Latent Diffusion (Conditional on SNR)...")
    print(f"  - Latent Shape: {image_size}x{image_size}, Channels: {latent_channels}")
    print(f"  - SNR Range: 0dB - 20dB (Randomized)")

    step = 0
    save_dir = os.path.join("results", "latent_diffusion_ckpt")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch[0].to(device)
            current_batch_size = images.shape[0]
            
            # -------------------------------------------------------
            # A. SNRのランダム選択
            # -------------------------------------------------------
            # 0dB ~ 20dB の整数値からランダムに選択
            # DeepJSCCの実装上、バッチ内で同一のSNRを使う必要がある場合が多いため
            # ここではバッチごとに1つのSNRを選んで適用する
            snr_int = torch.randint(0, 21, (1,)).item() # 0, 1, ..., 20
            
            # -------------------------------------------------------
            # B. DeepJSCCでエンコード (潜在変数 z を取得)
            # -------------------------------------------------------
            with torch.no_grad():
                # DeepJSCCエンコーダは given_SNR に応じて特徴マップを変える
                # 正規化: 出力zは概ね平均0分散1だが、拡散モデル用に微調整してもよい (ここではそのまま)
                z = djscc_model.encode(images, given_SNR=float(snr_int))
            
            # -------------------------------------------------------
            # C. 拡散モデルの学習
            # -------------------------------------------------------
            t = torch.randint(0, diffusion.num_timesteps, (current_batch_size,), device=device)
            
            # SNRを条件として渡す (class labelとして扱う)
            # モデル内部で Embedding される
            model_kwargs = {
                "y": torch.full((current_batch_size,), snr_int, device=device, dtype=torch.long)
            }

            losses = diffusion.training_losses(model, z, t, model_kwargs=model_kwargs)
            loss = losses["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "SNR": f"{snr_int}dB"})

            # 定期保存
            if step % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"model_{step}.pt"))

    print("Training Finished.")
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training images (e.g., ./testsets/ffhq_train_70k)")
    parser.add_argument("--djscc_ckpt", type=str, required=True, help="Path to DeepJSCC checkpoint")
    parser.add_argument("--diffcom_config", type=str, default="./configs/diffcom.yaml", help="Config file path")
    parser.add_argument("--channel_type", type=str, default="awgn", help="awgn or fading")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--learn_sigma", action='store_true', default=True, help="Whether to learn sigma (variance)")
    
    args = parser.parse_args()
    train(args)