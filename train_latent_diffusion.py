import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
import csv

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

    # Config Loading
    with open(args.diffcom_config, 'r') as f:
        diffcom_cfg = Config(yaml.safe_load(f))
    diffcom_cfg.CUDA = torch.cuda.is_available() # CUDAフラグの補完

    if not hasattr(diffcom_cfg, 'channel') or diffcom_cfg.channel is None:
        diffcom_cfg.channel = {}
    
    # チャンネル設定 (学習時はSNRを動的に変えるためダミー値で初期化)
    diffcom_cfg.channel['type'] = args.channel_type
    diffcom_cfg.channel['chan_param'] = 10 
    
    if not hasattr(diffcom_cfg, 'logger'):
        diffcom_cfg.logger = None
    
    latent_channels = diffcom_cfg.djscc.get('channel_num', 16)
    print(f"DeepJSCC Latent Channels (C): {latent_channels}")

    # -------------------------------------------------------------------------
    # 2. Prepare DeepJSCC (Frozen Encoder)
    # -------------------------------------------------------------------------
    channel_module = Channel(diffcom_cfg) 
    djscc_model = ADJSCC(C=latent_channels, channel=channel_module, device=device)
    
    # Load Pretrained Weights
    if os.path.exists(args.djscc_ckpt):
        checkpoint = torch.load(args.djscc_ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Key Correction
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('Encoder.'):
                new_k = k.replace('Encoder.', 'jscc_encoder.')
            elif k.startswith('Decoder.'):
                new_k = k.replace('Decoder.', 'jscc_decoder.')
            new_state_dict[new_k] = v
        
        djscc_model.load_state_dict(new_state_dict)
        print(f"Loaded DeepJSCC checkpoint from {args.djscc_ckpt} (Keys corrected)")
    else:
        raise FileNotFoundError(f"DeepJSCC checkpoint not found: {args.djscc_ckpt}")

    djscc_model.to(device)
    djscc_model.eval()
    for param in djscc_model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # 3. Create Diffusion Model
    # -------------------------------------------------------------------------
    image_size = 64 
    model, diffusion = script_util.create_model_and_diffusion(
        image_size=image_size,
        class_cond=True,
        learn_sigma=True,
        num_channels=128,
        num_res_blocks=2,
        channel_mult="",
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.1,
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
    
    # Layer Replacement
    old_conv_in = model.input_blocks[0][0]
    model.input_blocks[0][0] = nn.Conv2d(latent_channels, old_conv_in.out_channels, kernel_size=3, padding=1)
    
    old_conv_out = model.out[2]
    out_ch = latent_channels * 2 if args.learn_sigma else latent_channels
    model.out[2] = nn.Conv2d(old_conv_out.in_channels, out_ch, kernel_size=3, padding=1)
    
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -------------------------------------------------------------------------
    # 4. Resume & Logger Setup
    # -------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    # 保存先を新しいフォルダに変更することを推奨
    save_dir = os.path.join("results", "latent_diffusion_normalized_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    
    log_file_path = os.path.join(save_dir, "loss_log.csv")
    
    # Resume Logic
    if args.resume_ckpt:
        if os.path.isfile(args.resume_ckpt):
            print(f"Resuming from checkpoint: {args.resume_ckpt}")
            checkpoint = torch.load(args.resume_ckpt, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'step' in checkpoint:
                global_step = checkpoint['step']
            
            print(f"Resumed at Epoch {start_epoch}, Step {global_step}")
            log_file = open(log_file_path, 'a', newline='')
            csv_writer = csv.writer(log_file)
        else:
            print(f"Checkpoint file not found: {args.resume_ckpt}. Starting from scratch.")
            log_file = open(log_file_path, 'w', newline='')
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(['step', 'epoch', 'loss', 'snr'])
    else:
        print("Starting training from scratch.")
        log_file = open(log_file_path, 'w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['step', 'epoch', 'loss', 'snr'])

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    dataloader = get_test_loader(args.data_path, batch_size=args.batch_size, shuffle=True)

    print(f"Start Training Latent Diffusion (Correctly Normalized)...")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                images = batch[0].to(device)
                current_batch_size = images.shape[0]
                
                # SNR Randomization
                snr_int = torch.randint(0, 21, (1,)).item()
                
                with torch.no_grad():
                    # 1. DeepJSCC Encode
                    z_raw = djscc_model.encode(images, given_SNR=float(snr_int))
                    
                    # 2. ★★★ Scale Normalization (CRITICAL FIX) ★★★
                    # Channelクラスと同じロジックでパワーを1.0に正規化する
                    # これにより、テスト時と同じスケール分布になる
                    z, _ = channel_module.complex_normalize(z_raw, power=1)
                
                t = torch.randint(0, diffusion.num_timesteps, (current_batch_size,), device=device)
                
                model_kwargs = {
                    "y": torch.full((current_batch_size,), snr_int, device=device, dtype=torch.long)
                }

                losses = diffusion.training_losses(model, z, t, model_kwargs=model_kwargs)
                loss = losses["loss"].mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "SNR": f"{snr_int}dB"})

                csv_writer.writerow([global_step, epoch, loss.item(), snr_int])
                log_file.flush()
                if global_step % args.save_interval == 0:
                    save_path = os.path.join(save_dir, f"checkpoint_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                    log_file.flush()

        save_path = os.path.join(save_dir, "checkpoint_final.pt")
        torch.save({
            'epoch': args.epochs - 1,
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print("Training Finished.")

    finally:
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training images")
    parser.add_argument("--djscc_ckpt", type=str, required=True, help="Path to DeepJSCC checkpoint")
    parser.add_argument("--diffcom_config", type=str, default="./configs/diffcom.yaml", help="Config file path")
    parser.add_argument("--channel_type", type=str, default="awgn", help="awgn or fading")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=2200) # 約1epochごとに保存
    parser.add_argument("--learn_sigma", action='store_true', default=True, help="Whether to learn sigma")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)