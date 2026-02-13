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

    # Channel設定の注入
    if not hasattr(diffcom_cfg, 'channel') or diffcom_cfg.channel is None:
        diffcom_cfg.channel = {}
    
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
    
    # Load Pretrained Weights (with Fix for Key Mismatch)
    if os.path.exists(args.djscc_ckpt):
        checkpoint = torch.load(args.djscc_ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # -----------------------------------------------------------
        # ★ここが修正ポイント: キー名の不一致を解消する
        # Encoder. -> jscc_encoder.
        # Decoder. -> jscc_decoder.
        # -----------------------------------------------------------
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('Encoder.'):
                new_k = k.replace('Encoder.', 'jscc_encoder.')
            elif k.startswith('Decoder.'):
                new_k = k.replace('Decoder.', 'jscc_decoder.')
            new_state_dict[new_k] = v
        state_dict = new_state_dict
        # -----------------------------------------------------------

        djscc_model.load_state_dict(state_dict)
        print(f"Loaded DeepJSCC checkpoint from {args.djscc_ckpt} (Keys corrected)")
    else:
        raise FileNotFoundError(f"DeepJSCC checkpoint not found: {args.djscc_ckpt}")

    djscc_model.to(device)
    djscc_model.eval()
    for param in djscc_model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------------------------
    # 3. Create Diffusion Model for Latent Space
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
    
    old_conv_in = model.input_blocks[0][0]
    model.input_blocks[0][0] = nn.Conv2d(
        latent_channels, 
        old_conv_in.out_channels, 
        kernel_size=3, padding=1
    )
    
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
            
            # SNR Randomization (0-20)
            snr_int = torch.randint(0, 21, (1,)).item()
            
            with torch.no_grad():
                z = djscc_model.encode(images, given_SNR=float(snr_int))
            
            t = torch.randint(0, diffusion.num_timesteps, (current_batch_size,), device=device)
            
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

            if step % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"model_{step}.pt"))

    print("Training Finished.")
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pt"))

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
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--learn_sigma", action='store_true', default=True, help="Whether to learn sigma (variance)")
    
    args = parser.parse_args()
    train(args)