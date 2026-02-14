import argparse
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd

# DeepJSCC & DiffCom modules
from _djscc.network import ADJSCC
from _pdjscc.net.channel import Channel 
from data.datasets import get_test_loader 
from guided_diffusion import script_util
from utils.util import Config

# --- 評価指標用ライブラリ ---
try:
    import lpips
except ImportError:
    lpips = None
try:
    from DISTS_pytorch import DISTS
except ImportError:
    DISTS = None
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    InceptionResnetV1 = None

# -----------------------------------------------------------------------------
# Evaluator Class
# -----------------------------------------------------------------------------
class Evaluator:
    def __init__(self, device):
        self.device = device
        self.mse_fn = nn.MSELoss().to(device)
        
        if lpips is not None:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        if DISTS is not None:
            self.dists_fn = DISTS().to(device).eval()
        if InceptionResnetV1 is not None:
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    def calculate_metrics(self, img_pred, img_target):
        results = {}
        mse = self.mse_fn(img_pred, img_target).item()
        results['MSE'] = mse
        results['PSNR'] = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        
        img_pred_norm = img_pred * 2 - 1
        img_target_norm = img_target * 2 - 1
        
        if hasattr(self, 'lpips_fn'):
            with torch.no_grad():
                results['LPIPS'] = self.lpips_fn(img_pred_norm, img_target_norm).mean().item()
        else: results['LPIPS'] = float('nan')

        if hasattr(self, 'dists_fn'):
            with torch.no_grad():
                results['DISTS'] = self.dists_fn(img_pred, img_target).mean().item()
        else: results['DISTS'] = float('nan')

        if hasattr(self, 'facenet'):
            img_pred_face = nn.functional.interpolate(img_pred, size=(160, 160), mode='bilinear')
            img_target_face = nn.functional.interpolate(img_target, size=(160, 160), mode='bilinear')
            img_pred_face = (img_pred_face - 0.5) / 0.5
            img_target_face = (img_target_face - 0.5) / 0.5
            with torch.no_grad():
                emb_pred = self.facenet(img_pred_face)
                emb_target = self.facenet(img_target_face)
                results['IDLoss'] = (1 - nn.functional.cosine_similarity(emb_pred, emb_target)).mean().item()
        else: results['IDLoss'] = float('nan')

        return results

# -----------------------------------------------------------------------------
# Custom Sampling Function
# -----------------------------------------------------------------------------
def sample_range(diffusion, model, shape, z_start, start_step, end_step, model_kwargs):
    device = next(model.parameters()).device
    batch_size = shape[0]

    if hasattr(diffusion, 'timestep_map'):
        ts_map = diffusion.timestep_map
    else:
        ts_map = list(range(diffusion.num_timesteps))

    indices = list(range(diffusion.num_timesteps))[::-1]
    active_indices = []
    for i in indices:
        real_t = ts_map[i]
        if end_step <= real_t <= start_step:
            active_indices.append(i)
    
    if not active_indices:
        return z_start

    max_step = ts_map[-1]
    if start_step >= max_step:
        img = torch.randn(*shape, device=device)
    else:
        img = z_start

    for i in active_indices:
        t = torch.tensor([i] * batch_size, device=device)
        with torch.no_grad():
            out = diffusion.p_sample(
                model,
                img,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs
            )
            img = out["sample"]
    return img

# -----------------------------------------------------------------------------
# Main Test Function
# -----------------------------------------------------------------------------
def test(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.diffcom_config, 'r') as f:
        diffcom_cfg = Config(yaml.safe_load(f))
    diffcom_cfg.CUDA = torch.cuda.is_available()

    if not hasattr(diffcom_cfg, 'channel') or diffcom_cfg.channel is None:
        diffcom_cfg.channel = {}
    diffcom_cfg.channel['type'] = args.channel_type
    diffcom_cfg.channel['chan_param'] = args.test_snr 
    if not hasattr(diffcom_cfg, 'logger'): diffcom_cfg.logger = None
    
    latent_channels = diffcom_cfg.djscc.get('channel_num', 16)
    print(f"Latent Channels: {latent_channels}, Test SNR: {args.test_snr}dB")
    print(f"★ Scale Factor: {args.scale_factor} (Applying rescaling to match training distribution)")

    evaluator = Evaluator(device)

    # DeepJSCC
    channel_module = Channel(diffcom_cfg) 
    djscc_model = ADJSCC(C=latent_channels, channel=channel_module, device=device)
    
    if os.path.exists(args.djscc_ckpt):
        checkpoint = torch.load(args.djscc_ckpt, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('Encoder.'): new_k = k.replace('Encoder.', 'jscc_encoder.')
            elif k.startswith('Decoder.'): new_k = k.replace('Decoder.', 'jscc_decoder.')
            new_state_dict[new_k] = v
        djscc_model.load_state_dict(new_state_dict)
    else: raise FileNotFoundError(f"DeepJSCC checkpoint not found.")
    djscc_model.to(device).eval()

    # Diffusion
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
        dropout=0.0,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing=args.timestep_respacing,
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
    model.input_blocks[0][0] = nn.Conv2d(latent_channels, old_conv_in.out_channels, kernel_size=3, padding=1)
    old_conv_out = model.out[2]
    model.out[2] = nn.Conv2d(old_conv_out.in_channels, latent_channels * 2, kernel_size=3, padding=1)
    
    if os.path.exists(args.diffusion_ckpt):
        ckpt = torch.load(args.diffusion_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else: raise FileNotFoundError(f"Diffusion checkpoint not found.")
    model.to(device).eval()

    # Inference
    dataloader = get_test_loader(args.data_path, batch_size=args.batch_size, shuffle=False)
    
    print(f"Start Inference: Range [{args.start_step} -> {args.end_step}]")
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= args.num_batches: break
            
            images = batch[0].to(device) 
            batch_size = images.shape[0]

            # 1. DeepJSCC Transmission (Baseline)
            z_clean = djscc_model.encode(images, given_SNR=float(args.test_snr))
            z_noisy = channel_module(z_clean) # Normalized signal
            rec_jscc = djscc_model.decode(z_noisy, given_SNR=float(args.test_snr))

            # --- Evaluate DeepJSCC ---
            metrics_jscc = evaluator.calculate_metrics(rec_jscc, images)
            metrics_jscc = {f"JSCC_{k}": v for k, v in metrics_jscc.items()}

            # 2. Rescale & Diffusion Sampling (Proposed)
            model_kwargs = {"y": torch.full((batch_size,), int(args.test_snr), device=device, dtype=torch.long)}
            
            # 【重要】スケール補正: テスト入力(小) → 学習時スケール(大) に拡大
            z_input_scaled = z_noisy * args.scale_factor
            
            z_sampled_scaled = sample_range(
                diffusion, 
                model, 
                (batch_size, latent_channels, image_size, image_size),
                z_start=z_input_scaled,         
                start_step=args.start_step, 
                end_step=args.end_step, 
                model_kwargs=model_kwargs
            )
            
            # 【重要】スケール戻し: 学習時スケール(大) → テスト入力(小) に縮小
            z_sampled = z_sampled_scaled / args.scale_factor
            
            rec_diff = djscc_model.decode(z_sampled, given_SNR=float(args.test_snr))
            
            # --- Evaluate Proposed ---
            metrics_diff = evaluator.calculate_metrics(rec_diff, images)
            metrics_diff = {f"Diff_{k}": v for k, v in metrics_diff.items()}

            all_metrics.append({**metrics_jscc, **metrics_diff})

            comparison = torch.cat([images, rec_jscc, rec_diff], dim=0)
            save_path = os.path.join(args.output_dir, f"result_snr{args.test_snr}_{i}.png")
            save_image(comparison, save_path, nrow=batch_size, normalize=True)

    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        mean_metrics = df.mean()
        
        print("\n" + "="*60)
        print(f" Comparison (SNR={args.test_snr}dB, Step={args.start_step}, Scale=x{args.scale_factor})")
        print("="*60)
        
        columns = ['MSE', 'PSNR', 'LPIPS', 'DISTS', 'IDLoss']
        print(f"{'Metric':<10} | {'DeepJSCC':<15} | {'Proposed (Diff)':<15}")
        print("-" * 46)
        for col in columns:
            jscc_val = mean_metrics.get(f"JSCC_{col}", float('nan'))
            diff_val = mean_metrics.get(f"Diff_{col}", float('nan'))
            print(f"{col:<10} | {jscc_val:.6f}        | {diff_val:.6f}")
        print("="*60)
        
        csv_path = os.path.join(args.output_dir, f"metrics_snr{args.test_snr}.csv")
        df.to_csv(csv_path, index=False)
        
        summary_path = os.path.join(os.path.dirname(args.output_dir), "summary_metrics.csv")
        summary_data = {'SNR': args.test_snr, 'Start': args.start_step, 'Scale': args.scale_factor}
        summary_data.update(mean_metrics.to_dict())
        
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            df_new = pd.DataFrame([summary_data])
            df_summary = pd.concat([df_summary, df_new], ignore_index=True)
        else:
            df_summary = pd.DataFrame([summary_data])
        df_summary.to_csv(summary_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--djscc_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--diffcom_config", type=str, default="./configs/diffcom.yaml")
    parser.add_argument("--channel_type", type=str, default="awgn")
    parser.add_argument("--test_snr", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./results/test_images")
    parser.add_argument("--timestep_respacing", type=str, default="100")
    parser.add_argument("--start_step", type=int, default=1000)
    parser.add_argument("--end_step", type=int, default=0)
    
    # 【追加】スケール補正係数 (デフォルトは前回の診断値)
    parser.add_argument("--scale_factor", type=float, default=8.21, help="Rescale input z to match training distribution")
    
    args = parser.parse_args()
    test(args)