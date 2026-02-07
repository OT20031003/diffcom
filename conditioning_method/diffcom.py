import numpy as np
import torch
import torch.nn as nn

from utils import utils_model

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](**kwargs)


class ConsistencyLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        zeta = config.diffcom_series[config.conditioning_method]['zeta']
        gamma = config.diffcom_series[config.conditioning_method]['gamma']
        self.weight = {
            'x_mse': gamma,
            'ofdm_sig': zeta,
        }

    def forward(self, measurement, x_0_hat, cof, operator, operation_mode):
        x_0_hat = (x_0_hat / 2 + 0.5)  # .clip(0, 1)
        s = operator.encode(x_0_hat)
        if operation_mode == 'latent':
            recon_measurement = {
                'ofdm_sig': operator.forward(s, cof)
            }
        elif operation_mode == 'pixel':
            recon_measurement = {
                'x_mse': x_0_hat
            }
        elif operation_mode == 'joint':
            ofdm_sig = operator.forward(s, cof)
            s_hat = operator.transpose(ofdm_sig, cof)
            x_confirming = operator.decode(s_hat)
            recon_measurement = {
                'ofdm_sig': ofdm_sig,
                'x_mse': x_confirming
            }
        loss = {}
        for key in recon_measurement.keys():
            loss[key] = self.weight[key] * torch.linalg.norm(measurement[key] - recon_measurement[key])
        return loss


def get_lr(config, t, T):
    lr_base = config['learning_rate']
    # exponential decay to 0
    if config['lr_schedule'] == 'exp':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * np.exp(-t / T)
    # linear decay
    elif config['lr_schedule'] == 'linear':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * (t / T)
    # constant
    else:
        lr = lr_base
    return lr


@register_conditioning_method(name='diffcom')
class DiffCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'latent'

    def estimate_uncertainty(self, x_0_hat, t_step, ns, unet, num_perturbations=5):
        """
        De VitaらのAlgorithm 1に基づく不確実性推定 (Pixel-wise Aleatoric Uncertainty Estimation)
        
        Args:
            x_0_hat: 現在のステップで予測された原画像 (Predicted x_0)
            t_step: 現在のタイムステップ t
            ns: NoiseSchedule オブジェクト
            unet: 拡散モデルのUNet
            num_perturbations (M): 摂動サンプルの数 (Algorithm 1における M)
        
        Returns:
            uncertainty_map: 画素ごとの分散 (Variance)
        """
        # --- (b) Re-noising: 摂動サンプルの生成 ---
        # x_0_hat を現在のステップ t のノイズレベルまで再拡散させる
        # x_t^i = sqrt(alpha_bar_t) * x_0_hat + sqrt(1 - alpha_bar_t) * epsilon_i
        
        B, C, H, W = x_0_hat.shape
        M = num_perturbations
        
        # バッチサイズ方向に M 倍に拡張して並列計算
        x_0_repeated = x_0_hat.repeat_interleave(M, dim=0)
        t_repeated = torch.tensor([t_step] * (B * M), device=x_0_hat.device)
        
        # ノイズ係数の取得
        sqrt_alpha_bar = ns.sqrt_alphas_cumprod[t_step]
        sqrt_one_minus_alpha_bar = ns.sqrt_1m_alphas_cumprod[t_step]
        
        # ランダムノイズの生成 (epsilon_i ~ N(0, I))
        epsilon_i = torch.randn_like(x_0_repeated)
        
        # 摂動入力 x_t^i の作成
        x_t_perturbed = sqrt_alpha_bar * x_0_repeated + sqrt_one_minus_alpha_bar * epsilon_i
        
        # --- (c) Variance Calculation: 分散の計算 ---
        # モデル推論を実行してスコア (epsilon_theta) を取得
        
        with torch.no_grad():
             model_out = unet(x_t_perturbed, t_repeated)
             # model_out は通常 epsilon (または x_start) の予測値。
             # 論文では "variance of the denoising scores" とあるため、出力の分散を取る。
             # 出力が [B*M, C*2, ...] (learned variance) の場合は前半の平均のみを使用
             if model_out.shape[1] == C * 2:
                 model_out, _ = torch.split(model_out, C, dim=1)

        # M個のサンプルごとにグループ化して分散を計算
        # shape: [B, M, C, H, W]
        scores_grouped = model_out.view(B, M, C, H, W)
        
        # Variance calculation (Equation 8 in De Vita et al.)
        # Var(X) = E[|X - E[X]|^2]
        uncertainty_map = torch.var(scores_grouped, dim=1, unbiased=True) # [B, C, H, W]
        
        # チャンネル方向の平均を取ることで画素ごとの不確実性とする (任意だが一般的)
        uncertainty_map = torch.mean(uncertainty_map, dim=1, keepdim=True) # [B, 1, H, W]
        
        return uncertainty_map

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        
        # --- (a) Estimation of x_0 (Algorithm 1) ---
        # 勾配計算用に x_0_hat を取得 (既存ロジック)
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        # --- Uncertainty Estimation (Modified) ---
        # 計算頻度を制御するロジックを追加
        uncertainty_map = None
        if hasattr(config, 'calc_uncertainty') and config.calc_uncertainty:
             # main_diffcom.py で計算された interval を取得 (デフォルトは1)
             interval = getattr(config, 'uncertainty_interval', 1)
             
             # 現在のステップ i が interval で割り切れる場合のみ計算
             if i % interval == 0:
                 # M (摂動数) は config から取得、デフォルトは 5
                 M = getattr(config, 'uncertainty_perturbations', 5)
                 uncertainty_map = self.estimate_uncertainty(x_0_hat.detach(), t_step, ns, unet, num_perturbations=M)

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss, uncertainty_map
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_t, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad = torch.autograd.grad(outputs=total_loss, inputs=x_t)[0]
            learning_rate = get_lr(config.diffcom_series[config.conditioning_method], t_step,
                                   ns.t_start - 1)
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss, uncertainty_map


@register_conditioning_method(name='hifi_diffcom')
class HiFiDiffCom(DiffCom):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'joint'


@register_conditioning_method(name='blind_diffcom')
class BlindDiffCom(DiffCom):
    def __init__(self):
        super().__init__()

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        assert (config.conditioning_method == 'blind_diffcom')

        h_t = h_t.requires_grad_()
        h_score = - h_t / (power ** 2)
        h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
                h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
        h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
                            ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))
        
        # --- Uncertainty Estimation (Modified for Blind Mode) ---
        uncertainty_map = None
        if hasattr(config, 'calc_uncertainty') and config.calc_uncertainty:
             interval = getattr(config, 'uncertainty_interval', 1)
             
             if i % interval == 0:
                 M = getattr(config, 'uncertainty_perturbations', 5)
                 uncertainty_map = self.estimate_uncertainty(x_0_hat.detach(), t_step, ns, unet, num_perturbations=M)

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss, uncertainty_map
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad, h_t_grad = torch.autograd.grad(outputs=total_loss, inputs=[x_t, h_t])
            learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
            learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            lr_h = config.diffcom_series['blind_diffcom']['h_lr']
            lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
            h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
            h_t_minus_1 = h_t_minus_1.detach_()
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss, uncertainty_map