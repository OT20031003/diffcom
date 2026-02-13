import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss(log_file, output_img):
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    # CSV読み込み
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 移動平均を計算してトレンドを見やすくする (Windowサイズ=100)
    df['loss_smooth'] = df['loss'].rolling(window=100).mean()

    plt.figure(figsize=(12, 6))
    
    # 生のLoss（薄く表示）
    plt.plot(df['step'], df['loss'], alpha=0.2, color='gray', label='Raw Loss')
    
    # 移動平均（濃く表示）
    plt.plot(df['step'], df['loss_smooth'], color='blue', label='Smoothed Loss (MA=100)')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Latent Diffusion Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_img)
    print(f"Loss plot saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="results/latent_diffusion_ckpt/loss_log.csv")
    parser.add_argument("--output_img", type=str, default="loss_curve.png")
    args = parser.parse_args()
    
    plot_loss(args.log_file, args.output_img)