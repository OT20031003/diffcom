import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss(log_file, output_img, target_snr=None, window=100):
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    # CSV読み込み
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # カラム名の空白除去（念のため）
    df.columns = [c.strip() for c in df.columns]

    # SNRフィルタリングモード
    if target_snr is not None:
        if 'snr' not in df.columns:
            print("Error: 'snr' column not found in CSV. Cannot filter by SNR.")
            return
        
        # 指定されたSNRの行のみ抽出
        original_count = len(df)
        df = df[df['snr'] == target_snr]
        filtered_count = len(df)
        
        if filtered_count == 0:
            print(f"No data found for SNR={target_snr}")
            return
        
        print(f"Plotting mode: SNR={target_snr}dB (Found {filtered_count}/{original_count} steps)")
        plot_title = f'Latent Diffusion Training Loss (SNR={target_snr}dB)'
    else:
        print(f"Plotting mode: All Data")
        plot_title = 'Latent Diffusion Training Loss (All SNRs)'

    # 移動平均を計算 (min_periods=1にすることでデータ不足時のNaNを防ぐ)
    df['loss_smooth'] = df['loss'].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    
    # 生のLoss（薄く表示）
    plt.plot(df['step'], df['loss'], alpha=0.2, color='gray', label='Raw Loss')
    
    # 移動平均（濃く表示）
    plt.plot(df['step'], df['loss_smooth'], color='blue', label=f'Smoothed Loss (MA={window})')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_img)
    print(f"Loss plot saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="results/latent_diffusion_normalized_ckpt/loss_log.csv")
    parser.add_argument("--output_img", type=str, default="loss_curve.png")
    
    # 新機能: 特定のSNRを指定 (例: --target_snr 10)
    parser.add_argument("--target_snr", type=int, default=None, help="Specific SNR to plot (e.g. 10). If not set, plots all data.")
    
    # 新機能: 移動平均のウィンドウサイズ変更 (SNR指定時はデータが疎になるため小さくすると良い)
    parser.add_argument("--window", type=int, default=100, help="Window size for moving average.")
    
    args = parser.parse_args()
    
    # SNR指定時は出力ファイル名にSNRを含めるように自動調整しても便利ですが、
    # 今回はユーザー指定の output_img を優先します。
    
    plot_loss(args.log_file, args.output_img, args.target_snr, args.window)