import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os

# Ensure the parent directory (project root) is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.bench_vs_dense import CONFIGS, QUICK_CONFIGS, bench_config

def main():
    # Automatically select standard or high-end configs based on CUDA availability
    configs_to_run = QUICK_CONFIGS if not torch.cuda.is_available() else CONFIGS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    seq_lens = []
    speedups = []
    valid_names = []

    print(f"Running benchmarks on {device.upper()} for visualization...")
    for cfg in configs_to_run:
        name, B, H, S, D, lw, st, gb = cfg
        
        # Run the benchmark
        result = bench_config(name, B, H, S, D, lw, st, gb, device, dtype)
        
        # Capture valid results
        if result["sparse_ms"] > 0 and result["dense_ms"] > 0:
            speedup = result["dense_ms"] / result["sparse_ms"]
            seq_lens.append(S)
            speedups.append(speedup)
            valid_names.append(name)

    if not speedups:
        print("No valid benchmark results to plot.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    x_labels = [f"{s:,}\n({n})" for s, n in zip(seq_lens, valid_names)]
    sns.barplot(x=x_labels, y=speedups, hue=x_labels, palette="viridis", legend=False)
    plt.axhline(1.0, color='red', linestyle='--', label="Dense Baseline (1.0x)")

    plt.title(f"Sparse Attention Speedup vs Dense SDPA ({device.upper()})", fontsize=14, pad=15)
    plt.xlabel("Sequence Length & Config", fontsize=12)
    plt.ylabel("Speedup Multiplier (x)", fontsize=12)
    plt.legend()

    # Annotate bars
    for i, v in enumerate(speedups):
        plt.text(i, v + 0.05, f"{v:.2f}x", ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = "speedup_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved successfully to '{output_path}'.")
    print(f"You can view the image by double-clicking it in the Colab file explorer.")

if __name__ == "__main__":
    main()
