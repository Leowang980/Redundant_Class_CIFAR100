"""
Visualize training loss and accuracy for all 4 experiments (100, 101, 105, 110 classes).
Reads history_*.json from logs/ and generates plots.
Computes last 20 epochs avg ± std for each setting.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR / "plots"
LAST_N = 20


def load_histories():
    histories = {}
    for n in [100, 101, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]:
        p = LOGS_DIR / f"history_{n}.json"
        if p.exists():
            with open(p) as f:
                histories[n] = json.load(f)
        else:
            print(f"Warning: {p} not found, skipping.")
    return histories


def compute_last_n_stats(histories):
    """Compute avg ± std over last LAST_N epochs for each setting."""
    stats = {}
    for n, h in sorted(histories.items()):
        last = {k: np.array(v[-LAST_N:]) for k, v in h.items()}
        stats[n] = {
            "train_loss": (last["train_loss"].mean(), last["train_loss"].std()),
            "test_loss": (last["test_loss"].mean(), last["test_loss"].std()),
            "train_acc": (last["train_acc"].mean(), last["train_acc"].std()),
            "test_acc": (last["test_acc"].mean(), last["test_acc"].std()),
        }
    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = load_histories()
    if not histories:
        print("No history files found. Run run_all.py first.")
        return

    # Compute last 20 epochs avg ± std
    stats = compute_last_n_stats(histories)
    lines = [f"Last {LAST_N} epochs: avg ± std", "=" * 50]
    for n in sorted(stats):
        s = stats[n]
        lines.append(f"num_classes={n}:")
        lines.append(f"  train_loss: {s['train_loss'][0]:.4f} ± {s['train_loss'][1]:.4f}")
        lines.append(f"  test_loss:  {s['test_loss'][0]:.4f} ± {s['test_loss'][1]:.4f}")
        lines.append(f"  train_acc:  {s['train_acc'][0]:.2f} ± {s['train_acc'][1]:.2f}%")
        lines.append(f"  test_acc:   {s['test_acc'][0]:.2f} ± {s['test_acc'][1]:.2f}%")
        lines.append("")
    summary_text = "\n".join(lines)
    print(summary_text)
    with open(OUTPUT_DIR / "last20_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved {OUTPUT_DIR / 'last20_summary.txt'}")

    # Plot Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for n, h in sorted(histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label=f"Train (n={n})", alpha=0.8)
        ax.plot(epochs, h["test_loss"], "--", label=f"Test (n={n})", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Test Loss (Redundant Class Experiment)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'loss.png'}")

    # Plot Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for n, h in sorted(histories.items()):
        epochs = range(1, len(h["train_acc"]) + 1)
        ax.plot(epochs, h["train_acc"], label=f"Train (n={n})", alpha=0.8)
        ax.plot(epochs, h["test_acc"], "--", label=f"Test (n={n})", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Test Accuracy (Redundant Class Experiment)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'accuracy.png'}")

    # Combined: 2x2 subplots (Train Loss, Test Loss, Train Acc, Test Acc)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for n, h in sorted(histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0, 0].plot(epochs, h["train_loss"], label=f"n={n}")
        axes[0, 1].plot(epochs, h["test_loss"], label=f"n={n}")
        axes[1, 0].plot(epochs, h["train_acc"], label=f"n={n}")
        axes[1, 1].plot(epochs, h["test_acc"], label=f"n={n}")
    axes[0, 0].set_title("Train Loss")
    axes[0, 1].set_title("Test Loss")
    axes[1, 0].set_title("Train Accuracy (%)")
    axes[1, 1].set_title("Test Accuracy (%)")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Redundant Class Experiment: CIFAR-100 (100 vs 101 vs 105 vs 110 classes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "combined.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'combined.png'}")


if __name__ == "__main__":
    main()
