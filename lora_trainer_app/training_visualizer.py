"""
Training visualization utilities for LoRA fine-tuning.
Provides real-time progress tracking and post-training plots.
"""

import json
import time
from typing import Dict, List

from transformers import TrainerCallback

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.use("Agg")  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MetricsCallback(TrainerCallback):
    """Callback to display training metrics in real-time and save them."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history = {
            "loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": [],
            "eval_loss": [],
            "eval_steps": [],
        }
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print("\n" + "=" * 80)
        print("🚀 Training Started")
        print("=" * 80)
        print(
            f"📊 Total steps: {state.max_steps} | "
            f"Epochs: {args.num_train_epochs} | "
            f"Batch size: {args.per_device_train_batch_size}"
        )
        print("=" * 80 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens."""
        if logs is None:
            return

        current_time = time.time()

        # Extract metrics and convert to Python scalars (not tensors!)
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch")
        step = state.global_step

        # Convert tensors to scalars to avoid memory leak
        def to_scalar(value):
            """Convert tensor or any value to Python scalar."""
            if value is None:
                return None
            if hasattr(value, 'item'):
                return value.item()  # PyTorch tensor
            return float(value) if isinstance(value, (int, float)) else value

        # Save to history (ensure scalars, not tensors)
        if loss is not None:
            self.metrics_history["loss"].append(to_scalar(loss))
            self.metrics_history["learning_rate"].append(to_scalar(lr))
            self.metrics_history["epoch"].append(to_scalar(epoch))
            self.metrics_history["step"].append(int(step))

        # Check for eval metrics
        if "eval_loss" in logs:
            self.metrics_history["eval_loss"].append(to_scalar(logs["eval_loss"]))
            self.metrics_history["eval_steps"].append(int(step))
            
            # CRITICAL: aggressive cleanup after evaluation to prevent slowdown
            import torch
            import gc
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            gc.collect()

        # Calculate speed
        elapsed = current_time - self.start_time
        self.last_log_time = current_time

        # Clean GPU cache at every log step to prevent memory accumulation
        # This is critical to prevent memory leaks during training
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Display progress bar
        if loss is not None:
            total_steps = state.max_steps
            progress = step / total_steps * 100
            bar_length = 40
            filled = int(bar_length * step / total_steps)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Calculate ETA
            steps_remaining = total_steps - step
            if step > 0:
                avg_step_time = elapsed / step
                eta_seconds = avg_step_time * steps_remaining
                eta_min = int(eta_seconds / 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"{eta_min}m {eta_sec}s"
            else:
                eta_str = "calculating..."

            print(
                f"\r[{bar}] {progress:.1f}% | "
                f"Step {step}/{total_steps} | "
                f"Epoch {epoch:.2f} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"⏱️  {int(elapsed)}s | "
                f"ETA: {eta_str}",
                end="",
                flush=True,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None:
            return

        print("\n" + "-" * 80)
        print("📊 Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:.<40} {value:.4f}")
            else:
                print(f"  {key:.<40} {value}")
        print("-" * 80 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print("=" * 80 + "\n")

        # Save metrics to file
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_file = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"💾 Metrics saved to: {metrics_file}\n")

    def get_metrics(self):
        """Return the collected metrics."""
        return self.metrics_history


def generate_training_plots(metrics_history: Dict[str, List], output_dir: str):
    """
    Generate training visualization plots.

    Args:
        metrics_history: Dictionary of training metrics
        output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available - skipping plot generation")
        print("   Install with: pip install matplotlib")
        return

    if not metrics_history["loss"]:
        print("⚠️  No training data - skipping plot generation")
        return

    print("\n📊 Generating training plots...")

    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("LoRA Training Metrics", fontsize=18, fontweight="bold", y=0.995)

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(metrics_history["step"], metrics_history["loss"], "b-", linewidth=2.5, label="Training Loss", alpha=0.8)
    if metrics_history["eval_loss"]:
        ax1.plot(
            metrics_history["eval_steps"],
            metrics_history["eval_loss"],
            "ro-",
            linewidth=2.5,
            markersize=8,
            label="Eval Loss",
            alpha=0.9,
        )
    ax1.set_xlabel("Step", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax1.set_title("Training and Evaluation Loss", fontsize=15, fontweight="bold", pad=15)
    ax1.legend(fontsize=11, loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#f8f9fa")

    # Plot 2: Learning Rate Schedule
    ax2 = axes[0, 1]
    ax2.plot(metrics_history["step"], metrics_history["learning_rate"], "g-", linewidth=2.5, alpha=0.8)
    ax2.set_xlabel("Step", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Learning Rate", fontsize=13, fontweight="bold")
    ax2.set_title("Learning Rate Schedule", fontsize=15, fontweight="bold", pad=15)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax2.set_facecolor("#f8f9fa")

    # Plot 3: Loss vs Epoch
    ax3 = axes[1, 0]
    ax3.plot(metrics_history["epoch"], metrics_history["loss"], "purple", linewidth=2.5, alpha=0.8)
    ax3.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax3.set_title("Loss per Epoch", fontsize=15, fontweight="bold", pad=15)
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_facecolor("#f8f9fa")

    # Plot 4: Loss Smoothed (moving average)
    ax4 = axes[1, 1]
    window = min(max(5, len(metrics_history["loss"]) // 20), 50)
    if len(metrics_history["loss"]) > window:
        smoothed = np.convolve(metrics_history["loss"], np.ones(window) / window, mode="valid")
        ax4.plot(
            metrics_history["step"][window - 1 :],
            smoothed,
            "darkorange",
            linewidth=2.5,
            label=f"Smoothed (window={window})",
            alpha=0.8,
        )
        ax4.plot(
            metrics_history["step"], metrics_history["loss"], "b-", linewidth=0.5, alpha=0.3, label="Raw Loss"
        )
        ax4.set_xlabel("Step", fontsize=13, fontweight="bold")
        ax4.set_ylabel("Loss", fontsize=13, fontweight="bold")
        ax4.set_title("Smoothed Loss", fontsize=15, fontweight="bold", pad=15)
        ax4.legend(fontsize=10, loc="best", framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle="--")
        ax4.set_facecolor("#f8f9fa")
    else:
        ax4.text(
            0.5,
            0.5,
            "Not enough data\nfor smoothing",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax4.transAxes,
            color="gray",
        )
        ax4.set_title("Smoothed Loss", fontsize=15, fontweight="bold", pad=15)
        ax4.set_facecolor("#f8f9fa")

    plt.tight_layout()

    # Save combined plot
    plot_path = os.path.join(output_dir, "training_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Training plots saved to: {plot_path}")

    # Save individual plots
    individual_dir = os.path.join(output_dir, "plots")
    os.makedirs(individual_dir, exist_ok=True)

    # Individual: Loss
    fig_loss, ax = plt.subplots(figsize=(12, 7))
    ax.plot(metrics_history["step"], metrics_history["loss"], "b-", linewidth=2.5, label="Training Loss", alpha=0.8)
    if metrics_history["eval_loss"]:
        ax.plot(
            metrics_history["eval_steps"],
            metrics_history["eval_loss"],
            "ro-",
            linewidth=2.5,
            markersize=8,
            label="Eval Loss",
            alpha=0.9,
        )
    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax.set_title("Training Loss Over Time", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(os.path.join(individual_dir, "loss.png"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig_loss)

    # Individual: Learning Rate
    fig_lr, ax = plt.subplots(figsize=(12, 7))
    ax.plot(metrics_history["step"], metrics_history["learning_rate"], "g-", linewidth=2.5, alpha=0.8)
    ax.set_xlabel("Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Learning Rate", fontsize=14, fontweight="bold")
    ax.set_title("Learning Rate Schedule", fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(
        os.path.join(individual_dir, "learning_rate.png"), dpi=250, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig_lr)

    plt.close("all")
    print(f"✅ Individual plots saved to: {individual_dir}/")
    print()

