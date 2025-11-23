#!/usr/bin/env python3
"""
Plot training results from experiment losses.npz files

Usage:
    # Plot single experiment
    python scripts/plot_training_results.py --experiment output/exp1_synthetic

    # Compare multiple experiments
    python scripts/plot_training_results.py --compare output/exp1_synthetic output/exp2_synthetic_real output/exp3_real_threshold

    # Save plot instead of displaying
    python scripts/plot_training_results.py --experiment output/exp1_synthetic --save results.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_losses(experiment_dir):
    """Load losses from experiment directory"""
    losses_path = Path(experiment_dir) / "losses.npz"
    if not losses_path.exists():
        raise FileNotFoundError(f"No losses.npz found in {experiment_dir}")

    data = np.load(losses_path)
    return {
        'epochs': data['epochs'],
        'train_loss': data['train_loss'],
        'val_loss': data.get('val_loss', None),
        'best_val_loss': data.get('best_val_loss', None),
        'best_epoch': data.get('best_epoch', None),
    }


def plot_single_experiment(experiment_dir, save_path=None):
    """Plot training curves for a single experiment"""
    losses = load_losses(experiment_dir)
    exp_name = Path(experiment_dir).name

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot training loss
    ax.plot(losses['epochs'], losses['train_loss'],
            label='Training Loss', color='blue', linewidth=2, marker='o', markersize=4)

    # Plot validation loss if available
    if losses['val_loss'] is not None:
        ax.plot(losses['epochs'], losses['val_loss'],
                label='Validation Loss', color='red', linewidth=2, marker='s', markersize=4)

        # Mark best epoch
        if losses['best_epoch'] is not None:
            best_idx = losses['best_epoch'] - 1  # Convert to 0-indexed
            ax.axvline(losses['best_epoch'], color='green', linestyle='--', alpha=0.5,
                      label=f"Best Val Epoch ({losses['best_epoch']})")
            ax.scatter([losses['best_epoch']], [losses['val_loss'][best_idx]],
                      color='green', s=100, zorder=5, marker='*')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (DiceCE)', fontsize=12)
    ax.set_title(f'Training Progress: {exp_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    final_train = losses['train_loss'][-1]
    stats_text = f"Final Train Loss: {final_train:.6f}"
    if losses['val_loss'] is not None:
        final_val = losses['val_loss'][-1]
        best_val = losses['best_val_loss']
        stats_text += f"\nFinal Val Loss: {final_val:.6f}"
        stats_text += f"\nBest Val Loss: {best_val:.6f} (epoch {losses['best_epoch']})"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_comparison(experiment_dirs, save_path=None):
    """Compare multiple experiments on the same plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, exp_dir in enumerate(experiment_dirs):
        losses = load_losses(exp_dir)
        exp_name = Path(exp_dir).name
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Plot training loss
        ax1.plot(losses['epochs'], losses['train_loss'],
                label=exp_name, color=color, linewidth=2, marker=marker, markersize=4, alpha=0.7)

        # Plot validation loss if available
        if losses['val_loss'] is not None:
            ax2.plot(losses['epochs'], losses['val_loss'],
                    label=exp_name, color=color, linewidth=2, marker=marker, markersize=4, alpha=0.7)

    # Configure training loss plot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (DiceCE)', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Configure validation loss plot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss (DiceCE)', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()


def print_summary(experiment_dir):
    """Print summary statistics for an experiment"""
    losses = load_losses(experiment_dir)
    exp_name = Path(experiment_dir).name

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")
    print(f"Total epochs: {len(losses['epochs'])}")
    print(f"\nTraining Loss:")
    print(f"  Initial: {losses['train_loss'][0]:.6f}")
    print(f"  Final:   {losses['train_loss'][-1]:.6f}")
    print(f"  Best:    {losses['train_loss'].min():.6f} (epoch {losses['train_loss'].argmin() + 1})")

    if losses['val_loss'] is not None:
        print(f"\nValidation Loss:")
        print(f"  Initial: {losses['val_loss'][0]:.6f}")
        print(f"  Final:   {losses['val_loss'][-1]:.6f}")
        print(f"  Best:    {losses['best_val_loss']:.6f} (epoch {losses['best_epoch']})")

        # Check for overfitting
        train_val_gap = losses['train_loss'][-1] - losses['val_loss'][-1]
        if train_val_gap < -0.05:
            print(f"\n⚠ Warning: Underfitting detected (train > val by {abs(train_val_gap):.6f})")
        elif train_val_gap > 0.05:
            print(f"\n⚠ Warning: Possible overfitting (val > train by {train_val_gap:.6f})")
        else:
            print(f"\n✓ Good train/val balance (gap: {train_val_gap:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Plot training results from experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Plot single experiment")
    group.add_argument("--compare", nargs='+', help="Compare multiple experiments")

    parser.add_argument("--save", help="Save plot to file instead of displaying")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")

    args = parser.parse_args()

    if args.experiment:
        if args.summary:
            print_summary(args.experiment)
        plot_single_experiment(args.experiment, save_path=args.save)

    elif args.compare:
        if args.summary:
            for exp_dir in args.compare:
                print_summary(exp_dir)
        plot_comparison(args.compare, save_path=args.save)


if __name__ == "__main__":
    main()
