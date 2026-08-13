# -*- coding: utf-8 -*-
"""
Analyze a completed PPO training run
======================================

Generates a 2x3 panel of plots from a training run's monitor.csv:
    [reward / episode length / feasibility rate] (per episode)
    [B1, B2, B3, B4 margins] (per episode, averaged over steps)

Usage:
    python examples/analyze_training.py <run_dir>

    where <run_dir> is something like:
        output/rl_runs/ppo_jtg_d50_2017_seed0_20260518_153045

Outputs:
    <run_dir>/training_curves.png
    <run_dir>/training_summary.txt
"""
import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_monitor_csv(run_dir: str) -> Optional['pandas.DataFrame']:
    """Load SB3 monitor CSV (skipping the header comment line)."""
    try:
        import pandas as pd
    except ImportError:
        print('pandas not installed. Install with: pip install pandas')
        return None

    # SB3 writes monitor at <monitor_dir>/monitor.monitor.csv
    candidates = glob.glob(os.path.join(run_dir, 'monitor', '*.csv'))
    if not candidates:
        candidates = glob.glob(os.path.join(run_dir, '**', '*.csv'), recursive=True)
        candidates = [c for c in candidates if 'monitor' in c.lower()]
    if not candidates:
        print('No monitor.csv found in {}'.format(run_dir))
        return None

    csv_path = candidates[0]
    print('Loading {}'.format(csv_path))
    df = pd.read_csv(csv_path, skiprows=1)
    return df


def load_tensorboard_scalars(run_dir: str) -> dict:
    """
    Extract scalar metrics from TensorBoard event files.
    Returns: {tag_name: [(step, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print('tensorboard not installed. Install with: pip install tensorboard')
        return {}

    tb_dirs = glob.glob(os.path.join(run_dir, 'tensorboard', '**'), recursive=True)
    event_files = []
    for d in tb_dirs:
        if os.path.isdir(d):
            event_files.extend(glob.glob(os.path.join(d, 'events.out.tfevents.*')))
    if not event_files:
        print('No tensorboard events found')
        return {}

    # Use the most recent event file
    event_files.sort(key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(event_files[-1])
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    scalars = {}
    for tag in tags:
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]
    return scalars


def plot_training_curves(run_dir: str):
    """Generate the analysis plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed. Install with: pip install matplotlib')
        return

    df = load_monitor_csv(run_dir)
    scalars = load_tensorboard_scalars(run_dir)

    if df is None and not scalars:
        print('No training data found.')
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('PPO Training Curves: {}'.format(os.path.basename(run_dir)),
                 fontsize=14, fontweight='bold')

    # ─── Top-left: Episode reward ───────────────────────────────
    ax = axes[0, 0]
    if df is not None and 'r' in df.columns:
        ep = np.arange(1, len(df) + 1)
        ax.plot(ep, df['r'], 'b.-', alpha=0.6, label='Episode reward')
        if len(df) >= 5:
            # 5-episode moving average
            window = min(5, len(df))
            ma = df['r'].rolling(window=window, min_periods=1).mean()
            ax.plot(ep, ma, 'r-', linewidth=2, label='5-ep moving avg')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total reward')
        ax.set_title('Reward progression')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    # ─── Top-mid: Episode length ────────────────────────────────
    ax = axes[0, 1]
    if df is not None and 'l' in df.columns:
        ep = np.arange(1, len(df) + 1)
        ax.plot(ep, df['l'], 'g.-')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps per episode')
        ax.set_title('Episode length')
        ax.grid(True, alpha=0.3)

    # ─── Top-right: Feasibility rate ────────────────────────────
    ax = axes[0, 2]
    if 'feasibility/rate' in scalars:
        data = scalars['feasibility/rate']
        steps = [s for s, _ in data]
        vals  = [v for _, v in data]
        ax.plot(steps, vals, 'mo-')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Feasibility rate')
        ax.set_title('Feasibility rate (per rollout)')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)

    # ─── Bottom row: 4 margins ──────────────────────────────────
    margin_keys = [
        ('margins/B1_asphalt_fatigue_mean',          'B1 AC fatigue'),
        ('margins/B2_semi_rigid_fatigue_mean',       'B2 Semi-rigid fatigue'),
        ('margins/B3_ac_permanent_deformation_mean', 'B3 Permanent def'),
        ('margins/B4_subgrade_strain_mean',          'B4 Subgrade strain'),
    ]

    for i, (key, label) in enumerate(margin_keys):
        # Plot first 3 on bottom row, last one we overlay
        if i < 3:
            ax = axes[1, i]
        else:
            ax = axes[1, 2]  # overlay B4 on B3 panel actually let's leave it
            continue

        if key in scalars:
            data = scalars[key]
            steps = [s for s, _ in data]
            vals  = [v for _, v in data]
            ax.plot(steps, vals, 'o-', label='Mean margin')
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5,
                       label='Feasibility threshold')
            ax.set_xlabel('Training step')
            ax.set_ylabel('Margin (capacity / demand)')
            ax.set_title(label)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: {}'.format(out_path))

    # ─── Text summary ──────────────────────────────────────────
    summary_lines = [
        'PPO Training Summary: {}'.format(os.path.basename(run_dir)),
        '=' * 70,
    ]

    if df is not None:
        summary_lines.append('Episodes completed: {}'.format(len(df)))
        if len(df) > 0:
            summary_lines.append('Mean episode reward: {:.3f}'.format(df['r'].mean()))
            summary_lines.append('Final 5 ep mean reward: {:.3f}'.format(
                df['r'].tail(5).mean() if len(df) >= 5 else df['r'].mean()))
            summary_lines.append('Reward range: [{:.3f}, {:.3f}]'.format(
                df['r'].min(), df['r'].max()))
            # Improvement check
            if len(df) >= 10:
                first_half = df['r'].head(len(df)//2).mean()
                second_half = df['r'].tail(len(df)//2).mean()
                improvement = second_half - first_half
                summary_lines.append('Reward improvement (2nd half - 1st half): {:+.3f}'.format(improvement))
                if improvement > 0.05:
                    summary_lines.append('  → LEARNING (reward rising)')
                elif improvement < -0.05:
                    summary_lines.append('  → REGRESSING (reward dropping — check reward fn)')
                else:
                    summary_lines.append('  → STABLE (no clear learning yet — try more episodes)')

    if 'feasibility/rate' in scalars:
        data = scalars['feasibility/rate']
        vals = [v for _, v in data]
        if vals:
            summary_lines.append('Feasibility rate: mean={:.2f}, final={:.2f}'.format(
                np.mean(vals), vals[-1]))

    for key, label in margin_keys:
        if key in scalars:
            data = scalars[key]
            vals = [v for _, v in data]
            if vals:
                summary_lines.append('{}: mean={:.2f}, final={:.2f}'.format(
                    label, np.mean(vals), vals[-1]))

    summary_lines.append('')
    summary_lines.append('=' * 70)

    summary_text = '\n'.join(summary_lines)
    out_path = os.path.join(run_dir, 'training_summary.txt')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(summary_text)
    print('Saved: {}'.format(out_path))
    print()
    print(summary_text)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a PPO training run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python examples/analyze_training.py output/rl_runs/ppo_jtg_d50_2017_seed0_<ts>',
    )
    parser.add_argument('run_dir', help='Path to the training run directory')
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print('Error: directory not found: {}'.format(args.run_dir))
        sys.exit(1)

    plot_training_curves(args.run_dir)


if __name__ == '__main__':
    main()
