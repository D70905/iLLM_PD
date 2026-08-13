# -*- coding: utf-8 -*-
"""
rl.policy — PPO policy + training utilities
=============================================

Wraps stable-baselines3 PPO with:
    - Hyperparameters matching original MATLAB (lr=2e-3, γ=0.99, clip=0.2)
    - MLP actor-critic, hidden=[64, 64] (MATLAB used hidden_dim=64)
    - TensorBoard logging
    - Checkpoint callback
    - Custom evaluation callback (logs spec margins each rollout)

Phase 2A: PPO only. Phase 2F can add SAC for ablation comparison.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# stable-baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback, CheckpointCallback, EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.logger import configure as configure_logger
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None


@dataclass
class PPOHyperparams:
    """
    PPO hyperparameters.

    Defaults preserve original MATLAB iLLM-PD design where applicable.
    """
    # ── Core PPO (preserved from MATLAB) ────────────────────────
    learning_rate:     float = 2.0e-3       # MATLAB: 0.002
    gamma:             float = 0.99         # MATLAB: 0.99
    clip_range:        float = 0.2          # MATLAB: 0.2
    n_epochs:          int   = 4            # MATLAB: ppo_epochs = 4
    ent_coef:          float = 0.01         # MATLAB: entropy_coeff = 0.01
    vf_coef:           float = 0.5          # SB3 default
    gae_lambda:        float = 0.95         # MATLAB: gae_lambda = 0.95
    max_grad_norm:     float = 0.5          # SB3 default

    # ── Rollout / batch ────────────────────────────────────────
    n_steps:           int   = 64           # rollout length per env per update
    batch_size:        int   = 32           # MATLAB: batch_size = 32
    target_kl:         Optional[float] = 0.05    # early stop if KL exceeded

    # ── Network architecture ───────────────────────────────────
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    # MATLAB: hidden_dim = 64

    # ── Training schedule ──────────────────────────────────────
    total_timesteps:   int = 8000   # was 4000; R3-14 convergence diagnosis
    seed:              Optional[int] = 0
    device:            str = 'auto'

    def to_sb3_kwargs(self) -> Dict:
        """Convert to PPO(**kwargs) keyword arguments."""
        return dict(
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
            policy_kwargs={'net_arch': dict(
                pi=self.hidden_dims, vf=self.hidden_dims)},
            seed=self.seed,
            device=self.device,
            verbose=1,
        )


class SpecMarginsLoggingCallback(BaseCallback):
    """
    Custom callback: logs spec margins + feasibility rate to TensorBoard.

    SB3 by default logs only reward; we want to see margins evolving too.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_margins: List[Dict[str, float]] = []
        self.episode_feasible: List[bool] = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            eval_summary = info.get('evaluation', {})
            if 'margins' in eval_summary:
                self.episode_margins.append(eval_summary['margins'])
                self.episode_feasible.append(eval_summary.get('feasible', False))
        return True

    def _on_rollout_end(self) -> None:
        if not self.episode_margins:
            return
        # Average margins over this rollout
        all_keys = set()
        for m in self.episode_margins:
            all_keys.update(m.keys())
        for k in all_keys:
            values = [m[k] for m in self.episode_margins if k in m]
            if values:
                self.logger.record('margins/{}_mean'.format(k), float(np.mean(values)))
                self.logger.record('margins/{}_min'.format(k),  float(np.min(values)))
        # Feasibility rate
        if self.episode_feasible:
            feas_rate = float(np.mean([1.0 if f else 0.0 for f in self.episode_feasible]))
            self.logger.record('feasibility/rate', feas_rate)
        # Reset rollout buffers
        self.episode_margins = []
        self.episode_feasible = []


def build_ppo(env, hp: PPOHyperparams, tb_log_dir: Optional[str] = None) -> 'PPO':
    """Construct PPO model with given hyperparameters."""
    if not SB3_AVAILABLE:
        raise ImportError(
            'stable-baselines3 is not installed. Install with:\n'
            '    pip install stable-baselines3[extra]'
        )

    kwargs = hp.to_sb3_kwargs()
    if tb_log_dir:
        kwargs['tensorboard_log'] = tb_log_dir

    model = PPO('MlpPolicy', env, **kwargs)
    return model


def wrap_env(env_factory, n_envs: int = 1, monitor_dir: Optional[str] = None):
    """Wrap a single env factory into a vectorized + monitored env."""
    vec = DummyVecEnv([env_factory for _ in range(n_envs)])
    if monitor_dir:
        vec = VecMonitor(vec, filename=monitor_dir)
    return vec
