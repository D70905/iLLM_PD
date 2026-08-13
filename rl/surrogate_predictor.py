"""
Surrogate model inference wrapper (v3 — DUAL BASE TYPE).

Loads a trained MLP surrogate and provides a simple predict() interface.

v3 (Phase 2D dual-base):
    - Input dim: 11 → 12 (adds `is_semi_rigid` ∈ {0, 1} as 12th feature)
    - predict() now accepts `pavement_type` ∈ {"semi_rigid", "flexible"}
    - Backward-compat with v1/v2: if a 11-dim checkpoint is loaded
      (model_arch.in_dim == 11), we run in legacy mode (assume semi_rigid).

v2-compatible: reads `dropout` from checkpoint's `model_arch`.

Usage:
    from rl.surrogate_predictor import SurrogatePredictor

    pred = SurrogatePredictor("output/surrogate_model/surrogate_v3.pt")
    responses = pred.predict(
        thickness=[0.05, 0.06, 0.08, 0.30, 0.20],
        modulus=[12000.0, 9000.0, 6000.0, 1500.0, 400.0],
        E_subgrade=80.0,
        pavement_type="semi_rigid",     # ← new (default keeps v1/v2 behavior)
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn


# v3 labels (12 dims). For v1/v2 checkpoints (11 dims), we drop the last.
INPUT_LABELS_V3 = [
    "h_upper_AC", "h_mid_AC", "h_lower_AC", "h_base", "h_subbase",
    "E_upper_AC", "E_mid_AC", "E_lower_AC", "E_base", "E_subbase",
    "E_subgrade",
    "is_semi_rigid",        # ← v3 new
]
INPUT_LABELS_V2 = INPUT_LABELS_V3[:11]

OUTPUT_LABELS = [
    "epsilon_a_microstrain",
    "sigma_t_MPa",
    "epsilon_z_microstrain",
    "p_AC_upper_mid_MPa",
    "p_AC_mid_mid_MPa",
    "p_AC_lower_mid_MPa",
]


def _pavement_type_to_flag(pavement_type: str) -> float:
    """Map pavement_type string to is_semi_rigid ∈ {0.0, 1.0}."""
    pt = (pavement_type or "").lower().strip()
    if pt in ("semi_rigid", "semirigid", "semi-rigid"):
        return 1.0
    if pt in ("flexible", "unbound", "granular", "unbound_granular"):
        return 0.0
    raise ValueError(
        "Unknown pavement_type {!r}; expected 'semi_rigid' or 'flexible'"
        .format(pavement_type))


class _SurrogateMLP(nn.Module):
    """Mirror of training-time architecture; loaded from checkpoint."""
    def __init__(self, in_dim: int, out_dim: int, hidden, dropout: float = 0.0):
        super().__init__()
        dims = (in_dim,) + tuple(hidden) + (out_dim,)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SurrogatePredictor:
    """Loads a trained surrogate and provides batched / single-sample prediction."""

    def __init__(self, ckpt_path, device: str = "auto"):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found: {ckpt_path}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        arch = ck["model_arch"]
        self.model = _SurrogateMLP(
            in_dim  = arch["in_dim"],
            out_dim = arch["out_dim"],
            hidden  = arch["hidden"],
            dropout = arch.get("dropout", 0.0),
        ).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

        sc = ck["scaler"]
        self.x_mean   = np.asarray(sc["x_mean"],   dtype=np.float64)
        self.x_std    = np.asarray(sc["x_std"],    dtype=np.float64)
        self.y_mean   = np.asarray(sc["y_mean"],   dtype=np.float64)
        self.y_std    = np.asarray(sc["y_std"],    dtype=np.float64)
        self.log_mask = np.asarray(sc["log_mask"], dtype=bool)
        self.in_dim   = int(arch["in_dim"])
        self.input_labels  = ck.get("input_labels",
                                      INPUT_LABELS_V3 if self.in_dim == 12
                                      else INPUT_LABELS_V2)
        self.output_labels = ck.get("output_labels", OUTPUT_LABELS)
        self.meta = ck.get("training_meta", {})

        # Provenance
        if self.in_dim == 12:
            self._mode = "v3_dual_base"
        elif self.in_dim == 11:
            self._mode = "v2_semi_rigid_only"
        else:
            raise ValueError(
                "Unsupported surrogate in_dim={}; expected 11 (v2) or 12 (v3)"
                .format(self.in_dim))

    # ── single-sample API ────────────────────────────────────────────
    def predict(self,
                thickness: Sequence[float],
                modulus:   Sequence[float],
                E_subgrade: float,
                pavement_type: str = "semi_rigid") -> Dict[str, float]:
        """
        Single (h, E, pavement_type) sample → dict of 6 FEA responses.

        For v2 checkpoints, pavement_type must be 'semi_rigid' (only mode
        the checkpoint was trained on); requesting 'flexible' raises.
        """
        if len(thickness) != 5 or len(modulus) != 5:
            raise ValueError("thickness and modulus must each be 5-vectors")

        base_vec = np.concatenate([
            np.asarray(thickness, dtype=np.float64),
            np.asarray(modulus,   dtype=np.float64),
            [float(E_subgrade)],
        ])

        if self._mode == "v3_dual_base":
            flag = _pavement_type_to_flag(pavement_type)
            x = np.concatenate([base_vec, [flag]])
        else:  # v2 legacy
            if _pavement_type_to_flag(pavement_type) != 1.0:
                raise ValueError(
                    "Legacy v2 surrogate only supports pavement_type='semi_rigid'; "
                    "retrain v3 to use 'flexible'.")
            x = base_vec

        y = self._predict_batch(x[None, :])[0]
        return {lbl: float(y[i]) for i, lbl in enumerate(self.output_labels)}

    # ── batched API ──────────────────────────────────────────────────
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batched prediction. X shape [N, in_dim], returns [N, 6] in raw units."""
        return self._predict_batch(X)

    # ── internals ────────────────────────────────────────────────────
    @torch.no_grad()
    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.x_mean.shape[0]:
            raise ValueError(f"X must have shape [N, {self.x_mean.shape[0]}]; got {X.shape}")
        Xn = (X - self.x_mean) / self.x_std
        xb = torch.as_tensor(Xn, dtype=torch.float32, device=self.device)
        yn = self.model(xb).cpu().numpy()
        Yt = yn * self.y_std + self.y_mean
        if self.log_mask.any():
            cols = np.where(self.log_mask)[0]
            Yt[:, cols] = np.power(10.0, Yt[:, cols])
        return Yt

    @property
    def mode(self) -> str:
        """'v2_semi_rigid_only' or 'v3_dual_base'."""
        return self._mode
