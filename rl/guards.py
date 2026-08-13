# -*- coding: utf-8 -*-
"""
rl.guards — Numerical Guard (v0.5, 6-layer, DUAL BASE TYPE)
==============================================================

Hard validators wrapping every numerical I/O.

UPGRADE v0.5 (Phase 2D dual-base):
    - Per-layer bounds now split by `base_type` ∈ {"semi_rigid", "flexible"}
    - E_base / E_subbase ranges differ; AC and subgrade ranges unchanged.
    - GuardConfig.from_base_type() builds the right config.
    - NumericalGuard exposes .base_type for downstream inspection.

UPGRADE v0.4 (Phase 2A-1):
    - Layer count: 3 → 5 (top-down: upper_AC, mid_AC, lower_AC, base, subbase)

Failure mode: raises GuardViolation with specific code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class GuardViolation(Exception):
    """Numerical guard rejected an input or output."""
    def __init__(self, code: str, message: str, payload: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.payload = payload or {}


# ─────────────────────────────────────────────────────────────────────
# Dual base-type bounds tables
# ─────────────────────────────────────────────────────────────────────

# Common to both base types (AC layers, subgrade)
_AC_h_min = [0.02, 0.03, 0.04]    # upper, mid, lower AC
_AC_h_max = [0.10, 0.15, 0.25]
_AC_E_min = [4000.0, 3000.0, 2000.0]
_AC_E_max = [25000.0, 18000.0, 15000.0]

# Base / subbase by type
_BASE_H_MIN_BY_TYPE = {
    "semi_rigid": [0.15, 0.10],          # base, subbase
    "flexible":   [0.15, 0.10],
}
_BASE_H_MAX_BY_TYPE = {
    "semi_rigid": [0.50, 0.40],          # original tighter range
    "flexible":   [0.50, 0.45],          # flexible base/subbase tend thicker
}
_BASE_E_MIN_BY_TYPE = {
    "semi_rigid": [800.0,  200.0],       # cement / lime stabilized
    "flexible":   [150.0,  100.0],       # unbound granular
}
_BASE_E_MAX_BY_TYPE = {
    "semi_rigid": [3500.0, 800.0],
    "flexible":   [500.0,  400.0],
}


@dataclass
class GuardConfig:
    """
    Physical bounds (loose — only to prevent FEA crash / NaN).
    Order: [upper_AC, mid_AC, lower_AC, base, subbase]

    For backward compatibility, default values match v0.4 semi-rigid bounds.
    Use GuardConfig.from_base_type("flexible") to get the granular-base variant.
    """
    # Layer thickness bounds (m)
    h_min: List[float] = field(default_factory=lambda:
        list(_AC_h_min) + _BASE_H_MIN_BY_TYPE["semi_rigid"])
    h_max: List[float] = field(default_factory=lambda:
        list(_AC_h_max) + _BASE_H_MAX_BY_TYPE["semi_rigid"])

    # Layer modulus bounds (MPa)
    E_min: List[float] = field(default_factory=lambda:
        list(_AC_E_min) + _BASE_E_MIN_BY_TYPE["semi_rigid"])
    E_max: List[float] = field(default_factory=lambda:
        list(_AC_E_max) + _BASE_E_MAX_BY_TYPE["semi_rigid"])

    # Subgrade (same for both base types)
    E_subgrade_min: float = 20.0
    E_subgrade_max: float = 1000.0   # raised for LTPP stiff subgrades (e.g. 48_0001 = 700 MPa)

    # FEA output sanity bounds (same for both)
    epsilon_a_max_microstrain: float = 5000.0
    sigma_t_max_MPa: float = 20.0
    epsilon_z_max_microstrain: float = 5000.0

    # Tag for downstream introspection
    base_type: str = "semi_rigid"

    # ── Class constructors ─────────────────────────────────────
    @classmethod
    def from_base_type(cls, base_type: str) -> "GuardConfig":
        """Build a config matching the requested base type."""
        bt = (base_type or "").lower().strip()
        if bt in ("semi_rigid", "semirigid", "semi-rigid"):
            bt = "semi_rigid"
        elif bt in ("flexible", "unbound", "granular", "unbound_granular"):
            bt = "flexible"
        else:
            raise ValueError(
                "Unknown base_type {!r}; expected 'semi_rigid' or 'flexible'"
                .format(base_type))
        return cls(
            h_min=list(_AC_h_min) + _BASE_H_MIN_BY_TYPE[bt],
            h_max=list(_AC_h_max) + _BASE_H_MAX_BY_TYPE[bt],
            E_min=list(_AC_E_min) + _BASE_E_MIN_BY_TYPE[bt],
            E_max=list(_AC_E_max) + _BASE_E_MAX_BY_TYPE[bt],
            base_type=bt,
        )


class NumericalGuard:
    """Stateless validator. Use as `guard.check_design(...)` etc."""

    def __init__(self, config: Optional[GuardConfig] = None,
                 base_type: Optional[str] = None):
        """
        Args:
            config:     explicit GuardConfig. Takes precedence if provided.
            base_type:  'semi_rigid' | 'flexible'. Used if config is None.
                        Defaults to 'semi_rigid' (backward compatible).
        """
        if config is not None:
            self.config = config
        elif base_type is not None:
            self.config = GuardConfig.from_base_type(base_type)
        else:
            self.config = GuardConfig()  # defaults to semi_rigid
        self.base_type = self.config.base_type

    # ─── Pre-FEA: design parameters validation ───────────────────

    def check_design(self, thickness: np.ndarray, modulus: np.ndarray,
                     E_subgrade: float) -> None:
        """Validate 5-element design vectors before FEA call."""
        thickness = np.asarray(thickness, dtype=float)
        modulus = np.asarray(modulus, dtype=float)

        if thickness.shape != (5,):
            raise GuardViolation(
                'SHAPE_THICKNESS',
                'Expected thickness length 5 (upper/mid/lower AC + base + subbase), '
                'got shape={}'.format(thickness.shape))
        if modulus.shape != (5,):
            raise GuardViolation(
                'SHAPE_MODULUS',
                'Expected modulus length 5, got shape={}'.format(modulus.shape))

        if not np.all(np.isfinite(thickness)) or not np.all(np.isfinite(modulus)):
            raise GuardViolation('NONFINITE', 'NaN/Inf in design params')

        for i, (h, h_min, h_max) in enumerate(zip(
                thickness, self.config.h_min, self.config.h_max)):
            if h < h_min or h > h_max:
                raise GuardViolation(
                    'H_OUT_OF_BOUNDS',
                    'Layer {} thickness {:.4f} outside [{}, {}] '
                    '(base_type={})'
                    .format(i, h, h_min, h_max, self.base_type),
                    {'layer': i, 'value': float(h),
                     'base_type': self.base_type},
                )

        for i, (E, E_min, E_max) in enumerate(zip(
                modulus, self.config.E_min, self.config.E_max)):
            if E < E_min or E > E_max:
                raise GuardViolation(
                    'E_OUT_OF_BOUNDS',
                    'Layer {} modulus {:.1f} outside [{}, {}] '
                    '(base_type={})'
                    .format(i, E, E_min, E_max, self.base_type),
                    {'layer': i, 'value': float(E),
                     'base_type': self.base_type},
                )

        if not (self.config.E_subgrade_min <= E_subgrade <= self.config.E_subgrade_max):
            raise GuardViolation(
                'E_SUBGRADE_OUT_OF_BOUNDS',
                'Subgrade modulus {:.1f} outside [{}, {}]'
                .format(E_subgrade, self.config.E_subgrade_min,
                        self.config.E_subgrade_max),
            )

    # ─── Post-FEA: result sanity check ───────────────────────────

    def check_fea_result(self, fea_responses: Dict[str, float]) -> None:
        """Validate FEA result for NaN, Inf, and physically-insane values."""
        for k, v in fea_responses.items():
            if k.startswith('_'):
                continue
            if not isinstance(v, (int, float)):
                continue
            if not np.isfinite(v):
                raise GuardViolation('FEA_NONFINITE',
                                     'FEA returned NaN/Inf for {}'.format(k),
                                     {'key': k, 'value': v})

        eps_a = fea_responses.get('epsilon_a_microstrain', 0)
        if eps_a > self.config.epsilon_a_max_microstrain:
            raise GuardViolation(
                'FEA_INSANE_EPS_A',
                'eps_a={:.1f} ue exceeds sanity cap {}'.format(
                    eps_a, self.config.epsilon_a_max_microstrain),
            )

        sig_t = fea_responses.get('sigma_t_MPa', 0)
        if sig_t > self.config.sigma_t_max_MPa:
            raise GuardViolation(
                'FEA_INSANE_SIGMA_T',
                'sigma_t={:.3f} MPa exceeds sanity cap {}'.format(
                    sig_t, self.config.sigma_t_max_MPa),
            )

        eps_z = fea_responses.get('epsilon_z_microstrain', 0)
        if eps_z > self.config.epsilon_z_max_microstrain:
            raise GuardViolation(
                'FEA_INSANE_EPS_Z',
                'eps_z={:.1f} ue exceeds sanity cap {}'.format(
                    eps_z, self.config.epsilon_z_max_microstrain),
            )

    # ─── Action clipping (soft; doesn't raise) ───────────────────

    def clip_action_to_design(
        self,
        current_thickness: np.ndarray,
        current_modulus: np.ndarray,
        new_thickness: np.ndarray,
        new_modulus: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Soft-clip proposed design to physical bounds."""
        clipped_h = np.clip(new_thickness, self.config.h_min, self.config.h_max)
        clipped_E = np.clip(new_modulus, self.config.E_min, self.config.E_max)
        was_clipped = bool(
            np.any(clipped_h != new_thickness) or np.any(clipped_E != new_modulus)
        )
        return clipped_h, clipped_E, was_clipped
