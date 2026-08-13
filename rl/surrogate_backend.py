"""
SurrogateBackend (v2 — DUAL BASE TYPE)

Routes structural-response queries between the trained surrogate and real
ABAQUS FEA, with periodic validation and drift tracking.

v2 change (Phase 2D dual-base):
  - get_responses() now accepts `pavement_type` and forwards to predictor
  - For v3 surrogate (in_dim=12), each pavement_type uses the same
    network with different is_semi_rigid flag

Phase 2D policy (unchanged from v1):
  - Episode step 0 (reset): caller forces FEA externally.
  - Every `fea_validation_every` global steps: real FEA + parallel surrogate
    prediction → per-output drift logged.
  - Otherwise: surrogate prediction; if predicted B3 < b3_threshold,
    escalate to real FEA.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from rl.surrogate_predictor import SurrogatePredictor

logger = logging.getLogger(__name__)


class SurrogateBackend:
    """Routing layer between surrogate and FEA + drift tracking."""

    RESPONSE_KEYS = (
        "epsilon_a_microstrain",
        "sigma_t_MPa",
        "epsilon_z_microstrain",
        "p_AC_upper_mid_MPa",
        "p_AC_mid_mid_MPa",
        "p_AC_lower_mid_MPa",
    )

    def __init__(self,
                 model_path: str,
                 fea_validation_every: int = 10,
                 b3_threshold: float = 1.2,
                 device: str = "auto"):
        self.predictor = SurrogatePredictor(model_path, device=device)
        self.fea_validation_every = max(1, int(fea_validation_every))
        self.b3_threshold = float(b3_threshold)

        # Counters
        self.n_surrogate = 0
        self.n_fea_total = 0
        self.n_fea_validation = 0
        self.n_fea_escalation = 0
        self.drift_history: List[Dict[str, float]] = []

        logger.info(
            f"SurrogateBackend ready  model={model_path}  "
            f"mode={self.predictor.mode}  "
            f"validate-every={self.fea_validation_every}  "
            f"B3-escalate-below={self.b3_threshold}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────
    def get_responses(self,
                       thickness: List[float],
                       modulus: List[float],
                       E_subgrade: float,
                       episode_step: int,
                       global_step: int,
                       fea_fallback: Callable[[List[float], List[float]],
                                                Dict[str, float]],
                       evaluate_protocol: Callable[[Dict[str, float]],
                                                     Dict[str, float]],
                       pavement_type: str = "semi_rigid",
                       ) -> Dict:
        """
        Route the FEA-equivalent response query.

        New v2 param:
            pavement_type: 'semi_rigid' | 'flexible' — passed to predictor.
                           Defaults to 'semi_rigid' for backward compat.

        Returns
        -------
        dict with keys:
            'responses' : Dict[str, float] — the 6 mechanistic responses
            'source'    : str  ∈ {'surrogate', 'fea_validation',
                                   'surrogate_escalated', 'fea_fallback'}
            'drift'     : Optional[Dict[str, float]] — per-output % diff
        """
        is_validation_step = (
            global_step > 0
            and (global_step % self.fea_validation_every == 0)
        )
        if is_validation_step:
            return self._validation_path(thickness, modulus, E_subgrade,
                                          fea_fallback, pavement_type)

        return self._surrogate_path(thickness, modulus, E_subgrade,
                                     fea_fallback, evaluate_protocol,
                                     pavement_type)

    # ──────────────────────────────────────────────────────────────────
    # Path: scheduled validation
    # ──────────────────────────────────────────────────────────────────
    def _validation_path(self, thickness, modulus, E_subgrade,
                          fea_fallback, pavement_type) -> Dict:
        fea_responses = fea_fallback(thickness, modulus)
        self.n_fea_total += 1
        self.n_fea_validation += 1

        drift = None
        try:
            surr_responses = self.predictor.predict(
                thickness=thickness,
                modulus=modulus,
                E_subgrade=E_subgrade,
                pavement_type=pavement_type,
            )
            drift = self._compute_drift(surr_responses, fea_responses)
            self.drift_history.append(drift)
        except Exception as e:
            logger.warning(f"Surrogate prediction failed during validation: {e}")

        return {
            "responses": fea_responses,
            "source": "fea_validation",
            "drift": drift,
        }

    # ──────────────────────────────────────────────────────────────────
    # Path: surrogate with optional escalation
    # ──────────────────────────────────────────────────────────────────
    def _surrogate_path(self, thickness, modulus, E_subgrade,
                          fea_fallback, evaluate_protocol,
                          pavement_type) -> Dict:
        try:
            surr_responses = self.predictor.predict(
                thickness=thickness,
                modulus=modulus,
                E_subgrade=E_subgrade,
                pavement_type=pavement_type,
            )
        except Exception as e:
            logger.warning(f"Surrogate prediction failed; FEA fallback: {e}")
            fea_responses = fea_fallback(thickness, modulus)
            self.n_fea_total += 1
            return {"responses": fea_responses, "source": "fea_fallback",
                    "drift": None}

        # B3 escalation check
        b3_pred = float("inf")
        try:
            margins = evaluate_protocol(surr_responses)
            b3_pred = float(margins.get("B3_ac_permanent_deformation",
                                        float("inf")))
        except Exception as e:
            logger.warning(f"B3 escalation eval failed: {e}; using surrogate")
            b3_pred = float("inf")

        if b3_pred < self.b3_threshold:
            fea_responses = fea_fallback(thickness, modulus)
            self.n_fea_total += 1
            self.n_fea_escalation += 1
            drift = self._compute_drift(surr_responses, fea_responses)
            drift["__b3_predicted"] = b3_pred
            drift["__b3_threshold"] = self.b3_threshold
            self.drift_history.append(drift)
            logger.info(f"  surrogate escalated → FEA "
                        f"(B3 predicted={b3_pred:.2f} < threshold {self.b3_threshold})")
            return {"responses": fea_responses,
                    "source": "surrogate_escalated",
                    "drift": drift}

        self.n_surrogate += 1
        return {"responses": surr_responses, "source": "surrogate",
                "drift": None}

    # ──────────────────────────────────────────────────────────────────
    # Drift computation
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_drift(surr: Dict[str, float],
                        fea:  Dict[str, float]) -> Dict[str, float]:
        """Per-output signed percentage diff: (surr - fea) / |fea| * 100."""
        drift: Dict[str, float] = {}
        for k, fea_v in fea.items():
            if k not in surr:
                continue
            try:
                denom = max(abs(float(fea_v)), 1e-9)
                pct = (float(surr[k]) - float(fea_v)) / denom * 100.0
                drift[f"{k}_pct"] = float(pct)
            except (TypeError, ValueError):
                continue
        return drift

    # ──────────────────────────────────────────────────────────────────
    # Stats / reporting
    # ──────────────────────────────────────────────────────────────────
    def stats_summary(self) -> Dict:
        """End-of-training drift + routing summary."""
        total_calls = self.n_surrogate + self.n_fea_total
        out: Dict[str, float] = {
            "n_surrogate_calls":  self.n_surrogate,
            "n_fea_calls_total":  self.n_fea_total,
            "n_fea_validation":   self.n_fea_validation,
            "n_fea_escalation":   self.n_fea_escalation,
            "n_drift_records":    len(self.drift_history),
            "surrogate_fraction": (self.n_surrogate / total_calls
                                    if total_calls > 0 else 0.0),
            "predictor_mode":     self.predictor.mode,
        }
        if not self.drift_history:
            return out
        all_keys = sorted({k for d in self.drift_history for k in d.keys()
                            if k.endswith("_pct")})
        for k in all_keys:
            vals = [abs(d[k]) for d in self.drift_history if k in d]
            if vals:
                out[f"drift_{k}_max_abs"]  = float(np.max(vals))
                out[f"drift_{k}_mean_abs"] = float(np.mean(vals))
        return out

    def reset_stats(self) -> None:
        self.n_surrogate = 0
        self.n_fea_total = 0
        self.n_fea_validation = 0
        self.n_fea_escalation = 0
        self.drift_history.clear()
