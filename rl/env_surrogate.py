# -*- coding: utf-8 -*-
"""
rl/env_surrogate.py (v2 — DUAL BASE TYPE) — Phase 2D surrogate routing

Does NOT modify env.py. Subclasses PavementEnv and overrides:
  - reset()                  : sets `_in_reset` flag (forces FEA in reset path)
  - _run_fea_and_evaluate()  : routes through SurrogateBackend when appropriate
  - _audit_step()            : appends a 'routing' record (source + drift)

v2 change: pavement_type is now forwarded to backend.get_responses()
           so a v3 surrogate can use is_semi_rigid input flag.

Policy (matches user-confirmed Q1-Q4):
  Q1 fea_validation_every = 10
  Q2 reset → forces real FEA
  Q3 surrogate-vs-FEA drift on validation: log only, don't adjust reward
  Q4 surrogate-predicted B3 < 1.2 → escalate to real FEA
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rl.env import PavementEnv, EnvConfig
from rl.surrogate_backend import SurrogateBackend
from specs.protocol import DesignInputs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Extended config
# ─────────────────────────────────────────────────────────────

@dataclass
class SurrogateEnvConfig(EnvConfig):
    use_surrogate: bool = False
    surrogate_model_path: str = ""
    fea_validation_every: int = 10
    surrogate_b3_threshold: float = 1.2


# ─────────────────────────────────────────────────────────────
# Subclass
# ─────────────────────────────────────────────────────────────

class PavementEnvWithSurrogate(PavementEnv):
    """PavementEnv that routes the FEA call through a surrogate model."""

    def __init__(self, config: Optional[SurrogateEnvConfig] = None):
        super().__init__(config or SurrogateEnvConfig())

        self._in_reset = False
        self._last_response_source: str = "fea"
        self._last_drift_info: Optional[Dict[str, float]] = None

        self._surrogate_backend: Optional[SurrogateBackend] = None
        cfg = self.config
        if getattr(cfg, "use_surrogate", False) and getattr(cfg, "surrogate_model_path", ""):
            try:
                self._surrogate_backend = SurrogateBackend(
                    model_path=cfg.surrogate_model_path,
                    fea_validation_every=getattr(cfg, "fea_validation_every", 10),
                    b3_threshold=getattr(cfg, "surrogate_b3_threshold", 1.2),
                )
                logger.info(
                    "PavementEnvWithSurrogate: backend enabled "
                    "(validate every {} steps, B3 escalate < {}, "
                    "pavement_type={})".format(
                        cfg.fea_validation_every,
                        cfg.surrogate_b3_threshold,
                        cfg.pavement_type))
            except Exception as e:
                logger.warning(
                    "Surrogate backend init failed → FEA-only mode: {}".format(e))
                self._surrogate_backend = None
        else:
            logger.info("PavementEnvWithSurrogate: surrogate disabled")

    # ─── reset wrapper ─────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._in_reset = True
        try:
            obs, info = super().reset(seed=seed, options=options)
        finally:
            self._in_reset = False
        info["response_source"] = self._last_response_source
        if self._last_drift_info is not None:
            info["surrogate_drift"] = self._last_drift_info
        return obs, info

    # ─── Override the FEA call ─────────────────────────────────────
    def _run_fea_and_evaluate(self, design: Dict[str, np.ndarray]
                                ) -> Tuple[Dict[str, float], Any, float]:
        # Path 1: forced FEA (reset, or backend missing)
        if self._surrogate_backend is None or self._in_reset:
            responses, evaluation, fea_time = super()._run_fea_and_evaluate(design)
            self._last_response_source = "fea"
            self._last_drift_info = None
            return responses, evaluation, fea_time

        # Path 2: route through backend
        t0 = time.time()

        def fea_fallback(t_list, m_list):
            d = {
                "thickness": np.asarray(t_list, dtype=np.float64),
                "modulus":   np.asarray(m_list, dtype=np.float64),
                "poisson":   design["poisson"],
            }
            resp, _ev, _t = super(PavementEnvWithSurrogate,
                                    self)._run_fea_and_evaluate(d)
            return resp

        def eval_margins(responses_dict):
            inputs = self._build_design_inputs(design)
            ev = self.protocol.evaluate(inputs, responses_dict)
            return {k: float(v) for k, v in ev.margins.items()}

        try:
            result = self._surrogate_backend.get_responses(
                thickness=design["thickness"].tolist(),
                modulus=design["modulus"].tolist(),
                E_subgrade=float(self.config.E_subgrade),
                episode_step=int(self.episode_step),
                global_step=int(self.global_step),
                fea_fallback=fea_fallback,
                evaluate_protocol=eval_margins,
                pavement_type=str(self.config.pavement_type),   # ← v2 new
            )
        except Exception as e:
            logger.warning("Backend get_responses crashed → parent FEA: {}".format(e))
            responses, evaluation, fea_time = super()._run_fea_and_evaluate(design)
            self._last_response_source = "fea_hard_fallback"
            self._last_drift_info = None
            return responses, evaluation, fea_time

        responses = result["responses"]
        self._last_response_source = result["source"]
        self._last_drift_info = result["drift"]

        inputs = self._build_design_inputs(design)
        evaluation = self.protocol.evaluate(inputs, responses)
        return responses, evaluation, time.time() - t0

    # ─── Override _audit_step to add a 'routing' entry ─────────────
    def _audit_step(self, *args, **kwargs):
        try:
            super()._audit_step(*args, **kwargs)
        except Exception as e:
            logger.warning("Parent _audit_step failed: {}".format(e))

        if self.config.audit_chain is None:
            return
        try:
            self.config.audit_chain.record("routing", {
                "global_step":     int(self.global_step),
                "episode":         int(self.total_episodes),
                "episode_step":    int(self.episode_step),
                "response_source": str(self._last_response_source),
                "pavement_type":   str(self.config.pavement_type),
                "drift":           (self._last_drift_info
                                     if self._last_drift_info is not None else None),
            })
        except Exception as e:
            logger.warning("Routing audit record failed: {}".format(e))

    # ─── Helpers ────────────────────────────────────────────────────
    def _build_design_inputs(self, design: Dict[str, np.ndarray]):
        return super()._build_design_inputs(design)

    @property
    def last_response_source(self) -> str:
        return self._last_response_source

    @property
    def last_drift_info(self) -> Optional[Dict[str, float]]:
        return self._last_drift_info

    @property
    def backend_stats(self) -> Optional[Dict]:
        if self._surrogate_backend is None:
            return None
        return self._surrogate_backend.stats_summary()
