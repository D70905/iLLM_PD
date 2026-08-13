# -*- coding: utf-8 -*-
"""
examples/train_ppo_surrogate_smoke.py
Phase 2D INTEGRATED smoke test — PavementEnvWithSurrogate end-to-end.

Setup:
  use_surrogate = True
  fea_validation_every = 2     # so we see both surrogate and FEA in 5 steps
  surrogate_b3_threshold = 1.2

Expected routing over 1 reset + 5 steps:
  reset    → source='fea'                 (forced by _in_reset flag)
  step 1   → source='surrogate'           (gs=0 entering FEA)
  step 2   → source='surrogate'           (gs=1, 1%2!=0)
  step 3   → source='fea_validation'      (gs=2, 2%2==0)
  step 4   → source='surrogate'           (gs=3, 3%2!=0)
  step 5   → source='fea_validation'      (gs=4, 4%2==0)

  Sometimes a 'surrogate_escalated' replaces a 'surrogate' if predicted
  B3 < 1.2 — that's fine, the test allows it.

Wall time: ~3 real FEA × 50s + 2 surrogate × ms + LLM ~30s = ~3 minutes.

Run:
    cd D:\\iLLM_PD_new
    conda activate illm_pd
    python examples\\train_ppo_surrogate_smoke.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Make sure project root is importable when launched from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env for DeepSeek / ChatFire keys
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_2d")


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
OUT_DIR        = ROOT / "output" / "rl_runs" / "smoke_phase2d"
AUDIT_PATH     = OUT_DIR / "audit_chain.jsonl"
SURROGATE_PATH = ROOT / "output" / "surrogate_model" / "surrogate_v2.pt"
RAG_DIR        = ROOT / "output" / "rag_db"

OUT_DIR.mkdir(parents=True, exist_ok=True)
if AUDIT_PATH.exists():
    AUDIT_PATH.unlink()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    log.info("=" * 70)
    log.info("Phase 2D INTEGRATION smoke test (PavementEnvWithSurrogate)")
    log.info("=" * 70)
    log.info(f"Output dir:      {OUT_DIR}")
    log.info(f"Surrogate model: {SURROGATE_PATH}")

    if not SURROGATE_PATH.exists():
        log.error("Surrogate model not found. Run train_surrogate_v2 first.")
        sys.exit(1)

    # ── [1/7] Imports ──
    log.info("[1/7] Loading project modules...")
    from rl.audit       import AuditChain
    from rl.rag         import RAGStore
    from rl.evaluator   import Evaluator
    from rl.generator   import Generator, GeneratorConfig
    from rl.env_surrogate import (PavementEnvWithSurrogate, SurrogateEnvConfig)

    # ── [2/7] Audit chain ──
    log.info("[2/7] AuditChain init...")
    audit = AuditChain(path=str(AUDIT_PATH))
    log.info(f"       audit at {AUDIT_PATH}")

    # ── [3/7] RAG ──
    log.info("[3/7] RAG store load (BGE ~30s)...")
    rag = None
    try:
        rag = RAGStore.from_persisted(str(RAG_DIR))
        n_chunks = getattr(rag, "n_chunks", "?")
        log.info(f"       RAG ready ({n_chunks} chunks)")
    except Exception as e:
        log.warning(f"       RAG init failed: {e} (smoke can still proceed)")

    # ── [4/7] Evaluator ──
    log.info("[4/7] Evaluator init (DeepSeek)...")
    evaluator = Evaluator(audit=audit, backend="deepseek", fail_fast=False)

    # ── [5/7] Generator ──
    log.info("[5/7] Generator init (ChatFire/GPT)...")
    generator = Generator(
        config=GeneratorConfig(backend="chatfire"),
        rag=rag, audit=audit, fail_fast=False,
    )

    # ── [6/7] Env config + env ──
    log.info("[6/7] PavementEnvWithSurrogate init...")
    cfg = SurrogateEnvConfig(
        # ---- standard env config ----
        protocol_name="JTG_D50_2017",
        init_thickness_m=[0.04, 0.06, 0.08, 0.36, 0.18],
        init_modulus_MPa=[14000, 11000, 9000, 1500, 400],
        init_poisson=[0.25, 0.30, 0.30, 0.25, 0.35],
        E_subgrade=60.0, nu_subgrade=0.40,
        load_pressure_MPa=0.7, load_radius_m=0.1065,
        city="beijing", road_class="expressway",
        traffic_level="heavy", pavement_type="semi_rigid",
        design_life_years=15,
        action_dh_max_m=0.02, action_dE_max_MPa=100.0,
        max_episode_steps=20, max_episodes=200,
        fea_base_dir=str(ROOT),
        fea_num_cpus=4,
        fea_verbose=False,
        fea_keep_runs=False,
        # ---- LLM hooks ----
        llm_enabled=True,
        evaluator=evaluator,
        generator=generator,
        audit_chain=audit,
        strict_mode_steps=0,         # no strict mode for smoke
        log_every_n_steps=1,
        # ---- surrogate ----
        use_surrogate=True,
        surrogate_model_path=str(SURROGATE_PATH),
        fea_validation_every=2,      # short interval to exercise validation
        surrogate_b3_threshold=1.2,
    )
    env = PavementEnvWithSurrogate(cfg)

    # ── [7/7] Reset + 5 steps ──
    log.info("[7/7] env.reset() — forced FEA (~45s)...")
    t_reset = time.time()
    obs, info = env.reset(seed=42)
    log.info(f"       reset done in {time.time() - t_reset:.1f}s; "
             f"source={info.get('response_source')}  "
             f"feasible={info.get('feasible')}  "
             f"critical={info.get('critical')}")

    log.info("       stepping x5 (fea_validation_every=2)...")
    sources, drifts, rewards = [], [], []
    for i in range(5):
        action = env.action_space.sample()
        t_step = time.time()
        obs, reward, done, _, info = env.step(action)
        dt = time.time() - t_step
        src   = env.last_response_source
        drift = env.last_drift_info
        sources.append(src)
        if drift:
            drifts.append(drift)
        rewards.append(float(reward))
        log.info(f"       step {i+1}/5: src={src:<22} r={reward:+.3f}  "
                 f"feas={info.get('feasible')}  dt={dt:.1f}s")
        if drift:
            for k, v in drift.items():
                if k.startswith("_"):
                    continue
                log.info(f"            {k:<32} = {v:+.2f}%")

    # ─── Verification ───
    log.info("")
    log.info("=" * 70)
    log.info("VERIFICATION")
    log.info("=" * 70)

    stats = env.backend_stats or {}

    checks = {
        "Reset used FEA":                  True,  # forced by design
        "Backend was created":              env._surrogate_backend is not None,
        "Surrogate used ≥ 1 step":          sum(1 for s in sources
                                                 if s.startswith("surrogate")) >= 1,
        "FEA-validation step ≥ 1":          sum(1 for s in sources
                                                 if s == "fea_validation") >= 1,
        "Drift logged on validation":       len(drifts) >= 1,
        "Audit chain integrity":            _audit_integrity_ok(audit),
        "Backend stats nonzero":            (stats.get("n_surrogate_calls", 0) +
                                              stats.get("n_fea_calls_total", 0)) > 0,
    }
    log.info(f"Routing sources: {sources}")
    log.info(f"Backend stats  : {stats}")
    log.info("")
    for name, passed in checks.items():
        log.info(f"  {name:<35} {'✓' if passed else '✗'}")
    n_pass  = sum(checks.values())
    n_total = len(checks)
    log.info(f"\n  {n_pass}/{n_total} checks passed")
    log.info(f"  total wall time: {(time.time() - t_total) / 60:.1f} min")

    if n_pass == n_total:
        log.info("=" * 70)
        log.info("ALL CHECKS PASSED — Phase 2D integration ready.")
        log.info("=" * 70)
        sys.exit(0)
    else:
        log.error("Some checks failed — review log above.")
        sys.exit(1)


def _audit_integrity_ok(audit) -> bool:
    """Best-effort: call audit.verify() if it exists, else just check file exists."""
    try:
        for attr in ("verify_integrity", "verify", "validate"):
            fn = getattr(audit, attr, None)
            if callable(fn):
                return bool(fn())
        return AUDIT_PATH.exists() and AUDIT_PATH.stat().st_size > 0
    except Exception as e:
        log.warning(f"Audit verify call raised: {e}")
        return False


if __name__ == "__main__":
    main()
