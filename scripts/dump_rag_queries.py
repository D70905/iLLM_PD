# -*- coding: utf-8 -*-
"""
scripts/dump_rag_queries.py
===========================
Quick inference with LLM enabled to dump real RAG queries + retrieval results.
Runs 1 episode per section (20 steps), captures [RAG_DUMP] log lines.

Usage:
    conda activate illm_pd
    cd /d <PROJECT_ROOT>
    set PYTHONPATH=.
    set HF_HUB_OFFLINE=1
    python scripts/dump_rag_queries.py
"""
from __future__ import annotations
import logging, os, sys, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dump_rag")

import numpy as np

def main():
    # ---- Load policies ----
    from stable_baselines3 import PPO
    flex_ckpt = "output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048/ppo_model.zip"
    semi_ckpt = "output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024/ppo_model.zip"
    surrogate = "output/surrogate_model/surrogate_v3.pt"

    policy_flex = PPO.load(flex_ckpt)
    policy_semi = PPO.load(semi_ckpt)
    logger.info("Policies loaded OK")

    # ---- Sections to test ----
    # JTG climate_zone mapping from ltpp_inference.py
    SECTION_JTG_ZONE = {
        "04_1034": "hot", "16_1010": "temperate",
        "48_0001": "hot", "48_1076": "warm", "12_1060": "hot",
    }
    SECTIONS = [
        {"section_id": "04_1034", "E_subgrade": 91.3, "pavement_type": "flexible",
         "init_h": [0.04,0.06,0.08,0.30,0.25], "init_E": [14000,11000,9000,350,250],
         "init_nu": [0.25,0.30,0.30,0.35,0.35]},
        {"section_id": "16_1010", "E_subgrade": 77.8, "pavement_type": "flexible",
         "init_h": [0.04,0.06,0.08,0.30,0.25], "init_E": [14000,11000,9000,350,250],
         "init_nu": [0.25,0.30,0.30,0.35,0.35]},
    ]

    for sec in SECTIONS:
        sid = sec["section_id"]
        pt = sec["pavement_type"]
        jtg_zone = SECTION_JTG_ZONE.get(sid, "temperate")
        policy = policy_flex if pt == "flexible" else policy_semi

        logger.info("=" * 60)
        logger.info("Running %s (JTG=%s, LTPP climate via batch CSV)", sid, jtg_zone)

        # ---- Create RAG + Generator ----
        from rl.rag import RAGStore
        from rl.generator import Generator, GeneratorConfig
        rag = RAGStore()
        gen_cfg = GeneratorConfig()
        gen_cfg.use_rag = True
        gen_cfg.rag_top_k = 3
        generator = Generator(config=gen_cfg, rag=rag, fail_fast=False)
        logger.info("Generator+%d-chunk RAG ready", rag.count() if rag.enabled else 0)

        # ---- Build env with LLM enabled ----
        from rl.env_surrogate import PavementEnvWithSurrogate, SurrogateEnvConfig

        cfg = SurrogateEnvConfig(
            protocol_name="JTG_D50_2017",
            pavement_type=pt,
            init_thickness_m=list(sec["init_h"]),
            init_modulus_MPa=list(sec["init_E"]),
            init_poisson=list(sec["init_nu"]),
            E_subgrade=float(sec["E_subgrade"]),
            nu_subgrade=0.40,
            city="", climate_zone=jtg_zone,
            road_class="expressway", traffic_level="heavy",
            design_life_years=15, max_episode_steps=20,
            max_episodes=1, llm_enabled=True,  # ★ LLM ON
            generator=generator,  # ★ pass Generator with RAG
            fea_keep_runs=False,
            enable_lcc_eval=True, design_life_years_lcc=20.0,
            use_surrogate=True, surrogate_model_path=surrogate,
            surrogate_b3_threshold=1.0,
        )
        env = PavementEnvWithSurrogate(cfg)

        obs, info = env.reset(seed=0)
        for step in range(20):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()
        logger.info("Done %s\n", sid)

    logger.info("All done. Search above for [RAG_DUMP] lines to extract real queries + passages.")

if __name__ == "__main__":
    main()
