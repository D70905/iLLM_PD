# -*- coding: utf-8 -*-
"""
scripts/ltpp_inference.py 鈥?Inference-only run on 12 LTPP sections.

Reuses trained A v3 (flexible) and B (semi_rigid) PPO policies in a
deterministic, no-learning mode. For each (section, seed) pair, runs one
episode (20 steps) and logs design / margins / LCC / DSR / SCR_running to
disk. The output is the primary data substrate for the following reviewer
responses:

    R2-1  multi-section validation (12 LTPP, 4 climates 脳 2 GPS 脳 3 subgrade bins)
    R3-9  same (bundled by R2-1)
    R1-3  multi-seed reproducibility (>= 3 seeds per section; std reported)
    R3-15 reproducibility (deterministic policy + fixed seeds + audit chain)
    R3-11 ablation interaction (this script's outputs feed the ablation table)
    R1-2  long-term performance (LCC NPV reported per run via info dict)
    R2-2  100% vs 0.819 resolution (SCR + DSR per run, geomean / min)

Usage:
    # Dry-run: validate 12 sections + paths, no actual inference
    python -m scripts.ltpp_inference --dry-run \
        --policy-flex output/rl_runs/ppo_flexible_v3_1000ts_seed0_v3/checkpoints/ckpt_final_step_002048 \
        --policy-semi output/rl_runs/ppo_semi_rigid_v3_1000ts_seed0/checkpoints/ckpt_final_step_001024 \
        --sections experiments/ltpp_data/ltpp_12_sections_with_subgrade.xlsx \
        --surrogate-model-path output/surrogate_model/surrogate_v3.pt

    # Full quick scan (3 seeds): ~4.5 hours, 36 runs
    python -m scripts.ltpp_inference \
        --policy-flex ... --policy-semi ... \
        --sections ... \
        --surrogate-model-path output/surrogate_model/surrogate_v3.pt \
        --seeds 0,1,2 \
        --out-dir experiments/ltpp_data/deliverables/ltpp_inference

    # Single section / single seed (debugging)
    python -m scripts.ltpp_inference \
        --policy-flex ... --policy-semi ... \
        --sections ... \
        --surrogate-model-path output/surrogate_model/surrogate_v3.pt \
        --seeds 0 --only-section 48_0001
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ltpp_inference")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Per-type defaults (MUST mirror rl/train.py:_DEFAULT_INIT_BY_TYPE)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_DEFAULT_INIT_BY_TYPE: Dict[str, Dict[str, List[float]]] = {
    "semi_rigid": {
        "thickness": [0.04, 0.06, 0.08, 0.36, 0.18],
        "modulus":   [14000.0, 11000.0, 9000.0, 1500.0, 400.0],
        "poisson":   [0.25, 0.30, 0.30, 0.25, 0.35],
    },
    "flexible": {
        "thickness": [0.04, 0.06, 0.08, 0.30, 0.25],
        "modulus":   [14000.0, 11000.0, 9000.0, 350.0, 250.0],
        "poisson":   [0.25, 0.30, 0.30, 0.40, 0.35],
    },
}


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Section loading
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

# Tolerant column-name lookup (different naming conventions across CSV/XLSX)
_COL_ALIASES = {
    "section_id":   ["section_id", "id", "SHRP_ID", "shrp_id_str"],
    "state_code":   ["state_code", "STATE_CODE", "state"],
    "shrp_id":      ["shrp_id", "SHRP_ID"],
    "state_name":   ["state_name", "STATE_NAME"],
    "climate_zone": ["climate_zone", "climate", "CLIMATE_ZONE", "climate_zone_final"],
    "gps_family":   ["gps_family", "GPS_family", "gps", "inferred_gps_family"],
    "E_subgrade":   ["E_subgrade_MPa", "E_subgrade", "Esub_MPa", "subgrade_E_MPa", "E_sub_MPa"],
    "subgrade_bin": ["subgrade_bin", "subgrade_class", "bin"],
    "is_baseline":  ["is_baseline", "is_paper_baseline"],
}


# Hardcoded GPS family mapping (from SDR39 extracted experiment_section.csv).
# Used as fallback when the xlsx lacks a gps_family column.
# ACUB = Asphalt Concrete on Unbound Base = GPS-1 = flexible
# ACATB = Asphalt Concrete on Asphalt-Treated Base = GPS-2 = semi_rigid
_GPS_FAMILY_FALLBACK: Dict[str, str] = {
    "04_1034": "GPS-1", "12_1060": "GPS-1", "16_1010": "GPS-1",
    "27_1085": "GPS-1", "48_0001": "GPS-1", "48_1076": "GPS-1",
    "04_1065": "GPS-2", "06_2004": "GPS-2", "12_4097": "GPS-2",
    "27_2023": "GPS-2", "30_7076": "GPS-2", "48_1109": "GPS-2",
}


def _resolve_col(row_keys: List[str], aliases: List[str]) -> Optional[str]:
    """Find the first column name in `row_keys` that matches any alias."""
    row_keys_lower = {k.lower(): k for k in row_keys}
    for a in aliases:
        if a in row_keys:
            return a
        if a.lower() in row_keys_lower:
            return row_keys_lower[a.lower()]
    return None


def _normalize_gps(gps_raw: str) -> str:
    s = str(gps_raw).upper().strip()
    if "GPS-1" in s or "GPS1" in s or s == "1":
        return "GPS-1"
    if "GPS-2" in s or "GPS2" in s or s == "2":
        return "GPS-2"
    return s


def load_sections(xlsx_or_csv_path: Path) -> List[Dict[str, Any]]:
    """Load 12-section master file. Returns list of dicts with normalized keys."""
    p = Path(xlsx_or_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Sections file not found: {p}")

    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(p, sheet_name=0)
    else:
        df = pd.read_csv(p)

    cols = list(df.columns)
    logger.info("Loaded sections file: %s  (%d rows, %d columns)", p.name, len(df), len(cols))

    resolved: Dict[str, Optional[str]] = {}
    for canon, aliases in _COL_ALIASES.items():
        resolved[canon] = _resolve_col(cols, aliases)

    missing_required = [k for k in ("section_id", "E_subgrade", "gps_family")
                          if resolved[k] is None]
    if "gps_family" in missing_required and resolved["section_id"]:
        missing_required.remove("gps_family")
        logger.info("gps_family column not found; using hardcoded SDR39 fallback map")
    if missing_required:
        logger.error("Cannot find required columns: %s", missing_required)
        logger.error("Available columns: %s", cols)
        raise ValueError("Master file missing required columns; "
                         "edit _COL_ALIASES at top of script if column names differ")

    sections: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sid = str(row[resolved["section_id"]]).strip()
        try:
            esub = float(row[resolved["E_subgrade"]])
        except (ValueError, TypeError):
            logger.warning("Section %s has invalid E_subgrade, skipping", sid)
            continue
        if resolved["gps_family"]:
            gps = _normalize_gps(row[resolved["gps_family"]])
        else:
            gps = _GPS_FAMILY_FALLBACK.get(sid, "")
            if not gps:
                logger.warning("Section %s not in GPS fallback map, skipping", sid)
                continue
        if gps not in ("GPS-1", "GPS-2"):
            logger.warning("Section %s has unknown GPS family %r, skipping", sid, gps)
            continue

        sections.append({
            "section_id":   sid,
            "state_code":   str(row[resolved["state_code"]]) if resolved["state_code"] else "",
            "shrp_id":      str(row[resolved["shrp_id"]]) if resolved["shrp_id"] else "",
            "state_name":   str(row[resolved["state_name"]]) if resolved["state_name"] else "",
            "climate_zone": str(row[resolved["climate_zone"]]) if resolved["climate_zone"] else "",
            "gps_family":   gps,
            "E_subgrade":   esub,
            "subgrade_bin": str(row[resolved["subgrade_bin"]]) if resolved["subgrade_bin"] else "",
            "is_baseline":  bool(row[resolved["is_baseline"]]) if resolved["is_baseline"]
                              else (sid in ("48_0001", "48_1076")),
            # routing
            "pavement_type": "flexible" if gps == "GPS-1" else "semi_rigid",
        })

    if len(sections) == 0:
        raise ValueError("No valid sections loaded")
    logger.info("Valid sections: %d  (%d GPS-1, %d GPS-2)",
                  len(sections),
                  sum(1 for s in sections if s["gps_family"] == "GPS-1"),
                  sum(1 for s in sections if s["gps_family"] == "GPS-2"))
    return sections


def load_traffic_inputs(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load optional per-section LTPP traffic inputs for JTG N_e overrides."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Traffic input file not found: {p}")
    df = pd.read_csv(p)
    if "section_id" not in df.columns:
        raise ValueError("Traffic input file must contain a section_id column")

    numeric_cols = {
        "annual_ESAL_BZZ100",
        "total_ESAL_BZZ100",
        "traffic_growth_rate",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = str(row["section_id"]).strip()
        if not sid:
            continue
        item: Dict[str, Any] = {}
        for col in df.columns:
            if col == "section_id":
                continue
            value = row[col]
            if pd.isna(value):
                continue
            if col in numeric_cols:
                try:
                    item[col] = float(value)
                except (TypeError, ValueError):
                    continue
            else:
                item[col] = str(value)
        out[sid] = item
    logger.info("Loaded traffic inputs for %d sections from %s", len(out), p)
    return out


def attach_traffic_inputs(sections: List[Dict[str, Any]],
                          traffic_by_section: Dict[str, Dict[str, Any]]
                          ) -> List[Dict[str, Any]]:
    """Merge traffic fields into loaded section dictionaries."""
    attached = 0
    for section in sections:
        traffic = traffic_by_section.get(section["section_id"])
        if not traffic:
            continue
        section.update(traffic)
        attached += 1
    missing = [s["section_id"] for s in sections
               if s["section_id"] not in traffic_by_section]
    if missing:
        logger.warning("No traffic override for sections: %s", ", ".join(missing))
    logger.info("Attached traffic overrides to %d/%d sections", attached, len(sections))
    return sections

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Per-section climate 鈫?JTG D50-2017 climate_zone mapping
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# JTG Appendix G climate zones (annual mean air temperature):
#   cold:    <  5 掳C   鈫?kT_asphalt=0.85,  T_味=10 掳C
#   temperate: 5鈥?5 掳C 鈫?kT_asphalt=1.15,  T_味=17 掳C
#   warm:    15鈥?0 掳C  鈫?kT_asphalt=1.40,  T_味=22 掳C
#   hot:     20鈥?5 掳C  鈫?kT_asphalt=1.65,  T_味=26 掳C
#   tropical: > 25 掳C  鈫?kT_asphalt=1.85,  T_味=29 掳C
#
# MAAT from batch_climate_12sections_summary.csv (real LTPP monthly data).
# Because LTPP sections are in the USA (not China), we use the JTG climate_zone
# fallback table rather than a Chinese city lookup. city="" forces fallback.

_SECTION_MAAT_TO_JTG_ZONE: Dict[str, str] = {
    # MAAT from batch_climate_12sections_summary.csv (Cold鈫扝ot ordering)
    "27_2023": "temperate",    # MAAT  6.62 掳C
    "16_1010": "temperate",    # MAAT  6.70 掳C
    "27_1085": "temperate",    # MAAT  6.70 掳C
    "30_7076": "temperate",    # MAAT  7.48 掳C
    "04_1065": "temperate",    # MAAT 11.15 掳C
    "48_1076": "warm",         # MAAT 15.78 掳C
    "06_2004": "warm",         # MAAT 18.03 掳C
    "12_4097": "warm",         # MAAT 19.25 掳C
    "48_1109": "warm",         # MAAT 19.75 掳C
    "48_0001": "hot",          # MAAT 20.58 掳C
    "04_1034": "hot",          # MAAT 23.29 掳C
    "12_1060": "hot",          # MAAT 24.59 掳C
    # NCAT Test Track 2015-2021 Cracking Group (Opelika, AL, MAAT ~17.6 掳C)
    "NCAT_CG_N1": "warm", "NCAT_CG_N2": "warm", "NCAT_CG_N5": "warm",
    "NCAT_CG_N8": "warm", "NCAT_CG_S5": "warm", "NCAT_CG_S6": "warm",
    "NCAT_CG_S13": "warm",
}


def _get_jtg_climate_zone(section_id: str) -> str:
    """Return the JTG D50-2017 climate zone for an LTPP section.

    Uses MAAT-based mapping (see _SECTION_MAAT_TO_JTG_ZONE).
    Falls back to 'temperate' for unknown sections.
    """
    zone = _SECTION_MAAT_TO_JTG_ZONE.get(section_id, "")
    if zone:
        return zone
    # Fallback: try to infer from LTPP climate_zone field
    logger.warning("No JTG climate_zone mapping for %s; defaulting to 'temperate'", section_id)
    return "temperate"


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Environment construction (mirrors train.py:build_env_config but minimal)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def make_env_config(section: Dict[str, Any],
                     surrogate_model_path: Path,
                     b3_threshold: float = 1.0,
                     fea_validation_every: int = 10,
                     max_episode_steps: int = 20,
                     enable_lcc_eval: bool = True,
                     design_life_years_lcc: float = 20.0,
                     llm_enabled: bool = False,
                     generator: Any = None,
                     audit_chain: Any = None):
    """Build a SurrogateEnvConfig (or EnvConfig if surrogate missing) for this section."""
    from rl.env_surrogate import SurrogateEnvConfig

    pt = section["pavement_type"]
    init = _DEFAULT_INIT_BY_TYPE[pt]
    jtg_zone = _get_jtg_climate_zone(section["section_id"])

    return SurrogateEnvConfig(
        protocol_name="JTG_D50_2017",
        init_thickness_m=list(init["thickness"]),
        init_modulus_MPa=list(init["modulus"]),
        init_poisson=list(init["poisson"]),
        E_subgrade=float(section["E_subgrade"]),
        nu_subgrade=0.40,
        load_pressure_MPa=0.7,
        load_radius_m=0.1065,
        city="",                     # 鈽?disable Chinese-city lookup 鈫?use climate_zone
        climate_zone=jtg_zone,       # 鈽?per-section JTG zone from MAAT mapping
        road_class="expressway",
        traffic_level="heavy",
        annual_ESAL_BZZ100=section.get("annual_ESAL_BZZ100"),
        total_ESAL_BZZ100=section.get("total_ESAL_BZZ100"),
        traffic_growth_rate=float(section.get("traffic_growth_rate", 0.0) or 0.0),
        pavement_type=pt,
        design_life_years=15,
        max_episode_steps=max_episode_steps,
        max_episodes=1,            # inference: 1 episode per (section, seed)
        fea_keep_runs=False,
        llm_enabled=False,         # 鈫?inference: disable LLM (deterministic + fast)
        log_every_n_steps=5,
        # LCC/DSR/SCR post-eval (Patch from prev step)
        enable_lcc_eval=enable_lcc_eval,
        design_life_years_lcc=float(design_life_years_lcc),
        # Surrogate
        use_surrogate=True,
        surrogate_model_path=str(surrogate_model_path),
        fea_validation_every=int(fea_validation_every),
        surrogate_b3_threshold=float(b3_threshold),
    )


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Policy loading
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def find_policy_zip(checkpoint_path: Path) -> Path:
    """SB3 PPO checkpoint may be a .zip or a directory containing a .zip.
    Locate the actual policy file."""
    p = Path(checkpoint_path)
    if p.is_file() and p.suffix == ".zip":
        return p
    if p.is_dir():
        # train.py saves checkpoints as folders; look for model.zip / ppo_model.zip / *.zip inside
        for name in ("model.zip", "ppo_model.zip", "policy.zip"):
            cand = p / name
            if cand.exists():
                return cand
        # fall back: any *.zip in the dir
        zips = list(p.glob("*.zip"))
        if zips:
            return zips[0]
    # also try appending .zip (SB3 default)
    cand2 = p.with_suffix(".zip")
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Could not locate PPO policy zip in/under: {checkpoint_path}")


def load_policy(checkpoint_path: Path):
    """Return SB3 PPO instance ready for deterministic predict()."""
    from stable_baselines3 import PPO
    zip_path = find_policy_zip(checkpoint_path)
    logger.info("Loading policy from: %s", zip_path)
    # device='cpu' is fine for inference; avoids GPU memory contention
    model = PPO.load(str(zip_path), device="cpu")
    return model


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Per-run inference
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def run_one_inference(section: Dict[str, Any],
                       seed: int,
                       policy,
                       surrogate_model_path: Path,
                       out_jsonl: Path,
                       max_steps: int = 20,
                       b3_threshold: float = 1.0,
                       enable_llm: bool = False,
                       alpha_initial: float = 0.5,
                       alpha_decay: str = "linear_to_zero",
                       alpha_fallback: float = 0.0,
                       llm_temperature: float = 0.0,
                       gen_backend: str = "deepseek") -> Dict[str, Any]:
    """Run ONE episode (1 reset + up to max_steps steps), deterministic.
    Append step-by-step records to out_jsonl. Return run summary dict."""
    from rl.env_surrogate import PavementEnvWithSurrogate

    generator = None
    audit_chain = None
    if enable_llm:
        audit_dir = out_jsonl.parent / "audit" / f"{section['section_id']}_seed{seed}"
        generator, audit_chain = build_inference_generator(
            audit_dir=audit_dir, alpha_initial=alpha_initial,
            alpha_decay=alpha_decay, alpha_fallback=alpha_fallback,
            temperature=llm_temperature, backend=gen_backend)

    cfg = make_env_config(section, surrogate_model_path,
                            b3_threshold=b3_threshold,
                            max_episode_steps=max_steps,
                            llm_enabled=enable_llm,
                            generator=generator,
                            audit_chain=audit_chain)
    env = PavementEnvWithSurrogate(cfg)

    t_start = time.time()
    obs, info_reset = env.reset(seed=int(seed))
    steps_log: List[Dict[str, Any]] = []
    last_info: Dict[str, Any] = info_reset
    last_design = None
    total_reward = 0.0
    n_compliant = int(bool(info_reset.get("compliant", False)))
    n_steps = 0
    states: List[Dict[str, Any]] = []
    if bool(info_reset.get("compliant", False)):
        states.append({
            "step": 0,
            "dsr": _safe_float(info_reset.get("dsr")),
            "cost_cny": _safe_float((info_reset.get("lcc") or {}).get("C_construction_cny_per_m2")),
            "cost_usd": _safe_float((info_reset.get("lcc") or {}).get("C_construction_usd_per_m2")),
            "lcc_usd": _safe_float((info_reset.get("lcc") or {}).get("NPV_total_usd_m2")),
            "margins": (info_reset.get("evaluation") or {}).get("margins", {}),
            "compliant": True,
        })

    # Log reset state
    reset_record = {
        "phase":         "reset",
        "section_id":    section["section_id"],
        "seed":          int(seed),
        "step":          0,
        "pavement_type": section["pavement_type"],
        "E_subgrade":    section["E_subgrade"],
        "reward":        None,
        "dsr":           info_reset.get("dsr"),
        "scr_running":   info_reset.get("scr_running"),
        "compliant":     info_reset.get("compliant"),
        "lcc":           info_reset.get("lcc"),
        "evaluation":    info_reset.get("evaluation"),
        "response_source": info_reset.get("response_source"),
    }
    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(reset_record, default=_jsonable) + "\n")

    # Step loop
    for t in range(max_steps):
        try:
            action, _ = policy.predict(obs, deterministic=True)
        except Exception as e:
            logger.error("predict() failed at step %d: %s", t + 1, e)
            break

        try:
            step_out = env.step(action)
        except Exception as e:
            logger.error("env.step() crashed at step %d: %s", t + 1, e)
            break

        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated = bool(done); truncated = False

        n_steps += 1
        total_reward += float(reward)
        if bool(info.get("compliant", False)):
            n_compliant += 1
            states.append({
                "step": int(t + 1),
                "dsr": _safe_float(info.get("dsr")),
                "cost_cny": _safe_float((info.get("lcc") or {}).get("C_construction_cny_per_m2")),
                "cost_usd": _safe_float((info.get("lcc") or {}).get("C_construction_usd_per_m2")),
                "lcc_usd": _safe_float((info.get("lcc") or {}).get("NPV_total_usd_m2")),
                "margins": (info.get("evaluation") or {}).get("margins", {}),
                "compliant": True,
            })
        last_info = info
        last_design = info.get("new_design") or info.get("design")

        rec = {
            "phase":         "step",
            "section_id":    section["section_id"],
            "seed":          int(seed),
            "step":          int(t + 1),
            "reward":        float(reward),
            "dsr":           info.get("dsr"),
            "scr_running":   info.get("scr_running"),
            "compliant":     info.get("compliant"),
            "lcc":           info.get("lcc"),
            "evaluation":    info.get("evaluation"),
            "response_source": info.get("response_source"),
            "action":        action.tolist() if hasattr(action, "tolist") else list(action),
        }
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

        if terminated or truncated:
            logger.info("  [%s seed=%d] terminated/truncated at step %d", section["section_id"], seed, t + 1)
            break

    elapsed = time.time() - t_start
    delivered = None
    if states:
        delivered = min(
            states,
            key=lambda x: (
                float("inf") if x.get("lcc_usd") is None else float(x["lcc_usd"]),
                float("inf") if x.get("cost_usd") is None else float(x["cost_usd"]),
                int(x.get("step") or 0),
            ),
        )
    else:
        delivered = {
            "step": n_steps,
            "dsr": _safe_float(last_info.get("dsr")),
            "cost_cny": _safe_float((last_info.get("lcc") or {}).get("C_construction_cny_per_m2")),
            "cost_usd": _safe_float((last_info.get("lcc") or {}).get("C_construction_usd_per_m2")),
            "lcc_usd": _safe_float((last_info.get("lcc") or {}).get("NPV_total_usd_m2")),
            "margins": (last_info.get("evaluation") or {}).get("margins", {}),
        }

    # 鈹€鈹€ Summary 鈹€鈹€
    summary = {
        "section_id":    section["section_id"],
        "state_name":    section["state_name"],
        "climate_zone":  section["climate_zone"],
        "gps_family":    section["gps_family"],
        "subgrade_bin":  section["subgrade_bin"],
        "E_subgrade":    section["E_subgrade"],
        "is_baseline":   section["is_baseline"],
        "pavement_type": section["pavement_type"],
        "traffic_source": section.get("traffic_source", ""),
        "annual_ESAL_BZZ100": _safe_float(section.get("annual_ESAL_BZZ100")),
        "total_ESAL_BZZ100": _safe_float(section.get("total_ESAL_BZZ100")),
        "traffic_growth_rate": _safe_float(section.get("traffic_growth_rate")),
        "seed":          int(seed),
        "n_steps":       int(n_steps),
        "total_reward":  float(total_reward),
        "mean_reward":   float(total_reward / max(n_steps, 1)),
        "final_dsr":     _safe_float(last_info.get("dsr")),
        "final_scr_running": _safe_float(last_info.get("scr_running")),
        "n_compliant":   int(n_compliant),
        "compliance_rate_in_episode":
                          float(n_compliant / max(n_steps + 1, 1)),  # +1 for reset
        "final_lcc_npv_usd_m2":
                          _safe_float((last_info.get("lcc") or {}).get("NPV_total_usd_m2")),
        "final_C_const_usd_m2":
                          _safe_float((last_info.get("lcc") or {}).get("C_construction_usd_per_m2")),
        "final_C_const_cny_m2":
                          _safe_float((last_info.get("lcc") or {}).get("C_construction_cny_per_m2")),
        "final_n_maint_events":
                          (last_info.get("lcc") or {}).get("n_maint_events"),
        "delivered_dsr": _safe_float(delivered.get("dsr")),
        "delivered_step": delivered.get("step"),
        "delivered_C_const_cny_m2": _safe_float(delivered.get("cost_cny")),
        "delivered_C_const_usd_m2": _safe_float(delivered.get("cost_usd")),
        "delivered_lcc_npv_usd_m2": _safe_float(delivered.get("lcc_usd")),
        "delivered_margins": json.dumps(delivered.get("margins", {}), default=_jsonable),
        "llm_enabled": bool(enable_llm),
        "alpha_initial": float(alpha_initial) if enable_llm else 0.0,
        "alpha_fallback": float(alpha_fallback) if enable_llm else 0.0,
        "wall_clock_sec": float(elapsed),
        "status":        "ok",
    }
    return summary


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# JSON helper
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Main driver
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    parser = argparse.ArgumentParser(
        description="LTPP 12-section 脳 N-seed inference with A v3 + B policies")
    parser.add_argument("--policy-flex", type=Path, required=True,
                          help="Path to PPO A v3 (flexible) checkpoint dir or .zip")
    parser.add_argument("--policy-semi", type=Path, required=True,
                          help="Path to PPO B (semi_rigid) checkpoint dir or .zip")
    parser.add_argument("--sections", type=Path, required=True,
                          help="Path to 12-section master xlsx/csv (must have section_id, "
                                "E_subgrade_MPa, gps_family columns)")
    parser.add_argument("--surrogate-model-path", type=Path, required=True,
                          help="Path to surrogate_v3.pt")
    parser.add_argument("--traffic-inputs", type=Path, default=None,
                          help="Optional CSV with section_id and annual_ESAL_BZZ100 or total_ESAL_BZZ100 columns")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                          help="Comma-separated seeds, e.g. '0,1,2' (default: 0,1,2)")
    parser.add_argument("--max-steps", type=int, default=20,
                          help="Max steps per episode (default: 20)")
    parser.add_argument("--b3-threshold", type=float, default=1.0,
                          help="Surrogate B3 escalation threshold (default: 1.0)")
    parser.add_argument("--out-dir", type=Path,
                          default=Path("experiments/ltpp_data/deliverables/ltpp_inference"),
                          help="Output directory (default: experiments/.../ltpp_inference)")
    parser.add_argument("--only-section", type=str, default=None,
                          help="If set, run only this section_id (e.g. '48_0001')")
    parser.add_argument("--dry-run", action="store_true",
                          help="Print 12-section routing table and exit (no actual inference)")
    parser.add_argument("--enable-llm", action="store_true",
                          help="Enable canonical full-system LLM-PPO fusion during inference.")
    parser.add_argument("--gen-backend", type=str, default="deepseek",
                        choices=["deepseek", "chatfire", "siliconflow-qwen", "siliconflow-glm", "ollama", "ollama-llama"],
                        help="Generator LLM backend for full-system inference.")
    parser.add_argument("--gen-alpha-initial", type=float, default=0.5,
                          help="Generator alpha_initial for full-system/alpha sensitivity inference.")
    parser.add_argument("--gen-alpha-decay", type=str, default="linear_to_zero",
                          choices=["linear_to_zero", "cosine", "constant"],
                          help="Generator alpha schedule.")
    parser.add_argument("--gen-alpha-fallback", type=float, default=0.0,
                          help="Infeasible-state alpha fallback; canonical value is 0.0.")
    parser.add_argument("--llm-temperature", type=float, default=0.0,
                          help="Generator temperature for reproducible inference.")
    args = parser.parse_args()

    # Parse seeds
    seeds: List[int] = []
    for s in args.seeds.split(","):
        s = s.strip()
        if s:
            seeds.append(int(s))
    if not seeds:
        logger.error("No valid seeds parsed from --seeds %r", args.seeds)
        sys.exit(1)

    # Load sections
    sections = load_sections(args.sections)
    if args.traffic_inputs:
        sections = attach_traffic_inputs(sections, load_traffic_inputs(args.traffic_inputs))
    if args.only_section:
        sections = [s for s in sections if s["section_id"] == args.only_section]
        if not sections:
            logger.error("Section %s not found in master file", args.only_section)
            sys.exit(1)

    # Print routing table
    logger.info("=" * 78)
    logger.info("LTPP inference plan: %d sections 脳 %d seeds = %d runs",
                  len(sections), len(seeds), len(sections) * len(seeds))
    logger.info("=" * 78)
    logger.info("%-12s %-6s %-6s %-12s %-8s %-10s %-10s %-18s",
                  "section_id", "state", "GPS", "climate", "E_sub", "pavtype", "baseline", "traffic")
    logger.info("-" * 78)
    for s in sections:
        logger.info("%-12s %-6s %-6s %-12s %-8.1f %-10s %-10s %-18s",
                      s["section_id"], s["state_name"][:6], s["gps_family"],
                      s["climate_zone"][:12], s["E_subgrade"],
                      s["pavement_type"], "yes" if s["is_baseline"] else "",
                      str(s.get("traffic_source", ""))[:18])
    logger.info("=" * 78)
    logger.info("Seeds: %s", seeds)
    logger.info("Policy (flex): %s", args.policy_flex)
    logger.info("Policy (semi): %s", args.policy_semi)
    logger.info("Surrogate v3:  %s", args.surrogate_model_path)
    logger.info("Output dir:    %s", args.out_dir)

    # Sanity: surrogate exists
    if not args.surrogate_model_path.exists():
        logger.error("Surrogate model not found: %s", args.surrogate_model_path)
        sys.exit(1)

    if args.dry_run:
        logger.info("[Dry-run] exiting before any inference.")
        return

    # Resolve policy zips early (fail fast)
    try:
        flex_zip = find_policy_zip(args.policy_flex)
        semi_zip = find_policy_zip(args.policy_semi)
        logger.info("Resolved A v3 zip: %s", flex_zip)
        logger.info("Resolved B     zip: %s", semi_zip)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_jsonl = args.out_dir / f"ltpp_inference_steps_{run_id}.jsonl"
    out_csv   = args.out_dir / f"ltpp_inference_summary_{run_id}.csv"
    logger.info("Step-level JSONL: %s", out_jsonl)
    logger.info("Per-run CSV:      %s", out_csv)

    # Load both policies ONCE
    logger.info("Loading policies...")
    policy_flex = load_policy(args.policy_flex)
    policy_semi = load_policy(args.policy_semi)
    logger.info("Policies loaded OK.")

    # Run loop
    summaries: List[Dict[str, Any]] = []
    n_total = len(sections) * len(seeds)
    n_done = 0
    t_overall_start = time.time()

    for section in sections:
        policy = policy_flex if section["pavement_type"] == "flexible" else policy_semi
        for seed in seeds:
            n_done += 1
            label = f"[{n_done}/{n_total}] {section['section_id']} ({section['pavement_type']}) seed={seed}"
            logger.info("=" * 78)
            logger.info(label)
            logger.info("=" * 78)
            try:
                summ = run_one_inference(
                    section=section,
                    seed=seed,
                    policy=policy,
                    surrogate_model_path=args.surrogate_model_path,
                    out_jsonl=out_jsonl,
                    max_steps=args.max_steps,
                    b3_threshold=args.b3_threshold,
                    enable_llm=args.enable_llm,
                    alpha_initial=args.gen_alpha_initial,
                    alpha_decay=args.gen_alpha_decay,
                    alpha_fallback=args.gen_alpha_fallback,
                    llm_temperature=args.llm_temperature,
                    gen_backend=args.gen_backend,
                )
                summaries.append(summ)
                logger.info("  鈫?DSR=%s  SCR_run=%s  NPV=%s USD/m虏  reward=%.3f  (%.1fs)",
                              _fmt(summ["final_dsr"]),
                              _fmt(summ["final_scr_running"]),
                              _fmt(summ["final_lcc_npv_usd_m2"]),
                              summ["total_reward"], summ["wall_clock_sec"])
            except Exception as e:
                logger.error("Run FAILED: %s", e)
                logger.error(traceback.format_exc())
                summaries.append({
                    "section_id":  section["section_id"],
                    "gps_family":  section["gps_family"],
                    "pavement_type": section["pavement_type"],
                    "seed":        int(seed),
                    "status":      "failed",
                    "error":       str(e),
                })

            # 鈹€鈹€ Save summary CSV incrementally (so partial run is still useful) 鈹€鈹€
            _write_summary_csv(summaries, out_csv)

    elapsed_total = time.time() - t_overall_start
    n_ok = sum(1 for s in summaries if s.get("status") == "ok")
    n_fail = sum(1 for s in summaries if s.get("status") == "failed")
    logger.info("=" * 78)
    logger.info("DONE.  ok=%d  failed=%d  total_wall_clock=%.1f min",
                  n_ok, n_fail, elapsed_total / 60.0)
    logger.info("Step JSONL: %s", out_jsonl)
    logger.info("Summary CSV: %s", out_csv)


def _fmt(v) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.3f}"
    except (ValueError, TypeError):
        return str(v)


def _write_summary_csv(summaries: List[Dict[str, Any]], path: Path) -> None:
    """Write/overwrite summary CSV with current results. Uses first-row keys as schema."""
    if not summaries:
        return
    # Build union of keys to handle ok/failed schema difference
    all_keys: List[str] = []
    seen = set()
    for s in summaries:
        for k in s.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in all_keys})


if __name__ == "__main__":
    main()







