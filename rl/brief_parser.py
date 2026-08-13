# -*- coding: utf-8 -*-
"""
rl/brief_parser.py — Natural-language design-brief parser
==========================================================

Parses a free-text pavement-design brief into structured EnvConfig-compatible
fields.  The user enters a natural-language description (e.g. "北京地区高速公路,
重载交通, 软土地基, 预算约 300 元/m²") and the parser returns a dict that can
feed directly into EnvConfig or SurrogateEnvConfig.

Uses the Generator's LLM backend (GPT-4o-mini via ChatFire) with a structured
JSON output prompt.  This is the NLP input layer that the original MATLAB
iLLM-PD possessed and that the current Python system was missing.

Usage:
    from rl.brief_parser import parse_design_brief, BriefParseResult

    result = parse_design_brief(
        "上海郊区一级公路, 中等交通, 软粘土路基 E_sub≈40 MPa, 预算 350 元/m²",
    )
    print(result.city)          # 'shanghai'
    print(result.traffic_level) # 'medium'
    print(result.E_subgrade)    # 40.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rl.llm_client import get_client, LLMError, parse_json_from_text

logger = logging.getLogger("brief_parser")

# ── Allowed vocabulary (must match EnvConfig + jtg_d50.json) ──────────
KNOWN_CITIES = [
    "beijing", "shanghai", "guangzhou", "shenzhen", "chengdu",
    "wuhan", "nanjing", "hangzhou", "xian", "chongqing",
    "tianjin", "shenyang", "harbin", "kunming", "lanzhou",
    "urumqi", "lhasa", "guiyang", "nanning", "fuzhou",
    "zhengzhou", "jinan", "taiyuan", "changsha", "nanchang",
    "hefei", "changchun", "hohhot", "xining", "yinchuan",
    "haikou", "lasa", "shijiazhuang",
]
KNOWN_ROAD_CLASSES = ["expressway", "highway_1", "highway_2", "urban_trunk", "urban_branch"]
KNOWN_TRAFFIC_LEVELS = ["light", "medium", "heavy", "extra_heavy"]
KNOWN_PAVEMENT_TYPES = ["flexible", "semi_rigid"]


BRIEF_PARSER_SYSTEM = """You are a pavement-design specification translator.  Your job is to
read a natural-language engineering brief (Chinese or English) and map it
to structured, machine-readable parameters.  You must ONLY use the allowed
vocabularies listed below.  Do not invent values outside these lists.

Allowed city names (Pinyin lowercase):
{ALLOWED_CITIES}

Allowed road classes: {ALLOWED_ROAD_CLASSES}
Allowed traffic levels: {ALLOWED_TRAFFIC_LEVELS}
Allowed pavement types: {ALLOWED_TYPES}

Return ONLY a single JSON object with these keys:
  city:            string  — closest matching city from the allowed list
  road_class:      string  — from the allowed road classes
  traffic_level:   string  — from the allowed traffic levels
  pavement_type:   string  — "flexible" if granular/unbound base, "semi_rigid" if cement/CTB base
  E_subgrade:      number  — estimated subgrade resilient modulus in MPa (if the brief gives one; otherwise estimate from soil description: very soft ~20-40, soft ~40-80, medium ~80-150, stiff ~150-300, very stiff/rock >300)
  budget_cny:      number  — construction budget in CNY/m² if mentioned, otherwise 0 (unknown)
  design_life:     number  — design life in years (default 15 if not stated)
  confidence:      number  — 0-1 self-assessment of how well the brief maps to these parameters

Rules:
- If the brief mentions a city not in the list, choose the nearest large city.
- If the brief does not mention road class, default to "expressway".
- If the brief does not mention traffic, default to "heavy".
- Use your knowledge of Chinese pavement engineering practice to infer the pavement type from base-material descriptions.
"""

BRIEF_PARSER_USER = """Brief text:
{brief_text}

Return ONLY the JSON object."""


@dataclass
class BriefParseResult:
    """Structured output of brief parsing, compatible with EnvConfig fields."""

    city: str = "beijing"
    road_class: str = "expressway"
    traffic_level: str = "heavy"
    pavement_type: str = "flexible"
    E_subgrade: float = 60.0
    budget_cny: float = 0.0
    design_life: int = 15
    confidence: float = 0.5
    raw_brief: str = ""
    llm_raw: str = ""
    llm_model: str = ""
    elapsed_s: float = 0.0

    def as_env_kwargs(self) -> Dict:
        """Return the subset of fields that EnvConfig / SurrogateEnvConfig accepts."""
        return {
            "city": self.city,
            "road_class": self.road_class,
            "traffic_level": self.traffic_level,
            "pavement_type": self.pavement_type,
            "E_subgrade": self.E_subgrade,
            "design_life_years": self.design_life,
        }


def parse_design_brief(
    brief_text: str,
    backend: str = "chatfire",
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> BriefParseResult:
    """
    Parse a natural-language design brief into structured parameters.

    Args:
        brief_text: Free-text design description (Chinese or English).
        backend:    LLM backend name (default "chatfire" = GPT-4o-mini).
        model:      Override the default model for this backend.
        timeout:    LLM call timeout in seconds.

    Returns:
        BriefParseResult with structured fields.
    """
    client = get_client(backend)

    system_prompt = BRIEF_PARSER_SYSTEM.format(
        ALLOWED_CITIES=", ".join(KNOWN_CITIES),
        ALLOWED_ROAD_CLASSES=", ".join(KNOWN_ROAD_CLASSES),
        ALLOWED_TRAFFIC_LEVELS=", ".join(KNOWN_TRAFFIC_LEVELS),
        ALLOWED_TYPES=", ".join(KNOWN_PAVEMENT_TYPES),
    )
    user_prompt = BRIEF_PARSER_USER.format(brief_text=brief_text.strip())

    try:
        response = client.chat(
            system=system_prompt,
            user=user_prompt,
            model=model,
            temperature=0.0,  # deterministic: fixed seed for reproducibility
            max_tokens=800,                          # from 300: leave room for response
            timeout=timeout,
            response_format={'type': 'json_object'}, # GPT-4o-mini JSON mode
        )
    except LLMError as e:
        logger.warning("Brief-parser LLM call failed (%s); returning defaults", e.code)
        return BriefParseResult(raw_brief=brief_text, llm_raw=str(e), confidence=0.0)

    parsed = parse_json_from_text(response.text)
    result = BriefParseResult(
        raw_brief=brief_text,
        llm_raw=response.text,
        llm_model=response.model,
        elapsed_s=response.elapsed_s,
    )

    if parsed is None:
        logger.warning("Brief parser could not extract JSON; using defaults.")
        return result

    # ── Populate with validation ──────────────────────────────────────
    city = str(parsed.get("city", "")).strip().lower()
    result.city = city if city in KNOWN_CITIES else "beijing"

    rc = str(parsed.get("road_class", "")).strip().lower()
    result.road_class = rc if rc in KNOWN_ROAD_CLASSES else "expressway"

    tl = str(parsed.get("traffic_level", "")).strip().lower()
    result.traffic_level = tl if tl in KNOWN_TRAFFIC_LEVELS else "heavy"

    pt = str(parsed.get("pavement_type", "")).strip().lower()
    result.pavement_type = pt if pt in KNOWN_PAVEMENT_TYPES else "flexible"

    try:
        esub = float(parsed.get("E_subgrade", 60.0))
        result.E_subgrade = max(5.0, min(2000.0, esub))
    except (ValueError, TypeError):
        pass

    try:
        budget = float(parsed.get("budget_cny", 0.0))
        result.budget_cny = max(0.0, budget)
    except (ValueError, TypeError):
        pass

    try:
        dl = int(parsed.get("design_life", 15))
        result.design_life = max(5, min(50, dl))
    except (ValueError, TypeError):
        pass

    try:
        conf = float(parsed.get("confidence", 0.5))
        result.confidence = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        pass

    return result


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_briefs = [
        "北京地区高速公路, 重载交通, 软土地基 E_sub≈40 MPa, 预算约 300 元/m², 半刚性基层",
        "Guangzhou urban trunk with medium traffic, stiff subgrade ~180 MPa, flexible pavement",
        "Design a pavement for a heavy-traffic highway in Shanghai with cement-stabilised base",
    ]

    for i, brief in enumerate(test_briefs, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {brief[:80]}...")
        result = parse_design_brief(brief)
        print(f"  city:          {result.city}")
        print(f"  road_class:    {result.road_class}")
        print(f"  traffic_level: {result.traffic_level}")
        print(f"  pavement_type: {result.pavement_type}")
        print(f"  E_subgrade:    {result.E_subgrade} MPa")
        print(f"  budget:        {result.budget_cny} CNY/m²")
        print(f"  design_life:   {result.design_life} yr")
        print(f"  confidence:    {result.confidence:.2f}")
        print(f"  model:         {result.llm_model}")
        print(f"  elapsed:       {result.elapsed_s:.1f}s")
        print(f"  env_kwargs:    {result.as_env_kwargs()}")