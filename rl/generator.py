# -*- coding: utf-8 -*-
"""
rl.generator — LLM action proposer (Phase 2C)
================================================

Generator: GPT-4o-mini via ChatFire 中转.
Proposes a 10-dim action correction based on:
- Current 6-layer pavement state
- 4 JTG margins
- Critical indicator
- RAG-retrieved regulation context

Blending with PPO action:
    action_final = (1 - alpha) * action_PPO + alpha * action_generator
    where:
        alpha = alpha_base * generator_confidence
        alpha_base from training-progress schedule (decreases over time)
        generator_confidence self-reported by LLM in [0, 1]

Call schedule (avoids 4000 calls):
    early (tau < 0.3):  every step
    mid   (0.3 <= tau < 0.7):  every 2 steps
    late  (tau >= 0.7): every 5 steps

Total expected LLM calls for 4000-ts training: ~2240 (vs 4000 if every step).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rl.audit import AuditChain
from rl.llm_client import LLMClient, LLMError, get_client, parse_json_from_text
from rl.rag import RAGStore, RetrievedPassage

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    backend: str = 'chatfire'             # 'chatfire' or 'deepseek'
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: float = 25.0

    # Blending weight schedule (alpha = how much LLM influences PPO)
    alpha_initial: float = 0.50           # at tau=0.0
    alpha_decay:   str = 'linear_to_zero'  # 'linear_to_zero' | 'cosine' | 'constant'
    alpha_min:     float = 0.0
    # Canonical NC revision setting: no infeasible-state override.
    # The old fallback (0.80) is available through CLI/config only for sensitivity tests.
    alpha_fallback_infeasible: float = 0.0

    # Call schedule (when to actually call LLM)
    early_tau_threshold: float = 0.30
    mid_tau_threshold:   float = 0.70
    early_interval: int = 1
    mid_interval:   int = 2
    late_interval:  int = 5

    # RAG
    use_rag: bool = True
    rag_top_k: int = 3

    # RAG reranking: retrieve a larger candidate pool by vector similarity,
    # then let an LLM select the most relevant passages by SEMANTIC RELEVANCE.
    # The reranker picks reference text ONLY — it sets no design value and
    # makes no compliance/material/numerical decision (R1 authority boundary).
    use_reranker: bool = True
    rerank_pool_k: int = 20          # per-query candidates retrieved before reranking
    rerank_top_n: int = 3            # passages kept after reranking (final context size)
    rerank_temperature: float = 0.0  # deterministic reranking
    rerank_max_tokens: int = 120     # reranker returns only a short index list


@dataclass
class GeneratorResult:
    """One Generator proposal."""
    action: Optional[np.ndarray]      # 10-dim, or None on failure
    confidence: float                  # 0-1
    reasoning: str
    alpha_used: float                  # actual blending weight after all rules
    success: bool
    error_code: Optional[str] = None
    rag_sources: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    backend: str = ''
    model: str = ''
    episode: int = 0
    step: int = 0
    tau: float = 0.0
    was_called: bool = True            # False if skipped per schedule

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action':       self.action.tolist() if self.action is not None else None,
            'confidence':   self.confidence,
            'reasoning':    self.reasoning,
            'alpha_used':   round(self.alpha_used, 4),
            'success':      self.success,
            'error_code':   self.error_code,
            'rag_sources':  self.rag_sources,
            'elapsed_s':    round(self.elapsed_s, 3),
            'backend':      self.backend,
            'model':        self.model,
            'episode':      self.episode,
            'step':         self.step,
            'tau':          round(self.tau, 4),
            'was_called':   self.was_called,
        }


# ─── Prompts ────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """你是经验丰富的路面结构设计师, 熟悉 JTG D50-2017 沥青路面设计规范。

给定当前路面状态 + 4 个 margin + 临界指标 + 规范上下文, 建议一个 RL agent 的下一步 10 维 action (5 个 Δh + 5 个 ΔE)。

你的建议要:
1. 优先改善临界指标 (margin 接近 1.0 的那个)
2. 在改善临界指标的同时, 不大幅恶化其他 margin
3. 优先用"局部调整"(小幅修改), 避免大跨度变化
4. 考虑成本 — 增加厚度/模量会增加成本

每个 Δh 范围: [-2 cm, +2 cm] (在 action 中编码为 [-1, +1])
每个 ΔE 范围: [-100 MPa, +100 MPa] (在 action 中编码为 [-1, +1])

action 输出顺序: [Δh_上面层, Δh_中面层, Δh_下面层, Δh_基层, Δh_底基层,
                  ΔE_上面层, ΔE_中面层, ΔE_下面层, ΔE_基层, ΔE_底基层]

每个值都必须在 [-1, +1] 范围内。"""


RERANK_SYSTEM_PROMPT = """你是一个检索结果相关性排序器。给定一个路面设计情境的简要描述, 以及若干条带编号的候选参考资料, 你的唯一任务是: 按"与该设计情境的相关程度"从高到低, 选出最相关的若干条, 只输出它们的编号。

严格约束:
- 你只做"相关性判断", 不做任何设计决策。
- 不得提出、修改或推荐任何数值(层厚、模量、配比、标定系数等)、材料或结构方案。
- 不得编造候选资料中不存在的内容; 只能从给定候选编号中选择, 不得给出范围外的编号。

输出格式: 严格 JSON, 形如 {"selected": [3, 7, 1]}, 编号按相关性从高到低排列, 不要输出任何额外文字。"""


GENERATOR_USER_TEMPLATE = """[当前 6 层路面结构 (上→下)]
- 上面层 SMA-13:  h = {h0:.2f} cm,  E = {E0:.0f} MPa
- 中面层 AC-20:   h = {h1:.2f} cm,  E = {E1:.0f} MPa
- 下面层 AC-25:   h = {h2:.2f} cm,  E = {E2:.0f} MPa
- 基层 CTB:       h = {h3:.2f} cm,  E = {E3:.0f} MPa
- 底基层 GAB:     h = {h4:.2f} cm,  E = {E4:.0f} MPa

[当前 4 个 margin (JTG D50-2017)]
- B1 沥青疲劳:        {m_B1:.2f}
- B2 半刚性基层疲劳:  {m_B2:.2f}
- B3 沥青永久变形:    {m_B3:.2f}
- B4 路基应变:        {m_B4:.2f}

[临界指标]
{critical}

[规范参考]
{rag_context}

[输出 JSON]
{{
  "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "confidence": 0.5,
  "reasoning": "一句简短的工程理由"
}}"""


# ─── Generator ──────────────────────────────────────────────────

class Generator:
    """
    LLM-based action proposer with RAG + scheduled calling + confidence weighting.
    """

    def __init__(self,
                 config: Optional[GeneratorConfig] = None,
                 rag: Optional[RAGStore] = None,
                 audit: Optional[AuditChain] = None,
                 fail_fast: bool = False,
                 ):
        self.config = config or GeneratorConfig()
        self.rag = rag
        self.audit = audit
        self.fail_fast = fail_fast

        self._client: Optional[LLMClient] = None
        try:
            self._client = get_client(self.config.backend)
        except LLMError as e:
            if fail_fast:
                raise
            logger.warning('Generator init failed: {} (will silently skip)'.format(e))
            self._client = None

        self.n_calls = 0
        self.n_skipped = 0
        self.n_failures = 0

    # ── Call schedule ─────────────────────────────────────────

    def should_call(self, tau: float, episode_step: int) -> bool:
        """Decide whether to call LLM this step based on training progress."""
        cfg = self.config
        if tau < cfg.early_tau_threshold:
            interval = cfg.early_interval
        elif tau < cfg.mid_tau_threshold:
            interval = cfg.mid_interval
        else:
            interval = cfg.late_interval
        return (episode_step % interval == 0)

    # ── Alpha schedule ────────────────────────────────────────

    def alpha_base(self, tau: float) -> float:
        """Base blending weight, decreases with training progress."""
        cfg = self.config
        if cfg.alpha_decay == 'linear_to_zero':
            return max(cfg.alpha_min, cfg.alpha_initial * (1.0 - tau))
        elif cfg.alpha_decay == 'cosine':
            return cfg.alpha_min + (cfg.alpha_initial - cfg.alpha_min) * \
                0.5 * (1 + np.cos(np.pi * min(tau, 1.0)))
        elif cfg.alpha_decay == 'constant':
            return cfg.alpha_initial
        else:
            return cfg.alpha_initial

    # ── RAG context formatter ────────────────────────────────

    def _build_rag_context(self, query: str, region_query: str = "",
                           situation_desc: str = "") -> tuple:
        """Returns (formatted_text, source_list).

        Two-phase retrieval (spec query + region query), merged and
        deduplicated. When reranking is enabled a larger candidate pool is
        retrieved and an LLM selects the most relevant passages by semantic
        relevance (it picks reference text only; it sets no design value).
        Falls back to similarity order if reranking is off, has no client,
        or fails.
        """
        if not self.config.use_rag or self.rag is None:
            return '[未启用规范检索]', []

        # Candidate pool: large when reranking, small otherwise.
        pool_k = (self.config.rerank_pool_k
                  if self.config.use_reranker else self.config.rag_top_k)

        try:
            spec_passages = self.rag.retrieve(query, top_k=pool_k)
        except Exception:
            spec_passages = []
        region_passages = []
        if region_query:
            try:
                region_passages = self.rag.retrieve(region_query, top_k=pool_k)
            except Exception:
                region_passages = []

        # Merge: interleave region + spec, remove duplicate CHUNKS by text.
        # IMPORTANT: dedup by chunk TEXT, not by source. The regional
        # knowledge base is a single file (every chunk has source
        # 'regional_knowledge.md'); deduping by source would collapse all of
        # its distinct entries — including the SPS-1 entries — into one and
        # strip them from the candidate pool before reranking.
        seen_texts = set()
        merged = []
        max_len = max(len(spec_passages), len(region_passages))
        for i in range(max_len):
            for plist in (region_passages, spec_passages):
                if i < len(plist):
                    p = plist[i]
                    key = p.text.strip()
                    if key not in seen_texts:
                        merged.append(p)
                        seen_texts.add(key)
        # Bound the candidate pool so the reranker prompt stays compact.
        pool_cap = max(self.config.rerank_top_n, self.config.rerank_pool_k + 5)
        merged = merged[:pool_cap]

        if not merged:
            return '[无相关规范条目]', []

        # Rerank by LLM relevance, or fall back to similarity order.
        if self.config.use_reranker and self._client is not None and situation_desc:
            selected = self._rerank_passages(situation_desc, merged)
        else:
            selected = merged[:self.config.rag_top_k]

        lines = []
        sources = []
        for p in selected:
            lines.append('— [{}]: {}'.format(p.source, p.text[:200]))
            sources.append(p.source)
        return '\n'.join(lines), sources

    def _rerank_passages(self, situation_desc, passages):
        """LLM relevance reranking over a candidate pool.

        Sends the design situation + numbered candidates to the LLM and asks
        for the indices of the most relevant passages. The reranker selects
        reference text ONLY; it makes no design decision and sets no value.
        Every decision is logged (and recorded to the audit chain if present)
        so the selection is reproducible/inspectable. On any failure it falls
        back to similarity order so the system degrades to plain RAG.
        """
        top_n = self.config.rerank_top_n
        fallback = passages[:top_n]
        if self._client is None or not passages:
            return fallback

        cand_lines = []
        for i, p in enumerate(passages):
            cand_lines.append('[{}] ({}) {}'.format(i, p.source, p.text[:180]))
        user_prompt = (
            '[设计情境]\n{}\n\n'
            '[候选参考资料]\n{}\n\n'
            '请按与该设计情境的相关程度从高到低选出最相关的 {} 条, '
            '只输出 JSON: {{"selected": [编号, ...]}}.'
        ).format(situation_desc, '\n'.join(cand_lines), top_n)

        try:
            response = self._client.chat(
                system=RERANK_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=self.config.rerank_temperature,
                max_tokens=self.config.rerank_max_tokens,
                timeout=self.config.timeout_s,
                response_format={'type': 'json_object'},
            )
        except LLMError as e:
            logger.info('[RERANK] LLM call failed (%s) -> similarity fallback', e.code)
            return fallback

        parsed = parse_json_from_text(response.text)
        idxs = parsed.get('selected') if isinstance(parsed, dict) else None
        if not isinstance(idxs, list) or not idxs:
            logger.info('[RERANK] unparseable selection -> similarity fallback')
            return fallback

        chosen, seen = [], set()
        for x in idxs:
            try:
                j = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(passages) and j not in seen:
                chosen.append(passages[j]); seen.add(j)
            if len(chosen) >= top_n:
                break
        if not chosen:
            return fallback

        def _tag(pp):
            return pp.text.strip().replace('\n', ' ')[:26]

        def _has_sps1(ps):
            return any(('SPS-1' in pp.text) or ('典型结构-美国' in pp.text)
                       or ('FHWA-RD-01' in pp.text) for pp in ps)

        logger.info('[RERANK] pool=%d sps1_in_pool=%s sps1_selected=%s '
                    'selected=%s',
                    len(passages), _has_sps1(passages), _has_sps1(chosen),
                    [_tag(pp) for pp in chosen])
        if self.audit is not None:
            try:
                self.audit.record('reranker', {
                    'situation': situation_desc,
                    'pool_sources': [p.source for p in passages],
                    'selected_sources': [p.source for p in chosen],
                })
            except Exception:
                pass
        return chosen

    def _build_query(self, critical_indicator: Optional[str], margins: Dict,
                     climate_zone: str = "", pavement_type: str = "flexible") -> str:
        """Build RAG query string from current state, with optional climate context."""
        # Base query from critical indicator
        if critical_indicator == 'B1_asphalt_fatigue':
            base = 'JTG D50 沥青层疲劳寿命 Nf1 计算 应力应变 沥青层厚度'
        elif critical_indicator == 'B2_semi_rigid_fatigue':
            base = 'JTG D50 半刚性基层疲劳 σ_t 拉应力 水泥稳定碎石'
        elif critical_indicator == 'B3_ac_permanent_deformation':
            base = 'JTG D50 沥青混合料永久变形 车辙 高温稳定 多分层'
        elif critical_indicator == 'B4_subgrade_strain':
            base = 'JTG D50 路基顶面竖向压应变 ε_z 路基模量'
        else:
            base = 'JTG D50 沥青路面结构设计 控制指标'

        # Append climate-context keywords for regional knowledge retrieval
        if climate_zone:
            if climate_zone in ('hot', 'warm'):
                base += ' 高温多雨地区抗车辙策略 典型路面结构 本地标定'
            elif climate_zone in ('temperate',):
                base += ' 温和气候区路面结构组合 基层厚度'
            elif climate_zone in ('cold',):
                base += ' 冰冻区沥青路面 粒料基层 低温抗裂'

        return base

    def _build_region_query(self, climate_zone: str = "",
                            pavement_type: str = "flexible",
                            critical_indicator: str = "") -> str:
        """Build a region-only query using generic descriptive terms.

        Uses climate zone descriptors, indicator-relevant strategies, and
        pavement-type features — NOT geographic names — so that retrieval
        is driven by semantic relevance to the section's actual conditions,
        not by hardcoded location keywords. Any regional entry (calibration
        coefficients, provincial structures, climate strategies) that matches
        the same climate/indicator profile can be retrieved, regardless of
        its geographic origin.
        """
        if not climate_zone:
            return ""

        parts = []

        # Indicator-specific strategy terms
        if critical_indicator == 'B3_ac_permanent_deformation':
            parts.append('高温抗车辙策略 沥青混合料永久变形 车辙标定系数 本地校准')
        elif critical_indicator == 'B1_asphalt_fatigue':
            parts.append('沥青层疲劳寿命 层底拉应变 疲劳开裂标定 应力应变')
        elif critical_indicator == 'B2_semi_rigid_fatigue':
            parts.append('半刚性基层疲劳 弯拉应力 水泥稳定碎石 基层标定')
        elif critical_indicator == 'B4_subgrade_strain':
            parts.append('路基顶面竖向压应变 路基模量 粒料层厚度 路基稳定')

        # Climate-zone descriptors (generic, no geographic names)
        if climate_zone in ('hot', 'warm'):
            parts.append('高温多雨地区 抗车辙策略 PG高等级胶结料 '
                         '下面层加厚 硬质沥青 典型路面结构 沥青层厚度')
        elif climate_zone == 'temperate':
            parts.append('温和气候区 路面结构组合 基层厚度 面层设计参数 '
                         '温度调整系数 基准等效温度')
        elif climate_zone == 'cold':
            parts.append('冰冻区 低温抗裂 粒料基层 抗冻层设计 '
                         'PG低温等级 春融期模量折减')

        # Pavement-type features
        if pavement_type == 'flexible':
            parts.append('柔性基层 粒料基层 级配碎石 沥青稳定碎石')
        else:
            parts.append('半刚性基层 水泥稳定碎石 无机结合料 基层疲劳')

        return ' '.join(parts)

    # ── Main propose method ──────────────────────────────────

    def propose(self,
                thickness: List[float],
                modulus: List[float],
                margins: Dict[str, float],
                action_PPO: np.ndarray,
                episode: int,
                step: int,
                tau: float,
                critical_indicator: Optional[str] = None,
                last_step_was_infeasible: bool = False,
                climate_zone: str = "",
                pavement_type: str = "flexible",
                ) -> GeneratorResult:
        """
        Main entry: returns blended action.

        Returns GeneratorResult. If skipped per schedule or failure, .action is None
        and you should use action_PPO directly.
        """
        cfg = self.config

        # Check schedule
        if not self.should_call(tau, step):
            self.n_skipped += 1
            return GeneratorResult(
                action=None, confidence=0.0, reasoning='[skipped per schedule]',
                alpha_used=0.0, success=True, error_code=None,
                episode=episode, step=step, tau=tau, was_called=False,
            )

        # Check client
        if self._client is None:
            return GeneratorResult(
                action=None, confidence=0.0, reasoning='[client unavailable]',
                alpha_used=0.0, success=False, error_code='NO_CLIENT',
                episode=episode, step=step, tau=tau, was_called=False,
            )

        # Build prompt
        h_cm = [x * 100 for x in thickness]
        rag_query = self._build_query(critical_indicator, margins, climate_zone, pavement_type)
        region_query = self._build_region_query(climate_zone, pavement_type,
                                                  critical_indicator or "")
        # Neutral, compact description of the design situation for the reranker
        # (climate + binding indicator + pavement type + margins; NO geographic
        # names and NO target answer — relevance is judged, not hardcoded).
        crit_for_desc = critical_indicator or '无明显临界'
        situation_desc = (
            '气候区={}; 路面类型={}; 当前临界指标={}; '
            'margin: B1={:.2f}, B3={:.2f}, B4={:.2f}.'
        ).format(climate_zone or '未指定', pavement_type, crit_for_desc,
                 margins.get('B1_asphalt_fatigue', 0.0),
                 margins.get('B3_ac_permanent_deformation', 0.0),
                 margins.get('B4_subgrade_strain', 0.0))
        rag_text, rag_sources = self._build_rag_context(
            rag_query, region_query, situation_desc)

        crit_text = '无明显临界 (所有 margin > 2.0)'
        if critical_indicator:
            crit_text = '{}, margin = {:.2f}'.format(
                critical_indicator, margins.get(critical_indicator, 0.0))

        user_prompt = GENERATOR_USER_TEMPLATE.format(
            h0=h_cm[0], h1=h_cm[1], h2=h_cm[2], h3=h_cm[3], h4=h_cm[4],
            E0=modulus[0], E1=modulus[1], E2=modulus[2], E3=modulus[3], E4=modulus[4],
            m_B1=margins.get('B1_asphalt_fatigue', 0.0),
            m_B2=margins.get('B2_semi_rigid_fatigue', 0.0),
            m_B3=margins.get('B3_ac_permanent_deformation', 0.0),
            m_B4=margins.get('B4_subgrade_strain', 0.0),
            critical=crit_text,
            rag_context=rag_text,
        )

        # Call LLM
        try:
            response = self._client.chat(
                system=GENERATOR_SYSTEM_PROMPT,
                user=user_prompt,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout_s,
                response_format={'type': 'json_object'},   # GPT-4o-mini JSON mode; auto-fallback
            )
        except LLMError as e:
            self.n_failures += 1
            if self.fail_fast:
                raise
            result = GeneratorResult(
                action=None, confidence=0.0,
                reasoning='[LLM error: {}]'.format(e.code),
                alpha_used=0.0, success=False, error_code=e.code,
                rag_sources=rag_sources,
                episode=episode, step=step, tau=tau,
            )
            if self.audit is not None:
                self.audit.record('generator', result.to_dict())
            return result

        # Parse JSON
        parsed = parse_json_from_text(response.text)
        if parsed is None or 'action' not in parsed:
            self.n_failures += 1
            result = GeneratorResult(
                action=None, confidence=0.0,
                reasoning='[unparseable response: finish={}, chars={}]'.format(
                    response.finish_reason or 'unknown', len(response.text or '')),
                alpha_used=0.0, success=False, error_code='UNPARSEABLE',
                rag_sources=rag_sources,
                elapsed_s=response.elapsed_s,
                backend=response.backend, model=response.model,
                episode=episode, step=step, tau=tau,
            )
            if self.audit is not None:
                self.audit.record('generator', result.to_dict())
            return result

        # Validate action
        try:
            action_raw = np.array(parsed['action'], dtype=float)
            if action_raw.shape != (10,):
                raise ValueError('action shape != (10,)')
            action_gen = np.clip(action_raw, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            self.n_failures += 1
            result = GeneratorResult(
                action=None, confidence=0.0,
                reasoning='[invalid action: {}]'.format(e),
                alpha_used=0.0, success=False, error_code='INVALID_ACTION',
                rag_sources=rag_sources,
                episode=episode, step=step, tau=tau,
            )
            if self.audit is not None:
                self.audit.record('generator', result.to_dict())
            return result

        confidence = float(parsed.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(parsed.get('reasoning', ''))[:100]

        # Compute alpha
        a_base = self.alpha_base(tau)
        if last_step_was_infeasible and cfg.alpha_fallback_infeasible > 0.0:
            alpha = max(a_base * confidence, cfg.alpha_fallback_infeasible)
        else:
            alpha = a_base * confidence
        alpha = max(0.0, min(1.0, float(alpha)))

        self.n_calls += 1
        result = GeneratorResult(
            action=action_gen,
            confidence=confidence,
            reasoning=reasoning,
            alpha_used=alpha,
            success=True, error_code=None,
            rag_sources=rag_sources,
            elapsed_s=response.elapsed_s,
            backend=response.backend, model=response.model,
            episode=episode, step=step, tau=tau,
        )
        if self.audit is not None:
            self.audit.record('generator', result.to_dict())
        return result

    @staticmethod
    def blend(action_PPO: np.ndarray, action_gen: Optional[np.ndarray],
              alpha: float) -> np.ndarray:
        """Linear blend: (1-alpha) * PPO + alpha * generator. Clip to [-1, 1]."""
        if action_gen is None or alpha <= 0:
            return action_PPO
        return np.clip((1.0 - alpha) * action_PPO + alpha * action_gen,
                       -1.0, 1.0).astype(np.float32)

    def stats(self) -> Dict[str, Any]:
        total = self.n_calls + self.n_skipped + self.n_failures
        return {
            'n_calls':      self.n_calls,
            'n_skipped':    self.n_skipped,
            'n_failures':   self.n_failures,
            'success_rate': self.n_calls / max(total, 1),
        }
