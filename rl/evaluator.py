# -*- coding: utf-8 -*-
"""
rl.evaluator — Async LLM evaluator for PPO actions (Phase 2B)
================================================================

DeepSeek-based independent reviewer.
- Reviews each PPO action (BEFORE blending with Generator).
- Outputs: score 0-10 + one-line reasoning.
- Async via ThreadPoolExecutor: doesn't block FEA.
- Failures saved to audit chain. Optional fail-fast mode for debugging.
- NOT integrated into reward (audit-only, preserves PPO policy purity).

Usage:
    evaluator = Evaluator(audit=audit_chain, fail_fast=False)
    future = evaluator.evaluate_async(state_info, action, episode, step)
    # ... FEA runs in parallel ...
    result = evaluator.collect(future, timeout=10)   # picks up the result
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rl.audit import AuditChain
from rl.llm_client import LLMClient, LLMError, LLMResponse, get_client, parse_json_from_text

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """One Evaluator review."""
    score: float                # 0-10
    reasoning: str              # one-line
    success: bool               # whether LLM call succeeded
    error_code: Optional[str] = None
    raw_text: str = ''
    elapsed_s: float = 0.0
    backend: str = ''
    model: str = ''
    episode: int = 0
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score':       self.score,
            'reasoning':   self.reasoning,
            'success':     self.success,
            'error_code':  self.error_code,
            'elapsed_s':   round(self.elapsed_s, 3),
            'backend':     self.backend,
            'model':       self.model,
            'episode':     self.episode,
            'step':        self.step,
        }


# ─── Prompts ────────────────────────────────────────────────────

EVALUATOR_SYSTEM_PROMPT = """你是路面工程结构设计审核员, 熟悉中国 JTG D50-2017 沥青路面设计规范。

你的任务: 审核一个 RL agent 提出的路面结构设计修改 (10 维 action), 给出一个独立判断。

评分准则 (0-10 分):
- 9-10 分: 修改方向正确, 工程合理, 符合规范精神
- 7-8 分: 修改基本合理, 有小问题但可接受
- 5-6 分: 修改有疑问, 工程上不够合理
- 3-4 分: 修改方向有错, 可能违反工程实践
- 0-2 分: 严重违反工程常识或规范

仅根据下面给出的信息判断, 不要凭空猜测。"""


EVALUATOR_USER_TEMPLATE = """[当前 6 层路面结构 (上→下)]
- 上面层 SMA-13:  h = {h0:.2f} cm,  E = {E0:.0f} MPa
- 中面层 AC-20:   h = {h1:.2f} cm,  E = {E1:.0f} MPa
- 下面层 AC-25:   h = {h2:.2f} cm,  E = {E2:.0f} MPa
- 基层 CTB:       h = {h3:.2f} cm,  E = {E3:.0f} MPa
- 底基层 GAB:     h = {h4:.2f} cm,  E = {E4:.0f} MPa

[当前 4 个 margin (JTG D50-2017, capacity/demand, ≥ 1.0 = 通过)]
- B1 沥青疲劳:        {m_B1:.2f}
- B2 半刚性基层疲劳:  {m_B2:.2f}
- B3 沥青永久变形:    {m_B3:.2f}{m_B3_note}
- B4 路基应变:        {m_B4:.2f}

[临界指标]
{critical}

[RL agent 建议的修改 action]
- Δh_上面层:   {dh0:+.2f} cm
- Δh_中面层:   {dh1:+.2f} cm
- Δh_下面层:   {dh2:+.2f} cm
- Δh_基层:     {dh3:+.2f} cm
- Δh_底基层:   {dh4:+.2f} cm
- ΔE_上面层:   {dE0:+.0f} MPa
- ΔE_中面层:   {dE1:+.0f} MPa
- ΔE_下面层:   {dE2:+.0f} MPa
- ΔE_基层:     {dE3:+.0f} MPa
- ΔE_底基层:   {dE4:+.0f} MPa

[审核任务]
请审核这个修改的工程合理性, 输出 JSON 格式:
{{
  "score": <0-10 整数或小数>,
  "reasoning": "<一句话理由, 最多 50 字>"
}}"""


# ─── Evaluator ──────────────────────────────────────────────────

class Evaluator:
    """
    Async LLM evaluator using DeepSeek.

    Lifecycle:
      __init__       : Create with audit chain reference
      evaluate_async : Submit a review job (non-blocking)
      collect       : Block until result (timeout-bounded)
      close         : Shut down executor

    fail_fast=True: any LLM error raises immediately (debug mode).
    fail_fast=False: LLM errors logged but don't stop training (prod mode).
    """

    def __init__(self,
                 audit: Optional[AuditChain] = None,
                 backend: str = 'deepseek',
                 max_workers: int = 2,
                 default_timeout: float = 15.0,
                 temperature: float = 0.2,
                 max_tokens: int = 800,                      # from 300: leave room for response
                 fail_fast: bool = False,
                 model: Optional[str] = 'deepseek-chat',     # force non-reasoner for JSON scoring
                 ):
        self.audit = audit
        self.backend = backend
        self.default_timeout = default_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fail_fast = fail_fast
        self.model = model

        self._client: Optional[LLMClient] = None
        try:
            self._client = get_client(backend)
        except LLMError as e:
            if fail_fast:
                raise
            logger.warning('Evaluator init failed: {} (will silently skip)'.format(e))
            self._client = None

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.n_calls = 0
        self.n_failures = 0
        self.score_history: List[float] = []

    def _build_user_prompt(self,
                           thickness: List[float],
                           modulus: List[float],
                           margins: Dict[str, float],
                           action: np.ndarray,
                           max_dh_m: float = 0.02,
                           max_dE: float = 100.0,
                           critical_indicator: Optional[str] = None,
                           ) -> str:
        """Render the user prompt from state + action."""
        # Convert h to cm
        h_cm = [x * 100 for x in thickness]
        dh_cm = [action[i] * max_dh_m * 100 for i in range(5)]
        dE   = [action[i+5] * max_dE for i in range(5)]

        m = margins or {}
        m_B3 = m.get('B3_ac_permanent_deformation', 0.0)
        m_B3_note = '  (临界, 接近 1.0 = 高温车辙临界)' if 0.9 < m_B3 < 1.3 else ''

        crit_text = '无明显临界 (所有 margin > 2.0)'
        if critical_indicator:
            crit_text = '{}, margin = {:.2f}'.format(
                critical_indicator,
                m.get(critical_indicator, 0.0))

        return EVALUATOR_USER_TEMPLATE.format(
            h0=h_cm[0], h1=h_cm[1], h2=h_cm[2], h3=h_cm[3], h4=h_cm[4],
            E0=modulus[0], E1=modulus[1], E2=modulus[2], E3=modulus[3], E4=modulus[4],
            m_B1=m.get('B1_asphalt_fatigue', 0.0),
            m_B2=m.get('B2_semi_rigid_fatigue', 0.0),
            m_B3=m_B3, m_B3_note=m_B3_note,
            m_B4=m.get('B4_subgrade_strain', 0.0),
            critical=crit_text,
            dh0=dh_cm[0], dh1=dh_cm[1], dh2=dh_cm[2], dh3=dh_cm[3], dh4=dh_cm[4],
            dE0=dE[0], dE1=dE[1], dE2=dE[2], dE3=dE[3], dE4=dE[4],
        )

    def _do_evaluate_sync(self,
                          thickness: List[float],
                          modulus: List[float],
                          margins: Dict[str, float],
                          action: np.ndarray,
                          episode: int,
                          step: int,
                          critical_indicator: Optional[str] = None,
                          ) -> EvaluationResult:
        """Sync call (used inside executor thread)."""
        if self._client is None:
            return EvaluationResult(
                score=5.0, reasoning='[client unavailable]',
                success=False, error_code='NO_CLIENT',
                episode=episode, step=step,
            )

        user_prompt = self._build_user_prompt(
            thickness, modulus, margins, action,
            critical_indicator=critical_indicator)

        try:
            response = self._client.chat(
                system=EVALUATOR_SYSTEM_PROMPT,
                user=user_prompt,
                model=self.model,                          # deepseek-chat (non-reasoner)
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.default_timeout,
                response_format={'type': 'json_object'},   # DeepSeek JSON mode → parseable
            )
        except LLMError as e:
            self.n_failures += 1
            if self.fail_fast:
                raise
            return EvaluationResult(
                score=5.0,
                reasoning='[LLM error: {}]'.format(e.code),
                success=False, error_code=e.code,
                episode=episode, step=step,
            )

        # Parse JSON response
        parsed = parse_json_from_text(response.text)
        if parsed is None:
            # Fallback: try to find a number in [0,10] in the text
            import re
            m = re.search(r'\b([0-9](?:\.\d+)?|10)\b', response.text)
            score = float(m.group(1)) if m else 5.0
            reasoning = response.text[:80] if response.text else '[unparseable]'
            return EvaluationResult(
                score=max(0.0, min(10.0, score)),
                reasoning=reasoning,
                success=True, error_code='UNPARSEABLE_JSON',
                raw_text=response.text, elapsed_s=response.elapsed_s,
                backend=response.backend, model=response.model,
                episode=episode, step=step,
            )

        score = float(parsed.get('score', 5.0))
        score = max(0.0, min(10.0, score))
        reasoning = str(parsed.get('reasoning', ''))[:100]

        return EvaluationResult(
            score=score, reasoning=reasoning,
            success=True, error_code=None,
            raw_text=response.text, elapsed_s=response.elapsed_s,
            backend=response.backend, model=response.model,
            episode=episode, step=step,
        )

    def evaluate_async(self,
                       thickness: List[float],
                       modulus: List[float],
                       margins: Dict[str, float],
                       action: np.ndarray,
                       episode: int,
                       step: int,
                       critical_indicator: Optional[str] = None,
                       ) -> concurrent.futures.Future:
        """
        Submit an evaluation job. Returns a Future. Non-blocking.

        Pair with collect(future) after FEA to retrieve the result.
        """
        if self._client is None:
            # Return an already-resolved dummy future
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_result(EvaluationResult(
                score=5.0, reasoning='[evaluator disabled]',
                success=False, error_code='DISABLED',
                episode=episode, step=step,
            ))
            return fut

        return self._executor.submit(
            self._do_evaluate_sync,
            list(thickness), list(modulus), dict(margins), np.asarray(action).copy(),
            episode, step, critical_indicator,
        )

    def collect(self, future: concurrent.futures.Future,
                timeout: Optional[float] = None) -> EvaluationResult:
        """
        Block on a submitted future. Records result to audit chain.

        timeout=None means use default_timeout + 5s buffer.
        """
        if timeout is None:
            timeout = self.default_timeout + 5.0

        try:
            result: EvaluationResult = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self.n_failures += 1
            result = EvaluationResult(
                score=5.0,
                reasoning='[future.result timed out at {}s]'.format(timeout),
                success=False, error_code='FUTURE_TIMEOUT',
            )
            if self.fail_fast:
                raise LLMError('FUTURE_TIMEOUT', 'Evaluator future timed out')
        except Exception as e:
            self.n_failures += 1
            result = EvaluationResult(
                score=5.0, reasoning='[future error: {}]'.format(e),
                success=False, error_code='FUTURE_ERROR',
            )
            if self.fail_fast:
                raise

        self.n_calls += 1
        if result.success:
            self.score_history.append(result.score)

        # Record to audit chain
        if self.audit is not None:
            self.audit.record('evaluator', result.to_dict())

        return result

    def stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        mean = float(np.mean(self.score_history)) if self.score_history else 0.0
        return {
            'n_calls':       self.n_calls,
            'n_failures':    self.n_failures,
            'success_rate':  1.0 - (self.n_failures / max(self.n_calls, 1)),
            'mean_score':    mean,
            'min_score':     min(self.score_history) if self.score_history else 0.0,
            'max_score':     max(self.score_history) if self.score_history else 0.0,
        }

    def close(self):
        """Clean shutdown of thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
