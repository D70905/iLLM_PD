# -*- coding: utf-8 -*-
"""
Phase 2B + 2C Integration Test (no FEA, no PPO)
==================================================

Tests:
  [1/5] LLMClient: DeepSeek (Evaluator backend)
  [2/5] LLMClient: ChatFire (Generator backend)
  [3/5] AuditChain: write + verify hash chain
  [4/5] Evaluator: full audit cycle (async)
  [5/5] Generator: full propose cycle (with optional RAG)

Expected time: ~30-60 sec (mostly LLM API call latency)

Run from project root:
    conda activate illm_pd
    python examples/test_evaluator_rag.py

Prerequisites:
    pip install openai python-dotenv
    # Optional for RAG:
    pip install chromadb sentence-transformers pypdf

    # .env file populated with API keys

If LLM keys are missing or rate-limited, tests gracefully degrade.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    import numpy as np
    from rl.llm_client import get_client, LLMError
    from rl.audit import AuditChain
    from rl.evaluator import Evaluator
    from rl.generator import Generator, GeneratorConfig
    from rl.rag import RAGStore

    t_start = time.time()
    print('=' * 70)
    print('iLLM-PD Phase 2B+2C Integration Test')
    print('=' * 70)
    print()

    # Test state (typical 6-layer pavement)
    thickness = [0.04, 0.06, 0.08, 0.36, 0.18]  # m
    modulus = [14000, 11000, 9000, 1500, 400]    # MPa
    margins = {
        'B1_asphalt_fatigue': 31.66,
        'B2_semi_rigid_fatigue': 4.73,
        'B3_ac_permanent_deformation': 1.84,
        'B4_subgrade_strain': 3.26,
    }
    test_action = np.array([
        -0.2, 0.0, 0.0, -0.5, 0.0,    # decrease upper AC + base thickness
         0.0, 0.0, 0.0,  0.2, 0.0,    # slight base modulus increase
    ], dtype=np.float32)

    # ─── [1/5] DeepSeek client ────────────────────────────────
    print('[1/5] LLMClient: DeepSeek ping...')
    try:
        client = get_client('deepseek')
        resp = client.chat(
            system='You are a helpful assistant. Respond in JSON.',
            user='Output {"hello": "world"}',
            temperature=0.0,
            max_tokens=50,
            timeout=10.0,
        )
        print('  OK: backend={} model={} elapsed={:.2f}s'.format(
            resp.backend, resp.model, resp.elapsed_s))
        print('  Reply: {}'.format(resp.text[:80]))
    except LLMError as e:
        print('  FAIL [{}]: {}'.format(e.code, e))
        print('  (Skipping subsequent DeepSeek tests)')

    # ─── [2/5] ChatFire client ────────────────────────────────
    print()
    print('[2/5] LLMClient: ChatFire ping...')
    try:
        client = get_client('chatfire')
        resp = client.chat(
            system='You are a helpful assistant. Respond in JSON.',
            user='Output {"hello": "world"}',
            temperature=0.0,
            max_tokens=50,
            timeout=10.0,
        )
        print('  OK: backend={} model={} elapsed={:.2f}s'.format(
            resp.backend, resp.model, resp.elapsed_s))
        print('  Reply: {}'.format(resp.text[:80]))
    except LLMError as e:
        print('  FAIL [{}]: {}'.format(e.code, e))
        print('  (Skipping subsequent ChatFire tests)')

    # ─── [3/5] Audit chain ────────────────────────────────────
    print()
    print('[3/5] AuditChain: write + verify...')
    test_audit_path = './output/audit_test/test_audit.jsonl'
    if os.path.exists(test_audit_path):
        os.remove(test_audit_path)
    audit = AuditChain(test_audit_path)
    audit.record('test', {'msg': 'first entry'})
    audit.record('test', {'msg': 'second entry'})
    audit.record('test', {'msg': 'third entry'})
    audit.close()
    result = AuditChain.verify(test_audit_path)
    if result['ok'] and result['total'] == 3:
        print('  OK: chain verified, {} entries'.format(result['total']))
    else:
        print('  FAIL: {}'.format(result))

    # ─── [4/5] Evaluator (async cycle) ────────────────────────
    print()
    print('[4/5] Evaluator full cycle (async)...')
    evaluator_audit = AuditChain('./output/audit_test/eval_audit.jsonl')
    evaluator = Evaluator(audit=evaluator_audit, fail_fast=False)
    try:
        future = evaluator.evaluate_async(
            thickness=thickness, modulus=modulus, margins=margins,
            action=test_action, episode=1, step=1,
            critical_indicator='B3_ac_permanent_deformation',
        )
        print('  Future submitted, waiting for result (timeout=20s)...')
        result = evaluator.collect(future, timeout=20.0)
        print('  Score:       {:.1f}/10'.format(result.score))
        print('  Reasoning:   {}'.format(result.reasoning))
        print('  Success:     {}'.format(result.success))
        if result.error_code:
            print('  Error code:  {}'.format(result.error_code))
        print('  Elapsed:     {:.2f}s'.format(result.elapsed_s))
    except Exception as e:
        print('  FAIL: {}'.format(e))
    finally:
        evaluator.close()
        evaluator_audit.close()

    # ─── [5/5] Generator (with optional RAG) ──────────────────
    print()
    print('[5/5] Generator full cycle (RAG optional)...')
    generator_audit = AuditChain('./output/audit_test/gen_audit.jsonl')

    # Try RAG
    rag = RAGStore()
    rag_status = 'enabled' if rag.enabled else 'disabled (missing deps or empty)'
    print('  RAG status: {}'.format(rag_status))
    print('  RAG chunk count: {}'.format(rag.count()))

    gen_config = GeneratorConfig(use_rag=True, alpha_initial=0.5)
    generator = Generator(config=gen_config, rag=rag, audit=generator_audit,
                          fail_fast=False)
    try:
        action_PPO = np.zeros(10, dtype=np.float32)  # placeholder PPO action
        result = generator.propose(
            thickness=thickness, modulus=modulus, margins=margins,
            action_PPO=action_PPO, episode=1, step=1, tau=0.1,    # early phase
            critical_indicator='B3_ac_permanent_deformation',
        )
        print('  Success:      {}'.format(result.success))
        if result.was_called:
            print('  Was called:   yes')
            if result.success:
                print('  Action[:5]:   {}'.format(result.action[:5] if result.action is not None else None))
                print('  Confidence:   {:.2f}'.format(result.confidence))
                print('  Alpha used:   {:.2f}'.format(result.alpha_used))
                print('  Reasoning:    {}'.format(result.reasoning))
                print('  RAG sources:  {}'.format(result.rag_sources or '[none]'))

                # Test blending
                blended = Generator.blend(action_PPO, result.action, result.alpha_used)
                print('  Blend test:   PPO={} alpha={:.2f} → blended[:3]={}'
                    .format(action_PPO[:3], result.alpha_used, blended[:3]))
        else:
            print('  Was called:   no ({})'.format(result.reasoning))
        if result.error_code:
            print('  Error code:   {}'.format(result.error_code))
    except Exception as e:
        print('  FAIL: {}'.format(e))
    finally:
        generator_audit.close()

    # ─── Summary ──────────────────────────────────────────────
    print()
    elapsed = time.time() - t_start
    print('=' * 70)
    print('Integration test complete in {:.1f} sec'.format(elapsed))
    print('=' * 70)
    print()
    print('Test audit files written under: ./output/audit_test/')
    print()
    print('NEXT STEPS:')
    print('  1. If both LLM backends OK, ingest RAG corpus:')
    print('     python examples/ingest_rag_corpus.py')
    print('     (Requires JTG D50-2017 PDF + NCHRP 1-37A PDF in docs/regulations/)')
    print()
    print('  2. Integrate Evaluator + Generator into PavementEnv (Phase 2BC integration)')


if __name__ == '__main__':
    main()
