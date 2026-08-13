# -*- coding: utf-8 -*-
"""
rl.llm_client 鈥?Unified LLM client (Phase 2B + 2C)
=======================================================

Supports two backends with OpenAI-compatible API:
- DeepSeek (official): used by Evaluator
- ChatFire (proxy):    used by Generator (gpt-4o-mini via 涓浆)

Both use the openai Python SDK with custom base_url.

Loads credentials from .env (via python-dotenv):
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    CHATFIRE_API_KEY, CHATFIRE_BASE_URL, CHATFIRE_MODEL

Usage:
    from rl.llm_client import get_client, LLMError
    client = get_client('deepseek')
    response = client.chat(
        system="You are an auditor...",
        user="Review this action: ...",
        temperature=0.2,
        max_tokens=300,
        timeout=15,
    )
"""
from __future__ import annotations

import json
import os
import time
import requests as _requests
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package required. Install with: pip install openai python-dotenv"
    )

# Load .env on import
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass


class LLMError(Exception):
    """LLM call failed (timeout, auth, bad response, etc)."""
    def __init__(self, code: str, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.code = code
        self.original = original


@dataclass
class LLMResponse:
    text: str               # Raw text response
    model: str
    backend: str            # 'deepseek' or 'chatfire'
    elapsed_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_content: str = ''   # reasoner CoT (deepseek-reasoner); empty = non-reasoner
    finish_reason: str = ''       # 'stop' | 'length' ... ; 'length' = token budget exhausted


class LLMClient:
    """Single LLM backend wrapper."""

    def __init__(self, backend: str, api_key: str, base_url: str, default_model: str):
        if not api_key:
            raise LLMError('NO_API_KEY',
                'No API key for backend={}. Check .env file.'.format(backend))
        self.backend = backend
        self.default_model = default_model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        # Ollama native API doesn't use OpenAI SDK
        if backend in ('ollama', 'ollama-llama'):
            self.client = None  # native API, no SDK
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, system: str, user: str,
             model: Optional[str] = None,
             temperature: float = 0.3,
             max_tokens: int = 400,
             timeout: float = 30.0,
             response_format: Optional[dict] = None,
             ) -> LLMResponse:
        """Synchronous chat call. Raises LLMError on failure."""
        model_use = model or self.default_model
        t0 = time.time()

        # Ollama native API (POST /api/chat, not OpenAI-compatible)
        if self.backend in ('ollama', 'ollama-llama'):
            try:
                url = self.base_url.replace('/v1', '') + '/api/chat'
                payload = {
                    'model': model_use,
                    'messages': [
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user},
                    ],
                    'stream': False,
                    'options': {'temperature': temperature},
                }
                # Ollama needs longer timeout for first model load
                ollama_timeout = max(timeout, 120.0)
                r = _requests.post(url, json=payload, timeout=ollama_timeout)
                r.raise_for_status()
                body = r.json()
                text = body.get('message', {}).get('content', '')
                elapsed = time.time() - t0
                return LLMResponse(
                    text=text, model=model_use,
                    backend=self.backend, elapsed_s=elapsed,
                )
            except Exception as e:
                elapsed = time.time() - t0
                raise LLMError('API_ERROR',
                    '{} call failed: {}'.format(self.backend, e), e)

        create_kwargs = dict(
            model=model_use,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if response_format is not None:
            create_kwargs['response_format'] = response_format

        resp = None
        try:
            resp = self.client.chat.completions.create(**create_kwargs)
        except Exception as e:
            # Some proxies/models reject response_format 鈫?retry without it
            if response_format is not None and 'response_format' in str(e).lower():
                try:
                    create_kwargs.pop('response_format', None)
                    resp = self.client.chat.completions.create(**create_kwargs)
                except Exception as e2:
                    e = e2
            if resp is None:
                elapsed = time.time() - t0
                err_msg = str(e)
                if 'timeout' in err_msg.lower() or 'timed out' in err_msg.lower():
                    raise LLMError('TIMEOUT',
                        '{} timed out after {:.1f}s'.format(self.backend, elapsed), e)
                elif 'auth' in err_msg.lower() or '401' in err_msg or '403' in err_msg:
                    raise LLMError('AUTH', '{} auth failed: {}'.format(self.backend, e), e)
                elif 'rate' in err_msg.lower() or '429' in err_msg:
                    raise LLMError('RATE_LIMIT', '{} rate limited: {}'.format(self.backend, e), e)
                elif 'balance' in err_msg.lower() or 'quota' in err_msg.lower() or 'insufficient' in err_msg.lower():
                    raise LLMError('QUOTA', '{} quota exhausted: {}'.format(self.backend, e), e)
                else:
                    raise LLMError('API_ERROR', '{} call failed: {}'.format(self.backend, e), e)

        elapsed = time.time() - t0
        try:
            msg = resp.choices[0].message
            content = (msg.content or '').strip()
            reasoning_content = (getattr(msg, 'reasoning_content', '') or '').strip()
            finish_reason = (resp.choices[0].finish_reason or '')
        except Exception as e:
            raise LLMError('BAD_RESPONSE', 'Could not extract content: {}'.format(e), e)

        # When reasoner burns tokens on CoT and leaves content empty,
        # fall back to reasoning_content so downstream parsers can still recover JSON.
        text_out = content if content else reasoning_content

        usage = getattr(resp, 'usage', None)
        return LLMResponse(
            text=text_out,
            model=model_use,
            backend=self.backend,
            elapsed_s=elapsed,
            tokens_in=getattr(usage, 'prompt_tokens', 0) if usage else 0,
            tokens_out=getattr(usage, 'completion_tokens', 0) if usage else 0,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
        )


# 鈹€鈹€鈹€ Module-level singleton clients 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_CLIENTS: Dict[str, LLMClient] = {}


def get_client(backend: str) -> LLMClient:
    """Lazy-init singleton client.

    Supported backends:
        deepseek   鈥?DeepSeek official API
        chatfire   鈥?GPT-4o-mini via ChatFire proxy
        siliconflow-qwen 鈥?Qwen via SiliconFlow (OpenAI-compatible)
        siliconflow-glm  鈥?GLM-4 via SiliconFlow (OpenAI-compatible)
        ollama     鈥?Local Ollama (OpenAI-compatible endpoint)
    """
    if backend in _CLIENTS:
        return _CLIENTS[backend]

    if backend == 'deepseek':
        client = LLMClient(
            backend='deepseek',
            api_key=os.getenv('DEEPSEEK_API_KEY', ''),
            base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            default_model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
        )
    elif backend == 'chatfire':
        client = LLMClient(
            backend='chatfire',
            api_key=os.getenv('CHATFIRE_API_KEY', ''),
            base_url=os.getenv('CHATFIRE_BASE_URL', 'https://api.chatfire.cn/v1'),
            default_model=os.getenv('CHATFIRE_MODEL', 'gpt-4o-mini'),
        )
    elif backend == 'siliconflow-qwen':
        client = LLMClient(
            backend='siliconflow-qwen',
            api_key=os.getenv('SILICONFLOW_API_KEY', ''),
            base_url=os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'),
            default_model=os.getenv('SILICONFLOW_QWEN_MODEL', 'Qwen/Qwen2.5-72B-Instruct'),
        )
    elif backend == 'siliconflow-glm':
        client = LLMClient(
            backend='siliconflow-glm',
            api_key=os.getenv('SILICONFLOW_API_KEY', ''),
            base_url=os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'),
            default_model=os.getenv('SILICONFLOW_GLM_MODEL', 'THUDM/glm-4-9b-chat'),
        )
    elif backend == 'ollama':
        client = LLMClient(
            backend='ollama',
            api_key='ollama',
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
            default_model=os.getenv('OLLAMA_MODEL', 'qwen2.5:7b'),
        )
    elif backend == 'ollama-llama':
        client = LLMClient(
            backend='ollama-llama',
            api_key='ollama',
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1'),
            default_model=os.getenv('OLLAMA_LLAMA_MODEL', 'llama3:latest'),
        )
    else:
        raise ValueError('Unknown backend: {}'.format(backend))

    _CLIENTS[backend] = client
    return client


def parse_json_from_text(text: str) -> Optional[Dict]:
    """Extract first JSON object from an LLM text response. None on failure."""
    import re
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    def _relaxed(s: str) -> Optional[Dict]:
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        s2 = re.sub(r",\s*([}\]])", r"\1", s)                 # trailing commas
        s2 = (s2.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))  # smart quotes
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            pass
        if '"' not in s2 and "'" in s2:
            try:
                return json.loads(s2.replace("'", '"'))
            except json.JSONDecodeError:
                pass
        return None

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        obj = _relaxed(m.group(1))
        if obj is not None:
            return obj

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        obj = _relaxed(text[start:end + 1])
        if obj is not None:
            return obj
    return None


