# -*- coding: utf-8 -*-
"""
rl.audit — Tamper-evident audit chain (Phase 2B/2C)
======================================================

Every audited event is written to JSONL with a SHA-256 hash that includes
the previous entry's hash. Modifying any past entry breaks the chain.

Used by:
- Evaluator: records every LLM evaluation of a PPO action
- Generator: records every LLM action proposal
- Env: records every step's design/FEA/margin

Purpose:
- Answers R1-3 (reproducibility) and R3-11 (ablation auditability)
- Forensic trace: "what did Evaluator say at step 3.5 of episode 12?"

File format (JSONL):
    {"seq": 0, "ts": "...", "kind": "evaluator", "data": {...}, "prev_hash": "...", "hash": "..."}
    {"seq": 1, "ts": "...", "kind": "generator", "data": {...}, "prev_hash": "<hash of seq 0>", "hash": "..."}
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional


class AuditChain:
    """
    Thread-safe append-only audit chain.

    Single file per run (so concurrent Evaluator + Generator writes don't collide).
    Reset() opens a new file. Calling code must reset() at start of episode/run.
    """

    GENESIS_HASH = '0' * 64

    def __init__(self, path: Optional[str] = None):
        self._lock = threading.Lock()
        self._seq = 0
        self._prev_hash = self.GENESIS_HASH
        self._path: Optional[str] = None
        self._file = None
        if path:
            self.open(path)

    def open(self, path: str) -> None:
        """Open a new audit file. Closes previous one if any."""
        with self._lock:
            self.close_locked()
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._path = path
            self._file = open(path, 'a', encoding='utf-8', buffering=1)  # line-buffered
            self._seq = 0
            self._prev_hash = self.GENESIS_HASH

    def close_locked(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def close(self):
        with self._lock:
            self.close_locked()

    def record(self, kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a new audit entry. Returns the full entry (incl. hash).

        kind: 'evaluator', 'generator', 'step', 'reset', 'error', etc.
        data: arbitrary JSON-serializable dict
        """
        with self._lock:
            if self._file is None:
                # No file open — silent no-op (audit disabled)
                return {}
            entry = {
                'seq':       self._seq,
                'ts':        datetime.datetime.now().isoformat(timespec='milliseconds'),
                'kind':      kind,
                'data':      data,
                'prev_hash': self._prev_hash,
            }
            # Hash is over the serialized entry WITHOUT the 'hash' field itself
            serialized = json.dumps(entry, sort_keys=True, default=str)
            h = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
            entry['hash'] = h

            try:
                self._file.write(json.dumps(entry, default=str) + '\n')
                self._file.flush()
            except Exception:
                pass

            self._seq += 1
            self._prev_hash = h
            return entry

    @staticmethod
    def verify(path: str) -> Dict[str, Any]:
        """
        Verify the hash chain in a JSONL file.

        Returns: {'ok': bool, 'total': int, 'broken_at': int or None, 'reason': str}
        """
        if not os.path.exists(path):
            return {'ok': False, 'total': 0, 'broken_at': None, 'reason': 'file_not_found'}

        prev = AuditChain.GENESIS_HASH
        n = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return {'ok': False, 'total': n, 'broken_at': n,
                            'reason': 'json_decode_fail'}

                if entry.get('prev_hash') != prev:
                    return {'ok': False, 'total': n, 'broken_at': n,
                            'reason': 'prev_hash_mismatch'}

                # Recompute hash
                claimed_hash = entry.pop('hash', None)
                serialized = json.dumps(entry, sort_keys=True, default=str)
                expected = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
                if expected != claimed_hash:
                    return {'ok': False, 'total': n, 'broken_at': n,
                            'reason': 'hash_recompute_fail'}

                prev = claimed_hash
                n += 1

        return {'ok': True, 'total': n, 'broken_at': None, 'reason': 'chain_valid'}
