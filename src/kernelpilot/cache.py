"""Content-addressed cache for compile/validate/benchmark/profile results.

Keys hash everything that can change the answer: the candidate source, the
problem (op, shape, dtype), the stage-specific knobs (seed, warmup, reps) and
the environment fingerprint (backend, GPU, compute capability, driver, CUDA,
torch, triton, kernelpilot version). Change any of those and you get a miss.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from kernelpilot.problem import canonical_json, sha256

SCHEMA = """
CREATE TABLE IF NOT EXISTS result_cache (
    key TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
"""


def cache_key(
    stage: str, source_hash: str, problem_key: str, fingerprint: dict[str, Any], **extra: Any
) -> str:
    return sha256(
        canonical_json(
            {
                "stage": stage,
                "source": source_hash,
                "problem": problem_key,
                "env": fingerprint,
                "extra": extra,
            }
        )
    )


class ResultCache:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM result_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self._conn.execute("UPDATE result_cache SET hits = hits + 1 WHERE key = ?", (key,))
            self._conn.commit()
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, stage: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO result_cache (key, stage, payload, created_at) VALUES (?, ?, ?, ?)",
                (key, stage, json.dumps(payload), time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT stage, COUNT(*), SUM(hits) FROM result_cache GROUP BY stage"
            ).fetchall()
        return {
            "session_hits": self.hits,
            "session_misses": self.misses,
            "entries": {stage: {"count": n, "hits": h or 0} for stage, n, h in rows},
        }

    def clear(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM result_cache").rowcount
            self._conn.commit()
        return n

    def close(self) -> None:
        self._conn.close()
