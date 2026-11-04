"""SQLite persistence for experiments, candidates and optimizer decisions."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kernelpilot.backends.base import BenchmarkResult, CompileResult, ProfileResult
from kernelpilot.candidate import Candidate
from kernelpilot.correctness import ValidationResult

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    target TEXT NOT NULL,           -- json list of problem keys
    params TEXT NOT NULL,           -- json: budget, seed, reps, llm provider, ...
    environment TEXT NOT NULL,      -- json: detected hardware/software
    fingerprint TEXT NOT NULL,      -- json: what went into cache keys
    backend TEXT NOT NULL,
    simulated INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    summary TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    problem TEXT NOT NULL,
    op TEXT NOT NULL,
    dtype TEXT NOT NULL,
    origin TEXT NOT NULL,
    parent_id TEXT,
    config TEXT,
    label TEXT NOT NULL,
    source TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,

    compile_ok INTEGER,
    compile_error TEXT,
    compile_error_kind TEXT,
    compile_ms REAL,
    compile_log TEXT,
    compile_cached INTEGER,
    metadata TEXT,
    ptx TEXT,

    valid_status TEXT,
    validation TEXT,

    bench_ok INTEGER,
    bench_error TEXT,
    p50_ms REAL,
    p95_ms REAL,
    mean_ms REAL,
    std_ms REAL,
    min_ms REAL,
    max_ms REAL,
    samples TEXT,
    bench_method TEXT,
    bench_cached INTEGER,

    profile TEXT,

    status TEXT NOT NULL DEFAULT 'pending',
    retained INTEGER NOT NULL DEFAULT 0,
    speedup REAL
);
CREATE INDEX IF NOT EXISTS idx_cand_exp ON candidates(experiment_id, seq);
CREATE INDEX IF NOT EXISTS idx_cand_hash ON candidates(source_hash);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    problem TEXT,
    tool TEXT,
    args TEXT,
    result TEXT,
    rationale TEXT,
    evidence TEXT,
    candidate_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dec_exp ON decisions(experiment_id, id);
"""

JSON_COLS = {
    "target",
    "params",
    "environment",
    "fingerprint",
    "summary",
    "config",
    "metadata",
    "validation",
    "samples",
    "profile",
    "args",
    "result",
    "evidence",
}


def _dumps(v: Any) -> str | None:
    return None if v is None else json.dumps(v, default=str)


def _row(cur: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    out = {}
    for (name, *_), value in zip(cur.description, row, strict=True):
        if name in JSON_COLS and isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        out[name] = value
    return out


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = _row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()
        self._lock = threading.RLock()

    def close(self) -> None:
        self.conn.close()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    # -- experiments -----------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        strategy: str,
        target: list[str],
        params: dict[str, Any],
        environment: dict[str, Any],
        fingerprint: dict[str, Any],
        backend: str,
        simulated: bool,
    ) -> str:
        exp_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self._exec(
            """INSERT INTO experiments (id, name, strategy, target, params, environment, fingerprint,
               backend, simulated, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exp_id,
                name,
                strategy,
                _dumps(target),
                _dumps(params),
                _dumps(environment),
                _dumps(fingerprint),
                backend,
                int(simulated),
                time.time(),
            ),
        )
        return exp_id

    def finish_experiment(self, exp_id: str, status: str, summary: dict[str, Any]) -> None:
        self._exec(
            "UPDATE experiments SET status = ?, summary = ?, finished_at = ? WHERE id = ?",
            (status, _dumps(summary), time.time(), exp_id),
        )

    def experiment(self, exp_id: str) -> dict[str, Any] | None:
        row = self._exec("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if row is None:
            # allow unique prefixes, handy on the command line
            rows = self._exec(
                "SELECT * FROM experiments WHERE id LIKE ?", (exp_id + "%",)
            ).fetchall()
            row = rows[0] if len(rows) == 1 else None
        return row

    def experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._exec(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def latest(
        self, strategy: str | None = None, exclude_strategy: str | None = None
    ) -> dict[str, Any] | None:
        sql, args = "SELECT * FROM experiments WHERE status != 'running'", []
        if strategy:
            sql += " AND strategy = ?"
            args.append(strategy)
        if exclude_strategy:
            sql += " AND strategy != ?"
            args.append(exclude_strategy)
        sql += " ORDER BY created_at DESC LIMIT 1"
        return self._exec(sql, tuple(args)).fetchone()

    def delete_experiment(self, exp_id: str) -> None:
        self._exec("DELETE FROM experiments WHERE id = ?", (exp_id,))

    # -- candidates ------------------------------------------------------------

    def add_candidate(self, exp_id: str, cand: Candidate) -> int:
        with self._lock:
            seq = self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM candidates WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()["n"]
            self.conn.execute(
                """INSERT INTO candidates (id, experiment_id, seq, problem, op, dtype, origin, parent_id,
                   config, label, source, source_hash, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cand.id,
                    exp_id,
                    seq,
                    cand.problem.key,
                    cand.problem.op,
                    cand.problem.dtype,
                    cand.origin,
                    cand.parent_id,
                    _dumps(cand.config.to_dict() if cand.config else None),
                    cand.label,
                    cand.source,
                    cand.source_hash,
                    cand.note,
                    time.time(),
                ),
            )
            self.conn.commit()
        return seq

    def record_compile(self, cand_id: str, r: CompileResult) -> None:
        self._exec(
            """UPDATE candidates SET compile_ok = ?, compile_error = ?, compile_error_kind = ?,
               compile_ms = ?, compile_log = ?, compile_cached = ?, metadata = ?, ptx = ? WHERE id = ?""",
            (
                int(r.ok),
                r.error,
                r.error_kind,
                r.compile_ms,
                r.log,
                int(r.cached),
                _dumps(r.metadata),
                r.ptx,
                cand_id,
            ),
        )

    def record_validation(self, cand_id: str, r: ValidationResult) -> None:
        self._exec(
            "UPDATE candidates SET valid_status = ?, validation = ? WHERE id = ?",
            (r.status, _dumps(r.to_dict()), cand_id),
        )

    def record_benchmark(self, cand_id: str, r: BenchmarkResult) -> None:
        self._exec(
            """UPDATE candidates SET bench_ok = ?, bench_error = ?, p50_ms = ?, p95_ms = ?, mean_ms = ?,
               std_ms = ?, min_ms = ?, max_ms = ?, samples = ?, bench_method = ?, bench_cached = ?
               WHERE id = ?""",
            (
                int(r.ok),
                r.error,
                r.p50_ms,
                r.p95_ms,
                r.mean_ms,
                r.std_ms,
                r.min_ms,
                r.max_ms,
                _dumps(r.samples_ms),
                r.method,
                int(r.cached),
                cand_id,
            ),
        )

    def record_profile(self, cand_id: str, r: ProfileResult) -> None:
        self._exec("UPDATE candidates SET profile = ? WHERE id = ?", (_dumps(r.to_dict()), cand_id))

    def set_status(self, cand_id: str, status: str, speedup: float | None = None) -> None:
        self._exec(
            "UPDATE candidates SET status = ?, speedup = ? WHERE id = ?", (status, speedup, cand_id)
        )

    def set_retained(self, exp_id: str, problem: str, cand_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE candidates SET retained = 0 WHERE experiment_id = ? AND problem = ?",
                (exp_id, problem),
            )
            self.conn.execute("UPDATE candidates SET retained = 1 WHERE id = ?", (cand_id,))
            self.conn.commit()

    def candidates(
        self, exp_id: str, problem: str | None = None, with_source: bool = False
    ) -> list[dict[str, Any]]:
        cols = (
            "*"
            if with_source
            else (
                "id, experiment_id, seq, problem, op, dtype, origin, parent_id, config, label, source_hash, note, "
                "created_at, compile_ok, compile_error, compile_error_kind, compile_ms, compile_cached, metadata, "
                "valid_status, validation, bench_ok, bench_error, p50_ms, p95_ms, mean_ms, std_ms, min_ms, max_ms, "
                "samples, bench_method, bench_cached, profile, status, retained, speedup"
            )
        )
        sql = f"SELECT {cols} FROM candidates WHERE experiment_id = ?"
        args: tuple = (exp_id,)
        if problem:
            sql += " AND problem = ?"
            args = (exp_id, problem)
        return self._exec(sql + " ORDER BY seq", args).fetchall()

    def candidate(self, cand_id: str) -> dict[str, Any] | None:
        return self._exec("SELECT * FROM candidates WHERE id = ?", (cand_id,)).fetchone()

    # -- decisions -------------------------------------------------------------

    def add_decision(
        self,
        exp_id: str,
        step: int,
        kind: str,
        *,
        problem: str | None = None,
        tool: str | None = None,
        args: Any = None,
        result: Any = None,
        rationale: str | None = None,
        evidence: Any = None,
        candidate_id: str | None = None,
    ) -> int:
        cur = self._exec(
            """INSERT INTO decisions (experiment_id, step, kind, problem, tool, args, result, rationale,
               evidence, candidate_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exp_id,
                step,
                kind,
                problem,
                tool,
                _dumps(args),
                _dumps(result),
                rationale,
                _dumps(evidence),
                candidate_id,
                time.time(),
            ),
        )
        return int(cur.lastrowid or 0)

    def decisions(self, exp_id: str) -> list[dict[str, Any]]:
        return self._exec(
            "SELECT * FROM decisions WHERE experiment_id = ? ORDER BY id", (exp_id,)
        ).fetchall()
