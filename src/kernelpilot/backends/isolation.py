"""Where candidate code actually runs.

SubprocessExecutor gives crash and hang isolation: each request gets a fresh
interpreter, a wall-clock timeout, a CPU-time rlimit, no core dumps, a temp
working directory and an environment with obvious secrets stripped out.

That is *not* a security sandbox. Candidate code runs as your user with full
filesystem and network access. Don't feed it source you wouldn't run yourself.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Protocol

from kernelpilot.backends.worker import MARK, safe_handle
from kernelpilot.log import get_logger, kv

log = get_logger("isolation")
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


class Executor(Protocol):
    name: str

    def run(self, req: dict[str, Any], timeout: float) -> dict[str, Any]: ...


class InProcessExecutor:
    """No isolation at all. Fine for the simulated engine; timeouts are not enforced."""

    name = "inprocess"

    def run(self, req: dict[str, Any], timeout: float) -> dict[str, Any]:
        return safe_handle(req)


def scrubbed_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _limits(cpu_seconds: int):
    def apply() -> None:  # runs in the child between fork and exec
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        except (ImportError, ValueError, OSError):
            pass

    return apply


class SubprocessExecutor:
    name = "subprocess"

    def __init__(self, python: str | None = None):
        self.python = python or sys.executable

    def run(self, req: dict[str, Any], timeout: float) -> dict[str, Any]:
        payload = json.dumps(req)
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits(int(timeout * 2) + 10)
        with tempfile.TemporaryDirectory(prefix="kp-run-") as cwd:
            try:
                proc = subprocess.run(
                    [self.python, "-m", "kernelpilot.backends.worker"],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=cwd,
                    env=scrubbed_env(),
                    check=False,
                    **kwargs,
                )
            except subprocess.TimeoutExpired:
                log.warning("worker timed out", extra=kv(action=req.get("action"), timeout=timeout))
                return {"ok": False, "error_kind": "timeout",
                        "error": f"worker timed out after {timeout:.0f}s ({req.get('action')})"}

        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])

        tail = (proc.stderr or proc.stdout or "")[-3000:]
        log.warning("worker died", extra=kv(code=proc.returncode, action=req.get("action")))
        return {
            "ok": False,
            "error_kind": "crash",
            "error": f"worker exited with code {proc.returncode} without a result",
            "log": tail,
        }


def make_executor(name: str) -> Executor:
    if name == "inprocess":
        return InProcessExecutor()
    if name == "subprocess":
        return SubprocessExecutor()
    raise ValueError(f"unknown isolation mode {name!r}")
