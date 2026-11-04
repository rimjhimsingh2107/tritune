"""Worker process entry point: `python -m kernelpilot.backends.worker`.

Reads one JSON request on stdin, writes one result line prefixed with
KP_RESULT on stdout. Anything else the candidate (or Triton) prints is
ignored by the parent.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any

MARK = "KP_RESULT "


def handle(req: dict[str, Any]) -> dict[str, Any]:
    engine = req.get("engine", "sim")
    if engine == "sim":
        from kernelpilot.backends import sim_engine

        return sim_engine.handle(req)
    if engine == "triton":
        from kernelpilot.backends import triton_engine

        return triton_engine.handle(req)
    raise ValueError(f"unknown engine {engine!r}")


def safe_handle(req: dict[str, Any]) -> dict[str, Any]:
    try:
        return handle(req)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "error_kind": "crash",
            "log": traceback.format_exc(limit=8),
        }


def main(argv: list[str]) -> int:
    if argv[:1] == ["--profile-target"]:
        from kernelpilot.backends import triton_engine

        triton_engine.profile_target(argv[1], argv[2], int(argv[3]))
        return 0
    req = json.loads(sys.stdin.read())
    # test hook: lets the isolation tests prove that timeouts really kill us
    if req.get("_sleep"):
        time.sleep(float(req["_sleep"]))
    if req.get("_crash"):
        import os

        os._exit(int(req["_crash"]))
    result = safe_handle(req)
    sys.stdout.write("\n" + MARK + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
