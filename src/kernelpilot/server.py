"""FastAPI backend for the dashboard. `kernelpilot serve` runs it."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kernelpilot import __version__
from kernelpilot.analysis.compare import compare_experiments
from kernelpilot.analysis.ptx import summarize_ptx
from kernelpilot.analysis.trace import build_trace
from kernelpilot.config import Settings, load_settings
from kernelpilot.env import detect
from kernelpilot.kernels import REGISTRY, get_spec
from kernelpilot.log import get_logger
from kernelpilot.problem import Problem
from kernelpilot.report import build_report, to_html, to_json, to_markdown
from kernelpilot.store import Store
from kernelpilot.suites import SUITES

log = get_logger("server")


def _finite(obj: Any) -> Any:
    """JSON has no inf/nan. Validation errors are often exactly that, so keep them as strings."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_finite(content), separators=(",", ":"), default=str).encode()


class RunRequest(BaseModel):
    op: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    suite: str | None = None
    strategy: str = "agent"
    budget: int = Field(8, ge=1, le=256)
    planner: str = "auto"


def _dashboard_dir() -> Path | None:
    explicit = os.environ.get("KP_DASHBOARD_DIR")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path(__file__).resolve().parents[2] / "dashboard" / "dist")
    return next((p for p in candidates if (p / "index.html").exists()), None)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_home()
    store = Store(settings.db_path)
    env = detect()
    jobs: dict[str, dict[str, Any]] = {}
    app = FastAPI(title="KernelPilot", version=__version__, default_response_class=SafeJSONResponse)

    def exp_or_404(exp_id: str) -> dict[str, Any]:
        exp = store.experiment(exp_id)
        if exp is None:
            raise HTTPException(404, f"no experiment {exp_id}")
        return exp

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/env")
    def environment():
        backend = (
            settings.backend
            if settings.backend != "auto"
            else ("triton" if env.can_run_triton else "sim")
        )
        return {
            "environment": env.to_dict(),
            "backend": backend,
            "simulated": backend == "sim",
            "llm_provider": settings.llm_provider,
            "db": str(settings.db_path),
        }

    @app.get("/api/kernels")
    def kernels():
        return {
            "operators": [
                {
                    "name": s.name,
                    "summary": s.summary,
                    "dtypes": list(s.dtypes),
                    "default_shapes": [list(x) for x in s.default_shapes],
                    "variants": [
                        {"name": v.name, "summary": v.summary, "searchable": v.searchable}
                        for v in s.variants.values()
                    ],
                }
                for s in REGISTRY.values()
            ],
            "suites": SUITES,
        }

    @app.get("/api/experiments")
    def experiments(limit: int = 100):
        out = []
        for e in store.experiments(limit):
            s = e.get("summary") or {}
            out.append(
                {
                    "id": e["id"],
                    "name": e["name"],
                    "strategy": e["strategy"],
                    "status": e["status"],
                    "simulated": bool(e["simulated"]),
                    "backend": e["backend"],
                    "created_at": e["created_at"],
                    "finished_at": e["finished_at"],
                    "target": e["target"],
                    "params": e["params"],
                    "geomean_speedup": s.get("geomean_speedup"),
                    "candidates": s.get("candidates"),
                    "compile_failed": s.get("compile_failed"),
                    "invalid": s.get("invalid"),
                }
            )
        return out

    @app.get("/api/experiments/{exp_id}")
    def experiment(exp_id: str):
        return exp_or_404(exp_id)

    @app.get("/api/experiments/{exp_id}/candidates")
    def candidates(exp_id: str, problem: str | None = None):
        exp = exp_or_404(exp_id)
        return store.candidates(exp["id"], problem)

    @app.get("/api/experiments/{exp_id}/decisions")
    def decisions(exp_id: str):
        exp = exp_or_404(exp_id)
        return store.decisions(exp["id"])

    @app.get("/api/experiments/{exp_id}/trace")
    def trace(exp_id: str):
        exp = exp_or_404(exp_id)
        return build_trace(store, exp["id"])

    @app.get("/api/experiments/{exp_id}/report.{fmt}")
    def report(exp_id: str, fmt: str):
        exp = exp_or_404(exp_id)
        rep = build_report(store, exp["id"])
        headers = {"Content-Disposition": f'attachment; filename="kernelpilot-{exp["id"]}.{fmt}"'}
        if fmt == "md":
            return PlainTextResponse(to_markdown(rep), media_type="text/markdown", headers=headers)
        if fmt == "html":
            return HTMLResponse(to_html(rep), headers=headers)
        if fmt == "json":
            return PlainTextResponse(to_json(rep), media_type="application/json", headers=headers)
        raise HTTPException(400, "format must be md, html or json")

    @app.get("/api/candidates/{cand_id}")
    def candidate(cand_id: str):
        c = store.candidate(cand_id)
        if c is None:
            raise HTTPException(404, f"no candidate {cand_id}")
        parent = store.candidate(c["parent_id"]) if c.get("parent_id") else None
        c["parent_source"] = parent["source"] if parent else None
        c["parent_label"] = parent["label"] if parent else None
        c["ptx_summary"] = summarize_ptx(c["ptx"]).to_dict() if c.get("ptx") else None
        c.pop("ptx", None)
        return c

    @app.get("/api/compare")
    def compare(a: str, b: str):
        ea, eb = exp_or_404(a), exp_or_404(b)
        return compare_experiments(store, ea["id"], eb["id"])

    @app.post("/api/runs")
    def start_run(req: RunRequest):
        from kernelpilot.optimize import Context, run_experiment
        from kernelpilot.suites import suite_problems

        try:
            if req.suite:
                problems = suite_problems(req.suite)
            elif req.op:
                spec = get_spec(req.op)
                problems = [
                    Problem(
                        req.op,
                        tuple(req.shape or spec.default_shapes[0]),
                        req.dtype or spec.dtypes[0],
                    )
                ]
            else:
                raise ValueError("give op or suite")
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc).strip("'\"")) from exc

        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {"id": job_id, "status": "running", "experiment_id": None, "error": None}

        def work() -> None:
            try:
                ctx = Context.create(settings, env)
                exp_id, _ = run_experiment(
                    ctx, problems, req.strategy, req.budget, planner=req.planner
                )
                jobs[job_id].update(status="done", experiment_id=exp_id)
            except Exception as exc:
                log.exception("run failed")
                jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")

        threading.Thread(target=work, daemon=True).start()
        return jobs[job_id]

    @app.get("/api/runs/{job_id}")
    def job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id]

    dist = _dashboard_dir()
    if dist is not None:
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            target = dist / path
            if path and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        def no_dashboard() -> HTMLResponse:
            return HTMLResponse(
                "<p>API is up. Build the dashboard with <code>cd dashboard && npm install && "
                "npm run build</code>, or see <a href='/docs'>/docs</a>.</p>"
            )

    return app
