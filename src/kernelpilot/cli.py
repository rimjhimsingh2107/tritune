from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kernelpilot import __version__
from kernelpilot.agent.llm import llm_available
from kernelpilot.backends import BackendUnavailable
from kernelpilot.config import load_settings
from kernelpilot.env import detect
from kernelpilot.kernels import REGISTRY, get_spec
from kernelpilot.log import setup_logging
from kernelpilot.problem import Problem

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="KernelPilot: feedback-driven optimization for Triton kernels.")
console = Console()

SIM_NOTE = "[bold yellow]SIMULATED[/] no GPU in use; timings come from a model, numerics are emulated"


def _settings(backend: str | None = None, **kw):
    s = load_settings(backend=backend, **kw)
    setup_logging(s.log_level, s.log_json)
    return s


def _ctx(settings):
    from kernelpilot.optimize import Context

    try:
        return Context.create(settings)
    except BackendUnavailable as exc:
        console.print(f"[red]error:[/] {exc}")
        console.print("hint: use --backend sim (or KP_BACKEND=sim) to run without a GPU")
        raise typer.Exit(2) from exc


def _problems(op: str | None, shape: str | None, dtype: str | None, suite: str | None) -> list[Problem]:
    from kernelpilot.suites import suite_problems

    if suite:
        try:
            return suite_problems(suite)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from exc
    if not op:
        raise typer.BadParameter("give an operator (e.g. softmax) or --suite")
    try:
        spec = get_spec(op)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    dims = tuple(int(x) for x in shape.lower().split("x")) if shape else spec.default_shapes[0]
    return [Problem(op, dims, dtype or spec.dtypes[0])]


def _ms(v):
    if v is None:
        return "-"
    return f"{v * 1000:.1f} us" if v < 0.1 else f"{v:.3f} ms"


def _summary_table(summary: dict) -> Table:
    t = Table(title="results", title_justify="left")
    for col in ("problem", "baseline p50", "best p50", "speedup", "best config", "evals", "failed"):
        t.add_column(col, justify="right" if col not in ("problem", "best config") else "left")
    for p in summary["problems"]:
        sp = p.get("speedup")
        t.add_row(p["problem"], _ms(p.get("baseline_p50")), _ms(p.get("best_p50")),
                  f"[green]{sp:.2f}x[/]" if sp and sp > 1.01 else (f"{sp:.2f}x" if sp else "-"),
                  p.get("best_label") or "-", str(p["evaluated"]), str(p["compile_failed"] + p["invalid"]))
    return t


def _print_summary(exp_id: str, summary: dict) -> None:
    console.print(_summary_table(summary))
    gm = summary.get("geomean_speedup")
    console.print(f"geomean speedup [bold]{gm:.3f}x[/] over {summary['geomean_over']}/{summary['problems_total']} "
                  f"problems (vs untuned Triton, verified candidates only)" if gm else "no verified speedups")
    if summary.get("simulated"):
        console.print(SIM_NOTE)
    console.print(f"experiment [cyan]{exp_id}[/]  ->  kernelpilot report {exp_id}")


Backend = Annotated[str | None, typer.Option("--backend", "-b", help="auto | triton | sim")]
Shape = Annotated[str | None, typer.Option("--shape", "-s", help="e.g. 4096x1024")]
Dtype = Annotated[str | None, typer.Option("--dtype", "-d", help="fp16 | fp32")]


@app.command()
def doctor(backend: Backend = None) -> None:
    """Check hardware, toolchain and configuration."""
    s = _settings(backend)
    env = detect()
    t = Table(title=f"kernelpilot {__version__}", title_justify="left", show_header=False)
    t.add_column("probe")
    t.add_column("ok")
    t.add_column("detail")
    t.add_row("python", "[green]yes[/]", f"{env.python} on {env.platform} ({env.machine})")
    for p in env.probes:
        t.add_row(p.name, "[green]yes[/]" if p.ok else "[red]no[/]", escape(p.detail))
    ok, why = llm_available(s.llm_provider)
    t.add_row("llm planner", "[green]yes[/]" if ok else "[yellow]no[/]",
              f"{s.llm_provider}" if ok else f"{why} (heuristic planner will be used)")
    console.print(t)
    choice = s.backend if s.backend != "auto" else ("triton" if env.can_run_triton else "sim")
    console.print(f"backend: [bold]{choice}[/]" + ("" if env.can_run_triton else "  " + SIM_NOTE))
    console.print(f"database: {s.db_path}")


@app.command("list")
def list_kernels() -> None:
    """List operators, variants and benchmark suites."""
    from kernelpilot.suites import SUITES

    for spec in REGISTRY.values():
        t = Table(title=f"{spec.name}: {spec.summary}", title_justify="left")
        t.add_column("variant")
        t.add_column("searchable")
        t.add_column("notes")
        for v in spec.variants.values():
            t.add_row(v.name, "yes" if v.searchable else "[dim]no (deliberately broken)[/]", v.summary)
        console.print(t)
        console.print("  default shapes: " + ", ".join("x".join(map(str, s)) for s in spec.default_shapes) + "\n")
    for name, keys in SUITES.items():
        console.print(f"suite [bold]{name}[/] ({len(keys)} problems): " + ", ".join(keys))


@app.command()
def test(
    op: str,
    shape: Shape = None,
    dtype: Dtype = None,
    all_variants: Annotated[bool, typer.Option("--all-variants", help="include the broken variants")] = False,
    backend: Backend = None,
) -> None:
    """Correctness check of every variant against the fp64 reference."""
    from kernelpilot.candidate import Candidate

    s = _settings(backend)
    ctx = _ctx(s)
    spec = get_spec(op)
    if shape:
        problems = _problems(op, shape, dtype, None)
    else:
        problems = [Problem(op, sh, dtype or spec.dtypes[0]) for sh in spec.default_shapes]
    t = Table(title=f"{op} correctness", title_justify="left")
    for col in ("problem", "variant", "compile", "result", "max abs err", "max rel err"):
        t.add_column(col)
    failed = 0
    for p in problems:
        for v in spec.variants.values():
            if not v.searchable and not all_variants:
                continue
            cfg = spec.baseline(p).replace(variant=v.name)
            if spec.check(cfg, p):
                grid = spec.space(p).get(v.name)
                cfg = next((c for c in spec.configs(p) if c.variant == v.name), cfg) if grid else cfg
            cand = Candidate.from_config(p, cfg)
            comp = ctx.backend.compile(cand)
            if not comp.ok:
                t.add_row(p.key, v.name, f"[red]{comp.error_kind}[/]", (comp.error or "")[:60], "", "")
                failed += v.searchable
                continue
            val = ctx.backend.validate(cand, s.seed)
            w = val.worst
            status = "[green]pass[/]" if val.ok else f"[red]{val.status}[/]"
            failed += (not val.ok) and v.searchable
            t.add_row(p.key, v.name, "ok", status, f"{w.max_abs_err:.2e}" if w else "-",
                      f"{w.max_rel_err:.2e}" if w else "-")
    console.print(t)
    if ctx.backend.simulated:
        console.print(SIM_NOTE)
    raise typer.Exit(1 if failed else 0)


@app.command()
def benchmark(
    op: str,
    shape: Shape = None,
    dtype: Dtype = None,
    reps: Annotated[int | None, typer.Option(help="timed repetitions")] = None,
    backend: Backend = None,
) -> None:
    """Benchmark the untuned baseline and one config of each variant."""
    from kernelpilot.candidate import Candidate
    from kernelpilot.session import Session

    s = _settings(backend, reps=reps)
    ctx = _ctx(s)
    problems = _problems(op, shape, dtype, None)
    spec = get_spec(op)
    sess = Session(ctx.store, ctx.backend, s, ctx.env, name=f"benchmark {op}", strategy="benchmark",
                   target=problems)
    t = Table(title=f"{op} benchmark", title_justify="left")
    for col in ("problem", "candidate", "status", "compile", "p50", "p95", "speedup"):
        t.add_column(col)
    for p in problems:
        base = sess.baseline(p)
        evs = [base]
        for v in spec.space(p):
            if v != base.candidate.config.variant:
                cfg = next((c for c in spec.configs(p) if c.variant == v), None)
                if cfg:
                    evs.append(sess.evaluate(Candidate.from_config(p, cfg, origin="manual")))
        for ev in evs:
            sp = sess.speedup(ev)
            t.add_row(p.key, ev.candidate.label, ev.status,
                      f"{ev.compile.compile_ms:.0f} ms" if ev.compile and ev.compile.compile_ms else "-",
                      _ms(ev.p50), _ms(ev.p95), f"{sp:.2f}x" if sp else "-")
    sess.finish()
    console.print(t)
    if ctx.backend.simulated:
        console.print(SIM_NOTE)
    console.print(f"saved as experiment [cyan]{sess.exp_id}[/]")


@app.command()
def optimize(
    op: Annotated[str | None, typer.Argument(help="operator, or omit and use --suite")] = None,
    strategy: Annotated[str, typer.Option("--strategy", help="agent | grid | random | halving")] = "agent",
    budget: Annotated[int, typer.Option(help="candidates per problem")] = 8,
    shape: Shape = None,
    dtype: Dtype = None,
    suite: Annotated[str | None, typer.Option(help="quick | standard | full")] = None,
    planner: Annotated[str, typer.Option(help="agent planner: auto | heuristic | llm")] = "auto",
    seed: Annotated[int | None, typer.Option(help="RNG seed for inputs and random search")] = None,
    reps: Annotated[int | None, typer.Option(help="timed repetitions per benchmark")] = None,
    name: Annotated[str | None, typer.Option(help="experiment name")] = None,
    backend: Backend = None,
) -> None:
    """Optimize kernels with the agent or a deterministic autotuner."""
    from kernelpilot.optimize import run_experiment

    s = _settings(backend, seed=seed, reps=reps)
    ctx = _ctx(s)
    problems = _problems(op, shape, dtype, suite)
    console.print(f"[bold]{strategy}[/] on {len(problems)} problem(s), budget {budget}, backend {ctx.backend.describe()}")
    with console.status("optimizing..."):
        try:
            exp_id, summary = run_experiment(ctx, problems, strategy, budget, name=name, planner=planner)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]error:[/] {exc}")
            raise typer.Exit(2) from exc
    _print_summary(exp_id, summary)


@app.command()
def bakeoff(
    op: Annotated[str | None, typer.Argument()] = None,
    budget: int = 8,
    shape: Shape = None,
    dtype: Dtype = None,
    suite: Annotated[str | None, typer.Option(help="quick | standard | full")] = None,
    baselines: Annotated[str, typer.Option(help="comma-separated autotuner strategies")] = "grid,random",
    planner: str = "auto",
    backend: Backend = None,
) -> None:
    """Run the agent and autotuner(s) with the same budget and compare them."""
    from kernelpilot.analysis.compare import compare_experiments
    from kernelpilot.optimize import run_experiment

    s = _settings(backend)
    ctx = _ctx(s)
    problems = _problems(op, shape, dtype, suite)
    ids = {}
    for strat in ["agent", *[b.strip() for b in baselines.split(",") if b.strip()]]:
        with console.status(f"running {strat}..."):
            ids[strat], summary = run_experiment(ctx, problems, strat, budget, planner=planner)
        gm = summary.get("geomean_speedup")
        console.print(f"{strat:8s} geomean {gm:.3f}x  ({summary['candidates']} candidates)  [cyan]{ids[strat]}[/]"
                      if gm else f"{strat}: no verified speedup")
    for strat, eid in ids.items():
        if strat != "agent":
            _print_comparison(compare_experiments(ctx.store, ids["agent"], eid))
    if ctx.backend.simulated:
        console.print(SIM_NOTE)


def _print_comparison(cmp: dict) -> None:
    a, b = cmp["a"], cmp["b"]
    t = Table(title=f"{a['strategy']} ({a['id']})  vs  {b['strategy']} ({b['id']})", title_justify="left")
    for col in ("problem", f"{a['strategy']} speedup", f"{b['strategy']} speedup",
                "evals to best (a/b)", "failures (a/b)", "winner"):
        t.add_column(col)
    for r in cmp["rows"]:
        win = {a["id"]: a["strategy"], b["id"]: b["strategy"]}.get(r["winner"], r["winner"] or "-")
        t.add_row(r["problem"], f"{r['a_speedup']:.2f}x" if r["a_speedup"] else "-",
                  f"{r['b_speedup']:.2f}x" if r["b_speedup"] else "-",
                  f"{r['a_evals_to_best'] or '-'}/{r['b_evals_to_best'] or '-'}",
                  f"{r['a_failures']}/{r['b_failures']}", win)
    console.print(t)
    ga, gb = cmp["a_geomean_shared"], cmp["b_geomean_shared"]
    if ga and gb:
        console.print(f"geomean: {a['strategy']} {ga:.3f}x, {b['strategy']} {gb:.3f}x; wins {cmp['a_wins']}-"
                      f"{cmp['b_wins']} ({cmp['ties']} ties)")
    if cmp["mixed_backends"]:
        console.print("[red]warning:[/] one experiment is simulated and the other is not; don't compare them")


@app.command()
def compare(
    a: Annotated[str | None, typer.Argument(help="experiment id (default: latest agent run)")] = None,
    b: Annotated[str | None, typer.Argument(help="experiment id (default: latest non-agent run)")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compare two experiments problem by problem."""
    from kernelpilot.analysis.compare import compare_experiments
    from kernelpilot.store import Store

    s = _settings()
    store = Store(s.db_path)
    ea = store.experiment(a) if a else store.latest("agent")
    eb = store.experiment(b) if b else store.latest(exclude_strategy="agent")
    if not ea or not eb:
        console.print("[red]need two finished experiments[/] (run `kernelpilot bakeoff softmax` first)")
        raise typer.Exit(1)
    cmp = compare_experiments(store, ea["id"], eb["id"])
    if as_json:
        typer.echo(json.dumps(cmp, indent=2, default=str))
    else:
        _print_comparison(cmp)


@app.command()
def report(
    exp_id: Annotated[str | None, typer.Argument(help="experiment id (default: latest)")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="md | html | json | all")] = "md",
    out: Annotated[Path | None, typer.Option("--out", "-o", help="output directory")] = None,
) -> None:
    """Write an experiment report (markdown, html and/or json)."""
    from kernelpilot.report import build_report, to_html, to_json, to_markdown
    from kernelpilot.store import Store

    s = _settings()
    store = Store(s.db_path)
    exp = store.experiment(exp_id) if exp_id else store.latest()
    if not exp:
        console.print("[red]no such experiment[/]")
        raise typer.Exit(1)
    rep = build_report(store, exp["id"])
    out = out or Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    writers = {"md": to_markdown, "html": to_html, "json": to_json}
    fmts = list(writers) if fmt == "all" else [fmt]
    for f in fmts:
        if f not in writers:
            raise typer.BadParameter(f"unknown format {f!r}")
        path = out / f"{exp['id']}.{f}"
        path.write_text(writers[f](rep))
        console.print(f"wrote {path}")


@app.command()
def runs(limit: int = 20) -> None:
    """List recent experiments."""
    from kernelpilot.store import Store

    s = _settings()
    store = Store(s.db_path)
    t = Table(title="experiments", title_justify="left")
    for col in ("id", "name", "strategy", "status", "geomean", "candidates", "backend"):
        t.add_column(col)
    for e in store.experiments(limit):
        summ = e.get("summary") or {}
        gm = summ.get("geomean_speedup")
        t.add_row(e["id"], e["name"], e["strategy"], e["status"], f"{gm:.3f}x" if gm else "-",
                  str(summ.get("candidates", "-")), e["backend"] + (" (sim)" if e["simulated"] else ""))
    console.print(t)


@app.command()
def demo(
    scenario: Annotated[str, typer.Option(help="success | invalid-kernel | compile-failure | bad-source | regression | all")] = "all",
    out: Annotated[Path, typer.Option(help="where to write reports")] = Path("reports/demo"),
) -> None:
    """Run the deterministic, simulated walkthrough (no GPU, no API key)."""
    from kernelpilot.analysis.compare import compare_experiments
    from kernelpilot.demo import SCENARIOS
    from kernelpilot.optimize import run_experiment
    from kernelpilot.report import build_report, to_html, to_markdown

    s = _settings("sim", reps=40, seed=0)
    ctx = _ctx(s)
    names = list(SCENARIOS) if scenario == "all" else [scenario]
    out.mkdir(parents=True, exist_ok=True)
    console.print(Panel(SIM_NOTE + "\nEvery step below runs through the real planner, tools, store and reports.",
                        title="KernelPilot demo"))
    for n in names:
        sc = SCENARIOS.get(n)
        if sc is None:
            raise typer.BadParameter(f"unknown scenario {n!r}")
        console.rule(f"[bold]{sc.title}")
        console.print(f"[dim]{sc.blurb}[/]")
        problem = Problem.parse(sc.problem)
        exp_id, summary = run_experiment(ctx, [problem], "agent", sc.budget, name=f"demo/{sc.name}",
                                         planner="heuristic", opening=sc.opening or None)
        _narrate(ctx.store, exp_id)
        _print_summary(exp_id, summary)
        rep = build_report(ctx.store, exp_id)
        (out / f"{sc.name}.md").write_text(to_markdown(rep))
        (out / f"{sc.name}.html").write_text(to_html(rep))
        if sc.compare_with:
            other_id, _ = run_experiment(ctx, [problem], sc.compare_with, sc.budget,
                                         name=f"demo/{sc.name}-{sc.compare_with}")
            _print_comparison(compare_experiments(ctx.store, exp_id, other_id))
    console.print(f"\nreports in {out}/  ·  dashboard: kernelpilot serve")


def _narrate(store, exp_id: str) -> None:
    style = {"hypothesis": "cyan", "accept": "green", "reject": "red", "final": "bold"}
    labels = {c["id"]: c["label"] for c in store.candidates(exp_id)}
    for d in store.decisions(exp_id):
        k = d["kind"]
        if k not in style:
            continue
        cand = labels.get(d["candidate_id"] or "", "")
        console.print(f"  [{style[k]}]{k:10s}[/] {d['rationale']}" + (f"  [dim]{cand}[/]" if cand and k == "hypothesis" else ""))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API server + dashboard."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print("pip install 'kernelpilot[server]' to run the dashboard")
        raise typer.Exit(1) from exc
    from kernelpilot.server import create_app

    _settings()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command()
def cache(clear: Annotated[bool, typer.Option("--clear")] = False) -> None:
    """Show or clear the result cache."""
    from kernelpilot.cache import ResultCache

    s = _settings()
    s.ensure_home()
    c = ResultCache(s.db_path)
    if clear:
        console.print(f"removed {c.clear()} cached results")
    else:
        console.print_json(json.dumps(c.stats()))


if __name__ == "__main__":
    app()
