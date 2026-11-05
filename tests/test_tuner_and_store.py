import pytest

from tritune.analysis.compare import compare_experiments
from tritune.candidate import Candidate
from tritune.kernels import get_spec
from tritune.problem import Problem
from tritune.report import build_report, to_html, to_json, to_markdown
from tritune.session import Session
from tritune.stats import compare_latency, geomean
from tritune.suites import SUITES, suite_problems
from tritune.tuner import grid_configs, random_configs, run_search


def make_session(store, backend, settings, env, problems, strategy="grid", **params):
    return Session(
        store,
        backend,
        settings,
        env,
        name="test",
        strategy=strategy,
        target=problems,
        params=params,
    )


def test_geomean_and_latency_verdicts():
    assert geomean([2.0, 8.0]) == pytest.approx(4.0)
    assert geomean([]) is None
    assert geomean([None, 0, 4.0]) == pytest.approx(4.0)
    assert compare_latency(0.9, 0.91, 1.0, 1.01).better
    v = compare_latency(0.995, 1.0, 1.0, 1.005)
    assert not v.better and "noise" in v.reason
    assert "regression" in compare_latency(1.2, 1.25, 1.0, 1.01).reason


def test_standard_suite_shape_count():
    probs = suite_problems("standard")
    assert len(probs) >= 12
    ops = {p.op for p in probs}
    assert {"matmul", "softmax", "reduction"} <= ops
    assert len(SUITES["full"]) > len(SUITES["standard"])
    with pytest.raises(KeyError):
        suite_problems("nope")


def test_grid_is_strided_and_random_is_seeded():
    p = Problem("matmul", (1024, 1024, 1024), "fp16")
    full = get_spec("matmul").configs(p)
    g = grid_configs(p, 10)
    assert len(g) == 10
    assert {c.variant for c in g} == {"tiled", "grouped"}  # covers both halves of the grid
    assert g[0].key == full[0].key and g[-1].key == full[-1].key
    assert [c.key for c in random_configs(p, 8, seed=1)] == [
        c.key for c in random_configs(p, 8, seed=1)
    ]
    assert [c.key for c in random_configs(p, 8, seed=1)] != [
        c.key for c in random_configs(p, 8, seed=2)
    ]
    assert len(grid_configs(p, 10_000)) == len(full)


@pytest.mark.parametrize("strategy", ["grid", "random", "halving"])
def test_run_search_finds_improvement_and_persists(store, backend, settings, env, strategy):
    p = Problem("softmax", (1024, 16384), "fp32")
    sess = make_session(store, backend, settings, env, [p], strategy, budget=9)
    res = run_search(sess, p, strategy, budget=9)
    summary = sess.finish()
    assert res.best is not None and res.speedup and res.speedup > 1.0
    assert summary["geomean_speedup"] == pytest.approx(res.speedup)

    rows = store.candidates(sess.exp_id)
    assert len(rows) == 10  # baseline + 9
    assert sum(r["retained"] for r in rows) == 1
    retained = next(r for r in rows if r["retained"])
    assert retained["id"] == res.best.candidate.id
    assert retained["status"] == "measured" and retained["valid_status"] == "verified"
    decisions = store.decisions(sess.exp_id)
    assert decisions[0]["kind"] == "baseline"
    assert any(d["kind"] == "trial" for d in decisions)


def test_session_dedups_and_never_times_invalid_code(store, backend, settings, env):
    p = Problem("softmax", (256, 512), "fp16")
    sess = make_session(store, backend, settings, env, [p])
    spec = get_spec("softmax")
    a = Candidate.from_config(p, spec.baseline(p))
    b = Candidate.from_config(p, spec.baseline(p))
    ev_a, dup_a = sess.register(a)
    ev_b, dup_b = sess.register(b)
    assert not dup_a and dup_b and ev_b is ev_a

    bad = Candidate.from_config(p, spec.baseline(p).replace(variant="naive"))
    ev = sess.evaluate(bad)
    assert ev.status == "invalid"
    assert ev.bench is None
    assert store.candidate(bad.id)["p50_ms"] is None


def test_speedup_excludes_unverified(store, backend, settings, env):
    p = Problem("reduction", (1 << 20,), "fp16")
    sess = make_session(store, backend, settings, env, [p])
    base = sess.baseline(p)
    spec = get_spec("reduction")
    broken = Candidate.from_config(
        p, spec.baseline(p).replace(variant="atomic_fp16acc", TILES=16, BLOCK_SIZE=1024)
    )
    sess.evaluate(broken)
    best = sess.best(p)
    assert best is not None and best.candidate.id == base.candidate.id


def test_report_contains_required_sections(store, backend, settings, env):
    probs = [Problem("matmul", (2048, 2048, 2048), "fp16"), Problem("softmax", (512, 512), "fp16")]
    sess = make_session(store, backend, settings, env, probs, budget=12)
    for p in probs:
        run_search(sess, p, "grid", 12)
    spec = get_spec("softmax")
    sess.evaluate(Candidate.from_config(probs[1], spec.baseline(probs[1]).replace(variant="naive")))
    sess.finish()

    rep = build_report(store, sess.exp_id)
    md = to_markdown(rep)
    assert "SIMULATED RESULTS" in md
    for section in (
        "## Results",
        "## Latency distributions",
        "## Resource usage",
        "## Compilation failures",
        "## Numerical validation failures",
    ):
        assert section in md
    assert "out of resource" in md  # 2048^3 grid includes tiles that blow shared memory
    assert "NaN" in md
    assert "<svg" in to_html(rep)
    assert '"simulated": true' in to_json(rep)


def test_compare_experiments(store, backend, settings, env):
    p = Problem("reduction", (1 << 22,), "fp32")
    ids = []
    for strat in ("grid", "random"):
        sess = make_session(store, backend, settings, env, [p], strat)
        run_search(sess, p, strat, 6)
        sess.finish()
        ids.append(sess.exp_id)
    cmp = compare_experiments(store, *ids)
    assert cmp["shared_problems"] == 1
    row = cmp["rows"][0]
    assert row["a_curve"][-1]["speedup"] == pytest.approx(row["a_speedup"])
    assert cmp["a_wins"] + cmp["b_wins"] + cmp["ties"] == 1
    assert not cmp["mixed_backends"]


def test_store_prefix_lookup_and_latest(store, backend, settings, env):
    p = Problem("vector_add", (4096,), "fp32")
    sess = make_session(store, backend, settings, env, [p], "random")
    run_search(sess, p, "random", 2)
    sess.finish()
    assert store.experiment(sess.exp_id[:15])["id"] == sess.exp_id
    assert store.latest("random")["id"] == sess.exp_id
    assert store.latest("agent") is None
