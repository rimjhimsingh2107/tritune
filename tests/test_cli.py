import json

import pytest
from typer.testing import CliRunner

from tritune.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRITUNE_BACKEND", "sim")
    monkeypatch.setenv("TRITUNE_REPS", "10")
    monkeypatch.setenv("TRITUNE_WARMUP", "1")
    monkeypatch.chdir(tmp_path)


def run(*args):
    res = runner.invoke(app, list(args), catch_exceptions=False)
    return res


def test_doctor_and_list():
    r = run("doctor")
    assert r.exit_code == 0 and "backend" in r.output
    r = run("list")
    assert r.exit_code == 0 and "matmul" in r.output and "standard" in r.output


def test_test_command_passes_on_searchable_variants():
    r = run("test", "softmax", "--shape", "256x300")
    assert r.exit_code == 0, r.output
    assert "pass" in r.output and "SIMULATED" in r.output


def test_optimize_report_compare_flow(tmp_path):
    r = run(
        "optimize",
        "softmax",
        "--shape",
        "512x8192",
        "--dtype",
        "fp32",
        "--strategy",
        "agent",
        "--budget",
        "3",
    )
    assert r.exit_code == 0, r.output
    assert "geomean speedup" in r.output
    r = run(
        "optimize",
        "softmax",
        "--shape",
        "512x8192",
        "--dtype",
        "fp32",
        "--strategy",
        "grid",
        "--budget",
        "3",
    )
    assert r.exit_code == 0
    r = run("compare", "--json")
    data = json.loads(r.output)
    assert data["a"]["strategy"] == "agent" and data["b"]["strategy"] == "grid"
    r = run("report", "--format", "all", "--out", str(tmp_path / "rep"))
    assert r.exit_code == 0
    assert len(list((tmp_path / "rep").iterdir())) == 3
    assert "experiments" in run("runs").output


def test_bad_inputs_give_clean_errors():
    r = runner.invoke(app, ["optimize", "conv2d"])
    assert r.exit_code != 0 and "unknown operator" in r.output
    r = runner.invoke(app, ["optimize", "softmax", "--strategy", "annealing"])
    assert r.exit_code == 2 and "unknown strategy" in r.output


def test_triton_backend_unavailable_message(env, monkeypatch):
    if env.can_run_triton:
        pytest.skip("GPU present")
    monkeypatch.setenv("TRITUNE_BACKEND", "triton")
    r = runner.invoke(app, ["benchmark", "softmax"])
    assert r.exit_code == 2 and "--backend sim" in r.output


@pytest.mark.slow
def test_demo_all_scenarios(tmp_path):
    r = run("demo", "--out", str(tmp_path / "demo"))
    assert r.exit_code == 0, r.output
    for name in ("success", "invalid-kernel", "compile-failure", "bad-source", "regression"):
        md = (tmp_path / "demo" / f"{name}.md").read_text()
        assert "SIMULATED RESULTS" in md
    assert "failed validation" in r.output
    assert "out of resource" in r.output
    assert "regression" in r.output
