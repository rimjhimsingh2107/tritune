import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kernelpilot.config import load_settings  # noqa: E402
from kernelpilot.demo import SCENARIOS  # noqa: E402
from kernelpilot.optimize import Context, run_experiment  # noqa: E402
from kernelpilot.problem import Problem  # noqa: E402
from kernelpilot.server import create_app  # noqa: E402


@pytest.fixture
def client_and_ids(tmp_path):
    settings = load_settings(backend="sim", home=tmp_path, db_path=tmp_path / "kp.db", reps=10, warmup=1)
    ctx = Context.create(settings)
    sc = SCENARIOS["invalid-kernel"]
    agent_id, _ = run_experiment(ctx, [Problem.parse(sc.problem)], "agent", 3, planner="heuristic",
                                 opening=sc.opening)
    grid_id, _ = run_experiment(ctx, [Problem.parse(sc.problem)], "grid", 3)
    return TestClient(create_app(settings)), agent_id, grid_id


def test_read_endpoints(client_and_ids):
    client, agent_id, grid_id = client_and_ids
    assert client.get("/api/health").json()["ok"]
    assert client.get("/api/env").json()["simulated"] is True
    ops = {o["name"] for o in client.get("/api/kernels").json()["operators"]}
    assert ops == {"vector_add", "softmax", "reduction", "matmul"}

    exps = client.get("/api/experiments").json()
    assert {e["id"] for e in exps} == {agent_id, grid_id}
    assert client.get(f"/api/experiments/{agent_id}").json()["strategy"] == "agent"
    assert client.get("/api/experiments/nope").status_code == 404

    trace = client.get(f"/api/experiments/{agent_id}/trace").json()
    entries = trace["entries"]
    naive = next(e for e in entries if e["config"]["variant"] == "naive")
    assert naive["validation"]["status"] == "failed"
    assert naive["verdict"]["kind"] == "reject"
    assert naive["validation"]["worst"]["max_abs_err"] == "inf"  # survived JSON
    assert sum(e["retained"] for e in entries) == 1

    cand = client.get(f"/api/candidates/{naive['id']}").json()
    assert "@triton.jit" in cand["source"] and cand["parent_source"]

    cmp = client.get(f"/api/compare?a={agent_id}&b={grid_id}").json()
    assert cmp["shared_problems"] == 1


def test_report_downloads(client_and_ids):
    client, agent_id, _ = client_and_ids
    md = client.get(f"/api/experiments/{agent_id}/report.md")
    assert md.status_code == 200 and "SIMULATED RESULTS" in md.text
    assert "attachment" in md.headers["content-disposition"]
    assert "<svg" in client.get(f"/api/experiments/{agent_id}/report.html").text
    assert client.get(f"/api/experiments/{agent_id}/report.pdf").status_code == 400


def test_start_run(client_and_ids):
    client, *_ = client_and_ids
    assert client.post("/api/runs", json={"op": "conv"}).status_code == 400
    job = client.post("/api/runs", json={"op": "reduction", "shape": [65536], "strategy": "random",
                                         "budget": 2}).json()
    for _ in range(100):
        state = client.get(f"/api/runs/{job['id']}").json()
        if state["status"] != "running":
            break
        time.sleep(0.05)
    assert state["status"] == "done", state
    assert client.get(f"/api/experiments/{state['experiment_id']}").status_code == 200
