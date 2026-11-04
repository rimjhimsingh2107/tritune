from kernelpilot.agent.llm import LLMPlanner, Reply
from kernelpilot.agent.planner import HeuristicPlanner, ToolCall, Turn
from kernelpilot.agent.recommender import recommend
from kernelpilot.agent.runner import AgentBudget, AgentRunner
from kernelpilot.agent.tools import TOOLS, AgentState, ToolRegistry
from kernelpilot.analysis.observations import Observation
from kernelpilot.problem import KernelConfig, Problem
from kernelpilot.session import Session


def new_session(store, backend, settings, env, problem):
    return Session(
        store, backend, settings, env, name="agent test", strategy="agent", target=[problem]
    )


def kinds(store, exp_id, kind):
    return [d for d in store.decisions(exp_id) if d["kind"] == kind]


def test_tool_schemas_are_json_schema():
    for t in TOOLS:
        s = t.schema()
        assert s["input_schema"]["type"] == "object"
        assert s["description"]
    names = {t.name for t in TOOLS}
    assert {
        "inspect_kernel",
        "generate_candidate",
        "compile_candidate",
        "validate_correctness",
        "benchmark_candidate",
        "inspect_ptx",
        "inspect_compiler_metadata",
        "profile_candidate",
        "compare_candidates",
        "recommend_optimization",
        "save_experiment",
    } <= names


def test_heuristic_agent_fixes_spills(store, backend, settings, env):
    p = Problem("softmax", (1024, 16384), "fp32")
    sess = new_session(store, backend, settings, env, p)
    out = AgentRunner(sess, HeuristicPlanner(), AgentBudget(max_candidates=5)).run(p)
    summary = sess.finish()
    assert out.speedup and out.speedup > 2.0
    assert out.candidates <= 5
    first = kinds(store, sess.exp_id, "hypothesis")[0]
    assert "spill" in first["rationale"] or any(
        e.get("kind") == "spills" for e in first["evidence"] or []
    )
    assert kinds(store, sess.exp_id, "accept")
    assert kinds(store, sess.exp_id, "final")
    assert summary["problems"][0]["speedup"] == out.speedup


def test_recovers_from_numerical_failure(store, backend, settings, env):
    p = Problem("softmax", (2048, 1024), "fp16")
    opening = [{"variant": "naive", "hypothesis": "skip the max pass to save a reduction"}]
    sess = new_session(store, backend, settings, env, p)
    AgentRunner(sess, HeuristicPlanner(opening=opening), AgentBudget(max_candidates=4)).run(p)
    sess.finish()
    rejects = kinds(store, sess.exp_id, "reject")
    assert any("failed validation" in r["rationale"] for r in rejects)
    hyps = kinds(store, sess.exp_id, "hypothesis")
    assert "overflow" in hyps[1]["rationale"]  # the repair comes right after
    naive = [c for c in store.candidates(sess.exp_id) if c["config"]["variant"] == "naive"]
    assert naive and naive[0]["p50_ms"] is None  # never benchmarked


def test_repairs_shared_memory_overflow(store, backend, settings, env):
    p = Problem("matmul", (2048, 2048, 2048), "fp16")
    opening = [
        {
            "params": {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
            "num_stages": 4,
            "num_warps": 8,
            "hypothesis": "max out the tile for arithmetic intensity",
        }
    ]
    sess = new_session(store, backend, settings, env, p)
    out = AgentRunner(sess, HeuristicPlanner(opening=opening), AgentBudget(max_candidates=6)).run(p)
    sess.finish()
    rejects = kinds(store, sess.exp_id, "reject")
    assert any("shared memory" in r["rationale"] or "resources" in r["rationale"] for r in rejects)
    hyps = kinds(store, sess.exp_id, "hypothesis")
    assert "stage" in hyps[1]["rationale"].lower() or "BLOCK_K" in hyps[1]["rationale"]
    assert out.speedup and out.speedup >= 1.0


def test_regression_is_rejected_and_incumbent_kept(store, backend, settings, env):
    p = Problem("reduction", (1 << 24,), "fp32")
    opening = [
        {
            "params": {"BLOCK_SIZE": 8192},
            "num_warps": 2,
            "num_stages": 1,
            "hypothesis": "bigger blocks, fewer warps",
        }
    ]
    sess = new_session(store, backend, settings, env, p)
    AgentRunner(sess, HeuristicPlanner(opening=opening), AgentBudget(max_candidates=3)).run(p)
    summary = sess.finish()
    rejects = kinds(store, sess.exp_id, "reject")
    assert any("regression" in r["rationale"] for r in rejects)
    best = summary["problems"][0]["best_label"]
    assert "BLOCK_SIZE=8192" not in best or "w2" not in best


def test_budget_and_duplicates(store, backend, settings, env):
    p = Problem("vector_add", (1 << 20,), "fp32")
    sess = new_session(store, backend, settings, env, p)
    base = sess.register_baseline(p)
    st = AgentState(sess, p, max_candidates=2)
    reg = ToolRegistry(st)
    for t in ("compile_candidate", "validate_correctness", "benchmark_candidate"):
        assert reg.call(t, {"candidate_id": base.candidate.id}).ok
    a = reg.call("generate_candidate", {"num_warps": 8, "hypothesis": "more warps"})
    assert a.ok and a.content["budget_left"] == 1
    dup = reg.call("generate_candidate", {"num_warps": 8, "hypothesis": "more warps again"})
    assert dup.content["duplicate_of"] == a.content["candidate_id"]
    assert st.created == 1  # duplicates don't burn budget
    b = reg.call("generate_candidate", {"num_warps": 2, "hypothesis": "fewer warps"})
    assert b.ok
    over = reg.call("generate_candidate", {"num_warps": 16, "hypothesis": "one too many"})
    assert not over.ok and "budget" in over.error


def test_tool_errors_are_returned_not_raised(store, backend, settings, env):
    p = Problem("vector_add", (4096,), "fp32")
    sess = new_session(store, backend, settings, env, p)
    sess.register_baseline(p)
    reg = ToolRegistry(AgentState(sess, p, 4))
    assert "unknown tool" in reg.call("rm_rf", {}).error
    assert "invalid arguments" in reg.call("benchmark_candidate", {"reps": 3}).error
    assert "no candidate" in reg.call("compile_candidate", {"candidate_id": "nope"}).error
    bad = reg.call(
        "generate_candidate",
        {"variant": "warp_specialized", "hypothesis": "x", "base_candidate_id": sess.order[0]},
    )
    assert not bad.ok and "no variant" in bad.error
    assert (
        "no change"
        in reg.call(
            "generate_candidate", {"hypothesis": "nothing", "base_candidate_id": sess.order[0]}
        ).error
    )
    # benchmarking something that doesn't validate is refused
    gen = reg.call(
        "generate_candidate",
        {
            "variant": "unmasked",
            "params": {"BLOCK_SIZE": 512},
            "hypothesis": "skip masks",
            "base_candidate_id": sess.order[0],
        },
    )
    assert gen.ok
    # 4096 % 512 == 0, so the unmasked add is valid here, but edge shapes catch it
    v = reg.call("validate_correctness", {"candidate_id": gen.content["candidate_id"]})
    assert v.ok and v.content["ok"] is False
    bench = reg.call("benchmark_candidate", {"candidate_id": gen.content["candidate_id"]})
    assert bench.content["ok"] is False and "validation" in bench.content["error"]


def test_inspect_ptx_unavailable_in_sim(store, backend, settings, env):
    p = Problem("softmax", (128, 128), "fp32")
    sess = new_session(store, backend, settings, env, p)
    base = sess.register_baseline(p)
    reg = ToolRegistry(AgentState(sess, p, 2))
    out = reg.call("inspect_ptx", {"candidate_id": base.candidate.id})
    assert out.ok and out.content["available"] is False
    assert "simulated" in out.content["reason"]


def test_custom_source_candidate(store, backend, settings, env):
    p = Problem("softmax", (256, 1000), "fp32")
    sess = new_session(store, backend, settings, env, p)
    base = sess.register_baseline(p)
    reg = ToolRegistry(AgentState(sess, p, 3))
    src = base.candidate.source.replace("NUM_WARPS = 4", "NUM_WARPS = 8")
    g = reg.call("generate_candidate", {"source": src, "hypothesis": "edit by hand"})
    assert g.ok and "w8" in g.content["label"]
    broken = base.candidate.source.replace("BLOCK_N = 1024", "BLOCK_N = 1000")
    g2 = reg.call("generate_candidate", {"source": broken, "hypothesis": "exact fit block"})
    c = reg.call("compile_candidate", {"candidate_id": g2.content["candidate_id"]})
    assert c.content["ok"] is False and "power of 2" in c.content["error"]
    rec = reg.call("recommend_optimization", {"candidate_id": base.candidate.id})
    assert rec.content["failed_candidate"] == g2.content["candidate_id"]


def test_recommender_avoids_tried_and_regressed():
    p = Problem("matmul", (1024, 1024, 1024), "fp16")
    cfg = KernelConfig("tiled", {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 1}, 4, 2)
    obs = [Observation("small_tiles", "", evidence={"BLOCK_M": 64, "BLOCK_N": 64})]
    recs = recommend(p, cfg, obs)
    top = recs[0]
    assert top.changes == {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8}
    again = recommend(p, cfg, obs, tried={top.config.key})
    assert all(r.config.key != top.config.key for r in again)
    avoided = recommend(p, cfg, obs, avoid=[top.changes])
    assert all(r.changes != top.changes for r in avoided)


class FakeProvider:
    """Stands in for an LLM: replays a fixed script of turns."""

    name = "fake-llm"

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def reset(self, system, tools, first_message):
        self.system, self.tools, self.first = system, tools, first_message

    def step(self, results):
        self.seen.append(results)
        if not self.script:
            return Reply("nothing more", [], "end_turn")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(results) if callable(item) else item


def test_llm_planner_path_with_fake_provider(store, backend, settings, env):
    p = Problem("softmax", (1024, 16384), "fp32")
    sess = new_session(store, backend, settings, env, p)
    base_id = {}

    def first(_):
        return Reply("look first", [ToolCall("inspect_kernel", {}, "t1")], "tool_use")

    def pipeline(results):
        base_id["id"] = results[0].content["candidate_id"]
        b = base_id["id"]
        return Reply(
            "baseline pipeline",
            [
                ToolCall("compile_candidate", {"candidate_id": b}, "t2"),
                ToolCall("validate_correctness", {"candidate_id": b}, "t3"),
                ToolCall("benchmark_candidate", {"candidate_id": b}, "t4"),
            ],
            "tool_use",
        )

    def gen(_):
        return Reply(
            "spills; more warps",
            [
                ToolCall("generate_candidate", {"num_warps": 16, "hypothesis": "h"}, "t5"),
                ToolCall("not_a_tool", {}, "t6"),
            ],
            "tool_use",
        )

    def evaluate(results):
        cid = results[0].content["candidate_id"]
        assert results[1].ok is False
        return Reply(
            "evaluate",
            [
                ToolCall("compile_candidate", {"candidate_id": cid}, "t7"),
                ToolCall("validate_correctness", {"candidate_id": cid}, "t8"),
                ToolCall("benchmark_candidate", {"candidate_id": cid}, "t9"),
            ],
            "tool_use",
        )

    provider = FakeProvider(
        [
            first,
            pipeline,
            gen,
            evaluate,
            Reply("done", [ToolCall("save_experiment", {"summary": "warps fixed spills"}, "t10")]),
        ]
    )
    out = AgentRunner(sess, LLMPlanner(provider), AgentBudget(max_candidates=3)).run(p)
    sess.finish()
    assert out.stop_reason == "saved by planner"
    assert out.speedup and out.speedup > 1.5
    assert "save_experiment" in str(provider.tools)
    assert kinds(store, sess.exp_id, "final")[0]["rationale"] == "warps fixed spills"


def test_planner_crash_still_saves(store, backend, settings, env):
    p = Problem("vector_add", (1 << 16,), "fp32")
    sess = new_session(store, backend, settings, env, p)
    provider = FakeProvider([RuntimeError("API 529 overloaded")])
    out = AgentRunner(sess, LLMPlanner(provider), AgentBudget(max_candidates=2)).run(p)
    sess.finish()
    assert "planner error" in out.stop_reason
    assert kinds(store, sess.exp_id, "final")
    assert out.best_id is not None  # baseline gets measured anyway


def test_idle_planner_stops(store, backend, settings, env):
    class Idle:
        name = "idle"

        def start(self, *a):
            pass

        def next_turn(self, results):
            return Turn("thinking about it...")

    p = Problem("vector_add", (4096,), "fp32")
    sess = new_session(store, backend, settings, env, p)
    out = AgentRunner(sess, Idle()).run(p)
    assert out.stop_reason == "planner stopped calling tools"
    assert kinds(store, sess.exp_id, "final")


def test_consecutive_errors_abort(store, backend, settings, env):
    p = Problem("vector_add", (4096,), "fp32")
    sess = new_session(store, backend, settings, env, p)
    bad = [
        Reply("oops", [ToolCall("compile_candidate", {"candidate_id": "missing"}, f"x{i}")])
        for i in range(10)
    ]
    out = AgentRunner(
        sess, LLMPlanner(FakeProvider(bad)), AgentBudget(max_consecutive_errors=3)
    ).run(p)
    assert "consecutive" in out.stop_reason
