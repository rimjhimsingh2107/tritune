from pathlib import Path

from kernelpilot.analysis.ncu import parse_ncu_csv
from kernelpilot.analysis.ptx import summarize_ptx
from kernelpilot.backends.triton_engine import NCU_METRICS

FIX = Path(__file__).parent / "fixtures"


def test_ptx_summary_counts():
    s = summarize_ptx((FIX / "softmax_kernel.ptx").read_text())
    assert s.entries == ["softmax_kernel"]
    assert s.target == "sm_80"
    assert s.global_loads == 3
    assert s.vector_global_loads == 2
    assert s.global_stores == 1
    assert s.shared_loads == 1 and s.shared_stores == 1
    assert s.local_loads == 1 and s.local_stores == 1
    assert s.barriers == 1
    assert s.ex2 == 2
    assert s.declared_regs["f32"] == 57
    notes = " ".join(s.observations())
    assert "spills" in notes


def test_ncu_csv_parse_skips_chatter_and_reads_units():
    metrics, units = parse_ncu_csv((FIX / "ncu_raw.csv").read_text(), NCU_METRICS)
    assert metrics["dram_throughput_pct"] == 61.37
    assert metrics["duration_ns"] == 1412544
    assert metrics["registers_per_thread"] == 168
    assert metrics["shared_mem_per_block"] == 98304
    assert metrics["achieved_occupancy_pct"] == 23.94
    assert units["duration_ns"] == "nsecond"


def test_ncu_parse_empty():
    assert parse_ncu_csv("no csv here", NCU_METRICS) == ({}, {})
