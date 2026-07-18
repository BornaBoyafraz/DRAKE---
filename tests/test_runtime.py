import numpy as np

from codegen.fused_ops import execute_graph, init_step_inputs, init_weights
from ir import build_decode_step_graph, make_dims
from passes.fusion import FusionPass
from runtime import DrakeEngine


def test_fusion_is_semantics_preserving():
    """The core compiler-correctness property: running the fused graph must
    produce bit-identical results to running the original, unfused graph."""
    graph = build_decode_step_graph()
    fused_graph, records = FusionPass().run(graph)
    assert records, "expected at least one fusion to have happened"

    dims = make_dims(batch=3, seq_len=17, hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=256)
    weights = init_weights(dims, seed=42)
    step_inputs = init_step_inputs(dims, seed=7)

    tensors_in = {**weights, **step_inputs}

    unfused_out = execute_graph(graph, tensors_in, dims)
    fused_out = execute_graph(fused_graph, tensors_in, dims)

    for name in ("output", "cache_k_out", "cache_v_out"):
        np.testing.assert_allclose(unfused_out[name], fused_out[name], rtol=1e-6, atol=1e-6)


def test_engine_runs_multi_step_decode_with_growing_cache():
    engine = DrakeEngine(hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=256)
    batch = 2
    cache_k = np.zeros((batch, 0, 4, 16), dtype=np.float32)
    cache_v = np.zeros((batch, 0, 4, 16), dtype=np.float32)
    rng = np.random.default_rng(0)

    for step_idx in range(5):
        x = (rng.standard_normal((batch, 64)) * 0.1).astype(np.float32)
        result = engine.step(x, cache_k, cache_v)
        assert result.output.shape == (batch, 64)
        assert result.cache_k_out.shape == (batch, step_idx + 1, 4, 16)
        assert result.cache_v_out.shape == (batch, step_idx + 1, 4, 16)
        cache_k, cache_v = result.cache_k_out, result.cache_v_out

    assert engine.profiler.total_calls() == 5


def test_engine_caches_kernel_plans_per_bucket():
    engine = DrakeEngine(hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=256)
    batch = 1
    cache_k = np.zeros((batch, 5, 4, 16), dtype=np.float32)
    cache_v = np.zeros((batch, 5, 4, 16), dtype=np.float32)
    x = np.zeros((batch, 64), dtype=np.float32)

    engine.step(x, cache_k, cache_v)
    assert len(engine.plan_cache) == 1
    engine.step(x, cache_k, cache_v)  # same bucket -> no new plan
    assert len(engine.plan_cache) == 1

    cache_k_big = np.zeros((batch, 5000, 4, 16), dtype=np.float32)
    cache_v_big = np.zeros((batch, 5000, 4, 16), dtype=np.float32)
    engine.step(x, cache_k_big, cache_v_big)  # different bucket -> new plan
    assert len(engine.plan_cache) == 2


def test_fusion_summary_reports_node_reduction():
    engine = DrakeEngine(hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=256)
    summary = engine.fusion_summary()
    assert summary["original_op_count"] == 16
    assert summary["fused_op_count"] == 10
    assert len(summary["fusions"]) == 3


def test_traffic_saved_bytes_grows_with_sequence_length():
    engine = DrakeEngine(hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=256)
    batch = 1
    x = np.zeros((batch, 64), dtype=np.float32)

    short = engine.step(x, np.zeros((batch, 4, 4, 16), dtype=np.float32), np.zeros((batch, 4, 4, 16), dtype=np.float32))
    long = engine.step(x, np.zeros((batch, 2000, 4, 16), dtype=np.float32), np.zeros((batch, 2000, 4, 16), dtype=np.float32))

    assert long.traffic_saved_bytes > short.traffic_saved_bytes
