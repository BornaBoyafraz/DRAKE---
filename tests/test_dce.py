import numpy as np

from codegen.fused_ops import execute_graph, init_step_inputs, init_weights
from ir import Graph, Op, build_decode_step_graph, make_dims
from passes.dce import eliminate_dead_code
from passes.verify import verify_graph


def test_dce_is_a_noop_on_the_real_decode_graph():
    """Nothing in the hand-built decode graph is dead; DCE must leave it
    untouched (same op names, same order)."""
    graph = build_decode_step_graph()
    pruned, removed = eliminate_dead_code(graph)
    assert removed == []
    assert [op.name for op in pruned.ops] == [op.name for op in graph.ops]


def test_dce_is_a_noop_on_multi_layer_graph():
    graph = build_decode_step_graph(num_layers=3)
    pruned, removed = eliminate_dead_code(graph)
    assert removed == []
    assert len(pruned.ops) == len(graph.ops)


def _graph_with_dead_branch() -> Graph:
    """Live chain x -> a -> out, plus a dead branch x -> b -> c that nothing
    downstream of a graph output ever consumes."""
    shapes = {
        "x": ("n",),
        "t_a": ("n",),
        "out": ("n",),
        "t_b": ("n",),
        "t_c": ("n",),
    }
    ops = [
        Op("a", "op", ["x"], ["t_a"], {}),
        Op("b", "op", ["x"], ["t_b"], {}),  # dead
        Op("c", "op", ["t_b"], ["t_c"], {}),  # dead
        Op("finalize", "op", ["t_a"], ["out"], {}),
    ]
    return Graph(ops=ops, shapes=shapes, graph_inputs=["x"], graph_outputs=["out"])


def test_dce_removes_a_dead_branch():
    graph = _graph_with_dead_branch()
    pruned, removed = eliminate_dead_code(graph)
    assert set(removed) == {"b", "c"}
    assert [op.name for op in pruned.ops] == ["a", "finalize"]
    # order of survivors preserved
    verify_graph(pruned)


def test_dce_prunes_shapes_of_removed_tensors():
    graph = _graph_with_dead_branch()
    pruned, _ = eliminate_dead_code(graph)
    assert "t_b" not in pruned.shapes
    assert "t_c" not in pruned.shapes
    assert {"x", "t_a", "out"} <= set(pruned.shapes)


def test_dce_keeps_an_op_if_any_output_is_live():
    """An op with one live and one dead output must be kept whole."""
    shapes = {"x": ("n",), "live": ("n",), "dead": ("n",)}
    ops = [Op("multi", "op", ["x"], ["live", "dead"], {})]
    graph = Graph(ops=ops, shapes=shapes, graph_inputs=["x"], graph_outputs=["live"])
    pruned, removed = eliminate_dead_code(graph)
    assert removed == []
    assert len(pruned.ops) == 1


def test_dce_is_idempotent():
    graph = _graph_with_dead_branch()
    once, _ = eliminate_dead_code(graph)
    twice, removed_second = eliminate_dead_code(once)
    assert removed_second == []
    assert [op.name for op in once.ops] == [op.name for op in twice.ops]


def test_dce_is_semantics_preserving_when_dead_code_is_appended():
    """Append a genuinely dead op onto the real decode graph, run DCE, and
    confirm the executor produces identical outputs with and without it."""
    graph = build_decode_step_graph()
    dims = make_dims(batch=2, seq_len=6, hidden_dim=32, n_heads=4, head_dim=8, ffn_dim=64)

    # A dead op: reads a real tensor, writes a scratch nothing consumes.
    dead = Op("dead_scratch", "gelu", ["output"], ["scratch"], {})
    shapes = dict(graph.shapes)
    shapes["scratch"] = shapes["output"]
    dirty = Graph(
        ops=graph.ops + [dead],
        shapes=shapes,
        graph_inputs=list(graph.graph_inputs),
        graph_outputs=list(graph.graph_outputs),
    )

    pruned, removed = eliminate_dead_code(dirty)
    assert removed == ["dead_scratch"]

    weights = init_weights(dims, seed=3)
    step_inputs = init_step_inputs(dims, seed=5)
    tensors_in = {**weights, **step_inputs}

    baseline = execute_graph(graph, tensors_in, dims)
    after_dce = execute_graph(pruned, tensors_in, dims)
    for name in ("output", "cache_k_out", "cache_v_out"):
        np.testing.assert_array_equal(baseline[name], after_dce[name])
