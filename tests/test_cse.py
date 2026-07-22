import numpy as np

from codegen.fused_ops import execute_graph
from ir import Graph, Op, build_decode_step_graph, make_dims
from passes.cse import eliminate_common_subexpressions
from passes.verify import verify_graph


def test_cse_is_a_noop_on_the_real_decode_graph():
    """Every op in the hand-built graph is distinct; CSE removes nothing."""
    graph = build_decode_step_graph()
    deduped, removed = eliminate_common_subexpressions(graph)
    assert removed == []
    assert [op.name for op in deduped.ops] == [op.name for op in graph.ops]


def test_cse_is_a_noop_on_multi_layer_graph():
    graph = build_decode_step_graph(num_layers=3)
    _, removed = eliminate_common_subexpressions(graph)
    assert removed == []


def _graph_with_duplicate_op() -> Graph:
    """Two identical ops (`a` and `a_dup`: same kind, same input, same attrs)
    feed two consumers; CSE should keep `a`, drop `a_dup`, and rewire."""
    shapes = {
        "x": ("n",),
        "t_a": ("n",),
        "t_a_dup": ("n",),
        "left": ("n",),
        "right": ("n",),
        "out": ("n",),
    }
    ops = [
        Op("a", "gelu", ["x"], ["t_a"], {}),
        Op("a_dup", "gelu", ["x"], ["t_a_dup"], {}),  # identical to 'a'
        # Distinct consumer kinds so only 'a_dup' merges (the consumers
        # themselves are not duplicates of each other).
        Op("use_left", "op_l", ["t_a"], ["left"], {}),
        Op("use_right", "op_r", ["t_a_dup"], ["right"], {}),  # reads the dup
        Op("combine", "add", ["left", "right"], ["out"], {}),
    ]
    return Graph(ops=ops, shapes=shapes, graph_inputs=["x"], graph_outputs=["out"])


def test_cse_merges_a_duplicate_op_and_rewires_consumers():
    graph = _graph_with_duplicate_op()
    deduped, removed = eliminate_common_subexpressions(graph)

    assert removed == ["a_dup"]
    # 'use_right' must now read t_a (the canonical), not t_a_dup
    use_right = next(op for op in deduped.ops if op.name == "use_right")
    assert use_right.inputs == ["t_a"]
    # t_a_dup is gone from the graph and its shape is pruned
    assert "t_a_dup" not in deduped.shapes
    verify_graph(deduped)


def test_cse_does_not_merge_ops_differing_in_attrs():
    shapes = {"x": ("n",), "a": ("n",), "b": ("n",), "out": ("n",)}
    ops = [
        Op("n1", "rmsnorm", ["x"], ["a"], {"eps": 1e-6}),
        Op("n2", "rmsnorm", ["x"], ["b"], {"eps": 1e-5}),  # different eps
        Op("c", "add", ["a", "b"], ["out"], {}),
    ]
    graph = Graph(ops=ops, shapes=shapes, graph_inputs=["x"], graph_outputs=["out"])
    _, removed = eliminate_common_subexpressions(graph)
    assert removed == []


def test_cse_preserves_graph_output_names():
    """An op producing a declared graph output is never removed, so output
    tensor names stay stable even when it duplicates an earlier op."""
    shapes = {"x": ("n",), "scratch": ("n",), "out": ("n",)}
    ops = [
        Op("first", "gelu", ["x"], ["scratch"], {}),
        Op("second", "gelu", ["x"], ["out"], {}),  # same computation, but 'out' is a graph output
    ]
    graph = Graph(ops=ops, shapes=shapes, graph_inputs=["x"], graph_outputs=["out"])
    deduped, removed = eliminate_common_subexpressions(graph)
    assert removed == []
    assert deduped.graph_outputs == ["out"]


def test_cse_chains_collapse():
    """After a->a' dedup, downstream ops that become identical must dedup too."""
    shapes = {"x": ("n",), "a": ("n",), "a2": ("n",), "b": ("n",), "b2": ("n",), "out": ("n",)}
    ops = [
        Op("a", "gelu", ["x"], ["a"], {}),
        Op("a2", "gelu", ["x"], ["a2"], {}),  # dup of a
        Op("b", "op", ["a"], ["b"], {}),
        Op("b2", "op", ["a2"], ["b2"], {}),  # becomes dup of b after a2->a
        Op("combine", "add", ["b", "b2"], ["out"], {}),
    ]
    graph = Graph(ops=ops, shapes=shapes, graph_inputs=["x"], graph_outputs=["out"])
    deduped, removed = eliminate_common_subexpressions(graph)
    assert set(removed) == {"a2", "b2"}
    combine = next(op for op in deduped.ops if op.name == "combine")
    assert combine.inputs == ["b", "b"]
    verify_graph(deduped)


def test_cse_is_semantics_preserving():
    """Insert a redundant duplicate of a real op into the decode graph, run
    CSE, and confirm the executor produces identical output."""
    graph = build_decode_step_graph()
    dims = make_dims(batch=2, seq_len=6, hidden_dim=32, n_heads=4, head_dim=8, ffn_dim=64)

    # Duplicate norm1 (rmsnorm(x, w_norm1) -> x_norm, hidden_dim wide), then
    # fold the duplicate back into the residual stream so it stays live (a
    # dead dup would be removed by DCE instead of CSE). x_norm_dup and output
    # are both (batch, hidden_dim), so the extra add is shape-valid.
    ops = list(graph.ops)
    insert_at = next(i for i, op in enumerate(ops) if op.name == "norm1") + 1
    ops.insert(insert_at, Op("norm1_dup", "rmsnorm", ["x", "w_norm1"], ["x_norm_dup"], {"eps": 1e-6}))
    shapes = dict(graph.shapes)
    shapes["x_norm_dup"] = shapes["x_norm"]
    ops.append(Op("noop_add", "add", ["output", "x_norm_dup"], ["output_plus"], {}))
    shapes["output_plus"] = shapes["output"]
    dirty = Graph(
        ops=ops,
        shapes=shapes,
        graph_inputs=list(graph.graph_inputs),
        graph_outputs=["output_plus", "cache_k_out", "cache_v_out"],
    )
    verify_graph(dirty)

    deduped, removed = eliminate_common_subexpressions(dirty)
    assert "norm1_dup" in removed

    from codegen.fused_ops import init_step_inputs, init_weights

    tensors_in = {**init_weights(dims, seed=3), **init_step_inputs(dims, seed=5)}
    before = execute_graph(dirty, tensors_in, dims)
    after = execute_graph(deduped, tensors_in, dims)
    for name in ("output_plus", "cache_k_out", "cache_v_out"):
        np.testing.assert_array_equal(before[name], after[name])
