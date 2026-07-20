from ir import Graph, Op, build_decode_step_graph, make_dims
from passes.fusion import FusionPass, traffic_saved_bytes


def _fused():
    graph = build_decode_step_graph()
    fused_graph, records = FusionPass().run(graph)
    return graph, fused_graph, records


def test_fusion_reduces_node_count():
    graph, fused_graph, records = _fused()
    assert len(graph.ops) == 16
    assert len(fused_graph.ops) == 10
    assert len(records) == 3


def test_expected_fusion_groups_present():
    _, fused_graph, records = _fused()
    kinds = {r.fused_kind for r in records}
    assert kinds == {"fused_norm_matmul", "fused_attention_kvupdate", "fused_norm_matmul_gelu"}
    kv_record = next(r for r in records if r.fused_kind == "fused_attention_kvupdate")
    assert kv_record.sub_op_names == ["kv_update", "qk", "softmax", "av"]


def test_unfused_ops_remain_between_fusion_groups():
    _, fused_graph, _ = _fused()
    kinds_in_order = [
        op.attrs.get("fused_kind", op.kind) if op.kind == "fused" else op.kind
        for op in fused_graph.ops
    ]
    assert kinds_in_order == [
        "fused_norm_matmul",
        "split_qkv",
        "rope",
        "fused_attention_kvupdate",
        "reshape",
        "matmul",
        "add",
        "fused_norm_matmul_gelu",
        "matmul",
        "add",
    ]


def test_traffic_saved_bytes_for_kv_fusion_matches_analytic_model():
    graph, fused_graph, _ = _fused()
    dims = make_dims(batch=2, seq_len=10, hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=128)
    kv_fused_op = next(
        op for op in fused_graph.ops if op.attrs.get("fused_kind") == "fused_attention_kvupdate"
    )

    next_seq_len = dims["seq_len"] + 1
    cache_elems = dims["batch"] * next_seq_len * dims["n_heads"] * dims["head_dim"]
    scores_elems = dims["batch"] * dims["n_heads"] * next_seq_len
    # cache_k_out/cache_v_out are graph outputs -> only the internal read is saved (1x each).
    # scores/probs are purely internal -> both write and read are saved (2x each).
    expected = (cache_elems * 4) + (cache_elems * 4) + 2 * (scores_elems * 4) + 2 * (scores_elems * 4)

    assert traffic_saved_bytes(kv_fused_op, graph, dims) == expected


def test_traffic_saved_bytes_zero_for_non_fused_op():
    _, fused_graph, _ = _fused()
    plain_op = next(op for op in fused_graph.ops if op.kind != "fused")
    dims = make_dims(batch=2, seq_len=10, hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=128)
    assert traffic_saved_bytes(plain_op, fused_graph, dims) == 0


def test_no_fusion_across_a_shared_intermediate_that_leaves_the_pattern():
    """cache_v_out feeds attn_av from two hops away (not the immediate next op);
    the pass must still recognize the chain via the connectivity check rather
    than only checking adjacent-op dataflow."""
    graph, fused_graph, records = _fused()
    kv_record = next(r for r in records if r.fused_kind == "fused_attention_kvupdate")
    assert "kv_update" in kv_record.sub_op_names
    assert "av" in kv_record.sub_op_names


def test_cost_optimal_matches_greedy_on_the_real_graph():
    """On DRAKE's actual pattern table, every overlap is a strict
    superset (the 3-op norm+matmul+gelu group is a strict extension of the
    2-op norm+matmul group, so it can never score lower). Greedy and the DP
    selector should therefore agree exactly here -- this is what makes
    greedy a defensible default rather than a shortcut."""
    graph = build_decode_step_graph()
    dims = make_dims(batch=4, seq_len=200, hidden_dim=256, n_heads=8, head_dim=32, ffn_dim=1024)

    greedy_graph, greedy_records = FusionPass().run(graph)
    dp_graph, dp_records = FusionPass().run_cost_optimal(graph, dims)

    assert [r.fused_kind for r in greedy_records] == [r.fused_kind for r in dp_records]
    greedy_total = sum(traffic_saved_bytes(op, graph, dims) for op in greedy_graph.ops)
    dp_total = sum(traffic_saved_bytes(op, graph, dims) for op in dp_graph.ops)
    assert greedy_total == dp_total


def _adversarial_conflict_graph() -> Graph:
    """A synthetic 4-op chain (in0 -k1-> t01 -k2-> t12 -k3-> t23 -k4-> out3)
    with two competing, overlapping candidate patterns:

      - a 3-op group over (k1, k2, k3), which only elides two tiny tensors
      - a 2-op group over (k3, k4), which elides one huge tensor

    Both need op index 2 (kind k3), so only one can be taken. This is
    exactly the shape of conflict greedy-longest-first cannot reason about:
    it always prefers the 3-op group because it's longer, regardless of
    which grouping actually saves more traffic.
    """
    shapes = {
        "in0": ("small",),
        "t01": ("small",),
        "t12": ("small",),
        "t23": ("big",),
        "out3": ("big",),
    }
    ops = [
        Op("op0", "k1", ["in0"], ["t01"], {}),
        Op("op1", "k2", ["t01"], ["t12"], {}),
        Op("op2", "k3", ["t12"], ["t23"], {}),
        Op("op3", "k4", ["t23"], ["out3"], {}),
    ]
    return Graph(ops=ops, shapes=shapes, graph_inputs=["in0"], graph_outputs=["out3"])


def test_cost_optimal_beats_greedy_on_a_constructed_conflict():
    """The counter-example greedy-longest-first cannot handle: DP must find
    the higher-scoring 2-op grouping that greedy skips past because it
    commits to the longer 3-op match first."""
    graph = _adversarial_conflict_graph()
    dims = {"small": 1, "big": 1000}
    patterns = [
        (("k1", "k2", "k3"), "big3"),  # longer, but only elides two 4-byte tensors
        (("k3", "k4"), "small2"),  # shorter, but elides one 4000-byte tensor
    ]

    greedy_graph, greedy_records = FusionPass().run(graph, patterns=patterns)
    dp_graph, dp_records = FusionPass().run_cost_optimal(graph, dims, patterns=patterns)

    greedy_total = sum(traffic_saved_bytes(op, graph, dims) for op in greedy_graph.ops)
    dp_total = sum(traffic_saved_bytes(op, graph, dims) for op in dp_graph.ops)

    # Greedy commits to the 3-op group (t01, t12 elided: 2*4 + 2*4 = 16 bytes)
    # and leaves op3 stranded unfused.
    assert [r.fused_kind for r in greedy_records] == ["big3"]
    assert greedy_total == 16

    # DP skips the 3-op group entirely and takes the 2-op group instead
    # (t23 elided: 2*4000 = 8000 bytes) -- strictly better.
    assert [r.fused_kind for r in dp_records] == ["small2"]
    assert dp_total == 8000

    assert dp_total > greedy_total
