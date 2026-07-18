from ir import build_decode_step_graph, make_dims
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
