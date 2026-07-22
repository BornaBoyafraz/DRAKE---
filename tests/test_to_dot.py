from ir import build_decode_step_graph, make_dims
from passes.fusion import FusionPass


def test_to_dot_has_valid_graph_wrapper():
    dot = build_decode_step_graph().to_dot()

    assert dot.startswith("digraph")
    assert dot.count("{") == dot.count("}") == 1


def test_to_dot_contains_every_op_name():
    graph = build_decode_step_graph()
    dot = graph.to_dot()

    for op in graph.ops:
        assert op.name in dot


def test_to_dot_supports_multi_layer_graphs():
    graph = build_decode_step_graph(num_layers=3)
    dot = graph.to_dot()

    assert "l1_qkv_proj" in dot
    assert "l2_resid2" in dot


def test_to_dot_describes_fused_nodes_and_sub_ops():
    graph = build_decode_step_graph()
    fused_graph, _ = FusionPass().run(graph)
    dot = fused_graph.to_dot()
    fused_op = next(
        op for op in fused_graph.ops if op.attrs.get("fused_kind") == "fused_attention_kvupdate"
    )

    assert fused_op.name in dot
    assert "fused_attention_kvupdate" in dot
    assert "kv_cache_update -> attn_qk -> attn_softmax -> attn_av" in dot
    assert 'fillcolor="#fde68a"' in dot


def test_to_dot_annotates_edges_with_resolved_dimensions():
    graph = build_decode_step_graph()
    dims = make_dims(
        batch=2,
        seq_len=10,
        hidden_dim=32,
        n_heads=4,
        head_dim=8,
        ffn_dim=64,
    )
    dot = graph.to_dot(dims)

    assert 'label="x\\nshape=(2, 32)\\nelements=64"' in dot
