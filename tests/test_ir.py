from ir import build_decode_step_graph, make_dims, num_elements, resolve_shape


def test_graph_has_expected_op_count():
    graph = build_decode_step_graph()
    assert len(graph.ops) == 16


def test_graph_inputs_outputs_present_in_shapes():
    graph = build_decode_step_graph()
    for name in graph.graph_inputs + graph.graph_outputs:
        assert name in graph.shapes


def test_resolve_shape_and_num_elements():
    dims = make_dims(batch=2, seq_len=10, hidden_dim=64, n_heads=4, head_dim=16, ffn_dim=128)
    shape = ("batch", "seq_len", "n_heads", "head_dim")
    assert resolve_shape(shape, dims) == (2, 10, 4, 16)
    assert num_elements(shape, dims) == 2 * 10 * 4 * 16


def test_make_dims_rejects_inconsistent_heads():
    try:
        make_dims(batch=1, seq_len=0, hidden_dim=64, n_heads=3, head_dim=16, ffn_dim=128)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for n_heads * head_dim != hidden_dim")


def test_consumer_count_counts_graph_outputs():
    graph = build_decode_step_graph()
    # cache_k_out is both consumed nowhere further AND is a graph output
    assert graph.consumer_count("cache_k_out") >= 1
    # x is consumed by norm1 and by the residual add
    assert graph.consumer_count("x") == 2
