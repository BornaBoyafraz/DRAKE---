from ir import build_decode_step_graph, cache_io_names, make_dims, num_elements, resolve_shape, weight_names


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


def test_multi_layer_single_layer_is_byte_identical_to_default():
    """num_layers=1 must reproduce exactly today's single-layer graph --
    every caller written before multi-layer support existed depends on
    this (unprefixed tensor names, same op count, same op names)."""
    default_graph = build_decode_step_graph()
    explicit_graph = build_decode_step_graph(num_layers=1)
    assert explicit_graph.graph_inputs == default_graph.graph_inputs
    assert explicit_graph.graph_outputs == default_graph.graph_outputs
    assert explicit_graph.shapes == default_graph.shapes
    assert [(op.name, op.kind, op.inputs, op.outputs, op.attrs) for op in explicit_graph.ops] == [
        (op.name, op.kind, op.inputs, op.outputs, op.attrs) for op in default_graph.ops
    ]


def test_multi_layer_op_and_io_counts_scale_linearly():
    for num_layers in (1, 2, 5):
        graph = build_decode_step_graph(num_layers=num_layers)
        assert len(graph.ops) == 16 * num_layers
        # 1 shared "x" input, 2 cache tensors per layer
        assert len(graph.graph_inputs) == 1 + 2 * num_layers
        # 1 shared "output", 2 cache tensors per layer
        assert len(graph.graph_outputs) == 1 + 2 * num_layers


def test_multi_layer_residual_stream_chains_layers_in_order():
    graph = build_decode_step_graph(num_layers=3)
    # layer i's final "add" must write l{i}_output (or "output" for the last
    # layer) and layer i+1's first "rmsnorm" must read that same tensor.
    ops_by_layer = [graph.ops[i * 16 : (i + 1) * 16] for i in range(3)]
    assert ops_by_layer[0][-1].outputs == ["l0_output"]
    assert ops_by_layer[1][0].inputs[0] == "l0_output"
    assert ops_by_layer[1][-1].outputs == ["l1_output"]
    assert ops_by_layer[2][0].inputs[0] == "l1_output"
    assert ops_by_layer[2][-1].outputs == ["output"]


def test_multi_layer_each_layer_has_independent_kv_cache_and_weights():
    graph = build_decode_step_graph(num_layers=2)
    for layer in (0, 1):
        k_in, v_in, k_out, v_out = cache_io_names(2, layer)
        assert k_in in graph.graph_inputs and v_in in graph.graph_inputs
        assert k_out in graph.graph_outputs and v_out in graph.graph_outputs
        for tensor_name in weight_names(2, layer).values():
            assert tensor_name in graph.shapes
    # no name collisions between layers
    names0 = set(weight_names(2, 0).values())
    names1 = set(weight_names(2, 1).values())
    assert names0.isdisjoint(names1)


def test_build_decode_step_graph_rejects_zero_layers():
    try:
        build_decode_step_graph(num_layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for num_layers=0")
