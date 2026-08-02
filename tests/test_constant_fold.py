from ir import Graph, Op, build_decode_step_graph, weight_names
from passes.constant_fold import fold_constants


def test_real_decode_graph_has_no_all_constant_ops():
    graph = build_decode_step_graph()
    constants = set(weight_names(num_layers=1, layer=0).values())

    analyzed_graph, foldable = fold_constants(graph, constants)

    assert analyzed_graph is graph
    assert foldable == []


def test_all_constant_op_is_reported_and_grows_constant_set_transitively():
    graph = Graph(
        ops=[
            Op("sum_constants", "add", ["left", "right"], ["sum"], {}),
            Op("use_folded_sum", "add", ["sum", "bias"], ["output"], {}),
        ],
        shapes={name: ("n",) for name in ("left", "right", "bias", "sum", "output")},
        graph_inputs=["left", "right", "bias"],
        graph_outputs=["output"],
    )
    constants = {"left", "right", "bias"}

    analyzed_graph, foldable = fold_constants(graph, constants)

    assert analyzed_graph is graph
    assert foldable == ["sum_constants", "use_folded_sum"]
    assert constants == {"left", "right", "bias"}
