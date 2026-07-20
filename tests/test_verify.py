import pytest

from ir import Graph, Op, build_decode_step_graph
from passes.fusion import FusionPass
from passes.verify import GraphVerificationError, collect_graph_errors, verify_graph


def test_valid_single_layer_graph_passes():
    verify_graph(build_decode_step_graph())  # must not raise


def test_valid_multi_layer_graph_passes():
    for num_layers in (1, 2, 5):
        verify_graph(build_decode_step_graph(num_layers=num_layers))


def test_fused_graph_still_verifies():
    """Fusion must produce a graph that passes the same invariants as the
    unfused one -- this is a structural safety net on top of the numeric
    semantics-preservation test."""
    graph = build_decode_step_graph(num_layers=3)
    fused_graph, _ = FusionPass().run(graph)
    verify_graph(fused_graph)


def test_dangling_reference_without_shape_is_caught():
    """A reference to a tensor that is neither produced nor declared with a
    shape is a genuine dangling reference (a typo), caught as a missing
    shape. A reference to a shaped-but-unproduced tensor, by contrast, is a
    legal parameter (see test_weights_are_treated_as_parameters)."""
    graph = Graph(
        ops=[Op("a", "add", ["x", "ghost"], ["y"], {})],
        shapes={"x": ("n",), "y": ("n",)},  # 'ghost' has no shape -> dangling
        graph_inputs=["x"],
        graph_outputs=["y"],
    )
    errors = collect_graph_errors(graph)
    assert any("'ghost'" in e and "graph.shapes" in e for e in errors)


def test_weights_are_treated_as_parameters():
    """Tensors with a shape but no producer (model weights) are valid
    external values, not errors -- this is exactly how the real decode
    graph references w_qkv, w_norm1, etc."""
    graph = Graph(
        ops=[Op("a", "matmul", ["x", "w"], ["y"], {})],
        shapes={"x": ("n",), "w": ("n", "m"), "y": ("m",)},
        graph_inputs=["x"],  # 'w' is a parameter, not a graph input
        graph_outputs=["y"],
    )
    assert collect_graph_errors(graph) == []


def test_duplicate_producer_is_caught():
    graph = Graph(
        ops=[
            Op("a", "op", ["x"], ["y"], {}),
            Op("b", "op", ["x"], ["y"], {}),  # y produced twice
        ],
        shapes={"x": ("n",), "y": ("n",)},
        graph_inputs=["x"],
        graph_outputs=["y"],
    )
    errors = collect_graph_errors(graph)
    assert any("single-assignment" in e for e in errors)


def test_use_before_def_is_caught():
    graph = Graph(
        ops=[
            Op("a", "op", ["later"], ["y"], {}),  # reads 'later' before it's made
            Op("b", "op", ["x"], ["later"], {}),
        ],
        shapes={"x": ("n",), "later": ("n",), "y": ("n",)},
        graph_inputs=["x"],
        graph_outputs=["y"],
    )
    errors = collect_graph_errors(graph)
    assert any("before it is produced" in e for e in errors)


def test_unproduced_graph_output_is_caught():
    graph = Graph(
        ops=[Op("a", "op", ["x"], ["y"], {})],
        shapes={"x": ("n",), "y": ("n",), "z": ("n",)},
        graph_inputs=["x"],
        graph_outputs=["z"],  # z is never produced
    )
    errors = collect_graph_errors(graph)
    assert any("graph output 'z'" in e for e in errors)


def test_missing_shape_is_caught():
    graph = Graph(
        ops=[Op("a", "op", ["x"], ["y"], {})],
        shapes={"x": ("n",)},  # y has no shape
        graph_inputs=["x"],
        graph_outputs=["y"],
    )
    errors = collect_graph_errors(graph)
    assert any("'y'" in e and "graph.shapes" in e for e in errors)


def test_redefining_a_graph_input_is_caught():
    graph = Graph(
        ops=[Op("a", "op", ["x"], ["x"], {})],  # outputs a graph input
        shapes={"x": ("n",)},
        graph_inputs=["x"],
        graph_outputs=["x"],
    )
    errors = collect_graph_errors(graph)
    assert any("may not be redefined" in e for e in errors)


def test_verify_graph_raises_with_all_errors():
    graph = Graph(
        ops=[Op("a", "op", ["missing"], ["y"], {})],
        shapes={"y": ("n",)},  # both 'missing' unproduced AND no shape for it
        graph_inputs=[],
        graph_outputs=["y"],
    )
    with pytest.raises(GraphVerificationError) as exc:
        verify_graph(graph)
    assert len(exc.value.errors) >= 1
