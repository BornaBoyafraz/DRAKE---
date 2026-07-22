"""Minimal graph IR for a single transformer decode step.

This is intentionally not a general tensor-program IR (that's what MLIR/XLA/TVM
are for). It models exactly the op sequence in one autoregressive decode step
of a transformer layer, which is the workload DRAKE's fusion and
specialization passes target.

A tensor's shape is a tuple of *symbolic* dimension names (strings) resolved
against a concrete `dims` dict at analysis/execution time, e.g. shape
``("batch", "hidden_dim")`` resolves to ``(4, 4096)`` given
``dims = {"batch": 4, "hidden_dim": 4096, ...}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Tuple


Shape = Tuple[str, ...]
Dims = Dict[str, int]


def resolve_shape(shape: Shape, dims: Dims) -> Tuple[int, ...]:
    try:
        return tuple(dims[d] for d in shape)
    except KeyError as e:
        raise KeyError(f"dimension {e.args[0]!r} not present in dims={dims!r}") from e


def num_elements(shape: Shape, dims: Dims) -> int:
    resolved = resolve_shape(shape, dims)
    return prod(resolved) if resolved else 1


@dataclass
class Op:
    name: str
    kind: str
    inputs: List[str]
    outputs: List[str]
    attrs: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        attrs = f" {self.attrs}" if self.attrs else ""
        return f"{outs} = {self.kind}({ins}){attrs}"


@dataclass
class Graph:
    ops: List[Op]
    shapes: Dict[str, Shape]
    graph_inputs: List[str]
    graph_outputs: List[str]
    dtype_bytes: int = 4  # fp32

    def consumer_count(self, tensor: str) -> int:
        return sum(1 for op in self.ops if tensor in op.inputs) + (
            1 if tensor in self.graph_outputs else 0
        )

    def producer_index(self) -> Dict[str, int]:
        idx = {}
        for i, op in enumerate(self.ops):
            for out in op.outputs:
                idx[out] = i
        return idx

    def bytes_of(self, tensor: str, dims: Dims) -> int:
        return num_elements(self.shapes[tensor], dims) * self.dtype_bytes

    def to_dot(self, dims: Dims | None = None) -> str:
        """Return a deterministic Graphviz DOT rendering of the graph."""

        def quote(value: str) -> str:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{escaped}"'

        def edge_label(tensor: str) -> str:
            if dims is None:
                return tensor
            shape = resolve_shape(self.shapes[tensor], dims)
            elements = num_elements(self.shapes[tensor], dims)
            return f"{tensor}\nshape={shape}\nelements={elements}"

        lines = ['digraph DRAKE {', '  rankdir="LR";']
        input_nodes = {tensor: f"input_{i}" for i, tensor in enumerate(self.graph_inputs)}
        output_nodes = {tensor: f"output_{i}" for i, tensor in enumerate(self.graph_outputs)}
        op_nodes = [f"op_{i}" for i in range(len(self.ops))]

        for tensor in self.graph_inputs:
            label = quote(f"input\n{tensor}")
            lines.append(
                f'  {input_nodes[tensor]} [shape="ellipse", style="filled", '
                f'fillcolor="#dbeafe", label={label}];'
            )

        for i, op in enumerate(self.ops):
            label_parts = [op.name, op.kind]
            if op.kind == "fused":
                fused_kind = str(op.attrs.get("fused_kind", ""))
                sub_op_kinds = [sub_op.kind for sub_op in op.attrs.get("sub_ops", [])]
                label_parts.extend(
                    [f"fused_kind: {fused_kind}", f"sub-ops: {' -> '.join(sub_op_kinds)}"]
                )
                style = 'style="filled", fillcolor="#fde68a", color="#b45309", '
            else:
                style = ""
            label = quote("\n".join(label_parts))
            lines.append(
                f'  {op_nodes[i]} [shape="box", {style}label={label}];'
            )

        for tensor in self.graph_outputs:
            label = quote(f"output\n{tensor}")
            lines.append(
                f'  {output_nodes[tensor]} [shape="ellipse", style="filled", '
                f'fillcolor="#dcfce7", label={label}];'
            )

        producer_nodes = {
            tensor: op_nodes[i] for i, op in enumerate(self.ops) for tensor in op.outputs
        }
        for i, op in enumerate(self.ops):
            target = op_nodes[i]
            for tensor in op.inputs:
                source = producer_nodes.get(tensor, input_nodes.get(tensor))
                if source is not None:
                    lines.append(f"  {source} -> {target} [label={quote(edge_label(tensor))}];")

        for tensor in self.graph_outputs:
            source = producer_nodes.get(tensor, input_nodes.get(tensor))
            if source is not None:
                lines.append(
                    f"  {source} -> {output_nodes[tensor]} "
                    f"[label={quote(edge_label(tensor))}];"
                )

        lines.append("}")
        return "\n".join(lines)

    def dump(self) -> str:
        lines = [f"graph({', '.join(self.graph_inputs)}) -> ({', '.join(self.graph_outputs)}) {{"]
        for op in self.ops:
            lines.append(f"  {op!r}")
        lines.append("}")
        return "\n".join(lines)


def make_dims(
    *,
    batch: int,
    seq_len: int,
    hidden_dim: int,
    n_heads: int,
    head_dim: int,
    ffn_dim: int,
) -> Dims:
    assert n_heads * head_dim == hidden_dim, "n_heads * head_dim must equal hidden_dim"
    return {
        "batch": batch,
        "seq_len": seq_len,
        "next_seq_len": seq_len + 1,
        "hidden_dim": hidden_dim,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "qkv_dim": 3 * hidden_dim,
        "ffn_dim": ffn_dim,
    }


def layer_prefix(layer: int, num_layers: int) -> str:
    """Tensor/op-name prefix for a given layer. Empty for the single-layer
    case so ``build_decode_step_graph(num_layers=1)`` reproduces the exact
    unprefixed tensor names every existing test and caller depends on."""
    return f"l{layer}_" if num_layers > 1 else ""


def cache_io_names(num_layers: int, layer: int) -> Tuple[str, str, str, str]:
    """(cache_k_in, cache_v_in, cache_k_out, cache_v_out) tensor names for `layer`."""
    p = layer_prefix(layer, num_layers)
    return f"{p}cache_k_in", f"{p}cache_v_in", f"{p}cache_k_out", f"{p}cache_v_out"


def weight_names(num_layers: int, layer: int) -> Dict[str, str]:
    """Map logical weight name -> graph tensor name for `layer`."""
    p = layer_prefix(layer, num_layers)
    return {
        "w_norm1": f"{p}w_norm1",
        "w_qkv": f"{p}w_qkv",
        "w_o": f"{p}w_o",
        "w_norm2": f"{p}w_norm2",
        "w_up": f"{p}w_up",
        "w_down": f"{p}w_down",
    }


def _layer_ops(prefix: str, input_name: str, output_name: str) -> Tuple[List[Op], Dict[str, Shape]]:
    """Build one decode-step transformer layer: rmsnorm, QKV projection, RoPE,
    KV-cache append, attention, output projection, FFN. `input_name` /
    `output_name` are the residual-stream tensors that connect layers; every
    other tensor and op name is local to this layer, prefixed by `prefix`.
    """

    def n(local: str) -> str:
        return f"{prefix}{local}" if prefix else local

    shapes: Dict[str, Shape] = {
        input_name: ("batch", "hidden_dim"),
        n("w_norm1"): ("hidden_dim",),
        n("x_norm"): ("batch", "hidden_dim"),
        n("w_qkv"): ("hidden_dim", "qkv_dim"),
        n("qkv"): ("batch", "qkv_dim"),
        n("q"): ("batch", "n_heads", "head_dim"),
        n("k_new"): ("batch", "n_heads", "head_dim"),
        n("v_new"): ("batch", "n_heads", "head_dim"),
        n("q_rot"): ("batch", "n_heads", "head_dim"),
        n("k_rot"): ("batch", "n_heads", "head_dim"),
        n("cache_k_in"): ("batch", "seq_len", "n_heads", "head_dim"),
        n("cache_v_in"): ("batch", "seq_len", "n_heads", "head_dim"),
        n("cache_k_out"): ("batch", "next_seq_len", "n_heads", "head_dim"),
        n("cache_v_out"): ("batch", "next_seq_len", "n_heads", "head_dim"),
        n("scores"): ("batch", "n_heads", "next_seq_len"),
        n("probs"): ("batch", "n_heads", "next_seq_len"),
        n("attn_out"): ("batch", "n_heads", "head_dim"),
        n("attn_out_flat"): ("batch", "hidden_dim"),
        n("w_o"): ("hidden_dim", "hidden_dim"),
        n("attn_proj"): ("batch", "hidden_dim"),
        n("resid1"): ("batch", "hidden_dim"),
        n("w_norm2"): ("hidden_dim",),
        n("resid1_norm"): ("batch", "hidden_dim"),
        n("w_up"): ("hidden_dim", "ffn_dim"),
        n("ff_hidden"): ("batch", "ffn_dim"),
        n("ff_act"): ("batch", "ffn_dim"),
        n("w_down"): ("ffn_dim", "hidden_dim"),
        n("ff_out"): ("batch", "hidden_dim"),
        output_name: ("batch", "hidden_dim"),
    }

    ops = [
        Op(n("norm1"), "rmsnorm", [input_name, n("w_norm1")], [n("x_norm")], {"eps": 1e-6}),
        Op(n("qkv_proj"), "matmul", [n("x_norm"), n("w_qkv")], [n("qkv")], {}),
        Op(n("split"), "split_qkv", [n("qkv")], [n("q"), n("k_new"), n("v_new")], {}),
        Op(n("rope"), "rope", [n("q"), n("k_new")], [n("q_rot"), n("k_rot")], {"position": "seq_len"}),
        Op(
            n("kv_update"),
            "kv_cache_update",
            [n("k_rot"), n("v_new"), n("cache_k_in"), n("cache_v_in")],
            [n("cache_k_out"), n("cache_v_out")],
            {},
        ),
        Op(n("qk"), "attn_qk", [n("q_rot"), n("cache_k_out")], [n("scores")], {}),
        Op(n("softmax"), "attn_softmax", [n("scores")], [n("probs")], {}),
        Op(n("av"), "attn_av", [n("probs"), n("cache_v_out")], [n("attn_out")], {}),
        Op(n("flatten"), "reshape", [n("attn_out")], [n("attn_out_flat")], {}),
        Op(n("o_proj"), "matmul", [n("attn_out_flat"), n("w_o")], [n("attn_proj")], {}),
        Op(n("resid1"), "add", [input_name, n("attn_proj")], [n("resid1")], {}),
        Op(n("norm2"), "rmsnorm", [n("resid1"), n("w_norm2")], [n("resid1_norm")], {"eps": 1e-6}),
        Op(n("up_proj"), "matmul", [n("resid1_norm"), n("w_up")], [n("ff_hidden")], {}),
        Op(n("act"), "gelu", [n("ff_hidden")], [n("ff_act")], {}),
        Op(n("down_proj"), "matmul", [n("ff_act"), n("w_down")], [n("ff_out")], {}),
        Op(n("resid2"), "add", [n("resid1"), n("ff_out")], [output_name], {}),
    ]
    return ops, shapes


def build_decode_step_graph(num_layers: int = 1) -> Graph:
    """One decode-step across `num_layers` stacked transformer layers, as a
    DRAKE Graph. `num_layers=1` (the default) reproduces exactly the original
    single-layer graph -- unprefixed tensor/op names, identical shapes -- so
    every caller that predates multi-layer support is unaffected.

    For `num_layers > 1`, each layer gets its own KV cache
    (`l{i}_cache_k_in/out`, `l{i}_cache_v_in/out`) and weights
    (`l{i}_w_*`), threaded together along the residual stream: layer i's
    output feeds layer i+1's input, with the first layer's input named `x`
    and the last layer's output named `output`.

    Shapes use symbolic dims: batch, seq_len (existing KV-cache length),
    next_seq_len (= seq_len + 1, cache length after this step's append),
    hidden_dim, n_heads, head_dim, qkv_dim, ffn_dim.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    all_ops: List[Op] = []
    all_shapes: Dict[str, Shape] = {}
    graph_inputs = ["x"]
    graph_outputs = ["output"]

    layer_input = "x"
    for i in range(num_layers):
        prefix = layer_prefix(i, num_layers)
        layer_output = "output" if i == num_layers - 1 else f"l{i}_output"
        ops, shapes = _layer_ops(prefix, layer_input, layer_output)
        all_ops.extend(ops)
        all_shapes.update(shapes)

        cache_k_in, cache_v_in, cache_k_out, cache_v_out = cache_io_names(num_layers, i)
        graph_inputs.extend([cache_k_in, cache_v_in])
        graph_outputs.extend([cache_k_out, cache_v_out])
        layer_input = layer_output

    return Graph(
        ops=all_ops, shapes=all_shapes, graph_inputs=graph_inputs, graph_outputs=graph_outputs
    )
