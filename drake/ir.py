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


def build_decode_step_graph() -> Graph:
    """One decode-step, single transformer layer, as a DRAKE Graph.

    Shapes use symbolic dims: batch, seq_len (existing KV-cache length),
    next_seq_len (= seq_len + 1, cache length after this step's append),
    hidden_dim, n_heads, head_dim, qkv_dim, ffn_dim.
    """
    shapes: Dict[str, Shape] = {
        "x": ("batch", "hidden_dim"),
        "w_norm1": ("hidden_dim",),
        "x_norm": ("batch", "hidden_dim"),
        "w_qkv": ("hidden_dim", "qkv_dim"),
        "qkv": ("batch", "qkv_dim"),
        "q": ("batch", "n_heads", "head_dim"),
        "k_new": ("batch", "n_heads", "head_dim"),
        "v_new": ("batch", "n_heads", "head_dim"),
        "q_rot": ("batch", "n_heads", "head_dim"),
        "k_rot": ("batch", "n_heads", "head_dim"),
        "cache_k_in": ("batch", "seq_len", "n_heads", "head_dim"),
        "cache_v_in": ("batch", "seq_len", "n_heads", "head_dim"),
        "cache_k_out": ("batch", "next_seq_len", "n_heads", "head_dim"),
        "cache_v_out": ("batch", "next_seq_len", "n_heads", "head_dim"),
        "scores": ("batch", "n_heads", "next_seq_len"),
        "probs": ("batch", "n_heads", "next_seq_len"),
        "attn_out": ("batch", "n_heads", "head_dim"),
        "attn_out_flat": ("batch", "hidden_dim"),
        "w_o": ("hidden_dim", "hidden_dim"),
        "attn_proj": ("batch", "hidden_dim"),
        "resid1": ("batch", "hidden_dim"),
        "w_norm2": ("hidden_dim",),
        "resid1_norm": ("batch", "hidden_dim"),
        "w_up": ("hidden_dim", "ffn_dim"),
        "ff_hidden": ("batch", "ffn_dim"),
        "ff_act": ("batch", "ffn_dim"),
        "w_down": ("ffn_dim", "hidden_dim"),
        "ff_out": ("batch", "hidden_dim"),
        "output": ("batch", "hidden_dim"),
    }

    ops = [
        Op("norm1", "rmsnorm", ["x", "w_norm1"], ["x_norm"], {"eps": 1e-6}),
        Op("qkv_proj", "matmul", ["x_norm", "w_qkv"], ["qkv"], {}),
        Op("split", "split_qkv", ["qkv"], ["q", "k_new", "v_new"], {}),
        Op("rope", "rope", ["q", "k_new"], ["q_rot", "k_rot"], {"position": "seq_len"}),
        Op(
            "kv_update",
            "kv_cache_update",
            ["k_rot", "v_new", "cache_k_in", "cache_v_in"],
            ["cache_k_out", "cache_v_out"],
            {},
        ),
        Op("qk", "attn_qk", ["q_rot", "cache_k_out"], ["scores"], {}),
        Op("softmax", "attn_softmax", ["scores"], ["probs"], {}),
        Op("av", "attn_av", ["probs", "cache_v_out"], ["attn_out"], {}),
        Op("flatten", "reshape", ["attn_out"], ["attn_out_flat"], {}),
        Op("o_proj", "matmul", ["attn_out_flat", "w_o"], ["attn_proj"], {}),
        Op("resid1", "add", ["x", "attn_proj"], ["resid1"], {}),
        Op("norm2", "rmsnorm", ["resid1", "w_norm2"], ["resid1_norm"], {"eps": 1e-6}),
        Op("up_proj", "matmul", ["resid1_norm", "w_up"], ["ff_hidden"], {}),
        Op("act", "gelu", ["ff_hidden"], ["ff_act"], {}),
        Op("down_proj", "matmul", ["ff_act", "w_down"], ["ff_out"], {}),
        Op("resid2", "add", ["resid1", "ff_out"], ["output"], {}),
    ]

    graph_inputs = ["x", "cache_k_in", "cache_v_in"]
    graph_outputs = ["output", "cache_k_out", "cache_v_out"]
    return Graph(ops=ops, shapes=shapes, graph_inputs=graph_inputs, graph_outputs=graph_outputs)
