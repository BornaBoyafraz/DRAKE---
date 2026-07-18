"""Reference math backend: real, correct numpy implementations of every op
kind in ``drake.ir``, plus a generic executor that runs a ``Graph`` --
fused or not -- and produces numerically identical results either way.

This backend exists to prove the compiler passes are semantics-preserving
and to give the runtime something to actually execute end-to-end without
requiring a GPU. It is deliberately *not* where DRAKE's performance case is
made: on CPU/numpy, "fusing" python calls doesn't reduce HBM traffic the way
fusing real GPU kernels does. The traffic-savings numbers DRAKE reports
(``drake.passes.fusion.traffic_saved_bytes``) come from an analytic HBM-bytes
cost model, the same style of roofline argument production compilers
(XLA, TVM, TensorRT) use to justify fusion decisions. Swapping this module
for a Triton or CUTLASS backend -- without touching the IR, fusion pass, or
specializer -- is the natural next step to turn those analytic numbers into
measured GPU wall-clock ones.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ir import Dims, Graph, Op

Tensors = Dict[str, np.ndarray]


def _rmsnorm(t: Tensors, op: Op, dims: Dims) -> None:
    x = t[op.inputs[0]]
    w = t[op.inputs[1]]
    eps = op.attrs.get("eps", 1e-6)
    var = np.mean(x * x, axis=-1, keepdims=True)
    t[op.outputs[0]] = (x / np.sqrt(var + eps)) * w


def _matmul(t: Tensors, op: Op, dims: Dims) -> None:
    a, b = t[op.inputs[0]], t[op.inputs[1]]
    t[op.outputs[0]] = a @ b


def _split_qkv(t: Tensors, op: Op, dims: Dims) -> None:
    qkv = t[op.inputs[0]]
    batch, n_heads, head_dim = dims["batch"], dims["n_heads"], dims["head_dim"]
    q, k, v = np.split(qkv, 3, axis=-1)
    t[op.outputs[0]] = q.reshape(batch, n_heads, head_dim)
    t[op.outputs[1]] = k.reshape(batch, n_heads, head_dim)
    t[op.outputs[2]] = v.reshape(batch, n_heads, head_dim)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rope(x: np.ndarray, position: int, head_dim: int) -> np.ndarray:
    freqs = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angle = position * freqs
    emb = np.concatenate([angle, angle], axis=-1)
    cos = np.cos(emb).astype(x.dtype)
    sin = np.sin(emb).astype(x.dtype)
    return x * cos + _rotate_half(x) * sin


def _rope(t: Tensors, op: Op, dims: Dims) -> None:
    q, k = t[op.inputs[0]], t[op.inputs[1]]
    position = dims[op.attrs["position"]]
    head_dim = dims["head_dim"]
    t[op.outputs[0]] = _apply_rope(q, position, head_dim)
    t[op.outputs[1]] = _apply_rope(k, position, head_dim)


def _kv_cache_update(t: Tensors, op: Op, dims: Dims) -> None:
    k_new, v_new, cache_k, cache_v = (t[n] for n in op.inputs)
    t[op.outputs[0]] = np.concatenate([cache_k, k_new[:, None, :, :]], axis=1)
    t[op.outputs[1]] = np.concatenate([cache_v, v_new[:, None, :, :]], axis=1)


def _attn_qk(t: Tensors, op: Op, dims: Dims) -> None:
    q, cache_k = t[op.inputs[0]], t[op.inputs[1]]
    scale = 1.0 / np.sqrt(dims["head_dim"])
    t[op.outputs[0]] = np.einsum("bhd,bthd->bht", q, cache_k) * scale


def _attn_softmax(t: Tensors, op: Op, dims: Dims) -> None:
    scores = t[op.inputs[0]]
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    t[op.outputs[0]] = e / np.sum(e, axis=-1, keepdims=True)


def _attn_av(t: Tensors, op: Op, dims: Dims) -> None:
    probs, cache_v = t[op.inputs[0]], t[op.inputs[1]]
    t[op.outputs[0]] = np.einsum("bht,bthd->bhd", probs, cache_v)


def _reshape(t: Tensors, op: Op, dims: Dims) -> None:
    x = t[op.inputs[0]]
    t[op.outputs[0]] = x.reshape(dims["batch"], dims["hidden_dim"])


def _add(t: Tensors, op: Op, dims: Dims) -> None:
    t[op.outputs[0]] = t[op.inputs[0]] + t[op.inputs[1]]


def _gelu(t: Tensors, op: Op, dims: Dims) -> None:
    x = t[op.inputs[0]]
    t[op.outputs[0]] = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


OP_FUNCS: Dict[str, Callable[[Tensors, Op, Dims], None]] = {
    "rmsnorm": _rmsnorm,
    "matmul": _matmul,
    "split_qkv": _split_qkv,
    "rope": _rope,
    "kv_cache_update": _kv_cache_update,
    "attn_qk": _attn_qk,
    "attn_softmax": _attn_softmax,
    "attn_av": _attn_av,
    "reshape": _reshape,
    "add": _add,
    "gelu": _gelu,
}


def execute_op(tensors: Tensors, op: Op, dims: Dims) -> None:
    if op.kind == "fused":
        for sub_op in op.attrs["sub_ops"]:
            execute_op(tensors, sub_op, dims)
    else:
        OP_FUNCS[op.kind](tensors, op, dims)


def execute_graph(graph: Graph, tensors: Tensors, dims: Dims) -> Tensors:
    """Run every op in `graph` (fused or not) and return the full tensor dict."""
    tensors = dict(tensors)
    for op in graph.ops:
        execute_op(tensors, op, dims)
    return tensors


def init_weights(dims: Dims, seed: int = 0) -> Tensors:
    rng = np.random.default_rng(seed)
    hidden, qkv, ffn = dims["hidden_dim"], dims["qkv_dim"], dims["ffn_dim"]
    scale = 0.02
    return {
        "w_norm1": np.ones(hidden, dtype=np.float32),
        "w_qkv": (rng.standard_normal((hidden, qkv)) * scale).astype(np.float32),
        "w_o": (rng.standard_normal((hidden, hidden)) * scale).astype(np.float32),
        "w_norm2": np.ones(hidden, dtype=np.float32),
        "w_up": (rng.standard_normal((hidden, ffn)) * scale).astype(np.float32),
        "w_down": (rng.standard_normal((ffn, hidden)) * scale).astype(np.float32),
    }


def init_step_inputs(dims: Dims, seed: int = 1) -> Tensors:
    rng = np.random.default_rng(seed)
    batch, hidden = dims["batch"], dims["hidden_dim"]
    seq_len, n_heads, head_dim = dims["seq_len"], dims["n_heads"], dims["head_dim"]
    x = (rng.standard_normal((batch, hidden)) * 0.1).astype(np.float32)
    cache_shape = (batch, seq_len, n_heads, head_dim)
    cache_k = (rng.standard_normal(cache_shape) * 0.1).astype(np.float32)
    cache_v = (rng.standard_normal(cache_shape) * 0.1).astype(np.float32)
    return {"x": x, "cache_k_in": cache_k, "cache_v_in": cache_v}
