"""Operator fusion pass.

Greedily matches contiguous op-kind patterns (longest pattern first) and
replaces each match with a single ``FusedOp`` node, provided the match is a
*real* dataflow chain (not a coincidental kind match).

The KV-cache-aware pattern -- ``kv_cache_update -> attn_qk -> attn_softmax ->
attn_av`` -- is the one novel fusion here: it fuses the KV-cache append with
the attention read that immediately follows it, so the freshly written K/V
for the new token never round-trips through memory before being read back.
This mirrors the real optimization FlashDecoding-style and paged-attention
kernels rely on.

Fusion here does not change *what* is computed -- a ``FusedOp`` simply
remembers the original sub-ops and executes them in sequence (see
``codegen/fused_ops.py``). What changes is (a) node count, used by the
specializer to decide kernel variants per fused unit, and (b) the analytic
HBM traffic estimate below, which is the real payoff fusion is meant to
capture: tensors that only exist to hand data from one op to the next in the
same fused group never have to be written to and read back from global
memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ir import Graph, Op

# Longest patterns first so the greedy scan prefers larger fusion groups.
PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    (("rmsnorm", "matmul", "gelu"), "fused_norm_matmul_gelu"),
    (("kv_cache_update", "attn_qk", "attn_softmax", "attn_av"), "fused_attention_kvupdate"),
    (("attn_qk", "attn_softmax", "attn_av"), "fused_attention"),
    (("rmsnorm", "matmul"), "fused_norm_matmul"),
    (("matmul", "gelu"), "fused_matmul_gelu"),
]


@dataclass
class FusionRecord:
    fused_op_name: str
    fused_kind: str
    sub_op_names: List[str]


def _is_connected_chain(window: List[Op]) -> bool:
    """Reject coincidental kind-matches that aren't an actual dependency chain."""
    if len(window) == 1:
        return True
    reached = {0}
    for k in range(len(window)):
        if k not in reached:
            continue
        produced = set(window[k].outputs)
        for j in range(k + 1, len(window)):
            if produced & set(window[j].inputs):
                reached.add(j)
    return len(window) - 1 in reached


def _make_fused_op(window: List[Op], fused_kind: str, index: int) -> Op:
    name = f"{fused_kind}_{index}"
    all_inputs = [t for op in window for t in op.inputs]
    all_outputs = [t for op in window for t in op.outputs]
    internal = set(all_outputs) & set(all_inputs)
    inputs = []
    seen = set()
    for t in all_inputs:
        if t in internal:
            continue
        if t not in seen:
            seen.add(t)
            inputs.append(t)
    outputs = []
    seen = set()
    for t in all_outputs:
        if t not in seen:
            seen.add(t)
            outputs.append(t)
    return Op(
        name=name,
        kind="fused",
        inputs=inputs,
        outputs=outputs,
        attrs={"fused_kind": fused_kind, "sub_ops": list(window)},
    )


class FusionPass:
    def run(self, graph: Graph) -> Tuple[Graph, List[FusionRecord]]:
        ops = graph.ops
        n = len(ops)
        new_ops: List[Op] = []
        records: List[FusionRecord] = []
        i = 0
        fuse_idx = 0
        while i < n:
            matched = None
            for pattern, fused_kind in PATTERNS:
                length = len(pattern)
                if i + length > n:
                    continue
                window = ops[i : i + length]
                if tuple(op.kind for op in window) != pattern:
                    continue
                if not _is_connected_chain(window):
                    continue
                matched = (window, fused_kind)
                break
            if matched:
                window, fused_kind = matched
                fused = _make_fused_op(window, fused_kind, fuse_idx)
                fuse_idx += 1
                new_ops.append(fused)
                records.append(
                    FusionRecord(fused.name, fused_kind, [op.name for op in window])
                )
                i += len(window)
            else:
                new_ops.append(ops[i])
                i += 1

        new_graph = Graph(
            ops=new_ops,
            shapes=dict(graph.shapes),
            graph_inputs=list(graph.graph_inputs),
            graph_outputs=list(graph.graph_outputs),
            dtype_bytes=graph.dtype_bytes,
        )
        return new_graph, records


def traffic_saved_bytes(fused_op: Op, original_graph: Graph, dims: Dict[str, int]) -> int:
    """Analytic HBM-traffic bytes saved by fusing `fused_op`'s sub-ops.

    For every tensor produced inside the fused group and consumed by a later
    op in the same group: if nothing outside the group needs it, fusion
    elides both the write and the read-back (2x). If something outside the
    group still needs it (e.g. it's a graph output, like the updated KV
    cache), fusion still avoids the *internal* read-back, saving 1x.

    `original_graph` must be the *pre-fusion* graph (e.g. the one passed
    into ``FusionPass.run``), not the fused result -- once fusion has run,
    a fused op's own `.inputs` no longer lists the tensors internal to its
    sub-ops, so `consumer_count` against the fused graph would undercount
    external consumers (notably graph outputs like the KV cache) and wrongly
    treat them as fully elided.
    """
    if fused_op.kind != "fused":
        return 0
    sub_ops: List[Op] = fused_op.attrs["sub_ops"]
    total = 0
    for k, op in enumerate(sub_ops):
        for t in op.outputs:
            consumed_later_internally = any(t in later.inputs for later in sub_ops[k + 1 :])
            if not consumed_later_internally:
                continue
            total_consumers = original_graph.consumer_count(t)
            internal_consumers = sum(1 for later in sub_ops[k + 1 :] if t in later.inputs)
            external_consumers = total_consumers - internal_consumers
            multiplier = 2 if external_consumers <= 0 else 1
            total += multiplier * original_graph.bytes_of(t, dims)
    return total
