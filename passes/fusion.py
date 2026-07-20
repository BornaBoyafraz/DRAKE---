"""Operator fusion pass.

Two selection strategies over the same candidate patterns:

- ``FusionPass.run`` -- greedy, longest-pattern-first, structural only (no
  shape/dims needed). This is what runs once at engine-construction time,
  before any concrete shape has been observed.
- ``FusionPass.run_cost_optimal`` -- a DP over the op sequence that picks the
  set of non-overlapping pattern matches maximizing *total* analytic
  HBM-traffic bytes saved for a given concrete ``dims``, i.e. the provably
  best grouping for one specific deployment shape, not just "take the
  longest match you can". See ``tests/test_fusion.py`` for a constructed
  case where greedy is measurably suboptimal and the DP finds the true
  optimum -- on DRAKE's actual patterns/graph the two currently agree
  (every overlap here happens to be a strict superset, where longer never
  scores worse), but that's a property of this specific pattern table, not
  of greedy selection in general.

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
from typing import Dict, List, Optional, Tuple

from ir import Graph, Op

Pattern = Tuple[Tuple[str, ...], str]

# Longest patterns first so the greedy scan prefers larger fusion groups.
PATTERNS: List[Pattern] = [
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


def _rebuild_graph(graph: Graph, new_ops: List[Op]) -> Graph:
    return Graph(
        ops=new_ops,
        shapes=dict(graph.shapes),
        graph_inputs=list(graph.graph_inputs),
        graph_outputs=list(graph.graph_outputs),
        dtype_bytes=graph.dtype_bytes,
    )


class FusionPass:
    def run(
        self, graph: Graph, patterns: Optional[List[Pattern]] = None
    ) -> Tuple[Graph, List[FusionRecord]]:
        """Greedy, longest-pattern-first, structural selection (no dims
        needed). Scans left to right; at each position takes the first
        matching pattern in `patterns` order (default: `PATTERNS`, already
        longest-first) that both matches the op-kind sequence and passes the
        connectivity check, then jumps past it."""
        patterns = patterns if patterns is not None else PATTERNS
        ops = graph.ops
        n = len(ops)
        new_ops: List[Op] = []
        records: List[FusionRecord] = []
        i = 0
        fuse_idx = 0
        while i < n:
            matched = None
            for pattern, fused_kind in patterns:
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

        return _rebuild_graph(graph, new_ops), records

    def run_cost_optimal(
        self, graph: Graph, dims: Dict[str, int], patterns: Optional[List[Pattern]] = None
    ) -> Tuple[Graph, List[FusionRecord]]:
        """Dynamic-programming selection: the set of non-overlapping pattern
        matches that maximizes *total* analytic HBM-traffic bytes saved for
        this concrete `dims`, not just "take the longest match available".

        Classic weighted-interval-scheduling-style DP over positions
        `0..n`: `best[i]` is the best achievable total from position `i`
        onward, computed backward so every choice at `i` can look up
        `best[i + match_length]` already solved. O(n * len(patterns)).
        """
        patterns = patterns if patterns is not None else PATTERNS
        ops = graph.ops
        n = len(ops)

        best: List[int] = [0] * (n + 1)
        choice: List[Optional[Tuple[int, str, int]]] = [None] * n
        for i in range(n - 1, -1, -1):
            best_here = best[i + 1]  # leave ops[i] unfused
            best_choice: Optional[Tuple[int, str, int]] = None
            for pattern, fused_kind in patterns:
                length = len(pattern)
                if i + length > n:
                    continue
                window = ops[i : i + length]
                if tuple(op.kind for op in window) != pattern:
                    continue
                if not _is_connected_chain(window):
                    continue
                group_score = _group_traffic_saved(window, graph, dims)
                candidate = group_score + best[i + length]
                if candidate > best_here:
                    best_here = candidate
                    best_choice = (length, fused_kind, group_score)
            best[i] = best_here
            choice[i] = best_choice

        new_ops: List[Op] = []
        records: List[FusionRecord] = []
        i = 0
        fuse_idx = 0
        while i < n:
            picked = choice[i]
            if picked is None:
                new_ops.append(ops[i])
                i += 1
                continue
            length, fused_kind, _ = picked
            window = ops[i : i + length]
            fused = _make_fused_op(window, fused_kind, fuse_idx)
            fuse_idx += 1
            new_ops.append(fused)
            records.append(FusionRecord(fused.name, fused_kind, [op.name for op in window]))
            i += length

        return _rebuild_graph(graph, new_ops), records


def _group_traffic_saved(sub_ops: List[Op], original_graph: Graph, dims: Dict[str, int]) -> int:
    """Analytic HBM-traffic bytes saved by fusing this contiguous group of
    sub-ops together, against `original_graph`'s (pre-fusion) consumer
    counts and tensor byte sizes for `dims`.

    For every tensor produced inside the group and consumed by a later op in
    the same group: if nothing outside the group needs it, fusion elides
    both the write and the read-back (2x). If something outside the group
    still needs it (e.g. it's a graph output, like the updated KV cache),
    fusion still avoids the *internal* read-back, saving 1x.
    """
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


def traffic_saved_bytes(fused_op: Op, original_graph: Graph, dims: Dict[str, int]) -> int:
    """Analytic HBM-traffic bytes saved by fusing `fused_op`'s sub-ops.

    `original_graph` must be the *pre-fusion* graph (e.g. the one passed
    into ``FusionPass.run``), not the fused result -- once fusion has run,
    a fused op's own `.inputs` no longer lists the tensors internal to its
    sub-ops, so `consumer_count` against the fused graph would undercount
    external consumers (notably graph outputs like the KV cache) and wrongly
    treat them as fully elided.
    """
    if fused_op.kind != "fused":
        return 0
    return _group_traffic_saved(fused_op.attrs["sub_ops"], original_graph, dims)
