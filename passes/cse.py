"""Common subexpression elimination.

Every op in this IR is a pure function of its inputs, so two ops with the
same kind, the same (post-rewrite) input tensors, and the same attrs compute
the same values -- the second is redundant. CSE keeps the first (the
*canonical* op), drops the duplicate, and rewires everything downstream to
read the canonical outputs instead. Because ops are processed in
topological order and input names are rewritten through the running remap as
we go, chains collapse too: if `a` and `a'` dedup, an op reading `a'` is
rewritten to read `a`, so a later `b = f(a')` and `b' = f(a)` then also
become identical and dedup in turn.

Two conservative rules keep it obviously correct:

- **Graph-output names are never renamed.** An op that produces a declared
  graph output is never removed, so tensors like ``output`` / ``cache_k_out``
  keep their names and callers are unaffected. (Such an op may still *serve*
  as the canonical for later duplicates -- that only ever adds a reader.)
- **Equivalence is purely structural.** Same ``kind``, same rewritten input
  list (order matters), same ``attrs`` (compared by ``repr``), same output
  arity. No numeric or algebraic reasoning -- if two ops aren't structurally
  identical, they are left alone.

Like DCE, CSE is a no-op on DRAKE's current hand-built graph (its ops are all
distinct); it earns its keep once a graph is generated with repeated
sub-computation (e.g. a shared projection materialized twice).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ir import Graph, Op


def eliminate_common_subexpressions(graph: Graph) -> Tuple[Graph, List[str]]:
    """Return (deduplicated_graph, removed_op_names).

    Merges structurally-identical ops, rewiring consumers (and graph outputs)
    of a removed op's outputs to the canonical op's outputs. Op order among
    survivors is preserved; shapes for tensors no longer referenced are
    pruned (graph input/output shapes are always retained).
    """
    graph_outputs_set = set(graph.graph_outputs)
    remap: Dict[str, str] = {}  # duplicate tensor name -> canonical tensor name
    seen: Dict[Tuple, Op] = {}  # structural key -> canonical op
    new_ops: List[Op] = []
    removed: List[str] = []

    for op in graph.ops:
        rewritten_inputs = [remap.get(t, t) for t in op.inputs]
        key = (op.kind, tuple(rewritten_inputs), repr(op.attrs), len(op.outputs))
        produces_graph_output = any(out in graph_outputs_set for out in op.outputs)

        canonical = seen.get(key)
        if canonical is not None and not produces_graph_output:
            for dup_out, canon_out in zip(op.outputs, canonical.outputs):
                remap[dup_out] = canon_out
            removed.append(op.name)
            continue

        kept = Op(op.name, op.kind, rewritten_inputs, list(op.outputs), op.attrs)
        new_ops.append(kept)
        # First op with this key becomes the canonical; graph-output producers
        # may serve as canonical too (that only ever adds a reader downstream).
        seen.setdefault(key, kept)

    new_graph_outputs = [remap.get(t, t) for t in graph.graph_outputs]

    referenced = set(graph.graph_inputs) | set(new_graph_outputs)
    for op in new_ops:
        referenced.update(op.inputs)
        referenced.update(op.outputs)
    new_shapes = {t: s for t, s in graph.shapes.items() if t in referenced}

    deduped = Graph(
        ops=new_ops,
        shapes=new_shapes,
        graph_inputs=list(graph.graph_inputs),
        graph_outputs=new_graph_outputs,
        dtype_bytes=graph.dtype_bytes,
    )
    return deduped, removed
