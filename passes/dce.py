"""Dead-code elimination.

A textbook backward liveness pass: an op is *live* if any of its outputs is
transitively needed to compute a declared graph output; everything else is
dead and can be dropped. Because the IR keeps ops in topological order, one
reverse scan suffices -- by the time we visit op `i`, every op that could
consume its outputs has already been visited, so the live set is complete.

Why a decode compiler wants this: fusion, specialization, and any future
rewrite can leave behind ops whose results nothing reads (a projection that
was later bypassed, a scratch tensor from a partially-applied pattern). DCE
is the cleanup that keeps the graph -- and therefore the analytic traffic
model and the executor's work -- honest. On DRAKE's current hand-built
graph nothing is dead, so DCE is a no-op there (verified in tests); it
earns its keep as soon as a pass upstream introduces slack.

Removing an op never changes results (it produced only unused values), so
DCE is trivially semantics-preserving -- tests assert this against the
numpy executor anyway.
"""

from __future__ import annotations

from typing import List, Tuple

from ir import Graph


def eliminate_dead_code(graph: Graph) -> Tuple[Graph, List[str]]:
    """Return (pruned_graph, removed_op_names).

    Keeps exactly the ops needed to produce ``graph.graph_outputs``; drops
    the rest. Tensor shapes for names no longer referenced by any surviving
    op are pruned too (graph input/output shapes are always retained).
    Op order among survivors is preserved.
    """
    n = len(graph.ops)
    live = set(graph.graph_outputs)
    keep = [False] * n
    for i in range(n - 1, -1, -1):
        op = graph.ops[i]
        if any(out in live for out in op.outputs):
            keep[i] = True
            live.update(op.inputs)

    kept_ops = [op for i, op in enumerate(graph.ops) if keep[i]]
    removed = [op.name for i, op in enumerate(graph.ops) if not keep[i]]

    referenced = set(graph.graph_inputs) | set(graph.graph_outputs)
    for op in kept_ops:
        referenced.update(op.inputs)
        referenced.update(op.outputs)
    new_shapes = {t: s for t, s in graph.shapes.items() if t in referenced}

    pruned = Graph(
        ops=kept_ops,
        shapes=new_shapes,
        graph_inputs=list(graph.graph_inputs),
        graph_outputs=list(graph.graph_outputs),
        dtype_bytes=graph.dtype_bytes,
    )
    return pruned, removed
