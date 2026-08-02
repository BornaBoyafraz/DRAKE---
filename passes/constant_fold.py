"""Analysis of constant-folding opportunities without graph rewriting."""

from __future__ import annotations

from ir import Graph


def fold_constants(graph: Graph, constants: set[str]) -> tuple[Graph, list[str]]:
    """Analysis-only: return the unchanged graph and foldable op names.

    An op is foldable when every input is already known to be constant. Its
    outputs then become constants for later ops in the graph, allowing the
    analysis to discover transitive opportunities. Neither ``graph`` nor the
    caller's ``constants`` set is mutated.
    """
    known_constants = set(constants)
    foldable = []

    for op in graph.ops:
        if all(input_name in known_constants for input_name in op.inputs):
            foldable.append(op.name)
            known_constants.update(op.outputs)

    return graph, foldable
