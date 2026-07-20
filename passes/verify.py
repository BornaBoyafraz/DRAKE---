"""Graph verifier.

Every real compiler has a verifier: a cheap, always-runnable pass that
checks the IR's structural invariants and fails loudly with a precise
message the moment something upstream produced a malformed graph. It's the
difference between "a later pass crashed with an opaque KeyError" and "op
`qk` reads tensor `q_rot` which nothing produces". A verifier is what makes
every *other* pass safe to write, because each one can assume (and, in
tests, re-assert) a well-formed input.

A tensor is **external** (available before any op runs) if it has a declared
shape but is never produced by an op -- this covers both graph inputs and
model parameters (the weight tensors ``w_qkv``, ``w_norm1``, ... that ops
read but no op produces). A reference to a tensor that is neither produced
nor declared with a shape is a genuine dangling reference, caught by
invariant 5.

Invariants checked here, all of which DRAKE's own passes must preserve:

1. **Single assignment.** No tensor is produced by more than one op. (SSA-ish:
   a tensor name uniquely identifies the value and the op that made it.)
2. **No redefinition of graph inputs.** An op may not output a tensor whose
   name is a declared graph input.
3. **Def-before-use / topological order.** Every op input is either external
   (see above) or produced by an *earlier* op in the op list -- an input
   produced only by a *later* op is a use-before-def error.
4. **Outputs are produced.** Every declared graph output is a graph input or
   produced by some op.
5. **Shapes are declared.** Every tensor referenced by any op (input or
   output) has an entry in ``graph.shapes``.

Fused graphs are verified the same way: a ``FusedOp``'s top-level
inputs/outputs are ordinary tensors, and the checks above apply unchanged
(a fused op's internal tensors simply no longer appear at the top level).
"""

from __future__ import annotations

from typing import List

from ir import Graph


class GraphVerificationError(ValueError):
    """Raised by ``verify_graph`` when a graph violates an IR invariant.

    Carries the full list of problems (not just the first) so a caller sees
    everything wrong at once."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        joined = "\n  - ".join(errors)
        super().__init__(f"graph failed verification ({len(errors)} error(s)):\n  - {joined}")


def collect_graph_errors(graph: Graph) -> List[str]:
    """Return a list of invariant violations (empty if the graph is valid).

    Non-raising counterpart to ``verify_graph`` -- useful in tests and when
    you want to report every problem rather than stop at the first."""
    errors: List[str] = []
    graph_inputs = set(graph.graph_inputs)

    # Invariant 1 + 2: single assignment, no redefinition of graph inputs.
    producer_of: dict = {}
    for op in graph.ops:
        for out in op.outputs:
            if out in graph_inputs:
                errors.append(
                    f"op {op.name!r} outputs {out!r}, which is a declared graph input "
                    f"(graph inputs may not be redefined)"
                )
            if out in producer_of:
                errors.append(
                    f"tensor {out!r} is produced by more than one op "
                    f"({producer_of[out]!r} and {op.name!r}); tensors must be single-assignment"
                )
            else:
                producer_of[out] = op.name

    # External tensors: declared with a shape but produced by no op. These are
    # graph inputs and model parameters (weights), available from the start.
    external = {t for t in graph.shapes if t not in producer_of}

    # Invariant 3: def-before-use, in op-list order.
    produced_so_far = set(graph_inputs) | external
    for op in graph.ops:
        for inp in op.inputs:
            if inp not in produced_so_far and inp in producer_of:
                errors.append(
                    f"op {op.name!r} reads {inp!r} before it is produced "
                    f"(produced later by {producer_of[inp]!r}); ops must be topologically ordered"
                )
            # A reference with neither a producer nor a shape is a dangling
            # reference; invariant 5 reports it as a missing shape.
        produced_so_far.update(op.outputs)

    # Invariant 4: every graph output is produced (or is a graph input).
    for out in graph.graph_outputs:
        if out not in producer_of and out not in graph_inputs:
            errors.append(
                f"graph output {out!r} is neither produced by any op nor a graph input"
            )

    # Invariant 5: every referenced tensor has a declared shape.
    for op in graph.ops:
        for tensor in list(op.inputs) + list(op.outputs):
            if tensor not in graph.shapes:
                errors.append(f"tensor {tensor!r} (used by op {op.name!r}) has no entry in graph.shapes")

    return errors


def verify_graph(graph: Graph) -> None:
    """Raise ``GraphVerificationError`` if `graph` violates any IR invariant.

    Cheap enough to call at the boundary of every pass (after construction,
    after fusion, after DCE) as a safety net."""
    errors = collect_graph_errors(graph)
    if errors:
        raise GraphVerificationError(errors)
