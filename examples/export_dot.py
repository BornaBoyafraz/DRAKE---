"""Export a decode-step graph as Graphviz DOT.

Run with:  .venv/bin/python examples/export_dot.py --out decode.dot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ir import build_decode_step_graph
from passes.fusion import FusionPass


def build_dot(layers: int = 1, fused: bool = True) -> str:
    graph = build_decode_step_graph(num_layers=layers)
    if fused:
        graph, _ = FusionPass().run(graph)
    return graph.to_dot()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=1, metavar="N")
    parser.add_argument(
        "--fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export the fused graph (default: fused)",
    )
    parser.add_argument("--out", type=Path, metavar="PATH", help="output path (default: stdout)")
    args = parser.parse_args(argv)

    dot = build_dot(layers=args.layers, fused=args.fused)
    if args.out is None:
        print(dot)
    else:
        args.out.write_text(f"{dot}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
