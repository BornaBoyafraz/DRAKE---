"""End-to-end DRAKE demo: run a growing-KV-cache decode loop and print what
the compiler did at every stage -- fusion, bucket classification, chosen
kernel variants, and analytic HBM-traffic savings.

Run with:  .venv/bin/python examples/decode_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ir import build_decode_step_graph, make_dims
from passes.fusion import FusionPass, traffic_saved_bytes
from runtime import DrakeEngine


def single_layer_demo() -> None:
    engine = DrakeEngine(hidden_dim=256, n_heads=8, head_dim=32, ffn_dim=1024)

    print("=== Fusion pass (single layer) ===")
    summary = engine.fusion_summary()
    print(f"original ops: {summary['original_op_count']}  ->  fused ops: {summary['fused_op_count']}")
    for f in summary["fusions"]:
        print(f"  {f['fused_op']:<28} [{f['kind']}]  <-  {f['sub_ops']}")

    print("\n=== LLVM dispatch IR (drake_dispatch) ===")
    print(engine.dispatch_ir())

    print("=== Decode loop ===")
    batch = 4
    cache_k = np.zeros((batch, 0, 8, 32), dtype=np.float32)
    cache_v = np.zeros((batch, 0, 8, 32), dtype=np.float32)
    rng = np.random.default_rng(0)

    # Jump seq_len around to exercise every bucket, not just +1 each step.
    checkpoints = [1, 16, 130, 400, 1025, 2048]
    for target_len in checkpoints:
        while cache_k.shape[1] < target_len:
            x = (rng.standard_normal((batch, 256)) * 0.1).astype(np.float32)
            result = engine.step(x, cache_k, cache_v)
            cache_k, cache_v = result.cache_k_out, result.cache_v_out

        kv_variant = next(
            v for name, v in result.plan.variants.items() if "attention_kvupdate" in name
        )
        print(
            f"seq_len={cache_k.shape[1]:>5}  bucket={result.bucket.name:<24}"
            f"  attn_variant={kv_variant.name}{kv_variant.params or ''}"
            f"  traffic_saved={result.traffic_saved_bytes/1024:.1f} KiB"
        )

    print(f"\ntotal decode steps profiled: {engine.profiler.total_calls()}")
    print(f"kernel plans compiled (one per bucket actually visited): {len(engine.plan_cache)}")


def multi_layer_demo(num_layers: int = 8) -> None:
    print(f"\n=== Multi-layer decode ({num_layers} stacked layers) ===")
    engine = DrakeEngine(hidden_dim=128, n_heads=4, head_dim=32, ffn_dim=512, num_layers=num_layers)
    summary = engine.fusion_summary()
    print(
        f"original ops: {summary['original_op_count']}  ->  fused ops: {summary['fused_op_count']}"
        f"  ({len(summary['fusions'])} fusion groups, {len(summary['fusions']) // num_layers} per layer)"
    )

    batch = 2
    cache_k = [np.zeros((batch, 0, 4, 32), dtype=np.float32) for _ in range(num_layers)]
    cache_v = [np.zeros((batch, 0, 4, 32), dtype=np.float32) for _ in range(num_layers)]
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = (rng.standard_normal((batch, 128)) * 0.1).astype(np.float32)
        result = engine.step(x, cache_k, cache_v)
        cache_k, cache_v = result.cache_k_out, result.cache_v_out

    print(f"output shape: {result.output.shape}  |  per-layer KV cache lengths: "
          f"{[c.shape[1] for c in cache_k]}  (each layer's cache tracked independently)")


def cost_optimal_fusion_demo() -> None:
    print("\n=== Greedy vs. cost-optimal (DP) fusion selection ===")
    graph = build_decode_step_graph()
    dims = make_dims(batch=4, seq_len=200, hidden_dim=256, n_heads=8, head_dim=32, ffn_dim=1024)

    greedy_graph, greedy_records = FusionPass().run(graph)
    dp_graph, dp_records = FusionPass().run_cost_optimal(graph, dims)
    greedy_total = sum(traffic_saved_bytes(op, graph, dims) for op in greedy_graph.ops)
    dp_total = sum(traffic_saved_bytes(op, graph, dims) for op in dp_graph.ops)

    print(f"greedy (longest-pattern-first): {len(greedy_records)} groups, {greedy_total / 1024:.1f} KiB saved")
    print(f"DP (cost-optimal):              {len(dp_records)} groups, {dp_total / 1024:.1f} KiB saved")
    print(
        "-> identical here: every pattern overlap in DRAKE's table is a strict"
        " superset, so greedy can't lose. tests/test_fusion.py::"
        "test_cost_optimal_beats_greedy_on_a_constructed_conflict proves the DP"
        " formulation actually matters once patterns genuinely compete."
    )


def main() -> None:
    single_layer_demo()
    multi_layer_demo()
    cost_optimal_fusion_demo()


if __name__ == "__main__":
    main()
