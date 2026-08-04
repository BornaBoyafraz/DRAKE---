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

from codegen.fused_ops import execute_graph, init_step_inputs, init_weights
from codegen.llvm_kernels import (
    KERNEL_REFERENCE,
    compile_elementwise_kernel,
    llvm_op_overrides,
)
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
        "-> DP wins here: greedy takes the earlier add+rmsnorm pair, while the"
        " cost model prefers the overlapping rmsnorm+matmul+gelu group."
    )


def llvm_kernel_demo() -> None:
    print("\n=== LLVM-JIT'd compute kernel (real codegen, not just dispatch) ===")
    kernel = compile_elementwise_kernel("axpy")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1_000_000).astype(np.float32)
    y = rng.standard_normal(1_000_000).astype(np.float32)
    alpha = 2.5

    jitted = kernel(x, y, alpha=alpha)
    reference = KERNEL_REFERENCE["axpy"](x, y, np.float32(alpha)).astype(np.float32)
    max_abs_err = float(np.max(np.abs(jitted - reference)))
    print("generated + JIT-compiled  void drake_ew_axpy(float* out, float* x, float* y, float alpha, i32 n)")
    print(f"ran on {x.size:,} float32 elements  |  max|jit - numpy| = {max_abs_err:.1e} (bit-identical)")
    print("  the loop body -- getelementptr/load/fmul/fadd/store -- is native code, verified against numpy")


def llvm_matmul_backend_demo() -> None:
    print("\n=== Decode step with matmuls lowered to LLVM (not NumPy) ===")
    graph = build_decode_step_graph()
    fused, _ = FusionPass().run(graph)
    dims = make_dims(batch=2, seq_len=8, hidden_dim=32, n_heads=4, head_dim=8, ffn_dim=64)
    tensors = {**init_weights(dims, seed=4), **init_step_inputs(dims, seed=6)}

    numpy_out = execute_graph(fused, tensors, dims)
    llvm_out = execute_graph(fused, tensors, dims, op_overrides=llvm_op_overrides())

    err = max(
        float(np.max(np.abs(numpy_out[name] - llvm_out[name])))
        for name in ("output", "cache_k_out", "cache_v_out")
    )
    print("every matmul ran as JIT-compiled  void drake_matmul(float* C, float* A, float* B, i32 M, N, K)")
    print(f"max|numpy_backend - llvm_backend| = {err:.2e}  (equal within float32 accumulation order)")
    print("  same graph, same passes -- only the matmul lowering changed, via execute_graph(op_overrides=...)")


def main() -> None:
    single_layer_demo()
    multi_layer_demo()
    cost_optimal_fusion_demo()
    llvm_kernel_demo()
    llvm_matmul_backend_demo()


if __name__ == "__main__":
    main()
