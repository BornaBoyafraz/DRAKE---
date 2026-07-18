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

from runtime import DrakeEngine


def main() -> None:
    engine = DrakeEngine(hidden_dim=256, n_heads=8, head_dim=32, ffn_dim=1024)

    print("=== Fusion pass ===")
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


if __name__ == "__main__":
    main()
