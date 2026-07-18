# DRAKE — Dynamic Runtime Adaptive Kernel Engine

<p align="left">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="LLVM via llvmlite" src="https://img.shields.io/badge/codegen-LLVM%20(llvmlite)-4B8BBE.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-25%20passing-2ea043.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-research--grade%20prototype-orange.svg">
</p>

**Shape-specialized operator fusion for LLM decode inference, with the kernel-dispatch decision compiled to real LLVM IR and JIT-executed.**

Ahead-of-time inference compilers — TVM, XLA, TensorRT, `torch.compile` — commit to one fusion plan and one kernel schedule per shape, decided once at compile time. That assumption holds for vision models with fixed input sizes. It breaks for autoregressive LLM decoding, where `seq_len` (and the KV cache riding along with it) grows by one token on *every single call*, and a production server spends its entire life sliding along that curve rather than sitting at one point on it. The schedule that's optimal at token 16 — everything register-resident, no tiling needed — is not the schedule you want at token 8,000.

DRAKE's premise: don't pick one schedule. Profile the shapes a workload actually visits, bucket them, and lazily fuse + specialize a kernel plan the first time each bucket is hit — then compile the bucket-classification logic itself down to native code via LLVM, so that decision doesn't cost an interpreted Python branch on every token of every request.

This repository is a from-scratch, independently-tested implementation of that idea: a graph IR, a legality-checked fusion pass with an analytic memory-traffic cost model, a shape-bucket specializer, and an LLVM-JIT'd dispatcher — the same category of engineering problem as a deep learning compiler/runtime role sits squarely in.

---

## Why this exists

Built as a demonstration of the specific skill intersection a **Deep Learning Compiler Engineer** role asks for: compiler passes over a real workload's IR, LLVM as an actual code-generation target (not just a buzzword on a resume), and profiling-driven performance decisions — applied to the part of LLM inference (autoregressive decode) that ahead-of-time compilers systematically under-serve.

## What it actually does

| Module | Responsibility |
|---|---|
| [`ir.py`](ir.py) | A graph IR scoped to one transformer decode-step layer: RMSNorm, QKV projection, RoPE, KV-cache append, attention, output projection, FFN. Shapes are symbolic (`("batch", "seq_len", "n_heads", "head_dim")`), resolved against a concrete `dims` dict at analysis/execution time. |
| [`passes/fusion.py`](passes/fusion.py) | Pattern-based operator fusion, gated by a **connectivity check** that rejects coincidental kind-matches which aren't a real dependency chain. Includes a **KV-cache-aware fusion** — `kv_cache_update → attn_qk → attn_softmax → attn_av` collapses into one node, so the freshly written K/V for a token is consumed by attention without an intervening HBM round-trip, mirroring the idea behind FlashDecoding / paged-attention kernels. Reports savings via an analytic HBM-traffic-bytes model. |
| [`passes/specialize.py`](passes/specialize.py) | Buckets `(seq_len, batch)` and selects a kernel variant per bucket, per fused op — vectorized vs. tiled attention (tile size 64/128), single-block vs. batch-blocked matmul. |
| [`codegen/dispatch_jit.py`](codegen/dispatch_jit.py) | Compiles the bucket classifier to **real LLVM IR** via `llvmlite`, JIT-compiles it, and calls it through `ctypes`. Genuinely generated, verified, compiled, and executed — not a mock. Checked against a pure-Python reference over a dense `(seq_len, batch)` grid in [`tests/test_dispatch_jit.py`](tests/test_dispatch_jit.py). |
| [`codegen/fused_ops.py`](codegen/fused_ops.py) | Correct NumPy reference math for every op, and a generic executor that runs a graph — fused or not — by recursing into a fused op's original sub-ops. This is what makes the correctness claim below checkable rather than asserted. |
| [`runtime.py`](runtime.py) | `DrakeEngine`: wires profiling → LLVM dispatch → specialization (lazy, cached per bucket) → execution into a single `.step()` call per decode token. |

Full design writeup — including an explicit line between what's genuinely load-bearing and what's a deliberate stand-in — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## The one correctness property that matters

Fusion is only useful if it doesn't change the answer. `tests/test_runtime.py::test_fusion_is_semantics_preserving` runs the same inputs through the original 16-op graph and the fused 10-op graph and asserts the outputs — including the updated KV cache — are numerically identical. This isn't a nice-to-have: it's the property that separates a compiler transform from a bug.

## See it run

```
$ .venv/bin/python examples/decode_demo.py

=== Fusion pass ===
original ops: 16  ->  fused ops: 10
  fused_norm_matmul_0          [fused_norm_matmul]        <- ['norm1', 'qkv_proj']
  fused_attention_kvupdate_1   [fused_attention_kvupdate] <- ['kv_update', 'qk', 'softmax', 'av']
  fused_norm_matmul_gelu_2     [fused_norm_matmul_gelu]   <- ['norm2', 'up_proj', 'act']

=== LLVM dispatch IR (drake_dispatch) ===
define i32 @"drake_dispatch"(i32 %"seq_len", i32 %"batch")
{
entry:
  %".4" = icmp sge i32 %"seq_len", 128
  ...
  ret i32 %".14"
}
```

| seq_len | bucket | attention variant | analytic traffic saved |
|---:|---|---|---:|
| 1 | `seq[0,128)_batch[0,8)` | vector | 56.5 KiB |
| 130 | `seq[128,1024)_batch[0,8)` | tiled (tile=64) | 1,153.0 KiB |
| 1,025 | `seq[1024,inf)_batch[0,8)` | tiled (tile=128) | 8,760.5 KiB |

*(Analytic HBM-traffic-bytes model — see "Honest scope" below for exactly what that does and doesn't claim.)*

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/pytest -q                      # 25 tests: IR, fusion legality,
                                          # bucket/LLVM agreement, end-to-end
                                          # numeric equivalence

.venv/bin/python examples/decode_demo.py # fusion report + LLVM IR dump
                                          # + a growing-cache decode loop
```

No GPU, no external services, no network calls — the whole pipeline runs locally in a plain virtualenv.

## Honest scope

The compiler-technique pieces are real and independently testable: the IR, the fusion legality check, the analytic traffic-savings model, the shape-bucket specializer, and the LLVM codegen (genuinely generated, verified, JIT-compiled, and executed). The execution backend (`fused_ops.py`) runs on NumPy/CPU so the full pipeline is runnable without a GPU; it validates *correctness* — fusion never changes results — not GPU wall-clock performance, since CPU/NumPy doesn't have the HBM-bandwidth bottleneck that motivates kernel fusion in the first place. Swapping that one module for a Triton or CUTLASS backend — without touching the IR, fusion pass, or specializer — is the natural next step toward measured GPU numbers.

No benchmark claim in this README is dressed up as more than it is. That distinction is deliberate.

## Roadmap

- [ ] Triton backend behind the same `KernelPlan` interface, for measured GPU wall-clock and achieved-bandwidth numbers
- [ ] Cost-model-driven fusion selection (replace the greedy longest-pattern-first heuristic with one that compares candidate groupings)
- [ ] Multi-layer graphs (currently one decode-step layer; stacking layers is a straightforward IR extension)
- [ ] Autotuning over tile size / block size instead of the fixed heuristic table in `specialize.py`

## Author

**Seyedborna Boyafraz** (Borna Afraz)

## License

MIT — see [LICENSE](LICENSE).
