# DRAKE — Dynamic Runtime Adaptive Kernel Engine

<p align="left">
  <a href="https://github.com/BornaBoyafraz/drake/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/BornaBoyafraz/drake/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="LLVM via llvmlite" src="https://img.shields.io/badge/codegen-LLVM%20(llvmlite)-4B8BBE.svg">
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
| [`ir.py`](ir.py) | A graph IR for `num_layers` stacked transformer decode-step layers (default 1): RMSNorm, QKV projection, RoPE, KV-cache append, attention, output projection, FFN, threaded along a residual stream. Shapes are symbolic (`("batch", "seq_len", "n_heads", "head_dim")`), resolved against a concrete `dims` dict at analysis/execution time. Each layer gets its own weights and KV cache; `num_layers=1` reproduces the original single-layer graph exactly, tensor name for tensor name. `Graph.to_dot()` emits Graphviz DOT (fused nodes and per-edge shape annotations included) for visualizing any graph, fused or not. |
| [`passes/fusion.py`](passes/fusion.py) | Two fusion-selection strategies over the same candidate patterns, both gated by a **connectivity check** that rejects coincidental kind-matches which aren't a real dependency chain: `FusionPass.run` (greedy, longest-pattern-first, no shape needed — what runs once at engine construction) and `FusionPass.run_cost_optimal` (a DP over the op sequence that provably maximizes total analytic HBM-traffic bytes saved for a concrete `dims`, rather than just taking the longest available match). Includes a **KV-cache-aware fusion** — `kv_cache_update → attn_qk → attn_softmax → attn_av` collapses into one node, so the freshly written K/V for a token is consumed by attention without an intervening HBM round-trip, mirroring the idea behind FlashDecoding / paged-attention kernels. |
| [`passes/specialize.py`](passes/specialize.py) | Buckets `(seq_len, batch)` and selects a kernel variant per bucket, per fused op — vectorized vs. tiled attention (tile size 64/128), single-block vs. batch-blocked matmul. |
| [`passes/verify.py`](passes/verify.py) | The IR verifier every real compiler has: checks single-assignment, def-before-use (topological order), that graph outputs are produced, that model parameters vs. dangling references are distinguished, and that every referenced tensor has a declared shape. `DrakeEngine` runs it on the graph both before and after fusion, so a malformed graph fails loudly with a precise message instead of crashing a later pass. |
| [`passes/dce.py`](passes/dce.py) | Dead-code elimination via a single backward-liveness scan: drops any op whose outputs aren't transitively needed for a graph output, and prunes the now-unreferenced tensor shapes. Runs in the `DrakeEngine` build pipeline right after fusion (and re-verifies) — a no-op on the current hand-built graph, but correct-by-construction for future passes that introduce slack. |
| [`codegen/dispatch_jit.py`](codegen/dispatch_jit.py) | Compiles the bucket classifier to **real LLVM IR** via `llvmlite`, JIT-compiles it, and calls it through `ctypes`. Genuinely generated, verified, compiled, and executed — not a mock. Checked against a pure-Python reference over a dense `(seq_len, batch)` grid in [`tests/test_dispatch_jit.py`](tests/test_dispatch_jit.py). |
| [`codegen/fused_ops.py`](codegen/fused_ops.py) | Correct NumPy reference math for every op, and a generic executor that runs a graph — fused or not, any number of layers — by recursing into a fused op's original sub-ops. This is what makes the correctness claim below checkable rather than asserted. |
| [`runtime.py`](runtime.py) | `DrakeEngine`: wires profiling → LLVM dispatch → specialization (lazy, cached per bucket) → execution into a single `.step()` call per decode token. Takes `num_layers`; for `num_layers > 1`, `step()` takes/returns a list of per-layer KV caches instead of a single array pair. |

Full design writeup — including an explicit line between what's genuinely load-bearing and what's a deliberate stand-in — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## The one correctness property that matters

Fusion is only useful if it doesn't change the answer. `tests/test_runtime.py::test_fusion_is_semantics_preserving` runs the same inputs through the original 16-op graph and the fused 10-op graph and asserts the outputs — including the updated KV cache — are numerically identical. The multi-layer variant, `test_multi_layer_fusion_is_semantics_preserving`, checks the same property across a 3-layer stack with independent per-layer KV caches. This isn't a nice-to-have: it's the property that separates a compiler transform from a bug.

The DP fusion selector has its own correctness bar: `test_cost_optimal_beats_greedy_on_a_constructed_conflict` builds a small adversarial graph where the greedy longest-pattern-first heuristic provably picks a worse total than the DP optimum, and checks the DP selector actually finds it (8,000 bytes saved vs. greedy's 16). On DRAKE's real pattern table the two currently agree — every overlap there is a strict superset, so greedy can't lose — which `test_cost_optimal_matches_greedy_on_the_real_graph` pins down explicitly.

## See it run

```
$ .venv/bin/python examples/decode_demo.py

=== Fusion pass (single layer) ===
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

The same run continues with an 8-layer decode loop (`original ops: 128 -> fused ops: 80`, each layer's KV cache tracked independently) and a greedy-vs-DP fusion comparison.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/pytest -q                      # 35 tests: IR, multi-layer graphs,
                                          # fusion legality, cost-optimal
                                          # (DP) fusion selection, bucket/LLVM
                                          # agreement, end-to-end numeric
                                          # equivalence
.venv/bin/ruff check .                   # lint
.venv/bin/mypy --ignore-missing-imports ir.py profiler.py runtime.py passes/ codegen/

.venv/bin/python examples/decode_demo.py # fusion report + LLVM IR dump +
                                          # decode loop + multi-layer demo +
                                          # greedy-vs-DP fusion comparison
```

No GPU, no external services, no network calls — the whole pipeline runs locally in a plain virtualenv. The same three checks (`pytest`, `ruff`, `mypy`) run in CI on every push — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Honest scope

The compiler-technique pieces are real and independently testable: the IR, the fusion legality check, the analytic traffic-savings model, the shape-bucket specializer, and the LLVM codegen (genuinely generated, verified, JIT-compiled, and executed). The execution backend (`fused_ops.py`) runs on NumPy/CPU so the full pipeline is runnable without a GPU; it validates *correctness* — fusion never changes results — not GPU wall-clock performance, since CPU/NumPy doesn't have the HBM-bandwidth bottleneck that motivates kernel fusion in the first place. Swapping that one module for a Triton or CUTLASS backend — without touching the IR, fusion pass, or specializer — is the natural next step toward measured GPU numbers.

No benchmark claim in this README is dressed up as more than it is. That distinction is deliberate.

## Roadmap

- [ ] Triton backend behind the same `KernelPlan` interface, for measured GPU wall-clock and achieved-bandwidth numbers
- [x] Cost-model-driven fusion selection — `FusionPass.run_cost_optimal`, a DP over candidate groupings maximizing analytic traffic saved for a concrete shape
- [x] Multi-layer graphs — `build_decode_step_graph(num_layers=N)`, independent per-layer weights and KV cache
- [x] IR verifier (`passes/verify.py`) and dead-code elimination (`passes/dce.py`) passes
- [ ] Autotuning over tile size / block size instead of the fixed heuristic table in `specialize.py`
- [ ] Make the runtime's per-bucket specialization actually invoke `run_cost_optimal` (today `DrakeEngine` always fuses once, structurally, at construction — the DP selector is available and tested but not yet wired into the lazy per-bucket path, since concrete `dims` aren't known until the first `.step()` call for a shape)

## Author

**Seyedborna Boyafraz** (Borna Afraz)

## License

MIT — see [LICENSE](LICENSE).
