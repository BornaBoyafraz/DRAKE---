"""Compile the shape-bucket dispatch decision to real LLVM IR and JIT it.

Every decode step has to answer "which kernel plan applies to this
(seq_len, batch)?" before it can run a single matmul. In an interpreted
Python if-chain that decision costs real, measurable overhead on a
per-token hot path. DRAKE instead compiles the bucket classifier once,
ahead of the decode loop, into a native function via ``llvmlite`` and calls
it through ``ctypes`` -- so the dispatch logic that gates every fused
kernel is itself compiled code, not interpreted code.

The generated logic is branch-free: for each boundary ``b`` it computes
``zext(seq_len >= b)`` and sums those zero/one values to get the bucket
index along that axis, exactly mirroring ``passes.specialize.classify``.
Semantic equivalence between the two is checked exhaustively in
``tests/test_dispatch_jit.py``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

from llvmlite import binding as llvm
from llvmlite import ir

_LLVM_INITIALIZED = False


def _ensure_llvm_initialized() -> None:
    global _LLVM_INITIALIZED
    if _LLVM_INITIALIZED:
        return
    # llvmlite >= 0.44 initializes LLVM core automatically; only the native
    # target/asm-printer registration below is still required for JIT codegen.
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _LLVM_INITIALIZED = True


def build_dispatch_module(
    seq_boundaries: Sequence[int], batch_boundaries: Sequence[int]
) -> ir.Module:
    """Build the LLVM module defining ``i32 drake_dispatch(i32 seq_len, i32 batch)``."""
    module = ir.Module(name="drake_dispatch")
    i32 = ir.IntType(32)
    fn_ty = ir.FunctionType(i32, (i32, i32))
    fn = ir.Function(module, fn_ty, name="drake_dispatch")
    seq_len, batch = fn.args
    seq_len.name, batch.name = "seq_len", "batch"

    block = fn.append_basic_block("entry")
    builder = ir.IRBuilder(block)

    def bucket_index(value, boundaries):
        idx = ir.Constant(i32, 0)
        for b in boundaries:
            ge = builder.icmp_signed(">=", value, ir.Constant(i32, b))
            idx = builder.add(idx, builder.zext(ge, i32))
        return idx

    seq_idx = bucket_index(seq_len, seq_boundaries)
    batch_idx = bucket_index(batch, batch_boundaries)
    num_batch_buckets = ir.Constant(i32, len(batch_boundaries) + 1)
    bucket_id = builder.add(builder.mul(seq_idx, num_batch_buckets), batch_idx)
    builder.ret(bucket_id)
    return module


@dataclass
class DispatchEngine:
    """Owns the JIT-compiled module; keep it alive as long as `classify` is used."""

    ir_text: str
    _engine: llvm.ExecutionEngine
    _classify_fn: ctypes.CFUNCTYPE

    def classify(self, seq_len: int, batch: int) -> int:
        return self._classify_fn(seq_len, batch)


def compile_dispatch_engine(
    seq_boundaries: Sequence[int], batch_boundaries: Sequence[int]
) -> DispatchEngine:
    _ensure_llvm_initialized()
    module = build_dispatch_module(seq_boundaries, batch_boundaries)
    ir_text = str(module)

    llvm_module = llvm.parse_assembly(ir_text)
    llvm_module.verify()

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_module = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_module, target_machine)
    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()

    func_ptr = engine.get_function_address("drake_dispatch")
    c_func = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(func_ptr)

    return DispatchEngine(ir_text=ir_text, _engine=engine, _classify_fn=c_func)
