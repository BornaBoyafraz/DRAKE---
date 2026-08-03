"""LLVM-JIT'd elementwise compute kernels.

`dispatch_jit.py` compiles the *control-flow* decision (which bucket a shape
falls in) to LLVM. This module compiles actual *arithmetic*: it emits an
LLVM IR loop over a float32 array -- pointer arithmetic (``getelementptr``),
loads, an fp op, stores, and a phi-node induction variable -- JIT-compiles it
to native code, and calls it through ``ctypes`` on real NumPy buffers.

Why it matters. DRAKE's reference executor runs on NumPy, which is honest
about *what* is computed but says nothing about a lowered kernel. These
kernels close that gap for the elementwise ops in the decode graph (the
residual ``add``, and scaled variants): they are genuinely generated,
verified, compiled, and executed as native code, and their results are
checked bit-for-bit against NumPy in the tests. They are the seed of a real
codegen backend behind the ``KernelPlan`` interface -- the same shape of
lowering a Triton/CUTLASS backend would do per ``KernelVariant``, done here
for CPU via LLVM so it runs anywhere without a GPU.

The generated IR for ``add`` is, in essence::

    define void @drake_ew_add(float* %out, float* %x, float* %y, i32 %n) {
    entry:
      br label %loop.cond
    loop.cond:
      %i = phi i32 [0, %entry], [%i.next, %loop.body]
      %c = icmp slt i32 %i, %n
      br i1 %c, label %loop.body, label %loop.end
    loop.body:
      %vx = load float, float* getelementptr(float, float* %x, i32 %i)
      %vy = load float, float* getelementptr(float, float* %y, i32 %i)
      %r  = fadd float %vx, %vy
      store float %r, float* getelementptr(float, float* %out, i32 %i)
      %i.next = add i32 %i, 1
      br label %loop.cond
    loop.end:
      ret void
    }
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from llvmlite import binding as llvm
from llvmlite import ir

# op name -> (needs_alpha, combine(builder, vx, vy, alpha) -> value)
Combine = Callable[[ir.IRBuilder, ir.Value, ir.Value, ir.Value | None], ir.Value]

_LLVM_INITIALIZED = False


def _ensure_llvm_initialized() -> None:
    global _LLVM_INITIALIZED
    if _LLVM_INITIALIZED:
        return
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _LLVM_INITIALIZED = True


def _combine_add(b: ir.IRBuilder, vx: ir.Value, vy: ir.Value, alpha: ir.Value | None) -> ir.Value:
    return b.fadd(vx, vy)


def _combine_mul(b: ir.IRBuilder, vx: ir.Value, vy: ir.Value, alpha: ir.Value | None) -> ir.Value:
    return b.fmul(vx, vy)


def _combine_axpy(b: ir.IRBuilder, vx: ir.Value, vy: ir.Value, alpha: ir.Value | None) -> ir.Value:
    assert alpha is not None
    return b.fadd(b.fmul(alpha, vx), vy)  # alpha * x + y


# Each kernel: (needs_alpha, combine). NumPy reference lives in KERNEL_REFERENCE.
_KERNELS: dict[str, tuple[bool, Combine]] = {
    "add": (False, _combine_add),
    "mul": (False, _combine_mul),
    "axpy": (True, _combine_axpy),
}

KERNEL_REFERENCE: dict[str, Callable[..., np.ndarray]] = {
    "add": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    "axpy": lambda x, y, alpha: alpha * x + y,
}


def build_kernel_module(op: str) -> ir.Module:
    """Emit an LLVM module with ``void drake_ew_<op>(float* out, float* x,
    float* y, [float alpha,] i32 n)`` computing the elementwise op over n
    contiguous float32 elements."""
    if op not in _KERNELS:
        raise ValueError(f"unknown kernel op {op!r}; known: {sorted(_KERNELS)}")
    needs_alpha, combine = _KERNELS[op]

    module = ir.Module(name=f"drake_ew_{op}")
    f32 = ir.FloatType()
    f32p = ir.PointerType(f32)
    i32 = ir.IntType(32)

    arg_types = [f32p, f32p, f32p] + ([f32] if needs_alpha else []) + [i32]
    fn_ty = ir.FunctionType(ir.VoidType(), arg_types)
    fn = ir.Function(module, fn_ty, name=f"drake_ew_{op}")
    if needs_alpha:
        out, x, y, alpha, n = fn.args
        alpha.name = "alpha"
    else:
        out, x, y, n = fn.args
        alpha = None
    out.name, x.name, y.name, n.name = "out", "x", "y", "n"

    entry = fn.append_basic_block("entry")
    cond = fn.append_basic_block("loop.cond")
    body = fn.append_basic_block("loop.body")
    end = fn.append_basic_block("loop.end")

    b = ir.IRBuilder(entry)
    b.branch(cond)

    b.position_at_end(cond)
    i = b.phi(i32, name="i")
    i.add_incoming(ir.Constant(i32, 0), entry)
    b.cbranch(b.icmp_signed("<", i, n), body, end)

    b.position_at_end(body)
    vx = b.load(b.gep(x, [i], inbounds=True))
    vy = b.load(b.gep(y, [i], inbounds=True))
    result = combine(b, vx, vy, alpha)
    b.store(result, b.gep(out, [i], inbounds=True))
    i_next = b.add(i, ir.Constant(i32, 1))
    i.add_incoming(i_next, body)
    b.branch(cond)

    b.position_at_end(end)
    b.ret_void()
    return module


@dataclass
class ElementwiseKernel:
    """A JIT-compiled elementwise kernel. Keep it alive while calling it."""

    op: str
    needs_alpha: bool
    ir_text: str
    _engine: llvm.ExecutionEngine
    _fn_ptr: int

    def __call__(self, x: np.ndarray, y: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Run the kernel and return a fresh float32 output array.

        Inputs are coerced to contiguous float32; shapes must match.
        """
        xf = np.ascontiguousarray(x, dtype=np.float32)
        yf = np.ascontiguousarray(y, dtype=np.float32)
        if xf.shape != yf.shape:
            raise ValueError(f"shape mismatch: {xf.shape} vs {yf.shape}")
        out = np.empty_like(xf)
        n = xf.size

        f32p = ctypes.POINTER(ctypes.c_float)
        outp = out.ctypes.data_as(f32p)
        xp = xf.ctypes.data_as(f32p)
        yp = yf.ctypes.data_as(f32p)

        if self.needs_alpha:
            if alpha is None:
                raise ValueError(f"kernel {self.op!r} requires alpha")
            cfunc = ctypes.CFUNCTYPE(
                None, f32p, f32p, f32p, ctypes.c_float, ctypes.c_int32
            )(self._fn_ptr)
            cfunc(outp, xp, yp, ctypes.c_float(alpha), ctypes.c_int32(n))
        else:
            cfunc = ctypes.CFUNCTYPE(None, f32p, f32p, f32p, ctypes.c_int32)(self._fn_ptr)
            cfunc(outp, xp, yp, ctypes.c_int32(n))
        return out


def compile_elementwise_kernel(op: str) -> ElementwiseKernel:
    """Generate, verify, JIT-compile, and wrap the elementwise kernel `op`."""
    _ensure_llvm_initialized()
    needs_alpha, _ = _KERNELS[op]
    module = build_kernel_module(op)
    ir_text = str(module)

    llvm_module = llvm.parse_assembly(ir_text)
    llvm_module.verify()

    target_machine = llvm.Target.from_default_triple().create_target_machine()
    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), target_machine)
    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()

    fn_ptr = engine.get_function_address(f"drake_ew_{op}")
    return ElementwiseKernel(
        op=op, needs_alpha=needs_alpha, ir_text=ir_text, _engine=engine, _fn_ptr=fn_ptr
    )
