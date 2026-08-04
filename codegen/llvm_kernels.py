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


def _jit_module(module: ir.Module, fn_name: str) -> tuple[llvm.ExecutionEngine, int, str]:
    """Verify, MCJIT-compile a module, and return (engine, function pointer,
    IR text). The engine owns the compiled code and must be kept alive."""
    _ensure_llvm_initialized()
    ir_text = str(module)
    llvm_module = llvm.parse_assembly(ir_text)
    llvm_module.verify()
    target_machine = llvm.Target.from_default_triple().create_target_machine()
    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), target_machine)
    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()
    return engine, engine.get_function_address(fn_name), ir_text


def compile_elementwise_kernel(op: str) -> ElementwiseKernel:
    """Generate, verify, JIT-compile, and wrap the elementwise kernel `op`."""
    needs_alpha, _ = _KERNELS[op]
    engine, fn_ptr, ir_text = _jit_module(build_kernel_module(op), f"drake_ew_{op}")
    return ElementwiseKernel(
        op=op, needs_alpha=needs_alpha, ir_text=ir_text, _engine=engine, _fn_ptr=fn_ptr
    )


def build_matmul_module() -> ir.Module:
    """Emit ``void drake_matmul(float* C, float* A, float* B, i32 M, i32 N,
    i32 K)`` computing the row-major product ``C[M,N] = A[M,K] @ B[K,N]``.

    Three nested loops (m, n, k) with the dot-product accumulated in a
    register (an ``acc`` phi threaded through the k-loop), 2D indexing done
    explicitly as ``row * stride + col`` -- the canonical naive GEMM, but as
    genuine LLVM IR that JIT-compiles to native code.
    """
    module = ir.Module(name="drake_matmul")
    f32 = ir.FloatType()
    f32p = ir.PointerType(f32)
    i32 = ir.IntType(32)

    fn_ty = ir.FunctionType(ir.VoidType(), [f32p, f32p, f32p, i32, i32, i32])
    fn = ir.Function(module, fn_ty, name="drake_matmul")
    c, a, b, m_dim, n_dim, k_dim = fn.args
    c.name, a.name, b.name, m_dim.name, n_dim.name, k_dim.name = "C", "A", "B", "M", "N", "K"

    entry = fn.append_basic_block("entry")
    m_cond = fn.append_basic_block("m.cond")
    m_body = fn.append_basic_block("m.body")
    n_cond = fn.append_basic_block("n.cond")
    n_body = fn.append_basic_block("n.body")
    k_cond = fn.append_basic_block("k.cond")
    k_body = fn.append_basic_block("k.body")
    k_end = fn.append_basic_block("k.end")
    n_latch = fn.append_basic_block("n.latch")
    m_latch = fn.append_basic_block("m.latch")
    exit_bb = fn.append_basic_block("exit")

    zero = ir.Constant(i32, 0)
    one = ir.Constant(i32, 1)
    fzero = ir.Constant(f32, 0.0)

    bld = ir.IRBuilder(entry)
    bld.branch(m_cond)

    # for m in range(M)
    bld.position_at_end(m_cond)
    m = bld.phi(i32, name="m")
    m.add_incoming(zero, entry)
    bld.cbranch(bld.icmp_signed("<", m, m_dim), m_body, exit_bb)

    bld.position_at_end(m_body)
    bld.branch(n_cond)

    # for n in range(N)
    bld.position_at_end(n_cond)
    n = bld.phi(i32, name="n")
    n.add_incoming(zero, m_body)
    bld.cbranch(bld.icmp_signed("<", n, n_dim), n_body, m_latch)

    bld.position_at_end(n_body)
    bld.branch(k_cond)

    # acc = 0; for k in range(K): acc += A[m,k] * B[k,n]
    bld.position_at_end(k_cond)
    k = bld.phi(i32, name="k")
    acc = bld.phi(f32, name="acc")
    k.add_incoming(zero, n_body)
    acc.add_incoming(fzero, n_body)
    bld.cbranch(bld.icmp_signed("<", k, k_dim), k_body, k_end)

    bld.position_at_end(k_body)
    a_idx = bld.add(bld.mul(m, k_dim), k)  # m*K + k
    b_idx = bld.add(bld.mul(k, n_dim), n)  # k*N + n
    a_val = bld.load(bld.gep(a, [a_idx], inbounds=True))
    b_val = bld.load(bld.gep(b, [b_idx], inbounds=True))
    acc_next = bld.fadd(acc, bld.fmul(a_val, b_val))
    k_next = bld.add(k, one)
    k.add_incoming(k_next, k_body)
    acc.add_incoming(acc_next, k_body)
    bld.branch(k_cond)

    # C[m,n] = acc
    bld.position_at_end(k_end)
    c_idx = bld.add(bld.mul(m, n_dim), n)  # m*N + n
    bld.store(acc, bld.gep(c, [c_idx], inbounds=True))
    bld.branch(n_latch)

    bld.position_at_end(n_latch)
    n.add_incoming(bld.add(n, one), n_latch)
    bld.branch(n_cond)

    bld.position_at_end(m_latch)
    m.add_incoming(bld.add(m, one), m_latch)
    bld.branch(m_cond)

    bld.position_at_end(exit_bb)
    bld.ret_void()
    return module


@dataclass
class MatmulKernel:
    """A JIT-compiled row-major float32 GEMM. Keep it alive while calling."""

    ir_text: str
    _engine: llvm.ExecutionEngine
    _fn_ptr: int

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        af = np.ascontiguousarray(a, dtype=np.float32)
        bf = np.ascontiguousarray(b, dtype=np.float32)
        if af.ndim != 2 or bf.ndim != 2 or af.shape[1] != bf.shape[0]:
            raise ValueError(f"incompatible matmul shapes: {af.shape} @ {bf.shape}")
        m_dim, k_dim = af.shape
        k2, n_dim = bf.shape
        out = np.zeros((m_dim, n_dim), dtype=np.float32)

        f32p = ctypes.POINTER(ctypes.c_float)
        cfunc = ctypes.CFUNCTYPE(
            None, f32p, f32p, f32p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
        )(self._fn_ptr)
        cfunc(
            out.ctypes.data_as(f32p),
            af.ctypes.data_as(f32p),
            bf.ctypes.data_as(f32p),
            ctypes.c_int32(m_dim),
            ctypes.c_int32(n_dim),
            ctypes.c_int32(k_dim),
        )
        return out


def compile_matmul_kernel() -> MatmulKernel:
    """Generate, verify, JIT-compile, and wrap the row-major GEMM kernel."""
    engine, fn_ptr, ir_text = _jit_module(build_matmul_module(), "drake_matmul")
    return MatmulKernel(ir_text=ir_text, _engine=engine, _fn_ptr=fn_ptr)


_MATMUL_KERNEL: MatmulKernel | None = None


def llvm_op_overrides() -> dict[str, Callable]:
    """Op-kind -> implementation map for ``fused_ops.execute_graph``, lowering
    ``matmul`` through the LLVM-JIT'd GEMM (compiled once, then cached).

    Passing this to ``execute_graph(..., op_overrides=llvm_op_overrides())``
    runs an entire decode step's matmuls as native LLVM-compiled code instead
    of NumPy, without changing the IR or any pass. The decode graph's matmuls
    are all 2D, which is exactly what ``MatmulKernel`` handles.
    """
    global _MATMUL_KERNEL
    if _MATMUL_KERNEL is None:
        _MATMUL_KERNEL = compile_matmul_kernel()
    kernel = _MATMUL_KERNEL

    def _matmul(t, op, dims):  # type: ignore[no-untyped-def]
        t[op.outputs[0]] = kernel(t[op.inputs[0]], t[op.inputs[1]])

    return {"matmul": _matmul}
