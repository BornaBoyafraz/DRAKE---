import numpy as np
import pytest

from codegen.llvm_kernels import (
    KERNEL_REFERENCE,
    build_kernel_module,
    build_matmul_module,
    compile_elementwise_kernel,
    compile_matmul_kernel,
)


def test_generated_ir_has_the_expected_structure():
    ir_text = str(build_kernel_module("add"))
    assert "define" in ir_text and "drake_ew_add" in ir_text
    # a real loop: induction phi, comparison, the fp op, memory access
    assert "phi" in ir_text
    assert "icmp" in ir_text
    assert "fadd" in ir_text
    assert "getelementptr" in ir_text
    assert "load" in ir_text and "store" in ir_text


@pytest.mark.parametrize("op", ["add", "mul"])
def test_binary_kernel_matches_numpy_bit_for_bit(op):
    kernel = compile_elementwise_kernel(op)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096).astype(np.float32)
    y = rng.standard_normal(4096).astype(np.float32)

    got = kernel(x, y)
    expected = KERNEL_REFERENCE[op](x, y)
    # float32 add/mul are the same IEEE op in LLVM and NumPy -> exactly equal.
    np.testing.assert_array_equal(got, expected)


def test_axpy_kernel_matches_numpy():
    kernel = compile_elementwise_kernel("axpy")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000).astype(np.float32)
    y = rng.standard_normal(1000).astype(np.float32)
    alpha = 2.5

    got = kernel(x, y, alpha=alpha)
    # LLVM emits fmul+fadd (not a fused FMA), matching NumPy's two-op evaluation.
    expected = (np.float32(alpha) * x + y).astype(np.float32)
    np.testing.assert_array_equal(got, expected)


def test_kernel_reshapes_and_preserves_shape():
    kernel = compile_elementwise_kernel("add")
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.ones((3, 4), dtype=np.float32)
    got = kernel(x, y)
    assert got.shape == (3, 4)
    np.testing.assert_array_equal(got, x + y)


@pytest.mark.parametrize("n", [0, 1, 2, 17, 1000])
def test_kernel_handles_various_lengths_including_empty(n):
    kernel = compile_elementwise_kernel("add")
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32) * 3.0
    np.testing.assert_array_equal(kernel(x, y), x + y)


def test_kernel_coerces_non_float32_and_non_contiguous_input():
    kernel = compile_elementwise_kernel("add")
    x = np.arange(20, dtype=np.float64)[::2]  # non-contiguous, wrong dtype
    y = np.ones(10, dtype=np.int32)
    got = kernel(x, y)
    np.testing.assert_array_equal(got, x.astype(np.float32) + y.astype(np.float32))


def test_shape_mismatch_raises():
    kernel = compile_elementwise_kernel("add")
    with pytest.raises(ValueError, match="shape mismatch"):
        kernel(np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32))


def test_axpy_without_alpha_raises():
    kernel = compile_elementwise_kernel("axpy")
    with pytest.raises(ValueError, match="requires alpha"):
        kernel(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_unknown_op_raises():
    with pytest.raises(ValueError, match="unknown kernel op"):
        build_kernel_module("no_such_op")


def test_two_kernels_are_independent():
    add = compile_elementwise_kernel("add")
    mul = compile_elementwise_kernel("mul")
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([4, 5, 6], dtype=np.float32)
    np.testing.assert_array_equal(add(x, y), x + y)
    np.testing.assert_array_equal(mul(x, y), x * y)


# ---- matmul kernel -------------------------------------------------------


def test_matmul_ir_has_three_nested_loops_and_accumulator():
    ir_text = str(build_matmul_module())
    assert "drake_matmul" in ir_text
    # m, n, k induction phis + acc phi = 4 phi nodes
    assert ir_text.count("phi") >= 4
    assert "fmul" in ir_text and "fadd" in ir_text
    assert ir_text.count("getelementptr") >= 3  # A, B, C indexing


@pytest.mark.parametrize(
    "m,k,n",
    [(1, 1, 1), (2, 3, 4), (8, 8, 8), (1, 16, 5), (7, 1, 3), (16, 32, 24)],
)
def test_matmul_matches_numpy(m, k, n):
    kernel = compile_matmul_kernel()
    rng = np.random.default_rng(m * 100 + k * 10 + n)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    got = kernel(a, b)
    expected = a @ b
    # Scalar fp32 accumulation vs. NumPy/BLAS differs only in rounding order.
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert got.shape == (m, n)


def test_matmul_coerces_dtype_and_layout():
    kernel = compile_matmul_kernel()
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.int32).reshape(3, 4)
    got = kernel(a, b)
    np.testing.assert_allclose(got, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-4)


def test_matmul_shape_mismatch_raises():
    kernel = compile_matmul_kernel()
    with pytest.raises(ValueError, match="incompatible matmul shapes"):
        kernel(np.zeros((2, 3), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))
