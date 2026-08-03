import numpy as np
import pytest

from codegen.llvm_kernels import (
    KERNEL_REFERENCE,
    build_kernel_module,
    compile_elementwise_kernel,
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
