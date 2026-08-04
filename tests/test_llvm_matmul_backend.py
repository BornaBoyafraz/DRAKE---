"""The whole decode step, executed with matmuls lowered to LLVM.

This is the payoff of the pluggable executor: the same graph, same passes,
but every matmul runs as JIT-compiled native code from codegen.llvm_kernels
instead of NumPy -- and the result still matches the NumPy backend within
float32 tolerance.
"""

import numpy as np

from codegen.fused_ops import execute_graph, init_step_inputs, init_weights
from codegen.llvm_kernels import llvm_op_overrides
from ir import build_decode_step_graph, make_dims
from passes.fusion import FusionPass


def _run(num_layers: int, overrides):
    graph = build_decode_step_graph(num_layers=num_layers)
    fused, _ = FusionPass().run(graph)
    dims = make_dims(batch=2, seq_len=8, hidden_dim=32, n_heads=4, head_dim=8, ffn_dim=64)
    tensors = {
        **init_weights(dims, seed=4, num_layers=num_layers),
        **init_step_inputs(dims, seed=6, num_layers=num_layers),
    }
    return execute_graph(fused, tensors, dims, op_overrides=overrides)


def test_decode_step_with_llvm_matmul_matches_numpy_backend():
    numpy_out = _run(1, None)
    llvm_out = _run(1, llvm_op_overrides())
    for name in ("output", "cache_k_out", "cache_v_out"):
        np.testing.assert_allclose(numpy_out[name], llvm_out[name], rtol=1e-3, atol=1e-4)


def test_multi_layer_decode_step_with_llvm_matmul_matches_numpy():
    numpy_out = _run(3, None)
    llvm_out = _run(3, llvm_op_overrides())
    for name in ("output", "l0_cache_k_out", "l2_cache_v_out"):
        np.testing.assert_allclose(numpy_out[name], llvm_out[name], rtol=1e-3, atol=1e-4)


def test_override_only_replaces_matmul_other_ops_untouched():
    overrides = llvm_op_overrides()
    assert set(overrides) == {"matmul"}
