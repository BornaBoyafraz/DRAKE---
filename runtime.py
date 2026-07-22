"""DrakeEngine: wires profiling, fusion, specialization, LLVM dispatch, and
the reference execution backend into one callable per-decode-step engine.

Call sequence per `step()`:
  1. record the (seq_len, batch) shape with the ShapeProfiler
  2. classify it into a bucket using the LLVM-JIT'd dispatch function
  3. look up (or lazily build, then cache) the KernelPlan for that bucket
  4. execute the fused graph against the reference numpy backend
  5. report the analytic HBM-traffic bytes fusion saved for this call
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from codegen.dispatch_jit import DispatchEngine, compile_dispatch_engine
from codegen.fused_ops import Tensors, execute_graph, init_weights
from ir import Graph, build_decode_step_graph, cache_io_names
from passes.dce import eliminate_dead_code
from passes.fusion import FusionPass, traffic_saved_bytes
from passes.specialize import (
    DEFAULT_BATCH_BOUNDARIES,
    DEFAULT_SEQ_BOUNDARIES,
    KernelPlan,
    ShapeBucket,
    SpecializationPass,
    build_bucket_table,
)
from passes.verify import verify_graph
from profiler import ShapeProfiler

CacheArg = Union[np.ndarray, List[np.ndarray]]


def _as_list(arg: CacheArg) -> List[np.ndarray]:
    return arg if isinstance(arg, list) else [arg]


@dataclass
class StepResult:
    output: np.ndarray
    cache_k_out: CacheArg
    cache_v_out: CacheArg
    bucket: ShapeBucket
    plan: KernelPlan
    traffic_saved_bytes: int


class DrakeEngine:
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        ffn_dim: int,
        num_layers: int = 1,
        seq_boundaries: Sequence[int] = DEFAULT_SEQ_BOUNDARIES,
        batch_boundaries: Sequence[int] = DEFAULT_BATCH_BOUNDARIES,
        weight_seed: int = 0,
        graph: Optional[Graph] = None,
    ) -> None:
        self.num_layers = num_layers
        self.base_graph = graph if graph is not None else build_decode_step_graph(num_layers)
        verify_graph(self.base_graph)
        self.fused_graph, self.fusion_records = FusionPass().run(self.base_graph)
        # Fusion must preserve every structural invariant; verify it did.
        verify_graph(self.fused_graph)
        # Clean up any ops fusion left with no live consumers, then re-verify.
        # A no-op on today's graph, but it makes the pipeline correct-by-
        # construction for future passes that introduce dead nodes.
        self.fused_graph, self.dce_removed = eliminate_dead_code(self.fused_graph)
        verify_graph(self.fused_graph)

        self.seq_boundaries = tuple(seq_boundaries)
        self.batch_boundaries = tuple(batch_boundaries)
        self.bucket_table = build_bucket_table(self.seq_boundaries, self.batch_boundaries)
        self.dispatch_engine: DispatchEngine = compile_dispatch_engine(
            self.seq_boundaries, self.batch_boundaries
        )

        self.static_dims = {
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "qkv_dim": 3 * hidden_dim,
            "ffn_dim": ffn_dim,
        }
        self.weights: Tensors = init_weights(self.static_dims, seed=weight_seed, num_layers=num_layers)

        self.profiler = ShapeProfiler()
        self.specializer = SpecializationPass()
        self.plan_cache: Dict[int, KernelPlan] = {}

    def step(self, x: np.ndarray, cache_k_in: CacheArg, cache_v_in: CacheArg) -> StepResult:
        """Run one decode step.

        For `num_layers == 1` (the common case), `cache_k_in`/`cache_v_in`
        are plain arrays and `cache_k_out`/`cache_v_out` on the result are
        too -- unchanged from the original single-layer API. For
        `num_layers > 1`, pass/receive a list of one array per layer, in
        layer order.
        """
        k_list = _as_list(cache_k_in)
        v_list = _as_list(cache_v_in)

        batch = x.shape[0]
        seq_len = k_list[0].shape[1]
        self.profiler.record(seq_len, batch)

        bucket_id = self.dispatch_engine.classify(seq_len, batch)
        plan = self.plan_cache.get(bucket_id)
        if plan is None:
            bucket = self.bucket_table[bucket_id]
            plan = self.specializer.specialize(self.fused_graph, bucket)
            self.plan_cache[bucket_id] = plan

        dims = dict(self.static_dims)
        dims.update({"batch": batch, "seq_len": seq_len, "next_seq_len": seq_len + 1})

        tensors: Tensors = dict(self.weights)
        tensors["x"] = x
        for i in range(self.num_layers):
            k_in_name, v_in_name, _, _ = cache_io_names(self.num_layers, i)
            tensors[k_in_name] = k_list[i]
            tensors[v_in_name] = v_list[i]
        tensors = execute_graph(self.fused_graph, tensors, dims)

        saved = sum(
            traffic_saved_bytes(op, self.base_graph, dims)
            for op in self.fused_graph.ops
            if op.kind == "fused"
        )

        cache_k_out: CacheArg
        cache_v_out: CacheArg
        if self.num_layers > 1:
            names = [cache_io_names(self.num_layers, i) for i in range(self.num_layers)]
            cache_k_out = [tensors[k_out] for _, _, k_out, _ in names]
            cache_v_out = [tensors[v_out] for _, _, _, v_out in names]
        else:
            cache_k_out = tensors["cache_k_out"]
            cache_v_out = tensors["cache_v_out"]

        return StepResult(
            output=tensors["output"],
            cache_k_out=cache_k_out,
            cache_v_out=cache_v_out,
            bucket=plan.bucket,
            plan=plan,
            traffic_saved_bytes=saved,
        )

    def fusion_summary(self) -> Dict:
        return {
            "original_op_count": len(self.base_graph.ops),
            "fused_op_count": len(self.fused_graph.ops),
            "fusions": [
                {"fused_op": r.fused_op_name, "kind": r.fused_kind, "sub_ops": r.sub_op_names}
                for r in self.fusion_records
            ],
        }

    def dispatch_ir(self) -> str:
        return self.dispatch_engine.ir_text
