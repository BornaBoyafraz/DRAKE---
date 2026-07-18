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
from typing import Dict, Optional, Sequence

import numpy as np

from codegen.dispatch_jit import DispatchEngine, compile_dispatch_engine
from codegen.fused_ops import Tensors, execute_graph, init_weights
from ir import Graph, build_decode_step_graph
from passes.fusion import FusionPass, FusionRecord, traffic_saved_bytes
from passes.specialize import (
    DEFAULT_BATCH_BOUNDARIES,
    DEFAULT_SEQ_BOUNDARIES,
    KernelPlan,
    ShapeBucket,
    SpecializationPass,
    build_bucket_table,
)
from profiler import ShapeProfiler


@dataclass
class StepResult:
    output: np.ndarray
    cache_k_out: np.ndarray
    cache_v_out: np.ndarray
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
        seq_boundaries: Sequence[int] = DEFAULT_SEQ_BOUNDARIES,
        batch_boundaries: Sequence[int] = DEFAULT_BATCH_BOUNDARIES,
        weight_seed: int = 0,
        graph: Optional[Graph] = None,
    ) -> None:
        self.base_graph = graph if graph is not None else build_decode_step_graph()
        self.fused_graph, self.fusion_records = FusionPass().run(self.base_graph)

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
        self.weights: Tensors = init_weights(self.static_dims, seed=weight_seed)

        self.profiler = ShapeProfiler()
        self.specializer = SpecializationPass()
        self.plan_cache: Dict[int, KernelPlan] = {}

    def step(self, x: np.ndarray, cache_k_in: np.ndarray, cache_v_in: np.ndarray) -> StepResult:
        batch = x.shape[0]
        seq_len = cache_k_in.shape[1]
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
        tensors.update({"x": x, "cache_k_in": cache_k_in, "cache_v_in": cache_v_in})
        tensors = execute_graph(self.fused_graph, tensors, dims)

        saved = sum(
            traffic_saved_bytes(op, self.base_graph, dims)
            for op in self.fused_graph.ops
            if op.kind == "fused"
        )

        return StepResult(
            output=tensors["output"],
            cache_k_out=tensors["cache_k_out"],
            cache_v_out=tensors["cache_v_out"],
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
