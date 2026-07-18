"""Shape-bucket specialization.

Ahead-of-time kernel selection has to commit to one schedule for every shape
that might show up. DRAKE instead buckets the runtime-observed
``(seq_len, batch)`` pairs and picks a kernel variant *per bucket* for every
fused op -- e.g. a plain vectorized attention kernel while the KV cache is
short, and a tiled variant once it grows, without ever compiling a variant
for shapes that never actually occur.

The bucket boundaries here are the single source of truth shared with
``codegen/dispatch_jit.py``: the Python ``classify`` function below is the
reference semantics, and the LLVM IR built in dispatch_jit implements the
exact same boundary comparisons so the two are provably in agreement (see
``tests/test_dispatch_jit.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ir import Graph, Op

DEFAULT_SEQ_BOUNDARIES: Tuple[int, ...] = (128, 1024)
DEFAULT_BATCH_BOUNDARIES: Tuple[int, ...] = (8,)


@dataclass(frozen=True)
class ShapeBucket:
    bucket_id: int
    seq_lo: int
    seq_hi: int  # exclusive, -1 means unbounded
    batch_lo: int
    batch_hi: int  # exclusive, -1 means unbounded

    @property
    def name(self) -> str:
        seq = f"[{self.seq_lo},{self.seq_hi})" if self.seq_hi >= 0 else f"[{self.seq_lo},inf)"
        batch = f"[{self.batch_lo},{self.batch_hi})" if self.batch_hi >= 0 else f"[{self.batch_lo},inf)"
        return f"seq{seq}_batch{batch}"

    def contains(self, seq_len: int, batch: int) -> bool:
        seq_ok = self.seq_lo <= seq_len < self.seq_hi if self.seq_hi >= 0 else seq_len >= self.seq_lo
        batch_ok = (
            self.batch_lo <= batch < self.batch_hi if self.batch_hi >= 0 else batch >= self.batch_lo
        )
        return seq_ok and batch_ok


def build_bucket_table(
    seq_boundaries: Sequence[int] = DEFAULT_SEQ_BOUNDARIES,
    batch_boundaries: Sequence[int] = DEFAULT_BATCH_BOUNDARIES,
) -> List[ShapeBucket]:
    seq_edges = [0, *seq_boundaries, -1]
    batch_edges = [0, *batch_boundaries, -1]
    buckets = []
    bucket_id = 0
    for si in range(len(seq_edges) - 1):
        for bi in range(len(batch_edges) - 1):
            buckets.append(
                ShapeBucket(
                    bucket_id=bucket_id,
                    seq_lo=seq_edges[si],
                    seq_hi=seq_edges[si + 1],
                    batch_lo=batch_edges[bi],
                    batch_hi=batch_edges[bi + 1],
                )
            )
            bucket_id += 1
    return buckets


def classify(
    seq_len: int,
    batch: int,
    seq_boundaries: Sequence[int] = DEFAULT_SEQ_BOUNDARIES,
    batch_boundaries: Sequence[int] = DEFAULT_BATCH_BOUNDARIES,
) -> int:
    """Pure-Python reference bucket classifier (mirrored in LLVM IR)."""
    seq_idx = sum(1 for b in seq_boundaries if seq_len >= b)
    batch_idx = sum(1 for b in batch_boundaries if batch >= b)
    num_batch_buckets = len(batch_boundaries) + 1
    return seq_idx * num_batch_buckets + batch_idx


@dataclass(frozen=True)
class KernelVariant:
    name: str
    params: Dict[str, int] = field(default_factory=dict)


@dataclass
class KernelPlan:
    bucket: ShapeBucket
    variants: Dict[str, KernelVariant]


_ATTENTION_KINDS = {"fused_attention", "fused_attention_kvupdate"}
_MATMUL_KINDS = {"fused_norm_matmul", "fused_norm_matmul_gelu", "fused_matmul_gelu", "matmul"}


class SpecializationPass:
    def specialize(self, fused_graph: Graph, bucket: ShapeBucket) -> KernelPlan:
        variants: Dict[str, KernelVariant] = {}
        for op in fused_graph.ops:
            fused_kind = op.attrs.get("fused_kind") if op.kind == "fused" else op.kind
            variants[op.name] = self._variant_for(fused_kind, bucket)
        return KernelPlan(bucket=bucket, variants=variants)

    @staticmethod
    def _variant_for(fused_kind: str, bucket: ShapeBucket) -> KernelVariant:
        if fused_kind in _ATTENTION_KINDS:
            if bucket.seq_hi != -1 and bucket.seq_hi <= 128:
                return KernelVariant("vector")
            if bucket.seq_hi != -1 and bucket.seq_hi <= 1024:
                return KernelVariant("tiled", {"tile_size": 64})
            return KernelVariant("tiled", {"tile_size": 128})
        if fused_kind in _MATMUL_KINDS:
            if bucket.batch_hi != -1 and bucket.batch_hi <= 8:
                return KernelVariant("single_block")
            return KernelVariant("blocked_batch", {"block_size": 8})
        return KernelVariant("default")
