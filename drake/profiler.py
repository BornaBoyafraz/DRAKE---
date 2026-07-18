"""Runtime shape profiling.

Autoregressive decoding calls the same graph every step with a growing
``seq_len``. DRAKE records the (seq_len, batch) pair seen on every call so the
specialization pass can decide which shape *bucket* dominates traffic and is
therefore worth specializing for.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ShapeProfiler:
    observations: List[Tuple[int, int]] = field(default_factory=list)
    _counts: Counter = field(default_factory=Counter)

    def record(self, seq_len: int, batch: int) -> None:
        self.observations.append((seq_len, batch))
        self._counts[(seq_len, batch)] += 1

    def total_calls(self) -> int:
        return len(self.observations)

    def histogram(self) -> Counter:
        return Counter(self._counts)

    def dominant(self, top_k: int = 3) -> List[Tuple[Tuple[int, int], int]]:
        return self._counts.most_common(top_k)

    def seq_len_range(self) -> Tuple[int, int]:
        if not self.observations:
            return (0, 0)
        seqs = [s for s, _ in self.observations]
        return (min(seqs), max(seqs))
