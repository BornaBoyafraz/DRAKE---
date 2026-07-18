from codegen.dispatch_jit import compile_dispatch_engine
from passes.specialize import DEFAULT_BATCH_BOUNDARIES, DEFAULT_SEQ_BOUNDARIES, classify


def test_dispatch_ir_contains_expected_function():
    engine = compile_dispatch_engine(DEFAULT_SEQ_BOUNDARIES, DEFAULT_BATCH_BOUNDARIES)
    assert "define" in engine.ir_text
    assert "drake_dispatch" in engine.ir_text
    assert "icmp" in engine.ir_text


def test_jit_matches_python_reference_exhaustively():
    """The whole point of compiling this to LLVM is that it must be
    semantically identical to the Python reference classifier -- verified
    here over a dense grid, not just a handful of spot checks."""
    engine = compile_dispatch_engine(DEFAULT_SEQ_BOUNDARIES, DEFAULT_BATCH_BOUNDARIES)
    mismatches = []
    for seq_len in range(0, 4096, 3):
        for batch in range(0, 64, 2):
            expected = classify(seq_len, batch, DEFAULT_SEQ_BOUNDARIES, DEFAULT_BATCH_BOUNDARIES)
            actual = engine.classify(seq_len, batch)
            if actual != expected:
                mismatches.append((seq_len, batch, expected, actual))
    assert not mismatches, f"{len(mismatches)} mismatches, first few: {mismatches[:5]}"


def test_jit_handles_custom_boundaries():
    seq_boundaries = (16, 64, 256)
    batch_boundaries = (4,)
    engine = compile_dispatch_engine(seq_boundaries, batch_boundaries)
    for seq_len in (0, 15, 16, 63, 64, 255, 256, 1000):
        for batch in (0, 3, 4, 10):
            expected = classify(seq_len, batch, seq_boundaries, batch_boundaries)
            assert engine.classify(seq_len, batch) == expected


def test_two_independent_engines_do_not_interfere():
    engine_a = compile_dispatch_engine((128,), (8,))
    engine_b = compile_dispatch_engine((64, 512), (4, 16))
    assert engine_a.classify(200, 1) == classify(200, 1, (128,), (8,))
    assert engine_b.classify(200, 1) == classify(200, 1, (64, 512), (4, 16))
