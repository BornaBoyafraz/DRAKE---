
from ir import build_decode_step_graph
from passes.fusion import FusionPass
from passes.specialize import (
    DEFAULT_BATCH_BOUNDARIES,
    DEFAULT_SEQ_BOUNDARIES,
    SpecializationPass,
    build_bucket_table,
    classify,
)


def test_build_bucket_table_size():
    table = build_bucket_table(DEFAULT_SEQ_BOUNDARIES, DEFAULT_BATCH_BOUNDARIES)
    assert len(table) == (len(DEFAULT_SEQ_BOUNDARIES) + 1) * (len(DEFAULT_BATCH_BOUNDARIES) + 1)
    ids = [b.bucket_id for b in table]
    assert ids == list(range(len(table)))


def test_classify_matches_bucket_containment_exhaustively():
    table = build_bucket_table(DEFAULT_SEQ_BOUNDARIES, DEFAULT_BATCH_BOUNDARIES)
    for seq_len in range(0, 2000, 37):
        for batch in range(1, 32, 5):
            bucket_id = classify(seq_len, batch)
            bucket = table[bucket_id]
            assert bucket.contains(seq_len, batch), (seq_len, batch, bucket)
            # exactly one bucket should ever claim a given point
            claiming = [b for b in table if b.contains(seq_len, batch)]
            assert len(claiming) == 1


def test_specialization_picks_vector_attention_for_short_sequences():
    graph = build_decode_step_graph()
    fused_graph, _ = FusionPass().run(graph)
    table = build_bucket_table()
    short_bucket = table[classify(10, 1)]
    plan = SpecializationPass().specialize(fused_graph, short_bucket)
    kv_op = next(
        op for op in fused_graph.ops if op.attrs.get("fused_kind") == "fused_attention_kvupdate"
    )
    assert plan.variants[kv_op.name].name == "vector"


def test_specialization_picks_tiled_attention_for_long_sequences():
    graph = build_decode_step_graph()
    fused_graph, _ = FusionPass().run(graph)
    table = build_bucket_table()
    long_bucket = table[classify(5000, 1)]
    plan = SpecializationPass().specialize(fused_graph, long_bucket)
    kv_op = next(
        op for op in fused_graph.ops if op.attrs.get("fused_kind") == "fused_attention_kvupdate"
    )
    variant = plan.variants[kv_op.name]
    assert variant.name == "tiled"
    assert variant.params["tile_size"] == 128


def test_specialization_covers_every_fused_and_plain_op():
    graph = build_decode_step_graph()
    fused_graph, _ = FusionPass().run(graph)
    bucket = build_bucket_table()[0]
    plan = SpecializationPass().specialize(fused_graph, bucket)
    assert set(plan.variants.keys()) == {op.name for op in fused_graph.ops}
