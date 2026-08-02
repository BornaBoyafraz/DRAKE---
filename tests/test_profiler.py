from profiler import ShapeProfiler


def test_record_total_calls_and_histogram():
    profiler = ShapeProfiler()

    profiler.record(seq_len=8, batch=2)
    profiler.record(seq_len=16, batch=1)
    profiler.record(seq_len=8, batch=2)

    assert profiler.total_calls() == 3
    assert profiler.histogram() == {(8, 2): 2, (16, 1): 1}


def test_dominant_orders_by_count_and_honors_top_k():
    profiler = ShapeProfiler()
    for observation in [(32, 1), (8, 2), (16, 1), (8, 2), (32, 1), (8, 2)]:
        profiler.record(*observation)

    assert profiler.dominant() == [((8, 2), 3), ((32, 1), 2), ((16, 1), 1)]
    assert profiler.dominant(top_k=2) == [((8, 2), 3), ((32, 1), 2)]


def test_seq_len_range_is_zero_when_empty_and_spans_observations():
    profiler = ShapeProfiler()
    assert profiler.seq_len_range() == (0, 0)

    profiler.record(seq_len=12, batch=1)
    profiler.record(seq_len=3, batch=4)
    profiler.record(seq_len=20, batch=2)

    assert profiler.seq_len_range() == (3, 20)
