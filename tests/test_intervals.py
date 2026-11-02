from hypothesis import given
from hypothesis import strategies as st

from flowday.scheduler import merge_intervals, overlap_minutes, subtract_intervals

interval = st.tuples(st.integers(0, 200), st.integers(0, 200)).map(lambda t: (min(t), max(t)))
interval_lists = st.lists(interval, max_size=6)


def minutes(intervals):
    return {m for a, b in intervals for m in range(a, b)}


def test_merge_joins_overlapping_and_touching():
    assert merge_intervals([(5, 10), (0, 3), (3, 5), (20, 25), (22, 30)]) == [(0, 10), (20, 30)]


def test_subtract_examples():
    assert subtract_intervals([(0, 100)], [(10, 20), (50, 60)]) == [(0, 10), (20, 50), (60, 100)]
    assert subtract_intervals([(0, 100)], [(0, 100)]) == []
    assert subtract_intervals([(0, 10), (20, 30)], [(5, 25)]) == [(0, 5), (25, 30)]


def test_overlap_minutes():
    assert overlap_minutes((0, 10), (5, 20)) == 5
    assert overlap_minutes((0, 10), (10, 20)) == 0


@given(interval_lists, interval_lists)
def test_subtract_matches_set_difference(base, blocked):
    result = subtract_intervals(base, blocked)
    assert minutes(result) == minutes(base) - minutes(blocked)
    assert all(a < b for a, b in result)
    assert all(result[i][1] < result[i + 1][0] for i in range(len(result) - 1))  # disjoint, not touching


@given(interval_lists)
def test_merge_preserves_coverage(intervals):
    assert minutes(merge_intervals(intervals)) == minutes(intervals)
