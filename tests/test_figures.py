"""The figures are drawn to be published, so what is tested is what a reader
would notice: that both themes get written, that series stay distinguishable
without colour, and that a batch size the ceiling ruled out still leaves a mark.

Nothing here inspects pixels. The rendering is matplotlib's business; what this
file guards is the part that is ours -- which files exist, which ink each series
gets, and where a curve is allowed to stop.
"""

import pytest

from benchmarks.throughput.figures import THEMES, ink_ramp, plot, series_points
from benchmarks.throughput.report import build_rows, read_records
from benchmarks.throughput.sweep import read_manifest
from tests.test_report import entry, record, write_sweep

pytest.importorskip("matplotlib")


def sweep(tmp_path):
    """Two workloads over three batch sizes, the second stopped by memory."""
    return write_sweep(
        tmp_path,
        [
            (entry(workload="uniform", batch_size=1), record(workload="uniform")),
            (entry(workload="uniform", batch_size=2), record(workload="uniform")),
            (entry(workload="mixed-out", batch_size=1), record()),
            (entry(workload="mixed-out", batch_size=2, status="oom"), None),
        ],
        conditions={"model": "tiny", "dtype": "float32", "num_requests": 2},
    )


def test_both_themes_are_written_and_named_apart(tmp_path):
    """The page has a theme toggle, so a figure has to exist in both -- and the
    two names have to differ by exactly the word that swaps them."""
    written = plot(sweep(tmp_path), tmp_path / "figs")

    names = {path.name for path in written}
    assert len(names) == 6
    for name in names:
        assert name.replace("-light.", "-dark.") in names


def test_one_theme_writes_only_that_theme(tmp_path):
    written = plot(sweep(tmp_path), tmp_path / "figs", themes=("light",))

    assert {p.name for p in written} == {
        f"{tmp_path.name}-{panel}-light.png"
        for panel in ("throughput", "decode-step", "occupancy")
    }


def test_a_ceiling_is_carried_beside_the_points_it_ended(tmp_path):
    """A curve that stops and a curve that was stopped look the same until the
    batch size that failed is drawn, so it travels with the series."""
    rows = build_rows(read_manifest(sweep(tmp_path)), read_records(sweep(tmp_path)))

    points, ceiling = series_points(rows, "mixed-out", "out_tok_s", 1)

    assert [x for x, _ in points] == [1]
    assert ceiling == 2


def test_a_workload_that_never_ran_has_no_ceiling(tmp_path):
    rows = build_rows(read_manifest(sweep(tmp_path)), read_records(sweep(tmp_path)))

    assert series_points(rows, "uniform", "out_tok_s", 1)[1] is None


@pytest.mark.parametrize("theme", THEMES.values(), ids=list(THEMES))
def test_the_ink_ramp_spans_its_theme_end_to_end(theme):
    """Series are told apart by lightness alone, so the ramp has to use all of
    the range it is given -- a ramp bunched at one end is one colour."""
    ramp = ink_ramp(theme, 3)

    assert ramp[0] == theme.darkest
    assert ramp[-1] == theme.lightest
    assert len({*ramp}) == 3


def test_a_single_series_is_drawn_in_the_page_s_own_ink():
    assert ink_ramp(THEMES["light"], 1) == [THEMES["light"].darkest]
