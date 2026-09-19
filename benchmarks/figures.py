"""Draw a sweep's curves in the notebook's own hand.

The tables in `report.py` are for reading a folder; these are for publishing one
-- and they land in a page whose whole visual argument is ink on paper: an
editorial serif, one hairline rule, no accent colour. A chart that arrives in
matplotlib's defaults (Helvetica-ish sans, a boxed frame, category-10 blue and
orange) reads as a screenshot from somewhere else, no matter how right its
numbers are. So the page's own tokens are restated here, once, and every mark
below is drawn out of them.

Three consequences follow from "no accent colour", and they are the whole design:

  * Series are told apart by *ink*, not hue -- evenly spaced steps from the
    page's text colour down towards its faintest -- and every step is doubled by
    a marker shape and a label at the end of its own line. Identity is never
    carried by colour alone, which is the accessibility rule anyway; here it is
    simply not carried by colour at all.
  * The page has two themes and a toggle, so a baked-in figure has to be baked
    twice. `plot` writes a light and a dark variant by default; swapping them is
    the page's problem, and the README carries the rule that does it.
  * Text sizes are chosen in *displayed* pixels, not points. A figure rendered
    at 200 dpi and shown across the page's 41rem measure is halved on the way,
    so a 13pt title lands at roughly the 18px of an `h3` and a 8.5pt tick label
    at the 12px of the site's metadata line.

Nothing here recomputes a number. Every point comes from `report.build_rows`,
so a curve and the table above it cannot disagree.
"""

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path

from benchmarks.report import Row, build_rows, read_records
from benchmarks.sweep import read_manifest

# The site self-hosts these two faces; instanced copies live next to this module
# so a figure drawn on a laptop matches one drawn in CI, the same way a pinned
# commit makes two trials comparable. Absent, matplotlib walks the stacks below
# -- which are the page's own CSS fallbacks, in the page's own order.
FONT_DIR = Path(__file__).parent / "assets" / "fonts"

SERIF = [
    "Source Serif 4",
    "Charter",
    "Bitstream Charter",
    "Iowan Old Style",
    "Palatino",
    "Georgia",
    "DejaVu Serif",
]
MONO = ["JetBrains Mono", "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"]

# Marker shapes, in the order series are handed them. Six is not a limit that
# can be raised: past it the ink steps are closer than a reader can separate,
# and a sweep with that many workloads wants one figure each, not one figure.
MARKERS = ("o", "s", "^", "D", "v", "P")


@dataclass(frozen=True)
class Theme:
    """One of the page's two palettes, plus the two ends of its ink ramp.

    `paper` through `rule` are the site's custom properties verbatim. `darkest`
    and `lightest` are not: they bound the range series are drawn from, and the
    faint end is one step stronger than the page's `--fg-faint`. That token is
    tuned for text a reader can zoom, select and re-render at their own size; a
    figure is pixels, at whatever size the column gives it, so the faintest ink
    here clears 3:1 against the paper instead of the 2.8:1 the page allows
    itself.
    """

    paper: str  # --bg
    ink: str  # --fg
    muted: str  # --fg-muted
    faint: str  # --fg-faint, one step stronger (see above)
    rule: str  # --rule
    rule_strong: str  # --rule-strong
    darkest: str
    lightest: str


# The site's own tokens, copied out of its stylesheet rather than guessed at.
# The page is upstream of this file and lives in another repo, so nothing here
# notices when it changes; the values below were read off
#   https://lxynov.github.io/tiny-llm-serve/assets/main.css
# as of 2026-08, and re-reading that `:root` block is the whole of keeping them
# current. Every figure in the repo is drawn from this dict, so a palette that
# moved is one edit, not a sweep through the drawing code.
THEMES = {
    "light": Theme(
        paper="#fdfdfb",
        ink="#1c1b19",
        muted="#6a6862",
        faint="#8f8c83",
        rule="#e6e4dd",
        rule_strong="#cecbc1",
        darkest="#1c1b19",
        lightest="#8f8c83",
    ),
    "dark": Theme(
        paper="#141413",
        ink="#eae7e0",
        muted="#96938a",
        faint="#7b7870",
        rule="#2b2b28",
        rule_strong="#3e3d38",
        darkest="#eae7e0",
        lightest="#7b7870",
    ),
}


@dataclass(frozen=True)
class Panel:
    """One figure: which column of the report it draws, and how it reads.

    The title says what is measured, so no rotated y-axis label is needed --
    which is also why each panel is its own file rather than a subplot in a
    shared grid. A post drops the one it is arguing about into the paragraph
    that argues it.
    """

    name: str  # file stem suffix
    title: str  # what the y axis measures, spelled out
    attr: str  # the `Row` attribute behind it
    log: bool = False
    percent: bool = False  # a fraction in the record, a percentage on the page
    linear_guide: bool = False


PANELS = (
    Panel(
        name="throughput",
        title="Output tokens per second",
        attr="out_tok_s",
        log=True,
        linear_guide=True,
    ),
    Panel(
        name="decode-step",
        title="Milliseconds per decode step",
        attr="ms_per_step",
    ),
    Panel(
        name="occupancy",
        title="Share of decode slots still generating",
        attr="occupancy",
        percent=True,
    ),
)

# The sheet, in figure coordinates. Laid out by hand rather than by
# `tight_layout`, because the right margin is not slack -- it is where the
# series labels live, and a layout engine that reclaims it puts them off-page.
FIGSIZE = (6.6, 4.3)
DPI = 200
AXES_RECT = (0.10, 0.185, 0.675, 0.605)
TITLE_Y, SUBTITLE_Y, XLABEL_Y, FOOTER_Y = 0.955, 0.898, 0.088, 0.032

# Two labels closer than this share of the axes height are pushed apart. Sized
# to one line of the label font plus its leading.
LABEL_GAP = 0.058


def _rgb(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def ink_ramp(theme: Theme, count: int) -> list[str]:
    """`count` steps of ink, evenly spaced from the darkest to the faintest.

    Interpolated in sRGB, which is close enough to perceptually even for a
    ramp this short: sRGB's transfer curve is roughly the cube root that L*
    uses, so equal steps here land within a point or two of equal lightness.
    """
    if count < 2:
        return [theme.darkest]
    start, end = _rgb(theme.darkest), _rgb(theme.lightest)
    return [
        "#%02x%02x%02x"
        % tuple(round(a + (b - a) * i / (count - 1)) for a, b in zip(start, end))
        for i in range(count)
    ]


def tracked(text: str) -> str:
    """Letter-spacing, which matplotlib has no setting for.

    The page sets its small uppercase labels in mono with 0.04-0.06em of
    tracking; spaces are the only way to say that here, and at this size they
    land close enough.
    """
    return " ".join(text)


def rc(theme: Theme) -> dict:
    """The page's typography and palette, as matplotlib settings."""
    return {
        "font.family": "serif",
        "font.serif": SERIF,
        "font.monospace": MONO,
        "font.size": 8.5,
        "text.color": theme.ink,
        "figure.facecolor": theme.paper,
        "axes.facecolor": theme.paper,
        "savefig.facecolor": theme.paper,
        "axes.edgecolor": theme.rule_strong,
        "axes.linewidth": 0.7,
        "axes.labelcolor": theme.muted,
        "xtick.color": theme.rule_strong,
        "ytick.color": theme.rule_strong,
        "xtick.labelcolor": theme.muted,
        "ytick.labelcolor": theme.muted,
        "xtick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.size": 0,
        "grid.color": theme.rule,
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    }


def load_fonts() -> None:
    """Register the vendored faces, if they are there."""
    from matplotlib import font_manager

    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))


def series_points(rows: list[Row], workload: str, attr: str, scale: float):
    """One workload's curve, and the batch size that ended it.

    The ceiling is returned alongside the points because it is the one outcome
    with no point of its own: a batch size the sweep tried and could not fit,
    or ruled out after one that could not. Drawn, it is the difference between
    a curve that stops and a curve that was stopped.
    """
    group = [r for r in rows if r.workload == workload]
    points = [
        (r.batch_size, getattr(r, attr) * scale)
        for r in group
        if r.batch_size and getattr(r, attr) is not None
    ]
    failed = [r.batch_size for r in group if r.batch_size and r.status != "ok"]
    return points, min(failed) if failed else None


def format_value(value: float, _pos: int = 0) -> str:
    """Tick labels the way the report prints numbers: grouped, no trailing .0."""
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def draw_series(axes, rows: list[Row], panel: Panel, theme: Theme) -> list[tuple]:
    """Every workload's curve, and where each one's label wants to sit."""
    scale = 100 if panel.percent else 1
    workloads = list(dict.fromkeys(r.workload for r in rows))
    colours = ink_ramp(theme, len(workloads))
    labels = []
    for index, workload in enumerate(workloads):
        # Lighter over darker, which is the opposite of the instinct and the
        # right way round: where two curves coincide -- and on the occupancy
        # panel two of them nearly do -- a pale mark still leaves the dark line
        # readable around it, while a dark mark erases the pale one outright.
        layer = 3 + index
        points, ceiling = series_points(rows, workload, panel.attr, scale)
        if not points:
            continue
        colour = colours[index % len(colours)]
        axes.plot(
            *zip(*points),
            marker=MARKERS[index % len(MARKERS)],
            color=colour,
            markerfacecolor=colour,
            markeredgecolor=theme.paper,
            zorder=layer,
        )
        last_x, last_y = points[-1]
        if ceiling and ceiling > last_x:
            # A dotted stub out to the size that failed, ending in an open
            # cross: the curve did not level off there, it ran out of memory.
            axes.plot(
                [last_x, ceiling],
                [last_y, last_y],
                linestyle=(0, (1, 2)),
                linewidth=1,
                color=colour,
                zorder=layer - 2,
            )
            axes.plot(
                [ceiling],
                [last_y],
                marker="x",
                markersize=4.5,
                markeredgewidth=1.1,
                color=colour,
                zorder=layer,
            )
        labels.append((last_y, workload, colour))
    return labels


def place_labels(axes, labels: list[tuple]) -> None:
    """The legend, written at the end of each line instead of in a box.

    Lines that finish close together -- which on the occupancy panel is most of
    them -- would overprint, so the labels are nudged apart in axes space after
    the limits are final. Nudged, not re-ordered: a label that crossed its
    neighbour would point at the wrong curve.
    """
    to_axes = axes.transAxes.inverted()
    placed = sorted(
        (to_axes.transform(axes.transData.transform((1, y)))[1], name, colour)
        for y, name, colour in labels
    )
    for i in range(1, len(placed)):
        y, name, colour = placed[i]
        floor = placed[i - 1][0] + LABEL_GAP
        if y < floor:
            placed[i] = (floor, name, colour)
    for y, name, colour in placed:
        axes.text(
            1.035,
            y,
            name,
            transform=axes.transAxes,
            family="monospace",
            fontsize=8,
            color=colour,
            va="center",
            ha="left",
            clip_on=False,
        )


def draw_linear_guide(axes, rows: list[Row], theme: Theme) -> None:
    """What perfect scaling would have looked like, for the eye to fall short of.

    On log-log axes every curve looks like a line, so a reader cannot see that
    64x the batch bought 62x the throughput while 256x bought 24% of linear.
    Against a slope-one guide from the smallest batch that ran, they can. The
    guide leaves the top of the plot almost immediately, which is the point --
    the y limits come from the data, never from the guide.
    """
    points = [
        (r.batch_size, r.out_tok_s)
        for r in rows
        if r.batch_size and r.out_tok_s is not None
    ]
    if not points:
        return
    base_x = min(x for x, _ in points)
    base_y = max(y for x, y in points if x == base_x)
    left, right = axes.get_xlim()
    bottom, top = axes.get_ylim()
    axes.plot(
        [base_x, right],
        [base_y, base_y * right / base_x],
        linestyle=(0, (1, 2.5)),
        linewidth=1,
        color=theme.faint,
        zorder=1,
    )
    # Anchored where the guide is still on the sheet, and rotated to sit on it:
    # the angle is only knowable in display space, once the limits are fixed.
    exit_x = base_x * top / base_y
    anchor_x = base_x * (min(exit_x, right) / base_x) ** 0.55
    anchor_y = base_y * anchor_x / base_x
    (x0, y0), (x1, y1) = (
        axes.transData.transform((anchor_x, anchor_y)),
        axes.transData.transform((anchor_x * 2, anchor_y * 2)),
    )
    axes.annotate(
        "linear scaling",
        (anchor_x, anchor_y),
        xytext=(0, 4),
        textcoords="offset points",
        rotation=degrees(atan2(y1 - y0, x1 - x0)),
        rotation_mode="anchor",
        family="monospace",
        fontsize=7,
        color=theme.faint,
        ha="center",
        va="bottom",
    )
    axes.set_xlim(left, right)
    axes.set_ylim(bottom, top)


def caption(run_dir: Path, index: dict, records: dict[str, dict]) -> tuple[str, str]:
    """The two provenance lines: what was held fixed, and what produced it.

    Both are the report's own header, set the way the page sets a byline. A
    figure travels further than the folder it came from, so it carries the
    commit with it.
    """
    conditions = index.get("conditions") or {}
    environment = next(iter(records.values()), {}).get("environment") or {}
    parts = [
        str(conditions.get("model") or ""),
        str(conditions.get("dtype") or ""),
        str(environment.get("gpu") or conditions.get("device") or ""),
    ]
    if conditions.get("num_requests"):
        parts.append(f"{conditions['num_requests']} requests")
    commits = {str(r.get("commit")) for r in records.values()}
    commit = commits.pop()[:12] if len(commits) == 1 else "MIXED"
    return (
        " · ".join(p for p in parts if p),
        f"{index.get('sweep_id', run_dir.name)} · commit {commit}",
    )


def draw(panel: Panel, rows: list[Row], theme: Theme, subtitle: str, footer: str):
    """One panel, on one sheet of the page's paper."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (
        FuncFormatter,
        LogLocator,
        MaxNLocator,
        MultipleLocator,
        NullLocator,
    )

    figure = plt.figure(figsize=FIGSIZE)
    axes = figure.add_axes(AXES_RECT)

    axes.set_xscale("log", base=2)
    if panel.log:
        axes.set_yscale("log")
    labels = draw_series(axes, rows, panel, theme)

    sizes = sorted({r.batch_size for r in rows if r.batch_size})
    axes.set_xlim(sizes[0] / 1.6, sizes[-1] * 1.6)
    axes.set_xticks(sizes)
    axes.set_xticklabels([str(s) for s in sizes], family="monospace")
    axes.xaxis.set_minor_locator(NullLocator())  # every power of two is a major

    values = [y for y, _, _ in labels] + [
        getattr(r, panel.attr) * (100 if panel.percent else 1)
        for r in rows
        if getattr(r, panel.attr) is not None
    ]
    if panel.percent:
        axes.set_ylim(0, 105)
        axes.yaxis.set_major_locator(MultipleLocator(25))
        axes.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    elif panel.log:
        axes.set_ylim(min(values) / 1.2, max(values) * 1.3)
        axes.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axes.yaxis.set_major_formatter(FuncFormatter(format_value))
        axes.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    else:
        # Up to the first gridline that clears the data, rather than to a
        # fraction above it: a band of paper with no rule in it reads as a
        # margin, and puts the curve lower on the sheet than it belongs.
        axes.set_ylim(0, max(values) * 1.05)
        axes.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        axes.yaxis.set_major_formatter(FuncFormatter(format_value))
        above = [t for t in axes.get_yticks() if t >= max(values)]
        if above:
            axes.set_ylim(0, min(above))
    for label in axes.get_yticklabels():
        label.set_family("monospace")

    axes.grid(axis="y", zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)

    if panel.linear_guide:
        draw_linear_guide(axes, rows, theme)
    place_labels(axes, labels)

    left = AXES_RECT[0]
    figure.text(left, TITLE_Y, panel.title, fontsize=13, weight=600, color=theme.ink)
    figure.text(
        left, SUBTITLE_Y, subtitle, family="monospace", fontsize=8, color=theme.muted
    )
    figure.text(
        left + AXES_RECT[2] / 2,
        XLABEL_Y,
        tracked("BATCH SIZE"),
        family="monospace",
        fontsize=7,
        color=theme.faint,
        ha="center",
    )
    # The cross is the one mark on the sheet a reader cannot infer, so it is
    # glossed wherever it appears -- and only there.
    if any(r.batch_size and r.status != "ok" for r in rows):
        footer = f"{footer} · \u00d7 out of memory"
    figure.text(
        left, FOOTER_Y, footer, family="monospace", fontsize=6.5, color=theme.faint
    )
    return figure


def plot(
    run_dir: Path,
    out_dir: Path,
    themes: tuple[str, ...] = ("light", "dark"),
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Every panel of one sweep, once per theme, into `out_dir`.

    Both themes by default because the page has a toggle, and a figure that
    only exists in one of them is a white rectangle in a dark column half the
    time. The theme is in the file name rather than assumed from it: a reader
    swapping the pair needs the two names to differ by exactly that word.
    """
    import matplotlib

    matplotlib.use("Agg")  # a figure is written to a file, never to a window
    import matplotlib.pyplot as plt

    load_fonts()
    index = read_manifest(run_dir)
    records = read_records(run_dir)
    rows = build_rows(index, records)
    if not any(r.batch_size for r in rows):
        raise ValueError(f"{run_dir} holds no batched trial to plot")
    subtitle, footer = caption(run_dir, index, records)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in themes:
        theme = THEMES[name]
        with plt.rc_context(rc(theme)):
            for panel in PANELS:
                figure = draw(panel, rows, theme, subtitle, footer)
                for suffix in formats:
                    path = out_dir / f"{run_dir.name}-{panel.name}-{name}.{suffix}"
                    figure.savefig(path)
                    written.append(path)
                plt.close(figure)
    return written
