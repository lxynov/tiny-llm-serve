---
name: figure-style
description: This skill should be used whenever a visual for the tiny-llm-serve logbook is being made or changed — a benchmark chart, a diagram for a post, an SVG, a figure dropped into docs/ or the site — or when someone asks how the figures are styled, why a chart looks the way it does, or how to add a new panel.
version: 1.0.0
---

# Figure Style

Figures here are published into <https://lxynov.github.io/tiny-llm-serve/>, whose
whole visual argument is ink on paper: an editorial serif, one hairline rule, no
accent colour. A chart in matplotlib's defaults reads as a screenshot from
somewhere else no matter how right its numbers are.

## The one rule

**Every published figure comes out of `benchmarks/figures.py`.** Do not draw one
in a notebook, a scratch script, or an ad-hoc `plt.plot`. A new figure is a new
`Panel` in `PANELS`:

```python
Panel(name="bandwidth", title="Decode bytes read per second", attr="bytes_s")
```

That inherits the palette, the type, the layout, the ceiling marks, the
provenance lines and both themes. Nothing about the style has to be remembered,
which is the point of it living there.

If the thing you want cannot be expressed as a `Panel` over a `Row` attribute,
extend `figures.py` — a second drawing path is how a style stops being one.

## Never restate the palette

The tokens live in exactly one place: `THEMES` in `benchmarks/figures.py`, copied
from the site's stylesheet with the URL and the date beside them. Read them from
there. Do not paste hex values into a post, a diagram, another module, or this
file — a second copy is a copy that will not be updated, and the drift is
invisible until two figures on one page disagree.

When the site's design changes, re-read its `:root` block, update `THEMES`, bump
the as-of date in the comment, and re-render. That is the entire maintenance
story.

## Rules that outlive the palette

These hold whatever the tokens are:

- **Ink, not hue.** The site has no accent colour. Series are separated by steps
  of `ink_ramp`, and every step is doubled by a marker shape and a label at the
  end of its own line. Identity is never colour-alone because it is never colour.
  Past six series, split into one figure each rather than inventing steps.
- **Mono for every digit.** Tick labels, metadata, footers, axis labels — the
  page sets numbers in JetBrains Mono, so figures do too. Serif carries the
  title and nothing else.
- **The title states the measure**, spelled out, so there is no rotated y-axis
  label. One panel per file; a post drops in the one it is arguing about.
- **Labels at the end of the line, not a legend box.** `place_labels` nudges
  them apart when curves finish close together.
- **Hairline y-grid, bottom spine only.** No frame, no top/right/left spine, no
  fill behind the plot — the figure is the same paper as the page.
- **Provenance travels with the figure**: conditions under the title, sweep
  folder and commit in the footer. A figure outlives the folder it came from.
- **Both themes, always.** `plot` writes `-light` and `-dark`; the page swaps
  them on the theme toggle's own attribute (the rule is in the README). Never
  publish one theme's figure alone.
- **Never a dual y-axis.** Two measures of different scale are two panels.

## Visuals that are not charts

A diagram, an inline SVG, a table: same page, same rules. Take the tokens from
`THEMES`, keep to ink and rules with no accent colour, set every number in mono,
and produce both themes — inline SVG in the page can use the site's own custom
properties directly and skip the pair.

## What a finished figure looks like

```bash
uv run python -m benchmarks.report <sweep-folder> --plot docs/figures
```

Six files for three panels. `--theme light|dark` writes one; `--format svg|both`
writes vector. Open the PNGs and look at them before calling it done — the
things that go wrong are label collisions, a curve hidden under a lighter one,
and a band of empty paper above the top gridline, and none of those show up in
a test.

## Fonts

`benchmarks/assets/fonts/` holds static instances of the same Source Serif 4 and
JetBrains Mono the site serves, with their OFL licences. They are pinned so a
figure drawn on a laptop and one drawn in CI are the same figure. If they are
missing, matplotlib walks the page's own CSS fallback stack and the letterforms
change — check the fonts before concluding a figure regressed.
