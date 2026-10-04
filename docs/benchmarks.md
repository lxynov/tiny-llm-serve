# Benchmarks

Every performance claim here is a *delta* between two engine modes measured
under identical conditions, so the harness is fixed before the modes it
compares. What each number in a record or report means is in
[metrics.md](metrics.md).

- [Layout](#layout)
- [Rules](#rules)
- [Throughput](#throughput)
- [Sweeps](#sweeps)
- [Decode steps](#decode-steps)
- [Reports](#reports)
- [Curves](#curves)

## Layout

`benchmarks/` holds **one package per benchmark** — each answers a different
question, and little about how one measures carries over to another — over a
shared floor that all of them stand on:

```
benchmarks/
├── records.py    what every record carries: provenance, environment, commit
├── roofline.py   the bytes decode has to move, whatever is timing them
├── hardware.py   peak device bandwidth, so a measured rate reads as a fraction
├── assets/       the vendored typefaces a published figure is drawn in
├── throughput/   a full workload pass: bench, sweep, report, figures, workloads
├── decode_step/  one engine step at a chosen (batch, kv_len), in isolation
└── results/
    ├── throughput/   one folder per sweep
    └── decode-step/  one folder per day, commit, chip, model and dtype; a record per run
```

Records land in `results/<benchmark>/`, so which question a record answers is
readable from where it sits rather than from what is inside it — and a new
benchmark reaches for the shared floor rather than for another benchmark's
runner.

## Rules

1. **One variable per comparison.** Same model, dtype, device, commit,
   workload, seed, and sampling (greedy) — only the engine mode changes.
   Never compare runs from different machines or days without a same-day
   rerun of the baseline.
2. **Pre-tokenized prompts.** The engine consumes token ids directly, so
   tokenizer time never pollutes engine numbers.
3. **Bracket the timed pass.** A device barrier
   (`torch.{cuda,mps}.synchronize`) sits on either side of the timer:
   accelerator work is enqueued asynchronously, so an unbracketed timer
   measures kernel *submission*, not execution.
4. **Results are data.** Every run writes one JSON record under
   `benchmarks/results/<benchmark>/`; headline numbers must be reproducible from a
   committed record. Each record carries the git commit and whether the tree
   was dirty, plus the environment that produced it — OS, Python, torch, CPU
   model, core count, torch thread count, and GPU name, memory, and CUDA
   version on a GPU box — so rule 1 can be checked after the fact instead of
   taken on trust. A record with `"dirty": true` measures code that no commit
   describes; treat it as a scratch run. Dirty counts tracked modifications
   only — records are written *into* this repository, so counting untracked
   files would mark every run after the first for the record its predecessor
   left behind, and a flag that fires on every run says nothing about any of
   them. Sweeps and decode-step runs name their folders for the day, commit,
   chip, model and dtype; a lone throughput run writes
   `<mode>-<workload>-bs<NNN>.json` straight into `--output-dir`, so keep
   separate machines in separate `--output-dir`s.

## Throughput

```bash
# Run a seeded workload and write a JSON record to benchmarks/results/throughput/
uv run python -m benchmarks.throughput.bench --workload mixed-out --num-requests 8

# The batch=1 floor every batching win is read against
uv run python -m benchmarks.throughput.bench --workload mixed-out --num-requests 8 \
  --batch-size 1
```

`--mode` names the engine path a record came from and takes only `static`
today. `--batch-size` sets how many sequences decode together, `--dtype`
defaults to `bfloat16` (the checkpoints' own precision, and the only one the
fused attention kernels accept), and `--device` overrides auto-selection.

A run is successive full batches of `--batch-size`, one wave at a time, and
every request is drained, so runs at different batch sizes stay comparable.
Each wave reserves `batch × (longest prompt + longest output)` KV tokens —
every slot's preallocated worst case, which is what `kv_efficiency` weighs
against what the wave actually stored. At `--batch-size 1` a slot holds exactly
one sequence, so the ratio is 1.0 and the run is the sequential floor.

`--num-requests` must be a multiple of `--batch-size`. A short final wave would
otherwise be recorded under the full batch size, so a batch-size sweep would
compare trials that never ran the batch size they are plotted against;
requiring whole waves also keeps the workload byte-identical across the sweep,
which is what [rule 1](#rules) needs.

The workloads are described in [metrics.md](metrics.md#workloads). Continuous
batching lands next as a second mode, reusing these workloads and this record
format.

## Sweeps

A batch-size sweep is a grid of runs, not one run, so
`benchmarks/throughput/sweep.py` drives it:

```bash
uv run python -m benchmarks.throughput.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
  --num-requests 128 --batch-sizes 1,2,4,8,16,32,64,128
```

Each trial — one workload at one batch size — runs in **its own process**,
because the caching allocator's pool outlives a run and trials sharing a
process would report each other's peak memory, the metric that decides where
the sweep ends. A trial that dies does not stop the sweep: running out of
memory at a large batch size is *where capacity ran out*, which is a result,
and aborting there would discard every trial still queued behind it.

Batch sizes run ascending, so a sweep that dies at its memory ceiling has
already banked the rest of the curve.

A trial that runs out of memory writes a record of its own — `"status": "oom"`,
no metrics — and the sweep then stops raising the batch size for that workload
instead of attempting the rest: memory demand only rises with the batch size,
so every size above one that did not fit would spend a model load apiece to
confirm what the first failure already established. Ceilings are per workload,
so the other curves keep going. Only an allocation failure counts as one: any
other crash is a bug, and reading it as capacity would silently truncate a
curve.

Every trial of one sweep writes into one folder under `--output-dir`, named
for when the sweep started, the code it measures, and what it held fixed:

```
benchmarks/results/throughput/20260816-faa6722-cuda-h100-80gb-hbm3-qwen3-8b-bfloat16-n128/
├── static-mixed-out-bs001.json
├── static-mixed-out-bs002.json
├── ...
└── sweep.json                   # the grid, and how each trial ended
```

The folder name reads date, commit, device and chip, model, dtype, request
count. The date is the UTC day the sweep started, so `--output-dir` lists as a
history and a sweep that runs past midnight keeps one name. The commit is the
short hash, suffixed `-dirty` when the tree carries uncommitted changes, since
a dirty run is not the commit it names. The chip is there because the kind of
device says almost nothing: `cuda` covers an RTX 4090 and an H200, a 5× spread
in the bandwidth decode is bound by.

Inside, a trial is named for its mode, workload and batch size, zero-padded so
a listing sorts into the ladder the sweep climbed. That is everything that
varies between one folder's trials, so a second run of the same grid would
overwrite the first one record at a time; a sweep therefore refuses to write
into a folder that already exists, and rerunning one means moving the old folder
aside. `sweep.json` is rewritten after every trial, so a sweep that dies part way
through still leaves an index, and it carries the one outcome with no record of
its own: the batch sizes the ceiling ruled out before they were ever attempted.

## Decode steps

A sweep answers *how fast is a pass*; almost everything left to optimize moves
one number inside it — `s_per_decode_step` at a given batch size and KV length.
Draining a workload to read that number is waste, so
`benchmarks/decode_step/bench.py` buys it directly:

```bash
uv run python -m benchmarks.decode_step.bench --model Qwen/Qwen3-8B \
  --cells 1x576,8x576,64x576,256x576,64x2048 --warmup 10 --steps 30 --repeats 3
```

A cell is `BATCHxKV_LEN`. Each one runs one real prefill — so the state a step
reads is the state the engine would have built, down to the logits it samples
from — then `--warmup` steps that are thrown away, then `--repeats` windows of
`--steps` timed steps. The prefill stops short of `kv_len`, so the timed steps
center on it — the middle one attends exactly `kv_len` tokens — and the bytes
charged to a window follow the cache as it grows a token per step. `kv_len 576`
is the mean cached length of `uniform-512x128` over its run, so a cell at that
length lands next to the baseline sweep.

What is timed is the **engine** step — sampler, host bookkeeping and forward
pass — not the model forward, because at small batch sizes the step's cost is
decided by the loop around the forward rather than by the forward.

**One clock per window**, closed after a barrier, so it measures the steps the
device actually ran. A second clock closed before the barrier would say only how
long the host loop ran — and a step that reads anything back, as a serving step
must every step, blocks the host until the device catches up, so that clock
tracks this one whichever side is slow. Whether a step is paying for overhead or
for bytes is `mbu`'s question; where its time goes is the profiler's. The
columns it prints are described in [metrics.md](metrics.md#decode-step-columns).

Records land in `benchmarks/results/decode-step/`, one per invocation with
every cell in it, carrying the same commit/dirty/environment block a throughput
record does. Their folder is named the way a sweep's is — date, commit, device
and chip, model, dtype — and each record for the UTC time it started:

```
benchmarks/results/decode-step/20260819-faa6722-cuda-h100-80gb-hbm3-qwen3-8b-bfloat16/
├── 200125.json
└── 213410.json
```

Runs that can be read against each other sit together, reruns side by side in
the order they ran, and a run on another card or of other code lands somewhere
else. A cell that runs out of memory records `"status": "oom"` and the
run continues: the cells queued behind it have not been asked yet. Cells share
one process, unlike a sweep's trials — reloading an 8B checkpoint per cell
would cost more than the measurement — which is why allocated bytes are reset
per cell and *reserved* bytes, the metric that decides a sweep's ceiling, are
not reported here at all.

Each slot's KV past the prompt is filled with random values rather than left
zeroed. A step that reads past its own length — an off-by-one in a mask, a
`seqused_k` counting the wrong thing — adds a row of zeros to a softmax and
drifts quietly; against random values the same bug is loud enough for the
correctness gate to catch.

## Reports

A sweep folder is written to be read by a machine: one JSON record per trial
next to an index over them. `benchmarks/throughput/report.py` turns one back into
something a person can read.

```bash
uv run python -m benchmarks.throughput.report \
  benchmarks/results/throughput/20260819-faa6722-cuda-h100-80gb-hbm3-qwen3-8b-bfloat16-n512
```

It prints a table per workload — every trial in the grid, including the ones
with no record behind them — and four summary lines: where throughput peaked
and how far short of linear it fell, what the last doubling bought, the band of
memory bandwidth decode actually reached, and the batch size the ceiling ruled
out. `--markdown` emits the tables for pasting; `--plot DIR` writes throughput,
decode-step and occupancy curves there (the only thing here that needs
matplotlib, imported lazily so a report to a terminal does not pay for it). The
columns only the report has are described in
[metrics.md](metrics.md#report-columns).

The report also checks the one thing [rule 1](#rules) asks of a sweep that no
single record can confirm and the folder's name does not promise: that every
trial ran the same commit, dtype and GPU, and that none of them measured a dirty
tree. The name records the commit as it stood when the sweep started, but each
trial is a fresh process that imports the engine and reads the commit for
itself — so code committed or edited while a sweep runs reaches only the trials
after it, and a curve bends somewhere the engine did not change. Disagreements
print as `WARNING` lines above the tables rather than failing: the records are
still the results, they just are not a curve.

Nothing in the report re-derives a workload to recover its lengths. A name and a
seed reproduce a workload only while the generator behind the name is unchanged,
and a report that quietly regenerated against today's version would describe a
grid that was never run.

## Curves

`benchmarks/throughput/figures.py` draws them in the notebook's own palette and
type, so a figure dropped into a post reads as part of the page rather than as a
screenshot from somewhere else: ink on paper, one hairline rule, no accent
colour, and workloads told apart by lightness, marker shape and a label at the
end of each line rather than by hue. Where a sweep hit a ceiling the curve
carries a dotted stub out to the batch size that failed and ends in a cross;
on the throughput panel a slope-one guide from the smallest batch shows what
perfect scaling would have looked like, since on log-log axes falling short of
it is otherwise invisible.

Two consequences are worth knowing before publishing one:

**Every figure is written twice**, `-light` and `-dark`, because the site has a
theme toggle and a figure baked in one theme is a bright rectangle in a dark
column half the time. `--theme light` or `--theme dark` writes only one;
`--format svg` (or `both`) writes vector instead of, or alongside, the PNG. A
post asks for the pair by writing `{theme}` where the variant goes, and the
site serves whichever matches:

```markdown
![Output tokens per second](/images/20260819-faa6722-...-throughput-{theme}.png)
```

**The two typefaces are vendored**, under `benchmarks/assets/fonts/` — static
instances of the same Source Serif 4 and JetBrains Mono the site serves, with
their OFL licences beside them. They are pinned for the reason a commit is: so
a figure drawn on a laptop and one drawn in CI are the same figure. Delete them
and matplotlib falls back through the page's own CSS stack — Charter, then
Georgia — which changes the letterforms and nothing else.
