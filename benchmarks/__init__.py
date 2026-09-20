"""Benchmark harnesses, and what they record.

One package per benchmark, because each answers a different question and
nothing about the way one measures carries over to another:

    throughput/   a full workload pass: how many tokens per second, end to end

Anything at this level is shared by all of them, and is the only thing a new
benchmark should be reaching for:

    records.py    what every record carries -- provenance, environment, commit
    roofline.py   the bytes decode has to move, whatever is timing them
    hardware.py   peak device bandwidth, so a measured rate reads as a fraction
    assets/       the vendored typefaces a published figure is drawn in

Records land in `results/<benchmark>/`, so which question a record answers is
readable from where it sits rather than from what is inside it.
"""
