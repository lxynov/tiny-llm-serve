"""The throughput benchmark: a full workload pass through the engine.

bench.py      one trial -- a workload at one batch size, timed end to end
sweep.py      the grid driver: one trial per process, resumable
report.py     a sweep folder as tables
figures.py    the same folder as curves
workloads.py  the seeded request generators every trial draws from
"""
