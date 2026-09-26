"""What every benchmark record carries, and where the records land.

Rule 4 of the harness is that results are data: every run writes one JSON
record, and it carries enough of its own provenance -- commit, dirty tree,
machine, device -- that rule 1, one variable per comparison, can be checked
after the fact instead of taken on trust. None of that depends on what is being
measured, so it lives here rather than inside whichever benchmark needed it
first.
"""

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from tiny_llm_serve.models import loader

# Each benchmark files its records under its own folder in here. They answer
# different questions, and which question a record answers should be readable
# from where it sits rather than from what is inside it.
RESULTS_DIR = Path(__file__).parent / "results"
# Exit code for a run that recorded its own out-of-memory ceiling; see the
# bottom of a bench module. A ceiling is neither a success nor a crash, and a
# driver has to tell all three apart.
OOM_EXIT = 2
# The dtypes a benchmark runs in, keyed by the string its record carries.
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def git_state(repo: Path = Path(__file__).parent) -> tuple[str | None, bool | None]:
    """The commit and whether the tree carried uncommitted changes to it.

    A commit alone does not identify the code that ran: two runs from the
    same commit with different uncommitted edits are indistinguishable
    otherwise, which is exactly the claim a record is supposed to settle.

    Untracked files do not count, because this harness writes its records
    *into* the repository: counting them would mark every run after the first
    dirty for the file its predecessor left behind, and a flag that fires on
    every run says nothing about any of them.
    """
    git = ["git", "-C", str(repo)]
    try:
        commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True)
        status = subprocess.check_output(
            git + ["status", "--porcelain", "--untracked-files=no"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit.strip(), bool(status.strip())


def cpu_name() -> str | None:
    """The chip model, which `platform.processor()` is uselessly vague about
    ("arm" on macOS, "x86_64" on Linux) while it dominates any CPU run."""
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or None


def environment(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        # Torch sizes its thread pool from the machine, not from anything this
        # harness passes it, and CPU throughput scales with it.
        "torch_threads": torch.get_num_threads(),
    }
    if loader.is_cuda(device):
        properties = torch.cuda.get_device_properties(device)
        info |= {
            "gpu": properties.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_bytes": properties.total_memory,
            "cuda": torch.version.cuda,
        }
    return info


def provenance(model: str, device: str, name: str) -> dict:
    """Who ran what, where, and from which commit.

    `name` is the record's file name, and only has to tell it apart from the
    other runs in its folder -- a throughput trial's mode, workload and batch
    size, say. When, where and from which commit are the folder's to say, and
    are in the record below either way. Identical across benchmarks on purpose:
    two records that describe their conditions differently cannot be checked
    against each other by a reader or by a report.
    """
    now = datetime.now(timezone.utc)
    commit, dirty = git_state()
    return {
        "run_id": name,
        "date": now.isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model": model,
        "device": device,
        "environment": environment(device),
    }
