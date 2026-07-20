"""Run a test across the compute backends available on the current machine.

`@backends("all")` parametrizes a test's `device` argument over every backend
torch can use here: cpu is always present, with cuda and/or mps added when
available. Pass an explicit subset (e.g. `@backends("cpu", "mps")`) to restrict;
requested backends that aren't available are dropped, so the same test runs on
two devices on a CUDA box or an Apple-silicon Mac and on cpu alone in CI.
"""

import pytest
import torch

KNOWN_BACKENDS = ("cpu", "cuda", "mps")


def available_backends() -> list[str]:
    """Backends usable on this machine, cpu first."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def backends(*names: str):
    """Parametrize `device` over the requested backends that are available."""
    if not names:
        raise ValueError("backends() needs at least one backend name or 'all'")
    requested = KNOWN_BACKENDS if "all" in names else names
    unknown = set(requested) - set(KNOWN_BACKENDS)
    if unknown:
        raise ValueError(
            f"unknown backend(s) {sorted(unknown)}; known: {KNOWN_BACKENDS}"
        )
    available = available_backends()
    selected = [d for d in requested if d in available]
    return pytest.mark.parametrize("device", selected)
