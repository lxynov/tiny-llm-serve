from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

QWEN3 = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def qwen3_path() -> Path:
    """Path to the Qwen3 checkpoint, fetching it on first use (~1.4GB).

    Downloading from a test is a deliberate trade: the alternative, resolving
    from the cache alone, turns a fresh clone into a green run with a third of
    the suite silently skipped. A cached checkpoint still resolves offline --
    `snapshot_download` falls back to the cache when it can't reach the hub --
    so only a machine that is both uncached and offline skips.
    """
    try:
        return Path(snapshot_download(QWEN3))
    except Exception:
        pytest.skip(f"{QWEN3} is not cached and could not be fetched")
