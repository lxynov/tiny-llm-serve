from pathlib import Path

import pytest
import tinymodel
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


@pytest.fixture(scope="session")
def tiny_checkpoint_path(tmp_path_factory, qwen3_path) -> Path:
    """A loadable tiny random-weight checkpoint with the real Qwen3 tokenizer.

    max_position_embeddings is raised past the default so benchmark workloads
    (prompts and outputs up to 1024 tokens each) fit in the RoPE cache.
    """
    target = tmp_path_factory.mktemp("tiny-qwen3-checkpoint")
    return tinymodel.save_checkpoint(target, qwen3_path, max_position_embeddings=4096)
