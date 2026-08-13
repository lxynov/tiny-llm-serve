import pytest

from benchmarks.hardware import peak_hbm_bytes_s


@pytest.mark.parametrize(
    "reported, expected",
    [
        # The SXM5 part reports its memory, not its form factor, so matching
        # on "H100 SXM" would silently miss the headline GPU.
        ("NVIDIA H100 80GB HBM3", 3.35e12),
        ("NVIDIA H100 PCIe", 2.0e12),
        ("NVIDIA H100 NVL", 3.9e12),
        ("NVIDIA A100-SXM4-80GB", 2.039e12),
        ("NVIDIA A40", 0.696e12),
    ],
)
def test_known_skus_resolve_to_their_peak(reported, expected):
    assert peak_hbm_bytes_s(reported) == expected


def test_unknown_device_has_no_peak():
    """No ceiling beats a wrong one: utilization stays null instead."""
    assert peak_hbm_bytes_s("NVIDIA GeForce GTX 1080") is None
    assert peak_hbm_bytes_s(None) is None
