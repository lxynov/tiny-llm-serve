"""Peak hardware rates, so a measured rate can be read as a fraction of one.

A throughput number alone cannot say whether it is good. Divided by what the
device can physically do, it becomes a utilization -- and a low utilization
says how much room is left and roughly where to look for it.
"""

# Peak HBM bandwidth in bytes/second, keyed by a substring of the name torch
# reports for the SKU. Names are matched, not equated, because the same chip
# ships under several strings ("NVIDIA H100 80GB HBM3" is the SXM5 part); more
# specific keys come first, since the first match wins.
PEAK_HBM_BYTES_S = {
    "H100 NVL": 3.9e12,
    "H100 PCIe": 2.0e12,
    "H100 80GB HBM3": 3.35e12,  # SXM5
    "H200": 4.8e12,
    "A100-SXM4-80GB": 2.039e12,
    "A100 80GB PCIe": 1.935e12,
    "A100-SXM4-40GB": 1.555e12,
    "L40S": 0.864e12,
    "A40": 0.696e12,
    "RTX 4090": 1.008e12,
}


def peak_hbm_bytes_s(device_name: str | None) -> float | None:
    """Peak memory bandwidth for `device_name`, or None if it is unlisted.

    None rather than an estimate: a utilization computed against the wrong
    ceiling is a confidently wrong number, and worse than a missing one. Add
    the SKU to the table instead -- vendor spec sheets publish the figure.
    """
    if device_name is None:
        return None
    for name, peak in PEAK_HBM_BYTES_S.items():
        if name in device_name:
            return peak
    return None
