"""Peak hardware rates, so a measured rate can be read as a fraction of one.

A throughput number alone cannot say whether it is good. Divided by what the
device can physically do, it becomes a utilization -- and a low utilization
says how much room is left and roughly where to look for it.
"""

# Peak HBM bandwidth in bytes/second, keyed by a substring of the name torch
# reports for the SKU. Names are matched, not equated, because the same chip
# ships under several strings ("NVIDIA H100 80GB HBM3" is the SXM5 part); more
# specific keys come first, since the first match wins.
#
# Every figure is the vendor's published peak, in the decimal GB/s (10^9 bytes)
# NVIDIA quotes -- the same unit decode_bytes_read counts in. Sources, each
# checked against the spec table on the page as of 2026-08:
#   H100 SXM, H100 NVL  https://www.nvidia.com/en-us/data-center/h100/
#   H200 SXM, H200 NVL  https://www.nvidia.com/en-us/data-center/h200/
#   A100 80GB PCIe/SXM  https://www.nvidia.com/en-us/data-center/a100/
#   L40S                https://www.nvidia.com/en-us/data-center/l40s/
#   A40                 https://www.nvidia.com/en-us/data-center/a40/
# Three parts are not on a current product page, and need an older document:
#   H100 PCIe, 80GB HBM2e at 2,000 GB/s -- the product page now lists only SXM
#   and NVL, so the figure comes from the Nov 2022 product brief:
#   https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf
#   A100 40GB at 1,555 GB/s -- the June 2021 datasheet revision, whose table
#   still carries the 40GB columns the current page dropped:
#   https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
#   RTX 4090 -- NVIDIA publishes the parts but not the product, so 1,008 GB/s
#   is derived: 384-bit x 21 Gbps GDDR6X / 8 bits per byte.
#   https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
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
