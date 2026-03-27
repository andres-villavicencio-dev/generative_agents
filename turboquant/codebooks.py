"""Precomputed Lloyd-Max codebooks for Gaussian quantization.

Codebooks are optimized for N(0,1) distribution. The quantizer scales
them by sigma = 1/sqrt(head_dim) to match the expected KV cache
distribution N(0, 1/head_dim).

Supported bit widths: 2, 3, 4.
"""

import numpy as np

CODEBOOKS = {
    2: {
        "centroids": [-1.5104, -0.4528, 0.4528, 1.5104],
        "boundaries": [float("-inf"), -0.9816, 0.0, 0.9816, float("inf")],
    },
    3: {
        "centroids": [
            -2.1520, -1.3440, -0.7560, -0.2451,
            0.2451, 0.7560, 1.3440, 2.1520,
        ],
        "boundaries": [
            float("-inf"), -1.7480, -1.0500, -0.5006,
            0.0,
            0.5006, 1.0500, 1.7480, float("inf"),
        ],
    },
    4: {
        "centroids": [
            -2.7326, -2.0690, -1.6180, -1.2562,
            -0.9424, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9424,
            1.2562, 1.6180, 2.0690, 2.7326,
        ],
        "boundaries": [
            float("-inf"), -2.4008, -1.8435, -1.4370,
            -1.0993, -0.7996, -0.5224, -0.2582,
            0.0,
            0.2582, 0.5224, 0.7996, 1.0993,
            1.4370, 1.8435, 2.4008, float("inf"),
        ],
    },
}


def get_codebook(bits: int) -> tuple:
    """Return (centroids, boundaries) numpy arrays for given bit width.

    Args:
        bits: Quantization bit width (2, 3, or 4).

    Returns:
        Tuple of (centroids, boundaries) as float64 numpy arrays.

    Raises:
        ValueError: If bit width is not supported.
    """
    if bits not in CODEBOOKS:
        raise ValueError(
            f"Unsupported bit width {bits}. Choose from {sorted(CODEBOOKS.keys())}."
        )
    entry = CODEBOOKS[bits]
    return np.array(entry["centroids"], dtype=np.float64), np.array(
        entry["boundaries"], dtype=np.float64
    )
