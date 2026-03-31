"""TurboQuant two-stage KV cache quantizer.

Implements the TurboQuant algorithm (arxiv 2504.19874):
  Stage 1: Random orthogonal rotation + Lloyd-Max scalar quantization
  Stage 2: QJL (Quantized Johnson-Lindenstrauss) sign-bit correction
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .codebooks import get_codebook


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache quantization."""

    bits: int = 3
    """Quantization bit-width (2, 3, or 4)."""

    head_dim: int = 128
    """Dimension per attention head."""

    correction: bool = True
    """Enable QJL sign-bit correction (Stage 2)."""


@dataclass
class QuantizedKV:
    """Quantized key-value cache entry."""

    key_indices: np.ndarray
    """uint8 indices into codebook for keys."""

    value_indices: np.ndarray
    """uint8 indices into codebook for values."""

    key_rotation: np.ndarray
    """Orthogonal rotation matrix (head_dim x head_dim) used for keys."""

    value_rotation: np.ndarray
    """Orthogonal rotation matrix (head_dim x head_dim) used for values."""

    config: TurboQuantConfig
    """Config used during quantization."""

    key_residual_signs: Optional[np.ndarray] = None
    """Packed sign bits of projected key residual (Stage 2)."""

    value_residual_signs: Optional[np.ndarray] = None
    """Packed sign bits of projected value residual (Stage 2)."""

    key_projection: Optional[np.ndarray] = None
    """Random Gaussian projection matrix for keys (Stage 2)."""

    value_projection: Optional[np.ndarray] = None
    """Random Gaussian projection matrix for values (Stage 2)."""

    original_shape: tuple = ()
    """Original shape of the key/value tensors."""


def _generate_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random orthogonal matrix via QR decomposition of Gaussian matrix.

    Args:
        dim: Matrix dimension.
        rng: NumPy random generator.

    Returns:
        Orthogonal matrix of shape (dim, dim).
    """
    G = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(G)
    # Ensure proper rotation (det = +1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def _lloyd_max_quantize(
    x: np.ndarray, centroids: np.ndarray, boundaries: np.ndarray
) -> np.ndarray:
    """Quantize values using precomputed Lloyd-Max codebook.

    Args:
        x: Array of values to quantize.
        centroids: Codebook centroid values.
        boundaries: Decision boundaries (length = len(centroids) + 1).

    Returns:
        uint8 indices into the centroid array.
    """
    indices = np.digitize(x, boundaries[1:-1])
    return indices.astype(np.uint8)


def _lloyd_max_dequantize(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Dequantize indices back to centroid values.

    Args:
        indices: uint8 codebook indices.
        centroids: Codebook centroid values.

    Returns:
        Reconstructed values from centroids.
    """
    return centroids[indices]


def quantize_kv(
    keys: np.ndarray,
    values: np.ndarray,
    config: Optional[TurboQuantConfig] = None,
) -> QuantizedKV:
    """Quantize key and value tensors using TurboQuant.

    Stage 1: Random orthogonal rotation + Lloyd-Max scalar quantization.
    Stage 2 (if config.correction): QJL correction -- project residual, store sign bits.

    Args:
        keys: Array of shape (seq_len, head_dim) or (num_heads, seq_len, head_dim).
        values: Same shape as keys.
        config: TurboQuantConfig (uses defaults if None).

    Returns:
        QuantizedKV object containing compressed representation.
    """
    if config is None:
        config = TurboQuantConfig()

    rng = np.random.default_rng()
    centroids, boundaries = get_codebook(config.bits)
    sigma = 1.0 / np.sqrt(config.head_dim)
    scaled_centroids = centroids * sigma
    scaled_boundaries = boundaries * sigma

    original_shape = keys.shape

    # Stage 1: Rotate then quantize
    R_k = _generate_rotation(config.head_dim, rng)
    R_v = _generate_rotation(config.head_dim, rng)

    if keys.ndim == 2:
        rotated_k = keys @ R_k.T
        rotated_v = values @ R_v.T
    else:  # 3D: (num_heads, seq_len, head_dim)
        rotated_k = np.einsum("...d,Dd->...D", keys, R_k)
        rotated_v = np.einsum("...d,Dd->...D", values, R_v)

    k_indices = _lloyd_max_quantize(rotated_k, scaled_centroids, scaled_boundaries)
    v_indices = _lloyd_max_quantize(rotated_v, scaled_centroids, scaled_boundaries)

    qkv = QuantizedKV(
        key_indices=k_indices,
        value_indices=v_indices,
        key_rotation=R_k,
        value_rotation=R_v,
        config=config,
        original_shape=original_shape,
    )

    # Stage 2: QJL correction
    if config.correction:
        k_reconstructed = _lloyd_max_dequantize(k_indices, scaled_centroids)
        v_reconstructed = _lloyd_max_dequantize(v_indices, scaled_centroids)
        k_residual = rotated_k - k_reconstructed
        v_residual = rotated_v - v_reconstructed

        proj_dim = config.head_dim
        P_k = rng.standard_normal((config.head_dim, proj_dim)) / np.sqrt(proj_dim)
        P_v = rng.standard_normal((config.head_dim, proj_dim)) / np.sqrt(proj_dim)

        if k_residual.ndim == 2:
            k_proj = k_residual @ P_k
            v_proj = v_residual @ P_v
        else:
            k_proj = np.einsum("...d,dp->...p", k_residual, P_k)
            v_proj = np.einsum("...d,dp->...p", v_residual, P_v)

        qkv.key_residual_signs = np.packbits(
            (k_proj >= 0).astype(np.uint8), axis=-1
        )
        qkv.value_residual_signs = np.packbits(
            (v_proj >= 0).astype(np.uint8), axis=-1
        )
        qkv.key_projection = P_k
        qkv.value_projection = P_v

    return qkv


def dequantize_kv(qkv: QuantizedKV) -> tuple:
    """Dequantize back to approximate key/value tensors (lossy).

    Only uses Stage 1 reconstruction (rotation + codebook lookup).

    Args:
        qkv: QuantizedKV object.

    Returns:
        (keys, values) tuple of numpy arrays.
    """
    config = qkv.config
    centroids, _ = get_codebook(config.bits)
    sigma = 1.0 / np.sqrt(config.head_dim)
    scaled_centroids = centroids * sigma

    k_rotated = _lloyd_max_dequantize(qkv.key_indices, scaled_centroids)
    v_rotated = _lloyd_max_dequantize(qkv.value_indices, scaled_centroids)

    R_k_inv = qkv.key_rotation.T  # orthogonal: inverse = transpose
    R_v_inv = qkv.value_rotation.T

    if k_rotated.ndim == 2:
        keys = k_rotated @ R_k_inv.T
        values = v_rotated @ R_v_inv.T
    else:
        keys = np.einsum("...d,Dd->...D", k_rotated, R_k_inv)
        values = np.einsum("...d,Dd->...D", v_rotated, R_v_inv)

    return keys, values


def inner_product_corrected(query: np.ndarray, qkv: QuantizedKV) -> np.ndarray:
    """Compute corrected attention scores: query @ keys^T with QJL bias correction.

    This is the main use case -- avoids full dequantization by computing
    the inner product directly from the quantized representation.

    Args:
        query: Array of shape (head_dim,) or (num_queries, head_dim).
        qkv: QuantizedKV object.

    Returns:
        Attention scores (query @ keys^T approximation).
    """
    config = qkv.config
    centroids, _ = get_codebook(config.bits)
    sigma = 1.0 / np.sqrt(config.head_dim)
    scaled_centroids = centroids * sigma

    # Rotate query into same space as quantized keys
    if query.ndim == 1:
        q_rotated = query @ qkv.key_rotation.T
    else:
        q_rotated = np.einsum("...d,Dd->...D", query, qkv.key_rotation)

    # Stage 1: approximate inner product via codebook lookup
    k_reconstructed = _lloyd_max_dequantize(qkv.key_indices, scaled_centroids)

    if q_rotated.ndim == 1:
        scores = k_reconstructed @ q_rotated  # (seq_len,)
    else:
        scores = np.einsum("...d,qd->...q", k_reconstructed, q_rotated)

    # Stage 2: QJL correction
    if config.correction and qkv.key_residual_signs is not None:
        signs_unpacked = np.unpackbits(qkv.key_residual_signs, axis=-1)
        proj_dim = qkv.key_projection.shape[1]
        signs_unpacked = signs_unpacked[..., :proj_dim]
        signs = 2.0 * signs_unpacked.astype(np.float64) - 1.0

        if q_rotated.ndim == 1:
            q_proj = q_rotated @ qkv.key_projection
            correction = (signs * q_proj).sum(axis=-1) / proj_dim
        else:
            q_proj = np.einsum("...d,dp->...p", q_rotated, qkv.key_projection)
            correction = np.einsum("...p,...p->...", signs, q_proj) / proj_dim

        scores = scores + correction

    return scores
