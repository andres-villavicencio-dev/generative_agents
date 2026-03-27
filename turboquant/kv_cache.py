"""TurboQuant KV Cache -- drop-in replacement for list-of-tuples KV cache."""

import numpy as np

from .quantizer import TurboQuantConfig, QuantizedKV, quantize_kv, inner_product_corrected


class TurboQuantKVCache:
    """KV cache with TurboQuant compression.

    Drop-in replacement for a simple list-of-(k,v)-tuples cache.
    Stores quantized KV pairs per layer and provides corrected
    attention score computation.
    """

    def __init__(self, config: TurboQuantConfig | None = None):
        """Initialize cache with optional TurboQuantConfig.

        Args:
            config: Quantization configuration. Uses defaults if None.
        """
        self.config = config or TurboQuantConfig()
        self._layers: dict[int, list[QuantizedKV]] = {}
        self._original_bytes = 0
        self._compressed_bytes = 0

    def append(self, layer_idx: int, keys: np.ndarray, values: np.ndarray) -> None:
        """Append key-value pair for a given layer.

        Args:
            layer_idx: Transformer layer index.
            keys: Array of shape (seq_len, head_dim).
            values: Array of shape (seq_len, head_dim).
        """
        self._original_bytes += keys.nbytes + values.nbytes

        qkv = quantize_kv(keys, values, self.config)

        if layer_idx not in self._layers:
            self._layers[layer_idx] = []
        self._layers[layer_idx].append(qkv)

        compressed = qkv.key_indices.nbytes + qkv.value_indices.nbytes
        if qkv.key_residual_signs is not None:
            compressed += qkv.key_residual_signs.nbytes + qkv.value_residual_signs.nbytes
        self._compressed_bytes += compressed

    def get_scores(self, layer_idx: int, query: np.ndarray) -> np.ndarray:
        """Compute attention scores for query against cached keys at given layer.

        Uses corrected inner product for accuracy.

        Args:
            layer_idx: Transformer layer index.
            query: Array of shape (head_dim,) or (num_queries, head_dim).

        Returns:
            Array of attention scores.
        """
        if layer_idx not in self._layers:
            return np.array([])

        all_scores = []
        for qkv in self._layers[layer_idx]:
            scores = inner_product_corrected(query, qkv)
            all_scores.append(scores)

        return np.concatenate(all_scores, axis=0) if all_scores else np.array([])

    def stats(self) -> dict:
        """Return compression statistics.

        Returns:
            Dict with original_mb, compressed_mb, ratio, num_layers, total_entries.
        """
        orig_mb = self._original_bytes / (1024 * 1024)
        comp_mb = self._compressed_bytes / (1024 * 1024)
        ratio = orig_mb / comp_mb if comp_mb > 0 else float("inf")
        return {
            "original_mb": round(orig_mb, 3),
            "compressed_mb": round(comp_mb, 3),
            "ratio": round(ratio, 2),
            "num_layers": len(self._layers),
            "total_entries": sum(len(v) for v in self._layers.values()),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._layers.clear()
        self._original_bytes = 0
        self._compressed_bytes = 0
