"""TurboQuant -- KV cache quantization for LLM inference.

Implements the TurboQuant algorithm (arxiv 2504.19874, ICLR 2026):
  Stage 1: Random orthogonal rotation + Lloyd-Max scalar quantization
  Stage 2: QJL (Quantized Johnson-Lindenstrauss) sign-bit correction

Usage::

    from turboquant import TurboQuantConfig, TurboQuantKVCache

    config = TurboQuantConfig(bits=3, head_dim=128)
    cache = TurboQuantKVCache(config)
    cache.append(layer_idx=0, keys=k, values=v)
    scores = cache.get_scores(layer_idx=0, query=q)
"""

from .quantizer import (
    TurboQuantConfig,
    QuantizedKV,
    quantize_kv,
    dequantize_kv,
    inner_product_corrected,
)
from .kv_cache import TurboQuantKVCache
from .codebooks import get_codebook

__version__ = "0.1.0"
__all__ = [
    "TurboQuantConfig",
    "QuantizedKV",
    "quantize_kv",
    "dequantize_kv",
    "inner_product_corrected",
    "TurboQuantKVCache",
    "get_codebook",
]
