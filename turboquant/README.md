# TurboQuant

KV cache quantization for LLM inference, implementing the TurboQuant algorithm (arxiv 2504.19874, ICLR 2026).

## Overview

TurboQuant reduces KV cache memory via two stages:
1. **Stage 1**: Random orthogonal rotation (QR decomposition) + Lloyd-Max scalar quantization at configurable bit-width (2, 3, or 4 bits)
2. **Stage 2**: QJL (Quantized Johnson-Lindenstrauss) correction — random Gaussian projection of quantization residual with sign-bit storage for unbiased inner product estimation

## Installation

No installation needed — just ensure numpy is available:
```
pip install numpy
```

## Usage

### Basic quantization
```python
import numpy as np
from turboquant import TurboQuantConfig, quantize_kv, dequantize_kv, inner_product_corrected

config = TurboQuantConfig(bits=3, head_dim=128, correction=True)
keys = np.random.randn(1024, 128) / np.sqrt(128)
values = np.random.randn(1024, 128) / np.sqrt(128)
query = np.random.randn(128) / np.sqrt(128)

qkv = quantize_kv(keys, values, config)
scores = inner_product_corrected(query, qkv)
keys_approx, values_approx = dequantize_kv(qkv)
```

### KV Cache
```python
from turboquant import TurboQuantConfig, TurboQuantKVCache

cache = TurboQuantKVCache(TurboQuantConfig(bits=3))
cache.append(layer_idx=0, keys=k, values=v)
scores = cache.get_scores(layer_idx=0, query=q)
print(cache.stats())
```

### Benchmark
```
python -m turboquant.benchmark
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| bits | 3 | Quantization bit-width (2, 3, or 4) |
| head_dim | 128 | Attention head dimension |
| correction | True | Enable QJL sign-bit correction |

## Note on llama.cpp Integration

This is a standalone Python research implementation. For production KV cache quantization with llama.cpp, use the built-in `--cache-type-k` and `--cache-type-v` flags. TurboQuant applies only to Transformer attention KV cache (not Mamba/SSM state).

## Dependencies

- Python 3.8+
- NumPy
