"""
TurboQuant Benchmark — measure compression ratio and attention score accuracy.

Run with: python -m turboquant.benchmark
"""
import time
import numpy as np

def benchmark():
    from turboquant import TurboQuantConfig, TurboQuantKVCache, quantize_kv, dequantize_kv, inner_product_corrected

    head_dim = 128
    seq_lengths = [1024, 4096, 8192]
    bit_widths = [2, 3, 4]

    print("=" * 70)
    print("TurboQuant Benchmark")
    print("=" * 70)
    print(f"Head dimension: {head_dim}")
    print()

    # Table header
    print(f"{'Bits':>4} {'SeqLen':>7} {'Orig MB':>8} {'Comp MB':>8} {'Ratio':>6} {'MSE(scores)':>12} {'Time(ms)':>9}")
    print("-" * 70)

    rng = np.random.default_rng(42)

    for bits in bit_widths:
        for seq_len in seq_lengths:
            # Generate synthetic KV vectors ~ N(0, 1/sqrt(head_dim))
            sigma = 1.0 / np.sqrt(head_dim)
            keys = rng.normal(0, sigma, (seq_len, head_dim))
            values = rng.normal(0, sigma, (seq_len, head_dim))
            query = rng.normal(0, sigma, (head_dim,))

            config = TurboQuantConfig(bits=bits, head_dim=head_dim, correction=True)

            # Time the quantization
            t0 = time.perf_counter()
            qkv = quantize_kv(keys, values, config)
            t_quant = (time.perf_counter() - t0) * 1000

            # Compute true attention scores
            true_scores = keys @ query

            # Compute corrected scores
            corrected_scores = inner_product_corrected(query, qkv)

            # MSE
            mse = np.mean((true_scores - corrected_scores) ** 2)

            # Memory stats via cache
            cache = TurboQuantKVCache(config)
            cache.append(0, keys, values)
            stats = cache.stats()

            print(f"{bits:>4} {seq_len:>7} {stats['original_mb']:>8.3f} {stats['compressed_mb']:>8.3f} {stats['ratio']:>6.1f}x {mse:>12.2e} {t_quant:>8.1f}")

    print()
    print("=" * 70)

    # Compare with and without QJL correction
    print("\nQJL Correction Impact (3-bit, seq_len=4096):")
    print(f"{'Correction':>12} {'MSE':>12}")
    print("-" * 30)

    sigma = 1.0 / np.sqrt(head_dim)
    keys = rng.normal(0, sigma, (4096, head_dim))
    values = rng.normal(0, sigma, (4096, head_dim))
    query = rng.normal(0, sigma, (head_dim,))
    true_scores = keys @ query

    for correction in [False, True]:
        config = TurboQuantConfig(bits=3, head_dim=head_dim, correction=correction)
        qkv = quantize_kv(keys, values, config)
        corrected_scores = inner_product_corrected(query, qkv)
        mse = np.mean((true_scores - corrected_scores) ** 2)
        print(f"{'Yes' if correction else 'No':>12} {mse:>12.2e}")

    print()

if __name__ == "__main__":
    benchmark()
