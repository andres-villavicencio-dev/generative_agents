#!/usr/bin/env bash
# Start llama.cpp server for generative agents simulation
# Usage: ./start_server.sh /path/to/model.gguf [gpu_layers] [port]

set -euo pipefail

GGUF_PATH="${1:?Usage: $0 <path-to-gguf> [n-gpu-layers] [port]}"
N_GPU_LAYERS="${2:-18}"
PORT="${3:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
LLAMA_SERVER="${LLAMA_SERVER:-./build/bin/llama-server}"

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "Error: GGUF file not found: $GGUF_PATH"
    exit 1
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "Error: llama-server not found at $LLAMA_SERVER"
    echo "Build llama.cpp first or set LLAMA_SERVER env var"
    exit 1
fi

echo "Starting llama.cpp server..."
echo "  Model: $GGUF_PATH"
echo "  GPU layers: $N_GPU_LAYERS"
echo "  Context: $CTX_SIZE"
echo "  Port: $PORT"

exec "$LLAMA_SERVER" \
    -m "$GGUF_PATH" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --port "$PORT" \
    --host 0.0.0.0
