# llama.cpp Server for Generative Agents

Runs the Nemotron-Cascade-2 model locally via llama.cpp with CUDA acceleration on an RTX 3070 (8GB VRAM) + 32GB RAM system.

## Prerequisites

- CUDA Toolkit 12.x (`nvcc --version` to verify)
- CMake 3.21+ (`cmake --version`)
- Git
- ~20GB disk space for the GGUF model

## Building llama.cpp with CUDA

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

The server binary will be at `build/bin/llama-server`.

## Getting the GGUF Model

Download `nemotron-cascade-2-30b-a3b-q4_k_m.gguf` from HuggingFace:

```bash
# Via huggingface-cli
pip install huggingface_hub
huggingface-cli download <repo-id> nemotron-cascade-2-30b-a3b-q4_k_m.gguf --local-dir .
```

Or download directly from the model's HuggingFace page (Files tab).

## Running the Server

```bash
./build/bin/llama-server -m nemotron-cascade-2-30b-a3b-q4_k_m.gguf \
  --n-gpu-layers 18 \
  --ctx-size 8192 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --port 8080 \
  --host 0.0.0.0
```

Or use the bundled script:

```bash
chmod +x llama_cpp_server/start_server.sh
./llama_cpp_server/start_server.sh /path/to/nemotron-cascade-2-30b-a3b-q4_k_m.gguf
```

## Tuning `--n-gpu-layers`

Start at `18` for an 8GB VRAM card. Monitor VRAM usage:

```bash
watch -n 1 nvidia-smi
```

If VRAM headroom remains, increase `--n-gpu-layers` in increments of 4 and restart. Stop before you hit OOM. Layers that don't fit on GPU fall back to CPU (using system RAM), so the model will still run — just slower.

Typical safe ranges for RTX 3070 (8GB) with this model:
- `18` — conservative, ~7GB VRAM
- `24` — moderate, watch nvidia-smi
- `30+` — likely OOM, test carefully

## Note on Nemotron-Cascade / Mamba Layers

Nemotron-Cascade-2 is a hybrid **Mamba-2 + Transformer** architecture. Only the Transformer attention blocks have KV caches that benefit from `--cache-type-k/v q4_0` quantization (TurboQuant). Mamba SSM state is a fixed-size recurrent state — it is not part of the KV cache and is managed separately by llama.cpp's Mamba backend. The cache-type flags apply exclusively to Transformer attention layers and have no effect on Mamba layers.

## API Endpoint

The llama.cpp server exposes a **different API than Ollama**:

| | llama.cpp | Ollama |
|---|---|---|
| Endpoint | `POST /completion` | `POST /api/generate` |
| Response field | `{"content": "..."}` | `{"response": "..."}` |

Example:

```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "n_predict": 128}'
```

Response:

```json
{"content": "Hello! How can I help you today?", ...}
```

## Integration with Generative Agents

In `reverie/backend_server/persona/prompt_template/gpt_structure.py`, set:

```python
USE_LLAMA_CPP = True
LLAMA_CPP_URL = "http://localhost:8080"
```

This switches all LLM calls from the Ollama backend to the llama.cpp server.
