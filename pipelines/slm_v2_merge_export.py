#!/usr/bin/env python3
"""Merge the v2 LoRA into Qwen3.5-2B (text-only CausalLM) and export Q8_0 GGUF + Modelfile.

Mirrors ~/Projects/ga-chat-student/merge_lora.py + the Modelfile of qwen2b-ga-chat.
Does NOT run `ollama create` (prints the command).

usage: slm_v2_merge_export.py <adapter_dir> <out_root>
"""
import os
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-2B"
ADAPTER, ROOT = sys.argv[1], sys.argv[2]
MERGED = os.path.join(ROOT, "qwen2b-ga-chat-v2-merged")
GGUF = os.path.join(ROOT, "qwen2b-ga-chat-v2-q8_0.gguf")
# llama-cpp-latest rejects the qwen35 BPE pre-tokenizer hash; prism-llama.cpp's converter works
CONVERT = "/home/andus/Projects/prism-llama.cpp/convert_hf_to_gguf.py"
MODELFILE = os.path.join(ROOT, "Modelfile.qwen2b-ga-chat-v2")

if not os.path.exists(os.path.join(MERGED, "config.json")):
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model.save_pretrained(MERGED)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
    print("merged ->", MERGED, flush=True)

env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(CONVERT), "gguf-py"))
subprocess.run([sys.executable, CONVERT, MERGED, "--outtype", "q8_0", "--outfile", GGUF,
                "--model-name", "qwen2b-ga-chat-v2"], check=True, env=env)
with open(MODELFILE, "w") as f:
    f.write(f"FROM {GGUF}\nTEMPLATE {{{{ .Prompt }}}}\nRENDERER qwen3.5\nPARSER qwen3.5\n"
            "PARAMETER top_k 20\nPARAMETER top_p 0.95\nPARAMETER temperature 0.7\n")
print("gguf ->", GGUF)
print("modelfile ->", MODELFILE)
print(f"deploy: ollama create qwen2b-ga-chat-v2 -f {MODELFILE}")
