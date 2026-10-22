#!/usr/bin/env python3
"""LoRA SFT of Qwen3.5-2B -> qwen2b-ga-chat-v2 on filtered minimax-m3.1 call traces.

Answers-only targets. Loss on assistant content + <|im_end|> only (prompt and the empty
non-thinking `<think>\n\n</think>\n\n` header are masked, matching ollama's think=false
render). Logits are computed only at supervised positions (248k vocab x long seqs is
the OOM source on 8 GB). bf16 base, gradient checkpointing, VRAM capped at ~6 GB.

usage: slm_v2_sft.py <train.jsonl> <test.jsonl> <out_dir> [epochs]
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIN, TEST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
BASE = "Qwen/Qwen3.5-2B"
LR = 1e-4
GRAD_ACCUM = 8
TOK_BUDGET = 4600          # max tokens per micro-batch (padded)
SAVE_EVERY = 250           # optimizer updates
VAL_N = 120
MEM_FRACTION = 6.0 / 8.0   # hard cap so the live sim's embedding model never starves
CE_CHUNK = 256

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
tok = AutoTokenizer.from_pretrained(BASE)


def encode(msgs):
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    labels[-1] = -100  # trailing newline after <|im_end|> is not supervised
    return ids, labels


def load(path):
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        ids, labels = encode(r["messages"])
        out.append({"ids": ids, "labels": labels, "family": r["family"]})
    return out


def batches(items, rnd):
    items = items[:]
    rnd.shuffle(items)
    out = []
    for i in range(0, len(items), 64):  # length-sort inside chunks of 64, pack by token budget
        chunk = sorted(items[i:i + 64], key=lambda x: len(x["ids"]))
        cur = []
        for it in chunk:
            L = max([len(x["ids"]) for x in cur + [it]])
            if cur and L * (len(cur) + 1) > TOK_BUDGET:
                out.append(cur)
                cur = []
            cur.append(it)
        if cur:
            out.append(cur)
    rnd.shuffle(out)
    return out


def collate(b, pad):
    L = max(len(x["ids"]) for x in b)
    ids = torch.full((len(b), L), pad, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        n = len(x["ids"])
        ids[i, :n] = torch.tensor(x["ids"])
        lab[i, :n] = torch.tensor(x["labels"])
        att[i, :n] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


def loss_fn(model, ids, lab, att):
    inner = model.base_model.model
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    hs, ts = h[:, :-1][sel], tgt[sel]
    head = inner.lm_head

    def _ce(hc, tc):
        return torch.nn.functional.cross_entropy(head(hc).float(), tc, reduction="sum")
    total = 0.0
    for i in range(0, hs.size(0), CE_CHUNK):  # never materialise [n, 248k] logits at once
        if torch.is_grad_enabled():
            total = total + torch.utils.checkpoint.checkpoint(_ce, hs[i:i + CE_CHUNK], ts[i:i + CE_CHUNK],
                                                              use_reentrant=False)
        else:
            total = total + _ce(hs[i:i + CE_CHUNK], ts[i:i + CE_CHUNK])
    return total, int(sel.sum())


@torch.no_grad()
def val_loss(model, items):
    model.eval()
    tot, n = 0.0, 0
    for it in items:
        ids, lab, att = collate([it], tok.pad_token_id)
        l, k = loss_fn(model, ids, lab, att)
        tot += l.item()
        n += k
    model.train()
    return tot / max(1, n)


def main():
    os.makedirs(OUT, exist_ok=True)
    train = load(TRAIN)
    test = load(TEST)
    rnd = random.Random(7)
    val = rnd.sample(test, min(VAL_N, len(test)))
    print(f"train {len(train)} rows, {sum(len(x['ids']) for x in train)/1e6:.2f}M tokens, "
          f"{sum(sum(1 for t in x['labels'] if t != -100) for x in train)/1e6:.2f}M supervised", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model.config.use_cache = False
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                                      "out_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.99))
    ep_batches = [batches(train, random.Random(100 + e)) for e in range(EPOCHS)]
    total = sum(math.ceil(len(b) / GRAD_ACCUM) for b in ep_batches)
    warm = max(10, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    v0 = val_loss(model, val)
    print(f"updates total {total}; init val loss {v0:.4f}", flush=True)
    hist = [{"update": 0, "val_loss": v0}]
    t0 = time.time()
    upd = 0
    for ep in range(EPOCHS):
        bl = ep_batches[ep]
        run_l, run_n = 0.0, 0
        acc_l, acc_n = None, 0
        for bi, b in enumerate(bl):
            ids, lab, att = collate(b, tok.pad_token_id)
            l, k = loss_fn(model, ids, lab, att)
            # token-weighted mean over the accumulation window
            (l / 2000.0).backward()
            run_l += l.item()
            run_n += k
            acc_n += k
            if (bi + 1) % GRAD_ACCUM == 0 or bi + 1 == len(bl):
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(2000.0 / max(1, acc_n))
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                acc_n = 0
                upd += 1
                if upd % 25 == 0:
                    el = time.time() - t0
                    print(f"ep {ep+1} upd {upd}/{total} loss {run_l/max(1,run_n):.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"mem {torch.cuda.max_memory_allocated()/2**30:.2f}G elapsed {el/60:.1f}m "
                          f"eta {(total-upd)*el/upd/60:.0f}m", flush=True)
                    run_l, run_n = 0.0, 0
                if upd % SAVE_EVERY == 0:
                    model.save_pretrained(os.path.join(OUT, "checkpoint_latest"))
                    json.dump({"update": upd, "epoch": ep + 1}, open(os.path.join(OUT, "checkpoint_latest", "meta.json"), "w"))
        v = val_loss(model, val)
        hist.append({"update": upd, "epoch": ep + 1, "val_loss": v})
        print(f"=== epoch {ep+1} done: val loss {v:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        model.save_pretrained(os.path.join(OUT, f"epoch{ep+1}"))
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    json.dump({"base": BASE, "epochs": EPOCHS, "lr": LR, "grad_accum": GRAD_ACCUM, "tok_budget": TOK_BUDGET,
               "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": lcfg.target_modules and sorted(lcfg.target_modules)},
               "train_rows": len(train), "updates": upd, "minutes": round((time.time() - t0) / 60, 1),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "history": hist},
              open(os.path.join(OUT, "train_meta.json"), "w"), indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
