#!/usr/bin/env python3
"""Fine-tune Laya v3 -> v4 on the m3.1 teacher set (single GPU, <6 GB VRAM).

Single-GPU port of laya-main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
(train_ddp.py): RLCD = GRPO-style policy gradient over noisy logit projections rewarded
by proper scoring rules + full-weight soft cross-entropy, then one temperature per type
fitted by LBFGS on a calibration slice that never entered training. Memory savers:
gradient checkpointing (encoder + head), bf16 autocast, bitsandbytes AdamW8bit.

usage: laya_v4_train.py <init_ckpt> <data_dir> <out_dir> [epochs]
"""
import json
import math
import os
import random
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import QTYPES, build_model, build_sequence, proper_reward, serialize_state, encode_text

INIT, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.0e-5
LR_HEAD = 1.0e-4
SIGMA_START, SIGMA_END = 0.4, 0.1
OVERSAMPLE = {"poignancy": 1, "talk": 2, "react": 2}
device = torch.device("cuda")


def to_internal(qdef):
    crit = qdef["criteria"]
    return {"t": qdef["type"], "ins": qdef["instructions"], "crit": crit}


def make_item(tok, cfg, rec):
    (qid, qdef), = rec["questions"].items()
    q = to_internal(qdef)
    state_ids = encode_text(tok, serialize_state(rec["state"]).replace(tok.mask_token, " "),
                            add_special_tokens=False)["input_ids"]
    seq, markers = build_sequence(tok, rec["state"], q, cfg["max_len"], cfg["head_max_len"],
                                  state_ids=state_ids)
    if len(markers) != len(rec["target"]):
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": rec["target"],
            "label": rec["label"], "kind": rec["kind"]}


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def predict_logits(model, items, pad_id):
    model.eval()
    out = []
    for i in range(0, len(items), 16):
        ch = items[i:i + 16]
        b = collate(ch, pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        lg = lg.float().cpu()
        for r, it in enumerate(ch):
            out.append(lg[r, :len(it["markers"])].tolist())
    model.train()
    return out


def calib_report(logits, items):
    ce, acc = 0.0, 0
    for z, it in zip(logits, items):
        lp = torch.log_softmax(torch.tensor(z), -1)
        ce += -(torch.tensor(it["target"]) * lp).sum().item()
        acc += int(int(lp.argmax()) == it["label"])
    return ce / max(1, len(items)), acc / max(1, len(items))


def save(model, tok, cfg, out_dir, temps=None, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    c = dict(cfg)
    if temps is not None:
        c["temperature"] = temps
        c.pop("temperature_by_options", None)  # inherited buckets would mask the new fit
    if meta:
        c["training"] = meta
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(c, f, indent=2)


def main():
    cfg = json.load(open(os.path.join(INIT, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(INIT, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(INIT, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(INIT, "model.safetensors")), strict=True)
    model = model.float()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    def load_split(name):
        items = []
        for line in open(os.path.join(DATA, f"{name}.jsonl")):
            it = make_item(tok, cfg, json.loads(line))
            if it:
                items.append(it)
        return items
    train = load_split("train")
    calib = load_split("calib")
    train_os = [it for it in train for _ in range(OVERSAMPLE[it["kind"]])]
    print(f"train items {len(train)} (oversampled {len(train_os)}), calib {len(calib)}", flush=True)

    import bitsandbytes as bnb
    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = bnb.optim.AdamW8bit([{"params": enc_p, "lr": LR_ENCODER}, {"params": head_p, "lr": LR_HEAD}],
                              weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_os) / (MICRO_BATCH * GRAD_ACCUM))
    total = steps_per_epoch * EPOCHS
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.04, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    lc0 = predict_logits(model, calib, tok.pad_token_id)
    print("init calib CE %.4f acc %.3f" % calib_report(lc0, calib), flush=True)

    t0 = time.time()
    updates = 0
    hist = []
    for epoch in range(EPOCHS):
        rnd = random.Random(42 + epoch)
        # length-bucketed shuffle: sort within chunks so micro-batches pad less
        order = train_os[:]
        rnd.shuffle(order)
        chunks = [sorted(order[i:i + 256], key=lambda it: len(it["ids"])) for i in range(0, len(order), 256)]
        batches = [c[j:j + MICRO_BATCH] for c in chunks for j in range(0, len(c), MICRO_BATCH)]
        rnd.shuffle(batches)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        ep_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, chunk in enumerate(batches):
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                                    b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            qt = b["qtype"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qt, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            loss.backward()
            if (bi + 1) % GRAD_ACCUM == 0 or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
            ep_loss += loss.item() * GRAD_ACCUM
            nb += 1
            if nb % 50 == 0:
                print(f"  ep {epoch+1} mb {nb}/{len(batches)} loss {loss.item()*GRAD_ACCUM:.4f} "
                      f"ce {loss_ce.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"mem {torch.cuda.max_memory_allocated()/2**30:.2f}G t {time.time()-t0:.0f}s", flush=True)
        lc = predict_logits(model, calib, tok.pad_token_id)
        ce, acc = calib_report(lc, calib)
        hist.append({"epoch": epoch + 1, "train_loss": ep_loss / nb, "calib_ce": ce, "calib_acc": acc})
        print(f"=== epoch {epoch+1}: train loss {ep_loss/nb:.4f} calib CE {ce:.4f} acc {acc:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save(model, tok, cfg, os.path.join(OUT, "checkpoint_latest"),
             meta={"epoch": epoch + 1, "history": hist})

    lc = predict_logits(model, calib, tok.pad_token_id)
    temps = [1.0, 1.0, 1.0]
    for qtv in range(3):
        sel = [(z, it["target"]) for z, it in zip(lc, calib) if it["qtype"] == qtv]
        if sel:
            try:
                temps[qtv] = fit_one_temp(sel)
            except Exception as e:  # noqa: BLE001
                print("temperature fit fallback:", e)
                temps[qtv] = 1.2
    print("fitted temperatures (choice, score, noul):", [round(t, 3) for t in temps], flush=True)
    hours = (time.time() - t0) / 3600
    meta = {"updates": updates, "epochs_completed": EPOCHS, "hours": round(hours, 3), "world_size": 1,
            "fine_tuned_from_checkpoint": INIT, "data": DATA, "teacher": "minimax-m3.1-flash (blind)",
            "lr_encoder": LR_ENCODER, "lr_head": LR_HEAD, "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
            "oversample": OVERSAMPLE, "history": hist,
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    cfg["model_name"] = "laya-ga-v4"
    cfg["fine_tuned"] = True
    save(model, tok, cfg, OUT, temps=temps, meta=meta)
    print(f"saved {OUT} in {hours:.2f} h", flush=True)


if __name__ == "__main__":
    main()
