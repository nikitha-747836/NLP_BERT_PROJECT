
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import Sentence
from .utils import LOG, amp_dtype, gpu_mem_gb, load_json, save_json


def _cache_dir(cfg: dict) -> Path:
    name = Path(cfg["bert"]["model_name"]).name.replace("/", "_")
    d = Path(cfg["paths"]["cache_dir"]) / f"features_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _auto_batch(cfg: dict, device) -> int:
    bs = cfg["bert"].get("extract_batch_size", "auto")
    if bs != "auto":
        return int(bs)
    mem = gpu_mem_gb(device)
    return 64 if mem >= 20 else 32 if mem >= 10 else 16 if mem > 0 else 8


def load_bert(cfg: dict, device):
    from transformers import AutoModel, AutoTokenizer

    name = cfg["bert"]["model_name"]
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required (word_ids alignment)")
    model = AutoModel.from_pretrained(name, output_hidden_states=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device
    assert dev.type == device.type, f"model on {dev}, expected {device}"
    LOG.info("Loaded %s on %s | layers %d | hidden %d", name, dev,
             model.config.num_hidden_layers, model.config.hidden_size)
    return tok, model


@torch.inference_mode()
def extract_split(cfg: dict, split: str, sents: list[Sentence], device) -> dict:

    d = _cache_dir(cfg)
    meta_path = d / f"{split}_meta.json"
    sig = [len(s) for s in sents]
    if meta_path.is_file():
        meta = load_json(meta_path)
        files_ok = all((d / f"{split}_L{l}.npy").is_file() for l in range(meta["n_layers"] + 1))
        if meta["lengths"] == sig and files_ok:
            LOG.info("Using cached %s features (%s)", split, d)
            meta["dir"] = str(d)
            return meta

    tok, model = load_bert(cfg, device)
    n_layers, H = model.config.num_hidden_layers, model.config.hidden_size
    max_len = min(getattr(tok, "model_max_length", 512), model.config.max_position_embeddings)
    bs = _auto_batch(cfg, device)
    dt = amp_dtype(device) if cfg["bert"].get("mixed_precision", True) else None
    LOG.info("Extracting %s: %d sentences | batch %d | amp %s", split, len(sents), bs, dt)

    def clean(w: str) -> str:  # words that tokenize to nothing (e.g. control chars) -> [UNK]
        return w if tok.tokenize(w) else tok.unk_token

    order = np.argsort([-len(s) for s in sents], kind="stable")  # length-sorted -> less padding
    total_words = sum(sig)
    offsets = np.concatenate([[0], np.cumsum(sig)]).astype(np.int64)
    feats = [np.zeros((total_words, H), dtype=np.float16) for _ in range(n_layers + 1)]
    skipped = []
    for b in range(0, len(order), bs):
        idx = [int(i) for i in order[b:b + bs]]
        words = [[clean(w) for w in sents[i].tokens] for i in idx]
        enc = tok(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=False)
        keep_rows = [r for r in range(len(idx)) if int(enc["attention_mask"][r].sum()) <= max_len]
        for r in range(len(idx)):
            if r not in keep_rows:
                skipped.append(idx[r])
                LOG.warning("Sentence %s exceeds %d subwords - skipped", sents[idx[r]].sid, max_len)
        if not keep_rows:
            continue
        L = max(int(enc["attention_mask"][r].sum()) for r in keep_rows)
        inputs = {k: v[keep_rows, :L].to(device) for k, v in enc.items()}
        with torch.autocast(device.type, dtype=dt, enabled=dt is not None):
            hs = model(**inputs).hidden_states  # tuple(n_layers+1) of (b, L, H)
        hs = torch.stack(hs, 0).float()  # (layers, b, L, H)
        for bi, r in enumerate(keep_rows):
            wid = enc.word_ids(r)[:L]
            n_w = len(words[r])
            map_ = torch.zeros(n_w, L, device=device)
            for t, w in enumerate(wid):
                if w is not None:
                    map_[w, t] = 1.0
            cnt = map_.sum(1, keepdim=True)
            if (cnt == 0).any():
                raise RuntimeError(f"word without subwords in {sents[idx[r]].sid}")
            pooled = torch.einsum("wt,ltd->lwd", map_ / cnt, hs[:, bi])  # mean of subwords
            s0, s1 = offsets[idx[r]], offsets[idx[r] + 1]
            arr = pooled.to(torch.float16).cpu().numpy()
            for l in range(n_layers + 1):
                feats[l][s0:s1] = arr[l]
    for l in range(n_layers + 1):
        np.save(d / f"{split}_L{l}.npy", feats[l])
    meta = {"offsets": offsets.tolist(), "lengths": sig, "skipped": sorted(skipped),
            "n_layers": n_layers, "hidden": H, "model": cfg["bert"]["model_name"]}
    save_json(meta, meta_path)
    meta["dir"] = str(d)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    LOG.info("Saved %s features: %d words x %d dims x %d layers (%d skipped sents)",
             split, total_words, H, n_layers + 1, len(skipped))
    return meta


def layer_features(meta: dict, split: str, layer: int, mmap: bool = True) -> np.ndarray:
    return np.load(Path(meta["dir"]) / f"{split}_L{layer}.npy", mmap_mode="r" if mmap else None)
