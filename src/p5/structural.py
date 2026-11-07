
from __future__ import annotations

from pathlib import Path

import numpy as np

from .benchmarks import normalize
from .data import Sentence
from .utils import LOG, load_json, save_json


def prim_mst(dist: np.ndarray) -> set[tuple[int, int]]:
    n = dist.shape[0]
    if n < 2:
        return set()
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].astype(np.float64).copy()
    parent = np.zeros(n, dtype=int)
    edges = set()
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        v = int(np.argmin(cand))
        edges.add((min(v, parent[v]), max(v, parent[v])))
        in_tree[v] = True
        upd = (~in_tree) & (dist[v] < best)
        best[upd] = dist[v][upd]
        parent[upd] = v
    return edges


def gold_edges(sent: Sentence, keep: np.ndarray) -> set[tuple[int, int]]:
    """Undirected gold edges between kept tokens, re-indexed to positions in the kept subset."""
    pos = -np.ones(len(sent), dtype=int)
    pos[keep] = np.arange(keep.sum())
    out = set()
    for i, h in enumerate(sent.heads):
        if h > 0 and keep[i] and keep[h - 1]:
            a, b = pos[i], pos[h - 1]
            out.add((min(a, b), max(a, b)))
    return out


def uuas_for_distance(sent: Sentence, dist: np.ndarray) -> tuple[int, int]:
    """(correct, total) undirected attachments for one sentence given a full n x n distance matrix."""
    keep = ~sent.punct_mask
    if keep.sum() < 2:
        return 0, 0
    sub = dist[np.ix_(keep, keep)]
    pred = prim_mst(sub)
    gold = gold_edges(sent, keep)
    return len(pred & gold), len(gold)


def get_structural_sample(cfg: dict, test: list[Sentence]) -> list[int]:
    """Seeded sample of test-sentence indices, saved once and reused so Tasks 1 and 3 match."""
    path = Path(cfg["paths"]["output_dir"]) / "structural_sample.json"
    n, seed = cfg["structural_check"]["n_sentences"], cfg["structural_check"]["seed"]
    if path.is_file():
        saved = load_json(path)
        if saved["n"] == n and saved["seed"] == seed and saved["n_test"] == len(test):
            return saved["indices"]
    eligible = [i for i, s in enumerate(test) if (~s.punct_mask).sum() >= 2]
    rng = np.random.default_rng(seed)
    idx = sorted(int(i) for i in rng.choice(eligible, size=min(n, len(eligible)), replace=False))
    save_json({"n": n, "seed": seed, "n_test": len(test), "indices": idx,
               "sent_ids": [test[i].sid for i in idx],
               "texts": [" ".join(test[i].tokens) for i in idx]}, path)
    LOG.info("Saved structural-check sample (%d sentences) -> %s", len(idx), path)
    return idx


def static_vector_check(sents: list[Sentence], vocab: dict[str, int], M: np.ndarray) -> dict:

    Mn = normalize(M)
    tot_c = tot_g = n_tok = n_oov = 0
    per = []
    for s in sents:
        X = np.zeros((len(s), Mn.shape[1]), dtype=np.float32)
        for i, w in enumerate(s.tokens):
            j = vocab.get(w.lower())
            if j is None:
                n_oov += 1
            else:
                X[i] = Mn[j]
        n_tok += len(s)
        dist = 1.0 - X @ X.T
        np.fill_diagonal(dist, 0.0)
        c, g = uuas_for_distance(s, dist)
        tot_c, tot_g = tot_c + c, tot_g + g
        per.append(c / g if g else float("nan"))
    return {"uuas": tot_c / max(tot_g, 1), "correct": tot_c, "gold_edges": tot_g,
            "per_sentence": per, "oov_token_rate": n_oov / max(n_tok, 1)}


def baselines(sents: list[Sentence], seed: int, n_random: int = 100) -> dict:
    """Reference points: linear chain (left-to-right adjacency) and random spanning trees."""
    tot_g = lin_c = 0
    rand_c = np.zeros(n_random)
    rng = np.random.default_rng(seed)
    per_lin = []
    for s in sents:
        keep = ~s.punct_mask
        m = int(keep.sum())
        if m < 2:
            continue
        gold = gold_edges(s, keep)
        tot_g += len(gold)
        chain = {(i, i + 1) for i in range(m - 1)}
        lin_c += len(chain & gold)
        per_lin.append(len(chain & gold) / len(gold) if gold else float("nan"))
        for r in range(n_random):  # random tree: MST over random distances
            d = rng.random((m, m))
            d = (d + d.T) / 2
            rand_c[r] += len(prim_mst(d) & gold)
    return {"linear_chain_uuas": lin_c / tot_g, "random_tree_uuas": float((rand_c / tot_g).mean()),
            "linear_chain_per_sentence": per_lin}
