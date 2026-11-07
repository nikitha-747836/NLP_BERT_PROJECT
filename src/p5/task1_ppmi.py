
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .benchmarks import eval_analogy, eval_wordsim, load_analogy, load_wordsim, public
from .data import dataset_report, load_lines
from .structural import baselines, get_structural_sample, static_vector_check
from .utils import LOG, ensure_dirs, load_config, save_json, set_seed, setup_logging


def build_vocab(sents) -> dict[str, int]:
    """Lower-cased word types of the training split, frequency-sorted (ties alphabetical)."""
    cnt = collections.Counter(t.lower() for s in sents for t in s.tokens)
    words = sorted(cnt, key=lambda w: (-cnt[w], w))
    return {w: i for i, w in enumerate(words)}


def cooccurrence(sents, vocab: dict[str, int], window: int) -> sp.csr_matrix:
    """Symmetric word-word counts: every pair within `window` positions in the same sentence."""
    rows, cols = [], []
    for s in sents:
        ids = [vocab[t.lower()] for t in s.tokens]
        n = len(ids)
        for i in range(n):
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j != i:
                    rows.append(ids[i])
                    cols.append(ids[j])
    V = len(vocab)
    C = sp.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(V, V)).tocsr()
    C.sum_duplicates()
    assert (abs(C - C.T) > 0).nnz == 0, "co-occurrence matrix must be symmetric"
    return C


def ppmi(C: sp.csr_matrix, alpha: float = 1.0) -> sp.csr_matrix:
    """PPMI(w,c) = max(0, log P(w,c) / (P(w) P_alpha(c))). alpha=1 is plain PPMI;
    alpha=0.75 is context-distribution smoothing (Levy, Goldberg & Dagan 2015)."""
    C = C.tocoo()
    total = C.sum()
    row = np.asarray(C.sum(axis=1)).ravel()
    col = np.asarray(C.sum(axis=0)).ravel() ** alpha
    p_w = row / total
    p_c = col / col.sum()
    pmi = np.log((C.data / total) / (p_w[C.row] * p_c[C.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)


def svd_vectors(P: sp.csr_matrix, dim: int, power: float, seed: int) -> np.ndarray:
    """Truncated SVD; word vectors = U * S^power (power=1 -> U*Sigma, the standard choice)."""
    dim = min(dim, P.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=15, random_state=seed)
    US = svd.fit_transform(P)
    S = svd.singular_values_
    U = US / np.maximum(S, 1e-12)
    return (U * S ** power).astype(np.float32)


def save_vectors(vocab: dict[str, int], M: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", M)
    words = sorted(vocab, key=vocab.get)
    (out / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    with open(out / "vectors.w2v.txt", "w", encoding="utf-8") as f:  # word2vec text format
        f.write(f"{M.shape[0]} {M.shape[1]}\n")
        for w in words:
            f.write(w + " " + " ".join(f"{x:.5f}" for x in M[vocab[w]]) + "\n")


def load_vectors(d: Path) -> tuple[dict[str, int], np.ndarray]:
    words = (d / "vocab.txt").read_text(encoding="utf-8").splitlines()
    M = np.load(d / "vectors.npy")
    if len(words) != M.shape[0]:
        raise ValueError(f"vocab/vector size mismatch in {d}")
    return {w: i for i, w in enumerate(words)}, M


def run(cfg: dict) -> dict:
    paths = ensure_dirs(cfg)
    out = paths["output_dir"] / "task1"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    t1 = cfg["task1"]

    data = load_lines(cfg)
    dataset_report(data, paths["output_dir"] / "dataset")
    train, test = data["train"], data["test"]
    ws, an = load_wordsim(cfg), load_analogy(cfg)

    vocab = build_vocab(train)
    C = cooccurrence(train, vocab, t1["window"])
    LOG.info("Vocab %d types | co-occurrence nnz %d | total count %d", len(vocab), C.nnz, int(C.sum()))
    sample_idx = get_structural_sample(cfg, test)
    sample = [test[i] for i in sample_idx]

    rows, main = [], None
    for alpha in t1["ablation_alpha"]:
        P = ppmi(C, alpha)
        for power in t1["ablation_svd_power"]:
            M = svd_vectors(P, t1["dim"], power, cfg["seed"])
            w, a = eval_wordsim(vocab, M, ws), eval_analogy(vocab, M, an)
            st = static_vector_check(sample, vocab, M)
            is_main = alpha == t1["alpha"] and power == t1["svd_power"]
            rows.append({"alpha": alpha, "svd_power": power, "main": is_main,
                         "ws353_spearman_covered": w["spearman_covered"],
                         "ws353_spearman_all353": w["spearman_all353"],
                         "analogy_acc_all": a["acc_all"], "analogy_acc_covered": a["acc_covered"],
                         "struct_uuas": st["uuas"]})
            if is_main:
                main = (P, M, w, a, st)
            LOG.info("alpha=%.2f p=%.1f | WS353 rho=%.4f (cov %d) | analogy %.4f (cov %.4f) | MST UUAS %.4f%s",
                     alpha, power, w["spearman_covered"], w["n_covered"], a["acc_all"], a["acc_covered"],
                     st["uuas"], "  <- main" if is_main else "")
    if main is None:
        raise ValueError("task1.alpha / task1.svd_power must appear in the ablation lists")
    P, M, w, a, st = main
    save_vectors(vocab, M, paths["checkpoint_dir"] / "task1")
    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)
    np.save(out / "ws353_model_scores.npy", w["_model_scores"])
    np.save(out / "analogy_correct.npy", a["_correct"])

    base = baselines(sample, cfg["seed"])
    res = {
        "config": {"window": t1["window"], "alpha": t1["alpha"], "dim": M.shape[1],
                   "svd_power": t1["svd_power"], "lowercase": True, "min_count": 1},
        "data_source": data["_source"],
        "vocab_size": len(vocab), "cooc_nnz": int(C.nnz), "ppmi_nnz": int(P.nnz),
        "train_tokens": int(sum(map(len, train))),
        "wordsim353": public(w), "analogy": public(a),
        "structural_check": {k: v for k, v in st.items()},
        "structural_baselines": base,
    }
    save_json(res, out / "results.json")
    _plot(rows, out)
    LOG.info("TASK 1 | WordSim-353 Spearman %.4f on %d/353 covered pairs (all-353: %.4f)",
             w["spearman_covered"], w["n_covered"], w["spearman_all353"])
    LOG.info("TASK 1 | Google analogy accuracy %.4f overall (%d/19544 covered; %.4f on covered)",
             a["acc_all"], a["n_covered"], a["acc_covered"])
    LOG.info("TASK 1 | MST-vs-gold UUAS on %d test sentences: %.4f (linear chain %.4f, random %.4f)",
             len(sample), st["uuas"], base["linear_chain_uuas"], base["random_tree_uuas"])
    return res


def _plot(rows, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    df["label"] = df.apply(lambda r: f"a={r.alpha}, p={r.svd_power}" + (" *" if r.main else ""), axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, (col, t) in enumerate([("ws353_spearman_covered", "WordSim-353 Spearman (covered)"),
                                  ("analogy_acc_covered", "Analogy acc (covered)"),
                                  ("struct_uuas", "MST UUAS (20 test sents)")]):
        ax[k].bar(df["label"], df[col], color=["#C44E52" if m else "#8C8C8C" for m in df["main"]])
        ax[k].set_title(t)
        ax[k].tick_params(axis="x", rotation=30)
    fig.suptitle("Task 1 PPMI+SVD (* = specified configuration)")
    fig.tight_layout()
    fig.savefig(out / "task1_ablation.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "run.log")
    run(cfg)


if __name__ == "__main__":
    main()
