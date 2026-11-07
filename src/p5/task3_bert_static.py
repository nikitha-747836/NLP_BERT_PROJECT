from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.decomposition import PCA

from .benchmarks import eval_analogy, eval_wordsim, load_analogy, load_wordsim, public
from .data import load_lines
from .features import extract_split, layer_features
from .structural import get_structural_sample, static_vector_check
from .task1_ppmi import build_vocab, load_vectors
from .utils import LOG, ensure_dirs, get_device, load_config, load_json, save_json, set_seed, setup_logging


def pool_static(sents, meta: dict, layer: int, vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X = layer_features(meta, "train", layer, mmap=True)
    S = np.zeros((len(vocab), X.shape[1]), dtype=np.float64)
    cnt = np.zeros(len(vocab), dtype=np.int64)
    off, skip = meta["offsets"], set(meta["skipped"])
    for i, s in enumerate(sents):
        if i in skip:
            continue
        ids = np.fromiter((vocab[t.lower()] for t in s.tokens), dtype=np.int64, count=len(s))
        np.add.at(S, ids, np.asarray(X[off[i]:off[i + 1]], dtype=np.float64))
        np.add.at(cnt, ids, 1)
    M = (S / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    return M, cnt


def bootstrap_ws_diff(scores_a, scores_b, human, covered, n_boot: int, seed: int) -> dict:
    """Paired bootstrap over covered WordSim pairs for rho(B) - rho(A)."""
    from scipy.stats import spearmanr

    idx = np.flatnonzero(covered)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(spearmanr(scores_b[s], human[s]).statistic - spearmanr(scores_a[s], human[s]).statistic)
    diffs = np.array(diffs)
    return {"delta": float(spearmanr(scores_b[idx], human[idx]).statistic
                           - spearmanr(scores_a[idx], human[idx]).statistic),
            "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
            "p_boot_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def mcnemar(correct_a, correct_b, mask) -> dict:
    a, b = correct_a[mask], correct_b[mask]
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_task1_correct": only_a, "only_task3_correct": only_b, "p_exact_mcnemar": float(p)}


def run(cfg: dict) -> dict:
    paths = ensure_dirs(cfg)
    out = paths["output_dir"] / "task3"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    t3 = cfg["task3"]
    data = load_lines(cfg)
    train, test = data["train"], data["test"]
    ws, an = load_wordsim(cfg), load_analogy(cfg)

    best = t3.get("layer") or load_json(paths["output_dir"] / "task2" / "best_layer.json")["best_layer"]
    LOG.info("Using BERT layer l* = %d", best)
    meta = extract_split(cfg, "train", train, device)  # cached by Task 2
    vocab = build_vocab(train)
    t1_dir = paths["checkpoint_dir"] / "task1"
    v1, M1 = load_vectors(t1_dir)
    if v1 != vocab:
        raise RuntimeError("Task 1 vocabulary differs from Task 3's - rerun Task 1 with the same data")

    sample = [test[i] for i in get_structural_sample(cfg, test)]
    t1_res = load_json(paths["output_dir"] / "task1" / "results.json")

    M3, cnt = pool_static(train, meta, best, vocab)
    if (cnt == 0).any():
        LOG.warning("%d word types have no occurrence (skipped sentences)", int((cnt == 0).sum()))
    np.save(paths["checkpoint_dir"] / f"task3_vectors_L{best:02d}.npy", M3)
    (paths["checkpoint_dir"] / "task3_vocab.txt").write_text(
        "\n".join(sorted(vocab, key=vocab.get)) + "\n", encoding="utf-8")
    w3, a3, s3 = eval_wordsim(vocab, M3, ws), eval_analogy(vocab, M3, an), static_vector_check(sample, vocab, M3)
    M3p = PCA(n_components=min(100, M3.shape[1]), random_state=cfg["seed"]).fit_transform(M3)
    w3p, a3p, s3p = eval_wordsim(vocab, M3p, ws), eval_analogy(vocab, M3p, an), static_vector_check(sample, vocab, M3p)
    w1, a1, s1 = eval_wordsim(vocab, M1, ws), eval_analogy(vocab, M1, an), static_vector_check(sample, vocab, M1)

    sig = {
        "wordsim_bootstrap": bootstrap_ws_diff(w1["_model_scores"], w3["_model_scores"], w1["_human"],
                                               w1["_covered"], t3["n_bootstrap"], cfg["seed"]),
        "analogy_mcnemar": mcnemar(a1["_correct"], a3["_correct"], a1["_covered"]),
    }
    p1, p3 = np.array(s1["per_sentence"]), np.array(s3["per_sentence"])
    ok = ~(np.isnan(p1) | np.isnan(p3))
    d = p3[ok] - p1[ok]
    sig["structural_wilcoxon"] = {
        "mean_per_sentence_delta": float(d.mean()),
        "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n_tie": int((d == 0).sum()),
        "p_wilcoxon": float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0,
    }

    sweep = []
    for layer in range(0 if t3.get("sweep_include_layer0", True) else 1, meta["n_layers"] + 1):
        M, _ = pool_static(train, meta, layer, vocab)
        w, a, s = eval_wordsim(vocab, M, ws), eval_analogy(vocab, M, an), static_vector_check(sample, vocab, M)
        sweep.append({"layer": layer, "ws353_spearman": w["spearman_covered"], "analogy_acc_all": a["acc_all"],
                      "analogy_acc_covered": a["acc_covered"],
                      "analogy_sem_covered": a["acc_covered_semantic"],
                      "analogy_syn_covered": a["acc_covered_syntactic"], "struct_uuas": s["uuas"]})
        LOG.info("sweep L%02d | WS353 %.4f | analogy %.4f | MST UUAS %.4f", layer,
                 w["spearman_covered"], a["acc_all"], s["uuas"])
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out / "layer_sweep.csv", index=False)

    comp = pd.DataFrame([
        _row("Task 1: PPMI+SVD (100d)", w1, a1, s1),
        _row(f"Task 3: BERT L{best} mean-pooled (768d)", w3, a3, s3),
        _row(f"Task 3: BERT L{best} + PCA (100d)", w3p, a3p, s3p),
    ])
    comp.to_csv(out / "comparison.csv", index=False)
    verdict = _verdict(comp, sig, sweep_df, best)
    res = {
        "best_layer": best, "vocab_size": len(vocab),
        "wordsim353": public(w3), "analogy": public(a3), "structural_check": s3,
        "pca100": {"wordsim353": public(w3p), "analogy": public(a3p), "structural_check": s3p},
        "task1_reference": {"wordsim353": public(w1), "analogy": public(a1), "structural_check": s1},
        "task1_results_file_uuas": t1_res["structural_check"]["uuas"],
        "significance": sig, "comparison": comp.to_dict(orient="records"),
        "layer_sweep": sweep, "verdict": verdict,
    }
    save_json(res, out / "results.json")
    _plot(comp, sweep_df, best, out)
    LOG.info("\n%s", comp.to_string(index=False))
    for v in verdict["statements"]:
        LOG.info("TASK 3 | %s", v)
    return res


def _row(name, w, a, s):
    return {"model": name, "ws353_spearman_covered": w["spearman_covered"], "ws353_n_covered": w["n_covered"],
            "ws353_spearman_all353": w["spearman_all353"], "analogy_acc_all": a["acc_all"],
            "analogy_acc_covered": a["acc_covered"], "analogy_n_covered": a["n_covered"],
            "analogy_sem_covered": a["acc_covered_semantic"], "analogy_syn_covered": a["acc_covered_syntactic"],
            "struct_uuas_20": s["uuas"]}


def _verdict(comp: pd.DataFrame, sig: dict, sweep: pd.DataFrame, best: int) -> dict:
    """Data-driven statements (no hand-written numbers) for the report's discussion."""
    t1, t3 = comp.iloc[0], comp.iloc[1]
    st = []

    def cmp(metric, label, p):
        d = t3[metric] - t1[metric]
        word = "higher" if d > 0 else "lower" if d < 0 else "equal"
        sigw = "statistically significant" if p < 0.05 else "not statistically significant"
        st.append(f"{label}: BERT-L{best} {t3[metric]:.4f} vs PPMI {t1[metric]:.4f} "
                  f"({word}, delta {d:+.4f}; {sigw}, p={p:.3g}).")
        return d

    dw = cmp("ws353_spearman_covered", "WordSim-353 Spearman", sig["wordsim_bootstrap"]["p_boot_two_sided"])
    da = cmp("analogy_acc_covered", "Analogy accuracy (covered)", sig["analogy_mcnemar"]["p_exact_mcnemar"])
    ds = cmp("struct_uuas_20", "Structural MST UUAS (20 sents)", sig["structural_wilcoxon"]["p_wilcoxon"])
    sem_better = (dw > 0) + (da > 0)
    if sem_better == 2 and ds > 0:
        agree = "agree: BERT-derived vectors are better on both semantic benchmarks and the structural check"
    elif sem_better == 0 and ds <= 0:
        agree = "agree: the classical vectors are at least as good on both semantic and structural evaluations"
    else:
        agree = "do not fully agree: the semantic benchmarks and the structural check rank the two models differently"
    st.append(f"The two kinds of evaluation {agree}.")
    bw = int(sweep.loc[sweep.ws353_spearman.idxmax(), "layer"])
    bs = int(sweep.loc[sweep.struct_uuas.idxmax(), "layer"])
    st.append(f"Across pooled layers, WordSim-353 peaks at layer {bw} and the static-vector MST check peaks at "
              f"layer {bs}; the probe-selected layer is {best}.")
    return {"statements": st, "best_ws_layer_pooled": bw, "best_struct_layer_pooled": bs}


def _plot(comp: pd.DataFrame, sweep: pd.DataFrame, best: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    names = ["PPMI+SVD", f"BERT L{best}", f"BERT L{best}\nPCA-100"]
    cols = ["#8C8C8C", "#8172B2", "#B39DDB"]
    for a, (m, t) in zip(ax, [("ws353_spearman_covered", "WordSim-353 Spearman (covered pairs)"),
                              ("analogy_acc_covered", "Google analogy acc (covered)"),
                              ("struct_uuas_20", "MST UUAS, 20 test sentences")]):
        a.bar(names, comp[m], color=cols)
        for i, v in enumerate(comp[m]):
            a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        a.set_title(t)
    fig.suptitle("Task 3 vs Task 1 (same vocabulary, same benchmark items, same 20 sentences)")
    fig.tight_layout()
    fig.savefig(out / "task3_vs_task1.png", dpi=130)
    plt.close(fig)

    fig, a1 = plt.subplots(figsize=(8, 4.5))
    a1.plot(sweep.layer, sweep.ws353_spearman, "o-", label="WordSim-353 Spearman")
    a1.plot(sweep.layer, sweep.analogy_acc_covered, "s-", label="Analogy acc (covered)")
    a1.plot(sweep.layer, sweep.struct_uuas, "^-", label="MST UUAS (20 sents)")
    a1.axvline(best, color="k", ls="--", lw=1, label=f"l* = {best} (Task 2)")
    a1.set(xlabel="BERT layer used for pooling", ylabel="score", xticks=list(sweep.layer),
           title="Pooled static vectors: semantic vs structural quality by layer")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "task3_layer_sweep.png", dpi=130)
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
