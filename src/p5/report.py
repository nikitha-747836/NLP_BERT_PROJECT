
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import LOG, load_json


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: format(v, floatfmt))
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def build(cfg: dict) -> Path:
    out = Path(cfg["paths"]["output_dir"])
    r = ["# Project 5 - From Static to Contextual: results on UD English LinES", ""]
    ds = load_json(out / "dataset" / "dataset_stats.json") if (out / "dataset" / "dataset_stats.json").is_file() else None
    if ds:
        r += [f"Data source: `{ds['source']}`", "",
              _md(pd.DataFrame([{"split": k, **{x: v[x] for x in ("sentences", "tokens", "types_lowercased",
                                 "len_mean", "len_max", "non_projective", "sentences_also_in_train")}}
                                for k, v in ds["splits"].items()])), "", "![dataset](dataset/dataset_stats.png)", ""]
    if (out / "task1" / "results.json").is_file():
        t1 = load_json(out / "task1" / "results.json")
        w, a, s, b = t1["wordsim353"], t1["analogy"], t1["structural_check"], t1["structural_baselines"]
        r += ["## Task 1 - PPMI + SVD", "",
              f"Vocabulary {t1['vocab_size']} lower-cased types from {t1['train_tokens']} train tokens; "
              f"window +/-{t1['config']['window']}, PPMI alpha={t1['config']['alpha']}, SVD {t1['config']['dim']}d.", "",
              "| Metric | Value |", "|---|---|",
              f"| WordSim-353 Spearman (covered pairs) | {w['spearman_covered']:.4f} ({w['n_covered']}/353 pairs) |",
              f"| WordSim-353 Spearman (all 353, OOV sim = 0) | {w['spearman_all353']:.4f} |",
              f"| Google analogy accuracy (all 19,544, OOV = wrong) | {a['acc_all']:.4f} |",
              f"| Google analogy accuracy (covered) | {a['acc_covered']:.4f} ({a['n_covered']} questions) |",
              f"| MST UUAS vs gold, 20 test sentences | {s['uuas']:.4f} ({s['correct']}/{s['gold_edges']} edges) |",
              f"| - linear-chain baseline / random-tree baseline | {b['linear_chain_uuas']:.4f} / {b['random_tree_uuas']:.4f} |",
              "", "Ablation (PPMI smoothing alpha x SVD weighting power):", "",
              _md(pd.read_csv(out / "task1" / "ablation.csv")), "", "![t1](task1/task1_ablation.png)", ""]
    if (out / "task2" / "results.json").is_file():
        t2 = load_json(out / "task2" / "results.json")
        summ = pd.read_csv(out / "task2" / "layer_summary.csv")
        r += ["## Task 2 - Structural probe on bert-base-uncased", "",
              f"Best layer **l\\* = {t2['best_layer']}** (selected on {t2['selected_on']} UUAS). "
              f"Test-argmax layers: UUAS {t2['best_layer_by_test_uuas']}, depth Spearman "
              f"{t2['best_layer_by_test_depth_spearman']}.", "",
              "Per layer (max over ranks):", "", _md(summ), "",
              "![t2](task2/task2_uuas_spearman_vs_layer.png)", "",
              "All runs: `task2/probe_results.csv`.", ""]
    if (out / "task3" / "results.json").is_file():
        t3 = load_json(out / "task3" / "results.json")
        sig = t3["significance"]
        r += ["## Task 3 - BERT-derived static vectors vs PPMI+SVD", "",
              _md(pd.read_csv(out / "task3" / "comparison.csv")), "",
              f"- WordSim paired bootstrap: delta {sig['wordsim_bootstrap']['delta']:+.4f}, "
              f"95% CI [{sig['wordsim_bootstrap']['ci95'][0]:+.4f}, {sig['wordsim_bootstrap']['ci95'][1]:+.4f}]",
              f"- Analogy McNemar: only-PPMI-correct {sig['analogy_mcnemar']['only_task1_correct']}, "
              f"only-BERT-correct {sig['analogy_mcnemar']['only_task3_correct']}, "
              f"p = {sig['analogy_mcnemar']['p_exact_mcnemar']:.3g}",
              f"- Structural (per-sentence Wilcoxon): BERT better on {sig['structural_wilcoxon']['n_better']}, "
              f"worse on {sig['structural_wilcoxon']['n_worse']}, tied {sig['structural_wilcoxon']['n_tie']}; "
              f"p = {sig['structural_wilcoxon']['p_wilcoxon']:.3g}", "",
              "![t3](task3/task3_vs_task1.png)", "", "Layer sweep of pooled vectors:", "",
              _md(pd.read_csv(out / "task3" / "layer_sweep.csv")), "", "![sweep](task3/task3_layer_sweep.png)", "",
              "### Discussion (generated from the numbers above)", ""]
        r += [f"- {s}" for s in t3["verdict"]["statements"]]
        r += ["", "Interpretation guide: WordSim-353 and analogies test semantic/lexical structure; the MST check "
              "tests whether cosine geometry of type-level vectors mirrors syntactic attachment. Averaging over "
              "contexts removes the token-specific information the probe exploits, so a gain on one evaluation "
              "need not carry over to the other.", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(r), encoding="utf-8")
    LOG.info("Report -> %s", path)
    return path
