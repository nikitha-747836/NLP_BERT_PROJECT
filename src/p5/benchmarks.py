"""WordSim-353 and Google analogy benchmarks (shared, identical protocol for Tasks 1 and 3)."""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .utils import LOG


def _fetch(urls: list[str], dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errs = []
    for u in urls:
        try:
            LOG.info("Downloading %s", u)
            urllib.request.urlretrieve(u, dest)
            if dest.stat().st_size > 0:
                return dest
        except Exception as e:  # noqa: BLE001
            errs.append(f"{u}: {e}")
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"Could not obtain {dest.name}. Tried:\n  " + "\n  ".join(errs)
                       + f"\nPlace the file manually at {dest}.")


def load_wordsim(cfg: dict) -> list[tuple[str, str, float]]:
    path = Path(cfg["benchmarks"].get("wordsim_path") or "")
    if not path.is_file():
        path = _fetch(cfg["benchmarks"]["wordsim_urls"], Path(cfg["paths"]["data_dir"]) / "wordsim353.tsv")
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else re.split(r"[,\s]+", line.strip())
            if len(parts) < 3:
                continue
            try:
                pairs.append((parts[0].strip().lower(), parts[1].strip().lower(), float(parts[2])))
            except ValueError:  # header line such as "Word 1  Word 2  Human (mean)"
                continue
    if len(pairs) != 353:
        LOG.warning("WordSim-353 file has %d pairs (expected 353)", len(pairs))
    return pairs


def load_analogy(cfg: dict) -> list[tuple[str, str, str, str, str]]:
    path = Path(cfg["benchmarks"].get("analogy_path") or "")
    if not path.is_file():
        path = _fetch(cfg["benchmarks"]["analogy_urls"], Path(cfg["paths"]["data_dir"]) / "questions-words.txt")
    qs, section = [], "unknown"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                section = line[1:].strip()
                continue
            w = line.lower().split()
            if len(w) == 4:
                qs.append((section, *w))
    if len(qs) != 19544:
        LOG.warning("Analogy file has %d questions (expected 19544)", len(qs))
    return qs


def normalize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


def eval_wordsim(vocab: dict[str, int], M: np.ndarray, pairs) -> dict:
    "
    Mn = normalize(M)
    model, human, cov_mask = [], [], []
    for a, b, h in pairs:
        ok = a in vocab and b in vocab
        model.append(float(Mn[vocab[a]] @ Mn[vocab[b]]) if ok else 0.0)
        human.append(h)
        cov_mask.append(ok)
    model, human, cov_mask = np.array(model), np.array(human), np.array(cov_mask)
    rho_cov = spearmanr(model[cov_mask], human[cov_mask]).statistic if cov_mask.sum() > 2 else float("nan")
    rho_all = spearmanr(model, human).statistic
    return {"n_pairs": len(pairs), "n_covered": int(cov_mask.sum()),
            "spearman_covered": float(rho_cov), "spearman_all353": float(rho_all),
            "_model_scores": model, "_human": human, "_covered": cov_mask}


def eval_analogy(vocab: dict[str, int], M: np.ndarray, questions, batch: int = 2048) -> dict:
    
    Mn = normalize(M)
    cov_idx, correct_all = [], np.zeros(len(questions), dtype=bool)
    abcd = []
    for qi, (_, a, b, c, d) in enumerate(questions):
        if all(w in vocab for w in (a, b, c, d)):
            cov_idx.append(qi)
            abcd.append((vocab[a], vocab[b], vocab[c], vocab[d]))
    if abcd:
        abcd = np.array(abcd)
        for s in range(0, len(abcd), batch):
            q = abcd[s:s + batch]
            target = Mn[q[:, 1]] - Mn[q[:, 0]] + Mn[q[:, 2]]
            scores = target @ Mn.T
            rows = np.arange(len(q))
            for k in range(3):
                scores[rows, q[:, k]] = -np.inf
            pred = scores.argmax(1)
            correct_all[np.array(cov_idx[s:s + batch])] = pred == q[:, 3]
    cov = np.zeros(len(questions), dtype=bool)
    cov[cov_idx] = True
    syn = np.array([q[0].startswith("gram") for q in questions])

    def acc(mask):
        return float(correct_all[mask].mean()) if mask.sum() else float("nan")

    return {
        "n_questions": len(questions), "n_covered": int(cov.sum()),
        "acc_all": acc(np.ones_like(cov)), "acc_covered": acc(cov),
        "acc_all_semantic": acc(~syn), "acc_all_syntactic": acc(syn),
        "acc_covered_semantic": acc(cov & ~syn), "acc_covered_syntactic": acc(cov & syn),
        "n_covered_semantic": int((cov & ~syn).sum()), "n_covered_syntactic": int((cov & syn).sum()),
        "_correct": correct_all, "_covered": cov,
    }


def public(d: dict) -> dict:
    """Strip private arrays (keys starting with '_') before saving to JSON."""
    return {k: v for k, v in d.items() if not k.startswith("_")}
