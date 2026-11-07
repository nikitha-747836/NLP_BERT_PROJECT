
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr

from .data import Sentence, load_lines
from .features import extract_split, layer_features
from .structural import uuas_for_distance
from .utils import LOG, ensure_dirs, get_device, load_config, save_json, set_seed, setup_logging


class LinearProbe(nn.Module):
    def __init__(self, hidden: int, rank: int, kind: str):
        super().__init__()
        assert kind in ("distance", "depth")
        self.kind = kind
        self.proj = nn.Parameter(torch.empty(hidden, rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)  # same init as the reference implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x @ self.proj  # (b, L, k)
        if self.kind == "depth":
            return (t * t).sum(-1)  # (b, L)
        diff = t.unsqueeze(2) - t.unsqueeze(1)  # (b, L, L, k)
        return (diff * diff).sum(-1)  # (b, L, L)


class SplitTensors:
    """GPU-resident word features (one layer) + padded gold targets for one split."""

    def __init__(self, sents: list[Sentence], meta: dict, split: str, device):
        skip = set(meta["skipped"])
        self.idx = [i for i in range(len(sents)) if i not in skip]
        self.sents = [sents[i] for i in self.idx]
        self.lens = torch.tensor([len(s) for s in self.sents], device=device)
        self.Lmax = int(self.lens.max())
        off = meta["offsets"]
        n = len(self.sents)
        self.gather = torch.full((n, self.Lmax), -1, dtype=torch.long)
        self.dist = torch.zeros((n, self.Lmax, self.Lmax), dtype=torch.float16)
        self.depth = torch.zeros((n, self.Lmax), dtype=torch.float16)
        for k, i in enumerate(self.idx):
            m = len(sents[i])
            self.gather[k, :m] = torch.arange(off[i], off[i] + m)
            self.dist[k, :m, :m] = torch.from_numpy(sents[i].distances.astype(np.float16))
            self.depth[k, :m] = torch.from_numpy(sents[i].depths.astype(np.float16))
        self.gather, self.dist, self.depth = self.gather.to(device), self.dist.to(device), self.depth.to(device)
        self.meta, self.split, self.device, self.X = meta, split, device, None

    def load_layer(self, layer: int):
        arr = np.asarray(layer_features(self.meta, self.split, layer, mmap=False))
        zero = np.zeros((1, arr.shape[1]), dtype=arr.dtype)  # padding row at index -1 -> last
        self.X = torch.from_numpy(np.concatenate([arr, zero])).to(self.device)

    def batch(self, rows: torch.Tensor):
        L = int(self.lens[rows].max())
        g = self.gather[rows, :L]
        x = self.X[g].float()  # index -1 -> zero padding row
        mask = g >= 0
        return x, mask, self.dist[rows, :L, :L].float(), self.depth[rows, :L].float(), self.lens[rows].float()


def probe_loss(pred, gold, mask, lens, kind):
    if kind == "distance":
        m = (mask.unsqueeze(1) & mask.unsqueeze(2)).float()
        per = ((pred - gold).abs() * m).sum((1, 2)) / (lens ** 2)
    else:
        per = ((pred - gold).abs() * mask.float()).sum(1) / lens
    return per.mean()


def train_probe(tr: SplitTensors, kind: str, rank: int, cfg: dict, device, gen: torch.Generator):
    p2 = cfg["task2"]
    probe = LinearProbe(tr.X.shape[1], rank, kind).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=p2["lr"])
    n, bs = len(tr.sents), p2["batch_size"]
    history = []
    for ep in range(p2["epochs"]):
        perm = torch.randperm(n, generator=gen).to(device)
        tot = 0.0
        for b in range(0, n, bs):
            rows = perm[b:b + bs]
            x, mask, dist, depth, lens = tr.batch(rows)
            loss = probe_loss(probe(x), dist if kind == "distance" else depth, mask, lens, kind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(rows)
        history.append(tot / n)
    return probe, history


def _rowwise_spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of A with the same row of B (average ranks for ties)."""
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / den, np.nan)


@torch.no_grad()
def evaluate(probe: LinearProbe, ds: SplitTensors, bs: int = 64) -> dict:
    probe.eval()
    out: dict = {}
    by_len = collections.defaultdict(list)
    c_tot = g_tot = root_ok = root_n = 0
    for b in range(0, len(ds.sents), bs):
        rows = torch.arange(b, min(b + bs, len(ds.sents)), device=ds.device)
        x, mask, *_ = ds.batch(rows)
        pred = probe(x).cpu().numpy()
        for r, k in enumerate(rows.tolist()):
            s = ds.sents[k]
            n, keep = len(s), ~s.punct_mask
            if probe.kind == "distance":
                P = pred[r, :n, :n]
                c, g = uuas_for_distance(s, P)
                c_tot, g_tot = c_tot + c, g_tot + g
                G = s.distances[np.ix_(keep, keep)]
                Pk = P[np.ix_(keep, keep)]
                if 5 <= keep.sum() <= 50:
                    by_len[int(keep.sum())].append(np.nanmean(_rowwise_spearman(Pk, G)))
            else:
                p, gd = pred[r, :n], s.depths
                if keep.sum() >= 1:
                    root = s.heads.index(0)
                    if keep[root]:
                        root_n += 1
                        root_ok += int(np.argmin(np.where(keep, p, np.inf)) == root)
                if 5 <= keep.sum() <= 50:
                    by_len[int(keep.sum())].append(spearmanr(p[keep], gd[keep]).statistic)
    dspr = float(np.nanmean([np.nanmean(v) for v in by_len.values()])) if by_len else float("nan")
    if probe.kind == "distance":
        out = {"uuas": c_tot / max(g_tot, 1), "dist_dspr": dspr}
    else:
        out = {"depth_spearman": dspr, "root_acc": root_ok / max(root_n, 1)}
    probe.train()
    return out


def run(cfg: dict) -> dict:
    paths = ensure_dirs(cfg)
    out = paths["output_dir"] / "task2"
    ck = paths["checkpoint_dir"] / "task2"
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    p2 = cfg["task2"]
    data = load_lines(cfg)
    metas = {s: extract_split(cfg, s, data[s], device) for s in ("train", "dev", "test")}
    tr, dv, te = (SplitTensors(data[s], metas[s], s, device) for s in ("train", "dev", "test"))
    LOG.info("Probe data: train %d / dev %d / test %d sentences on %s",
             len(tr.sents), len(dv.sents), len(te.sents), tr.dist.device)

    rows, curves = [], {}
    layers = p2["layers"]
    for layer in layers:
        for ds in (tr, dv, te):
            ds.load_layer(layer)
        for kind in ("distance", "depth"):
            for rank in p2["ranks"]:
                f = ck / f"{kind}_L{layer:02d}_r{rank}.pt"
                t0 = time.time()
                if f.is_file() and p2.get("resume", True):
                    state = torch.load(f, map_location=device, weights_only=False)
                    probe = LinearProbe(tr.X.shape[1], rank, kind).to(device)
                    probe.load_state_dict(state["state_dict"])
                    hist = state["history"]
                else:
                    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 * layer + rank)
                    torch.manual_seed(cfg["seed"] + 1000 * layer + rank)
                    probe, hist = train_probe(tr, kind, rank, cfg, device, gen)
                    torch.save({"state_dict": probe.state_dict(), "history": hist, "layer": layer,
                                "rank": rank, "kind": kind, "model": cfg["bert"]["model_name"]}, f)
                dev_m, test_m = evaluate(probe, dv), evaluate(probe, te)
                curves[f"{kind}_L{layer}_r{rank}"] = hist
                row = {"layer": layer, "kind": kind, "rank": rank, "final_train_loss": hist[-1]}
                row.update({f"dev_{k}": v for k, v in dev_m.items()})
                row.update({f"test_{k}": v for k, v in test_m.items()})
                rows.append(row)
                LOG.info("L%02d %-8s k=%-3d | loss %.4f | dev %s | test %s | %.1fs", layer, kind, rank,
                         hist[-1], _fmt(dev_m), _fmt(test_m), time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_csv(out / "probe_results.csv", index=False)
    save_json(curves, out / "loss_curves.json")

    # ---- select l*: best dev UUAS (max over ranks); tie-break by dev depth Spearman ----
    dist = df[df.kind == "distance"].groupby("layer")
    dep = df[df.kind == "depth"].groupby("layer")
    summ = pd.DataFrame({
        "dev_uuas": dist["dev_uuas"].max(), "test_uuas": dist["test_uuas"].max(),
        "test_uuas_best_dev_rank": dist.apply(lambda g: g.loc[g.dev_uuas.idxmax(), "test_uuas"],
                                              include_groups=False),
        "dev_depth_spearman": dep["dev_depth_spearman"].max(),
        "test_depth_spearman": dep["test_depth_spearman"].max(),
        "test_root_acc": dep["test_root_acc"].max(),
    })
    sel = p2.get("select_on", "dev")
    key = f"{sel}_uuas"
    best = summ.sort_values([key, f"{sel}_depth_spearman"], ascending=False).index[0]
    summ.to_csv(out / "layer_summary.csv")
    res = {
        "best_layer": int(best), "selected_on": sel,
        "best_layer_by_test_uuas": int(summ["test_uuas"].idxmax()),
        "best_layer_by_test_depth_spearman": int(summ["test_depth_spearman"].idxmax()),
        "best_layer_metrics": summ.loc[best].to_dict(),
        "layer_summary": summ.reset_index().to_dict(orient="records"),
        "n_skipped_long_sentences": {s: len(metas[s]["skipped"]) for s in metas},
        "device": str(device),
    }
    save_json(res, out / "results.json")
    save_json({"best_layer": int(best)}, out / "best_layer.json")
    _plot(df, summ, int(best), out)
    LOG.info("TASK 2 | l* = %d (selected on %s) | test UUAS %.4f (dev-selected rank) | test depth Spearman %.4f",
             best, sel, summ.loc[best, "test_uuas_best_dev_rank"], summ.loc[best, "test_depth_spearman"])
    if res["best_layer_by_test_uuas"] != best:
        LOG.info("TASK 2 | note: test-argmax UUAS layer is %d", res["best_layer_by_test_uuas"])
    return res


def _fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in m.items())


def _plot(df: pd.DataFrame, summ: pd.DataFrame, best: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for (metric, kind, title), a in zip(
            [("uuas", "distance", "UUAS (distance probe)"),
             ("depth_spearman", "depth", "Depth-probe Spearman (DSpr, lengths 5-50)")], ax):
        sub = df[df.kind == kind]
        for ci, (rank, g) in enumerate(sub.groupby("rank")):
            g, col = g.sort_values("layer"), f"C{ci}"
            a.plot(g.layer, g[f"test_{metric}"], "o-", color=col, label=f"test, k={rank}")
            a.plot(g.layer, g[f"dev_{metric}"], ":", color=col, alpha=0.6, label=f"dev, k={rank}")
        a.axvline(best, color="k", ls="--", lw=1)
        a.text(best, a.get_ylim()[1], f" l*={best}", va="top")
        a.set(title=title, xlabel="BERT layer", xticks=sorted(df.layer.unique()))
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Task 2: structural probe on bert-base-uncased, UD English LinES")
    fig.tight_layout()
    fig.savefig(out / "task2_uuas_spearman_vs_layer.png", dpi=130)
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
