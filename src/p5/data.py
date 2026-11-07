
from __future__ import annotations

import collections
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .utils import LOG, save_json

SPLITS = ("train", "dev", "test")
HF_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


@dataclass
class Sentence:
    sid: str
    tokens: list[str]
    heads: list[int]          # 1-based head index per token, 0 = root
    deprels: list[str]
    _dist: np.ndarray | None = field(default=None, repr=False)
    _depth: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def punct_mask(self) -> np.ndarray:
        """True for punctuation tokens (deprel == punct), excluded from UUAS as in Hewitt & Manning."""
        return np.array([d.split(":")[0] == "punct" for d in self.deprels], dtype=bool)

    @property
    def distances(self) -> np.ndarray:
        if self._dist is None:
            self._dist = tree_distances(self.heads)
        return self._dist

    @property
    def depths(self) -> np.ndarray:
        if self._depth is None:
            root = self.heads.index(0)
            self._depth = self.distances[root].copy()
        return self._depth



def tree_distances(heads: list[int]) -> np.ndarray:
    """All-pairs path length in the undirected dependency tree (BFS from every node)."""
    n = len(heads)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, h in enumerate(heads):
        if h > 0:
            adj[i].append(h - 1)
            adj[h - 1].append(i)
    dist = np.full((n, n), -1, dtype=np.int32)
    for s in range(n):
        dist[s, s] = 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s, v] < 0:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    if (dist < 0).any():
        raise ValueError("Dependency graph is disconnected")
    return dist


def is_valid_tree(heads: list[int]) -> tuple[bool, str]:
    n = len(heads)
    if n == 0:
        return False, "empty"
    if any(h < 0 or h > n for h in heads):
        return False, "head out of range"
    if sum(h == 0 for h in heads) != 1:
        return False, "root count != 1"
    for i in range(n):  # cycle check: walk to root
        seen, cur = set(), i
        while heads[cur] != 0:
            if cur in seen:
                return False, "cycle"
            seen.add(cur)
            cur = heads[cur] - 1
    return True, ""


def is_projective(heads: list[int]) -> bool:
    arcs = [(min(i + 1, h), max(i + 1, h)) for i, h in enumerate(heads) if h > 0]
    for a1, b1 in arcs:
        for a2, b2 in arcs:
            if a1 < a2 < b1 < b2:
                return False
    return True



def parse_conllu(path: Path) -> list[Sentence]:
    sents, toks, heads, rels, sid = [], [], [], [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            elif line.startswith("#"):
                continue
            elif not line.strip():
                if toks:
                    sents.append(Sentence(sid or f"s{len(sents)}", toks, heads, rels))
                toks, heads, rels, sid = [], [], [], None
            else:
                cols = line.split("\t")
                if len(cols) != 10:
                    raise ValueError(f"Malformed CoNLL-U line in {path}: {line[:80]!r}")
                if "-" in cols[0] or "." in cols[0]:  # multiword-token ranges / empty nodes
                    continue
                toks.append(cols[1])
                heads.append(int(cols[6]))
                rels.append(cols[7])
    if toks:
        sents.append(Sentence(sid or f"s{len(sents)}", toks, heads, rels))
    return sents


def _from_hf_rows(rows, split: str) -> list[Sentence]:
    out = []
    for k, r in enumerate(rows):
        keep = [i for i, h in enumerate(r["head"]) if str(h).isdigit()]  # drop MWT/empty rows if any
        out.append(Sentence(
            str(r.get("idx") or f"{split}-{k}"),
            [r["tokens"][i] for i in keep],
            [int(r["head"][i]) for i in keep],
            [str(r["deprel"][i]) for i in keep],
        ))
    return out


def _load_hf(cfg: dict) -> dict[str, list[Sentence]] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    name, conf = cfg["data"]["hf_name"], cfg["data"]["hf_config"]
    attempts = [
        ("hf-script", dict(trust_remote_code=True)),
        ("hf-parquet", dict(revision="refs/convert/parquet")),
    ]
    for tag, kw in attempts:
        try:
            ds = load_dataset(name, conf, **kw)
            out = {s: _from_hf_rows(ds[HF_SPLIT[s]], s) for s in SPLITS if HF_SPLIT[s] in ds}
            if "train" in out and "test" in out:
                LOG.info("Loaded LinES from HuggingFace (%s)", tag)
                out["_source"] = tag  # type: ignore[assignment]
                return out
        except Exception as e:  # noqa: BLE001 - any failure -> next source
            LOG.warning("HF load (%s) failed: %s", tag, str(e).splitlines()[0][:160])
    return None


def _load_github(cfg: dict, data_dir: Path) -> dict[str, list[Sentence]]:
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in SPLITS:
        fn = raw / f"en_lines-ud-{s}.conllu"
        if not fn.is_file() or fn.stat().st_size == 0:
            url = cfg["data"]["github_url"].format(split=s)
            LOG.info("Downloading %s", url)
            try:
                urllib.request.urlretrieve(url, fn)
            except Exception as e:
                fn.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download {url} ({e}). Place the CoNLL-U file manually at {fn}."
                ) from e
        out[s] = parse_conllu(fn)
    LOG.info("Loaded LinES from UD GitHub CoNLL-U files (%s)", raw)
    out["_source"] = "ud-github-" + cfg["data"]["github_url"].split("/")[-2]  # type: ignore[assignment]
    return out


def load_lines(cfg: dict) -> dict[str, list[Sentence]]:
    """Return {'train','dev','test'} -> list[Sentence], validated. Cached in-process."""
    key = id(cfg)
    if key in _CACHE:
        return _CACHE[key]
    data_dir = Path(cfg["paths"]["data_dir"])
    src = cfg["data"].get("source", "auto")
    data = None
    if src in ("auto", "hf"):
        data = _load_hf(cfg)
        if data is None and src == "hf":
            raise RuntimeError("HuggingFace loading requested but failed")
    if data is None:
        data = _load_github(cfg, data_dir)
    source = data.pop("_source")
    for s in list(data):
        valid, bad = [], collections.Counter()
        for sent in data[s]:
            ok, why = is_valid_tree(sent.heads)
            if ok and len(sent.tokens) == len(sent.heads) == len(sent.deprels):
                valid.append(sent)
            else:
                bad[why or "length mismatch"] += 1
        if bad:
            LOG.warning("%s: dropped %d invalid sentences %s", s, sum(bad.values()), dict(bad))
        data[s] = valid
        LOG.info("%-5s : %5d sentences, %6d tokens", s, len(valid), sum(map(len, valid)))
    data["_source"] = source  # type: ignore[assignment]
    _CACHE[key] = data
    return data


_CACHE: dict[int, dict] = {}



def dataset_report(data: dict, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"source": data["_source"], "splits": {}}
    train_set = {tuple(s.tokens) for s in data["train"]}
    for s in SPLITS:
        sents = data[s]
        lens = np.array([len(x) for x in sents])
        toks = [t for x in sents for t in x.tokens]
        seqs = [tuple(x.tokens) for x in sents]
        stats["splits"][s] = {
            "sentences": len(sents),
            "tokens": int(lens.sum()),
            "types_lowercased": len({t.lower() for t in toks}),
            "len_mean": float(lens.mean()), "len_median": float(np.median(lens)),
            "len_min": int(lens.min()), "len_max": int(lens.max()),
            "duplicate_sentences_within_split": len(seqs) - len(set(seqs)),
            "sentences_also_in_train": (sum(q in train_set for q in seqs) if s != "train" else None),
            "non_projective": int(sum(not is_projective(x.heads) for x in sents)),
            "punct_token_ratio": float(np.mean([p for x in sents for p in x.punct_mask])),
            "max_tree_depth_mean": float(np.mean([x.depths.max() for x in sents])),
        }
    rel = collections.Counter(d.split(":")[0] for x in data["train"] for d in x.deprels)
    stats["train_deprel_top15"] = rel.most_common(15)
    save_json(stats, out_dir / "dataset_stats.json")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for s in SPLITS:
        ax[0].hist([len(x) for x in data[s]], bins=40, alpha=0.5, label=s, density=True)
    ax[0].set(title="Sentence length (tokens)", xlabel="tokens", ylabel="density")
    ax[0].legend()
    names, counts = zip(*rel.most_common(15))
    ax[1].barh(names[::-1], counts[::-1], color="#4C72B0")
    ax[1].set(title="Train: top-15 dependency relations", xlabel="count")
    for s in SPLITS:
        ax[2].hist([x.depths.max() for x in data[s]], bins=range(0, 20), alpha=0.5, label=s, density=True)
    ax[2].set(title="Max tree depth per sentence", xlabel="depth")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_stats.png", dpi=130)
    plt.close(fig)
    LOG.info("Dataset stats -> %s", out_dir / "dataset_stats.json")
    for s in SPLITS:
        d = stats["splits"][s]
        LOG.info("  %-5s %5d sents | len %.1f±(max %d) | dup %d | in-train %s | non-proj %d",
                 s, d["sentences"], d["len_mean"], d["len_max"],
                 d["duplicate_sentences_within_split"], d["sentences_also_in_train"], d["non_projective"])
    return stats
