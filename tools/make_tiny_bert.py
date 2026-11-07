"""Build a tiny, randomly initialised BERT (12 layers, 32 dims) + WordPiece vocab, offline.

Used only by config.smoke.yaml to exercise the whole pipeline without downloading
bert-base-uncased. Its scores are meaningless.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import collections  # noqa: E402

from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from p5.data import load_lines  # noqa: E402
from p5.utils import load_config, setup_logging  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/tiny-bert")
setup_logging()
cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml", {"data.source": "github"})
data = load_lines(cfg)
cnt = collections.Counter(t.lower() for s in data["train"] for t in s.tokens)
chars = sorted({c for w in cnt for c in w})
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars] \
        + [w for w, _ in cnt.most_common(3000) if len(w) > 1]
vocab = list(dict.fromkeys(vocab))
out.mkdir(parents=True, exist_ok=True)
(out / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)
tok.save_pretrained(out)
BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=12, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512)).save_pretrained(out)
print("tiny BERT saved to", out, "vocab", len(vocab))
