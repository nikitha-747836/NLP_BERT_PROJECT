"""Generate notebooks/Project5_LinES_Colab.ipynb from the repository sources (single source of truth)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
FILES = ["config.yaml", "run_all.py"] + [f"src/p5/{f}" for f in
         ("__init__.py", "utils.py", "data.py", "benchmarks.py", "structural.py", "task1_ppmi.py",
          "features.py", "task2_probe.py", "task3_bert_static.py", "report.py")]

nb = nbf.v4.new_notebook()
C = nb.cells
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C.append(md("""# Project 5 — From Static to Contextual (UD English LinES)
End-to-end, self-contained: **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

| Task | What runs | Approx. time on T4 |
|---|---|---|
| 1 | PPMI+SVD vectors, WordSim-353, Google analogy, 20-sentence MST check | < 1 min |
| 2 | BERT feature extraction (13 layers, cached) + 72 structural probes (12 layers × 3 ranks × 2 probes × 30 epochs) | ~15–40 min |
| 3 | Layer-ℓ* pooled static vectors, same evaluations, significance tests, layer sweep | ~2 min |

Results: `outputs/REPORT.md` (rendered at the end), plots in `outputs/task*/`, probe checkpoints in `checkpoints/`.
Probe checkpoints are reused if the runtime restarts (`task2.resume: true`); mount Drive (optional cell) to survive disconnects."""))
C.append(code("""# 0. GPU check
!nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv || echo "No GPU - will run on CPU (slow)"
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")"""))
C.append(code("""# 1. Dependencies (Colab already has torch/transformers/datasets; this only fills gaps)
!pip -q install "transformers>=4.44,<6" "scikit-learn>=1.3" "pandas>=2.2" pyyaml scipy matplotlib"""))
C.append(code("""# 2. (Optional) persist everything on Google Drive so a disconnect does not lose probe checkpoints
USE_DRIVE = False
import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/nlp_project5'
else:
    PROJECT = '/content/project5'
os.makedirs(PROJECT + '/src/p5', exist_ok=True)
%cd {PROJECT}"""))
C.append(md("## 3. Project source files\n(Generated from the repository — identical to the Docker build.)"))
for f in FILES:
    C.append(code(f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))
C.append(md("## 4. Task 1 — PPMI + SVD baseline"))
C.append(code("!python run_all.py --tasks 1"))
C.append(md("## 5. Task 2 — Structural probe (GPU)"))
C.append(code("!python run_all.py --tasks 2"))
C.append(md("## 6. Task 3 — BERT-derived static vectors vs Task 1"))
C.append(code("!python run_all.py --tasks 3"))
C.append(md("## 7. Results"))
C.append(code("""from IPython.display import Markdown, Image, display
import pandas as pd
display(Markdown(open('outputs/REPORT.md').read().replace('](', '](outputs/')))
for p in ['outputs/dataset/dataset_stats.png', 'outputs/task1/task1_ablation.png',
          'outputs/task2/task2_uuas_spearman_vs_layer.png', 'outputs/task3/task3_vs_task1.png',
          'outputs/task3/task3_layer_sweep.png']:
    display(Image(p))"""))
C.append(code("""# 8. Download results + checkpoints (BERT feature cache excluded - it is ~2 GB and regenerable)
!zip -qr project5_results.zip outputs checkpoints config.yaml
try:
    from google.colab import files; files.download('project5_results.zip')
except ImportError:
    print('Not on Colab: project5_results.zip written to', os.getcwd())"""))
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = ROOT / "notebooks" / "Project5_LinES_Colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
