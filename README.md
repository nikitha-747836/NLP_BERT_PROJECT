# Project 5 — From Static to Contextual: Distributional Semantics and What's Inside BERT

**Assigned dataset:** UD English LinES (`universal_dependencies`, config `en_lines`).
All three tasks use LinES only (train for vectors/probe training, dev for layer selection, test for UUAS and the 20-sentence check).

## Run it

### Option A — Docker, one command (GPU)
Requirements: Docker + NVIDIA driver ≥ 550 + NVIDIA Container Toolkit. The image pins the tested versions in `requirements-docker.txt` (torch 2.5.1 from the base image, transformers 4.46.3).
```bash
./run.sh                    # builds the image, detects the GPU, runs Tasks 1-3, writes outputs/REPORT.md
./run.sh --tasks 1          # any run_all.py arguments pass through
# equivalent: docker compose up --build
```
Without the NVIDIA runtime `run.sh` runs on CPU (works, but Task 2 takes hours).

### Option B — Google Colab
Open `notebooks/Project5_LinES_Colab.ipynb` in Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*.
The notebook is self-contained (it writes the same source files the Docker image uses) and renders `REPORT.md` and all plots at the end.

### Option C — local venv (Ubuntu / VS Code)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python run_all.py                      # all tasks
python run_all.py --tasks 1            # Task 1 only
python run_all.py --tasks 2            # Task 2 only (needs nothing from Task 1)
python run_all.py --tasks 3            # needs Task 1 vectors + Task 2 best layer
python run_all.py --device cpu         # force CPU
```




## What was executed vs. not (honest status)

| Component | Status |
|---|---|
| Data loading, validation, statistics (real LinES r2.7) | **Executed** |
| Task 1 end-to-end on real data | **Executed** — results below and in `outputs/` |
| Task 2 + Task 3 code paths (extraction, 13-layer cache, probe training, resume, MST/UUAS, l* selection, pooling, PCA, layer sweep, significance tests, report) | **Executed** with a tiny randomly initialised 12-layer BERT (`config.smoke.yaml`), on CPU, with both transformers 5.22 and 4.46.3. Results were bit-identical across reruns. |
| Task 2 + Task 3 with real `bert-base-uncased` on a GPU | **Not executed** — the build sandbox had no GPU and could not reach huggingface.co. Run `./run.sh` or the Colab notebook to produce them. |
| Docker image build | **Not executed** — Docker Hub was unreachable from the sandbox; the image's Python steps were verified separately with the same pinned versions. |
| Colab notebook | Generated from the repo; validated that every `%%writefile` cell is byte-identical to the tested sources. |

### Actual Task 1 results (LinES train, 57,372 tokens, 7,606 types)
| Metric | Value |
|---|---|
| WordSim-353 Spearman, covered pairs | **0.1621** (116/353 pairs covered) |
| WordSim-353 Spearman, all 353 (OOV sim = 0) | −0.0424 |
| Google analogy accuracy, all 19,544 | 0.0007 |
| Google analogy accuracy, covered (2,374 questions) | 0.0059 |
| MST UUAS vs gold, 20 test sentences | **0.1273** (41/322 edges) |
| Reference: linear-chain / random-tree UUAS | 0.4348 / 0.1184 |

Low absolute numbers are expected: LinES train is ~57k tokens (the brief notes this trade-off). Cosine-MST over PPMI vectors is barely above random trees.

### What to expect from Tasks 2–3 (illustrative, not measured here)
Hewitt & Manning found syntax most linearly recoverable in BERT's middle layers; expect the UUAS-vs-layer curve to rise from layer 1, peak in the middle layers, and fall towards layer 12, with test UUAS well above the 0.43 linear-chain baseline. Terminal output per probe looks like:
```
L07 distance k=128 | loss 0.xxxx | dev uuas=0.xxxx dist_dspr=0.xxxx | test uuas=0.xxxx dist_dspr=0.xxxx | 12.3s
TASK 2 | l* = <layer> (selected on dev) | test UUAS 0.xxxx (dev-selected rank) | test depth Spearman 0.xxxx
TASK 3 | WordSim-353 Spearman: BERT-L<l*> 0.xxxx vs PPMI 0.1621 (higher/lower, delta ..., p=...)
```
Runtime on a T4: ~5 min BERT extraction, ~15–40 min for the 72 probes, ~2 min Task 3.

## Layout
```
run_all.py                 entry point (Task 1 -> 2 -> 3 -> REPORT.md)
config.yaml                every hyperparameter and path
config.smoke.yaml          tiny-random-BERT config used to test the pipeline
requirements.txt           loose deps for venv/Colab; requirements-docker.txt = exact pins for the image
src/p5/
  data.py                  LinES loading (HF -> HF parquet -> UD GitHub r2.7), validation, stats, tree distances
  benchmarks.py            WordSim-353 + Google analogy (shared protocol for Tasks 1 and 3)
  structural.py            Prim MST, UUAS (Hewitt & Manning convention), 20-sentence sample, baselines
  task1_ppmi.py            co-occurrence -> PPMI -> SVD-100 -> evaluations
  features.py              frozen BERT, subword-averaged word vectors for all 13 layers, cached
  task2_probe.py           distance + depth probes, 12 layers x 3 ranks, UUAS / DSpr, l* selection
  task3_bert_static.py     layer-l* pooled static vectors, same evaluations, significance, layer sweep
  report.py                builds outputs/REPORT.md from results files
tools/                     tiny-BERT builder, notebook generator
notebooks/                 Colab notebook
Dockerfile, docker-compose.yml, run.sh
```
Generated: `dataset/` (downloads; originals never modified), `cache/` (BERT features, ~2 GB), `checkpoints/` (Task 1 vectors, 72 probe checkpoints, Task 3 vectors), `outputs/` (results, plots, `REPORT.md`, `run.log`).

## Design decisions (and why)

| Item | Choice | Reason |
|---|---|---|
| Data loading | Try `load_dataset("universal_dependencies","en_lines")`, then its parquet export, then UD GitHub **r2.7** | `datasets>=4` removed script datasets, so the prescribed call fails on current installs; r2.7 is the release the HF dataset wraps. Source is logged in `outputs/dataset/dataset_stats.json`. |
| Tokens | `tokens/head/deprel` only; lower-cased word types; all tokens kept (min count 1) | Benchmarks are lower-case; BERT is uncased; every test-sentence word that occurs in train gets a vector. |
| Co-occurrence | symmetric ±5, within sentence, unweighted counts | Spec. |
| PPMI / SVD | plain PPMI (α=1), TruncatedSVD (randomized, seeded) 100-d, vectors = UΣ | Spec. Ablation over α∈{1,0.75} and Σ-power∈{1,0.5} reported separately. |
| WordSim-353 | Spearman on covered pairs (main) + on all 353 with OOV sim=0 | LinES (57k tokens) covers only part of WS-353; both numbers are reported, coverage is identical for Tasks 1 and 3. |
| Analogy | 3CosAdd, the 3 query words excluded; accuracy over all 19,544 (OOV=wrong) and over covered questions; semantic/syntactic split | Spec. |
| UUAS | Prim MST on non-punctuation words, gold edges between non-punctuation words, micro-averaged | Hewitt & Manning's evaluation. Same function for the probe and the static-vector checks. |
| Subwords | mean of WordPiece vectors per word | As in the structural-probes code. |
| Probe | B ∈ R^{k×768}, init U(−0.05,0.05); L1 loss / n² (distance), / n (depth); Adam 1e-3, batch 20, 30 epochs, final epoch kept | Spec + paper. |
| Depth metric | Spearman per sentence, averaged per length 5–50, macro over lengths (DSpr); root accuracy also reported | Paper's metric. |
| ℓ* | argmax **dev** UUAS (max over ranks), tie-break dev depth Spearman; test-argmax also reported | Selecting on test would leak test information into Task 3. Set `task2.select_on: test` for the literal reading. |
| Mixed precision | bf16 (Ampere+) / fp16 autocast for BERT inference only; probes in fp32 | Probes are tiny; precision matters more than speed there. |
| Batch size | BERT extraction batch auto-sized from VRAM (16/32/64) | Avoids OOM on T4, uses big GPUs. |
| Task 3 | mean over all train occurrences of each word type at ℓ*; also PCA-100 (dimension-matched) and a layer sweep 0–12 | Same vocabulary as Task 1 by construction, so the comparison uses exactly the same benchmark items and sentences. |
| Significance | paired bootstrap (WS-353), exact McNemar (analogy), Wilcoxon over the 20 sentences (MST) | 20 sentences and ~100 WS pairs are small; differences need uncertainty. |

## Outputs per task

**Task 1** — `outputs/task1/results.json`, `ablation.csv`, `task1_ablation.png`; vectors in `checkpoints/task1/` (`vectors.npy`, `vocab.txt`, `vectors.w2v.txt`).
**Task 2** — `outputs/task2/probe_results.csv` (72 rows: layer, probe, rank, dev/test metrics), `layer_summary.csv`, `best_layer.json`, `results.json`, `loss_curves.json`, `task2_uuas_spearman_vs_layer.png`; probes in `checkpoints/task2/{distance,depth}_LXX_rK.pt`.
**Task 3** — `outputs/task3/comparison.csv`, `layer_sweep.csv`, `results.json` (incl. significance and generated verdict), `task3_vs_task1.png`, `task3_layer_sweep.png`; vectors in `checkpoints/`.
**All** — `outputs/REPORT.md`, `outputs/run.log`, `outputs/dataset/dataset_stats.{json,png}`, `outputs/structural_sample.json` (the 20 test sentences, shared by Tasks 1 and 3).

### Interpreting the metrics
- **WordSim-353 Spearman**: rank agreement with human similarity; 1 = same ordering. Values from a 57k-token corpus are low; compare models on the same covered pairs.
- **Analogy accuracy**: exact top-1 match. Most questions are OOV in LinES, so `acc_all` is near zero for any model; `acc_covered` is the informative number.
- **UUAS**: fraction of gold undirected edges recovered by the MST. Reference points are printed: linear chain (adjacent-word tree) and random tree.
- **Depth Spearman**: does the probe order words by tree depth correctly.

## Requirement checklist

| Brief requirement | Where |
|---|---|
| T1: word-word co-occurrence, train raw words, symmetric window 5 | `task1_ppmi.cooccurrence` |
| T1: PPMI weighting | `task1_ppmi.ppmi` (α=1) |
| T1: truncated SVD to 100 d | `task1_ppmi.svd_vectors` |
| T1: WordSim-353 Spearman over 353 pairs | `benchmarks.eval_wordsim` (covered + all-353) |
| T1: Google analogy, vector offset, query words excluded | `benchmarks.eval_analogy` |
| T1 bonus: 20 test sentences, cosine-distance MST, undirected attachment vs gold | `structural.static_vector_check` |
| T2: layers 1–12 of bert-base-uncased, B ∈ R^{k×768}, k ∈ {32,64,128} | `config.yaml: task2`, `task2_probe.LinearProbe` |
| T2: distance + depth probes from gold `head` | `data.tree_distances`, `Sentence.depths` |
| T2: Adam 1e-3, batch 20, 30 epochs, train split | `task2_probe.train_probe` |
| T2: MST decoding, UUAS on gold test trees | `task2_probe.evaluate` |
| T2: plot UUAS and depth Spearman vs layer, ℓ* | `task2_uuas_spearman_vs_layer.png`, `best_layer.json` |
| T3: layer-ℓ* embeddings for every train occurrence, averaged per type | `task3_bert_static.pool_static` |
| T3: same WordSim/analogy protocol | shared `benchmarks.py` |
| T3: same 20-sentence check, compared with T1 | shared `structural_sample.json` |
| T3: discussion of both evaluations and agreement | `verdict` in `task3/results.json`, REPORT.md |
| LinES only; train/test as specified | `data.py` |
| Docker one-command / Colab end-to-end | `run.sh`, `docker-compose.yml`, `notebooks/` |

## Verification checklist after a run
1. `outputs/run.log` contains `GPU: <name> | VRAM ...` and `Loaded bert-base-uncased on cuda:0`.
2. `checkpoints/task2/` contains 72 `.pt` files.
3. `outputs/task2/best_layer.json` exists; Task 3 log starts with `Using BERT layer l* = <same>`.
4. `outputs/REPORT.md` has all three sections and five figures.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Install NVIDIA Container Toolkit, `sudo systemctl restart docker`. |
| `Dataset scripts are no longer supported` in log | Expected on `datasets>=4`; the loader falls back automatically. |
| No internet to huggingface.co | Pre-download `bert-base-uncased` into a folder and set `bert.model_name` to that path. |
| Benchmark download fails | Put `wordsim353.tsv` / `questions-words.txt` in `dataset/` or set `benchmarks.*_path`. |
| CUDA OOM during extraction | Set `bert.extract_batch_size: 8`. |
| Colab disconnected mid Task 2 | Re-run; finished probes are loaded from `checkpoints/` (`task2.resume: true`). Use the Drive cell to persist. |
| Changed data/model but stale results | Delete `cache/` and `checkpoints/`. |

## References
Hewitt & Manning, *A Structural Probe for Finding Syntax in Word Representations*, NAACL 2019. Levy & Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*, NeurIPS 2014. Bommasani, Davis & Cardie, *Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings*, ACL 2020.
