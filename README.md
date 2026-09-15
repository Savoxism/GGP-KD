# GGPKD: Subgraph Distillation of Text Embedding Geometry

This repository implements GGPKD, a relational knowledge distillation method
that transfers a large teacher embedding model's local geometry to a compact
student. Each step, the student encodes neighbourhoods of the teacher's kNN
graph. It is then supervised on the teacher's neighbour row of every text it
encoded, plus one pool-wide calibration term. `method.md` is the method
specification, `experiments.md` the run plan, and `story.md` the paper plan.

## Supported Distillation Pairs

The framework supports arbitrary teacher-student pairs. Tested configurations include:

| Pair key | Teacher | Student |
|:---|:---|:---|
| `qwen3_0_6b_to_minilmv2_h384` | `Qwen/Qwen3-Embedding-0.6B` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base` |
| `bge_m3_to_minilmv2_h768` | `BAAI/bge-m3` | `nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base` |
| `qwen3_4b_to_bert_base` | `Qwen/Qwen3-Embedding-4B` | `google-bert/bert-base-uncased` |

`qwen3_0_6b_to_minilmv2_h384` is the default in `config/ggpkd_config.py`. The
launchers take the pair key as their first argument and derive the model names,
pooling, cache and log paths from it, so two pairs never share a cache file.

The training corpus follows the TALAS paper setup: ~15K unlabeled sentences sampled from three in-domain datasets. The default corpus is `data/train_set/merged_3_data_5k_each.csv`.

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Method

Full specification: `method.md`.

### 1. Offline graph

The teacher embeds the deduplicated corpus once, and the embeddings are
$\ell_2$-normalized and cached. Each text $j$ keeps its directed teacher top-$k$
$N(j)$ (`graph_k`, default 100) whole. Its row temperature is read off the
teacher's top-$k$ span:

$$\tau_j=\frac{s^{(1)}_j-s^{(k)}_j}{\log k},\qquad P^T_j(u)=\mathrm{softmax}_{u\in N(j)}\big(s^T_{ju}/\tau_j\big).$$

Each row is invariant to affine rescaling of the teacher's cosines. The
artifact (version 12) stores rows, probabilities and bandwidths. It also logs
the reciprocal kNN graph's connectivity: `reciprocal_components`,
`reciprocal_isolated`, `reciprocal_largest_frac` and `reciprocity`.

### 2. Per step

Anchors $B$ are sampled uniformly. The pool is $\mathcal P_B=B\cup\bigcup_{i\in B}N(i)$,
deduplicated (~4.7k texts at $k=100$, $B=64$), and the student encodes it once.

### 3. Objective

$$\mathcal L=\lambda_{\rm row}\,\mathcal L_{\rm row}+\lambda_{\rm cal}\,\mathcal L_{\rm cal}$$

- $\mathcal L_{\rm row}$ is the mean over every pool text $j$ with
  $|\Omega_j|\ge2$, anchors included, of
  $\mathrm{KL}(P^T_j|_{\Omega_j}\,\|\,P^S_j|_{\Omega_j})$. Here
  $\Omega_j=N(j)\cap\mathcal P_B$, and both sides are softmaxes at $\tau_j$
  renormalized on $\Omega_j$. A partial row is an exact constraint, so this term
  pins student cosines on graph edges up to one offset per reciprocal-graph
  component.
- $\mathcal L_{\rm cal}$ is, per anchor, the KL between teacher and student
  softmaxes over the whole pool at $\bar\tau=\mathrm{median}_j\tau_j$. It is the
  only term that scores non-edge pairs, and it fixes the offsets between
  components.

The method has two knobs: `graph_k`, and the ratio `cal_weight : row_weight`
(default 0.5 : 1.0). Columns, rows and temperatures are derived.

## Configuration

Defaults live in `config/ggpkd_config.py`. With no GGPKD flags, a run is the
method. Every other value is an ablation switch for one row of `story.md`
Tables 3–6; `method.md` §4 maps each switch to its row.

| Group | Flags | Default | Description |
|:---|:---|:---|:---|
| Graph | `--graph_k` | 100 | kNN width and, through $\tau_j$, row sharpness |
| Graph | `--neighbor_source` | `teacher` | `student`: neighbours from the frozen base student; targets stay the teacher's |
| Graph | `--knn_mode` | `directed` | `mutual`: keep reciprocal neighbours only |
| Graph | `--fixed_bandwidth` | off | one median $\tau$ for every row |
| Graph | `--holdout_edge_frac`, `--holdout_seed` | 0, 12345 | withhold a symmetric edge subset from all training terms (held-out probe) |
| Objective | `--cal_weight`, `--row_weight` | 0.5, 1.0 | $\lambda_{\rm cal}$, $\lambda_{\rm row}$ |
| Row ablations | `--row_set` | `all` | `anchors` or `non_anchors` |
| Row ablations | `--row_target` | `teacher` | `uniform`: equal mass on the same columns |
| Row ablations | `--row_columns` | `graph` | `random`: same width, random pool columns |
| Row ablations | `--row_reweight` | off | weight rows by $1/P(\text{text in pool})$ (GraphSAINT) |
| Batch ablations | `--batch_local` | off | pool = batch, $\mathcal L_{\rm cal}$ only (implies `row_weight` 0) |
| Batch ablations | `--batch_sampler` | `random` | `neighbor`: one graph neighbourhood per batch; also valid for `--method pointwise` |
| Paths | `--cache_path`, `--ggpkd_cache_path`, `--ggpkd_log_dir`, `--pooling_method` | per pair | teacher cache, graph artifact, graph log, teacher pooling |
| Training | `--batch_size`, `--epochs`, `--lr`, `--seed` | 64, 5, 3e-5, 42 | standard setup (`min_lr` 3e-6) |

`config.validate()` rejects incoherent combinations. For example, row switches
require `row_weight > 0`, `--row_columns random` cannot be combined with a
holdout, and `--row_reweight` needs the random sampler.

## Training

### GGPKD

```bash
source venv/bin/activate
bash scripts/ggpkd/train.sh                           # default pair
bash scripts/ggpkd/train.sh bge_m3_to_minilmv2_h768   # another pair
bash scripts/ggpkd/train.sh qwen3_4b_to_bert_base --graph_k 50 --seed 43
```

Extra arguments are passed through to `main.py`. The launcher reads `SEED`,
`GPU`, `BATCH_SIZE`, `EPOCHS`, `LR`, `MAX_LENGTH`, `CACHE_PATH`,
`GGPKD_CACHE_PATH`, `GGPKD_LOG_DIR`, `SAVE_DIR`, `WEIGHTS_DIR` and `PYTHON_BIN`.
`METHOD=pointwise` reuses the same launcher, pair table and paths for the
pointwise baseline.

```bash
WEIGHTS_DIR="/path/to/weights" bash scripts/ggpkd/train.sh
```

Arms that change the graph (`--graph_k`, `--neighbor_source`, `--knn_mode`,
`--fixed_bandwidth`, `--holdout_edge_frac`) need their own `GGPKD_CACHE_PATH`.
The artifact metadata is checked on load. A mismatch triggers a rebuild that
overwrites the file, so two arms sharing one path rebuild each other's graph.

### Using Python directly

```bash
python3 main.py \
  --method ggpkd \
  --train_data data/train_set/merged_3_data_5k_each.csv \
  --student_model nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base \
  --teacher_model Qwen/Qwen3-Embedding-0.6B \
  --pooling_method last_token \
  --cache_path cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt \
  --ggpkd_cache_path cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/graph_k100.pt \
  --ggpkd_log_dir logs/ggpkd/qwen3_0_6b_to_minilmv2_h384 \
  --batch_size 64 --epochs 5 --lr 3e-5 --seed 42 \
  --save_dir models/ggpkd/qwen3_0_6b_to_minilmv2_h384/default
```

### Paper experiments

`scripts/exp/` has one script per table of `story.md` §5. `experiments.md`
gives the arms, stop rules and budget for each.

| Script | Table | Arms |
|:---|:---|:---|
| `table5_graph_k.sh` | 5 | `graph_k` ∈ {25, 50, 100, 200} |
| `table3_exposure.sh` | 3 | `pointwise_random`, `pointwise_neighbor`, `in_batch_random`, `in_batch_neighbor`, `anchor_rows`, `ours`, `uniform_target`, `random_columns` |
| `table4_loss_terms.sh` | 4 | holdout 0.2: `ours`, `cal_off`, `row_off`, `anchors_only`, `anchors_excluded` |
| `table4_heldout.sh` | 4 | post-hoc held-out Spearman split by reciprocal component (`heldout_geometry.py`) |
| `table6_robustness.sh` | 6 | `cal_0p25`, `cal_1p0`, `row_reweight`, `student_knn`, `mutual_knn`, `global_tau` |
| `table1_main.sh` | 1 | pairs `bge_m3_to_minilmv2_h768`, `qwen3_4b_to_bert_base` |
| `table2_cost.sh` | 2 | `pointwise`, `in_batch`, `ours`; one job per GPU |
| `run_all.sh` | all | Table 5, stop until `GRAPH_K` is set, then 3, 4, 4-heldout, 6, 1, 2 |

```bash
DRY_RUN=1 GPUS=0 bash scripts/exp/table3_exposure.sh          # print the arm matrix
GPUS=0,1,2 SEEDS=42,43,44 bash scripts/exp/table5_graph_k.sh
GRAPH_K=100 GPUS=0,1,2 SEEDS=42,43,44 bash scripts/exp/run_all.sh
GRAPH_K=100 GPUS=0 ARMS=cal_off bash scripts/exp/table4_loss_terms.sh   # re-run one arm
```

Environment: `GPUS`, `SEEDS` (default `42`; use `42,43,44` for the paper),
`GRAPH_K`, `ARMS`, `DRY_RUN=1`, `JOBS_PER_GPU`. The shared runner is
`scripts/exp/lib/run_arms.sh` with `lib/stage_common.sh`. `export_runs.py`
collects run trees into CSV, and `coverage.py` computes exposure statistics.

### RKD baseline

RKD uses the paper-default RKD-DA objective (distance weight 1, angle weight 2,
no task loss) and the original metric-learning optimizer schedule (Adam,
batch 128, 80 epochs, learning rate `1e-4`, decays at epochs 40 and 60). Teacher
embeddings are cached before student training.

```bash
bash scripts/rkd/train.sh
```

Select another supported teacher-student pair or prepare only its cache:

```bash
bash scripts/rkd/train.sh bge_m3_to_minilmv2_h768
bash scripts/rkd/train.sh qwen3_4b_to_bert_base --prepare-cache
```

For a quick smoke run, override the expensive paper defaults through environment
variables:

```bash
BATCH_SIZE=16 EPOCHS=1 bash scripts/rkd/train.sh
```

### Run cost: time and memory

Every launcher wraps its `main.py` invocation in `scripts/common/run_stats.sh`, so
each run writes a `run_stats.json` next to its own outputs (`SAVE_DIR` /
`RUN_DIR`) and echoes a one-line summary into its log:

```
[stats] wall 1:47:12 (6432.4s) | peak host RSS 11.20 GiB | peak GPU 18.60 GiB | exit 0 -> .../run_stats.json
```

The file records `wall_seconds`, `peak_host_rss_mib`, `peak_gpu_mib`,
`exit_code`, the start/finish timestamps, the pinned device and the exact
command. Both figures are measured from outside the process over the whole
tree -- dataloader workers included -- so a run that dies mid-epoch still leaves
its numbers behind. Host memory comes from the kernel's own high-water mark
(`VmHWM`), GPU memory from per-process `nvidia-smi` accounting sampled every
`RUN_STATS_POLL_SECONDS` (default 2), which is what bounds the GPU spike a run
can hide; `peak_gpu_mib` is `null` where `nvidia-smi` is unavailable. Set
`RUN_STATS_DISABLE=1` to run the command bare, or `STATS_FILE` to redirect the
file.

The per-step cost used in Table 2 comes from the training loop itself:
`mean_step_seconds`, `peak_memory_mb` and `encoded_texts_cum` in each epoch
record of `metrics.jsonl`.

## Outputs

Model checkpoints are saved under the configured `save_dir`. Training metrics are
written to `metrics.jsonl` in the same directory. For GGPKD they include
`loss_total`, `loss_cal`, `loss_row`, `row_share`, `pool_size`, `row_count`,
`row_eff_denom`, `row_exposed_mass` and `row_ess_ratio`.

Teacher embedding and graph caches are written to `cache/ggpkd/<pair_key>/`. If you
change the training corpus or teacher model, delete that pair's caches before
retraining:

```bash
rm -f cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/*.pt
```

kNN graph diagnostics, including the reciprocal-component stats, are logged to
`<ggpkd_log_dir>/knn_graph_neighbors.jsonl`.

## Benchmarks

The training loop evaluates on 9 benchmarks after each epoch:

| Family | Benchmarks |
|:---|:---|
| Classification | Banking77, Emotion, Tweet |
| Pair Classification | MRPC, SciTail, WiC |
| Semantic Textual Similarity | SICK, STS12, STS-B |

Validation runs after each epoch. Test evaluation runs once after training.

## Adding a Method

`distiller.py` never compares `config.distill_method` against a method name. It
reads a `MethodSpec` from `src/methods/`, whose flags answer the questions the
shared pipeline asks and whose hooks replace the parts that differ:

```python
SPEC = MethodSpec(
    name="ggpkd",
    config_cls=GGPKDConfig,
    step=step,                  # src/distill/steps/ggpkd.py
    uses_teacher_cache=True,    # precomputed teacher embeddings
    batch_relational=True,      # -> DataLoader(drop_last=True)
    prepare_frame=prepare_frame,
    build_data=build_data,
    build_criterion=build_criterion,
    on_epoch_start=on_epoch_start,
)
```

Every hook defaults to `None`, meaning "use the shared path", so a spec carries
only what actually differs. A new method is one module under `src/methods/` plus
one line in its `REGISTRY`; `--method` choices and the config class follow from
there. `src/methods/spec.py` documents each field.

## Other Distillation Methods

This repository also includes implementations of other distillation baselines for comparison:

- **TALAS** (`config/talas_config.py`, `scripts/talas/train.sh`)
- **RKD** (`config/rkd_config.py`, `scripts/rkd/train.sh`)
- **CDM** (`config/cdm_config.py`, `scripts/cdm/train.sh`)
- **DSKD** (`config/dskd_config.py`, `scripts/dskd/train.sh`)
- **EMO** (`config/emo_config.py`, `scripts/emo/train.sh`)
- **Stella** (`config/stella_config.py`, `scripts/stella/train.sh`)
- **Pointwise KD** (`config/pointwise_config.py`, `METHOD=pointwise scripts/ggpkd/train.sh`):
  $1-\cos(Ws_i,t_i)$ with a trained linear map; the floor of Table 3.
  `--batch_sampler neighbor` reads a prebuilt GGPKD graph artifact.

## Project Structure

```
.
├── main.py                          # Entry point
├── distiller.py                     # Method-agnostic training loop and evaluation
├── method.md                        # GGPKD specification
├── experiments.md                   # Run plan for the paper tables
├── story.md                         # Paper plan: claims, method, tables
├── config/
│   ├── base_config.py               # Shared defaults, override checking, validate()
│   ├── ggpkd_config.py              # GGPKD defaults and ablation switches
│   └── ...                          # Other method configs
├── src/
│   ├── methods/                     # One module per method: what it *is*
│   │   ├── spec.py                  # MethodSpec: the flags and hooks
│   │   ├── __init__.py              # REGISTRY -- the only list of methods
│   │   └── ggpkd.py, talas.py, ...  # Per-method data, criterion, optimizer, KD term
│   ├── criterions/
│   │   ├── ggpkd_distillation.py    # L_row and L_cal
│   │   └── ...                      # Other method losses
│   ├── ggpkd/
│   │   ├── graph_builder.py         # kNN graph, bandwidths, holdout, reciprocal stats
│   │   └── policy.py                # Numerical and runtime constants
│   ├── distill/
│   │   ├── steps/                   # One training step per shape of step
│   │   └── ...                      # Checkpointing, telemetry, geometry, benchmarks
│   ├── data_utils/                  # Datasets, pool collate, batch samplers
│   ├── models/                      # Students that are not a plain AutoModel
│   ├── evaluation/                  # Benchmark evaluation
│   ├── cache_teacher.py             # Teacher embedding caching
│   ├── pooling.py                   # Pooling strategies
│   └── loss.py                      # Shared loss utilities
├── scripts/
│   ├── <method>/train.sh            # Bash launcher per method
│   ├── common/run_stats.sh          # Wall clock + peak host/GPU memory
│   ├── ggpkd/run_metrics.py         # Run-tree reader used by exp/export_runs.py
│   ├── exp/table{1,2,3,4,5,6}_*.sh  # One script per paper table (see above)
│   ├── exp/table4_heldout.sh        # Post-hoc held-out probe
│   ├── exp/run_all.sh               # Tables in order, with the GRAPH_K stop
│   ├── exp/heldout_geometry.py      # Held-out teacher geometry on checkpoints
│   ├── exp/lib/                     # run_arms.sh, stage_common.sh
│   └── talas/                       # + run_paper.sh, summarize.py
├── data/                            # Train/val/test CSV datasets
├── tests/                           # pytest suite
├── docs/                            # Reference papers, LaTeX, result tables
```

The earlier sweep scripts (`scripts/exp/stage*`, `scripts/exp/exp1`–`exp4`, the
GGPKD `run_paper`/sensitivity/calibration-gate helpers), the Colab notebook
and `docs/paper_flow.md` have been removed.
