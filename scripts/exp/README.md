# Experiments

One script per table of `story.md` §5. Every training arm runs through the same
runner (`lib/run_arms.sh`) and `scripts/ggpkd/train.sh`, so two arms that differ
in one flag differ in nothing else.

## One command (server)

`launch.sh` runs the whole plan detached: preflight, Table 5, the `graph_k`
choice (`select_graph_k.py`), then Tables 3, 4, 4h, 6, 1, 2 at that `k`.

```bash
GPUS=4,5 JOBS_PER_GPU=2 bash scripts/exp/launch.sh start                 # pilot: SEEDS=42, 28 runs
PROFILE=paper GPUS=4,5 JOBS_PER_GPU=2 bash scripts/exp/launch.sh start   # SEEDS=42,43,44, 84 runs
bash scripts/exp/launch.sh status          # phase, graph_k, completed/failed runs, log tail
bash scripts/exp/launch.sh stop            # TERM the whole process group
GRAPH_K=100 GPUS=4,5 bash scripts/exp/launch.sh start   # skip the choice
```

The `graph_k` rule: the larger of (a) the smallest `k` whose mean Avg is within
`TOLERANCE` (0.2) of the best `k`, and (b) the smallest `k` whose reciprocal graph
is about connected (`graph_reciprocal_largest_frac >= MIN_LARGEST_FRAC`, 0.995).
Everything the controller needs is pinned in `runs/launch/<run_id>/env.sh`; the
phases, the chosen `k` and the exit code sit beside it. The launcher warns when
the working tree is dirty: commit first, so `SOURCE_COMMIT` names the code that ran.

## Order

Table 5 chooses the operating point `k` and every other table runs at it, so it
goes first. The orchestrator stops after Table 5 on purpose: choosing `k` is a
judgement call and no script may guess it.

```bash
GPUS=0,1,2,3              bash scripts/exp/run_all.sh   # Table 5, then stop
column -s, -t runs/table5_graph_k/results.csv | less -S
GRAPH_K=<k> GPUS=0,1,2,3  bash scripts/exp/run_all.sh   # Tables 3, 4, 4h, 6, 1, 2
DRY_RUN=1 GRAPH_K=100     bash scripts/exp/run_all.sh   # print the plan only
```

`FROM` / `TO` restrict the range (`5`, `3`, `4`, `4h`, `6`, `1`, `2`). On the
second pass Table 5 is skipped when its CSV already holds `graph_k_<GRAPH_K>`,
and otherwise run at that one `k`, because that run is Table 6's default row and
Table 1's pair (c).

| table | script | arms | runs / seed |
|---|---|---|---|
| 5 | `table5_graph_k.sh` | `graph_k_25`, `graph_k_50`, `graph_k_100`, `graph_k_200` | 4 |
| 3 | `table3_exposure.sh` | `pointwise_random`, `pointwise_neighbor`, `in_batch_random`, `in_batch_neighbor`, `anchor_rows`, `ours`, `uniform_target`, `random_columns` | 8 |
| 4 | `table4_loss_terms.sh` | `ours`, `cal_off`, `row_off`, `anchors_only`, `anchors_excluded` | 5 |
| 4 (held-out) | `table4_heldout.sh` | post-hoc on Table 4's checkpoints | 0 |
| 6 | `table6_robustness.sh` | `cal_0p25`, `cal_1p0`, `row_reweight`, `student_knn`, `mutual_knn`, `global_tau` | 6 |
| 1 | `table1_main.sh` | `ours` on each of `PAIRS` | 1 per pair |
| 2 | `table2_cost.sh` | `pointwise`, `in_batch`, `ours` | 3 |

What each arm changes, relative to the method with no flags:

* **Table 5** — `--graph_k k`, one graph per `k`. Rule: the smallest `k` at which
  Avg saturates and the reciprocal graph is about connected
  (`graph_reciprocal_components`, `graph_reciprocal_isolated`).
* **Table 3** — `--method pointwise` (± `--batch_sampler neighbor`);
  `--batch_local` (± `--batch_sampler neighbor`); `--row_set anchors`;
  `--row_target uniform`; `--row_columns random`. After training it runs
  `coverage.py` on the graph and writes `runs/table3_exposure/coverage_<pair>.csv`:
  the measured in-batch neighbours per row under random and neighbour batching,
  beside the analytic `(B-1)k/(N-1)`.
* **Table 4** — every arm trains on graph `main_holdout`
  (`--holdout_edge_frac 0.2 --holdout_seed 12345`, env `HOLDOUT_FRAC` /
  `HOLDOUT_SEED`): `--cal_weight 0`, `--row_weight 0`, `--row_set anchors`,
  `--row_set non_anchors`. Its `ours` arm is the reference, not Table 5's.
* **Table 4 held-out** — `heldout_geometry.py` over the latest Table 4 run tree
  (or `RUN_ROOT`) against `graph_main_holdout.pt`. The two columns are `spearman`
  on `heldout_same_component` and `heldout_cross_component`: held-out edges split
  by whether both endpoints share a component of the reciprocal graph built on
  the training edges. Writes `runs/table4_heldout/heldout_geometry.csv`.
* **Table 6** — `--cal_weight 0.25`, `--cal_weight 1.0`, `--row_reweight`, and
  three graph variants with their own keys: `student` (`--neighbor_source
  student`), `mutual` (`--knn_mode mutual`), `median_tau` (`--fixed_bandwidth`).
  The default row is Table 5's `graph_k_<GRAPH_K>`; the script warns if it is
  missing.
* **Table 1** — `PAIRS` defaults to `bge_m3_to_minilmv2_h768,qwen3_4b_to_bert_base`;
  `qwen3_0_6b_to_minilmv2_h384` comes from Table 5. Every pair exports into the
  same `runs/table1_main/results.csv` with merge, and each pair gets its own
  result tree `results/table1_main/<pair>/<run_id>`.
* **Table 2** — `JOBS_PER_GPU` is forced to 1 so step time and peak memory are
  read on unshared GPUs.

## Environment

| variable | default | meaning |
|---|---|---|
| `GRAPH_K` | required (not Table 5) | operating point from Table 5 |
| `GPUS` | every GPU `nvidia-smi` lists | comma-separated device ids |
| `SEEDS` | `42` | `42,43,44` for paper numbers |
| `PAIR` | `qwen3_0_6b_to_minilmv2_h384` | teacher->student pair (Tables 2-6) |
| `PAIRS` | see Table 1 | pairs for Table 1 |
| `GRAPH_KS` | `25,50,100,200` | Table 5 sweep |
| `CORPUS` | `data/train_set/merged_3_data_5k_each.csv` | training CSV |
| `CACHE_ROOT` | `cache/exp` | teacher caches and graphs, `<pair>/<corpus>/graph_<key>.pt` |
| `JOBS_PER_GPU` | `1` | concurrent runs per GPU (ignored by Table 2) |
| `ARMS` | all | comma-separated subset of a table's arms |
| `RUN_ID`, `RESULT_BASE`, `CSV_OUT` | timestamp, `results/<table>`, `runs/<table>/results.csv` | output locations |
| `DRY_RUN` | `0` | `1` prints the matrix and runs nothing |

Graph-defining flags (`--graph_k --neighbor_source --knn_mode --fixed_bandwidth
--holdout_edge_frac --holdout_seed`) belong in a table's `GRAPH_SPEC`, never in
an arm's flags: they change the cached artifact, and the runner builds each graph
once, serially, before any arm starts.

## Re-running one arm

`ARMS=a,b` runs only those arms of a table and builds only the graphs they use.
The export merges by `(experiment, pair, arm, seed)`, so the new rows replace the
old ones in the same `results.csv` and every other row is kept.

```bash
GRAPH_K=100 GPUS=0 ARMS=uniform_target bash scripts/exp/table3_exposure.sh
GRAPH_K=100 GPUS=0 ARMS=student_knn    bash scripts/exp/table6_robustness.sh
GRAPH_K=100 GPUS=0 SEEDS=43 ARMS=cal_off bash scripts/exp/table4_loss_terms.sh
bash scripts/exp/table4_heldout.sh     # rescore after any Table 4 re-run
```

## Exporting

`run_arms` exports automatically when every run in a sweep succeeds, and writes
the run root to `runs/<table>/last_run_root.txt`. To export a tree by hand — a
sweep with a failed run, or one synced back from a cluster:

```bash
.venv/bin/python scripts/exp/export_runs.py results/table3_exposure/<run_id> \
    --experiment table3_exposure --pair qwen3_0_6b_to_minilmv2_h384 \
    --out runs/table3_exposure/results.csv --merge
```

One row per (arm, seed):

* `score_*` — nine benchmarks plus Avg In / Avg Out / Avg All, in points.
* `cfg_*` — read from each run's own `run.json`, with the graph fields taken from
  the artifact the run trained on, so a mislabelled arm is visible.
* `train_*` — final-epoch means of the criterion's metrics (`loss_cal`,
  `loss_row`, `pool_size`, `row_count`, `row_exposed_mass`, ...) and the
  distiller's cost counters (`encoded_texts_cum`, `mean_step_seconds`,
  `peak_memory_mb`).
* `graph_*` — reciprocal connectivity and target sharpness of that artifact.
  Blank for pointwise runs, which build no graph.
* `cost_*` — wall clock and peak host/GPU memory measured outside the process.
* `geom_*` — the final-epoch geometry probe.
* `status` — `ok`, or why the row is incomplete. A partial sweep still exports.

## Things to check before trusting a number

* **`graph_target_kl_uniform`.** Near zero means the rows went flat: too large a
  `k` measures the distance out of the neighbourhood, not the local decay.
* **`holdout_starved_rows` in Table 4's graph build log.** A row whose every edge
  was withheld keeps its nearest neighbour; `heldout_geometry.py` drops that pair
  from the held-out sets.
* **`n_pairs` for `heldout_cross_component`.** At `k=100` the reciprocal graph has
  few components, so cross-component held-out pairs are rare. Under 100 pairs the
  metrics are left blank rather than reported on noise; `knn_recall` also needs
  more than `--knn-k` positives per anchor, so read `spearman` and `pair_order`
  there.
* **Pointwise arms do not deduplicate the corpus**; GGPKD does. They use separate
  teacher caches (`teacher_pointwise.pt` vs `teacher_train.pt`) for that reason.
