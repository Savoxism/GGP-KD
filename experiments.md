# Experiments — run plan for `story.md` Tables 1–6

This file lists the runs that fill the six tables in `story.md` §5, and no
others. The method and its flags are in `method.md`.

**No number recorded before the 2026-09-15 code is citable.** Every earlier
result used the student graph, the per-anchor $\mathcal L_{r=1}$ term, anchors
left out of $\mathcal L_{\rm row}$, or an older loss. The `[prov]` values in
`story.md` only size the expected effects. Cite a value only once it has 3
seeds on the current code (`[final]`).

**Common setup.** Setting (c), `qwen3_0_6b_to_minilmv2_h384`: teacher graph,
directed, $k$ = `GRAPH_K` (100 until Table 5 decides), $B=64$, 5 epochs,
lr 3e-5, 1055 optimizer steps. Arms that change $k$, the neighbour source, the
kNN mode, the bandwidth or the holdout each build their own graph artifact.

**Noise.** Seed sd is about 0.08, so the sd of a difference is about 0.11.
Gaps below 0.2 are noise. Paper numbers are mean ± sd over seeds 42, 43, 44.

---

## Scripts and environment

All scripts are in `scripts/exp/`. They share `lib/run_arms.sh` and
`lib/stage_common.sh`. `export_runs.py` collects run trees into CSV, and
`coverage.py` computes exposure statistics.

| variable | meaning | default |
|---|---|---|
| `GPUS` | comma-separated GPU ids | — |
| `SEEDS` | comma-separated seeds; use `42,43,44` for the paper | `42` |
| `GRAPH_K` | operating-point $k$ for every table after Table 5 | 100 |
| `ARMS` | run only the named arms (re-run a failed arm) | all |
| `DRY_RUN=1` | print the arm matrix and exit | off |
| `JOBS_PER_GPU` | concurrent runs per GPU (Table 2 forces 1) | script default |

`scripts/exp/run_all.sh` runs Table 5, then **stops until `GRAPH_K` is set**.
It then runs Tables 3, 4, 4-heldout, 6, 1 and 2.

---

## Table 5 — $k$ and reciprocal connectivity (C3)

**Question.** Does Avg saturate at the $k$ where $G^\leftrightarrow$ becomes
connected? This table also sets the operating point and supplies the default
reference for Tables 3 and 6.

**Script.** `table5_graph_k.sh`. Arms: `--graph_k` ∈ {25, 50, 100, 200}, no
other flags.

**Read.**
- Avg, Avg-In and Avg-Out.
- `reciprocal_components`, `reciprocal_isolated` and `reciprocal_largest_frac`
  from the graph-build log.
- `row_exposed_mass` and `pool_size` from `metrics.jsonl`.
- Peak GPU.

**Rule.**
- **Choosing `GRAPH_K`.** Set it to the smallest $k$ whose Avg is within 0.2
  of the best and whose $G^\leftrightarrow$ is connected up to a few isolated
  nodes. Record the choice before starting Table 3.
- **Falsification of C3.** C3 is falsified if Avg keeps rising after
  $G^\leftrightarrow$ connects.
- **Timing.** Do not report step time from this table; its runs share GPUs.

---

## Table 3 — exposure, not mechanism (C2)

**Question.** Is the gain exposure to graded teacher neighbour rows, or
neighbour batching alone, or knowing which texts are neighbours?

**Script.** `table3_exposure.sh`. No holdout.

| # | arm | flags |
|---|---|---|
| 1 | `pointwise_random` | `--method pointwise` |
| 2 | `pointwise_neighbor` | `--method pointwise --batch_sampler neighbor` (reads the prebuilt teacher graph at `GRAPH_K`) |
| 3 | `in_batch_random` | `--batch_local --row_weight 0` |
| 4 | `in_batch_neighbor` | `--batch_local --row_weight 0 --batch_sampler neighbor` |
| 5 | `anchor_rows` | `--row_set anchors` |
| 6 | `ours` | none |
| 7 | `uniform_target` | `--row_target uniform` |
| 8 | `random_columns` | `--row_columns random` |

**Read.**
- Avg, Avg-In and Avg-Out for every row.
- The "nbrs / row" column: `in_batch_neighbors_per_row` from the neighbour
  sampler for row 4; $(B-1)k/(N-1)$ for row 3; `row_eff_denom` and
  `row_exposed_mass` for rows 5–8.
- Compare the pairs 1→2, 3→4, 5→6 and 6→7.

**Rules.**
- **Falsification of C2.** C2 is falsified by any of: 1→2 gains; 3→4 does not
  gain; 7 ties 6.
- **Stop rule.** If 5 ties 6, C3 loses its payoff and the paper becomes
  C1 + C2.

---

## Table 4 — loss terms and held-out relations (C3)

**Question.** Does $\mathcal L_{\rm row}$ pin within-component geometry, and
$\mathcal L_{\rm cal}$ the cross-component geometry? Does supervising every
encoded row matter?

**Script.** `table4_loss_terms.sh`. Every arm uses
`--holdout_edge_frac 0.2` with the same `--holdout_seed`, so all arms withhold
the same edges. This table has its own `ours` reference: runs without a
holdout (Tables 3, 5, 6) are not comparable here. `anchors_only` is the same
configuration as Table 3 row 5, but a separate run.

| arm | flags (plus `--holdout_edge_frac 0.2`) |
|---|---|
| `ours` | none |
| `cal_off` | `--cal_weight 0` |
| `row_off` | `--row_weight 0` |
| `anchors_only` | `--row_set anchors` |
| `anchors_excluded` | `--row_set non_anchors` |

`anchors_excluded` replaces the "legacy $+\mathcal L_{r=1}$" row in `story.md`.
The $\mathcal L_{r=1}$ term no longer exists in the code. This arm is the
current objective with anchors removed from $\mathcal L_{\rm row}$.

**Held-out probe.** `table4_heldout.sh` runs `heldout_geometry.py` on the five
checkpoints (no training). For each arm it reports the teacher/student
Spearman on withheld pairs, split by whether both endpoints are in the same
component of $G^\leftrightarrow$ built on training edges.

**Read.** Avg, plus held-out ρ on same-component and cross-component pairs.
The prediction is: `cal_off` ≈ `ours` on same-component pairs and below it on
cross-component pairs; `anchors_only` below `ours` on Avg.

**Stop rules.**
- **$\mathcal L_{\rm cal}$ role.** If `cal_off` ties `ours` on cross-component
  pairs, drop $\mathcal L_{\rm cal}$'s role from C3 and present it as a
  calibration add-on.
- **One-term method.** If `cal_off` also ties on Avg, the method drops to one
  term (`story.md` §6).

---

## Table 6 — robustness (C3)

**Question.** Is the default a narrow peak? This is a check, not a search: one
change at a time, and no further points.

**Script.** `table6_robustness.sh`. The default row is Table 5 at `GRAPH_K`.

| arm | flags |
|---|---|
| `cal_0p25` | `--cal_weight 0.25` |
| `cal_1p0` | `--cal_weight 1.0` |
| `row_reweight` | `--row_reweight` |
| `student_knn` | `--neighbor_source student` |
| `mutual_knn` | `--knn_mode mutual` |
| `global_tau` | `--fixed_bandwidth` |

**Read.** Avg and Δ against the default. For `row_reweight`, also
`row_ess_ratio`.

**Rule.** If `row_reweight` beats the default by ≥ 0.2, it becomes the default
(it adds no knob). Re-run the default rows of Tables 1 and 2 with it.

---

## Table 1 — main results (C1)

**Script.** `table1_main.sh`, with pairs `bge_m3_to_minilmv2_h768` (b) and
`qwen3_4b_to_bert_base` (a). Default flags at `GRAPH_K`. Setting (c) is the
Table 5 default run.

**Read.**
- The nine task scores, Avg-In, Avg-Out and Avg, as mean ± sd.
- `reciprocal_components` from each pair's graph build, so that the Table 5
  choice of $k$ can be checked per teacher.
- Checkpoints: the same selection rule in all three settings.

**Rule.** C1 is falsified if Avg does not beat TALAS in at least 2 of the 3
settings. Baseline rows are copied from the TALAS paper and are not re-run.

---

## Table 2 — cost per optimizer step (C1)

**Script.** `table2_cost.sh`. Arms: `pointwise` (`--method pointwise`),
`in_batch` (`--batch_local --row_weight 0`) and `ours`. One job per GPU; the
GPU must not be shared.

**Read.** From the epoch records in `metrics.jsonl`:

| column | source |
|---|---|
| texts encoded | `pool_size` (64 for the batch arms) |
| supervised rows | `row_count` (in-batch: 64) |
| column evaluations | `row_count` × `row_eff_denom` + $B$ × `cal_columns` |
| step (s) | `mean_step_seconds` |
| peak GPU (MiB) | `peak_memory_mb`; cross-check `peak_gpu_mib` in `run_stats.json` |
| Avg | from Table 3 rows 1, 3, 6 |

Offline cost: the graph-build time from the build log, given cached teacher
embeddings.

---

## Budget (3 seeds)

| table | arms | training runs | notes |
|---|---|---|---|
| 5 | 4 | 12 | 4 graph builds |
| 3 | 8 | 24 | 21 if `ours` is reused from Table 5 at `GRAPH_K` |
| 4 | 5 | 15 | holdout artifact; nothing reusable from other tables |
| 4-heldout | — | 0 | post-hoc on Table 4 checkpoints |
| 6 | 6 | 18 | 3 extra graph builds (student, mutual, fixed bandwidth) |
| 1 | 2 pairs | 6 | two teacher caches; (a) is the expensive pair |
| 2 | 3 | 9 | timing needs 1 seed (3 runs), unshared GPU |
| **total** | | **84** | 81 with reuse; 75 with a 1-seed Table 2 |

Setting (c) runs take about 5 min each on an unshared GPU. Peak memory for
`ours` grows with $k$ (roughly 14 GiB at $k=100$ for H384). Check memory for
BERT-base in (a) before launching Table 1.

---

## Order

```
Table 5   graph_k sweep            12   -> set GRAPH_K; stop
Table 3   exposure                 24   \
Table 4   loss terms (holdout)     15    | parallel across GPUs
Table 4   held-out probe            0   /  after Table 4 finishes
Table 6   robustness               18
Table 1   pairs (a), (b)            6
Table 2   cost, unshared GPU        9
```

Apply the Table 3 and Table 4 stop rules before spending on Tables 6 and 1.
Either rule changes what the paper claims, but not which runs Tables 1 and 2
need.
