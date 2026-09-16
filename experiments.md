# Experiments — run plan for `story.md` Tables 1–8

Tables 1–6 fill the result tables of the run plan; Tables 7 and 8 are the
attribution controls and the budget curve `story.md` §6 asks for, and they are
what decides which of the claims survives. The method and its flags are in
`method.md`.

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
It then runs Tables 3, 7, 4, 4-heldout, 6, 1, 2 and 8.

---

## One command

`launch.sh` runs the whole plan detached: preflight, Table 5, the `graph_k`
choice (`scripts/exp/select_graph_k.py`), then Tables 3, 7, 4, 4h, 6, 1, 2, 8 at
that `k`.

```bash
GPUS=4,5 JOBS_PER_GPU=2 bash scripts/exp/launch.sh start                 # pilot: SEEDS=42, 40 runs
PROFILE=paper GPUS=4,5 JOBS_PER_GPU=2 bash scripts/exp/launch.sh start   # SEEDS=42,43,44, 106 runs
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
| `holdout_strict` | `--holdout_bandwidth surviving` |

`holdout_strict` is the method again, under a holdout that also keeps the
withheld teacher scores out of $\tau_j$. With the default (`full`) the split
withholds the *target values* but not the information: $\tau_j=(s^{(1)}-s^{(k)})
/\log k$ is read off the raw top-$k$, so a withheld pair still shapes every
target in its row, and $\bar\tau$ is the median of those. It trains on its own
artifact, `graph_main_holdout_strict.pt`; the withheld pairs are the same hash of
(pair, seed, fraction), so `table4_heldout.sh` scores it on the same relations.
Read it as a validity check on the probe, not as a method arm: if it matches
`ours` on the held-out columns, the leak did not matter and the default numbers
stand; if it does not, the held-out columns have to be reported from this arm and
the others described as target-only.

`anchors_excluded` replaces the "legacy $+\mathcal L_{r=1}$" row in `story.md`.
The $\mathcal L_{r=1}$ term no longer exists in the code. This arm is the
current objective with anchors removed from $\mathcal L_{\rm row}$.

**Held-out probe.** `table4_heldout.sh` runs `heldout_geometry.py` on the six
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

## Table 7 — what the gain is attributable to (`story.md` §6.2)

**Question.** Three competing explanations for the same number: the teacher's
graded values, encoding 4.7k texts instead of 64, and relational KD in general.
Each arm removes exactly one of them.

**Script.** `table7_controls.sh`. No holdout. The reference row is Table 3's
`ours`: same graph key, same $k$, no holdout, so it is not re-run.

| arm | flags | removes |
|---|---|---|
| `shuffled_target` | `--row_target shuffled` | the assignment of teacher values to neighbours, keeping the support, the entropy and the histogram |
| `random_pool` | `--pool_source random` | the pool's structure, keeping its size — the same texts-per-step, drawn uniformly |
| `dense_rows` | `--row_columns pool` | the sparsity: every pool text is a centre against every other |
| `dense_rows_only` | `--row_columns pool --cal_weight 0` | the same, as a one-term objective (with $\mathcal L_{\rm cal}$ on, each anchor's pool row is supervised twice at two temperatures) |

**Read.** Avg against Table 3's `ours` and `uniform_target`, plus `row_count`,
`row_eff_denom` (pair evaluations per row) and `row_exposed_mass`.

**Rules (`story.md` §10).**
- **Graded values.** Graded > uniform but ≈ shuffled means the ordering of the
  values is not evidenced; entropy is the competing explanation and the claim
  narrows to "a peaked local target helps".
- **Sampler.** `random_pool` ≈ `ours` means the neighbourhood sampler does not
  belong at the centre of a causal claim; the honest statement is then about the
  pool's size, which is §2.6's confound.
- **Sparsity.** `dense_rows` ≈ `ours` is not a loss: it moves the claim from
  quality to cost, and `row_eff_denom` is what makes that claim measurable. Report
  it that way rather than dropping the arm.

**Caveat to carry into the text.** In both `shuffled_target` and
`uniform_target`, $\mathcal L_{\rm cal}$ still carries the teacher's values over
the pool. These arms measure the *incremental* contribution of graded row
targets, not a student that never saw teacher grading.

---

## Table 8 — quality against budget (`story.md` §6.1, P2)

**Question.** Every other table fixes the optimizer steps, which is not a fixed
budget: a step of the method encodes a subgraph pool, a step of the baselines
encodes 64 texts. Do the baselines catch up when they spend what they saved?

**Script.** `table8_budget.sh`. One seed by default (`BUDGET_SEEDS` for more):
this is a curve, and its longest arms cost four ordinary runs each.

| arm | flags |
|---|---|
| `ours_e5` | `--epochs 5` (the method's budget) |
| `pointwise_e{5,10,20}` | `--method pointwise --epochs E` |
| `in_batch_e{5,10,20}` | `--batch_local --epochs E` |

`BUDGET_EPOCHS` sets the ladder. The arms are not budget-*matched* — matching the
method's tokens would take roughly 70× the epochs — they trace the curve the
matched point would sit on.

**Read.** Avg against `encoded_tokens_cum` (the budget axis), with
`encoded_texts_cum` and `wall_seconds` beside it. Step time and peak memory come
from Table 2, which owns an unshared GPU; this table may share GPUs because its
axis is tokens.

**Rule.** A win on steps that does not survive on tokens or GPU-hours is a
trade-off to report, not training efficiency to claim.

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
| 7 | 4 | 12 | no new graph; reference row comes from Table 3 |
| 4 | 6 | 18 | 2 holdout artifacts (the strict arm rebuilds $\tau_j$) |
| 4-heldout | — | 0 | post-hoc on Table 4 checkpoints |
| 6 | 6 | 18 | 3 extra graph builds (student, mutual, fixed bandwidth) |
| 1 | 2 pairs | 6 | two teacher caches; (a) is the expensive pair |
| 2 | 3 | 9 | timing needs 1 seed (3 runs), unshared GPU |
| 8 | 7 | 7 | one seed; the 20-epoch arms cost ~4 runs each, so ~19 run-equivalents |
| **total** | | **106** | 103 with reuse; 97 with a 1-seed Table 2 |

Setting (c) runs take about 5 min each on an unshared GPU. Peak memory for
`ours` grows with $k$ (roughly 14 GiB at $k=100$ for H384). Check memory for
BERT-base in (a) before launching Table 1.

---

## Order

```
Table 5   graph_k sweep            12   -> set GRAPH_K; stop
Table 3   exposure                 24   \
Table 7   attribution controls     12    | parallel across GPUs
Table 4   loss terms (holdout)     18    |
Table 4   held-out probe            0   /  after Table 4 finishes
Table 6   robustness               18
Table 1   pairs (a), (b)            6
Table 2   cost, unshared GPU        9
Table 8   budget curve, 1 seed      7
```

Apply the Table 3, Table 7 and Table 4 stop rules before spending on Tables 6
and 1. They change what the paper claims, not which runs Tables 1 and 2 need.

`story.md` §10 argues for a different order: check the baselines' provenance,
then run Tables 7 and 3 (all-rows vs anchors, graded vs shuffled, random pool),
then Table 4, and leave the `k` sweep until after the claims are settled. The
scripts allow it — every table except 5 takes `GRAPH_K` as a given, so
`GRAPH_K=100 FROM=3 TO=7 bash scripts/exp/run_all.sh` runs the controls first at
a provisional $k$ — but then Table 5 has to run before any number is reported at
the chosen operating point.
