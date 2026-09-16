# GGPKD — method specification

Subgraph distillation of a teacher's kNN geometry into a compact student. This
file specifies what `main.py --method ggpkd` computes. The paper plan is
`story.md` (§2 method, §3 theory, §5 tables); this file follows it.

Notation: $N$ corpus size, $B$ the anchor batch, $k$ = `graph_k`,
$s^T_{ju}$ and $s^S_{ju}$ teacher and student cosines ($\ell_2$-normalized
embeddings).

---

## 1. Offline graph (once per teacher, corpus, `graph_k`)

Built by `src/ggpkd/graph_builder.py`, cached at `--ggpkd_cache_path`
(artifact version 12).

1. **Encode.** The teacher embeds the deduplicated corpus once (cached at
   `--cache_path`).
2. **Neighbours.** $N(j)$ is the directed top-$k$ of text $j$ under the teacher,
   self excluded. Rows are kept whole: every row has $k$ columns.
3. **Bandwidth.** From the teacher's raw top-$k$ cosines,
   $$\tau_j=\frac{s^{(1)}_j-s^{(k)}_j}{\log k}.$$
   The $k$-th neighbour gets $1/k$ of the nearest neighbour's unnormalized
   weight. Under $s\mapsto as+b$, $\tau_j\mapsto a\tau_j$ and the softmax is
   shift-invariant, so each row is invariant to the teacher's cosine scale.
4. **Target row.**
   $P^T_j(u)=\mathrm{softmax}_{u\in N(j)}\big(s^T_{ju}/\tau_j\big)$.
5. **Holdout (Table 4 only).** `--holdout_edge_frac f` withholds a fraction $f$
   of edges before the rows are normalized. The split is a symmetric hash of the
   unordered pair and `--holdout_seed`, so it is the same across arms and seeds,
   and the evaluation can recompute it. $\tau_j$ is still read off the raw
   top-$k$, so a withheld pair's teacher score shapes its row's targets: the
   default is a holdout of the *target values*, not of the information.
   `--holdout_bandwidth surviving` closes that path and is the arm that says how
   much it mattered.

Graph-build stats go to `<ggpkd_log_dir>/knn_graph_neighbors.jsonl`. They
include `reciprocal_components`, `reciprocal_isolated`,
`reciprocal_largest_frac` and `reciprocity`, all computed on the reciprocal
graph $G^\leftrightarrow$ ($u\sim j$ iff $u\in N(j)$ and $j\in N(u)$) of the
training edges. They also include `target_kl_uniform`, and the build warns
when it falls below 0.05 (rows close to uniform).

---

## 2. Per step

1. Sample $B$ anchors uniformly without replacement (`drop_last`).
2. The pool is $\mathcal P_B=B\cup\bigcup_{i\in B}N(i)$, deduplicated. At $k=100$,
   $B=64$ this is about 4.7k texts.
3. The student encodes the pool once. This forward pass is essentially the
   whole step cost.

---

## 3. Objective

$$\mathcal L=\lambda_{\rm row}\,\mathcal L_{\rm row}+\lambda_{\rm cal}\,\mathcal L_{\rm cal}$$

### 3.1 $\mathcal L_{\rm row}$: every encoded row, local shape

$$\mathcal L_{\rm row}=\frac{1}{|R_B|}\sum_{j\in R_B}\mathrm{KL}\big(P^T_j|_{\Omega_j}\,\|\,P^S_j|_{\Omega_j}\big),\qquad \Omega_j=N(j)\cap\mathcal P_B,\quad R_B=\{j\in\mathcal P_B:|\Omega_j|\ge2\}.$$

- Both sides are softmaxes at $\tau_j$, renormalized on $\Omega_j$.
- Anchors are included. An anchor's $\Omega_i=N(i)$ is its whole row.
- **What it pins (Lemma 1, Proposition 2).** A partial row has zero KL iff
  $s^S_{ju}-s^T_{ju}=a_j$ on $\Omega_j$. A reciprocal edge forces $a_j=a_u$.
  $\mathcal L_{\rm row}$ therefore fixes student cosines on graph edges up to
  one offset per connected component of $G^\leftrightarrow$. No importance
  correction is needed for the fixed points. The objective is still
  inclusion-weighted (§5).

### 3.2 $\mathcal L_{\rm cal}$: cross-neighbourhood levels

$$\mathcal L_{\rm cal}=\frac{1}{|B|}\sum_{i\in B}\mathrm{KL}\big(Q^T_i\,\|\,Q^S_i\big),\qquad Q_i=\mathrm{softmax}_{u\in\mathcal P_B\setminus i}\big(s_{iu}/\bar\tau\big).$$

- One temperature on both sides, $\bar\tau=\mathrm{median}_j\,\tau_j$ (logged as
  `cal_temp`).
- **What it pins.** It is the only term that scores non-edge pairs, and so the
  offsets between reciprocal components. It is SEED-style, and it is the only
  term that depends on the pool.
- With a holdout, withheld pairs are also masked out of $Q_i$.

### 3.3 Derived quantities

| quantity | value | set by |
|---|---|---|
| row columns $\Omega_j$ | $N(j)\cap\mathcal P_B$ | graph and pool |
| supervised rows $R_B$ | pool texts with $\lvert\Omega_j\rvert\ge2$ | graph and pool |
| row temperature $\tau_j$ | $(s^{(1)}_j-s^{(k)}_j)/\log k$ | teacher, `graph_k` |
| calibration temperature $\bar\tau$ | median $\tau_j$ | teacher, `graph_k` |

### 3.4 Knobs

| flag | default | role |
|---|---|---|
| `--graph_k` | 100 | neighbourhood width, and (through $\tau_j$) row sharpness. Choose the smallest $k$ at which $G^\leftrightarrow$ is connected up to a few isolated nodes (Table 5) |
| `--cal_weight` : `--row_weight` | 0.5 : 1.0 | $\lambda_{\rm cal}:\lambda_{\rm row}$. AdamW is nearly scale-invariant, so only the ratio matters (Table 6) |

The training setup is batch 64, 5 epochs, lr 3e-5 (min 3e-6), seed 42. With
no GGPKD flags, a run is the method.

---

## 4. Ablation switches

Each flag changes one thing and leaves the rest of §1–§3 at the method's
value. The table and row numbers refer to `story.md` §5.

| flag | method value | ablation value | changes | story |
|---|---|---|---|---|
| `--method pointwise` | — | — | $1-\cos(Ws_i,t_i)$ with a trained linear map $W$; no relations | T3 row 1 |
| `--batch_sampler` | `random` | `neighbor` | each batch is filled from one teacher graph neighbourhood; step and update counts are unchanged. Also valid with `--method pointwise` | T3 rows 2, 4 |
| `--batch_local` | off | on | pool = the batch; loss is $\mathcal L_{\rm cal}$ only (`row_weight` becomes 0 unless set; a positive value is rejected) | T3 rows 3, 4; T2 |
| `--row_set` | `all` | `anchors` | $\mathcal L_{\rm row}$ over anchor rows only | T3 row 5; T4 |
| | | `non_anchors` | $\mathcal L_{\rm row}$ excludes anchors | T4 last row |
| `--row_target` | `teacher` | `uniform` | same columns, equal mass on each | T3 row 7 |
| | | `shuffled` | same columns and the teacher's own values, permuted among them: support, entropy and the histogram of probabilities are preserved, only which neighbour carries which value is destroyed | T7 |
| `--row_columns` | `graph` | `random` | same width $\lvert\Omega_j\rvert$, columns drawn uniformly from the pool; target is the teacher softmax at $\tau_j$ over them. Not allowed with a holdout | T3 row 8 |
| | | `pool` | every other pool text is a column and every pool text is a row centre (dense relational KD on the same pool); the eligibility rule is dropped, so `row_eff_denom` becomes $\lvert\mathcal P_B\rvert-1$. Not allowed with a holdout | T7 |
| `--pool_source` | `graph` | `random` | the pool is the anchors plus uniformly drawn corpus texts, as many as that batch's graph pool would have held; rows are still $N(j)\cap\mathcal P_B$ under the same eligibility rule. Holds the pool's compute fixed and removes only its structure | T7 |
| `--cal_weight` | 0.5 | 0 | $\mathcal L_{\rm row}$ only | T4 |
| | | 0.25, 1.0 | ratio robustness | T6 |
| `--row_weight` | 1.0 | 0 | $\mathcal L_{\rm cal}$ over the subgraph pool only | T4 |
| `--holdout_edge_frac` | 0 | 0.2 | withhold edges from every term, for the held-out probe | T4 |
| `--holdout_bandwidth` | `full` | `surviving` | recompute $\tau_j$ on the edges the holdout left. Under `full` a withheld pair's teacher score still enters $\tau_j$ (and through the median, $\bar\tau$), which makes the default a *target-only* holdout | T4 |
| `--graph_k` | 100 | 25, 50, 200 | width and sharpness | T5 |
| `--row_reweight` | off | on | weight row $j$ by $1/p_j$, with $p_j=1-\binom{N-\mathrm{indeg}(j)-1}{B}/\binom{N}{B}$ exact (GraphSAINT normalization). Needs `row_set` ≠ `anchors` and the random sampler | T6 |
| `--neighbor_source` | `teacher` | `student` | $N(j)$ from the frozen base student's top-$k$; target values and $\tau_j$ stay the teacher's | T6 |
| `--knn_mode` | `directed` | `mutual` | keep $u\in N(j)$ only if $j\in N(u)$; an emptied row falls back to its raw top-$k$ | T6 |
| `--fixed_bandwidth` | off | on | one temperature, median $\tau_j$, for every row | T6 |

---

## 5. Stated properties

- **Inclusion weighting.** Row $j$ enters a pool with probability
  $p_j=1-\binom{N-\mathrm{indeg}(j)-1}{B}/\binom{N}{B}$, so $\mathcal L_{\rm row}$
  is weighted by $p_j$. `--row_reweight` removes this weighting, and
  `row_ess_ratio` logs the effective-sample-size cost.
- **Anchor coverage.** With `drop_last`, $N \bmod B$ texts are not anchors in a
  given epoch (49 at $N=13{,}553$, $B=64$). Every other row is seen whole once
  per epoch.
- **Logged per step** (`metrics.jsonl`): `loss_total`, `loss_cal`, `loss_row`,
  `loss_cal_weighted`, `loss_row_weighted`, `row_share`, `pool_size`,
  `cal_columns`, `row_count`, `row_eff_denom` (mean $\lvert\Omega_j\rvert$),
  `row_exposed_mass` (mean teacher mass of $N(j)$ inside the pool, with p10/p90),
  `row_teacher_entropy`, `row_kl_p50`/`p90`, `row_ess_ratio`. Table 2's column
  evaluations are `row_count` × `row_eff_denom` plus $B$ × `cal_columns`.

Code: `src/ggpkd/graph_builder.py` (graph, bandwidths, holdout, inclusion
probabilities, reciprocal components), `src/data_utils/ggpkd_dataset.py` (pool
collate), `src/criterions/ggpkd_distillation.py` (loss),
`src/methods/ggpkd.py` (wiring), `config/ggpkd_config.py` (defaults and
validation).
