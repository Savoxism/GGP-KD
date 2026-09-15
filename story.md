# GGPKD — the story (v2: subgraph distillation)

Three claims, the method that serves them, and the tables that decide each one.
This replaces story v1 and the scope–computation framing (the deleted `docs/paper_flow.md`).

**Decisions this rests on (2026-09-15).** The per-anchor full-row term
$\mathcal L_{r=1}$ is removed. Row truncation is removed. The graph is the
teacher's kNN.

**Status tags.**
- `[final]`: 3 seeds on the method in §2; citable.
- `[prov]`: 1 seed, an older objective, or the student graph; never cited.
- `TBD`: not run.

---

## 1. Story

> Relational distillation of text embeddings is **starved, not biased**. With
> $N=13{,}553$, $B=64$, $k=100$, a batch-local row compares an anchor with 63
> texts, of which only $(B-1)k/(N-1)\approx 0.46$ are its neighbours. Those rows
> are valid, just nearly empty of local structure. Because a softmax row stays
> exact on any subset of its columns, the fix is to encode **neighbourhoods** and
> supervise the teacher's neighbour row of **every text encoded**.

Working title: *Partial Rows, Exact Constraints: Subgraph Distillation of Text
Embedding Geometry.* Rename GGPKD before submission; the story makes no claim to
"global geometry preservation".

| | claim | falsified if (3 seeds) | tables |
|---|---|---|---|
| **C1 — Result** | Subgraph distillation gives the best compact student in all three teacher→student settings, at a stated per-step encoder cost | Avg does not beat TALAS in ≥ 2 of 3 settings | 1, 2 |
| **C2 — Mechanism** | The gain is exposure to **graded teacher neighbour rows**. Neighbour batching alone does not produce it, and neither does knowing which texts are neighbours | neighbour batching without row targets helps; batch-local KD does not improve when batches contain neighbours; uniform targets tie graded ones | 3 |
| **C3 — Principle** | Partial rows are exact constraints, so every encoded text should be a supervised row. $\mathcal L_{\rm row}$ fixes local geometry, $\mathcal L_{\rm cal}$ fixes the rest, and $k$ is set by when the reciprocal graph connects | all-rows ties anchor-rows-only; $\lambda_{\rm cal}=0$ ties on cross-component held-out pairs; score keeps rising after the reciprocal graph connects | 4, 5, 6 |

---

## 2. Method

**Offline, once per (teacher, corpus).**
- Encode the corpus with the teacher and $\ell_2$-normalize.
- $N(j)$ is the directed top-$k$ of text $j$, kept whole.
- $\tau_j=(s^{(1)}_j-s^{(k)}_j)/\log k$.
- $P^T_j=\mathrm{softmax}_{u\in N(j)}(s^T_{ju}/\tau_j)$.

**Per step.**
- Sample anchors $B$ without replacement.
- The pool is $\mathcal P_B=B\cup\bigcup_{i\in B}N(i)$, deduplicated
  (~4.7k texts at $k=100$).
- The student encodes the pool once.

**Objective.**

$$\mathcal L=\lambda_{\rm row}\,\underbrace{\frac{1}{|R_B|}\sum_{j\in R_B}\mathrm{KL}\big(P^T_j|_{\Omega_j}\,\|\,P^S_j|_{\Omega_j}\big)}_{\mathcal L_{\rm row}:\ \text{every encoded row, local shape}}
\;+\;\lambda_{\rm cal}\,\underbrace{\frac{1}{|B|}\sum_{i\in B}\mathrm{KL}\big(Q^T_i\,\|\,Q^S_i\big)}_{\mathcal L_{\rm cal}:\ \text{cross-neighbourhood levels}}$$

- **$\mathcal L_{\rm row}$.**
  - $\Omega_j=N(j)\cap\mathcal P_B$ and $R_B=\{j\in\mathcal P_B:|\Omega_j|\ge2\}$.
  - Both sides are softmaxes at $\tau_j$, renormalized on $\Omega_j$.
  - Anchors are included. An anchor's $\Omega_i=N(i)$ is its whole row, so the
    removed $\mathcal L_{r=1}$ is a special case of $\mathcal L_{\rm row}$.
- **$\mathcal L_{\rm cal}$.**
  $Q_i=\mathrm{softmax}_{u\in\mathcal P_B\setminus i}(s_{iu}/\bar\tau)$ with
  $\bar\tau=\mathrm{median}_j\tau_j$. SEED-style, over the pool.
- **Knobs.** Two: `graph_k` and the ratio $\lambda_{\rm cal}:\lambda_{\rm row}$
  (default 0.5 : 1). Columns, rows and temperatures are derived.

---

## 3. Why it works (the theory behind C3)

**Lemma 1 (partial rows are exact).** For $\Omega\subseteq N(j)$ with
$|\Omega|\ge 2$,
$\mathrm{KL}(P^T_j|_\Omega\|P^S_j|_\Omega)=0 \iff s^S_{ju}-s^T_{ju}=a_j$ for all
$u\in\Omega$.
- **Subsets combine.** Constants agree on overlapping subsets, and a row is seen
  whole whenever its text is an anchor (every epoch, except the $N \bmod B = 49$
  texts `drop_last` skips in that epoch). The joint zero set therefore equals
  full-row matching.
- **No importance correction.** Subset sampling changes weights, not fixed
  points. Sampled softmax and top-$K$ logit caching are biased only because their
  targets are not renormalized.
- **Support.** McFadden (1978); Hajek, Oh & Xu (NeurIPS'14).

**Proposition 2 (what each term pins).** Cosine is symmetric, so a reciprocal
edge ($u\in N(j)$ and $j\in N(u)$) forces $a_j=a_u$.
- **$\mathcal L_{\rm row}$ pins graph edges.** The student's cosines equal the
  teacher's on every graph edge, up to **one offset per connected component of
  the reciprocal kNN graph $G^\leftrightarrow$**. A global offset changes no
  ranking.
- **$\mathcal L_{\rm cal}$ pins the rest.** Offsets between components and all
  non-edge pairs are left free by $\mathcal L_{\rm row}$. $\mathcal L_{\rm cal}$
  rows span 64 neighbourhoods and constrain exactly those pairs.
- **Consequences.**
  - Choose $k$ as the smallest value at which $G^\leftrightarrow$ is connected up
    to a few isolated nodes.
  - Held-out edges inside a component should be recovered without
    $\mathcal L_{\rm cal}$; pairs across components should need it.

**Stated, not hidden.** Row $j$ enters a pool with probability
$p_j=1-\binom{N-\mathrm{indeg}(j)-1}{B}\big/\binom{N}{B}$ (exact), so the objective is
inclusion-weighted. On the student graph at $k=100$,
$\max p/\mathrm{median}\,p=3.0$ and the effective number of rows is $0.75N$.
GraphSAINT's $1/p_j$ correction is an ablation (Table 6).

---

## 4. Prior art we credit, and what is ours

| component | prior art | ours |
|---|---|---|
| teacher kNN offline, neighbours put in the batch | BINGO (ICLR'22), CoSS (2409.13939), TAS-B, B3 | they supervise anchors only, pointwise or contrastively |
| KL between SNE-style conditionals | PKT (ECCV'18), parametric t-SNE (AISTATS'09) | fixed corpus rows restricted to a random subset, renormalized |
| KL on graph-neighbour rows | LSP (CVPR'20) | theirs processes the whole graph; ours samples it |
| dense similarity KL over a pool ($\mathcal L_{\rm cal}$) | SEED (ICLR'21), CompRess (NeurIPS'20) | nothing; presented as theirs |
| per-point temperature | self-tuning spectral clustering, UMAP | closed form only |
| subgraph batches and their bias | GraphSAINT, Cluster-GCN | we ablate their correction |

- **Ours:** the partial-row lemma and reciprocal-component identifiability;
  supervising every encoded row; the exposure decomposition (Table 3).
- **Not claimed:** global geometry preservation; unbiased estimation of a uniform
  corpus loss; that the teacher selects important relations.

---

## 5. Evidence

**Setup for all ablations.** Setting (c), Qwen3-0.6B → MiniLMv2-H384: $k=100$,
$B=64$, 5 epochs, lr 3e-5, teacher graph.

**Provisional numbers.** `[prov]` values come from
`runs/ggp_kd_run_all_20260915_results`: 1 seed, student graph, with
$\mathcal L_{r=1}$ present unless stated.

**Noise.** Two runs of the same config agree to 0.02. Seed sd is about 0.08, so
the sd of a difference is about 0.11. Gaps below 0.2 are noise.

### C1 — Result

**Table 1. Main results.**
- **Metrics.** Classification F1, pair-classification AP, STS Spearman. $^\star$
  marks in-domain tasks; Avg is the mean over nine tasks.
- **Baselines.** Rows are copied from `docs/latex/tables/main_results.tex`
  (source: TALAS paper). TALAS is mean±sd over 3 seeds.
- **Legacy GGPKD.** The pre-2026-09-13 objective scored Avg 78.43 / 77.46 /
  75.58 on (a) / (b) / (c). It is not citable and only sizes the target.

| Method | Banking77 | Tweet | Emotion$^\star$ | MRPC | SciTail | WiC$^\star$ | SICK | STS12 | STS-B$^\star$ | Avg-In | Avg-Out | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ***(a) Qwen3-Embedding-4B → BERT-base (109M)*** | | | | | | | | | | | | |
| Teacher | 93.63 | 72.22 | 68.94 | 84.79 | 91.77 | 66.80 | 83.43 | 82.55 | 86.87 | 74.20 | 84.73 | 81.22 |
| Student base | 84.86 | 47.79 | 68.02 | 77.36 | 67.74 | 59.22 | 42.43 | 21.54 | 20.30 | 42.44 | 60.33 | 54.36 |
| SimCSE-unsup | 89.21 | 69.13 | 53.55 | 82.86 | 76.59 | 71.64 | 68.19 | 61.68 | 70.36 | 65.18 | 74.61 | 71.47 |
| CDM | 89.75 | 70.58 | 53.11 | 84.83 | 75.19 | 71.06 | 69.37 | 68.48 | 73.92 | 66.03 | 76.37 | 72.92 |
| DSKD | 89.66 | 69.93 | 54.51 | 83.13 | 77.95 | 71.53 | 68.65 | 63.37 | 71.42 | 65.82 | 75.45 | 72.24 |
| Jasper & Stella | 89.71 | 73.83 | 65.98 | 85.33 | 80.99 | 70.50 | 73.96 | 67.90 | 75.65 | 70.71 | 78.62 | 75.98 |
| DistillCSE | 90.94 | 70.90 | 60.80 | 82.86 | 78.98 | 68.63 | 75.12 | 72.61 | 77.08 | 68.84 | 78.57 | 75.32 |
| EMO | 90.78 | 73.54 | 67.48 | 84.46 | 81.57 | 69.76 | 75.66 | 68.83 | 77.17 | 71.47 | 79.14 | 76.58 |
| TALAS | 90.44±0.13 | 74.62±0.17 | 70.87±0.52 | 85.90±0.10 | 81.78±0.29 | 68.84±0.15 | 77.56±0.10 | 73.12±0.30 | 79.24±0.10 | 72.98±0.19 | 80.57±0.01 | 78.04±0.07 |
| **Ours** | TBD | | | | | | | | | | | |
| ***(b) BGE-M3 → MiniLMv2-H768 (66M)*** | | | | | | | | | | | | |
| Teacher | 93.52 | 73.85 | 68.56 | 85.81 | 91.87 | 61.47 | 79.18 | 78.73 | 84.87 | 71.63 | 83.83 | 79.76 |
| Student base | 81.84 | 47.70 | 71.67 | 79.67 | 65.06 | 60.89 | 49.46 | 26.76 | 24.12 | 44.24 | 62.41 | 56.35 |
| SimCSE-unsup | 87.89 | 71.45 | 55.65 | 83.87 | 76.32 | 67.96 | 66.93 | 59.66 | 63.97 | 62.53 | 74.35 | 70.41 |
| CDM | 89.15 | 70.48 | 58.71 | 84.16 | 75.91 | 70.00 | 68.14 | 62.74 | 67.78 | 65.50 | 75.10 | 71.90 |
| DSKD | 89.51 | 70.53 | 57.27 | 84.39 | 75.36 | 69.96 | 68.81 | 66.78 | 70.05 | 65.76 | 75.90 | 72.52 |
| Jasper & Stella | 88.38 | 74.13 | 62.81 | 85.98 | 80.33 | 68.38 | 74.35 | 66.80 | 74.55 | 68.58 | 78.33 | 75.08 |
| DistillCSE | 90.50 | 72.70 | 59.88 | 84.39 | 78.48 | 69.49 | 73.97 | 70.00 | 73.73 | 67.70 | 78.34 | 74.79 |
| EMO | 90.19 | 74.26 | 61.48 | 85.31 | 80.82 | 68.30 | 75.41 | 67.06 | 76.35 | 68.71 | 78.84 | 75.46 |
| TALAS | 89.72±0.19 | 75.38±0.11 | 65.83±0.58 | 87.58±0.02 | 82.90±0.12 | 66.96±0.07 | 75.82±0.11 | 68.11±0.20 | 77.51±0.18 | 70.10±0.13 | 79.92±0.05 | 76.64±0.04 |
| **Ours** | TBD | | | | | | | | | | | |
| ***(c) Qwen3-Embedding-0.6B → MiniLMv2-H384 (22M)*** | | | | | | | | | | | | |
| Teacher | 93.21 | 71.33 | 66.53 | 83.37 | 90.00 | 66.85 | 80.64 | 77.11 | 84.60 | 72.66 | 82.61 | 79.29 |
| Student base | 68.29 | 41.07 | 69.90 | 78.39 | 64.56 | 59.58 | 47.60 | 21.95 | 22.08 | 40.91 | 58.45 | 52.60 |
| SimCSE-unsup | 86.67 | 69.56 | 54.00 | 82.29 | 73.76 | 65.71 | 66.04 | 59.11 | 62.08 | 60.60 | 72.91 | 68.80 |
| CDM | 86.07 | 70.32 | 53.04 | 81.97 | 72.97 | 68.34 | 65.97 | 61.12 | 65.75 | 62.38 | 73.07 | 69.51 |
| DSKD | 85.66 | 69.86 | 52.25 | 82.17 | 73.52 | 68.58 | 66.54 | 62.52 | 66.70 | 62.51 | 73.38 | 69.76 |
| Jasper & Stella | 85.94 | 71.25 | 57.99 | 83.36 | 76.17 | 68.25 | 70.57 | 63.81 | 70.11 | 65.45 | 75.18 | 71.94 |
| DistillCSE | 88.03 | 69.87 | 54.24 | 83.20 | 77.66 | 67.88 | 70.09 | 68.19 | 71.53 | 64.55 | 76.17 | 72.30 |
| EMO | 87.03 | 71.67 | 59.63 | 83.45 | 78.17 | 68.89 | 70.81 | 65.78 | 71.42 | 66.65 | 76.15 | 72.98 |
| TALAS | 84.16±0.15 | 72.82±0.11 | 61.13±0.52 | 84.85±0.10 | 80.91±0.27 | 66.32±0.07 | 70.72±0.15 | 67.80±0.08 | 72.39±0.31 | 66.61±0.24 | 76.88±0.07 | 73.46±0.10 |
| Ours `[prov]` (no $\mathcal L_{r=1}$; anchors not yet in $\mathcal L_{\rm row}$; student graph) | 87.86 | 71.28 | 63.64 | 83.83 | 81.90 | 65.61 | 74.96 | 74.20 | 78.20 | 69.15 | 79.01 | 75.72 |
| **Ours** | TBD | | | | | | | | | | | |

Rules for the final table:
- **Marking.** Bold and underline mark best and second best, excluding Teacher
  and Student base.
- **Checkpoints.** One selection rule for all settings.
- **WiC.** It is the known weak task (65–66 against 68–72 for contrastive
  baselines); say so.

**Table 2. Cost per optimizer step** (same GPU, not shared).

| method | texts encoded | supervised rows | column evaluations | step (s) | peak GPU (MiB) | Avg |
|---|---|---|---|---|---|---|
| pointwise KD | 64 | 0 | 0 | TBD | 1,986 `[prov]` | 69.10 `[prov]` |
| in-batch relational KD | 64 | 64 | 4,032 | TBD | 2,628 `[prov]` | 71.54 `[prov]` |
| **ours** | ~4.7k | ~4.7k | TBD ($\sum_j\lvert\Omega_j\rvert$) | TBD | ~14,400 `[prov]` | TBD |

Offline cost:
- **Graph build.** One teacher encode plus exact top-$k$. The build took 27 s
  given cached teacher embeddings.
- **At scale.** Exact top-$k$ can be replaced by ANN.
- **Inference.** None.

### C2 — Mechanism

**Table 3. Exposure, not mechanism.** Encoder, optimizer, steps and seed are
fixed. Only which texts are co-encoded, and what rows are matched on them,
varies. The nbrs / row column counts the row centre's teacher neighbours among
the texts it is compared with.

| # | objective | batch | rows / step | nbrs / row | target | Avg | Avg-In | Avg-Out |
|---|---|---|---|---|---|---|---|---|
| 1 | pointwise cosine | random | — | — | — | 69.10 `[prov]` | 61.89 | 72.70 |
| 2 | pointwise cosine (CoSS-style) | neighbour-composed | — | — | — | 67.05 `[prov]` | 60.28 | 70.44 |
| 3 | in-batch KL (PKT/SEED-style) | random | 64 | 0.46 | graded | 71.54 `[prov]` | 63.56 | 75.54 |
| 4 | in-batch KL (BINGO-style batches) | neighbour-composed | 64 | TBD | graded | 73.74 `[prov]` | 66.44 | 77.40 |
| 5 | anchor rows only + $\mathcal L_{\rm cal}$ | subgraph | 64 | 100 | graded | TBD | | |
| 6 | **ours: every encoded row + $\mathcal L_{\rm cal}$** | subgraph | ~4.7k | ~61% of mass | graded | TBD | | |
| 7 | ours, uniform targets | subgraph | ~4.7k | ~61% of mass | uniform | TBD (prov. 71.93, −3.64) | 63.18 | 76.30 |
| 8 | ours, random columns | subgraph | ~4.7k | random | graded | TBD (prov. 75.18, −0.39) | 68.38 | 78.58 |

The `[prov]` deltas in rows 7–8 are relative to 75.57, the three-term run on the
student graph.

How to read it:
- **1→2.** Neighbour batching alone does not help; it cost −2.05 in the
  provisional run.
- **3→4.** Batch-local KD improves exactly when batches contain neighbours.
- **5→6.** Supervising every encoded row is the C3 step.
- **6→7.** Graded values carry most of the gain. The worst provisional losses
  were STS-B −9.3 and STS12 −6.7. This is the opposite of Böhm et al. (JMLR'22),
  where binary kNN affinities suffice for t-SNE.

### C3 — Principle

**Table 4. Loss terms and held-out relations.**
- **Held-out split.** Withhold 20% of teacher-graph edges from training, with the
  same split for every arm. Withheld pairs are masked out of $\mathcal L_{\rm cal}$
  too, so no term ever sees them.
- **Metric.** Teacher/student Spearman on held-out pairs, split by whether both
  endpoints lie in the same $G^\leftrightarrow$ component (built on training
  edges).

| arm | $\mathcal L_{\rm row}$ rows | $\mathcal L_{\rm cal}$ | Avg | Avg-In | Avg-Out | held-out ρ, same component | held-out ρ, cross component |
|---|---|---|---|---|---|---|---|
| **ours** | all encoded | ✓ | TBD | | | TBD | TBD |
| $\lambda_{\rm cal}=0$ | all encoded | — | TBD | | | TBD | TBD |
| $\lambda_{\rm row}=0$ | — | ✓ | TBD | | | TBD | TBD |
| anchors only | anchors | ✓ | TBD | | | TBD | TBD |
| anchors excluded (`--row_set non_anchors`) | non-anchor | ✓ | TBD (prov. 75.72) | 69.15 | 79.01 | TBD | TBD |

- **Prediction.** $\lambda_{\rm cal}=0$ matches ours on same-component pairs and
  loses on cross-component pairs; anchors-only loses on Avg.
- **Provisional context.** On 2026-09-13, folding anchors into the row mean gave
  +0.26.

**Table 5. $k$ and reciprocal connectivity.**

| $k$ | Avg | Avg-In | Avg-Out | $G^\leftrightarrow$ components | isolated nodes | exposed row mass | texts / step | peak GPU (MiB) |
|---|---|---|---|---|---|---|---|---|
| 25 | TBD (prov. 75.00) | 68.00 | 78.50 | 238 `[prov]` | 218 | 0.39 | ~1.5k | 7,424 |
| 50 | TBD (prov. 75.27) | 68.40 | 78.70 | 77 | 69 | 0.49 | ~2.7k | 10,136 |
| **100** | TBD (prov. 75.57) | 68.83 | 78.94 | 15 | 13 | 0.61 | ~4.7k | 14,438 |
| 200 | TBD (prov. 75.52) | 68.83 | 78.86 | 6 | 5 | 0.75 | ~7.2k | 22,822 |

- **Source.** Connectivity comes from the student-graph logs; rebuild it on the
  teacher graph.
- **Timing.** Step time is omitted because those runs shared GPUs.
- **Reading.** Components fall to ~15 exactly where Avg saturates.

**Table 6. Robustness** (one change at a time from the default).

| arm | Avg | Δ | Avg-In | Avg-Out | `[prov]` precursor |
|---|---|---|---|---|---|
| **default** ($\lambda_{\rm cal}:\lambda_{\rm row}$ = 0.5:1, teacher kNN, directed, $\tau_j$, no reweighting) | TBD | — | | | 75.57 (three-term) |
| $\lambda_{\rm cal}:\lambda_{\rm row}$ = 0.25:1 | TBD | | | | 75.48 |
| $\lambda_{\rm cal}:\lambda_{\rm row}$ = 1:1 | TBD | | | | 75.65 |
| $1/\hat p_j$ row reweighting (GraphSAINT) | TBD | | | | — |
| student kNN graph | TBD | | | | teacher 75.73 vs student 75.52 at 63 cols |
| mutual kNN filter | TBD | | | | −0.28 under holdout |
| global temperature $\bar\tau$ for every row | TBD | | | | — |

- **Reweighting.** If $1/\hat p_j$ wins, it becomes the default; it adds no knob.
- **Purpose.** The other rows show the default is not a narrow peak. They are not
  a search.

---

## 6. Objections

- **"PKT/SEED on a kNN graph."** The KL is theirs (§4). The new parts are what
  gets supervised and why that is exact. Table 3 rows 3→6 hold the loss family
  fixed.
- **"BINGO/CoSS already batch neighbours."** Table 3 row 2 reproduces that
  recipe without row targets, and it does not help.
- **"Top-$K$ targets are biased."** Only when they are not renormalized. By
  Lemma 1, ours keep their fixed points. The inclusion weighting is stated and
  ablated (Table 6).
- **"Binary kNN affinities suffice (Böhm et al.)."** For visualization, yes. For
  distillation, no: Table 3 row 7.
- **"$\mathcal L_{\rm cal}$ depends on the pool."** It is the only term that
  does. If Table 4 shows $\lambda_{\rm cal}=0$ ties, the method drops to one
  term.

---

## 7. To do

**Code.** Done (2026-09-15): teacher graph default, anchors in $\mathcal L_{\rm row}$,
`r1_weight` and every unused option removed, `--row_reweight`, reciprocal-component
stats, one script per table plus `scripts/exp/launch.sh`, held-out probe split by
component. PKT/SEED-style is `--batch_local`; CoSS-style is pointwise with
`--batch_sampler neighbor`, on the same deduplicated corpus as the graph.

**Runs** (3 seeds, teacher graph; `PROFILE=paper bash scripts/exp/launch.sh start`):
1. Table 5, the $k$ sweep. It also provides the default reference.
2. Table 3, then Table 4. Table 4 runs its own arms under the 0.2 holdout (its
   Avg is not comparable with Tables 3, 5, 6); the held-out probe scores its
   checkpoints.
3. Table 6.
4. Table 1, settings (a) and (b).
5. Table 2, on an unshared GPU.

**Stop rules:**
- **All rows ties anchors only** (Table 3, row 5 vs 6). C3 loses its payoff; the
  paper becomes C1 + C2.
- **$\lambda_{\rm cal}=0$ ties on cross-component pairs** (Table 4). Drop
  $\mathcal L_{\rm cal}$'s role from C3 and present it as a calibration add-on.

---

## 8. Paper outline

1. **Introduction.** The 0.46 count; starved, not biased; C1–C3.
2. **Related work.** Relational and embedding KD; neighbour embedding;
   neighbour-composed batches; subgraph sampling and partial-ranking theory.
3. **Method.** §2, with Lemma 1 and Proposition 2 inline.
4. **Results (C1).** Tables 1–2.
5. **Where the gain comes from (C2, C3).** Tables 3–6.
6. **Limitations.**
   - Inclusion weighting.
   - $\mathcal L_{\rm cal}$ depends on the pool.
   - Identifiability is edge-level only.
   - The graph build is exact.
   - WiC and pair classification lag.
   - The training corpus is 13.5k texts.
