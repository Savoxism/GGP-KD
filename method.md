# GGPKD — Method Specification

Relational knowledge distillation from an embedding teacher to a small student.
The frozen base student's kNN selects each row's columns; the teacher supplies
the similarities and transition targets on those columns.

The method has four explicit structure/loss settings: `graph_k`, `r0_weight`,
`r1_weight`, and `row_weight`. The three loss coefficients are independent and
are applied directly; none is normalized against another.
Bandwidths, temperatures, the candidate set, and supervised rows are derived
from the cached corpus embeddings.

The radius is one hop: the objective matches the teacher's transition rows
directly. Multi-hop diffusion has been removed, not merely defaulted off.

---

## 1. Offline: a student-selected, teacher-valued graph

Built once per (teacher, corpus) and cached.

**1.1 Encode.** The teacher and the frozen base student each embed the
deduplicated corpus once; vectors are L2-normalized, so all similarities below
are cosines.

**1.2 Retrieve.** For each node `i`, retrieve the `k = graph_k` highest-cosine
other nodes under the base student. This student kNN is the method; teacher kNN
is the source ablation. Self is excluded during retrieval.

**1.3 Teacher bandwidth.** Each row reads its temperature from the teacher's own
raw top-k span:

$$\tau_i = \frac{s_i^{(1)} - s_i^{(k)}}{\log k}$$

where $s_i^{(j)}$ is the teacher's $j$-th largest corpus cosine from `i`. On the
teacher's own reference top-$k$, the $k$-th neighbour sits $\log k$ nats below
the nearest and receives $1/k$ of its unnormalized weight. The actual target is
then evaluated on the student-selected columns at this teacher-derived scale;
it does not imply the same endpoint ratio within that different set.

*Affine invariance.* Under $s \mapsto as + b$ the bandwidth scales as
$\tau \mapsto a\tau$, so the logits become $s_j/\tau_i + b/(a\tau_i)$. The second
term does not depend on $j$ and a softmax is shift-invariant, so **the row is
unchanged**. This is the property that rules out a single global temperature,
which would otherwise have to be retuned for every teacher whose cosines are
spread differently.

*Fixed sample size.* The scores come from the raw top-$k$, read **before** the
mutual filter, so all $k$ values exist for every node whenever $k < n$. There is
no degree to fall short of and no target entropy to miss.

**1.4 Edge rule.** None. The neighbour set is the base student's retrieved list,
`N_S(i) = topk_S(i)`, so every node has degree `graph_k`. A mutual filter remains
the `--knn_mode mutual` ablation. With no filter, no node can be isolated.

**1.5 Teacher transition row.** Over the student-selected neighbours,

$$P^T(j \mid i) = \operatorname{softmax}_{j\in N_S(i)}\!\left(s^T_{ij} / \tau_i\right)$$

Thus the student chooses which texts are compared, but never supplies its own
target values.

**1.6 No truncation.** The row is kept whole. Every student-selected column
carries its teacher probability into the objective, so all rows have exactly
`graph_k` columns, nothing about the target depends on a numerical constant, and
collation needs no padding.

> A mass-prefix truncation (`--truncation_tolerance 0.01`: keep the smallest
> prefix holding 99% of the row) remains as an arm, and the multi-hop arms
> require it — a diffused row is dense and cannot be carried whole.

The artifact stores the rows and their bandwidths.

---

## 2. Training: the candidate set

For each anchor, the candidate set is **its whole transition row** — every
column retrieved by the frozen base student, and nothing else. No negatives are drawn.

There is no budget, no selection and no RNG. The set is a deterministic function
of the graph, is **identical in every epoch**, and is the same width `graph_k`
for every anchor. (The padding path — short rows padded with the anchor's own
index, removed from every softmax by the self-mask — is still there for the arms
that produce ragged rows, and is inert for the method.)

Each step encodes the deduplicated union of the batch's candidate sets — the
**shared pool**. This is essentially the whole step cost, and it is now the
union of `|B|` rows of width `graph_k` rather than of truncated rows: re-measure
it before quoting a number (the ~1,400 texts per batch of 64 recorded here was a
mutual graph with 99%-prefix rows).

---

## 3. Objective

$$\mathcal{L} = \underbrace{\lambda_0\mathcal{L}_{r=0} +
\lambda_1\mathcal{L}_{r=1}}_{\mathcal{L}_{\text{rel}}}
+ \lambda_{\text{row}}\,\mathcal{L}_{\text{row}}$$

The defaults are $\lambda_0=\lambda_1=0.5$, recovering the previous equal split.
The coefficients are absolute: changing both by the same factor changes the
relational objective's scale relative to $\mathcal L_{\rm row}$, and deleting
one term does not rescale the other.

### 3.1 Graph scale, `r=1`

For each anchor, KL between the teacher's transition row and the student's
softmax, over **the anchor's own candidate columns**, at $\tau_i$ on both sides.
Because the `r=1` target *is* the transition row, the student reuses the
bandwidth stored with that row — there is no temperature to choose.

### 3.2 Ambient scale, `r=0`

For each anchor, KL over the **whole shared pool** at a single temperature,
applied to teacher and student alike (same-temperature distillation, Hinton et
al. 2015). That temperature is derived as the median bandwidth of the graph.

The two groups deliberately use **different column sets**: `r=1` ranks *within*
the neighbourhood, `r=0` calibrates similarity levels *across* the batch. Neither
holds an opinion the other contradicts. The ambient group carries the same total
weight as the graph group.

### 3.3 Row supervision

Non-anchor nodes `j` already in the shared pool are promoted to auxiliary rows:

$$\mathcal{L}_{\text{row}} = \sum_j \nu_B(j)\, \mathrm{KL}\!\left(P^T_j|_{\Omega_j} \,\|\, p^S_j|_{\Omega_j}\right), \qquad \Omega_j = N_S(j) \cap \text{pool}$$

with the dense transition row as target — row-kernel matching, not a trajectory
likelihood — each row at its own stored bandwidth $\tau_j$, weighted uniformly.
Batch anchors are excluded: $\mathcal{L}_{\text{rel}}$ already matches their row
at `r=1`. Rows with fewer than two available columns are dropped.

The row set is a deterministic function of the pool, so this term carries **no
selection hyperparameter** and reuses computation $\mathcal{L}_{\text{rel}}$ has
already paid for.

### 3.4 Temperature ties

None of these is a free parameter; the criterion rejects them by name.

| | tied to |
|:---|:---|
| $\tau_1(i)$ | $\tau_i$ — the `r=1` target *is* the transition row |
| $\tau_{\text{row}}(j)$ | $\tau_j$ — row targets are transition rows |
| ambient | one temperature on both sides, the median $\tau_i$ |
| relational group weights | explicit `r0_weight`/`r1_weight`, applied directly without normalization |

---

## 4. Hyperparameters

| | default | role |
|:---|---:|:---|
| `graph_k` | 200 | retrieval width **and**, through $\tau_i$, row sharpness |
| `r0_weight` | 0.5 | independent coefficient $\lambda_0$ for ambient calibration |
| `r1_weight` | 0.5 | independent coefficient $\lambda_1$ for graph-transition matching |
| `row_weight` | 1.0 | independent coefficient $\lambda_{\text{row}}$ for auxiliary-row supervision |

Plus standard training settings (batch size, epochs, learning rate, seed).

> `graph_k` does two jobs. A sweep over it cannot separate neighbourhood width
> from row sharpness — that is the price of not carrying a second constant, and it
> must be stated rather than hidden. Too large a `k` measures the distance out of
> the anchor's neighbourhood rather than the local decay, and the rows go uniform;
> `scripts/ggpkd/pick_graph_k.py` reports the induced sharpness per `k` from one
> teacher-embedding pass.
