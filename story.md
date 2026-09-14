# GGPKD — the story

What the paper claims, what it deliberately does not claim, and how each claim
is allowed to be phrased. Written after the survey in which the objective turned
out to be prior art; the conclusion of that survey is that the objective
*should* stay prior art, and the paper should sell the finding instead.

A note on wording, because the whole framing turns on it. Avoid "support of the
target distribution", "teacher mass", "relational budget" and "teacher
relevance". They are either jargon or they smuggle in a claim of authority we do
not need and cannot defend. Say instead: *which texts are actually similar to
this one*, *which comparisons a student is shown*, *how many comparisons per
anchor*. The argument is stronger in plain words, and it stops a reader from
hearing "the teacher decides what matters" — which invites the question *matters
for what?*, a question we would then have to answer.

---

## 1. Thesis

> A frozen base student already provides a cheap, per-text shortlist of related
> corpus items; what it lacks is the teacher's geometry over that shortlist.
> Mini-batch training almost never shows it those pairs: it compares each text
> against whatever happens to share its batch, which is almost always unrelated.
> GGPKD keeps the student's shortlist fixed and asks the teacher how those texts
> should be ordered and calibrated.

Everything else in the paper is in service of that paragraph.

---

## 2. The gap, and the number that carries it

Take the corpus this paper trains on: 13,553 texts, batch size 64. Count a
text's candidate neighbours generously — say its 200 nearest under the frozen
base student.
The expected number of them landing in the same batch is

$$63 \times \frac{200}{13{,}552} \approx 0.93 .$$

So of the **63 comparisons** an anchor receives per step, fewer than **one** is
with a text it is actually related to. Count neighbours at 20 instead of 200 and
it falls to 0.09: one informative comparison every eleven batches. Everything
else is *unrelated vs unrelated* — something the student already gets right and
would keep getting right without being told.

This is not a subtle inefficiency. It is where almost all of the supervision
goes.

Every relational KD method has to choose which comparisons to make, and the
literature chooses them by **what is cheap to have on hand**:

| method | what an anchor is compared against | chosen by | labels |
|---|---|---|---|
| SP, RKD, PKT, IRG | the other examples in the batch | batch composition | no |
| CompRess, SEED, ISD | a memory bank / momentum queue | availability | no |
| XBM | cross-batch feature memory | availability | no |
| ANCE, RocketQA | top-$k$ of an ANN index | an online student, negatives only | yes |
| LSP (GCN KD) | graph neighbours | the task's own graph | yes |
| **GGPKD** | a frozen base student's top-$k$ | **offline student retrieval; teacher-valued targets** | **no** |

The relevant design point is a fixed, label-free, per-anchor comparison set from
the deployable model, with a frozen teacher supplying soft relational targets
rather than labels or negatives.

The reason is not oversight, it is cost: a per-anchor comparison set means
encoding a different group of texts for every anchor. Queues and memory banks
exist precisely to avoid that. Section 5 is the answer to that cost.

---

## 3. What we claim

**C1 (the finding).** Replacing an anchor's batch-local comparisons with the
texts retrieved by the frozen base student improves the distilled model. What
pays is that the comparisons are with related texts: drawing the same number
uniformly from the corpus does not reproduce the gain. The teacher supplies the
target similarities and ordering on every support. The effect tracks *how many*
related texts an objective is shown: composing batches from one neighbourhood
improves the baseline. Enlarging the batch buys the same count only linearly, at
far more encoder work — a consequence of the count formula, not a trained result.

*The like-for-like count is experimental control, not a constraint we impose on
the method.* It is there so the difference cannot be attributed to one arm
simply making more comparisons, exactly as we fix the seed and the learning
rate. Practical cost is a separate question with its own table.

**C2 (the loss is held fixed).** Every support arm runs the complete objective:
$\lambda_0\mathcal L_{r=0}+\lambda_1\mathcal L_{r=1}+\lambda_{\rm row}\mathcal
L_{\rm row}$, with the same weights in every arm. The anchor-row quota is matched; pool width and usable auxiliary
rows are reported as outcomes of the support. Only the loss-decomposition study
turns terms off.

**C3 (it is affordable).** Deduplicating the comparison sets of a batch into one
shared pool, and reusing already-encoded pool members as extra anchors, makes a
per-anchor comparison set cost roughly what a batch-local objective costs.

**C4 (explicit loss balance).** The per-row temperature, calibration
temperature, comparison set and supervised rows are determined offline.
`r0_weight` and `r1_weight` are independent coefficients applied directly,
without normalization; both default to 0.5. The per-row
temperature $\tau_i$ comes in closed form from the retrieval span, which makes
each row invariant to the scale of a given teacher's cosines.

---

## 4. What we explicitly do NOT claim

Stating these is what keeps C1–C4 credible.

- **Not a new loss.** Softmax over a comparison set, matched by KL, is CompRess
  (NeurIPS 2020) and SEED (ICLR 2021); the same operation restricted to graph
  neighbours is LSP (CVPR 2020). We use them as-is and say so.
- **Not the first to argue that in-batch sampling is a bad sampler.** ANCE
  (ICLR 2021) makes that argument for dense-retrieval negatives and fixes it
  with a global ANN index. Ours is the unlabelled, frozen-teacher case, where
  the chosen texts carry the *targets* as well, not just the contrast.
- **Not a new temperature rule.** Per-point bandwidths from the $k$-th neighbour
  are self-tuning spectral clustering (Zelnik-Manor & Perona, 2004); the
  $\log k$ normalisation is UMAP's. Ours is a closed form instead of a solve —
  an engineering simplification, not a concept.
- **Not a claim that geometric distortion controls downstream score.** The
  alignment/coverage split motivates the design; it does not predict benchmark
  movement.
- **Not global geometry preservation** until the held-out probe is repaired
  (`experiments.md`, Stage 2E). Until then the honest phrasing is that the
  student is supervised on the comparisons that carry information, and the title
  should say that.
- **Not online hard-example mining.** The base-student graph is built once and
  never follows training error. It selects by similarity, not by where the
  evolving student currently fails; the latter is ANCE's axis.

---

## 5. The method, as the cheapest realisation

Offline, once per (teacher, base student, corpus): encode both models and
$\ell_2$-normalise. The base student retrieves the top $k$ columns for every
text; the teacher supplies the cosine score on those columns. No reciprocity
filter is applied, so every node has degree exactly $k$. Read the temperature
from the teacher's own top-$k$ span, $\tau_i =
(s_i^{(1)}-s_i^{(k)})/\log k$. On that teacher reference list, the $k$-th item
receives $1/k$ of the nearest item's unnormalized weight. The target row is the
softmax of teacher cosines over the student-selected neighbours at $\tau_i$,
kept whole — no truncation; it need not have the same endpoint ratio because its
columns came from a different encoder.

During training the comparison set of an anchor **is** its retrieved
neighbourhood: no sampling, no negatives, no RNG, identical in every epoch. A
batch encodes the deduplicated union of those sets once, and that single
encoding pass serves three terms:

$$\mathcal L = \underbrace{\lambda_1\mathcal L_{r=1}}_{\text{order within the neighbourhood}}
 + \underbrace{\lambda_0\mathcal L_{r=0}}_{\text{calibrate levels across the pool}}
 + \lambda_{\text{row}}\underbrace{\mathcal L_{\text{row}}}_{\text{reuse pool members as extra anchors}}$$

The default is $\lambda_0=\lambda_1=0.5$. These are absolute coefficients:
neither is divided by their sum or changed when the other term is removed.

The design principle in one sentence, for the intro:

> A frozen student graph decides which comparisons are evaluated; the teacher
> decides what those relations should be; mini-batches only schedule computation.

---

## 6. Evidence map

Derived from the claims, not from what is already in `runs/`. Each entry states
what a reader has to accept, what would falsify it, and the one experiment that
settles it. A claim whose falsifier cannot be stated is not a claim; an entry
whose experiment has not been run is marked as such rather than quietly replaced
by the nearest thing on disk.

**Premise P.** *Batch composition shows a student almost no informative
comparisons, and the shortfall grows with the corpus.*
Falsified if the count $(B-1)k/(N-1)$ is not small at realistic $N$, or if
related pairs are over-represented in batches relative to unrelated ones.
→ **A. The count**: 0.93 per step here, $1/N$ thereafter, and $B \approx 4{,}270$
to fix it by batch size alone. A formula and one figure; no training. (The
per-pair version — every pair has the same probability
$\binom{N-2}{B-2}/\binom{N}{B}$ — belongs in a footnote.)

**C1a.** *Comparisons with the frozen student's neighbours beat comparisons
with batch co-occupants, at the same anchor-row count.*
Falsified if the two tie when the number of comparisons, the loss, the
temperature and the encoder budget are equal.
→ **B. Comparison ladder**, arms `in_batch` vs `student_knn`.

**C1b.** *What pays is that the texts are related, not that they come from
outside the batch.*
Falsified if drawing comparisons uniformly from the whole corpus — batch-free
but unrelated — matches the student's neighbours.
→ **B**, arm `corpus_uniform`.

**C1c.** *The effect tracks the number of related texts an objective sees, not
the mechanism that supplies them.*
Falsified if raising that number for a batch-local objective — by composing
batches from one neighbourhood — leaves its score unmoved.
→ **C. Dose–response**. Buying the count with batch size is not trained: no
setting matches steps, data passes and learning rate across $B$ at once. It is
stated from A's formula and listed as a limitation.

**C1d.** *The shortlist need not be retrieved by the teacher.*
Falsified if replacing the frozen student's kNN by the teacher's own kNN gives a
robust improvement. The teacher still values both supports, so this isolates
whose retrieval chooses the columns.
→ **B**, arm `teacher_knn`.

**C1e.** *Batch composition changes what a batch-local objective learns and does
not change what ours learns.*
Falsified if ours moves with batch composition by more than the pointwise floor
moves.
→ **C. Composition intervention**, read as difference-in-differences.

**C2.** *The support result holds for the objective we ship.*
Falsified if the full-objective ladder in B does not separate related support
from batch-local or corpus-uniform support. Every B arm keeps
$\mathcal L_{r=0}$, $\mathcal L_{r=1}$ and $\mathcal L_{\rm row}$ active; only
the loss-decomposition stages remove terms.
→ **B. Comparison ladder**, with **D. Component ablation** explaining which
terms earn their place.

**C3.** *A per-anchor comparison set costs about what a batch-local one costs.*
Falsified if the encoder budget per step is materially higher at the same number
of comparisons.
→ **B**'s cost columns. Already exported per arm; no new runs.

**C4.** *The method is not brittle to its explicit weights.*
Falsified if the relational balance or row weight sits on a narrow peak.
→ **G. Sensitivity**, plus the deletion tests in `experiments.md` Stage 1,
which ask whether each loss group earns its place at all.

**Coverage.** *The student reproduces teacher relations it was never shown.*
Falsified if held-out teacher pairs are reproduced no better than by an arm that
never saw related pairs at all. Currently **failing** on the existing probe,
whose `pair_order` metric is degenerate.
→ **E. Held-out relations**.

---

## 7. Objections, and the prepared answer

**"This is CompRess with a kNN comparison set."** The comparison operation is
prior art; the contribution is the support intervention and its amortized
implementation. The full objective is held fixed across the ladder. The
anchor-row quota is matched, while the resulting shared-pool width, auxiliary
row exposure and encoder cost are measured and reported rather than hidden.

**"ANCE already argued this."** For negatives in a labelled discriminative
objective, with an index rebuilt from the student during training. Here there
are no labels, the base-student index is frozen, and the teacher supplies soft
targets on the selected texts rather than only a contrast.

**"You should compare where the student is wrong, not only where it retrieves
similar texts."** A fair objection, and a different axis. The frozen base
student chooses by similarity once; it does not mine the evolving student's
current errors. Online error mining is not tested and remains a limitation.

**"$\mathcal L_{r=0}$ compares against whatever the batch supplies, which is what
you criticise."** Two answers, decided by measurement (Stage 1.2 and 2C): either
the full objective is measurably unmoved by batch composition, in which case
that term calibrates levels rather than choosing the anchor support and we say
so with the number; or it is not, in which case the claim is narrowed and the
pool dependence is reported explicitly.

**"Only +0.15 from $\mathcal L_{\text{row}}$."** Reported as what it is: a cheap
add-on that reuses encoded texts, not a pillar. If Stage 1.1 shows its gain does
not depend on the extra anchors being graph-selected, it is removed and the
method drops one loss coefficient and the corresponding sensitivity sweep.

**"The graph costs $O(N^2d)$."** Stated in the cost paragraph, once, with the
ANN alternative named. It is paid once per (teacher, corpus) and never at
inference.

---

## 8. Paper outline

1. **Introduction** — what a teacher knows about a text; the 0.93 count; the two
   decisions mini-batch training conflates; contributions as *finding first,
   method second*.
2. **Related work** — three blocks: (i) embedding/relational KD and where each
   family gets its comparisons, with the table from §2; (ii) beyond-batch
   supervision (queues, memory banks, full-kernel, ANCE) and why none of them
   compares against related texts specifically; (iii) neighbourhood graphs and
   per-point bandwidths, crediting self-tuning/UMAP/LSP.
3. **Which comparisons** — formalise the choice; alignment and coverage; the
   exposure statement.
4. **GGPKD** — §5 above, presented as the cheapest realisation, not as novelty.
5. **Controlled study** — the ladder and the composition intervention as the
   centrepiece, with cost columns.
6. **Downstream results** — main table, component ablation, sensitivity.
7. **Limitations** — the fraction of a neighbourhood a pool exposes, graph cost,
   the calibration term's comparison set, the held-out result, the student-error
   axis we do not test, what happens at very large batch (not trained), and
   pair-classification being less uniform.
