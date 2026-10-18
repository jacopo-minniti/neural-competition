# Source selection and assumption audit

Search/read date: **18 October 2026**. Sources were selected against the
repository's reports and model implementation, using original papers and
proceedings. This is a targeted explanation of these experiments, not an
exhaustive literature review. Links below identify the versions examined.

The substantive paper claims are summarized where they are used. The extended
calculations in the notes are deductions for our model; predictions and
interventions remain proposals unless accompanied by an explicit saved-state
check. A source supporting a mechanism in another setting is not a theorem
about our Adam runs.

## Papers that changed an explanation, measure or proposed experiment

| Source | Exact part examined | Why it is retained | Where the application is worked out |
| --- | --- | --- | --- |
| Safran, Yehudai, Shamir (COLT 2021), [The Effects of Mild Over-parameterization on the Optimization Landscape of Shallow ReLU Neural Networks](https://proceedings.mlr.press/v134/safran21a.html) · [paper](https://proceedings.mlr.press/v134/safran21a/safran21a.pdf) | §2 assumptions; Theorems 3, 6–9, 14; §4.1 caveat about newly formed wider minima | Separates escape from local conditioning; motivates a function-preserving width intervention | [Width, escape and scaling](width_escape_and_scaling.md) |
| Liu, Wu, Wang (NeurIPS 2019), [Splitting Steepest Descent for Growing Neural Architectures](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a01fc0853ebeba94fde4d1cc6fb842a-Paper.pdf) | §2.1–2.2, local expansion and splitting matrix | Supplies a second-order split diagnostic; checking it in our model reveals the fixed-gate limitation | [Width, escape and scaling](width_escape_and_scaling.md) |
| Liu, Zhu, Belkin (2020 preprint; ACHA 59:85–116, 2022), [Loss landscapes and optimization in over-parameterized non-linear systems and neural networks](https://arxiv.org/abs/2003.00307) · [v2 paper](https://arxiv.org/pdf/2003.00307v2) | §§2, 4–5; PL* definition, tangent kernel, sufficient-region requirements | Derives a residual-weighted progress diagnostic; exposes why full-kernel lambda_min cannot work in our sample/parameter regime | [Residual access and tests](residual_access_and_tests.md) |
| Chizat, Oyallon, Bach (NeurIPS 2019), [On Lazy Training in Differentiable Programming](https://papers.neurips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html) · [paper](https://papers.neurips.cc/paper_files/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf) | §§2–3, scale dependence and linearized dynamics | Makes feature-learning strength a concrete confound in the width-128 slowdown and optimizer comparison | [Width, escape and scaling](width_escape_and_scaling.md) |
| Boursier, Flammarion (JMLR 26(183), 2025), [Early Alignment in Two-Layer Networks Training is a Two-Edged Sword](https://www.jmlr.org/papers/v26/24-1523.html) · [paper](https://www.jmlr.org/papers/volume26/24-1523/24-1523.pdf) | §2 initialization; Theorems 1–2; §4 regression counterexample | Motivates measuring effective feature supply and checking sign/initialization assumptions | [Adam, initialization and features](adam_initialization_and_features.md) |
| Sandbrink, Martinelli, van Meegen, Gerstner, Brea (September 2026 preprint), [Teacher Geometry Shapes Learnability in Teacher-Student Networks](https://arxiv.org/html/2609.09595v1) | §§2.2, 4–6; Appendix B.1 training; E fast-readout reduction; F.3 Adam control | Generates gate-coverage/readout-oracle checks and a differential-layer-rate intervention | [Adam, initialization and features](adam_initialization_and_features.md) |

The 2026 preprint is useful chiefly because its mechanism is checkable here.
Its empirical endpoint protocol uses gradient-flow integration followed by
second-order refinement, excludes runs without saved outputs, and includes
an output bias. We do not transplant its convergence frequencies. Our saved
failures also give counterevidence to the simplest always-off/on explanation.

## What the requested classics do and do not imply here

**Cooper (2018), [The loss landscape of overparameterized neural networks](https://arxiv.org/abs/1804.10200),
Theorem 2.1.** For smooth prediction maps with p>N, the zero-loss set is
generically a p−N dimensional manifold, possibly empty. This concerns generic
regular targets and geometry at interpolation, not an optimizer finding it.
Our targets are specially realizable, ReLU maps are piecewise smooth, and
p=22m<N=8,192 throughout the documented widths up to 128. Thus the generic
p−N dimension statement does not explain these runs. The useful lesson is
to distinguish solution degeneracy from dynamic access to a solution.

**Nguyen (ICML 2019), [On Connected Sublevel Sets in Deep Learning](https://proceedings.mlr.press/v97/nguyen19a.html),
[Theorem 6.1](https://proceedings.mlr.press/v97/nguyen19a/nguyen19a.pdf).** In its
ReLU setting, all hidden widths at least N remove bad local valleys; at least
2N give connected sublevel sets. For one hidden layer these are widths at
least 8,192 or 16,384 here, not 10 or 16. The earlier single-wide-layer
theorems use stricter activation assumptions; do not cite them as a ReLU
guarantee. Connected paths do not identify which directions our optimizer
takes or how long it needs. These results are a boundary on interpretation,
not a predicted width threshold for our realizable teacher task.

**Belkin with Liu and Zhu** supplies a useful quantitative route, developed
in the residual-access note. Its sufficient kernel conditioning and our
necessarily singular full-training kernel must be distinguished. This also
avoids treating that framework and Safran et al.'s local-PL failure as
contradictory results about one common setting.

In these experiments, “overparameterized” primarily means **wider than a
known teacher representation**, whereas many classical theorems require
more parameters or hidden units than arbitrary training constraints. Neither
regime can be silently substituted for the other.

## One tempting Adam result that should not be overinterpreted

Vasudeva, Lee, Sharan and Soltanolkotabi,
[The Rich and the Simple: On the Implicit Bias of Adam and SGD](https://arxiv.org/html/2505.24022v2)
(2025), is a useful warning that optimizer-dependent feature selection can
matter. However, its theoretical setup fixes the signed final layer, uses
classification/correlation objectives, and the main contrasting Adam theorem
uses beta1=beta2=0, i.e. signGD. It cannot prove that default Adam avoids
feature collapse or output-polarity failure in our jointly trained regression
model. We therefore do not use it to explain away our counterexamples or
derive a promised optimizer-universal measure.
