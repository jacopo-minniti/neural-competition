# Mentor update

A four-page English report titled **Mitigating Neural Competition Through
Overparameterization**, with four unnumbered subsections, four figures and one table.
It covers gradient distributions, neuron interactions, signed feature roles and
readout adaptation. Stability and optimizer-history controls are integrated into
the first subsection, whose tentative interpretations address the original
distribution questions. The one-step neuron comparison is a table; a cumulative
cost figure makes the contrasting momentum and Adam results explicit. Negative
results qualify the narrative rather than appearing as isolated observations.
The network, loss, gradient-flow assumptions and readout projection identities
are defined in the text. Captions are numbered, left aligned and single spaced
(9-point type on an 11-point baseline); Figures 2–4 use 88% of the text width.
Every plotting panel has a frame on all four sides.

- [Editable LaTeX](main.tex)
- [Compiled PDF](../../output/pdf/neural_competition_update.pdf)
- [Portable LaTeX sources](../../output/pdf/neural_competition_latex.zip)
- [Figure builder](build_figures.py)
- [Evidence manifest](evidence.json)

The attached private conversation was used to identify the scientific requests;
its personal content is not copied into this project or the output archive.

## Evidence selection

| Subsection | Retained evidence |
| --- | --- |
| 1. Gradient distributions | `docs/tables/history/teacher/pier_tail_summary.csv`, `pier_tail_effects.csv`, `pier_agreement_summary.csv`; original report `b03c53f:docs/reports/batch_distributions.md` |
| 1. Stability controls | `pier_continuation_summary.csv`, `pier_outcomes_summary.csv`; detailed finite-step/sharpness results in the same original report |
| 1. Optimizer-history controls | `momentum_gradient_summary.csv`, `momentum_actual_step_summary.csv`; original `b03c53f:docs/reports/momentum.md` |
| 2. Neuron interactions | `pair_interference_summary.csv`, `pair_interference_effects.csv`; [research history](../reports/research_history.md) and its cumulative/temporal tables |
| 3. Signed roles | [Residual-growth report](../reports/residual_growth.md), [derivation](../theory/residual_growth.md), `summary.csv` and `roles.csv` in its table directory |
| 4. Readout corrections | [Feature-profile report](../reports/feature_profile.md) and [complete evidence](../tables/feature_profile/README.md) |

The original batch-distribution and stability documentation was recovered from
Git and checked against retained results. No experiment was rerun for this report:
those completed studies already cover the original shallow-ReLU requests.
The report makes no claim to completed normalization, attention, CIFAR or language
experiments. MNIST remains in the repository but is outside this report. The report also distinguishes
reported negative results from incomplete causal or predictive claims.

Numerical conventions: original teacher errors are ordinary MSE; squared-loss
identities use half-MSE. Residual/profile table losses are half-MSE on unit-training-
variance targets, so the report's MSE percentages equal 200 times table loss.
The first figure is the unchanged original 128-probe distribution plot, recovered
from Git; the qualitative tail discussion also uses the fresh 512-probe validation.
All new plots use full-cohort summaries, not the purposively retained old states.
Error plots show cohort means. The report preserves the limits of loss matching
and early/late intervention timing.

## Figures

- `distributions.png`: original complete CDFs, raw versus normalized, from
  `b03c53f:docs/figures/distributions/gradient_distributions_0.2.png`.
- Table 1 (in `main.tex`): frequency, count and objective-relative harm from
  the early momentum cohort, using the original128 full-step comparison.
- `cumulative_cost.pdf`: positive pair cost versus final reference MSE from
  `history/effect_magnitude/cumulative_per_seed.csv`, reference view, widths 8/16.
  It uses the complete `teacher8/momentum` and `k8_teacher2027/adam_001` cohorts;
  arrows connect cohort means, not time points. Different protocols prevent an
  optimizer ranking. Reference MSE is twice the stored reference half-MSE.
  The matched-helpful-movement interpretation is retained from the research history.
- `signed_roles.pdf`: same/opposing-sign errors and positive-role availability
  from the original 20-start residual-growth cohort.
- `readout_timing.pdf`: ordinary/early/late means and the wide baseline from
  the fresh main feature-profile cohort; these are distinct from the preceding cohort.

## Rebuild

From the repository root, with the project's analysis dependencies:

```sh
MPLCONFIGDIR=/tmp/neural-competition-mpl .venv/bin/python docs/mentor_report/build_figures.py
tectonic --outdir output/pdf docs/mentor_report/main.tex
```

The desktop LaTeX editor can open this folder and compile it directly. For
portable use, unpack the source archive, keeping `main.tex` and `images/` together.
The PDF was checked with the desktop compiler and Tectonic, rendered page by page,
and inspected for clipping, table overflow and page breaks. It has four A4 pages.
Generated review images and TeX logs are not retained in the research tree.
