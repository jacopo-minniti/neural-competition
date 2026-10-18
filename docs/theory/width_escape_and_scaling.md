# Mild width expansion, escape directions and slow refinement

The specific facts to explain are the [teacher-8 SGD results](../reports/training_width.md):
6/20 good fits at width 8, 8/20 at 9, 18/20 at 10, and slower final refinement
at width 128 under unchanged rates. There are also narrow successes and wider
failures. A theory should allow both improved basin access and different
within-basin dynamics.

## The closest landscape theorem

**Paper result.** Safran, Yehudai and Shamir,
[The Effects of Mild Over-parameterization on the Optimization Landscape of Shallow ReLU Neural Networks](https://proceedings.mlr.press/v134/safran21a.html)
(COLT 2021), study Gaussian population squared loss with orthogonal unit
teacher vectors, no biases and fixed positive readout coefficients. Theorem 14:
for a non-global local minimum with student width n≤K and sum_r||w_r||≤K,
there exists a neuron whose split into two positive collinear copies turns
the point into a saddle. This does not eliminate every wider local minimum.
Theorems 3–9 also distinguish locally strongly convex minimal-width solutions
from overparameterized solutions without local convexity or even a local PL
inequality at the differentiable global minima considered.

**Inference for our observations.** These results motivate an explanation
involving extra escape directions, rather than universally improved convexity
or smaller gradient conflict. Easier arrival at a good solution can coexist
with slow refinement along degenerate directions. This is a qualitative
mechanism template: our empirical, signed, trainable-readout, biased model
does not satisfy the theorem's assumptions. Even at minimal width our model
has ReLU rescaling symmetries, so the paper's strong-convexity result cannot
be applied to our raw parameters.

The observed jump at K+2 does not establish a universal two-neuron threshold.
It could reflect changed basin access, coverage probabilities, finite budgets
or initialization dynamics. To identify escape, expand an already failed
state while preserving its function; independent-from-start wide training
does not isolate that mechanism.

## A useful caution about splitting in our exact model

**Our derivation.** For a neuron contribution u(theta), a symmetric split has

\[
\tfrac12u(\theta+\varepsilon v)+\tfrac12u(\theta-\varepsilon v)-u(\theta)
=\tfrac12\varepsilon^2v^\top\nabla^2u(\theta)v+o(\varepsilon^2).
\]

The leading loss change is epsilon² v^T S v/2, with
S=E[e Hessian(u)]. A negative eigenvalue suggests an escape direction,
provided the expansion is valid. This is related to the splitting-matrix
idea in Liu, Wu and Wang's
[Splitting Steepest Descent for Growing Neural Architectures](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a01fc0853ebeba94fde4d1cc6fb842a-Paper.pdf)
(NeurIPS 2019), §2. Their smooth local expansion should not be silently
substituted for a nonsmooth ReLU gate-crossing argument.

There is an especially restrictive fact here. On a finite evaluation set
with fixed gates, write z=(w,b), x_tilde=(x,1), and

\[
c_r=\frac{1}{\sqrt m}\mathbb E[e\,1_{z_r^\top\tilde x>0}\tilde x].
\]

Then grad_z L=a_r c_r. The residual-weighted second derivative of u_r has
only mixed a,z entries, equal to c_r. At an exact differentiable stationary
point with a_r≠0, c_r=0, so this splitting matrix is **zero**. Consequently
splitting a live neuron does not automatically reveal a negative second-order
direction while all its sample gates stay fixed. Zero-output units are an
exception to that implication, since grad_z L=0 then says nothing about c_r.

Population Gaussian objectives integrate moving boundaries; a finite sample
has an open fixed-gate region whenever all preactivations are nonzero. This
is one reason the population splitting theorem need not transfer directly.
A useful rescue might require crossing gates, a near-zero outgoing unit,
or supplying a new feature, rather than infinitesimal splitting of a live unit.
A smooth surrogate can help explore this distinction, but changes the model.

## What an informative expansion would compare

For m→m' expansion, multiply the old outgoing coefficients by sqrt(m'/m).
This preserves their current contribution under our width normalization.
Compare continuations from the same failed state:

| Expansion | Function-preserving construction | Mechanism it tests |
| --- | --- | --- |
| One live neuron split | Two copies share its outgoing coefficient after the normalization adjustment | Escape from a redundant representation; add opposite perturbations and examine gate changes |
| One fresh zero-output unit | New random incoming vector, outgoing coefficient zero | New feature availability; its incoming gradient initially vanishes |
| Two canceling units | Same new incoming vector and bias, outgoing coefficients +A and −A | Zero initial function with immediate incoming gradients and both output polarities |
| Two fresh zero-output units | Same added incoming directions, both outgoing coefficients zero | Controls additional width against the canceling-pair mechanism |

The canceling pair proposal is particularly relevant to the initialization
failure: a_r=0 blocks hidden learning initially; opposite nonzero coefficients
can cancel the function without blocking their individual hidden gradients.
Its relevance to the K+2 success pattern is a hypothesis, not an explanation
already demonstrated by those from-start runs.

There is a simple first-order statement behind this proposal. Hold the old
function fixed, use the expanded normalization, and define
c=E[e 1_(z^T x_tilde>0) x_tilde]/sqrt(m') for the new direction. The signed
pair has incoming gradients +A*c and −A*c. Their combined incoming-parameter
contribution to the full-gradient squared norm is 2A²||c||², whereas two
zero-output units supply zero incoming contribution initially. The pair
therefore opens immediate incoming descent whenever c is nonzero, despite
adding no initial prediction. This is a local gradient fact; the size of A
also changes step strength and must be controlled when comparing trajectories.

Identical copies with identical optimizer states remain symmetric under the
same minibatches. An unperturbed split alone cannot demonstrate specialization.
Use multiple perturbation radii and independent random directions, retain
the original neuron functions, and compare against a same-width perturbation
control. A sufficiently large perturbation can cross gates even when the
infinitesimal split calculation is flat.

Adam history needs explicit treatment: rescaling parameters and copying
neurons does not have a unique state mapping that preserves its ordinary
scalar-rate dynamics. A clean first comparison can reset moments in **all**
branches, including the no-expansion control; that tests expansion following
a common reset. Preserve-history comparisons need their mapping stated and
a reset-only control. Match initial loss exactly, and control first useful
prediction change or step size so a rescue is not just a larger update.

## Width also changes the learning regime

**Paper result.** Chizat, Oyallon and Bach,
[On Lazy Training in Differentiable Programming](https://papers.neurips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html)
(NeurIPS 2019), show that lazy, approximately linearized training is induced
by scaling and is not a consequence of parameter count alone. Its relevance
here is a required control on the mechanism, rather than a claim that our
finite-width ReLU trajectories satisfy their smooth-model bounds.

**Our model's gradients** make the issue visible:

\[
g_{a_r}=\frac{1}{\sqrt m}\mathbb E[e\phi_r],\qquad
g_{z_r}=\frac{a_r}{\sqrt m}\mathbb E[e\,1_{z_r^\top\tilde x>0}\tilde x].
\]

Starting at zero output, SGD's first outgoing movement is O(eta/sqrt(m))
if feature correlations are O(1); the ensuing early incoming gradient can
therefore be O(eta/m). Adam normalizes coordinates by their moment scales,
so this magnitude argument does not transfer unchanged. These are early
scale estimates, not asymptotic laws for the trained trajectories.

The slow width-128 refinement could reflect weaker feature development or
different effective rates, alongside any landscape degeneracy. A useful
comparison records hidden-feature displacement, gate changes, readout floors,
and tangent-kernel drift, then varies layer rates or parameterization while
controlling initial predictions. Equal scalar learning rates, parameter-step
norms or one matched loss do not supply that control. Do not infer a universal
optimal width, or prescribe one width-rate power law, from this budget.

The distinction between these representation-relative widths and classical
sample-relative overparameterization is recorded in [the source audit](sources.md).
