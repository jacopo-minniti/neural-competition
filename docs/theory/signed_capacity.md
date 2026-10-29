# A quantitative signed-capacity obstruction

21 October 2026. Derived during the sign-survival follow-up. This extends the
exact-representation sign quota into a population approximation bound. It is
a conditional obstruction, not a theorem predicting the training trajectory.

## Second Hermite moment

Let x be standard Gaussian in d dimensions, and let a scalar target y(x) have
finite second moment. Define the symmetric matrix

\[
T_y=\mathbb E[y(x)(xx^\top-I)].
\]

For unit u, Gaussian integration gives

\[
\mathbb E[\operatorname{ReLU}(u^\top x)(xx^\top-I)]
  =\frac{1}{\sqrt{2\pi}}uu^\top.
\]

For example, the component along u is E[ReLU(Z)(Z²−1)] = 1/√(2π);
all perpendicular and mixed components vanish by independence and symmetry.
Consequently, for f=Σ c_r ReLU(u_rᵀx),

\[
T_f=\frac{1}{\sqrt{2\pi}}\sum_r c_r u_ru_r^\top=P-N,
\]

where P and N are positive semidefinite. If k+ and k− are the numbers of
nonzero positive and negative coefficients, rank(P)≤k+ and rank(N)≤k−.
It follows that T_f has at most k+ positive and k− negative eigenvalues.
This statement is independent of the directions or the size of other weights.

The degree-two Hermite projection of g=y−f is

\[
g_2(x)=\tfrac12\{x^\top(T_y-T_f)x-\operatorname{tr}(T_y-T_f)\},
\quad \mathbb E[g_2^2]=\tfrac12\|T_y-T_f\|_F^2.
\]

Orthogonality gives E[(y−f)²]≥E[g₂²]. Let λ_i⁺ be the positive eigenvalues of
T_y in decreasing order, and λ_i⁻ the absolute negative eigenvalues, also in
decreasing order. Spectral approximation with the above inertia limits yields

\[
\boxed{\mathbb E[(y-f)^2]\ge
\frac12\sum_{i>k_+}(\lambda_i^+)^2+
\frac12\sum_{i>k_-}(\lambda_i^-)^2.}
\]

Proof of the last step: by the eigenvalue/Frobenius inequality for symmetric
matrices, the squared Frobenius distance bounds the squared distance between
ordered eigenvalues. After its first k+ positive eigenvalues, the approximant
has nonpositive eigenvalues, so each unmatched positive target eigenvalue costs
at least its square. Apply the analogous argument to the negative tail. The
two target index sets are disjoint. Truncating the target spectrum attains this
distance over the class of all matrices with those inertia limits. It need not
attain the bound over complete ReLU functions: higher Hermite components also
contribute to error.

An output intercept or arbitrary linear skip has zero second Hermite moment,
so neither can evade this bound. More generally, a biased unit
a ReLU(wᵀx+b), with rho=||w||>0, contributes
a rho φ(b/rho) uuᵀ, where φ is the standard Gaussian density. Its matrix still
has the output coefficient's sign. Thus the inertia bound also holds with
hidden biases, though the numerical experiments here train bias-free networks.

## The current eight-feature teacher

For y=Σ β_i ReLU(v_iᵀx), T_y=Σ β_i v_iv_iᵀ/√(2π). In each orthogonal pair
plane, write β+=(b), β−=−rb, r>0, and v₊ᵀv₋=cos(2θ). The two eigenvalues are

\[
\lambda_\pm=\frac{b}{2\sqrt{2\pi}}
\left(1-r\ \pm\sqrt{(1-r)^2+4r\sin^2(2\theta)}\right).
\]

For nondegenerate pairs one is positive and one negative. Each is repeated
four times across the four orthogonal planes. At half-angle 30°, r=3 and
the repository's data-seed-9027 normalization, λ+=0.1007473887451444.
With enough negative units, the MSE bound is

    (4−k+)₊ × 0.00507501816948262.

The training target variance is one. Each missing positive role therefore
forces at least **0.5075 percentage points of population MSE**. Four missing
roles force 2.0300 points. These are MSE units; half-MSE is half as large.
The population target variance is not assumed numerically identical to the
finite training variance used for normalization.

The general formula depends on the signed spectrum, not raw teacher-feature
correlation. It applies even where the old initial-evidence sign test fails.
It does not prove that a run will lose a role, that four roles suffice, or that
the same scalar predicts a width-benefit boundary.

## From capacity loss to dynamics

The residual field in Section 3 governs the coefficients c_r, and thus whether
the available signed capacity disappears. Under equal unit-rate gradient flow,
D_r=||w_r||²−a_r² is constant. If D_r>0, the coordinate
z_r=asinh(a_r/√D_r) obeys

\[
\dot z_r=H_f(u_r)/\sqrt m,
\qquad z_r(t)=z_r(0)+\frac1{\sqrt m}\int_0^tH_{f_s}(u_r(s))\,ds.
\]

An initial sign test misses the duration of adverse evidence and its later
release. This integral supplies an exact hitting condition under the flow
assumptions, but using the observed future H would be retrospective accounting.
A predictive theory must approximate that evolving field *before* the crossing.
If D_r<0, a_r cannot cross zero under the same flow. Finite steps, momentum,
Adam and explicit projection require their actual update dynamics instead.

The norm invariant is established background, not a new result:
[Du, Hu and Lee (2018)](https://arxiv.org/abs/1806.00900).
The related [early-alignment analysis of Boursier and Flammarion (2025)](https://www.jmlr.org/papers/v26/24-1523.html)
shows how small initialization can reduce diversity and obstruct fitting in its
own settings. Those assumptions do not establish convergence or failure here.
The present contribution within this repository is the explicit spectral
approximation certificate and its combination with a targeted sign-crossing test;
no claim of literature novelty is made.

## Interpretation and limits

This is a route from an observable state constraint to objective units:
residual-driven sign loss can reduce signed capacity, which then forces error.
The [experiment](../reports/sign_survival.md) tests whether preventing that loss
actually changes learning. A certified floor is not a uniquely mediated fraction
of the width advantage. Wide and narrow models differ in their directions,
rates of feature formation and residual fields as well as their sign counts.

T_y can in principle be estimated from labels and inputs without a known teacher
under Gaussian input assumptions. Empirical estimation of its spectrum needs
uncertainty control; a naive finite-sample matrix is not an exact population
certificate. General inputs, deeper networks and cross-entropy need new arguments.
