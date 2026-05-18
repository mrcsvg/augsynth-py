# Methodology

This document maps the code in `src/augsynth_py/` to the published literature.
Every public estimator should have an entry here citing the paper, the
specific equations, and any deviations from the original formulation.

This is the document the dissertation chapter is built from later. Keep it
honest and current.

## 1. Classical synthetic control (Abadie, Diamond & Hainmueller, 2010)

**Paper:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic
Control Methods for Comparative Case Studies: Estimating the Effect of
California's Tobacco Control Program. *Journal of the American Statistical
Association*, 105(490), 493–505.

**Code location:** [`src/augsynth_py/synth/classical.py`](../src/augsynth_py/synth/classical.py)
— class `Synth`.

**Estimator.** Given a treated unit $j=1$ and $J$ donor units, the synthetic
control assigns weights $w \in \Delta^J$ (the simplex) that minimize the
pre-period MSE between the treated outcome and the weighted donor outcomes.

Formally, with $X_1 \in \mathbb{R}^k$ denoting the pre-period feature vector
of the treated unit and $X_0 \in \mathbb{R}^{k \times J}$ the donor matrix:

$$
w^* = \arg\min_{w \in \Delta^J} (X_1 - X_0 w)^\top V (X_1 - X_0 w)
$$

where $V$ is a positive semi-definite weighting matrix over predictors. In
the canonical Abadie-Diamond-Hainmueller (2010) implementation $V$ is itself
optimized via a nested minimization (see Abadie et al., Section 2.2).

**Implementation notes.**

The shipped `Synth` estimator implements the **outcome-only, simplex-constrained
formulation** analyzed in Doudchenko & Imbens (2016) and adopted by the R
`augsynth` package with `progfunc = "None"`. Concretely, $X_1$ and $X_0$ are the
pre-treatment outcome trajectories themselves (no predictor optimization) and
$V$ is the identity, so the optimization reduces to

$$
w^* = \arg\min_{w \geq 0,\; \mathbf{1}^\top w = 1} \big\| y_1^{\text{pre}} - Y_0^{\text{pre}} w \big\|_2^2.
$$

The QP is solved with `cvxpy` using the Clarabel solver. Returned weights are
clipped at zero and renormalized so the simplex constraints hold to machine
precision.

When `fixedeff=True` (the default), each unit's pre-treatment mean is subtracted
from its full trajectory before the QP, then the treated unit's pre-period mean
is added back to the counterfactual. This is the analogue of
`augsynth(fixedeff = TRUE)` and is the form used throughout the showcase
notebook.

The Abadie-Diamond-Hainmueller (2010) form with predictor selection and nested
$V$ optimization is **not** implemented in this estimator; it will live in a
separate class if and when needed by an MVP user. It is not on the critical path
for the augmented variant (Ben-Michael, Feller & Rothstein 2021), which builds
directly on the simplex form above.

**Validation.** [`tests/validation_against_r/test_classical_synth.py`](../tests/validation_against_r/test_classical_synth.py)
asserts numerical agreement with
`augsynth(progfunc = 'None', scm = TRUE, fixedeff = TRUE)` on the `GeoLift_PreTest`
panel. The **counterfactual path** is compared strictly (`atol=1e-4, rtol=1e-4`,
i.e. ~1e-9 relative agreement on sales-scale data).

**Weights are not compared element-wise.** The simplex QP can have a flat
minimum along an affine subspace of equivalent convex combinations whenever
donor outcome trajectories are linearly dependent in the pre-period — common on
real geo panels with multiple similar markets. R `augsynth` and our Clarabel
solver land on different argmins of the *same* minimum, producing different
weight vectors that map to the same counterfactual. The invariants tested are
therefore the constraints (`w >= 0`, `sum(w) == 1`) and the synthetic path
itself.

**Numerical scaling.** Internally `_solve_simplex_qp` divides the outcome
matrix by its max absolute value before calling Clarabel. The simplex
constraint is scale-invariant, so this is purely a conditioning step — without
it, Clarabel falsely reports infeasibility on data with natural scales above
~10^6 (e.g. daily sales in BRL).

---

## 2. Augmented synthetic control (Ben-Michael, Feller & Rothstein, 2021)

**Paper:** Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The
Augmented Synthetic Control Method. *Journal of the American Statistical
Association*, 116(536), 1789–1803.

**Code location:** `src/augsynth_py/synth/augmented.py` *(not yet implemented)*

**Estimator.** ASCM corrects the bias of classical SC when pre-period fit is
imperfect. The augmented estimator is

$$
\hat\tau_{\text{ASCM}} = \hat\tau_{\text{SC}}
                     - \big(\hat m(X_1) - \hat m(X_0) w^*\big)
$$

where $\hat m$ is an outcome model fit on the donor pool. The default choice
in Ben-Michael et al. is ridge regression; the bias correction term then has
a closed form.

**Implementation notes.** *(to be filled in when the estimator lands)*

---

## 3. Generalized synthetic control (Xu, 2017)

**Paper:** Xu, Y. (2017). Generalized Synthetic Control Method: Causal
Inference with Interactive Fixed Effects Models. *Political Analysis*, 25(1),
57–76.

**Status:** Out of scope for v0.1; planned for v0.2.

---

## 4. Conformal inference for synthetic controls (Chernozhukov, Wuthrich & Zhu, 2021)

**Paper:** Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021). An Exact and
Robust Conformal Inference Method for Counterfactual and Synthetic Controls.
*Journal of the American Statistical Association*, 116(536), 1849–1864.

**Code location:** `src/augsynth_py/inference/conformal.py` *(not yet implemented)*

**Method.** Provides finite-sample valid p-values for the null of no
treatment effect by permuting residuals across the pre-period.

---

## 5. Power analysis and market selection (orchestration layer)

**Reference:** GeoLift R package documentation and underlying use of ASCM
for simulation-based power calculation.

**Code location:** `src/augsynth_py/power/` *(not yet implemented)*

**Method.** Given a panel and a candidate set of treatment markets / dates,
inject a synthetic effect of varying magnitude, refit ASCM, and record
detection rates. Repeat over an embarrassingly parallel grid via `joblib`.

The minimum detectable effect (MDE) is the smallest injected effect for
which detection rate exceeds a chosen power threshold (typically 0.80) at a
chosen significance level (typically 0.10 for one-sided geo-tests).
