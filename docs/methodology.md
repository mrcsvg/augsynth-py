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

**Code location:** [`src/augsynth_py/synth/augmented.py`](../src/augsynth_py/synth/augmented.py)
— class `AugSynth`, plus the private helpers `_ridge_augment`,
`_period_demean_pre`, `_fit_at_lambda`, `_loo_cv_lambda`, and
`_build_auto_lambda_grid`.

**Estimator.** AugSynth corrects the bias of classical synthetic control when
the pre-period fit is imperfect, by composing the simplex SCM weights
$\omega \in \Delta^J$ with a ridge augmentation $\gamma \in \mathbb{R}^J$
fit on the SCM residual. The effective AugSynth weights are
$\omega + \gamma$ (BFR 2021 §2.4 Lemma 1). They are real-valued and need
not lie on the simplex.

Given the pre-period donor matrix $Y_{0,\text{pre}} \in \mathbb{R}^{T_0 \times J}$,
the pre-period treated vector $Y_{1,\text{pre}} \in \mathbb{R}^{T_0}$, and
the simplex SCM solution $\omega$, the ridge augmentation solves

$$
\gamma \;=\; \arg\min_{\gamma \in \mathbb{R}^J} \big\| Y_{1,\text{pre}} \;-\; Y_{0,\text{pre}} (\omega + \gamma) \big\|_2^2 \;+\; \lambda \|\gamma\|_2^2
$$

with closed form

$$
\gamma \;=\; \big(Y_{0,\text{pre}}^\top Y_{0,\text{pre}} + \lambda I_J\big)^{-1} Y_{0,\text{pre}}^\top \big(Y_{1,\text{pre}} - Y_{0,\text{pre}} \omega\big).
$$

The counterfactual at every period $t$ is then

$$
\hat Y_{1,t}(0) \;=\; Y_{0,t}^\top (\omega + \gamma) \;+\; \mu_1
$$

where $\mu_1$ is the treated unit's pre-period mean (added back when
`fixedeff=True`). The bias decomposition `weights_ = scm_weights_ +
augmentation_weights_` is the BFR Lemma 1 surface and is asserted in
unit test U3 at `atol=1e-10`.

**Implementation notes.**

`AugSynth` composes the existing classical `Synth` estimator for $\omega$
(via `Synth._solve_simplex_qp`) with the ridge step in `_ridge_augment`,
then projects across the full period. This composition is the per-design-doc
D2 architectural choice: separate class, not inheritance, since the fitted-
object surface differs (`AugSynth` adds `lambda_`, `lambda_cv_path_`,
`synthetic_scm_`, `ridge_correction_`, `scm_weights_`,
`augmentation_weights_`).

Before the SCM and ridge solves, both donor and treated pre-period matrices
are **period-demeaned** by `_period_demean_pre`: at each pre-period column,
the donor-mean across donors is subtracted from every unit's value. This
absorbs time fixed effects in the ridge step. The simplex SCM is invariant
to this subtraction (since $\omega$ sums to 1, the period-mean cancels in
the objective); the ridge $\gamma$ is not. This is the M3.5 fix documented
in [`docs/clean-room-audit-2026-05-26-augsynth.md`](clean-room-audit-2026-05-26-augsynth.md)
§4.1 (deviation D-1). Without it, the augmented path diverges from R
`augsynth(progfunc='Ridge')` by orders of magnitude exceeding the D8
tolerance; with it, Test A passes strict on all three documented λ regimes.

The ridge is stated above as a $J \times J$ primal system. R `augsynth`
uses the equivalent $T_0 \times T_0$ dual; the two formulations produce
identical $\gamma$ under the Woodbury identity
$(X^\top X + \lambda I_p)^{-1} X^\top = X^\top (X X^\top + \lambda I_n)^{-1}$.
The primal is preferred here because the $J \times J$ matrix is the natural
home of the donor-weight interpretation that BFR §2.4 Lemma 1 calls out.

**Cross-validation.** Per BFR 2021 §3.2, $\lambda$ is selected by leave-one-
out CV over pre-treatment periods. For each candidate $\lambda$ and each
held-out pre-period $t$, `_loo_cv_lambda` refits $\omega$ and $\gamma$ on
the remaining $T_0 - 1$ rows and records the squared prediction error at
$t$. The argmin over the grid is returned. The default grid (per design-doc
D6) is

$$
\Lambda \;=\; \mathrm{logspace}(-4, 4, 50) \cdot \mathrm{Var}(Y_{1,\text{pre}}),
$$

constructed by `_build_auto_lambda_grid`. The variance scaling mirrors the
numerical-scaling rationale used in classical `Synth` (see §1 above):
without it, a grid fixed in raw units misbehaves on series whose natural
scale differs by orders of magnitude. A `UserWarning` fires when CV selects
a grid endpoint, indicating the auto-grid was too narrow.

**Validation.** Three layers:

1. **Unit tests** ([`tests/unit/test_augsynth.py`](../tests/unit/test_augsynth.py)) cover the U1–U10 inventory from the design doc: closed-form `_ridge_augment` (U1), effective-weights-can-be-negative (U2), Lemma 1 decomposition (U3), large-λ limit collapses onto `Synth` (U4), small-λ OLS limit in the period-demeaned space (U5), boundary warning (U6), determinism (U7), perfect-donor regime (U8), input validation (U9), and CV interior-selection (U10). All green.
2. **Parity tests** ([`tests/validation_against_r/test_augsynth.py`](../tests/validation_against_r/test_augsynth.py)) compare `AugSynth` to `augsynth(progfunc='Ridge', scm=TRUE, fixedeff=TRUE)` on `GeoLift_PreTest`. Test A pins $\lambda$ at three regimes ($\{0.1, 1.0, 10.0\} \times \mathrm{Var}(Y_{1,\text{pre}})$) and asserts strict path agreement plus per-donor effective-weight agreement; Test B pins an 11-entry log-spaced grid and asserts the CV-chosen $\lambda$ matches R within one grid cell. On `GeoLift_PreTest`, both implementations select the same cell.
3. **Replication notebook.** Optional M6 milestone; not required for the v0.2 audit.

See [`docs/clean-room-audit-2026-05-26-augsynth.md`](clean-room-audit-2026-05-26-augsynth.md)
for the full deviation ledger (D-1 period demeaning; D-2 R-side CV argument
convention; D-3 `ridge_correction_` tolerance refinement; U4 / U5 unit-test
rewrites) and the empirical max-delta table.

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
