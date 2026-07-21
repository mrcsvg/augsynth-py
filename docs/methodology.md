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

## 3. Multi-treated pooling and L2 imbalance (P0.1)

Two cross-cutting facilities added for the geo-experimentation surface. Neither
introduces a new estimator: both are shared infrastructure consumed by
`Synth` (§1) and `AugSynth` (§2). They live in
[`src/augsynth_py/synth/_panel.py`](../src/augsynth_py/synth/_panel.py).

### 3.1 Multi-treated pooling

**References:** Doudchenko & Imbens (2016) and Ben-Michael, Feller & Rothstein
(2021) both frame synthetic control for a treated *unit*; the pooled-group
extension for a block treatment applied to several units at a common
intervention time is the standard reduction used by the R `augsynth` package.
R `augsynth` is the parity **oracle** here, not the definitional source.

**Code location:** [`long_to_wide`](../src/augsynth_py/synth/_panel.py) in
`_panel.py`. `treated` accepts either a single unit value or an iterable of unit
values.

**Method.** When `treated` names a group $G = \{g_1, \dots, g_m\}$, the treated
units are collapsed into a single fit target: their **elementwise mean** across
units at each time period,

$$
\bar y_{G,t} \;=\; \frac{1}{m} \sum_{g \in G} y_{g,t},
$$

placed at column 0 of the outcome matrix, with all $g \in G$ removed from the
donor pool. The estimator then fits this pooled series exactly as it would a
single treated unit. Consequently `actual_`, `synthetic_`, `att_`, and
`att_pct_` all refer to the treated-**group mean**; for a flow outcome the total
incremental effect is `att_` $\times\, m \times T_{\text{post}}$ (number of
treated units times number of post periods). `units_[0]` is the comma-joined,
string-sorted label of the treated group (e.g. `[2, 10]` labels as `"10,2"`).

**Why collapsing before demeaning is valid.** `Synth` subtracts each unit's
pre-period mean (`apply_unit_fixedeff`) before the QP. It would be equally
correct to demean every treated unit first and then average the demeaned
columns; the shipped code averages first and demeans once. These give the
identical result because **per-unit demeaning is a linear map applied
identically to each column, so it distributes over the cross-column average**
(collapse-then-demean equals demean-then-collapse). Averaging first is the
cheaper of the two equivalent orders. (This is a distributivity property of a
shared linear operator over an average — not a claim that two different
operations commute.)

**Validation.**
[`tests/validation_against_r/test_multitreated.py`](../tests/validation_against_r/test_multitreated.py)
designates `chicago` and `portland` as a hypothetical treated group on
`GeoLift_PreTest` and asserts the pooled counterfactual path matches
`augsynth`'s block-treatment fit strictly. Unit coverage for the pooling logic
(mean collapse, donor-pool removal, label formatting, order invariance) is in
[`tests/unit/test_multitreated.py`](../tests/unit/test_multitreated.py).

### 3.2 L2 imbalance diagnostics

**Paper:** Ben-Michael, Feller & Rothstein (2021), §3 — the pre-treatment fit /
imbalance quantity. The exact scalar reported (`l2_imbalance`,
`scaled_l2_imbalance`) mirrors what R `augsynth` surfaces; R is the parity
oracle, not the definitional source.

**Code location:** [`imbalance`](../src/augsynth_py/synth/_panel.py) in
`_panel.py`, exposed as the fitted attributes `l2_imbalance_` and
`scaled_l2_imbalance_` on both `Synth` and `AugSynth`.

**Definitions.** For a treated pre-period vector $y_{1,\text{pre}}$, donor
pre-period matrix $Y_{0,\text{pre}} \in \mathbb{R}^{T_0 \times J}$, and fitted
weights $w$,

$$
\text{l2} \;=\; \big\| y_{1,\text{pre}} - Y_{0,\text{pre}}\, w \big\|_2,
\qquad
\text{scaled} \;=\;
\frac{\big\| y_{1,\text{pre}} - Y_{0,\text{pre}}\, w \big\|_2}
     {\big\| y_{1,\text{pre}} - Y_{0,\text{pre}}\, w_{\text{unif}} \big\|_2},
\qquad w_{\text{unif}} = \tfrac{1}{J}\mathbf{1}.
$$

`l2_imbalance_` is the plain (unnormalized) residual norm in the space the
weights were fit in. `scaled_l2_imbalance_` divides it by the imbalance of the
uniform $1/J$ donor average. Because the scaled version is a **ratio of two
norms in the same space**, it is invariant to any constant normalization of the
norm — R's `l2_imbalance` uses the plain, unnormalized norm (so the two agree at
ratio 1.0 on the scaled quantity regardless of that convention).

**Zero-denominator (IEEE) convention.** When the uniform baseline fits
$y_{1,\text{pre}}$ exactly, the denominator is 0. The code follows the IEEE
semantics that an unguarded R division produces: denominator $> 0$ gives
$\text{l2}/\text{denom}$; denominator $= 0$ with $\text{l2} > 0$ gives `inf`;
$0/0$ gives `nan`.

**Space and weights per estimator.** `imbalance` is space-agnostic — the caller
supplies $y_{1,\text{pre}}$, $Y_{0,\text{pre}}$, and $w$ in whatever space the
weights were fit:

- **`Synth`** computes it in the **unit-fixed-effect-demeaned** space
  (`apply_unit_fixedeff`, §1) using the **simplex** weights $w^\ast$.
- **`AugSynth`** computes it in the **period-demeaned** space
  (`_period_demean_pre`, §2) using the **effective** weights $\omega + \gamma$
  (SCM plus ridge augmentation), matching R `augsynth(progfunc='Ridge')`. The
  choice of *weights* (effective, not SCM-only $\omega$) is what the parity test
  pins; the choice of *space* is immaterial for AugSynth because $\gamma$ sums to
  0 whenever $\lambda > 0$ and $\omega$ sums to 1, so the period-mean term
  cancels in the residual — only the weights choice is load-bearing.

**Validation.**
[`tests/validation_against_r/test_imbalance.py`](../tests/validation_against_r/test_imbalance.py)
asserts both `l2_imbalance_` (raw) and `scaled_l2_imbalance_` match
`augsynth` for `Synth` (`progfunc='None'`) and `AugSynth` (`progfunc='Ridge'`)
on `GeoLift_PreTest`. The AugSynth case is the one that pins the effective-weights
choice: had R used SCM-only $\omega$, both the raw and scaled numbers would
disagree. Unit coverage (the IEEE inf/nan branches, the ratio invariance) is in
[`tests/unit/test_imbalance.py`](../tests/unit/test_imbalance.py).

---

## 4. Generalized synthetic control (Xu, 2017)

**Paper:** Xu, Y. (2017). Generalized Synthetic Control Method: Causal
Inference with Interactive Fixed Effects Models. *Political Analysis*, 25(1),
57–76.

**Status:** Out of scope for v0.1; planned for v0.2.

---

## 5. Conformal inference for synthetic controls (Chernozhukov, Wüthrich & Zhu, 2021)

**Paper:** Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and
Robust Conformal Inference Method for Counterfactual and Synthetic Controls.
*Journal of the American Statistical Association*, 116(536), 1849–1864
(hereafter CWZ 2021).

**Code location:** [`src/augsynth_py/inference.py`](../src/augsynth_py/inference.py)
— public functions `conformal_pvalue` and `conformal_interval`, built on the
private permutation core (`_post_statistic`, `_permutation_pvalue`) and the
estimator hook `_conformal_null_residuals` on both `Synth` (§1) and `AugSynth`
(§2). Neither function is a new estimator: both operate on an already-fitted
`Synth` or `AugSynth`.

**Estimator surface.** Two functions operating on any fitted estimator:

- `conformal_pvalue(fit, h0=0.0, *, permutation_type="block", side="two-sided",
  ns=1000, rng=None) -> float` — the conformal p-value for the null of a
  **constant** post-period effect equal to `h0`.
- `conformal_interval(fit, *, alpha=0.05, grid_size=100, permutation_type="block",
  side="two-sided", ns=1000, rng=None) -> tuple[float, float]` — a
  $(1-\alpha)$ confidence interval for that constant effect, by test inversion.

### 5.1 Refit-under-null residual construction

CWZ 2021 (§3) tests $H_0$: the post-treatment effect is constant and equal to
$h_0$. The test is exact because, under $H_0$, the full-window residuals are
**exchangeable**. To realize that exchangeability the treated unit's
post-period outcomes are adjusted by $h_0$,

$$
\tilde y_{1,t} \;=\;
\begin{cases}
y_{1,t} & t \in \text{pre}, \\
y_{1,t} - h_0 & t \in \text{post},
\end{cases}
$$

and the synthetic control is **refit over the entire window** — all $T$
periods, not the pre-period only (for `fixedeff` the demeaning is likewise taken
over the full $h_0$-adjusted window). The residual handed to the permutation
core is the full-length $T$-vector $\hat u_t = \tilde y_{1,t} - \hat y_{1,t}(0)$
from that refit. This is the `_conformal_null_residuals(h0)` hook on each
estimator.

The full-window refit is the load-bearing step: only a control fit on the
$h_0$-adjusted *full* window makes the pre- and post-period residuals draws from
a common law, which is what CWZ 2021 §3 requires for the permutation
distribution to be valid in finite samples. Refitting on the pre-period alone
and reusing those weights out-of-sample leaves post-period residuals with a
different distribution than the pre-period ones, breaking exchangeability.

**The point estimate stays pre-period-fit; only inference refits.** The reported
`att_` / `gap_` / `synthetic_` come from the pre-period counterfactual (§1, §2)
and are *not* recomputed here — the ATT is a pre-period-anchored quantity. The
refit-under-null is exclusively an inference construct: it produces exchangeable
residuals for the permutation test and never touches the point estimate.

> **Note (rejected shortcut).** An initial "no-refit" approximation — reuse the
> pre-period weights and set the post-period residual to $\text{gap}_t - h_0$ —
> was implemented first and **rejected**. Those residuals are not exchangeable
> (the pre-period residuals are in-sample, the post-period ones are not), and the
> resulting p-values did **not** match R `augsynth`. Do not reintroduce it as the
> default. (A no-refit *fast approximation* is a possible future opt-in for the
> `geolift-py` simulation loop, never the exact path.)

### 5.2 Test statistic and permutation schemes

On the full-window residual vector $\hat u$, the CWZ test statistic is evaluated
on the **fixed post positions** (`_post_statistic`):

$$
S(\hat u) \;=\;
\begin{cases}
\sum_{t \in \text{post}} |\hat u_t| & \text{`side="two-sided"`}, \\
\sum_{t \in \text{post}} \hat u_t & \text{`side="right"`}, \\
-\sum_{t \in \text{post}} \hat u_t & \text{`side="left"`}.
\end{cases}
$$

CWZ's $1/\sqrt{T_1}$ normalization is constant across permutations, so it
cancels in the rank and is omitted. Two permutation schemes produce the p-value:

- **`block`** (default, deterministic): the reference set is the $T$ cyclic
  shifts $\texttt{np.roll}(\hat u, j)$, $j = 0, \dots, T-1$, with the statistic
  re-read on the fixed post positions. Since the observed statistic ($j=0$) is
  always in the set, $p = \#\{j : S_j \ge S_\text{obs}\}\,/\,T$. This is the
  moving-block scheme of CWZ 2021 §3. **Granularity floor:** $p$ is a multiple
  of $1/T$ and peaks near $1/T$, so a $(1-\alpha)$ CI is only feasible when
  $T \gtrsim 1/\alpha$ (e.g. a 95% interval needs $T \ge 20$ total periods).
- **`iid`**: `ns` random permutations drawn from `rng`, with the
  finite-sample-valid convention
  $p = \bigl(1 + \#\{S_\text{perm} \ge S_\text{obs}\}\bigr)\,/\,(1 + ns)$.
  Requires a non-`None` `rng`.

### 5.3 Confidence interval by test inversion

`conformal_interval` inverts the two-sided test: the $(1-\alpha)$ interval is
the acceptance region

$$
\mathrm{CI}_{1-\alpha} \;=\; \{\, h_0 : \texttt{conformal\_pvalue}(fit, h_0) \ge \alpha \,\},
$$

approximated on a finite grid centred at `att_` and spanning
$\pm 6\,\mathrm{sd}(\text{post gap})$ (with `0.0` always unioned in so the sharp
null is scored). Each grid point is a **separate refit-under-null**, so a CI
costs roughly `grid_size` refits. Only the **two-sided** test can be inverted
into a bounded interval — a one-sided acceptance region is a half-line, so any
other `side` raises `ValueError`.

**Auto-expansion / truncation guard.** The span is seeded from the
*point-estimate* gap dispersion, but each point is scored by the *full-window
refit* acceptance region, a different and often wider quantity (tight donors give
a small `sd`, yet a low-power conformal test accepts a broad range of $h_0$). If
an accepted endpoint reaches a grid boundary the span is doubled (centre,
`grid_size` and the unioned `0.0` preserved) and re-scored, up to 8 doublings.
If truncation persists at the cap a `UserWarning` is emitted and the widest
computed bounds are returned as a lower bound on the true interval. Under `block`
the $p(h_0)$ curve peaks near a well-specified $h_0$ and decays to a **floor of
$1/T$** (the $j=0$ identity shift always ties itself) at extreme $h_0$; when
$T \le 1/\alpha$ that floor keeps $p(h_0) \ge 1/T \ge \alpha$ at *every* $h_0$,
so the acceptance region is **unbounded** and lands in exactly this
truncation/`UserWarning` branch.

This is distinct from an **empty** acceptance region, which returns `(nan, nan)`
and which widening cannot fix. Empty requires the *peak* of $p(h_0)$ to fall
below $\alpha$ — driven by residual **non-exchangeability** (a poor / trending
fit), and only *possible* when $1/T < \alpha$ (i.e. $T > 1/\alpha$); it is **not**
caused by, and does not occur under, the small-$T$ ($T \le 1/\alpha$) floor
regime above.

### 5.4 Validation

Parity is pinned in
[`tests/validation_against_r/test_conformal.py`](../tests/validation_against_r/test_conformal.py):

- **`block` p-value — exact.** Matches R `augsynth`'s conformal inference
  **exactly** at multiple $h_0$, for both `Synth` (`progfunc='None'`) and
  `AugSynth` (`progfunc='Ridge'`) on `GeoLift_PreTest`. The refit-under-null is
  what makes this hold; the rejected no-refit shortcut did not.
- **`iid` p-value — statistical.** Agrees with R `augsynth`'s conformal
  p-value within a documented Monte-Carlo tolerance (`0.03`) for a fixed `rng`
  and large `ns`; seeds do not transfer between R and Python, so exact parity is
  not expected on this path.
- **Confidence interval — transitive.** R `augsynth` exposes no *aggregate*
  conformal CI (only per-period intervals), so there is no direct oracle for
  `conformal_interval`. It is validated transitively: the interval is pure test
  inversion over `conformal_pvalue`, and $p(h_0)$ matches R exactly at every
  tested $h_0$, so the inverted region is correct by construction.
- **Second fixture — the $p(h_0)$ curve on the Basque panel.** The block
  p-value also matches R exactly on the Abadie & Gardeazabal (2003) Basque
  panel (`Synth(fixedeff=False)`), at probes spanning the full structure of
  the curve — both $1/T$ floors, the left cliff, the rejected gap at the
  point estimate, and the off-centre peak (§5.5). An 89-point sweep of
  $h_0 \in [-12, 16]$ agreed with
  max $|p_\text{py} - p_\text{r}| = 4.4 \times 10^{-16}$ (one ulp; identical
  shift-counts $k/43$ at every point). This fixture needs only the `augsynth`
  R package, decoupling conformal parity from `GeoLift_PreTest`.

### 5.5 Interval asymmetry is a property of test inversion (I-1)

On the Basque panel (classical `Synth(fixedeff=False)`, treatment 1975) the
point estimate is `att_ = -0.6915` yet the 95% block interval is
`(-3.2169, 7.6864)` — skewed strongly positive, with `att_` far left of centre.
This was tracked as known-issue I-1 (opened 2026-07-07, resolved 2026-07-20)
under the suspicion of a defect in the grid construction, the null-refit, or
the interval extraction. It is none of these: the full $p(h_0)$ curve matches
R `augsynth` pointwise to one ulp (see §5.4), so R inverts the *identical*
acceptance region; the grid is symmetric about `att_` (§5.3); and the
extraction faithfully reports the accepted region's min/max. The asymmetry is
the method's.

What the curve actually looks like on this panel, identically in both
implementations:

- $p(h_0)$ is **non-monotone** and peaks at $h_0 \approx +4.5$ with
  $p = 36/43$ — nowhere near `att_`. It holds a broad high plateau over
  roughly $h_0 \in [+1, +5]$, decays through $p = 5/43$ at $+7.5$, and hits
  the $1/T$ floor at $+8.25$.
- On the left it collapses from $p = 17/43$ at $-3.0$ to the floor by
  $-3.75$ — a cliff.
- The acceptance region is **non-contiguous**: a rejected window at the
  $1/T$ floor sits essentially *at the point estimate* — $p = 1/43 \approx
  0.023 < \alpha$ for $h_0 \in$ roughly $[-0.70, -0.35]$ (0.05-step scan), a
  gap that contains `att_` $= -0.6915$ itself, flanked by $p = 3/43$ at
  $-0.75$ and $-0.25$. R returns the identical $1/43$ at the swept points
  (e.g. $h_0 = -0.5$). `conformal_interval` reports min/max of the accepted
  set, so the published $(-3.2169, 7.6864)$ silently **bridges** this gap:
  the non-contiguity caveat documented at the extraction step is not
  hypothetical — it binds on this very panel (audit D-6).

**Mechanism.** The test statistic is the *rank* of the post-window block sum
$\sum_{\text{post}} |\hat u_t|$ among all $T$ cyclic shifts of the full-window
residual vector, and the residuals come from a refit against a donor pool
whose convex hull cannot reach the treated series from below (no intercept
with `fixedeff=False`; the Basque Country sits near the top of the donor
span). Testing a *negative* $h_0$ raises the adjusted treated post-period
further above the hull: no simplex combination follows, residual mass
concentrates in the post window (share $0.79$–$0.96$ of $\sum_t |\hat u_t|$),
the post block ranks extreme, and $p$ dies. Testing a *positive* $h_0$ lowers
the treated post-period into the donors' range: the full-window refit
compromises between pre and post, spreading misfit across the window (post
share $\approx 0.48$, *below* the post period-share $23/43 \approx 0.53$), so
the post block ranks unexceptional among shifts and $p$ stays high across the
plateau.

The rejection *at* the well-specified null is the same geometry seen from the
other side: near $h_0 \approx$ `att_` the full-window refit drives the
pre-window residuals to near zero — the Basque pre-fit is exceptionally tight
— while the post window keeps its genuine dispersion, so the post block sum
ranks first among all $T$ cyclic shifts and $p$ sits on the $1/T$ floor.
Exchangeability of the full-window residuals, the premise of CWZ exactness,
fails *locally* through pre/post dispersion imbalance; only once $h_0$ is
far enough from the data to degrade the pre-fit as well does the post block
stop standing out — which is what produces the acceptance plateau away from
the point estimate.

Three practical consequences. First, a conformal interval from test inversion
need not be centred on — nor even *accept* — the point estimate: here
$p(\texttt{att\_}) = 1/43 < \alpha$, and `att_` lies inside the reported
bounds only because min/max bridges the rejected gap. Second, on panels like
this the reported bounds must be read as the **envelope** of a non-contiguous
acceptance set; the audit's D-6 recommendation (assert/flag contiguity at
extraction) is thereby upgraded from prudence to demonstrated need. Third,
the Basque interval remains usable as a published illustrative example — the
entire curve, including the asymmetry, the plateau, and the rejected gap, is
shared exactly with the R reference — provided the text presents it as the
envelope of the acceptance region rather than as a connected interval.
Parity is pinned by `test_basque_pvalue_curve_matches_r_exact` in
[`tests/validation_against_r/test_conformal.py`](../tests/validation_against_r/test_conformal.py).

### 5.6 Cost note for `geolift-py`

Because inference now refits the synthetic control once per $h_0$ (≈ `grid_size`
refits per CI), conformal inference is **no longer cheap**. The earlier
`geolift-py` simulation fast-path assumption that conformal inference reuses the
pre-period fit (no refit) is **void**. A no-refit fast approximation remains a
possible future opt-in for the power-simulation loop, but it is not the default
and is not the exact CWZ path.

---

## 6. Power analysis and market selection (orchestration layer)

**Reference:** GeoLift R package documentation and underlying use of ASCM
for simulation-based power calculation.

**Code location:** `src/augsynth_py/power/` *(not yet implemented)*

**Method.** Given a panel and a candidate set of treatment markets / dates,
inject a synthetic effect of varying magnitude, refit ASCM, and record
detection rates. Repeat over an embarrassingly parallel grid via `joblib`.

The minimum detectable effect (MDE) is the smallest injected effect for
which detection rate exceeds a chosen power threshold (typically 0.80) at a
chosen significance level (typically 0.10 for one-sided geo-tests).
