# Clean-Room Audit — Conformal inference (`inference.py`, v0.3)

**Date.** 2026-07-07
**Auditor.** Claude Code (assistant), with the project maintainer.
**Code under audit.** [`src/augsynth_py/inference.py`](../src/augsynth_py/inference.py)
(public `conformal_pvalue`, `conformal_interval`; private `_post_statistic`,
`_permutation_pvalue`, `_FittedSC`), together with the estimator hook
`_conformal_null_residuals` on
[`Synth`](../src/augsynth_py/synth/classical.py) and
[`AugSynth`](../src/augsynth_py/synth/augmented.py), and the tests in
[`tests/unit/test_inference_*.py`](../tests/unit/) and
[`tests/validation_against_r/test_conformal.py`](../tests/validation_against_r/test_conformal.py).

**Verdict.** **PASS, with one open question.** Every step of the procedure maps
to a specific construct in CWZ 2021 §3, and the deterministic (`block`) p-value
path matches R `augsynth` **exactly** (`abs=1e-8`) at four separate null values,
which is the strongest parity result anywhere in this repository. Six deviations
are documented in §4; five are benign (normalisation cancellation, p-value
conventions, engineering heuristics not specified by the paper). The sixth —
holding λ fixed under the null refit for `AugSynth` (D-3) — is a genuine
methodological choice that should be stated in the paper rather than left
implicit. Separately, one **unexplained empirical result** (§6.2, tracked as
[`known-issues.md`](known-issues.md) I-1) prevents this audit from being
unconditional: the inverted confidence interval is markedly asymmetric about the
point estimate on the Basque panel, and that asymmetry has not been checked
against R.

> **Update (2026-07-20).** The §6.2 open question is resolved: the full
> $p(h_0)$ curve was compared against R `augsynth` on the same Basque panel
> (89 points, $h_0 \in [-12, 16]$) and agrees to one ulp
> (max $|p_\text{py} - p_\text{r}| = 4.4 \times 10^{-16}$, identical
> shift-counts at every point). The asymmetry is a property of CWZ test
> inversion on this panel, shared exactly by R — not an implementation defect.
> The sweep also revealed that the acceptance region is **non-contiguous** at
> $\alpha = 0.05$ on that panel (a rejected $1/T$-floor gap containing
> `att_`, identical in R), which makes the D-6 contiguity assertion of
> Recommendation 4 a demonstrated need rather than a precaution.
> See the update in §6.2, `methodology.md` §5.5, and
> `test_basque_pvalue_curve_matches_r_exact` in the parity harness. The §1.1
> disclosure condition in §6.1 is unaffected and remains open.

---

## 1. What "clean-room" means in this project

Policy unchanged from the `Synth` audit; see
[`docs/clean-room-audit-2026-05-16.md`](clean-room-audit-2026-05-16.md) §1 for
the full statement. In summary:

> All algorithms are implemented from the published papers, not by translating
> the R `augsynth` source code. (...) Reference implementations in R are used
> **only as oracles for validation tests**, never as a source to translate from.

Two operational criteria for PASS:

1. **Algorithm derived from a paper.** Each substantive piece maps to a
   construct in CWZ 2021 below (§3).
2. **Numerical agreement with R `augsynth`** within documented tolerances
   (§5).

### 1.1 Honest disclosure — R namespace surface used by the test harness

The parity harness calls R's **internal** function
`augsynth:::compute_permute_pval(...)` directly (triple-colon access), and the
`_conformal_null_residuals` docstrings name `compute_permute_test_stats`.
Knowing these internal names is more R-side surface than a pure black-box
oracle requires.

This is disclosed rather than resolved. Calling an internal function is still
*using R as an oracle* — it is a stronger oracle than the public wrapper,
because it isolates the exact quantity under test instead of forcing a
comparison through R's public reporting layer. But the audit cannot certify
from the code alone whether the R **source** was read to obtain the argument
signature and semantics, or whether these were obtained from R's help/namespace
listing.

**{{Maintainer to confirm}}:** how the internal signature was obtained. The
precedent is the `AugSynth` audit (§"Honest disclosure"), which records that R
source inspection *was* authorised during M4 debugging. If the same happened
here, say so in the same place. This does not invalidate the clean-room claim
for the *algorithm* — which traces to CWZ 2021 §3 independently, see §3 — but
the paper's blanket "no R source consulted" phrasing must be qualified
consistently across all three audits.

---

## 2. Sources consulted during this audit

- Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). *An Exact and Robust
  Conformal Inference Method for Counterfactual and Synthetic Controls.*
  JASA 116(536), 1849–1864. §3 (the permutation test) is the load-bearing
  section.
- [`docs/methodology.md`](methodology.md) §5, which already carries the
  equation-level mapping; this audit checks that mapping against the code
  rather than restating it.
- The module and estimator-hook source listed above.
- The unit suite (`test_inference_core.py`, `test_inference_pvalue.py`,
  `test_inference_interval.py`) and the R parity suite.

**Not consulted:** the R `augsynth` source for the *algorithm*. See §1.1 for
the test-harness caveat.

---

## 3. The algorithm, by paper equation

### 3.1 Refit-under-null residuals ↔ `_conformal_null_residuals` ↔ CWZ 2021 §3

**Paper.** CWZ 2021 §3 tests $H_0$: the post-treatment effect is constant and
equal to $h_0$. Exactness rests on the full-window residuals being
**exchangeable** under $H_0$. The treated series is adjusted,

$$
\tilde y_{1,t} = y_{1,t} - h_0 \cdot \mathbb{1}[t \in \text{post}],
$$

and the counterfactual is re-estimated on the adjusted series so that pre- and
post-period residuals are draws from a common law.

**Code.** `Synth._conformal_null_residuals(h0)` copies the outcome matrix,
subtracts `h0` from the treated unit's post positions, recomputes fixed-effect
offsets **over the full adjusted window**, solves the simplex QP on all $T$
periods, and returns `adjusted_treated - synthetic` over all $T$.
`AugSynth._conformal_null_residuals(h0)` does the same through the ω/γ
machinery of BFR 2021 §2, over all $T$ periods.

**Verdict.** Faithful. The critical detail — refitting over the **entire**
window rather than the pre-period — is implemented and is the reason parity
holds; §4 of `methodology.md` records that a no-refit shortcut was implemented
first, failed parity against R, and was rejected. That negative result is
strong evidence the current construction is the paper's.

**Non-obvious correctness point worth preserving.** The point estimate
(`att_`, `gap_`, `synthetic_`) is *not* recomputed by this hook. The two
quantities are deliberately different: the ATT is a pre-period-anchored
counterfactual, while the conformal residual is CWZ's full-window
exchangeability construct. A future contributor "simplifying" these into one
fit would silently break both.

### 3.2 Test statistic ↔ `_post_statistic` ↔ CWZ 2021 §3 ($S_q$)

**Paper.** The statistic $S_q(\hat u) = \bigl(\frac{1}{\sqrt{T_1}}\sum_{t \in
\text{post}} |\hat u_t|^q\bigr)^{1/q}$; the implementation targets $q = 1$.

**Code.** `sum(abs(post))` for two-sided; `sum(post)` / `-sum(post)` for the
one-sided variants.

**Verdict.** Faithful up to a constant; see D-1 for the omitted
$1/\sqrt{T_1}$ factor.

### 3.3 Permutation p-value ↔ `_permutation_pvalue` ↔ CWZ 2021 §3

**Paper.** The reference distribution is generated by permuting the residual
vector and re-reading the statistic on the fixed post positions. CWZ's
moving-block scheme uses cyclic shifts, which preserves serial dependence.

**Code.** `block` enumerates all $T$ cyclic shifts `np.roll(u, j)`,
$j = 0,\dots,T-1$, and returns $\#\{j : S_j \ge S_{\text{obs}}\}/T$. `iid`
draws `ns` random permutations and returns $(1 + \#)/(1 + ns)$.

**Verdict.** Faithful. The two different conventions are correct rather than
inconsistent — see D-2.

### 3.4 Test inversion ↔ `conformal_interval` ↔ CWZ 2021 §3

**Paper.** The $(1-\alpha)$ confidence set is the acceptance region
$\{h_0 : p(h_0) \ge \alpha\}$.

**Code.** Scores a finite grid of $h_0$ via `conformal_pvalue` and returns
`(min, max)` of the accepted points, with an auto-widening guard.

**Verdict.** Faithful in principle; the grid construction, widening policy, and
`min`/`max` recovery are engineering decisions the paper does not specify — see
D-4, D-5, D-6.

### 3.5 What we do *not* implement

- **Non-constant effect paths.** Only the constant-effect null is testable;
  CWZ 2021 permits general $h_0(t)$ trajectories. The public API takes a scalar
  `h0`.
- **Per-period intervals.** R `augsynth` reports per-period conformal
  intervals; this package reports an aggregate interval for the constant
  post-period effect only.
- **$q \ne 1$ statistics.** The $S_q$ family is implemented at $q = 1$ only.

These are scope limits, not deviations, and are correctly absent from the
public API rather than silently defaulted.

---

## 4. Deviations from paper specification

### 4.1 D-1 — $1/\sqrt{T_1}$ normalisation omitted

**Code.** `_post_statistic` returns the unnormalised sum.
**Paper.** CWZ 2021 §3 defines $S_q$ with a $1/\sqrt{T_1}$ factor.
**Why we deviate.** $T_1$ (the post-period length) is identical across every
member of the reference set, because the statistic is always evaluated on the
same fixed `post_mask`. The factor therefore cancels in the comparison
$S_j \ge S_{\text{obs}}$ and cannot change the p-value.
**Risk.** None for the p-value. It *would* matter if the statistic were ever
exposed publicly or compared across different post-window lengths; it is not.
**Action.** Keep. Documented in the function docstring.

### 4.2 D-2 — Asymmetric p-value conventions between `block` and `iid`

**Code.** `block` returns $\#/T$; `iid` returns $(1 + \#)/(1 + ns)$.
**Paper.** CWZ 2021 uses the finite-sample-valid "add-one" convention.
**Why we deviate — and why it is not actually a deviation.** Under `block` the
identity shift $j = 0$ *is* a member of the reference set, so the observed
statistic always ties itself and the count is at least 1; the add-one is
already built in, and applying it again would double-count. Under `iid` the
observed statistic is *not* among the `ns` random draws, so the explicit
add-one is required for finite-sample validity. The two expressions implement
the same convention under different reference-set constructions.
**Consequence worth knowing.** The `block` p-value has a hard floor of $1/T$
and is quantised to multiples of $1/T$. A $(1-\alpha)$ interval is only
meaningful when $T \gtrsim 1/\alpha$ — a 95% interval needs $T \ge 20$ total
periods. This is a property of the exact test, not a limitation of the code,
but it is the single most likely source of user confusion and is documented in
`methodology.md` §5.2.
**Action.** Keep. Correct as written.

### 4.3 D-3 — `AugSynth` holds λ fixed under the null refit

**Code.** `AugSynth._conformal_null_residuals(h0)` refits over the full
$h_0$-adjusted window **at the already-selected penalty `self.lambda_`**; it
does not re-run leave-one-out cross-validation under each null.
**Paper.** CWZ 2021 treats the counterfactual estimator as given and does not
address penalty selection; BFR 2021 does not address inference. Neither paper
specifies the interaction, so there is no equation to be faithful to.
**Why it matters.** A strict reading of "refit the estimator under the null"
would re-select λ under each $h_0$, since λ is part of the estimator. Holding
it fixed treats λ as a tuning constant fixed by the observed data rather than
as part of the fitted object. Both are defensible; they are not the same
procedure, and the difference is invisible to a user.
**Practical justification.** Re-selecting λ per grid point would multiply the
cost of a confidence interval by the CV grid size (already `grid_size` refits;
this would make it `grid_size × n_lambda`), and it would introduce a
discontinuous, data-dependent estimator into the null distribution, whose effect
on exactness is not established by either paper.
**Risk.** The p-value inherits whatever selection uncertainty λ carries. Since
λ is selected once on the observed data and then held fixed, the test conditions
on that selection.
**Action.** Keep, but **state it explicitly in the paper**. This is the one
deviation a methodologically minded reviewer is likely to ask about. It is
currently documented only in the method docstring.

### 4.4 D-4 — Confidence-interval grid heuristics

**Code.** Grid centred on `att_`, spanning $\pm 6\,\mathrm{sd}(\text{post gap})$
with `ddof=1`, `grid_size` points, `0.0` always unioned in; degenerate fallback
to `abs(center)` (or `1.0`) when `sd` is below `1e-6 × scale`.
**Paper.** CWZ 2021 specifies the acceptance region, not how to search it.
**Why we deviate.** A pointwise-only p-value forces a finite search. The
choices are pragmatic: centring on the point estimate puts the grid where the
region is expected; six standard deviations is a wide default; unioning `0.0`
guarantees the sharp null is always scored regardless of grid placement; the
degenerate fallback prevents a near-perfect pre-fit from collapsing the grid to
a single point.
**Risk.** Bound resolution is one grid spacing,
$2 \times \text{spread}/(\text{grid\_size} - 1)$ — documented in the docstring.
The `1e-6` threshold is a magic number tuned to solver noise.
**Action.** Keep. Consider exposing `spread_sd` as a keyword if users hit the
default.

### 4.5 D-5 — Auto-widening truncation guard

**Code.** If an accepted endpoint reaches a grid boundary, the span doubles
(up to 8 times) and re-scores; on persistent truncation a `UserWarning` is
emitted and the widest bounds are returned as a *lower bound* on the interval.
**Paper.** Not specified.
**Why we deviate.** The span is seeded from point-estimate dispersion but
scored by the full-window-refit acceptance region — a different and often much
wider quantity. Without the guard, `min`/`max` would clip silently to the grid
edges and **understate** the interval, which is anti-conservative and therefore
the dangerous direction to fail in.
**Verdict.** This is good defensive engineering and the failure mode it guards
is real. The warning text correctly tells the user the bounds are a lower
bound.
**Action.** Keep.

### 4.6 D-6 — `min`/`max` interval recovery assumes a contiguous acceptance region

**Code.** The interval is `(min(accepted), max(accepted))`.
**Paper.** The acceptance region is a set; the paper does not assert it is an
interval.
**Why it matters.** If the accepted set were ever non-contiguous, `min`/`max`
would silently fill the interior gap and report a *wider* region than the test
actually accepts. The source comment acknowledges this explicitly: contiguity
is *empirical, not guaranteed by convexity*, because under refit-under-null the
residuals themselves depend on $h_0$, so the statistic is not a fixed convex
function of $h_0$. The boundary-interiority check does not detect interior
gaps.
**Risk.** Conservative in direction (reports too wide, not too narrow), so it
fails safe. But it is unverified.
**Action.** Keep, and consider a cheap diagnostic: assert the accepted indices
are consecutive in the grid, warning if not. That would convert an unverified
assumption into a checked one at negligible cost. **Recommended follow-up.**

---

## 5. Validation

Pinned in [`tests/validation_against_r/test_conformal.py`](../tests/validation_against_r/test_conformal.py),
against R's internal `augsynth:::compute_permute_pval` on the
`GeoLift_PreTest` fixture:

| Test | Path | Result |
|---|---|---|
| `test_block_pvalue_h0_zero_matches_r_exact` | `block`, $h_0 = 0$ | **Exact**, `abs=1e-8` |
| `test_block_pvalue_nonzero_h0_matches_r_exact` | `block`, $h_0 \in \{-20, 10, 50\}$ | **Exact**, `abs=1e-8` |
| `test_iid_pvalue_matches_r_within_tolerance` | `iid`, `ns=5000` | Within `0.03` |
| `test_conformal_interval_brackets_accepted_nulls` | interval | Finite, `lo < hi`, brackets accepted nulls |

**Assessment of this evidence.**

- The `block` result is **strong**. Matching a deterministic p-value to `1e-8`
  at four separate $h_0$ values validates the entire pipeline at once — the
  null adjustment, the full-window refit, the fixed-effect handling, the
  statistic, and the cyclic-shift reference set. Any error in any of those
  would break it. Exercising nonzero $h_0$ specifically validates the *curve*
  $p(h_0)$, not just one point on it, and hence the acceptance region the
  interval inverts.
- The `iid` tolerance of `0.03` is **appropriate but weak**. Seeds do not
  transfer between R and Python, so exact agreement is impossible and only a
  Monte-Carlo band is assertable. It confirms no gross error; it would not
  catch a subtle one. See §6.2 / issue I-2.
- The interval test is **transitive, not direct**, and correctly labelled as
  such in `methodology.md` §5.4: R exposes no aggregate conformal CI, so there
  is no oracle. The argument is that the interval is pure inversion over a
  p-value proven exact, so correctness follows by construction. **This argument
  is sound for the acceptance region but does not cover the search
  machinery** — the grid placement (D-4), widening (D-5), and contiguity
  assumption (D-6) are untested against any external reference. §6.2 is
  precisely a symptom in that untested layer.

Unit coverage is separately substantial: `test_inference_core.py`,
`test_inference_pvalue.py`, and `test_inference_interval.py` within the 88-test
unit suite.

---

## 6. Verdict

### 6.1 PASS on the algorithm

The conformal implementation traces cleanly to CWZ 2021 §3. The refit-under-null
construction — the step that makes the method exact and the step a naive
implementation gets wrong — is correct, and the repository preserves the
negative result that establishes it (the rejected no-refit shortcut). The
deterministic parity result against R is the strongest in this codebase. The six
deviations in §4 are either provably immaterial (D-1, D-2) or clearly documented
engineering outside the paper's scope (D-4, D-5, D-6), with one genuine
methodological choice (D-3) that needs surfacing rather than fixing.

The phrasing "clean-room implementation" may be used for the conformal
inference module, **subject to the §1.1 disclosure being resolved consistently
with the `AugSynth` audit**.

### 6.2 Open question blocking an unconditional PASS

**The inverted confidence interval is markedly asymmetric about the point
estimate, and this is unexplained.** On the Basque panel a `Synth(fixedeff=False)`
fit gives ATT = −0.6915 with a 95% interval of (−3.2169, 7.6864) — the point
estimate sits far left of centre. The grid is *symmetric* about `att_` by
construction (§4.4), and the result is stable under grid refinement (widths
9.84 / 10.66 / 10.90 at `grid_size` 40 / 100 / 400), so this is neither grid
placement nor discretisation. The asymmetry is therefore in the $p(h_0)$ curve
itself.

That may well be a genuine property: under refit-under-null the residuals
depend on $h_0$ through the refit, so $p(h_0)$ has no symmetry guarantee. But
"may well be" is not an audit finding. Tracked as
[`known-issues.md`](known-issues.md) **I-1**, with the closing criterion being a
direct comparison against R's $p(h_0)$ curve on the same panel — which the
existing exact-parity harness makes cheap to run.

> **Update (2026-07-20) — resolved, property confirmed.** The closing
> criterion was executed: an 89-point sweep of $h_0 \in [-12, 16]$ on the same
> panel and fit matches R `augsynth`'s conformal p-value pointwise with
> max $|p_\text{py} - p_\text{r}| = 4.4 \times 10^{-16}$ (one ulp; the
> integer shift-counts $k/43$ are identical at all 89 points), so R inverts
> the identical acceptance region. The curve itself is non-monotone: it peaks
> at $h_0 \approx +4.5$ ($p = 36/43$), not at `att_`, holds a broad plateau on
> the right, collapses to the $1/T$ floor by $h_0 = -3.75$ on the left — and
> the acceptance region is in fact **non-contiguous**: a rejected window at
> the $1/T$ floor ($p = 1/43 < \alpha$) spans roughly $h_0 \in [-0.70, -0.35]$
> and **contains `att_` $= -0.6915$ itself**, flanked by $p = 3/43$ at
> $-0.75$ and $-0.25$; R returns the identical $1/43$ (e.g. at $h_0 = -0.5$).
> `conformal_interval`'s min/max extraction bridges this gap silently, so
> **D-6 is no longer hypothetical** — Recommendation 4's contiguity assertion
> is demonstrated to bind on a real panel and rises in priority.
> Mechanism: with `fixedeff=False` the donor simplex has no
> intercept and the Basque series sits near the top of the donor span, so
> raising the treated post-period (negative $h_0$) concentrates residual mass
> in the post window (share 0.79–0.96), while lowering it (positive $h_0$)
> lets the full-window refit spread misfit across the window (share ≈ 0.48),
> keeping the post block unexceptional among cyclic shifts. Near
> $h_0 \approx$ `att_` the same geometry rejects the truth: the refit drives
> the pre-window residuals to near zero while the post window keeps genuine
> dispersion, so the post block ranks first among all $T$ shifts — a local
> failure of residual exchangeability via pre/post dispersion imbalance.
> Full analysis in [`methodology.md`](methodology.md) §5.5; pinned by
> `test_basque_pvalue_curve_matches_r_exact`. I-1 has been removed from
> `known-issues.md` per that file's convention; **I-2 remains open** (its
> block side is now exactly verified on this panel by the same test, narrowing
> it to the `iid` magnitude question).

A second, lower-priority item is tracked as **I-2**: the `block` and `iid`
schemes give 0.186 vs 0.065 on that panel. The direction is expected (block is
conservative on autocorrelated series) but the magnitude is unverified, and the
`0.03` Monte-Carlo tolerance above is too loose to constrain it.

---

## 7. Recommendations

In descending order of value:

1. **Resolve I-1** by extending the existing parity harness to compare the whole
   $p(h_0)$ curve against R, not just four points. The machinery already exists;
   this is a loop over an existing test.
   **Done (2026-07-20)** — see the §6.2 update; property confirmed, exact
   parity across the curve, pinned in the harness on a second (Basque) fixture.
2. **Resolve §1.1** — state how the R internal signature was obtained, and
   align the disclosure wording across all three audits. The paper's
   "no R source consulted" claim must be qualified identically everywhere or it
   is a credibility risk.
3. **State D-3 in the paper.** Holding λ fixed under the null is the deviation
   most likely to draw a reviewer question.
4. **Add the contiguity assertion from D-6.** Converts an acknowledged
   unverified assumption into a checked one for a few lines of code.
   **Upgraded (2026-07-20):** no longer a precaution — the Basque acceptance
   region is non-contiguous (§6.2 update) and the current min/max extraction
   silently bridges a rejected gap that contains `att_`. Without the
   assertion/flag, `conformal_interval`'s bounds can be misread as a
   connected acceptance region.
5. **Tighten the `iid` parity test** if I-2 is to be closed — a synthetic series
   with analytically known autocorrelation would constrain the block/iid ratio
   far better than the current `0.03` band on real data.
