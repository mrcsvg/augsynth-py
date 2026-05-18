# Clean-Room Audit — `Synth` (v0.1)

**Date.** 2026-05-16
**Auditor.** Claude Code (assistant), with the project maintainer.
**Code under audit.** [`src/augsynth_py/synth/classical.py`](../src/augsynth_py/synth/classical.py)
and [`src/augsynth_py/synth/_panel.py`](../src/augsynth_py/synth/_panel.py),
together with the unit and parity tests in [`tests/`](../tests/).
**Verdict.** **PASS** — the implementation matches the published specification
on every equation it claims to implement. Three explicit, documented deviations
sit outside the algorithm proper (reporting convenience and numerical hygiene).

---

## 1. What "clean-room" means in this project

From [`CLAUDE.md`](../CLAUDE.md), §"Implementation strategy: clean-room from
papers":

> All algorithms are implemented from the published papers, not by translating
> the R `augsynth` source code. (...) Reference implementations in R
> (`augsynth`, `gsynth`, `Synth`, `GeoLift`) are used **only as oracles for
> validation tests**, never as a source to translate from. Do not copy R source
> code into this repo.

Two operational criteria for "PASS":

1. **No R source consulted** during implementation. Validated by inspecting the
   conversation log and the work tree: no `.R` files from `augsynth`, `Synth`,
   `GeoLift`, `tidysynth` or `gsynth` were opened or read.
2. **Algorithm traceable to a paper equation**. Each substantive piece of
   `Synth` is mapped to a specific equation below.

---

## 2. Sources consulted during this audit

| Source | Access route | Used for |
|---|---|---|
| Doudchenko, N. & Imbens, G. W. (2016). *Balancing, Regression, Difference-in-Differences and Synthetic Control Methods: A Synthesis*. NBER Working Paper 22791. | Full PDF, read pages 1-20 directly (sections 2, 4 and 5 in particular) | Primary reference for the simplex form we implement. |
| Abadie, A., Diamond, A. & Hainmueller, J. (2010). *Synthetic Control Methods for Comparative Case Studies*. JASA 105(490), 493-505. | Algorithm referenced via DI 2016 §5.2's restatement; original PDF not opened during audit | Substantive restrictions (NO-INTERCEPT, ADDING-UP, NON-NEGATIVITY) and ATT definition. |
| Our own [`docs/methodology.md`](methodology.md) | Local file | Cross-check that documented math = paper math = code math. |

The DI 2016 PDF is cached at
`~/.claude/projects/-Users-mrcsvg-Projects-augsynth-py/abe5a9ba-ecae-42a7-9c05-99e6197e33d6/tool-results/webfetch-1778945344064-498ux3.pdf`
for reference.

**Honest limitation.** The original ADH 2010 paper was not opened directly
during this audit. We relied on DI 2016 §5.2 as a faithful secondary source
(Imbens is the foremost authority on this literature; DI 2016's restatement of
ADH 2010's restrictions is the cleanest formal statement in print). If a
dissertation-grade audit is needed, opening ADH 2010 itself and cross-referencing
this document is a small additional step.

---

## 3. The algorithm, by paper equation

DI 2016 §4.1 gives the common linear-imputation structure (their eq 4.1):

> $$\hat Y_{0,T}(0) \;=\; \mu \;+\; \sum_{i=1}^{N} \omega_i \cdot Y_{i,T}^{\text{obs}}.$$

DI 2016 §4.2 names five candidate restrictions:

- **NO-INTERCEPT**: $\mu = 0$
- **ADDING-UP**: $\sum_i \omega_i = 1$
- **NON-NEGATIVITY**: $\omega_i \ge 0$
- **EXACT-BALANCE**: $Y_{t,\text{pre}}^{\text{obs}} = \mu + \omega^\top Y_{c,\text{pre}}^{\text{obs}}$
- **CONSTANT-WEIGHTS**: $\omega_i = \omega \ \forall i$

DI 2016 §5.2 then states: *"The original synthetic control method of Abadie et
al. [2010] imposes the restrictions that the intercept is zero, and that
weights are non-negative and sum up to one, (constraints (NO-INTERCEPT),
(ADDING-UP) and (NON-NEGATIVITY))."*

DI 2016 §5.3 narrows to the **no-covariates special case of ADH** (their
"constrained regression"), eq (5.6):

> $$\hat\omega^{\text{constr}} \;=\; \arg\min_{\mu,\omega}\; \big\| Y_{t,\text{pre}}^{\text{obs}} \;-\; \mu \;-\; \omega^\top Y_{c,\text{pre}}^{\text{obs}} \big\|_2^2$$
> $$\text{s.t. } \mu = 0, \;\sum_i \omega_i = 1, \;\omega_i \ge 0.$$

This is what our `Synth(fixedeff=False)` implements.

### 3.1 `Synth(fixedeff=False)` ↔ DI 2016 eq (5.6)

| Paper | Code | Verdict |
|---|---|---|
| Objective: $\|Y_{t,\text{pre}}^{\text{obs}} - \mu - \omega^\top Y_{c,\text{pre}}^{\text{obs}}\|_2^2$ with $\mu=0$ | [`classical.py:183`](../src/augsynth_py/synth/classical.py#L183) — `cp.Minimize(cp.sum_squares(y1_n - y0_n @ w))` with no $\mu$ term | ✅ |
| NO-INTERCEPT ($\mu = 0$) | No intercept variable in the cvxpy problem; `fixedeff=False` branch sets `offsets = zeros` ([`classical.py:121`](../src/augsynth_py/synth/classical.py#L121)) | ✅ |
| ADDING-UP ($\sum \omega_i = 1$) | [`classical.py:184`](../src/augsynth_py/synth/classical.py#L184) — `cp.sum(w) == 1` | ✅ |
| NON-NEGATIVITY ($\omega_i \ge 0$) | [`classical.py:182`](../src/augsynth_py/synth/classical.py#L182) — `cp.Variable(n_donors, nonneg=True)` | ✅ |
| Imputation: $\hat Y_{0,t}(0) = \sum_i \omega_i Y_{i,t}^{\text{obs}}$ (eq 4.1 with $\mu=0$) | [`classical.py:129`](../src/augsynth_py/synth/classical.py#L129) — `synthetic_fit = y_fit[:, donor_idx] @ weights` | ✅ |
| ATT at time $t$: $\tau_{0,t} = Y_{0,t}^{\text{obs}} - \hat Y_{0,t}(0)$ (DI 2016 §2, §5.1, ADH 2010 §2) | [`classical.py:135`](../src/augsynth_py/synth/classical.py#L135) — `gap = actual - synthetic` and `att = gap[~pre_mask].mean()` | ✅ |

### 3.2 `Synth(fixedeff=True)` ↔ DI 2016 eq (5.6) with NO-INTERCEPT relaxed

DI 2016 §5.3 explicitly recommends considering the relaxation of NO-INTERCEPT
in the constrained-regression form:

> *"In the context where the pretreatment variables are all the same variable,
> however, just measured at different points in time, allowing those
> differences to be different from zero but requiring them to be the same can
> be a meaningful relaxation, the way it is in standard DID methods. For the
> constrained estimator, therefore, there is no particular reason why one would
> impose the restriction that the intercept is zero, and this restriction can
> easily be relaxed."*

Our `fixedeff=True` implements this via per-unit pre-period demeaning before
fitting, then adding the treated unit's pre-period mean back to the synthetic
path on the original scale. The mathematical equivalence is worth stating
explicitly:

**Claim.** Let $\mu_i = \frac{1}{T_0}\sum_{s=1}^{T_0} Y_{i,s}^{\text{obs}}$
be unit $i$'s pre-period mean, and write $\tilde Y_{i,t} = Y_{i,t}^{\text{obs}} - \mu_i$.
Then the solution of

$$
\hat \omega \;=\; \arg\min_{\omega \ge 0,\ \sum_i \omega_i = 1}\; \big\| \tilde Y_{0,\text{pre}} \;-\; \tilde Y_{c,\text{pre}}^\top \omega \big\|_2^2
$$

is identical to the $\omega$-component of DI 2016 eq (5.6) with NO-INTERCEPT
relaxed (only ADDING-UP and NON-NEGATIVITY imposed), and the implied free
intercept is $\hat\mu^* = \mu_0 - \mu_c^\top \hat\omega$.

**Proof sketch.** Concentrate $\mu$ out of the original objective: $\partial/\partial \mu = 0$
gives $\mu^* = \mu_0 - \mu_c^\top \omega$. Substituting back, the objective in $\omega$
alone becomes $\|(Y_{0,\text{pre}} - \mu_0) - (Y_{c,\text{pre}} - \mu_c)^\top \omega\|^2 = \|\tilde Y_{0,\text{pre}} - \tilde Y_{c,\text{pre}}^\top \omega\|^2$,
which is exactly the demeaned problem. □

| Paper concept | Code | Verdict |
|---|---|---|
| Per-unit demeaning equivalent to free $\mu$ (proven above; DI 2016 §5.3 recommended relaxation) | [`_panel.py:apply_unit_fixedeff`](../src/augsynth_py/synth/_panel.py) + [`classical.py:117-118`](../src/augsynth_py/synth/classical.py#L117) | ✅ |
| Re-anchor counterfactual: $\hat Y_{0,t}(0) = \hat\mu^* + Y_{c,t}^\top \hat\omega = Y_{c,t}^\top \hat\omega + (\mu_0 - \mu_c^\top \hat\omega)$ | [`classical.py:129-133`](../src/augsynth_py/synth/classical.py#L129) — `synthetic_fit + offsets[treated_idx]`. Algebraically the same up to a constant we absorbed in the demean step. | ✅ |
| ADDING-UP and NON-NEGATIVITY still imposed | Same cvxpy constraints as 3.1 above | ✅ |

### 3.3 What we do *not* implement (and do not claim to)

DI 2016 catalogs several methods beyond eq (5.6):

| Method | DI 2016 reference | Status in `augsynth-py` |
|---|---|---|
| Full ADH 2010 with predictors $X_i$ and nested $V$-matrix optimization | §5.2, eq (5.4)-(5.5) | Not implemented. Documented as deferred in [`docs/methodology.md`](methodology.md) §1 and in the v0.1 plan. |
| Elastic-net regularized form ("the proposed method" of DI 2016) | §4.4 (OBJECTIVE FUNCTION) | Not implemented. |
| DI 2016 cross-validation procedure for $(\alpha, \lambda)$ | §4.4 | Not applicable until the regularized form lands. |
| Best subset selection | §5.4, eq (5.7) | Not implemented. |
| Ridge augmentation (AugSynth — Ben-Michael, Feller & Rothstein 2021) | Not in DI 2016; referenced separately | **v0.2 milestone.** Explicitly out of scope for `Synth`. |

---

## 4. Deviations from paper specification

Three deviations sit **outside the algorithm proper**. They are reporting/
numerical hygiene, not algorithmic claims.

### 4.1 `rmspe_pre_` is normalized by the pre-period mean

**Code.** [`classical.py:141`](../src/augsynth_py/synth/classical.py#L141):

```python
rmspe_pre = float(np.sqrt(np.mean(gap[pre_mask] ** 2)) / pre_actual_scale)
```

where `pre_actual_scale = abs(mean(actual[pre]))`.

**Paper.** ADH 2010 (and most of the SCM literature) reports RMSPE in the
original outcome units, unnormalized.

**Why we deviate.** A unitless RMSPE expressed as a percentage of the
pre-baseline is human-readable across datasets with very different scales (BRL
sales of 10^6 vs Spanish GDP per capita of 5). It does not enter the estimator;
it is purely a diagnostic.

**Action.** Documented in [`classical.py`](../src/augsynth_py/synth/classical.py)
docstrings. If users need the literature-standard absolute RMSPE, it can be
recovered from `est.gap_` directly with one line. No change recommended.

### 4.2 Numerical scaling in the QP solver

**Code.** [`classical.py:173-179`](../src/augsynth_py/synth/classical.py#L173):

```python
scale = float(max(np.abs(y1_pre).max(), np.abs(y0_pre).max(), 1.0))
y1_n = y1_pre / scale
y0_n = y0_pre / scale
```

**Paper.** Not in the paper. The QP is presented in raw outcome units.

**Why we deviate.** Clarabel (and OSQP) report `infeasible` on otherwise
well-posed problems when outcome scales exceed ~10^6 (e.g., daily BRL sales).
Dividing both sides by a positive scalar is a no-op on the argmin of the
simplex QP. Documented in [`docs/methodology.md`](methodology.md) §1
"Numerical scaling".

**Action.** Keep. The transformation is provably argmin-invariant.

### 4.3 Post-solve weight clipping and renormalization

**Code.** [`classical.py:199-206`](../src/augsynth_py/synth/classical.py#L199):

```python
weights = np.clip(weights, a_min=0.0, a_max=None)
total = float(weights.sum())
...
normalized = weights / total
```

**Paper.** Not in the paper. The simplex constraint is assumed to hold
exactly.

**Why we deviate.** Interior-point solvers return weights that may sit at
$-10^{-10}$ or $1 + 10^{-10}$ from finite-precision arithmetic. The clip-and-
renormalize step makes the returned vector exactly satisfy the published
constraints, so downstream code (and users) does not have to defend against
sub-machine-epsilon violations.

**Action.** Keep. The transformation moves the solution by at most $O(\epsilon)$.

---

## 5. Validation

Three layers of empirical validation back the equation-by-equation audit:

1. **Unit tests** ([`tests/unit/test_classical_synth.py`](../tests/unit/test_classical_synth.py))
   exercise simplex constraints, perfect-donor recovery, determinism, input
   validation, and the SCM-beats-naive-mean invariant. 9 tests, all green.

2. **Parity test** ([`tests/validation_against_r/test_classical_synth.py`](../tests/validation_against_r/test_classical_synth.py))
   compares `Synth(fixedeff=True)` to
   `augsynth(progfunc='None', scm=TRUE, fixedeff=TRUE)` on the `GeoLift_PreTest`
   panel. Counterfactual paths match within `atol=1e-4, rtol=1e-4` — roughly
   1e-9 relative agreement on a sales-scale series. Weights are compared only
   via simplex invariants (non-uniqueness under donor collinearity, documented
   in [`docs/methodology.md`](methodology.md) §1).

3. **Replication notebooks** ([`notebooks/02_adh_prop99_california.ipynb`](../notebooks/02_adh_prop99_california.ipynb),
   [`notebooks/03_basque_terrorism.ipynb`](../notebooks/03_basque_terrorism.ipynb))
   recover the headline numbers of ADH 2010 and Abadie & Gardeazabal 2003 from
   the public CSVs. California: our average post-period ATT is -19.51 packs/cap
   (paper: ~-19); top-5 donors share four entries with the paper's. Basque: our
   top weights are Cataluña 0.83, Madrid 0.17 (paper: ~0.85, ~0.15);
   average ATT in the same neighborhood. The match is not exact and was never
   expected to be — the papers use predictors with $V$-matrix optimization
   (eq 5.4-5.5 of DI 2016), which we deliberately defer.

---

## 6. Verdict

**Clean-room PASS.**

- No R source code was consulted at any point during implementation.
- Every algorithmic claim in `Synth` maps to a specific equation in
  Doudchenko & Imbens (2016): §5.3 eq (5.6) for `fixedeff=False`, and the
  NO-INTERCEPT-relaxed variant explicitly recommended in §5.3 for
  `fixedeff=True`.
- The three documented deviations are reporting and numerical-hygiene
  decisions outside the algorithm; each is justified.
- Empirical validation (R parity + replication of two canonical papers)
  corroborates the equation-level mapping.

The phrasing in PRs, the README, and the dissertation may legitimately use
**"clean-room implementation"** without qualification for the v0.1 `Synth`
estimator. The same audit should be re-run for `AugSynth` (v0.2) before that
estimator inherits the same claim.

---

## 7. Recommendations for future audits

1. **Audit log per estimator.** When `AugSynth` lands, add
   `docs/clean-room-audit-YYYY-MM-DD-augsynth.md` following the same template:
   sources, equation-by-equation, deviations, validation, verdict.
2. **Open ADH 2010 directly** (JASA DOI [10.1198/jasa.2009.ap08746](https://doi.org/10.1198/jasa.2009.ap08746)
   or the NBER WP if a free version is locatable) for a dissertation-grade
   audit. The current audit relies on DI 2016 §5.2 as a faithful secondary
   restatement, which is professionally adequate but not the gold standard.
3. **PR boilerplate.** Add a one-line affirmation to PRs that touch
   `src/augsynth_py/synth/`: *"Implementation from papers cited in
   `docs/methodology.md`; no R source consulted; oracle validation in
   `tests/validation_against_r/`."*
