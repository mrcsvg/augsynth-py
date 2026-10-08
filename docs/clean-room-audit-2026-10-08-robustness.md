# Clean-Room Audit — Robustness checks (`robustness.py`, v0.6)

**Date.** 2026-10-08
**Auditor.** Claude Code (assistant), with the project maintainer.
**Code under audit.** [`src/augsynth_py/robustness.py`](../src/augsynth_py/robustness.py)
(public `placebo_in_space`, `leave_one_out`, `PlaceboInSpaceResult`,
`LeaveOneOutResult`; private `_RefittableSC`, `_refit`, `_run_all`, `_rmspe`,
`_ratio`, `_surface_warnings`), the panel helpers `wide_to_long` and
`periods_to_series` in
[`src/augsynth_py/synth/_panel.py`](../src/augsynth_py/synth/_panel.py), and
the tests in [`tests/unit/test_robustness.py`](../tests/unit/test_robustness.py),
[`tests/unit/test_panel.py`](../tests/unit/test_panel.py) and
[`tests/validation_against_r/test_robustness.py`](../tests/validation_against_r/test_robustness.py).

**Verdict.** **PASS, qualified.** Both diagnostics map to specific constructs
in ADH 2010 §III.B, ADH 2015 §4.4 and Abadie 2021 §5–6, and the equation-level
mapping in [`methodology.md`](methodology.md) §7 checks against the code. The
numerical evidence is layered rather than single-shot (§5): every refit is
validated **by composition** against R `augsynth` on the same modified panel,
and the placebo aggregation — ratios, discard rule, rank p-value — is
**exact** against `SCtools::mspe.test` fed with our fitted series (ratios
within 6.3e-16 relative, p-values equal). What is *not* validated is stated
in §6.2: `leave_one_out` has no external oracle beyond composition, and the
third validation layer (a genuine `Synth`/ipop placebo run) is a
characterization, not a parity claim. The qualification on the verdict is the
§1.1 disclosure: unlike the `Synth` and conformal audits, this module's
*harness* was built after reading two R `SCtools` source files, so "no R
source consulted" cannot be said of v0.6 without qualification.

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
   construct in ADH 2010, ADH 2015 or Abadie 2021 below (§3).
2. **Numerical agreement with an R reference** within documented tolerances
   (§5) — here R `augsynth` for the refits and R `SCtools` for the aggregation.

### 1.1 Honest disclosure — R source read for the oracle harness

Two files of the R `SCtools` package (0.3.3.1) were **read** while this
module was built: `mspe.test.R` and `generate.placebos.R`. They were read for
two purposes, both on the *validation* side:

- to pin the **aggregation convention** that the parity test asserts against
  — the treated unit counted in the denominator of the rank p-value, the
  `>=` comparison, the discard rule as `pre_j**2 / pre_1**2 >= k` with the
  treated row exempt, the treated row appended last in `mspe.test`'s output;
- to learn the **`tdf` object layout** (`df` column order, `mspe.placs`,
  `loss.v`, `t0`/`t1`/`tr`, `names.and.numbers`, `n`, `treated.name`, and
  that `SCtools:::is_tdf` is `inherits(x, "tdf")`) so that
  `_build_tdf_in_r` could hand `mspe.test` *our* fitted series and isolate
  the aggregation arithmetic from any solver.

They were **not translated**. The Python module contains no code derived from
them: the post/pre RMSPE ratio and the rank p-value are one-line formulas in
ADH 2010 §III.B and Abadie 2021 §5, and the discard rule is the ADH 2010
practice (the 20× / 5× / 2× pre-MSPE cut-offs). The
in-house prototype that preceded `robustness.py` is the NR-35 notebook
([`notebooks/_build_nr35.py`](../notebooks/_build_nr35.py), Atos 5 and 7):
a placebo loop with the treated unit removed from the pool and a rank
p-value, and a leave-one-out loop over the donors, both written from the
papers before `SCtools` was opened.

This disclosure is **recorded, not resolved**, because it cannot be resolved:
the files were read. The reading was scoped (two files, oracle-side) and is
of the same kind as the `AugSynth` audit's M4 inspection (see
[`clean-room-audit-2026-05-26-augsynth.md`](clean-room-audit-2026-05-26-augsynth.md)
§1). The cross-audit alignment therefore becomes:

- **`Synth` (v0.1):** no R source consulted — unqualified.
- **`AugSynth` (v0.2):** R source inspected once, scoped and disclosed
  (M4 period-demeaning diagnosis, `fit_ridgeaug_formatted`); not translated.
- **Conformal inference (v0.3):** no R source consulted — unqualified;
  internal R functions are *called* as a stronger oracle.
- **Robustness checks (v0.6):** R `SCtools` source read (`mspe.test.R`,
  `generate.placebos.R`) to fix the aggregation convention and the `tdf`
  layout for the oracle harness; not translated; no module code derived
  from it.

Any blanket "no R source consulted" phrasing in the paper must be qualified
for `AugSynth` **and** for the robustness module. Per the house rule, the
claim is "implemented from the papers; R read only where disclosed, never
translated" — not "no R source consulted".

A secondary disclosure, same spirit: `test_characterize_against_real_sctools_placebos`
(layer 3, §5) drives `Synth::dataprep`, `Synth::synth` and
`SCtools::generate.placebos` through their *public* interfaces; the two
harness details it needed (sorted `controls.identifier`, `Sigf.ipop = 4`)
were found empirically while setting the run up, not read from source, and
are explained inline in the test.

---

## 2. Sources consulted during this audit

- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic Control
  Methods for Comparative Case Studies.* JASA 105(490), 493–505. §III.B (the
  in-space placebo and the post/pre RMSPE ratio) and Fig. 7 are the
  load-bearing passages.
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). *Comparative Politics
  and the Synthetic Control Method.* AJPS 59(2), 495–510. §4.4 (leave-one-out
  robustness).
- Abadie, A. (2021). *Using Synthetic Controls: Feasibility, Data
  Requirements, and Methodological Aspects.* JEL 59(2), 391–425. §5 (the rank
  p-value and the argument against discarding poor-fit placebos) and §6
  (leave-one-out).
- R `SCtools` 0.3.3.1 source, `mspe.test.R` and `generate.placebos.R` —
  **oracle-side only**, see §1.1.
- [`docs/methodology.md`](methodology.md) §7, which already carries the
  equation-level mapping; this audit checks that mapping against the code
  rather than restating it.
- [`docs/plans/2026-10-06-robustness-v0.6.md`](plans/2026-10-06-robustness-v0.6.md),
  whose "Decisions made while planning" section (items 1–16) is condensed
  into §4 here, and the design doc `2026-10-06-robustness-design.md` beside
  it. Note: `docs/plans/` is gitignored, so external readers will not see
  either file; these cross-references are for in-repo navigation, and §4
  reproduces everything the audit relies on.
- The module, the panel helpers and the three test files listed above.

**Not consulted:** the R `augsynth` source; the R `Synth` source; any
`SCtools` file other than the two named.

---

## 3. The algorithm, by paper equation

### 3.1 Panel reconstruction and deepcopy refit ↔ the "same estimator, modified pool" premise

**Paper.** Both ADH 2010 §III.B and ADH 2015 §4.4 describe the diagnostics
as re-running *the same estimation* with a different treated unit or a
reduced donor pool. The estimator is held fixed; only the data changes.

**Code.** `wide_to_long` (the inverse of `long_to_wide`) rebuilds the long
panel from the fitted object's stored outcome matrix `_y_matrix`, its
`units_` and `periods_`; `_refit` then calls `.fit(...)` on a
`copy.deepcopy` of the fitted estimator with the modified panel. Because the
copy carries the constructor state, an `AugSynth` with a CV-selected penalty
re-runs the CV in every refit, and `AugSynth(lambda_=x)` keeps `x`; `Synth`'s
`fixedeff` carries over. Nothing in the module touches the solver.

**Verdict.** Faithful, and the right construction: it is the one that makes
layer-1 validation (§5) a *composition* argument — each refit is an
estimator call the parity suite already covers, on a panel R `augsynth` can
be given verbatim.

**Non-obvious correctness point.** The period column of the rebuilt panel
must keep the user's time dtype, or a `Datetime("ns")` fit would be refit on
epoch integers and `treatment_time_` would no longer compare. `periods_to_series`
exists for this (decision 8 in §4); `test_panel.py` pins the round trip.

### 3.2 In-space placebo ↔ `placebo_in_space` ↔ ADH 2010 §III.B

**Paper.** For treated unit $1$ and donors $j = 2, \dots, J+1$, each donor is
in turn designated treated at the same $T_0$ and a synthetic control is
built for it from the remaining donors; the real treated unit is removed
from the placebo pools. The result is the "spaghetti" plot of placebo gaps
against the treated gap.

**Code.** `_split_units` separates the treated column from the donors;
`pool = wide_to_long(y[:, donor_idx], ...)` builds the donor-only panel; one
`_refit` per donor with that donor as `treated`. The treated row and every
successful placebo's `actual - synthetic` go into the long `gaps` frame.

**Verdict.** Faithful. Removing the real treated unit from the pool is the
ADH 2010 practice; it is also the detail that makes the p-value a
*permutation-shaped* statistic rather than a strictly exact one (§3.4,
decision 16).

### 3.3 Test statistic ↔ `_rmspe`, `_ratio` ↔ ADH 2010 Fig. 7

**Paper.** The post/pre RMSPE ratio,

$$
r_j = \frac{\mathrm{RMSPE}_{\text{post}}(j)}{\mathrm{RMSPE}_{\text{pre}}(j)},
\qquad
\mathrm{RMSPE}_{\cdot}(j) = \sqrt{\tfrac{1}{|\cdot|}\sum_{t \in \cdot}\bigl(Y_{jt} - \hat Y^{N}_{jt}\bigr)^2},
$$

on the raw outcome scale, which rewards a large post-treatment gap only
relative to the pre-period fit.

**Code.** `_rmspe(actual, synthetic, mask)` over `pre_mask_` and `~pre_mask_`;
`_ratio(pre, post)` returns `post / pre` with two deliberate conventions:
`inf` for a perfect pre-fit with post error, `nan` for $0/0$.

**Verdict.** Faithful. `SCtools::mspe.test` reports $r_j^2$ (a ratio of
MSPEs); the square is monotone, so ranks and p-value coincide and layer 2
compares $r_j^2$. The `inf`/`nan` conventions are outside the paper and are
handled explicitly in the aggregation (decision 10).

### 3.4 Rank p-value ↔ `placebo_in_space` aggregation ↔ Abadie 2021 §5

**Paper.** The p-value is the rank of the treated ratio among all $J+1$
units, the treated unit included:

$$
p = \frac{\#\{\, j \in \{1, \dots, J+1\} : r_j \ge r_1 \,\}}{J + 1},
\qquad p \ge \frac{1}{J+1}.
$$

**Code.** `kept_ratios` holds every successful, non-discarded row with a
defined ratio (treated row included); `pvalue = sum(kept >= statistic) /
len(kept)`; `rank` is `1 + #{kept > ratio}`, so ties share the smaller rank.
This is `SCtools::mspe.test`'s `sum(ratios >= treated) / nrow` convention
and is what layer 2 pins exactly.

**Verdict.** Faithful, with the exactness claim **qualified** (decision 16):
the statistic has the form of a permutation p-value, but excluding the real
treated unit from every placebo pool means the treated unit's synthetic
control uses $J$ donors and each placebo's uses $J-1$, so the units are not
exchangeable in the strict sense. `methodology.md` §7.1 and the docstring
say so; the hypothesis is one about the design, not about sampling.

### 3.5 Discard rule ↔ `mspe_limit` ↔ ADH 2010 practice

**Paper.** ADH 2010 drop placebos whose pre-period MSPE exceeds the treated
unit's by a factor ($20\times$, $5\times$, $2\times$) before comparing gaps.
Abadie 2021 §5 argues the ratio statistic already penalises a poor pre-fit,
so discarding is unnecessary and changes $J$ (hence the floor of $p$).

**Code.** `mspe_limit=None` by default (Abadie 2021 stance). When given, a
placebo is flagged `discarded` if `(pre_j / pre_1) ** 2 >= mspe_limit`;
discarded rows stay in `ratios` with a null `rank`, and the treated row is
never discarded.

**Verdict.** Faithful to ADH 2010 as a *default-off* option. The rule is
written `(pre_j/pre_1)**2` where `SCtools` writes `pre_j**2/pre_1**2`; the
two differ by a few ulps (decision 14), which cannot flip a decision on the
fixture (nearest placebos at 5.43 and 1.88 for limits 5 and 2) but keeps
this from being a bit-match. Recorded as a follow-up (§7).

### 3.6 Leave-one-out ↔ `leave_one_out` ↔ ADH 2015 §4.4 / Abadie 2021 §6

**Paper.** Iteratively re-estimate the synthetic control omitting one of the
donors that received a positive weight, and report how the estimate moves;
if it hinges on a single donor, dropping that donor exposes it.

**Code.** For each donor with $|w_j| > \texttt{tol}$ in the full fit
(`donors="nonzero"`; `"all"` and an explicit list are also accepted), the
pool minus that donor is rebuilt via `wide_to_long` and refit. The result
carries `att_full`, a per-donor frame (`dropped_unit`, `weight`, `att`,
`delta_att`, `rmspe_pre`, `lambda_`, `error`), the envelope
`att_range = (min, max)` over refits with a finite ATT, `sign_stable`,
`most_influential = argmax |delta_att|` (ties to the first dropped donor),
and the `paths` frame of counterfactuals.

**Verdict.** Faithful. Two readings go beyond the paper and are deliberate:
"positive weight" is read as $|w_j| > \texttt{tol}$ so that `AugSynth`'s
negative ridge-corrected weights count as contributing (BFR 2021 §2.4), and
the result *names* the most influential donor rather than reporting a range
alone — the NR-35 application is the motivating case, where the
less-negative end of the range was a single donor carrying the 2019
Brumadinho dam failure. The `tol` default (decision 3) is a solver fact, not
a methodological choice.

### 3.7 What we do *not* implement

- **In-time placebos** (backdating $T_0$; ADH 2015 §4.3). The power module
  already runs placebo-in-time windows for a different purpose; a dedicated
  diagnostic is not in v0.6.
- **Permutation confidence sets from placebo gaps.** Only the ratio-rank
  p-value is returned; inversion to an interval is conformal inference's
  job (`inference.py`).
- **Covariate-aware refits.** The estimators are outcome-only, so the
  diagnostics are too.
- **Leave-one-out over pre-periods.** Only donors are dropped.
- **A public refit Protocol.** `_RefittableSC` stays private this release
  (decision 11).

These are scope limits, not deviations, and are correctly absent from the
public API rather than silently defaulted.

---

## 4. Deviations and decisions

Condensed from "Decisions made while planning" in
[`docs/plans/2026-10-06-robustness-v0.6.md`](plans/2026-10-06-robustness-v0.6.md),
numbering preserved; item 17 was added at the final pre-tag review and is
recorded here only. "Design" is `2026-10-06-robustness-design.md`.

| # | Decision | Reason |
|---|----------|--------|
| 1 | `ratios` gains `is_treated`, `lambda_`, `error`; `att` gains `lambda_`, `error`. | `lambda_` makes the penalty policy observable per refit (the design said it "follows the prototype" with no way to see it); `error` is the `on_error="record"` channel, mirroring `power`. |
| 2 | `att_full` carried on `LeaveOneOutResult`. | `delta_att` is self-explanatory without the original fit at hand. |
| 3 | `tol` parameter on `leave_one_out`, default `1e-6`. | "Nonzero" needs a threshold: Clarabel at default tolerances leaves 1.2e-7 to 2.7e-7 on weights that are zero in exact arithmetic (measured on the test panel), so `1e-8` selected every donor. `1e-6` is negligible against weights that sum to 1. Not in the design; corrected during Task 2. |
| 4 | Oracle split: `SCtools` validates aggregation on a `tdf` built from our series; refits validate by composition against `augsynth`; a real ipop run is characterized (ρ > 0.9), not asserted. | `SCtools` consumes `Synth` objects whose ipop / nested-$V$ optimiser does not bit-match our simplex QP, so direct parity at repository tolerance is unavailable. Already recorded in the design. |
| 5 | Donor labels are strings (`fit.units_`); explicit `donors=[...]` compared after `str()`. | Consequence of `long_to_wide` stringifying unit values; an integer unit `7` is reported as `"7"`. Documented in the module docstring. |
| 6 | Period column dtype in `gaps` / `paths` is the user's time dtype (Date, Datetime at any resolution, Int64). | Joins back to the user's panel without casting. Via `periods_to_series` (see 8). |
| 7 | The treated row is never discarded by `mspe_limit`; errored placebos leave the denominator of $p$. | The treated unit is the reference the rule is defined against. `SCtools` has no error channel, so the second point is an extension. |
| 8 | `_panel.periods_to_series`: infer dtype from numpy, `.tolist()` only for object arrays; used in `wide_to_long` and `_tiled_periods`. | The plan's `.tolist()` reconstruction turned `Datetime("ns")` into `Int64` (numpy renders `datetime64[ns]` as epoch ints). Time zones are not recoverable from `periods_` — pre-existing (`long_to_wide` drops them); documented, not fixed. Task 1 review. |
| 9 | Refit contract names `actual_`, `synthetic_`, `pre_mask_`, `weights_`, `att_`, `rmspe_pre_` explicitly and drops `treated_`; `most_influential` ties go to the first dropped donor; non-finite refit ATTs excluded from `att_range` / `most_influential`; explicit `donors` deduplicated preserving order; `paths` / `gaps` carry explicit schemas; `lambda_` coerced with `float()`. | A group `treated_` is a list and the refit needs the collapsed label `units_[_treated_idx]`. The rest keeps the all-failed and tie cases deterministic and typed. Task 2 review. |
| 10 | `_ratio` yields `nan` for $0/0$ and `inf` for a perfect pre-fit; NaN placebo ratios are excluded from rank and p-value; `inf` stays and ranks first; a NaN treated statistic gives `pvalue = nan` with a warning; only-the-treated-row-survives gives $p = 1$ with a warning. Discard rule written as `(pre_j / pre_1)**2 >= k`. | `nan` compares false against everything and would otherwise rank first while padding $J$. Warn rather than raise because the fit is valid — only the ratio test is undefined. $p = 1$ is what the formula gives and what `SCtools` would give. Task 3 review. |
| 11 | `_RefittableSC` stays private. | It names two underscore attributes (`_y_matrix`, `_treated_idx`) the maintainer chose not to promote this release; publishing a Protocol over private names would freeze them as API by the back door. The contract is documented in the module docstring and `methodology.md` §7; a public Protocol is a one-line addition once the attributes are promoted. Task 5 review. |
| 12 | Layer-1 ATT tolerance `abs=1e-4, rel=1e-4` (plan snippet had `abs=1e-4` only). | The `baton rouge` refit differs from R by 2.6e-4 absolute (2.5e-6 relative) on an ATT of about −104 — Clarabel-vs-R solver noise (the full fit already differs by 9.5e-5). This is the documented ATT tolerance in `tests/validation_against_r/README.md`, so it applies the policy rather than relaxing it. Paths were already at that tolerance. Task 6. |
| 13 | `SCtools` harness: classed list suffices for `tdf` (`is_tdf` is `inherits`); `tdf` rebuilt per `mspe_limit` case; R recomputes `mspe.placs` and `loss.v` from the pre rows and checks them against ours at `1e-12` before `mspe.test` runs; `discard.extreme = TRUE` wrapped in `suppressWarnings`. | Rebuilding is ~0.15 s per case, not worth a fixture. R-side recomputation means the denominators are oracle-checked, not merely supplied. The wrap is defensive: the repo runs pytest with `filterwarnings = error` and `SCtools` warns when no placebo exceeds the limit (none of the tested limits does: 5.0 discards 6, 2.0 discards 9 of 39). Measured: ratios within 6.3e-16 relative, p-values equal, for limits None / 5 / 2. Task 7. |
| 14 | **Follow-up, not in v0.6:** our discard rule evaluates `(pre_j/pre_1)**2 >= k`, `SCtools` evaluates `pre_j**2/pre_1**2 >= k`. | They differ by ≤ 3.5e-16 relative on the fixture; no flip possible there (nearest placebos at 5.43 and 1.88 for limits 5 and 2). Switching would bit-match `SCtools`; left out because it touches library code after review and the boundary unit test uses the current form. |
| 15 | Layer-3 characterization: (a) `Synth(fixedeff=False)` is the variant characterized (ρ = 0.9739; median relative gap 0.02114; $p$ 0.8250 ours vs 0.7750 `SCtools`), with the `fixedeff=True` ρ = 0.7617 printed for the record; (b) `controls.identifier` passed **sorted**; (c) `Sigf.ipop = 4`, not `Synth`'s default 5. | (a) R `Synth` has no unit demeaning. (b) `dataprep` returns matrices in sorted-id order but `names.and.numbers` keeps the given order, and `generate.placebos` labels placebo $i$ by `names.and.numbers[i]` while indexing by position — unsorted ids scramble every placebo label (ρ read −0.03 before the fix); arguably a `Synth`/`SCtools` labelling bug worth reporting upstream. (c) With 45 single-period predictors on 38 donors, `kernlab::ipop` at 5 digits diverges on the `nashville` placebo; 3 and 4 converge, 5–7 fail; precision is the only ipop knob `generate.placebos` exposes. The ρ > 0.9 threshold was not touched. Wall time ≈ 90 s. Task 8. |
| 16 | **Qualified exactness claim.** The placebo p-value is described as having the form of a permutation p-value (Abadie 2021 §5), not as an exact test. The all-discarded case gives $p = 1$ by the same formula, but no parity is claimed there. Layer-3 test marked `slow`, prints the gap at 4 significant figures, suppresses tibble messages, cleans up its R globals. | Excluding the real treated unit from every placebo pool (ADH 2010) means the treated statistic uses $J$ donors and each placebo $J-1$, so the units are not exchangeable in the strict sense. `SCtools::mspe.test` cannot run with zero placebos (its `for (i in 1:n)` with `n = 0` indexes missing columns). Task 8 review. |
| 17 | **Two `rmspe_pre` columns, two scales — kept as they are** (final review, 2026-10-08). `LeaveOneOutResult.att["rmspe_pre"]` is the estimator's *normalized* `rmspe_pre_`; `PlaceboInSpaceResult.ratios["rmspe_pre"]` is *raw* on the outcome scale. Both column docstrings and `methodology.md` §7.2 cross-reference each other. Result dataclasses are `@dataclass(frozen=True, eq=False)`, like `ConformalTestResult`, so they compare by identity. | The leave-one-out column is a fit-quality diagnostic comparable to `fit.rmspe_pre_`; the placebo column feeds the ratio statistic, the discard rule and `SCtools::mspe.test`, which all operate on the raw scale. Renaming either would break the comparison it exists for. `eq=False` because the frames inside make the generated `__eq__` raise `TypeError`. |

Decisions 3, 10, 12 and 14–16 are the ones a methodologically minded reader
should know about; the rest are result-shape and harness engineering.

---

## 5. Validation

Pinned in [`tests/validation_against_r/test_robustness.py`](../tests/validation_against_r/test_robustness.py)
on the `GeoLift_PreTest` fixture (`new york` treated from 2021-02-15,
`Synth(fixedeff=True)` unless stated), in three layers
([`methodology.md`](methodology.md) §7.4):

| Layer | Test | Oracle | Result |
|---|---|---|---|
| 1 | `test_placebo_refits_match_r_augsynth_by_composition` | R `augsynth(progfunc='None', scm=TRUE, fixedeff=TRUE)` on the same panel with the real treated unit removed, for three pseudo-treated donors | Counterfactual paths within `atol=rtol=1e-4` |
| 1 | `test_loo_refits_match_r_augsynth_by_composition` | Same, with each of three dropped donors removed | Paths within `atol=rtol=1e-4`; refit ATT within `abs=rel=1e-4` (largest gap 2.6e-4 absolute on about −104, `baton rouge`) |
| 2 | `test_placebo_aggregation_matches_sctools_mspe_test_exactly[None/5.0/2.0]` | `SCtools::mspe.test` on a `tdf` built in R from *our* fitted series; R recomputes `mspe.placs` / `loss.v` from the pre rows and checks them against ours at `1e-12` first | MSPE ratios within 6.3e-16 relative (`rtol=1e-12` asserted), p-values **equal** (`==`), identical kept sets; limits None / 5 / 2 discard 0 / 6 / 9 of 39 placebos |
| 3 | `test_characterize_against_real_sctools_placebos` (`slow`, ≈ 90 s) | Genuine `SCtools::generate.placebos` on `Synth::synth` (ipop, 45 single-period `special.predictors`, `Sigf.ipop = 4`, sorted `controls.identifier`) vs our `Synth(fixedeff=False)` | Spearman ρ = 0.9739 between $r_j^2$ (asserted ρ > 0.9); median relative gap 0.02114; $p$ 0.8250 (33/40) ours vs 0.7750 (31/40) `SCtools`; ρ = 0.7617 against the `fixedeff=True` fit, printed |

`SCtools` (0.3.3.1 at the time of writing) and `Synth` are installed in the
R validation job (`.github/workflows/validation.yml`) for layers 2 and 3.

**Assessment of this evidence.**

- **Layer 1 is the parity claim for the refits.** It exercises the whole
  reconstruction path — `wide_to_long`, `periods_to_series`, the deepcopy,
  the pool manipulation — not just the estimator, because R is given the
  modified panel independently and must land on the same counterfactual.
  The tolerance is the repository's documented one for paths and ATTs
  (decision 12); the `baton rouge` gap shows why an absolute-only `1e-4`
  would have sat at the solver noise floor.
- **Layer 2 is the parity claim for the aggregation, and it is exact.**
  Feeding `mspe.test` our series isolates the convention (treated in the
  denominator, `>=`, discard rule, treated row exempt) from any solver;
  6.3e-16 relative on the ratios is floating-point identity, and `p` is
  compared with `==`. R recomputing the pre-MSPE denominators (decision 13)
  closes the obvious hole — if our pre-window definition drifted, a supplied
  `loss.v` would have moved both sides together.
- **Layer 3 is a characterization, not a parity claim.** ipop with nested
  $V$ weighting over 45 predictors and Clarabel on the outcome-only simplex
  QP are different optimisers; the ρ = 0.97 rank agreement and the 2 %
  median gap say the two procedures rank the same units the same way, which
  is the property a user of the p-value cares about. The two harness
  details it surfaced (decision 15 b, c) are worth more than the number:
  the labelling issue would silently scramble any `generate.placebos` run
  with unsorted ids.
- **What the layers do not cover.** `leave_one_out` has **no oracle beyond
  composition**: R `augsynth` has no leave-one-out routine and `SCtools`
  does not implement ADH 2015 §4.4, so `att_range`, `sign_stable` and
  `most_influential` are checked only by the unit suite (23 tests in
  `tests/unit/test_robustness.py` at the time of the audit) and by the
  composition argument that
  each refit ATT is right. The all-discarded / all-errored placebo case has
  no `SCtools` counterpart (decision 16). Neither gap is closable with the
  oracles available; both are stated.

Unit coverage: 23 tests in `test_robustness.py` (selection modes, `tol`,
`on_error="record"`, NaN / inf ratios, discard boundary, treated groups,
warning aggregation, dtype preservation, `AugSynth` penalty carry-over),
plus the `wide_to_long` / `periods_to_series` round trips in `test_panel.py`,
within the 203-test unit suite.

---

## 6. Verdict

### 6.1 PASS on the algorithm, qualified on the clean-room claim

Both diagnostics trace to the papers construct by construct (§3), the
deviations are either result-shape engineering or recorded and reasoned
(§4), and the numerical evidence is as strong as the available oracles
allow: composition against R `augsynth` for every refit, floating-point
identity against `SCtools::mspe.test` for the aggregation (§5).

The phrasing "clean-room implementation" may be used for the robustness
module **only with the §1.1 qualification**: two `SCtools` source files were
read to build the oracle harness and fix the convention it asserts. They
were not translated and no module code derives from them, but "no R source
consulted" would be false and must not be written about v0.6. The paper's
blanket claim is now qualified for `AugSynth` and for this module.

### 6.2 What is not validated

Stated here so the PASS is not read as broader than it is:

1. **`leave_one_out` has no external oracle beyond composition.** Each
   refit's path and ATT match R `augsynth`; the aggregation over refits
   (envelope, sign stability, most-influential donor) is unit-tested only.
   It is arithmetic over validated numbers, but it is not parity-tested.
2. **Layer 3 is not a parity result.** ρ = 0.9739 and a 2 % median gap
   against a genuine `Synth`/ipop placebo run characterise agreement between
   two different optimisers; the assertion is ρ > 0.9, nothing tighter, and
   the printed numbers are a record, not a tolerance.
3. **The vacuous case** (only the treated row survives discard / errors /
   undefined ratios, $p = 1$ with a warning) has no `SCtools` counterpart.
4. **The discard-rule form** differs from `SCtools` by a few ulps
   (decision 14); exactness in layer 2 holds because no fixture placebo
   sits within that distance of a tested limit.

---

## 7. Recommendations

In descending order of value:

1. **Switch the discard rule to the `pre_j**2 / pre_1**2 >= k` form**
   (decision 14) so layer 2 is a bit-match of `SCtools` rather than an
   agreement to a few ulps. Library code plus the one boundary unit test
   that uses the current form; small, after-review change, hence deferred
   from v0.6.
2. **Promote `_y_matrix` and `_treated_idx` to public attributes and publish
   the refit Protocol** (decision 11). The conformal module took the same
   path in 0.5.0 (`_conformal_null_residuals` → `conformal_null_residuals`);
   until then, third-party estimators implement a private-named contract.
   One line once the attributes are promoted.
3. **Migrate the NR-35 notebook** (`notebooks/_build_nr35.py`, Atos 5 and 7)
   from its hand-rolled placebo and leave-one-out loops to
   `placebo_in_space` / `leave_one_out`, and check the notebook's reported
   p-value and leave-one-out range are reproduced. The prototype should not
   outlive the library it motivated. Tracked as a follow-up issue after
   merge.
4. **Report the `Synth`/`SCtools` labelling issue upstream** (decision 15 b):
   `generate.placebos` labels placebo $i$ from `names.and.numbers[i]` while
   indexing `dataprep`'s sorted matrices by position, so an unsorted
   `controls.identifier` attaches every placebo gap to another unit's name.
   The reproduction is in the layer-3 test.
5. **Qualify the clean-room claim in the paper** for this module alongside
   `AugSynth`, using the §1.1 wording. The three earlier audits already fix
   the per-module alignment; this one adds a fourth line.
6. **Consider an in-time placebo diagnostic** (ADH 2015 §4.3) as a sibling
   of these two if demand appears; the power module's placebo-in-time
   windows are the obvious scaffolding.
