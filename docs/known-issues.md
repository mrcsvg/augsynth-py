# Known issues — open investigations

Working record of behaviours that are suspected bugs or unexplained results.
Each entry states what was observed, how to reproduce it, why it matters, and
what would close it. Remove an entry when it is resolved (and, if it was a real
bug, point at the fix commit).

---

*I-1 (conformal interval markedly asymmetric around the point estimate,
opened 2026-07-07) was resolved 2026-07-20: the full $p(h_0)$ curve matches R
`augsynth` to one ulp on the same panel, so the asymmetry is a property of CWZ
2021 test inversion, not a bug. The same sweep showed the acceptance region is
non-contiguous at $\alpha = 0.05$ there — a rejected $1/T$-floor gap containing
`att_`, identical in R, bridged by the reported min/max envelope — which
upgrades audit recommendation 4 (D-6 contiguity flag) to a demonstrated need.
See `methodology.md` §5.5, the 2026-07-20 updates in
`clean-room-audit-2026-07-07-inference.md` §6.2, and
`test_basque_pvalue_curve_matches_r_exact` in
[`tests/validation_against_r/test_conformal.py`](../tests/validation_against_r/test_conformal.py).*

---

## I-2 — Block and iid permutation schemes disagree substantially

**Status:** open, probably expected behaviour but unquantified. Opened 2026-07-07.
Tracked publicly as [issue #4](https://github.com/mrcsvg/augsynth-py/issues/4).
**Module:** [`src/augsynth_py/inference.py`](../src/augsynth_py/inference.py) — `conformal_pvalue`.

**Observed.** On the Abadie & Gardeazabal (2003) Basque panel — classical
`Synth(fixedeff=False)` fit, `gdpcap`, treated "Basque Country (Pais Vasco)",
treatment 1975, Spain aggregate dropped — two-sided:

| `permutation_type` | p-value |
|---|---|
| `"block"` (default) | 0.1860 |
| `"iid"` | 0.0649 |

The two schemes straddle the conventional 0.05/0.10 thresholds, so the choice
changes the qualitative conclusion.

**Reproduce.**

```python
import numpy as np, polars as pl
from augsynth_py import Synth, conformal_pvalue

bq = pl.read_csv("notebooks/_data/basque_ag2003.csv").filter(
    pl.col("regionname") != "Spain (Espana)"
)
fit = Synth(fixedeff=False).fit(
    bq,
    unit="regionname",
    time="year",
    outcome="gdpcap",
    treated="Basque Country (Pais Vasco)",
    treatment_time=1975,
)

for pt in ("block", "iid"):
    print(pt, conformal_pvalue(fit, permutation_type=pt, rng=np.random.default_rng(42)))
```

**Expected direction, unverified magnitude.** The moving-block scheme preserves
serial dependence while the iid scheme assumes it away, so block *should* be
more conservative on an autocorrelated series — and GDP per capita is strongly
autocorrelated. A factor of ~3 in the p-value is therefore plausible rather
than alarming. What is missing is confirmation that the magnitude is right and
not inflated by, for example, an off-by-one in block length or wrap-around
handling.

**Why it matters.** Lower priority than I-1: this is most likely correct
behaviour. But the paper (¶3.4) uses the gap between the two schemes as a
pedagogical point about the cost of autocorrelation, and that sentence should
not ship on an unverified magnitude.

**What would close it.**
1. Compare both schemes against R `augsynth` on the same panel. *(The `block`
   side is done as of 2026-07-20: exact parity on this exact panel and fit at
   9 probes — and to one ulp over an 89-point sweep — via
   `test_basque_pvalue_curve_matches_r_exact`. What remains is the `iid` side,
   i.e. the magnitude of the block/iid gap.)*
2. Unit-test the block permutation on a synthetic series with *known*
   autocorrelation, where the conservative direction and rough magnitude can be
   predicted analytically.
3. Confirm the block length actually used and how the final partial block is
   handled.

---

## I-3 — Placebo discard rule differs from `SCtools` by a few ulps

**Status:** open, cosmetic; no observable effect on the fixture. Opened
2026-10-08 (robustness plan decision 14; audit
[`clean-room-audit-2026-10-08-robustness.md`](clean-room-audit-2026-10-08-robustness.md)
§3.5, Recommendation 1).
**Module:** [`src/augsynth_py/robustness.py`](../src/augsynth_py/robustness.py) — `placebo_in_space`, `mspe_limit`.

**Observed.** Our discard rule evaluates `(pre_j / pre_1) ** 2 >= k`
(`_ratio(t_pre, rmspe_pre_j) ** 2`), while `SCtools::mspe.test` evaluates
`pre_j**2 / pre_1**2 >= k` on MSPEs. The two expressions are equal in exact
arithmetic and differ by at most 3.5e-16 relative in floating point on
`GeoLift_PreTest`. The nearest placebos to the tested limits sit at 5.43
(for `k = 5`) and 1.88 (for `k = 2`), so no discard decision can flip there,
and layer 2 of `tests/validation_against_r/test_robustness.py` passes
exactly (ratios within 6.3e-16, p-values equal) for limits None / 5 / 2.

**Why it matters.** A placebo whose squared pre-RMSPE ratio lies within a
few ulps of `k` could be discarded by one implementation and kept by the
other, changing `J` and the p-value floor by one unit. Nothing in the test
corpus is that close, but the parity claim for the aggregation is "exact",
and this is the one place where it holds by a margin rather than by
construction.

**What would close it.** Rewrite the rule as `rmspe_pre_j**2 /
rmspe_pre_1**2 >= k` so the arithmetic bit-matches `SCtools`, and update the
boundary unit test in `tests/unit/test_robustness.py` that pins the current
form. Deferred from 0.6.0 because it touches library code after review;
small, isolated change.

---

## I-4 — Upstream `Synth`/`SCtools` harness hazards: placebo label scrambling and ipop divergence

**Status:** open upstream, worked around in our harness. Opened 2026-10-08
(robustness plan decision 15; audit
[`clean-room-audit-2026-10-08-robustness.md`](clean-room-audit-2026-10-08-robustness.md)
§5, Recommendation 4). Candidate reports to the `Synth` / `SCtools`
maintainers; neither is a defect in `augsynth_py`.
**Module:** [`tests/validation_against_r/test_robustness.py`](../tests/validation_against_r/test_robustness.py) — `test_characterize_against_real_sctools_placebos` (layer 3).

**Observed.**

1. *Label scrambling.* `Synth::dataprep` returns `X0`, `Z0` and `Y0plot` in
   sorted-id order regardless of the order of `controls.identifier`, but
   `names.and.numbers` (and `colnames(X0)`) keep the given order.
   `SCtools::generate.placebos` indexes the matrices by position while
   labelling placebo `i` from `names.and.numbers[i]`, so with the natural
   `setdiff(unique(id), treated)` every one of the 39 placebos on
   `GeoLift_PreTest` carried another unit's name. The tell was a Spearman
   ρ of −0.03 between our MSPE ratios and SCtools's, with pre-period MSPEs
   *negatively* correlated across the two sides — impossible if the labels
   were right, since fit difficulty is a property of the unit. The ratios
   themselves are correct; only the labels are permuted. Passing
   `controls.identifier` sorted makes the two orders coincide (ρ = 0.9739).
2. *ipop divergence.* With 45 single-period `special.predictors` on 38
   donors, the inner QP matrix $X_0^\top V X_0$ has reciprocal condition
   number about $5 \cdot 10^{-8}$, and `kernlab::ipop` at `Synth`'s default
   `Sigf.ipop = 5` diverges on the `nashville` placebo (`infinite or missing
   values in 'x'` from its iterate-norm `svd`). It converges at 3 and 4 and
   fails at 5, 6 and 7. Precision is the only ipop knob `generate.placebos`
   exposes, so the harness uses `Sigf.ipop = 4` for the treated fit and the
   placebos alike.

**Why it matters.** (1) silently attaches every placebo gap to the wrong
unit in any `generate.placebos` run with unsorted ids — the p-value is
unaffected (it is a rank over the multiset of ratios), but per-unit
reporting and the spaghetti plot are wrong, and a user comparing against
another implementation would conclude the *other* one is broken. (2) means
a default-settings `generate.placebos` call can abort on a panel of this
shape with an error message that does not name the cause.

**What would close it.** Upstream: a report (with the layer-3 reproduction)
to `SCtools` for (1) — either reorder `names.and.numbers` to match
`dataprep`'s sorted output or index the matrices by name — and a note in
`generate.placebos`'s documentation for (2). On our side, nothing: both
workarounds are in place and explained inline in the test, and this entry
exists so the next person reading `Sigf.ipop = 4` and `sort(...)` in the
harness knows why.
