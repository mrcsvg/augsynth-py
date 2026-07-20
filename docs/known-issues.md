# Known issues — open investigations

Working record of behaviours that are suspected bugs or unexplained results.
Each entry states what was observed, how to reproduce it, why it matters, and
what would close it. Remove an entry when it is resolved (and, if it was a real
bug, point at the fix commit).

---

## I-1 — Conformal confidence interval is markedly asymmetric around the point estimate

**Status:** open, unexplained. Opened 2026-07-07.
**Module:** [`src/augsynth_py/inference.py`](../src/augsynth_py/inference.py) — `conformal_interval`.

**Observed.** On the Abadie & Gardeazabal (2003) Basque panel, a classical
`Synth(fixedeff=False)` fit gives a point ATT of **−0.6915**, but the 95%
conformal interval is **(−3.2169, 7.6864)** — skewed strongly positive, with
the point estimate sitting far left of centre.

**Reproduce.**

```python
import numpy as np, polars as pl
from augsynth_py import Synth, conformal_interval

bq = pl.read_csv("notebooks/_data/basque_ag2003.csv").filter(
    pl.col("regionname") != "Spain (Espana)")
fit = Synth(fixedeff=False).fit(
    bq, unit="regionname", time="year", outcome="gdpcap",
    treated="Basque Country (Pais Vasco)", treatment_time=1975)

print(fit.att_)                                    # -0.6915
for g in (40, 100, 400):
    print(g, conformal_interval(fit, alpha=0.05, grid_size=g,
                                rng=np.random.default_rng(42)))
# 40  (-2.7421, 7.1005)   width  9.8426
# 100 (-3.1149, 7.5479)   width 10.6628
# 400 (-3.2169, 7.6864)   width 10.9033
```

**Not a discretisation artifact.** The interval converges as the grid is
refined (widths 9.84 → 10.66 → 10.90) and the asymmetry persists at every grid
size, so it is not grid coarseness.

**Why it matters.** This interval is a candidate illustrative example in the
SoftwareX paper (¶3.4/¶3.5 of `paper/sections/softwarex_v2.md`). Publishing it
as a demonstration requires knowing whether the asymmetry is a genuine property
of test inversion under CWZ 2021 — plausible, since the inverted test statistic
need not be symmetric in the hypothesised effect — or an implementation defect
in the grid construction, the null-refit, or the interval-extraction step.

**What would close it.**
1. Run R `augsynth`'s conformal interval on the same panel and compare
   endpoints. If R is asymmetric the same way, this is a property, not a bug.
2. Inspect the p-value-vs-candidate-effect curve directly (not just its
   0.05 crossings) to see whether the asymmetry is in the test statistic or in
   the interval extraction.
3. Check the grid's construction: if candidate effects are centred on something
   other than the point estimate, or the grid bounds are asymmetric, that alone
   could produce this.

**Related.** The inference module has **no clean-room audit** yet — that audit
is the natural place to resolve and document this.

---

## I-2 — Block and iid permutation schemes disagree substantially

**Status:** open, probably expected behaviour but unquantified. Opened 2026-07-07.
**Module:** [`src/augsynth_py/inference.py`](../src/augsynth_py/inference.py) — `conformal_pvalue`.

**Observed.** Same Basque fit as I-1, two-sided:

| `permutation_type` | p-value |
|---|---|
| `"block"` (default) | 0.1860 |
| `"iid"` | 0.0649 |

The two schemes straddle the conventional 0.05/0.10 thresholds, so the choice
changes the qualitative conclusion.

**Reproduce.**

```python
from augsynth_py import conformal_pvalue
for pt in ("block", "iid"):
    print(pt, conformal_pvalue(fit, permutation_type=pt,
                               rng=np.random.default_rng(42)))
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
1. Compare both schemes against R `augsynth` on the same panel.
2. Unit-test the block permutation on a synthetic series with *known*
   autocorrelation, where the conservative direction and rough magnitude can be
   predicted analytically.
3. Confirm the block length actually used and how the final partial block is
   handled.
