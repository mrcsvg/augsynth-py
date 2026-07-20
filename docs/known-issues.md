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
    pl.col("regionname") != "Spain (Espana)")
fit = Synth(fixedeff=False).fit(
    bq, unit="regionname", time="year", outcome="gdpcap",
    treated="Basque Country (Pais Vasco)", treatment_time=1975)

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
