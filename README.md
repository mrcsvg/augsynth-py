# augsynth-py

A Python implementation of Augmented Synthetic Control Methods, with conformal
inference. Methodologically faithful to the published literature, validated
against the R reference implementations.

> **Status: alpha.** The estimator API is functional and validated against R,
> but not yet stable across releases. Pin the exact version in production.

## What this is

Synthetic control methods for causal inference, with a focus on the use case
that matters most in modern marketing science: measuring the lift of geo-level
ad campaigns. The roadmap targets feature parity with:

- The R [`augsynth`](https://github.com/ebenmichael/augsynth) package
  (estimators).
- Meta's R [`GeoLift`](https://github.com/facebookincubator/GeoLift) package
  (orchestration: power analysis, market selection, multi-cell).

## Why another synthetic control package

There are good Python options for parts of this problem (`CausalPy`,
`pysyncon`, `tfcausalimpact`), but none provides Augmented Synthetic Control
Methods (Ben-Michael, Feller & Rothstein 2021) together with the
GeoLift-style orchestration layer in a single Python-native package.

## Installation

```bash
pip install augsynth-py
```

For the validation test suite (requires R and the `augsynth` R package):

```bash
pip install "augsynth-py[validation]"
```

### Environments pinned to numpy 1.x

Both numpy 1.26+ and numpy 2.x are supported. When augsynth-py shares an
environment with libraries that cannot move off numpy 1.x yet (`econml`,
`tslearn`/`numba`, and similar), install the `numpy1` extra, which resolves the
dependency set against the last releases built for that ABI:

```bash
pip install "augsynth-py[numpy1]"
```

`pip` only reconciles what a single command names, so when augsynth-py is added
to an environment that already holds such pins, state them explicitly — in the
same command or through a constraints file:

```bash
# constraints.txt
numpy>=1.26.4,<2
cvxpy>=1.4.1,<1.5

pip install -c constraints.txt augsynth-py
```

The oldest supported dependency set — numpy 1.26, scipy 1.12, cvxpy 1.4.1,
polars 1.0, joblib 1.3 on Python 3.11 — runs the unit suite on every push.
See [`docs/compatibility.md`](docs/compatibility.md) for the full matrix.

## Quickstart

```python
import numpy as np
import polars as pl

from augsynth_py import AugSynth, conformal_pvalue

# Simulated geo panel: 20 markets x 90 days, +10% lift in geo_00 from day 70.
rng = np.random.default_rng(7)
days = np.arange(90)
base = rng.uniform(80, 120, 20)
trend = rng.normal(0.1, 0.05, 20)
seasonal = 5 * np.sin(2 * np.pi * days / 7)

panel = pl.concat(
    pl.DataFrame(
        {
            "geo": f"geo_{i:02d}",
            "day": days,
            "sales": base[i] + trend[i] * days + seasonal + rng.normal(0, 1.0, 90),
        }
    )
    for i in range(20)
).with_columns(
    pl.when((pl.col("geo") == "geo_00") & (pl.col("day") >= 70))
    .then(pl.col("sales") * 1.10)
    .otherwise(pl.col("sales"))
    .alias("sales")
)

fit = AugSynth(lambda_=1.0).fit(
    panel,
    unit="geo",
    time="day",
    outcome="sales",
    treated="geo_00",
    treatment_time=70,
)

print(f"ATT: {fit.att_:.2f} ({fit.att_pct_:+.1%})")
# ATT: 10.29 (+9.8%)

# The noise in this simulated panel is iid; on real (autocorrelated) series
# keep the default permutation_type="block".
p = conformal_pvalue(fit, permutation_type="iid", rng=np.random.default_rng(0))
print(f"conformal p-value: {p:.4f}")
# conformal p-value: 0.0120
```

### Treating more than one market

`treated` takes either a single unit value or any iterable of them — `list`,
`tuple`, `set`, `np.ndarray`, `pl.Series`. An iterable designates a treated
*group*: the units are collapsed to their elementwise mean and dropped from the
donor pool, so `att_` is the effect on that group mean.

```python
group_fit = AugSynth(lambda_=1.0).fit(
    panel,
    unit="geo",
    time="day",
    outcome="sales",
    treated={"geo_00", "geo_01"},  # <- a group, not a single market
    treatment_time=70,
)

print(group_fit.units_[0])
# geo_00,geo_01

print(f"ATT on the group mean: {group_fit.att_:.2f}")
# ATT on the group mean: 4.83
# (lower than the 10.29 above because only geo_00 got the lift; geo_01 dilutes it)
```

`str` and `bytes` are the one exception: they are read as a single unit value,
not as a sequence of characters. To treat exactly one market, either form works
(`treated="geo_00"` or `treated=["geo_00"]`).

The aggregate lift across the group is `att_ * n_treated * n_post_periods` —
`att_` itself stays per-unit-per-period, so it is comparable to a single-market
fit.

What's available today:

- `Synth` — classical simplex-constrained synthetic control
  (outcome-only form, `augsynth(progfunc = "None", scm = TRUE)` analogue),
  with optional unit fixed effects.
- `AugSynth` — ridge-augmented synthetic control (Ben-Michael, Feller &
  Rothstein 2021), with leave-one-out CV for the ridge penalty by default.
- `conformal_pvalue` / `conformal_interval` — exact conformal inference for
  a constant post-period effect (Chernozhukov, Wuthrich & Zhu 2021), block
  and iid permutation schemes.
- Multi-treated fits: pass an iterable of units as `treated` to estimate the
  effect on the treated-group mean (see
  [Treating more than one market](#treating-more-than-one-market)).
- `simulate_power` — GeoLift-style power analysis / MDE for a *given* treated
  set: placebo-in-time windows, effect injection, conformal detection, with
  a joblib-parallel grid:

  ```python
  from augsynth_py import AugSynth, simulate_power

  # Same panel as the quickstart above. AugSynth() re-runs its penalty CV in
  # every simulation (the R-faithful default); freeze it with lambda_= when
  # the grid is large.
  res = simulate_power(
      panel,
      estimator=AugSynth(lambda_=1.0),
      unit="geo",
      time="day",
      outcome="sales",
      treated="geo_00",
      durations=15,  # periods per pseudo-experiment
      lookback_window=10,  # placebo windows sliding back from the end
  )
  res.power_curve()  # detection rate per effect size; row 0.0 = false-positive rate
  res.mde()  # smallest lift with power >= 0.8
  ```

Market selection — *choosing* the treated set, budgets, multi-cell designs —
is the planned sibling package (`geoexp`), which will consume this one
through its public API.

## Methodological references

The implementation is based on the published literature, not translated from
the R sources. Key references:

- Abadie, Diamond & Hainmueller (2010). Synthetic Control Methods for
  Comparative Case Studies. *JASA*.
- Ben-Michael, Feller & Rothstein (2021). The Augmented Synthetic Control
  Method. *JASA*.
- Xu (2017). Generalized Synthetic Control Method. *Political Analysis*.
- Chernozhukov, Wuthrich & Zhu (2021). An Exact and Robust Conformal Inference
  Method for Counterfactual and Synthetic Controls. *JASA*.

See `docs/methodology.md` for the mapping between code and equations.

## Validation against R

Every estimator in this package is validated numerically against the R
reference implementation. The validation suite lives in
`tests/validation_against_r/` and runs as a separate CI job.

If you find a discrepancy with the R output beyond documented tolerances,
please open an issue — that is a bug.

## Contributing

Read [`CLAUDE.md`](CLAUDE.md) first. It contains the architectural decisions,
coding conventions, and the validation rule that PRs must satisfy.

## License

MIT. See [`LICENSE`](LICENSE).
