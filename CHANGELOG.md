# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the usual 0.x caveat: minor versions may break the API).

## [Unreleased]

### Added

- `conformal_test` / `ConformalTestResult` — the transparent counterpart of
  `conformal_pvalue`: same test, bit-identical p-value, but returning the
  intermediate quantities between the refit-under-null residuals and the
  scalar p-value — the observed statistic, the permutation reference
  distribution it is ranked against (one statistic per permutation), the null
  residuals, and the p-value convention — so the observed post-period
  behaviour can be *shown* against the permutation distribution (histogram +
  observed line) instead of collapsing straight to `p`.
- Flexible permutation blocks: `conformal_pvalue`, `conformal_test`,
  `conformal_interval` and `simulate_power` accept `block_size`. With
  `permutation_type="block"`, an integer `block_size` switches from the
  deterministic cyclic-shift scheme to `ns` random shuffles of contiguous
  length-`block_size` blocks (within-block order preserved, add-one p-value
  convention, requires `rng`) — a caller-tunable middle ground between `iid`
  (`block_size=1` is equivalent) and the cyclic shifts, for serially
  dependent residuals when the `1/T` granularity of the deterministic scheme
  is too coarse. `block_size=None` (default) keeps the existing deterministic
  scheme, unchanged and still exactly R-parity-tested.
- `adjust_pvalues` — multiple-testing adjustment for collections of conformal
  p-values (per placebo window, candidate treated set, or outcome): Holm
  (default), Bonferroni, and Benjamini-Hochberg, matching R's `p.adjust`
  (`"holm"`, `"bonferroni"`, `"BH"`) and returned in input order.

- `augsynth_py.power` — simulation-based power analysis for a given treated
  set (v0.4 milestone). `simulate_power` runs GeoLift-style placebo-in-time
  simulations: sliding pseudo-treatment windows at the end of the panel
  (`lookback_window`), multiplicative effect injection by default
  (`1 + effect_size`, GeoLift's fractional-lift convention; additive
  available), a fresh estimator fit per simulation, and detection via
  `conformal_pvalue(h0=0)` at `pvalue <= alpha`. `PowerResults.power_curve()`
  aggregates detection rates (the zero-effect row is the empirical size) and
  `PowerResults.mde()` reads off the minimum detectable effect at a target
  power. The grid parallelizes with joblib (`n_jobs`), is reproducible under
  `permutation_type="iid"` for a given `rng` regardless of `n_jobs`, and
  R-compatible defaults follow `GeoLiftPower` (effect grid `0..0.25` by
  `0.05`, `lookback_window=1`, `alpha=0.1`). Parity is asserted against
  `GeoLiftPower` (`model="none"`, deterministic block scheme) in
  `tests/validation_against_r/test_power.py`; the estimator itself was
  already parity-tested, so that test isolates the simulation harness.

### Changed

- The planned sibling package is now named `geoexp` throughout the docstrings
  and docs that reference it, replacing the working name `geolift-py`. That
  name implied an association with Meta's GeoLift that this clean-room MIT
  project does not have. Documentation only — no code, API or behaviour
  changed.
- The package description no longer advertises "geo-experimentation tooling"
  (`pyproject.toml`, `README.md`). Per the package boundary recorded in
  `CLAUDE.md`, geo-experiment orchestration lands in the sibling; what ships
  here is generic synthetic control, and the description now says so.
- `AugSynth.fit` is roughly 35x faster when the ridge penalty is chosen by
  cross-validation (39.1s -> 1.1s on a 40-unit, 90-day panel with the 50-point
  auto-grid). `_loo_cv_lambda` was re-solving the simplex QP inside the lambda
  loop, but `Synth._solve_simplex_qp` takes no penalty argument, so each fold's
  `omega` is identical at every grid point: 3750 QP solves where 75 suffice.
  The solve is now hoisted to one per fold. This is caching, not a change of
  algorithm — `cv_path`, the selected `lambda_`, and every downstream estimate
  are bit-for-bit identical, which
  `test_loo_cv_omega_is_lambda_invariant` pins against the un-hoisted form.
  The cost mattered because power analysis (v0.4) refits in a loop, and market
  selection multiplies that by each candidate treated set.
- The `RuntimeError` raised when a CV fold fails now distinguishes the two
  steps: the simplex half reports the fold only (`LOO-CV simplex fit failed at
  held-out t=...`), since it is lambda-invariant and no longer runs inside the
  grid loop, while the ridge half still reports both (`LOO-CV ridge fit failed
  at lambda=..., held-out t=...`). Callers matching on the previous combined
  message need to update.

## [0.3.1] - 2026-08-09

Packaging and documentation only — no estimator behaviour changed, and the
numerical parity suite against R is unaffected.

### Added

- `numpy1` optional-dependency extra (`pip install "augsynth-py[numpy1]"`),
  resolving the dependency set against the numpy 1.x ABI for environments
  shared with libraries that cannot move to numpy 2 yet.
- `docs/compatibility.md`: supported dependency ranges, the cvxpy/numpy version
  boundary, and the constraints-file recipe for co-installing augsynth-py into
  an already-pinned environment.
- CI job `unit-tests-min-deps`, running the unit suite against the exact
  declared dependency floors on Python 3.11.
- README section "Treating more than one market", showing the treated-group
  form of `treated` with a runnable example.
- Regression test covering every iterable container accepted by `treated`
  (`list`, `tuple`, `set`, `frozenset`, `np.ndarray`, `pl.Series`, generator)
  across both estimators.

### Changed

- Lowered the `cvxpy` floor from `>=1.5` to `>=1.4.1`. cvxpy 1.4.1 already
  ships the CLARABEL solver `Synth` targets, and the previous floor made
  augsynth-py uninstallable alongside a `cvxpy<1.5` pin. Because cvxpy >= 1.8
  requires `numpy>=2.0.0`, the old floor also pulled numpy 2 into environments
  that were pinned to numpy 1.x.
- `treated` is annotated `Any | Iterable[Any]` instead of `Any` on `Synth.fit`,
  `AugSynth.fit` and `long_to_wide`. No behaviour change — multi-treated fits
  have worked since 0.3.0 — but neither the signature nor the README quickstart
  showed that a group was accepted, and it was reported as the feature being
  missing. The union collapses to `Any` for a type checker: it documents the
  group form where a reader looks first, it does not enforce it.

## [0.3.0] - 2026-08-03

First public release on PyPI. Everything below is new in the sense that no
earlier version was ever published; internally this tree evolved through the
v0.1 (classical SCM) and v0.2 (ridge ASCM) design milestones.

### Added

- `Synth`: classical simplex-constrained synthetic control
  (Abadie, Diamond & Hainmueller 2010; outcome-only Doudchenko & Imbens 2016
  form, analogue of R `augsynth(progfunc = "None", scm = TRUE)`), with
  optional unit fixed effects (`fixedeff=True` default).
- `AugSynth`: ridge-augmented synthetic control (Ben-Michael, Feller &
  Rothstein 2021, §2.3–2.4), with leave-one-out cross-validation of the
  ridge penalty (auto-grid or user grid) or a fixed `lambda_`.
- `conformal_pvalue` / `conformal_interval`: exact conformal inference for a
  constant post-period effect (Chernozhukov, Wuthrich & Zhu 2021, §3), with
  moving-block (default) and iid permutation schemes and refit-under-the-null
  residuals.
- Multi-treated fits: `treated` accepts an iterable of units; the effect is
  estimated on the treated-group mean.
- `conformal_interval` detects a non-contiguous acceptance region (accepted
  grid points with interior rejected gaps — demonstrated on the Basque panel,
  where the gap contains the point estimate itself) and emits a `UserWarning`
  clarifying that the returned bounds are the conservative min/max envelope
  (clean-room audit D-6, Recommendation 4).
- Numerical parity suite against R `augsynth` (`tests/validation_against_r/`),
  including exact p-value-curve parity on the Basque panel.
- PEP 561 `py.typed` marker; `mypy --strict` clean on `src/`.

[Unreleased]: https://github.com/mrcsvg/augsynth-py/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/mrcsvg/augsynth-py/releases/tag/v0.3.1
[0.3.0]: https://github.com/mrcsvg/augsynth-py/releases/tag/v0.3.0
