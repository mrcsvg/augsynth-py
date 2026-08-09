# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the usual 0.x caveat: minor versions may break the API).

## [Unreleased]

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

[Unreleased]: https://github.com/mrcsvg/augsynth-py/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mrcsvg/augsynth-py/releases/tag/v0.3.0
