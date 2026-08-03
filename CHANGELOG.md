# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(with the usual 0.x caveat: minor versions may break the API).

## [Unreleased]

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
- Numerical parity suite against R `augsynth` (`tests/validation_against_r/`),
  including exact p-value-curve parity on the Basque panel.
- PEP 561 `py.typed` marker; `mypy --strict` clean on `src/`.

[Unreleased]: https://github.com/mrcsvg/augsynth-py/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mrcsvg/augsynth-py/releases/tag/v0.3.0
