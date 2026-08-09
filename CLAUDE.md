# augsynth-py

A Python implementation of Augmented Synthetic Control Methods (ASCM) and related
geo-experimentation methodology. The end goal is feature parity with the R
`augsynth` package and a higher-level orchestration layer comparable to Meta's
GeoLift (R).

This file is the entry point for AI coding assistants working on this repo.
Read it before starting any task.

---

## Project goal

Provide a Python-native, methodologically faithful implementation of:

1. Classical synthetic control (Abadie, Diamond & Hainmueller, 2010).
2. Augmented synthetic control with ridge augmentation (Ben-Michael, Feller & Rothstein, 2021).
3. Generalized synthetic control (Xu, 2017).
4. Power analysis and market selection for geo-experiments (the orchestration
   layer that GeoLift provides on top of `augsynth`).

The package must be usable in production data science workflows. Methodology
correctness is the top priority, performance is secondary, ergonomics third.

## Implementation strategy: clean-room from papers

All algorithms are implemented **from the published papers**, not by translating
the R `augsynth` source code. This keeps the project under a permissive license
(MIT) and forces a deeper understanding of the methods.

Reference implementations in R (`augsynth`, `gsynth`, `Synth`, `GeoLift`) are
used **only as oracles for validation tests**, never as a source to translate
from. Do not copy R source code into this repo.

## Canonical references

The implementation must be traceable to these papers. New code should cite the
relevant paper and equation in docstrings.

- Abadie, Diamond & Hainmueller (2010). *Synthetic Control Methods for
  Comparative Case Studies*. JASA.
- Abadie (2021). *Using Synthetic Controls: Feasibility, Data Requirements,
  and Methodological Aspects*. JEL.
- Ben-Michael, Feller & Rothstein (2021). *The Augmented Synthetic Control
  Method*. JASA.
- Xu (2017). *Generalized Synthetic Control Method*. Political Analysis.
- Doudchenko & Imbens (2016). *Balancing, Regression, Difference-in-Differences
  and Synthetic Control Methods: A Synthesis*. NBER WP 22791.
- Chernozhukov, Wuthrich & Zhu (2021). *An Exact and Robust Conformal Inference
  Method for Counterfactual and Synthetic Controls*. JASA.

## Architectural decisions (already made — do not relitigate without reason)

| Concern              | Choice                            | Why                                                    |
|----------------------|-----------------------------------|--------------------------------------------------------|
| Language             | Python 3.11+                      | Practical adoption, scientific ecosystem               |
| Numerics             | `numpy`, `scipy`                  | Native BLAS/LAPACK, stable APIs                        |
| Convex optimization  | `cvxpy` (OSQP / Clarabel backends)| Cleanest expression of constrained QPs                 |
| Tabular data         | `polars`                          | Fast, sane API, scales to panel datasets               |
| Parallelism          | `joblib`                          | Power analysis is embarrassingly parallel              |
| Build backend        | `hatchling`                       | Modern, simple, PEP 621 native                         |
| Layout               | `src/` layout                     | Avoids accidental imports during testing               |
| Lint + format        | `ruff`                            | One tool, fast                                         |
| Type checking        | `mypy --strict` on `src/`         | Catch shape/contract bugs early                        |
| Tests                | `pytest`                          | Standard                                               |
| Validation oracle    | R `augsynth` via `rpy2`           | Ground truth for parity tests                          |

**Do not introduce new dependencies without justification in the PR.**
**Do not reach for Rust/C extensions before profiling identifies a real hot
loop.** This is a project rule, not a suggestion. Premature optimization in a
foreign language has killed similar projects.

## Validation rule (non-negotiable)

Every public estimator must have at least one test in
`tests/validation_against_r/` that:

1. Runs the same input through both the Python implementation and the R
   reference (`augsynth` / `gsynth` / `Synth`) via `rpy2`.
2. Asserts numerical agreement on weights, counterfactuals, and ATT estimates
   within a documented tolerance (default `atol=1e-6, rtol=1e-5`; relax with
   justification only).
3. Uses a fixture from the `GeoLift_PreTest` panel (40 US cities, 90 days)
   loaded once per session.

PRs that add a new estimator without a parity test should be rejected.

## Roadmap and scope

Shipped so far: classical `Synth` (v0.1), ridge-augmented `AugSynth` with
LOO-CV (v0.2), conformal inference per CWZ 2021 (v0.3) — each with unit +
R-parity tests and a clean-room audit in `docs/`.

Critical path toward end-to-end geo-experiments (the project's reason to
exist), in order:

1. **v0.4 — power analysis / MDE** (`augsynth_py.power`): effect injection +
   refit + detection rate via conformal inference; parallel grid with joblib.
   Parity oracle: `GeoLiftPower`.
2. **v0.5 — market selection**: candidate ranking over treatment sets ×
   durations, `GeoLiftMarketSelection` analogue. Completes the single-cell
   GeoLift flow.
3. **v0.6+ — multi-cell experiments**; then, if demand appears, auxiliary
   covariates $X_i$ and GSC (Xu 2017, incl. BFR 2021 §4 augmentation).

Deferred until further notice (do not start without maintainer sign-off):

- AugSynth with auxiliary covariates / predictors $X_i$ — needs `Synth`
  predictor support first.
- Matrix-completion augmentation (Athey et al. 2021) — would land as a
  sibling estimator.
- GSC / gsynth (Xu 2017) and BFR 2021 §4 GSC augmentation.

Permanently out of scope:

- Bayesian variants — that's CausalPy's territory.
- GPU acceleration — not needed.
- Rust extensions — not needed, see above.

## Repository layout

```
augsynth-py/
├── CLAUDE.md                       <- you are here
├── README.md
├── CHANGELOG.md                    <- Keep a Changelog format
├── LICENSE                         <- MIT
├── pyproject.toml
├── src/augsynth_py/
│   ├── __init__.py                 <- public exports
│   ├── _version.py                 <- single source of truth for the version
│   ├── exceptions.py               <- domain exceptions
│   ├── inference.py                <- conformal inference (CWZ 2021)
│   ├── py.typed                    <- PEP 561 marker
│   └── synth/                      <- estimator implementations
├── tests/
│   ├── conftest.py                 <- shared fixtures
│   ├── unit/                       <- pure-Python unit tests, no R required
│   └── validation_against_r/       <- parity tests, require R + augsynth
├── docs/
│   ├── methodology.md              <- maps code to paper sections/equations
│   ├── compatibility.md            <- dependency floors, numpy 1.x/2.x support
│   ├── known-issues.md             <- open investigations register
│   ├── releasing.md                <- PyPI release runbook
│   └── clean-room-audit-*.md       <- one audit per estimator/module
├── papers/
│   └── SOURCES.md                  <- link index (PDFs are not tracked)
├── notebooks/                      <- exploratory notebooks, not shipped
└── .github/workflows/
    ├── ci.yml                      <- lint, type, unit tests
    ├── validation.yml              <- parity tests against R
    └── release.yml                 <- tag-triggered PyPI trusted publishing
```

## Code conventions

- Type hints everywhere in `src/`. Public APIs use `numpy.typing.NDArray` for
  arrays.
- Docstrings: NumPy style. Every public function cites paper + equation when
  implementing a known method.
- No `print()` in library code. Use the `logging` module at INFO/DEBUG.
- No global mutable state. Estimators are classes with `fit()` populating
  trailing-underscore fitted attributes (`att_`, `weights_`, ...); a separate
  `predict()` has not been needed so far — introduce one only with maintainer
  sign-off on the API shape.
- Random seeds: every stochastic function takes `rng: np.random.Generator`,
  never reads global state.
- Errors: raise `ValueError` for bad input, custom exceptions in
  `augsynth_py.exceptions` for domain errors.

## Testing conventions

- `tests/unit/` runs in CI on every push, no R required.
- `tests/validation_against_r/` runs in a separate CI job that installs R
  and the reference packages. Use `pytest.importorskip("rpy2")` so local
  developers without R can still run the unit suite.
- Parity fixtures are loaded from the R packages themselves via `rpy2`
  (`GeoLift_PreTest` in `tests/validation_against_r/conftest.py`) or from
  public CSVs under `notebooks/_data/` (Basque panel). There is no
  `tests/fixtures/` directory today; create one only if a fixture cannot
  live in either of those places.
- Snapshot tests for plotting outputs are NOT used; we test data, not pixels.

## Common task recipes

### Adding a new estimator

1. Create `src/augsynth_py/synth/<name>.py` with a class implementing `fit` and
   `predict`.
2. Cite the paper and equation in the class docstring.
3. Add unit tests in `tests/unit/test_<name>.py` covering edge cases (single
   treated unit, perfect pre-period fit, degenerate donor pool).
4. Add a parity test in `tests/validation_against_r/test_<name>.py` against the
   R reference.
5. Document the estimator in `docs/methodology.md` with the mapping to paper
   sections.
6. Export from `src/augsynth_py/__init__.py`.

### Updating dependencies

- Edit `pyproject.toml`, not `requirements.txt` (we don't have one).
- Justify the dependency in the PR description.
- Prefer `scipy` / `numpy` primitives over new packages.
- Keep floors at the oldest release whose API the code actually uses, and keep
  the base dependencies uncapped — the package must stay co-installable with
  numpy 1.x environments. A floor change means updating the pins in the
  `unit-tests-min-deps` CI job too. See `docs/compatibility.md`.

## Things to confirm with the maintainer before doing

- Adding a dependency.
- Changing the public API of any class already exported from `__init__.py`.
- Relaxing a validation tolerance.
- Adding a Rust / Cython / C extension.
- Adding GPU support.
