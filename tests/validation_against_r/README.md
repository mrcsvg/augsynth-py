# Validation against R

Tests in this directory compare the output of `augsynth_py` against the R
reference implementations (`augsynth`, `gsynth`, `Synth`, `GeoLift`) using
`rpy2` as a bridge.

These tests are the project's contract with reality. If a Python estimator or
geo-experiment helper does not match the R output within tolerance, that is a
bug. The GeoLift-backed tests are marked with `requires_r_pkg("GeoLift")` and
skip when GeoLift is not installed.

## Why a separate directory

Running these tests requires:

- R >= 4.0
- The R packages `augsynth`, `Synth`, and (for geo-experiment tests) `GeoLift`.
- The Python optional dependency group `validation` (which installs `rpy2`).

Most contributors will not have all of this set up locally. Keeping the
parity tests in their own directory lets `pytest tests/unit/` work for
everyone, while `pytest tests/validation_against_r/` is gated on the heavier
environment and runs in its own CI job.

## Running locally

```bash
# Install validation extras
pip install -e ".[validation,dev]"

# Install R dependencies (one-time)
Rscript -e 'install.packages(c("remotes", "Synth"))'
Rscript -e 'remotes::install_github("ebenmichael/augsynth")'

# Run only the parity suite
pytest tests/validation_against_r/ -v
```

If R or `rpy2` is not available, the suite skips cleanly via
`pytest.importorskip`. If a specific R package is missing, only the tests
marked with that package are skipped. They never fail spuriously due to a
missing R install.

## Tolerances

Default tolerances for numerical comparisons are:

- `atol=1e-6, rtol=1e-5` for synthetic control weights.
- `atol=1e-4, rtol=1e-4` for ATT estimates and counterfactual paths.

Relax with explicit justification (a comment citing the reason) only.

## Canonical fixture: `GeoLift_PreTest`

The shared fixture is the `GeoLift_PreTest` panel from the `GeoLift` R package:
40 US cities x 90 days. It is loaded once per session via the
`r_geolift_pretest` fixture in `conftest.py`.

This is the same dataset the GeoLift documentation uses, which makes
comparison with published examples straightforward.
