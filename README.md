# augsynth-py

A Python implementation of Augmented Synthetic Control Methods and
geo-experimentation tooling. Methodologically faithful to the published
literature, validated against the R reference implementations.

> **Status: pre-alpha.** API is not stable. Do not use in production yet.

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

## Quickstart

```python
# Coming soon — API still under design.
```

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
