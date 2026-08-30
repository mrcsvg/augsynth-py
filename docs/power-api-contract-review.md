# `augsynth_py.power` — pre-freeze API contract review

> **Status — accepted 2026-08-30**: the maintainer accepted the
> *recommended* bundle (all of R-1 through R-4, including the two breaking
> items). It ships as augsynth-py **0.5.0**; `geoexp`'s dependency floor is
> that release.

**Date**: 2026-08-28. **Version reviewed**: 0.4.0 (the shape published to PyPI).
**Reviewer context**: `CLAUDE.md` requires that "the public shape of
`augsynth_py.power` should be learned before it is frozen as a cross-package
contract" — this review is that step, gating the start of v0.5 (market
selection in the sibling package `geoexp`). Once `geoexp` depends on this
module, every item below that changes the public surface becomes a
coordinated breaking change; today it is a cheap edit.

**Surface reviewed**: `simulate_power` (signature, defaults, validation,
parallelism, reproducibility), `PowerResults` (fields, `power_curve`, `mde`),
the `PowerEstimator` protocol, module exports, and the seam with
`augsynth_py.inference`. Tests consulted: 21 unit tests in
`tests/unit/test_power.py`, `GeoLiftPower` parity in
`tests/validation_against_r/test_power.py`.

## Verdict

The API is sound and can be frozen **after** resolving four findings: two
require maintainer decisions (R-1, R-4, both cheapest now), one is an
additive gap that would otherwise be re-implemented downstream (R-2), and one
is documentation that becomes part of the contract (R-3). Everything else
reviewed is fine as published (see "Reviewed and endorsed as-is").

Per `CLAUDE.md`, R-1/R-2/R-4 change or extend the public API of exported
names and therefore need maintainer sign-off before implementation. This
document proposes; it does not decide.

## Findings

### R-1 — `PowerEstimator` is not importable, and its contract hangs on a private method

`simulate_power(estimator=...)` is typed against the `PowerEstimator`
protocol, but the protocol is not exported from `augsynth_py` (nor listed in
any `__all__`), and it requires the underscore method
`_conformal_null_residuals(h0)` — the same private seam `_FittedSC` uses in
`inference.py`. Two consequences for the contract:

- `geoexp` cannot type its own `estimator:` parameters without importing a
  non-exported name from a module internal (`augsynth_py.power.PowerEstimator`).
- Any third party implementing a custom estimator must implement a
  private-named method to satisfy a public entry point. That is a contract
  on a name we have documented as private.

**Recommendation**: export `PowerEstimator` from `augsynth_py.__init__`, and
decide the method-name question now, in one of two ways:

1. *(preferred)* Rename `_conformal_null_residuals` →
   `conformal_null_residuals` on `Synth`, `AugSynth`, `_FittedSC`, and
   `PowerEstimator`. It is already the documented extension point of the
   inference module ("any object exposing ...") — the underscore contradicts
   that role. Breaking only for code calling the private name, which no
   external code should be doing at 0.4.0.
2. Keep the underscore but state explicitly in the protocol docstring that
   the name is stable and part of the public estimator contract despite its
   prefix. Zero code churn, permanent awkwardness.

Also worth exporting alongside: `EffectType` and `DEFAULT_EFFECT_SIZES`
(both are part of `simulate_power`'s documented signature).

### R-2 — `PowerResults` does not carry the run's design parameters

`PowerResults` records `simulations`, `alpha`, and `effect_type` — but not
what was simulated: `treated`, the requested `durations`, `lookback_window`,
`permutation_type`/`block_size`, `side`, `ns`, or any description of the
estimator. Market selection runs `simulate_power` once per candidate treated
set and then ranks candidates side by side; with self-describing results,
that is "concatenate and rank". Without them, `geoexp` must carry a parallel
bookkeeping structure mapping result objects back to the design that
produced them — the first thing every downstream consumer would rebuild.

**Recommendation**: add a frozen `params` field (a small frozen dataclass or
a plain dict) capturing the call's design arguments, populated by
`simulate_power`. Purely additive — existing fields, `power_curve`, and
`mde` are untouched — so it ships in any minor release. Estimator identity
can be recorded as `repr(estimator)`-before-fit; that is enough for labeling
and honest about not being a reconstruction recipe.

### R-3 — The multi-candidate reproducibility and parallelism patterns are the contract, but are written nowhere

Two behaviours `geoexp` will lean on are currently implicit:

- **Reproducibility across candidates.** `simulate_power` draws per-task
  seeds from `rng` up front, so one run is reproducible for a given `rng`
  regardless of `n_jobs`. But a *sequence* of calls sharing one generator is
  reproducible only if the calls happen in a fixed order — which candidate-
  level parallelism destroys. The correct pattern is one child generator per
  candidate via `rng.spawn(n_candidates)`, which is order-independent and
  restartable. This should be stated in `simulate_power`'s Notes, because it
  is a property of this API, not of `geoexp`.
- **Nested parallelism.** The efficient split is candidate-level parallelism
  outside (in `geoexp`) with `n_jobs=1` inside, since joblib does not nest
  workers — inner `Parallel` calls inside a worker run sequentially. Stating
  this in the docstring prevents the natural mistake (`n_jobs=-1` at both
  levels) and freezes an expectation `geoexp` can rely on.

**Recommendation**: extend `simulate_power`'s Notes section with both
patterns. Documentation-only; no version implications beyond a docs release.

### R-4 — `rng` leniency contradicts the inference layer

`conformal_pvalue` refuses to run a random permutation scheme without a
caller-supplied generator: `permutation_type="iid"` and `"block"` with a
`block_size` both raise `ValueError` when `rng is None`
(`inference.py:206-224`). `simulate_power`, sitting directly on top of it,
silently substitutes `np.random.default_rng()` when `rng` is omitted —
producing irreproducible results from the one function in the package whose
outputs (power, MDE) feed a go/no-go decision downstream.

**Recommendation**: adopt the inference layer's stance — raise `ValueError`
when a random scheme is requested without `rng`. This is a small breaking
change (calls that omitted `rng` were irreproducible by construction, so no
one can be depending on their values), and under the 0.x policy it makes the
release that carries it a minor bump. The deterministic default
(`permutation_type="block"`, `block_size=None`) keeps needing no `rng`,
so the quickstart path is unaffected.

## Reviewed and endorsed as-is

- **Argument names and defaults** track `GeoLiftPower` deliberately
  (`durations`/`effect_sizes`/`lookback_window`/`alpha=0.1`/`side`), with
  the correspondence table in the docstring. Keep.
- **`estimator` required, no default.** Explicit methodology beats a hidden
  `Synth()` default. Keep.
- **`alpha` stored on results, overridable at aggregation** without
  re-simulating. Good separation; keep.
- **`on_error="record"`** semantics (null `pvalue`, `error` column, failed
  simulations excluded from the power denominator) are what a long
  market-selection sweep needs. Keep.
- **`mde()` returns a grid point, no interpolation**, and refuses
  mixed-sign grids. Honest about resolution; `geoexp` can interpolate from
  `power_curve()` if it wants ranking granularity. A future
  `mde(interpolate=True)` would be additive — not needed for the freeze.
- **`PowerResults` as a frozen dataclass over a polars frame** matches the
  project's data conventions; the `simulations` frame is the documented
  escape hatch for aggregations we did not anticipate.
- **Per-fit artifacts beyond `att`/`att_pct`/`rmspe_pre` are discarded**
  (weights, gaps). Correct for memory at market-selection scale; revisit
  only if `geoexp` demonstrates a need.
- **Panel immutability and prototype `deepcopy`** (never fit in place,
  worker-side copies) — pinned by `test_prototype_is_never_fitted_in_place`.

## Explicit non-contract

For the avoidance of doubt when `geoexp` is written: private helpers
(`_windows`, `_inject_effect`, `_run_one`, `_Window`, `_Task`), the *order*
of columns in `simulations` (names and dtypes are contract; order is not),
and log messages are not part of the public surface and may change without
notice.

## Suggested sequencing

| Bundle | Contents | Version implication |
|---|---|---|
| Additive-only | R-2 (params on results), R-3 (docs), export half of R-1 | 0.4.1 |
| Recommended | All of the above + R-1 rename + R-4 strict `rng` | 0.5.0 of augsynth-py |

The recommended bundle takes the two breaking items while breaking is free.
Note the version-name collision: *project milestone* v0.5 is market
selection (shipping in `geoexp`); an augsynth-py *package release* 0.5.0
carrying this contract cleanup is a different thing. `geoexp`'s dependency
floor becomes whichever release carries the accepted bundle — see
`docs/geoexp-bootstrap-spec.md`.
