"""Parity test for augsynth_py.power against R GeoLift's ``GeoLiftPower``.

Runs the same power simulation through both implementations on the
``GeoLift_PreTest`` panel and asserts agreement, per the validation rule in
``CLAUDE.md`` (oracle for v0.4: ``GeoLiftPower``).

Alignment choices (decision log)
--------------------------------
- ``model="none", fixed_effects=TRUE`` on the R side corresponds to our
  ``Synth(fixedeff=True)``; the estimator itself is already parity-tested in
  ``test_classical_synth.py``, so a mismatch here isolates the *simulation
  harness* (window arithmetic, truncation, injection, detection), not the fit.
- ``conformal_type="block"`` makes both sides deterministic — no seeds to
  reconcile — and matches our ``permutation_type="block"``;
  ``side_of_test="two_sided"`` matches ``side="two-sided"``.
- ``lookback_window=2`` exercises the sliding-window truncation semantics
  (an earlier window must drop the periods after its end). GeoLiftPower's
  per-simulation row order within an effect size is not part of its contract,
  so p-values are compared as *sorted multisets* per effect size.
- ``alpha=0.105``: block p-values on this panel are multiples of ``1/90``
  (window 0) and ``1/89`` (window 1). Neither grid contains 0.105, and no
  grid point falls in ``(0.1, 0.105]``, so the strict-vs-non-strict detection
  convention cannot change any power number at this alpha — the comparison is
  convention-free. (Our library convention is ``pvalue <= alpha``.)
- The GeoLiftPower return schema has varied across versions (per-simulation
  ``pvalue`` rows vs pre-aggregated ``power`` rows). The test asserts on
  whichever is present, preferring the stronger per-p-value comparison, and
  fails with the actual column list if neither is recognizable.

R is driven through the same rpy2 session fixtures as the other parity tests;
GeoLift's ``GeoDataRead`` maps the date axis to integer periods 1..90, which
matches our sorted-date axis window-for-window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from augsynth_py.power import simulate_power
from augsynth_py.synth.classical import Synth
from tests.validation_against_r._r_frames import from_r_data_frame

TREATED = ["chicago", "portland"]
DURATION = 15
LOOKBACK = 2
EFFECTS = [0.0, 0.10, 0.25]
ALPHA = 0.105  # tie-free for the p-value grids in play; see module docstring


def _r_power_frame(r_session: Any) -> pl.DataFrame:
    """Run GeoLiftPower in R and return its result as a polars DataFrame."""
    formals = set(r_session("names(formals(GeoLift::GeoLiftPower))"))
    needed = {"conformal_type", "side_of_test", "lookback_window"}
    missing = needed - formals
    if missing:
        pytest.skip(
            f"installed GeoLift's GeoLiftPower lacks argument(s) {sorted(missing)}; "
            "a deterministic parity run needs them."
        )

    r_session(
        'gl_power_data <- GeoDataRead(data = GeoLift_PreTest, date_id = "date", '
        'location_id = "location", Y_id = "Y", format = "yyyy-mm-dd", summary = FALSE)'
    )
    effects_r = ", ".join(str(e) for e in EFFECTS)
    treated_r = ", ".join(f'"{u}"' for u in TREATED)
    r_session(
        "gl_power <- GeoLiftPower("
        "data = gl_power_data, "
        f"locations = c({treated_r}), "
        f"effect_size = c({effects_r}), "
        f"lookback_window = {LOOKBACK}, "
        f"treatment_periods = {DURATION}, "
        "cpic = 1, "
        f"alpha = {ALPHA}, "
        'model = "none", '
        "fixed_effects = TRUE, "
        'conformal_type = "block", '
        'side_of_test = "two_sided", '
        "parallel = FALSE"
        ")"
    )
    return _normalize_columns(from_r_data_frame(r_session("as.data.frame(gl_power)")))


def _normalize_columns(out: pl.DataFrame) -> pl.DataFrame:
    """Map GeoLiftPower's column names onto the snake_case names asserted on.

    The installed GeoLift (2026-08 CI run) returns per-simulation rows with
    columns ``location, pvalue, duration, EffectSize, treatment_start,
    investment, cpic, ScaledL2Imbalance, att_estimator, detected_lift, power``.
    Lowercasing alone turns ``EffectSize`` into ``effectsize`` (no underscore),
    so known aliases are mapped explicitly after the generic normalization.
    """
    out = out.rename({c: c.lower().replace(".", "_") for c in out.columns})
    aliases = {"effectsize": "effect_size", "scaledl2imbalance": "scaled_l2_imbalance"}
    return out.rename({old: new for old, new in aliases.items() if old in out.columns})


@pytest.mark.requires_r_pkg("GeoLift")
@pytest.mark.requires_r_pkg("augsynth")
def test_power_simulation_matches_geoliftpower(
    r_session: Any, r_geolift_pretest: pl.DataFrame
) -> None:
    r_power = _r_power_frame(r_session)

    panel = r_geolift_pretest.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    ours = simulate_power(
        panel,
        estimator=Synth(fixedeff=True),
        unit="location",
        time="date",
        outcome="Y",
        treated=TREATED,
        durations=DURATION,
        effect_sizes=EFFECTS,
        lookback_window=LOOKBACK,
        alpha=ALPHA,
        permutation_type="block",
        side="two-sided",
    )

    columns = set(r_power.columns)
    if "pvalue" in columns:
        # Strong form: per-simulation p-values, compared as sorted multisets
        # within each effect size (R's row order is not part of its contract).
        assert r_power.height == len(EFFECTS) * LOOKBACK, r_power
        for effect in EFFECTS:
            r_p = sorted(
                r_power.filter(pl.col("effect_size") == effect).get_column("pvalue").to_list()
            )
            py_p = sorted(
                ours.simulations.filter(pl.col("effect_size") == effect)
                .get_column("pvalue")
                .to_list()
            )
            assert len(r_p) == LOOKBACK, (effect, r_power)
            np.testing.assert_allclose(
                py_p,
                r_p,
                atol=1e-8,
                rtol=0,
                err_msg=f"conformal p-values diverge from GeoLiftPower at effect={effect}",
            )
    elif "power" in columns:
        # Aggregated form: detection rates per effect size. alpha=0.105 makes
        # this comparison independent of the <= vs < detection convention.
        curve = ours.power_curve()
        for effect in EFFECTS:
            r_pow = r_power.filter(pl.col("effect_size") == effect).get_column("power").to_list()
            assert r_pow, (effect, r_power)
            py_pow = curve.filter(pl.col("effect_size") == effect).get_column("power").to_list()[0]
            np.testing.assert_allclose(
                py_pow,
                r_pow[0],
                atol=1e-6,
                rtol=0,
                err_msg=f"power diverges from GeoLiftPower at effect={effect}",
            )
    else:
        raise AssertionError(
            f"GeoLiftPower returned neither 'pvalue' nor 'power'; columns: {sorted(columns)}"
        )
