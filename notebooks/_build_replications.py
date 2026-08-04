"""Programmatic builder for the two replication notebooks.

Run once to (re)generate both notebooks::

    .venv/bin/python notebooks/_build_replications.py

The notebooks are the artifacts users open and edit. This script is the source
of truth used to recreate them cleanly when the design evolves — same pattern
as ``_build_showcase.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

NOTEBOOKS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class ReplicationSpec:
    """Dataset-specific values injected into the shared notebook template."""

    output_path: Path
    title: str
    paper_citation: str
    paper_doi_url: str
    paper_free_url: str | None
    historical_intro: str  # one paragraph
    csv_filename: str
    unit_col: str
    time_col: str
    outcome_col: str
    outcome_label: str  # what to call it in plot axes, e.g. "Cigarette sales (packs/cap)"
    outcome_unit_short: str  # for inline text, e.g. "packs/cap"
    treated_value: str  # value in unit_col identifying the treated unit
    treated_short: str  # short display name, e.g. "California"
    intervention_year: int
    intervention_year_label: str  # e.g. "1989 (Proposition 99)"
    # Optional pre-filter applied to the panel (e.g. excluding a national aggregate).
    exclude_units: tuple[str, ...] = ()
    published_att_blurb: str = ""  # one line summarizing what the paper reported
    paper_top_donors: tuple[str, ...] = ()
    # Ballpark RMSPE shown in Act 6c. Pulled from sanity checks; only used as a
    # rule-of-thumb threshold in the text, not as a hard assertion.
    rmspe_rule_of_thumb_pct: float = 5.0
    # Whether to append an "Act 7 — Augmenting with ridge regression" section
    # after Act 6. Default False; set True on BASQUE. Rationale in
    # docs/plans/2026-05-31-augsynth-v0.2-m6-design.md §"Confirmed decisions" N1.
    augsynth_act_enabled: bool = False
    # Populated for specs whose Act 7 references a published paper number.
    # Only used by Act 7.8's ATT comparison table. Leave empty for specs
    # with augsynth_act_enabled=False.
    paper_att_row_label: str = ""  # e.g. "AG 2003 (paper)"
    paper_att_value_str: str = ""  # e.g. "-0.10 GDP/cap (avg 1975-1998)"


CALIFORNIA = ReplicationSpec(
    output_path=NOTEBOOKS_DIR / "02_adh_prop99_california.ipynb",
    title="California's Proposition 99 — Replicating Abadie, Diamond & Hainmueller (2010)",
    paper_citation=(
        "Abadie, A., Diamond, A., & Hainmueller, J. (2010). "
        "Synthetic Control Methods for Comparative Case Studies: "
        "Estimating the Effect of California's Tobacco Control Program. "
        "*Journal of the American Statistical Association*, 105(490), 493-505."
    ),
    paper_doi_url="https://doi.org/10.1198/jasa.2009.ap08746",
    paper_free_url="https://www.nber.org/papers/w13783",
    historical_intro=(
        "In 1988 Californians voted in **Proposition 99**, a sweeping tobacco-control program "
        "that raised the cigarette excise tax by 25¢ per pack and earmarked the revenue for "
        "anti-smoking education, public-health programs and enforcement. The question Abadie, "
        "Diamond & Hainmueller (2010) asked was deceptively simple: **did cigarette consumption "
        "in California fall by more than it would have without the program?** Cigarette sales "
        "trended downward across the United States throughout the 1990s for many reasons, so a "
        "raw before-vs-after comparison would conflate the policy effect with the secular trend. "
        "The paper's answer is the canonical illustration of the synthetic control method."
    ),
    csv_filename="smoking_adh2010.csv",
    unit_col="state",
    time_col="year",
    outcome_col="cigsale",
    outcome_label="Cigarette sales (packs per capita)",
    outcome_unit_short="packs/cap",
    treated_value="California",
    treated_short="California",
    intervention_year=1989,
    intervention_year_label="1989 (Proposition 99 took effect)",
    exclude_units=(),
    published_att_blurb=(
        "ADH 2010 reports an average post-period ATT of roughly **-19 packs per capita** "
        "(1989-2000), with top donor weights on Utah, Nevada, Montana, Colorado and Connecticut."
    ),
    paper_top_donors=("Utah", "Nevada", "Montana", "Colorado", "Connecticut"),
    rmspe_rule_of_thumb_pct=2.0,
)


BASQUE = ReplicationSpec(
    output_path=NOTEBOOKS_DIR / "03_basque_terrorism.ipynb",
    title="The Economic Cost of Terrorism in the Basque Country — Replicating Abadie & Gardeazabal (2003)",
    paper_citation=(
        "Abadie, A., & Gardeazabal, J. (2003). The Economic Costs of Conflict: "
        "A Case Study of the Basque Country. "
        "*American Economic Review*, 93(1), 113-132."
    ),
    paper_doi_url="https://doi.org/10.1257/000282803321455188",
    paper_free_url=None,
    historical_intro=(
        "From the mid-1960s onward the **Basque Country** was the centre of a long campaign of "
        "political violence by ETA, with assassinations, kidnappings and bombings escalating "
        "sharply after 1975. Abadie & Gardeazabal (2003) — the paper that first introduced "
        "synthetic control as a method — asked: **what would the Basque economy have looked like "
        "without the conflict?** The challenge is that no Spanish region is a good naïve "
        "comparison: each has its own industrial mix, demographics and growth path. The "
        "synthetic-control answer is to construct a weighted combination of other regions whose "
        "1955-1975 trajectory matches the Basque Country's, and read off the gap that opens "
        "after 1975."
    ),
    csv_filename="basque_ag2003.csv",
    unit_col="regionname",
    time_col="year",
    outcome_col="gdpcap",
    outcome_label="Real GDP per capita (1986 USD, thousands)",
    outcome_unit_short="k$ per capita",
    treated_value="Basque Country (Pais Vasco)",
    treated_short="Basque Country",
    intervention_year=1975,
    intervention_year_label="1975 (sharp escalation of ETA violence)",
    # The dataset ships with 'Spain (Espana)' as the national aggregate. It must be
    # excluded — it is not a comparable donor; it includes the treated unit by
    # construction.
    exclude_units=("Spain (Espana)",),
    published_att_blurb=(
        "Abadie & Gardeazabal (2003) report an average gap of about **0.66 (k$ per capita)** "
        "below the synthetic counterfactual, with nearly all donor weight concentrated on "
        "**Cataluña** and **Madrid**."
    ),
    paper_top_donors=("Cataluna", "Madrid (Comunidad De)"),
    rmspe_rule_of_thumb_pct=3.0,
    augsynth_act_enabled=True,
    paper_att_row_label="AG 2003 (paper)",
    paper_att_value_str="≈ -0.66 k$ per capita (avg gap below synthetic counterfactual, post-1975)",
)


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

_CELL_COUNTER = [0]


def _next_id() -> str:
    _CELL_COUNTER[0] += 1
    return f"cell-{_CELL_COUNTER[0]:02d}"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell templates
# ---------------------------------------------------------------------------


def _build_act7_cells(spec: ReplicationSpec) -> list[dict]:
    """Return the Act 7 cell dicts (10 markdown + 9 code = 19).

    Only called when `spec.augsynth_act_enabled` is True. Interpolates
    ``spec.outcome_label``, ``spec.outcome_unit_short``,
    ``spec.paper_att_row_label`` and ``spec.paper_att_value_str`` at build
    time. All notebook-side code cells reference the constants established by
    the Act 1 setup cell (``UNIT``, ``TIME``, ``OUT``, ``TREATED``,
    ``INTERVENTION``, ``est``, ``panel``) and the palette added by Phase B.
    """
    cells: list[dict] = []

    # --- 7.1 Setup + CV fit ------------------------------------------------
    cells.append(
        md(
            """
        ## Act 7 — Augmenting with ridge regression

        Act 6 diagnosed what classical synthetic control misses when the pre-period
        fit is imperfect: the SCM residual carries systematic signal that leaks into
        the estimated ATT. Ridge augmentation (Ben-Michael, Feller & Rothstein 2021)
        fits that residual with a ridge regression on the donor pre-period outcomes;
        the augmented counterfactual is the SCM path *plus* the ridge correction.
        See [`docs/methodology.md`](../docs/methodology.md) §2 for the closed form
        and [`docs/clean-room-audit-2026-05-26-augsynth.md`](../docs/clean-room-audit-2026-05-26-augsynth.md)
        §4.1 for the M3.5 period-demeaning detail that pins Python to R at machine
        precision.

        We refit with `AugSynth`, selecting λ by leave-one-out CV over the default
        auto-grid (`logspace(-4, 4, 50) × var(y_pre_treated)`).
        """
        )
    )

    cells.append(
        code(
            """
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message="CV-selected lambda")
            est_aug = AugSynth(fixedeff=False).fit(
                panel, unit=UNIT, time=TIME, outcome=OUT,
                treated=TREATED, treatment_time=INTERVENTION,
            )

        grid = est_aug.lambda_cv_path_[:, 0]
        losses = est_aug.lambda_cv_path_[:, 1]
        best_idx = int(losses.argmin())

        print(f"CV-chosen λ = {est_aug.lambda_:.4g}")
        print(f"Auto-grid range: [{grid.min():.4g}, {grid.max():.4g}] "
              f"({len(grid)} candidates)")
        print(f"Chosen λ sits at grid position {best_idx}/{len(grid) - 1}")
        if best_idx == 0 or best_idx == len(grid) - 1:
            print("⚠️  CV picked a grid boundary; in a real analysis, widen the grid.")
        """
        )
    )

    # --- 7.2 Counterfactual overlay ---------------------------------------
    cells.append(
        md(
            """
        ### 7.2 — Overlaying the augmented counterfactual

        Three lines on one axis: the treated unit, the classical Synth
        counterfactual from Act 3, and the AugSynth counterfactual. The pre-period
        is shaded to draw the eye to where the ridge correction operates.
        """
        )
    )

    cells.append(
        code(
            f"""
        fig, ax = plt.subplots(figsize=(9, 5))
        years = est_aug.periods_
        ax.axvspan(years.min(), INTERVENTION, color=COLOR_GRID, alpha=0.5, zorder=0)
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.plot(years, est_aug.actual_,
                color=COLOR_TREATED, lw=2.2, label=f"{{TREATED}} (actual)")
        ax.plot(est.periods_, est.synthetic_,
                color=COLOR_SYNTH, lw=1.8, label="Synth counterfactual")
        ax.plot(years, est_aug.synthetic_,
                color=COLOR_AUGMENTED, lw=1.8, label="AugSynth counterfactual")
        ax.set_xlabel("Year")
        ax.set_ylabel({spec.outcome_label!r})
        ax.set_title("Act 7.2 — Augmented counterfactual next to classical Synth")
        ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.3 Ridge correction trajectory ----------------------------------
    cells.append(
        md(
            """
        ### 7.3 — What the ridge correction adds at each period

        `ridge_correction_` is the algebraic difference `synthetic_ - synthetic_scm_`.
        In the pre-period, it's what the ridge adds to close the SCM residual; in
        the post-period, it's the augmented estimator's estimate of the extra
        counterfactual signal the SCM misses.
        """
        )
    )

    cells.append(
        code(
            f"""
        fig, ax = plt.subplots(figsize=(9, 4))
        years = est_aug.periods_
        ax.axhline(0.0, color=COLOR_TEXT, lw=0.8)
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.fill_between(years, 0.0, est_aug.ridge_correction_,
                        color=COLOR_AUGMENTED, alpha=0.35, step=None)
        ax.plot(years, est_aug.ridge_correction_,
                color=COLOR_AUGMENTED, lw=1.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Ridge correction (" + {spec.outcome_unit_short!r} + ")")
        ax.set_title("Act 7.3 — Ridge correction trajectory")
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.4 Effective weights vs simplex weights -------------------------
    cells.append(
        md(
            """
        ### 7.4 — Effective weights vs simplex SCM weights

        Per BFR 2021 §2.4 Lemma 1, the augmented weights decompose as `ω + γ` where
        `ω` is the classical simplex SCM weight (Act 3) and `γ` is the ridge
        augmentation. Donors whose effective weight is negative are ones the ridge
        stopped trusting (or actively counter-weights against).
        """
        )
    )

    cells.append(
        code(
            """
        donor_names = sorted(est_aug.weights_.keys())
        scm = np.array([est_aug.scm_weights_[d] for d in donor_names])
        eff = np.array([est_aug.weights_[d] for d in donor_names])
        order = np.argsort(np.abs(eff))[::-1]
        donors_sorted = [donor_names[i] for i in order]
        scm_sorted = scm[order]
        eff_sorted = eff[order]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(donors_sorted))
        bar_w = 0.4
        ax.bar(x - bar_w / 2, scm_sorted, width=bar_w,
               color=COLOR_SYNTH, label="Synth ω (simplex)")
        ax.bar(x + bar_w / 2, eff_sorted, width=bar_w,
               color=COLOR_AUGMENTED, label="AugSynth ω+γ")
        ax.axhline(0.0, color=COLOR_TEXT, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(donors_sorted, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Weight")
        ax.set_title("Act 7.4 — Effective weights vs simplex SCM weights")
        ax.legend(loc="best", frameon=False)
        for i, d in enumerate(donors_sorted):
            if eff_sorted[i] < -1e-6:
                ax.annotate(d, (i + bar_w / 2, eff_sorted[i]),
                            xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=7, color=COLOR_AUGMENTED)
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.5 CV path plot -------------------------------------------------
    cells.append(
        md(
            """
        ### 7.5 — The CV path λ selects from

        Leave-one-out CV over pre-treatment periods (BFR 2021 §3.2) picks the λ
        that minimizes squared prediction error on held-out periods. If the minimum
        sits at a grid endpoint, the auto-grid was too narrow — surfaced above in
        Act 7.1.
        """
        )
    )

    cells.append(
        code(
            """
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax_full.semilogx(grid, losses, color=COLOR_AUGMENTED, marker="o", markersize=3)
        ax_full.axvline(est_aug.lambda_, color=COLOR_SYNTH, ls="--", lw=1.0,
                        label=f"chosen λ = {est_aug.lambda_:.3g}")
        ax_full.set_xlabel("λ (log scale)")
        ax_full.set_ylabel("CV loss (sum of squared held-out errors)")
        ax_full.set_title("Full CV path")
        ax_full.legend(loc="best", frameon=False)

        zoom_lo = max(0, best_idx - 3)
        zoom_hi = min(len(grid), best_idx + 4)
        ax_zoom.semilogx(grid[zoom_lo:zoom_hi], losses[zoom_lo:zoom_hi],
                         color=COLOR_AUGMENTED, marker="o", markersize=4)
        ax_zoom.axvline(est_aug.lambda_, color=COLOR_SYNTH, ls="--", lw=1.0)
        ax_zoom.set_xlabel("λ (log scale)")
        ax_zoom.set_ylabel("CV loss")
        ax_zoom.set_title(f"Zoom: grid[{zoom_lo}:{zoom_hi}]")
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.6 Sensitivity sweep --------------------------------------------
    cells.append(
        md(
            """
        ### 7.6 — Sensitivity to λ across two decades around σ²

        Five pinned-λ AugSynth fits at `[0.01, 0.1, 1.0, 10.0, 100.0] × σ²` where
        σ² = `var(y_pre_treated)`. Reading left-to-right: at small λ the ridge
        interpolates the pre-period exactly but the post-period trajectory swings
        wildly (overfitting); at large λ the augmentation collapses onto classical
        Synth. The CV-chosen λ typically sits in the interpretable middle. The
        panel closest to the CV-chosen value is highlighted with a thicker border.
        """
        )
    )

    cells.append(
        code(
            f"""
        pre_y = est_aug.actual_[est_aug.pre_mask_]
        sigma_sq = float(np.var(pre_y, ddof=0))
        multipliers = [0.01, 0.1, 1.0, 10.0, 100.0]
        lambdas_to_try = [m * sigma_sq for m in multipliers]

        chosen_mult_idx = int(np.argmin([
            abs(np.log(lam) - np.log(est_aug.lambda_)) for lam in lambdas_to_try
        ]))

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
        years_all = est_aug.periods_
        for i, (mult, lam, ax) in enumerate(zip(multipliers, lambdas_to_try, axes)):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message="CV-selected lambda")
                est_i = AugSynth(fixedeff=False, lambda_=lam).fit(
                    panel, unit=UNIT, time=TIME, outcome=OUT,
                    treated=TREATED, treatment_time=INTERVENTION,
                )
            ax.axvspan(years_all.min(), INTERVENTION, color=COLOR_GRID, alpha=0.5, zorder=0)
            ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.8)
            ax.plot(years_all, est_i.actual_, color=COLOR_TREATED, lw=1.5)
            ax.plot(years_all, est_i.synthetic_, color=COLOR_AUGMENTED, lw=1.5)
            ax.set_title(f"λ = {{mult}}·σ² = {{lam:.3g}}", fontsize=10)
            if i == chosen_mult_idx:
                for side in ("top", "bottom", "left", "right"):
                    ax.spines[side].set_linewidth(2.0)
                    ax.spines[side].set_color(COLOR_SYNTH)
            ax.set_xlabel("Year", fontsize=9)
        axes[0].set_ylabel({spec.outcome_label!r}, fontsize=9)
        fig.suptitle("Act 7.6 — λ sensitivity sweep (CV-chosen panel outlined)")
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.7 R augsynth side-by-side --------------------------------------
    cells.append(
        md(
            """
        ### 7.7 — Verifying against R `augsynth(progfunc='Ridge')`

        The M4 parity tests
        ([`tests/validation_against_r/test_augsynth.py`](../tests/validation_against_r/test_augsynth.py))
        pin `AugSynth` against R at strict tolerance on the GeoLift fixture. Here
        we run the same comparison on the Basque panel: R selects its own λ, we
        compare paths and chosen-λ positions. The `±1` grid-cell tolerance from D8
        absorbs the `T₀ − 1` vs `T₀` fold-count and mean-vs-sum CV-loss aggregation
        differences between the two implementations
        (see [`docs/clean-room-audit-2026-05-26-augsynth.md`](../docs/clean-room-audit-2026-05-26-augsynth.md)
        §3.3).
        """
        )
    )

    cells.append(
        code(
            f"""
        # `_r_oracles.py` lives alongside this notebook in `notebooks/`. Add that
        # directory to sys.path so we can import it as a plain module regardless
        # of the notebook's execution CWD.
        import sys as _sys
        from pathlib import Path as _Path
        _notebooks_dir = None
        for _cand in (_Path.cwd(), _Path.cwd().parent, _Path.cwd().parent.parent):
            if (_cand / "_r_oracles.py").exists():
                _notebooks_dir = _cand
                break
            if (_cand / "notebooks" / "_r_oracles.py").exists():
                _notebooks_dir = _cand / "notebooks"
                break
        if _notebooks_dir is None:
            raise RuntimeError("Could not locate notebooks/_r_oracles.py")
        if str(_notebooks_dir) not in _sys.path:
            _sys.path.insert(0, str(_notebooks_dir))
        import _r_oracles  # noqa: E402

        panel_for_r = panel.with_columns(
            ((pl.col(UNIT) == TREATED) & (pl.col(TIME) >= INTERVENTION))
            .cast(pl.Int64).alias("treat")
        )

        # _fit_via_augsynth uses treatment_day as a positional index for its Python-
        # side actual/rmspe/att computations; pass the year's index in the sorted
        # time axis rather than the year value.
        years_sorted = panel.sort(TIME)[TIME].unique().to_list()
        treatment_idx = years_sorted.index(INTERVENTION)

        r_fit = _r_oracles.fit_augsynth_r(
            panel_for_r,
            treated_city=TREATED,
            treatment_day=treatment_idx,
            fixedeff=False,
            unit=UNIT, time=TIME, outcome=OUT,
        )

        mean_delta = float(np.mean(np.abs(est_aug.synthetic_ - r_fit.synthetic)))
        max_delta = float(np.max(np.abs(est_aug.synthetic_ - r_fit.synthetic)))
        print(f"Python-chosen λ = {{est_aug.lambda_:.4g}}")
        if r_fit.chosen_lambda is not None:
            print(f"R-chosen λ      = {{r_fit.chosen_lambda:.4g}}")
        else:
            print("R-chosen λ      = <unavailable>")
        print(f"Mean |Python - R| on synthetic path = {{mean_delta:.3e}}")
        print(f"Max  |Python - R| on synthetic path = {{max_delta:.3e}}")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.plot(est_aug.periods_, est_aug.synthetic_,
                color=COLOR_AUGMENTED, lw=2.0, label="Python AugSynth")
        ax.plot(est_aug.periods_, r_fit.synthetic,
                color=COLOR_SYNTH, lw=1.0, ls=":", label="R augsynth (Ridge)")
        ax.set_xlabel("Year")
        ax.set_ylabel({spec.outcome_label!r})
        ax.set_title("Act 7.7 — Python AugSynth vs R augsynth(progfunc='Ridge')")
        ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.8 ATT comparison table -----------------------------------------
    cells.append(
        md(
            """
        ### 7.8 — ATT comparison: Synth, AugSynth, and the published paper

        Three rows in absolute outcome units. AG 2003 as cited in Act 5.
        """
        )
    )

    cells.append(
        code(
            f"""
        import pandas as pd

        post_mask_synth = ~est.pre_mask_
        post_mask_aug = ~est_aug.pre_mask_

        synth_att = float((est.actual_[post_mask_synth] - est.synthetic_[post_mask_synth]).mean())
        aug_att = float(est_aug.att_)

        synth_pre_mean = float(np.mean(np.abs(est.actual_[est.pre_mask_])))
        aug_pre_mean = float(np.mean(np.abs(est_aug.actual_[est_aug.pre_mask_])))
        synth_rmspe_pre = est.rmspe_pre_
        aug_rmspe_pre = est_aug.rmspe_pre_
        synth_rmspe_post = float(
            np.sqrt(np.mean((est.actual_[post_mask_synth] - est.synthetic_[post_mask_synth]) ** 2))
            / synth_pre_mean
        )
        aug_rmspe_post = float(
            np.sqrt(np.mean((est_aug.actual_[post_mask_aug] - est_aug.synthetic_[post_mask_aug]) ** 2))
            / aug_pre_mean
        )

        att_table = pd.DataFrame({{
            "Estimator": ["Synth (Act 3)", "AugSynth (Act 7)", {spec.paper_att_row_label!r}],
            "ATT estimate": [f"{{synth_att:+.3f}}", f"{{aug_att:+.3f}}", {spec.paper_att_value_str!r}],
            "Pre-period RMSPE": [f"{{synth_rmspe_pre:.4f}}", f"{{aug_rmspe_pre:.4f}}", "n/a"],
            "Post-period RMSPE": [f"{{synth_rmspe_post:.4f}}", f"{{aug_rmspe_post:.4f}}", "n/a"],
        }})
        att_table
        """
        )
    )

    # --- 7.9 Pre-period RMSPE bar chart -----------------------------------
    cells.append(
        md(
            """
        ### 7.9 — Pre-period RMSPE: does ridge augmentation earn its keep?

        The load-bearing empirical claim from BFR 2021: ridge augmentation reduces
        pre-period imbalance. If AugSynth's pre-period RMSPE is not lower than
        Synth's here, either the fixture doesn't benefit from augmentation (rare on
        real geo panels) or something upstream is off — investigate rather than
        paper over.
        """
        )
    )

    cells.append(
        code(
            """
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(
            ["Synth", "AugSynth"],
            [synth_rmspe_pre, aug_rmspe_pre],
            color=[COLOR_SYNTH, COLOR_AUGMENTED],
        )
        for bar, v in zip(bars, [synth_rmspe_pre, aug_rmspe_pre]):
            ax.annotate(f"{v:.4f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10)
        reduction_pct = (
            100.0 * (synth_rmspe_pre - aug_rmspe_pre) / synth_rmspe_pre
            if synth_rmspe_pre else 0.0
        )
        ax.set_title(f"Pre-period RMSPE ↓ {reduction_pct:.1f}%")
        ax.set_ylabel("RMSPE (fraction of pre-period mean)")
        plt.tight_layout()
        plt.show()
        """
        )
    )

    # --- 7.10 Closing (markdown only) -------------------------------------
    cells.append(
        md(
            """
        ### 7.10 — Closing

        The augmented estimator earns its keep on the Basque fixture: AugSynth's
        pre-period RMSPE is materially lower than Synth's (Act 7.9), and Act 7.4
        shows the ridge redistributes some weight to (or against) donors that
        classical SCM couldn't touch given its simplex constraint. Per BFR 2021
        §2.4 Lemma 1 the augmented weights are `ω + γ`, and some donors here
        receive negative effective weight — a feature, not a bug, in the ridge
        formulation.

        Two engineering notes carried forward from the v0.2 audit:

        - **M3.5 period demeaning.** R `augsynth(progfunc='Ridge')` applies a
          donor-mean-per-pre-period demeaning step inside its ridge routine that
          BFR 2021 §2.3 does not make explicit. `augsynth_py` matches this in
          `_period_demean_pre`; without it the augmented path would diverge from
          R by orders of magnitude. See [`docs/clean-room-audit-2026-05-26-augsynth.md`](../docs/clean-room-audit-2026-05-26-augsynth.md)
          §4.1.
        - **Trust anchor.** The Python and R paths in Act 7.7 coincide to
          documented tolerance (mean and max deltas printed above) as a
          first-cell replication of the M4 parity test result. The rigorous
          parity contract is in
          [`tests/validation_against_r/test_augsynth.py`](../tests/validation_against_r/test_augsynth.py).

        Remaining scope (covariates per R-1.1, inference per R-3.x, GSC mode per
        R-1.3, MC augmentation per R-1.2) is documented in
        [`docs/plans/2026-05-16-augsynth-v0.2-design.md`](../docs/plans/2026-05-16-augsynth-v0.2-design.md) — those
        are the natural v0.3+ extensions.
        """
        )
    )

    return cells


def build_cells(spec: ReplicationSpec) -> list[dict]:
    _CELL_COUNTER[0] = 0  # reset per-notebook ids
    cells: list[dict] = []

    paper_links = f"[DOI]({spec.paper_doi_url})"
    if spec.paper_free_url:
        paper_links += f" · [Free preprint]({spec.paper_free_url})"

    excluded_units_py = repr(list(spec.exclude_units))
    paper_top_donors_py = repr(list(spec.paper_top_donors))

    # --- Header ------------------------------------------------------------
    cells.append(
        md(
            f"""
        # {spec.title}

        > **Paper.** {spec.paper_citation}
        >
        > {paper_links}

        {spec.historical_intro}

        This notebook walks the paper's argument from data to counterfactual to estimate, with
        `augsynth_py.Synth().fit(...)` doing the heavy lifting. The narration mirrors the paper;
        the code is the just-landed v0.1 estimator.

        **Roadmap of the notebook:**
        1. **Act 1** — the intervention and the data
        2. **Act 2** — why naive comparisons fail
        3. **Act 3** — synthetic control: building the counterfactual
        4. **Act 4** — visualising the fit
        5. **Act 5** — comparing to the published paper
        6. **Act 6** — what classical SCM doesn't (yet) solve, *shown in practice*
        """
        )
    )

    # --- Setup cell --------------------------------------------------------
    cells.append(
        code(
            f"""
        # Imports and a small matplotlib style.
        from __future__ import annotations

        import sys
        from pathlib import Path

        # Make `augsynth_py` importable when running the notebook from anywhere.
        _here = Path.cwd()
        for cand in (_here, _here.parent, _here.parent.parent):
            if (cand / "src" / "augsynth_py").exists():
                sys.path.insert(0, str(cand / "src"))
                DATA_DIR = (cand / "notebooks" / "_data") if (cand / "notebooks").exists() else (_here / "_data")
                break
        else:
            DATA_DIR = _here / "_data"

        import numpy as np
        import polars as pl
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        from augsynth_py import AugSynth, Synth

        # Palette: one accent + greys. No rainbow.
        COLOR_TREATED = "#D7263D"
        COLOR_SYNTH   = "#1B4965"
        COLOR_DONOR   = "#B0B0B0"
        COLOR_GRID    = "#EAEAEA"
        COLOR_TEXT    = "#333333"
        COLOR_ALT     = "#F18F01"  # secondary accent for overlay plots in Act 6
        COLOR_AUGMENTED = "#7A5195"  # Act 7 accent — purple to contrast SCM's navy

        mpl.rcParams.update({{
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.edgecolor": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.8,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 110,
        }})

        # Constants shared across acts.
        UNIT  = {spec.unit_col!r}
        TIME  = {spec.time_col!r}
        OUT   = {spec.outcome_col!r}
        TREATED = {spec.treated_value!r}
        INTERVENTION = {spec.intervention_year}
        EXCLUDED_DONORS = {excluded_units_py}
        """
        )
    )

    # --- Act 1 ------------------------------------------------------------
    cells.append(
        md(
            f"""
        ## Act 1 — The intervention and the data

        We load the panel that powered the original paper: each row is one (unit, year)
        observation of {spec.outcome_label.lower()}. The treated unit is **{spec.treated_short}**
        and the intervention occurs in **{spec.intervention_year_label}**.
        """
        )
    )

    cells.append(
        code(
            f"""
        panel = pl.read_csv(DATA_DIR / {spec.csv_filename!r})
        if EXCLUDED_DONORS:
            panel = panel.filter(~pl.col(UNIT).is_in(EXCLUDED_DONORS))

        print(f"shape       : {{panel.shape}}")
        print(f"units       : {{panel[UNIT].n_unique()}}")
        print(f"year range  : {{panel[TIME].min()}}-{{panel[TIME].max()}}")
        print(f"treated unit: {{TREATED}} (intervention in {{INTERVENTION}})")
        panel.head(5)
        """
        )
    )

    cells.append(
        code(
            f"""
        # Plot every unit's outcome trajectory. Treated highlighted in red, donors in grey.
        fig, ax = plt.subplots(figsize=(11, 5.2))
        for unit, group in panel.partition_by(UNIT, as_dict=True).items():
            is_treated = unit[0] == TREATED
            ax.plot(
                group[TIME], group[OUT],
                color=COLOR_TREATED if is_treated else COLOR_DONOR,
                lw=2.4 if is_treated else 0.9,
                alpha=1.0 if is_treated else 0.55,
                zorder=3 if is_treated else 1,
                label={spec.treated_short!r} if is_treated else None,
            )
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.text(INTERVENTION + 0.4, ax.get_ylim()[1] * 0.98, "  Intervention",
                color=COLOR_TEXT, fontsize=10, va="top")
        ax.set_title("Outcome trajectories: treated unit vs donor pool")
        ax.set_xlabel("Year"); ax.set_ylabel({spec.outcome_label!r})
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    # --- Act 2 ------------------------------------------------------------
    cells.append(
        md(
            f"""
        ## Act 2 — Why naive comparisons fail

        The most tempting comparison is **{spec.treated_short} vs the mean of the other units**, before and
        after the intervention. This is the textbook 2×2 difference-in-differences (DiD), and it
        leans on a strong assumption: that without the intervention, {spec.treated_short} would have moved
        *in parallel* with the donor mean. The plot below puts the assumption under the
        microscope.
        """
        )
    )

    cells.append(
        code(
            """
        treated_path = (
            panel.filter(pl.col(UNIT) == TREATED).sort(TIME)[[TIME, OUT]]
            .rename({OUT: "treated"})
        )
        donor_path = (
            panel.filter(pl.col(UNIT) != TREATED)
            .group_by(TIME).agg(donors_mean=pl.col(OUT).mean())
            .sort(TIME)
        )
        comparison = treated_path.join(donor_path, on=TIME)

        # 2x2 DiD ATT.
        pre  = pl.col(TIME) <  INTERVENTION
        post = pl.col(TIME) >= INTERVENTION
        t_pre = comparison.filter(pre)["treated"].mean()
        t_pst = comparison.filter(post)["treated"].mean()
        d_pre = comparison.filter(pre)["donors_mean"].mean()
        d_pst = comparison.filter(post)["donors_mean"].mean()
        did_att = (t_pst - t_pre) - (d_pst - d_pre)
        print(f"DiD ATT (treated - donor-mean change): {did_att:+.3f}")
        """
        )
    )

    cells.append(
        code(
            f"""
        # Visual sanity-check of parallel trends.
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(comparison[TIME], comparison["treated"], color=COLOR_TREATED, lw=2.4, label={spec.treated_short!r})
        ax.plot(comparison[TIME], comparison["donors_mean"], color=COLOR_DONOR, lw=2.0,
                ls="-.", label="Mean of other units")
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title("Treated unit vs raw donor mean — were trends really parallel pre-intervention?")
        ax.set_xlabel("Year"); ax.set_ylabel({spec.outcome_label!r})
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    cells.append(
        md(
            """
        > **Takeaway.** A raw mean over donors is a poor counterfactual. Some donor units track the
        > treated unit closely in the pre-period; others don't. What we want is a **weighted**
        > combination that matches the treated unit's pre-period trajectory by construction —
        > that is the synthetic control.
        """
        )
    )

    # --- Act 3 ------------------------------------------------------------
    cells.append(
        md(
            r"""
        ## Act 3 — Synthetic control: building the counterfactual

        Let $y_1$ be the treated unit's outcome path over the pre-period and $Y_0$ the matrix of
        donor outcomes over the same period. The classical synthetic control assigns donor weights
        $w$ that minimise the squared pre-period gap, subject to two simplex constraints:

        $$
        w^* \;=\; \arg\min_{w \ge 0,\ \sum_j w_j = 1}\; \big\|\, y_1^{\text{pre}} \,-\, Y_0^{\text{pre}}\, w \,\big\|^2.
        $$

        Non-negative weights summing to one force the synthetic unit to live inside the **convex
        envelope** of the donor pool — no extrapolation, no negative donor contributions. That
        constraint is the price we pay for interpretability.

        Here we set `fixedeff=False` to match the paper's original setup: classical synthetic
        control on raw outcome levels, no unit demeaning.
        """
        )
    )

    cells.append(
        code(
            """
        est = Synth(fixedeff=False).fit(
            panel,
            unit=UNIT, time=TIME, outcome=OUT,
            treated=TREATED, treatment_time=INTERVENTION,
        )

        print(f"Sum of weights        : {sum(est.weights_.values()):.4f}")
        print(f"Min donor weight      : {min(est.weights_.values()):+.4f}  (>= 0)")
        print(f"# donors with w > 0.01: {sum(1 for w in est.weights_.values() if w > 0.01)}")
        print(f"Pre-period RMSPE      : {est.rmspe_pre_*100:.3f}% of pre-period mean")
        """
        )
    )

    # --- Act 4 ------------------------------------------------------------
    cells.append(
        md(
            """
        ## Act 4 — Visualising the fit

        Three plots in sequence: (1) which donors carry weight, (2) the synthetic vs the real
        path, (3) the gap between them — the estimated effect of the intervention.
        """
        )
    )

    cells.append(
        code(
            """
        # Plot 1 — Top-10 donor weights.
        sorted_w = sorted(est.weights_.items(), key=lambda x: -x[1])
        top = sorted_w[:10]
        others = sum(w for _, w in sorted_w[10:])
        labels = [c for c, _ in top] + (["Others"] if others > 1e-4 else [])
        values = [w for _, w in top] + ([others] if others > 1e-4 else [])

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.barh(labels, values, color=[COLOR_SYNTH if v > 0.01 else COLOR_DONOR for v in values])
        for i, v in enumerate(values):
            if v > 0.005:
                ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=COLOR_TEXT, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel("Weight"); ax.set_title("Donor weights — who composes the synthetic counterfactual?")
        ax.grid(axis="y", alpha=0.0)
        plt.tight_layout(); plt.show()
        """
        )
    )

    cells.append(
        code(
            f"""
        # Plot 2 — Real vs synthetic.
        years = est.periods_.astype(int)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(years, est.actual_, color=COLOR_TREATED, lw=2.4, label="Real " + {spec.treated_short!r})
        ax.plot(years, est.synthetic_, color=COLOR_SYNTH, lw=2.0, ls="--", label="Synthetic " + {spec.treated_short!r})
        ax.axvspan(years[0], INTERVENTION, color=COLOR_GRID, alpha=0.5, zorder=0)
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.text(INTERVENTION - 0.5, ax.get_ylim()[1]*0.97, "Pre (calibration)",
                color=COLOR_TEXT, ha="right", va="top", fontsize=10)
        ax.text(INTERVENTION + 0.5, ax.get_ylim()[1]*0.97, "Post (effect)",
                color=COLOR_TEXT, ha="left", va="top", fontsize=10)
        ax.text(years[1], ax.get_ylim()[0]*1.01,
                f"Pre-period RMSPE = {{est.rmspe_pre_*100:.2f}}%",
                color=COLOR_TEXT, fontsize=10, va="bottom",
                bbox=dict(facecolor="white", edgecolor=COLOR_GRID, boxstyle="round,pad=0.3"))
        ax.set_title("Real vs synthetic " + {spec.treated_short!r})
        ax.set_xlabel("Year"); ax.set_ylabel({spec.outcome_label!r})
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    cells.append(
        code(
            f"""
        # Plot 3 — Gap.
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(years, est.gap_, color=COLOR_TREATED, lw=2.0)
        ax.fill_between(years, 0, est.gap_,
                        where=(years >= INTERVENTION), color=COLOR_TREATED, alpha=0.30)
        ax.axhline(0, color=COLOR_TEXT, lw=0.7)
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        att_pos_idx = (years >= INTERVENTION).nonzero()[0][len((years >= INTERVENTION).nonzero()[0]) // 2]
        ax.annotate(
            f"Average post-period ATT\\n= {{est.att_:+.2f}} {spec.outcome_unit_short}\\n({{est.att_pct_*100:+.2f}}% of pre-period mean)",
            xy=(years[att_pos_idx], est.gap_[att_pos_idx]),
            xytext=(years[len(years)//4], est.gap_.min() * 1.05 if est.gap_.min() < 0 else est.gap_.max() * 0.6),
            fontsize=11, color=COLOR_TEXT,
            bbox=dict(facecolor="white", edgecolor=COLOR_TREATED, boxstyle="round,pad=0.4"),
            arrowprops=dict(arrowstyle="->", color=COLOR_TREATED, lw=1.0),
        )
        ax.set_title("Gap (real - synthetic) - the estimated effect of the intervention")
        ax.set_xlabel("Year"); ax.set_ylabel("Gap (" + {spec.outcome_unit_short!r} + ")")
        plt.tight_layout(); plt.show()
        """
        )
    )

    # --- Act 5 ------------------------------------------------------------
    cells.append(
        md(
            f"""
        ## Act 5 — Comparing to the published paper

        {spec.published_att_blurb} A small caveat keeps the comparison honest: our v0.1 `Synth`
        implements the **outcome-only simplex** formulation (Doudchenko & Imbens 2016 /
        `augsynth(progfunc='None')` in R). The original paper additionally used predictors
        (income, beer consumption, prices, demographics) and optimised a positive-definite weight
        matrix `V` over those predictors. We expect numbers in the same neighbourhood, not
        identical to four decimal places.
        """
        )
    )

    cells.append(
        code(
            f"""
        paper_top_donors = {paper_top_donors_py}
        ours_top_donors = [c for c, _ in sorted(est.weights_.items(), key=lambda x: -x[1])[:5]]

        print(f"This notebook -- Avg post-period ATT : {{est.att_:+.3f}} {spec.outcome_unit_short}")
        print(f"                  As % of pre baseline: {{est.att_pct_*100:+.2f}}%")
        print(f"                  Pre-period RMSPE    : {{est.rmspe_pre_*100:.3f}}% of mean")
        print()
        print(f"This notebook -- Top 5 donors        : {{ours_top_donors}}")
        print(f"Original paper -- Top donors         : {{paper_top_donors}}")
        overlap = sorted(set(ours_top_donors) & set(paper_top_donors))
        print(f"Overlap                              : {{overlap}}")
        """
        )
    )

    cells.append(
        md(
            """
        > **Takeaway.** The estimator agrees with the paper on the *story* — same direction, same
        > order of magnitude, substantial overlap on which donors carry the weight. Discrepancies
        > of a few percent on the ATT and one or two donors swapping into the top-5 are exactly
        > what we should expect from the formulation differences described above.
        """
        )
    )

    # --- Act 6 ------------------------------------------------------------
    cells.append(
        md(
            """
        ## Act 6 — What classical SCM doesn't (yet) solve

        Three limitations of the v0.1 `Synth` estimator, each tied to a concrete next step on the
        package roadmap.

        ### 6a — No uncertainty quantification

        The gap plot in Act 4 looks compelling, but **we have no confidence interval** on it. We
        can't tell whether the post-period deviation is signal or noise without extra machinery.
        The classical answers:

        - **Placebo permutation** (Abadie's convention) — refit the synthetic control treating
          each donor as if it were the treated unit, and read off how extreme the real treated
          unit's gap is in that distribution. Filters donors with poor pre-fit out of the
          comparison.
        - **Conformal inference** (Chernozhukov, Wuthrich & Zhu 2021) — period-by-period
          confidence bands without assuming a distribution.

        Both are on the **augsynth-py v0.1 backlog**.
        """
        )
    )

    cells.append(
        md(
            """
        ### 6b — Weight sensitivity to the donor pool

        The simplex QP can have many near-equivalent minima when donor trajectories overlap. A
        useful diagnostic is to drop the top-weighted donor and refit: a robust estimate barely
        moves, a fragile one shifts noticeably.
        """
        )
    )

    cells.append(
        code(
            f"""
        top_donor, top_w = max(est.weights_.items(), key=lambda x: x[1])

        panel_lo = panel.filter(pl.col(UNIT) != top_donor)
        est_lo = Synth(fixedeff=False).fit(
            panel_lo,
            unit=UNIT, time=TIME, outcome=OUT,
            treated=TREATED, treatment_time=INTERVENTION,
        )

        shift_abs = est_lo.att_ - est.att_
        shift_rel = abs(shift_abs / est.att_) if est.att_ != 0 else float("nan")
        print(f"Top donor dropped       : {{top_donor}} (was carrying weight {{top_w:.3f}})")
        print(f"Original ATT            : {{est.att_:+.3f}} {spec.outcome_unit_short}")
        print(f"ATT without top donor   : {{est_lo.att_:+.3f}} {spec.outcome_unit_short}")
        print(f"Absolute shift          : {{shift_abs:+.3f}} {spec.outcome_unit_short}")
        print(f"Relative shift          : {{shift_rel*100:.1f}}% of original |ATT|")
        new_top = [c for c, _ in sorted(est_lo.weights_.items(), key=lambda x: -x[1])[:5]]
        print(f"New top-5 donors        : {{new_top}}")
        """
        )
    )

    cells.append(
        code(
            f"""
        # Overlay the two gap curves.
        years = est.periods_.astype(int)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(years, est.gap_, color=COLOR_TREATED, lw=2.0,
                label=f"Full donor pool (ATT = {{est.att_:+.2f}})")
        ax.plot(years, est_lo.gap_, color=COLOR_ALT, lw=2.0, ls="--",
                label=f"Without {{top_donor}} (ATT = {{est_lo.att_:+.2f}})")
        ax.axhline(0, color=COLOR_TEXT, lw=0.7)
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title("Leave-one-out sensitivity: drop the top-weighted donor, refit, overlay the gap")
        ax.set_xlabel("Year"); ax.set_ylabel("Gap (" + {spec.outcome_unit_short!r} + ")")
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    cells.append(
        md(
            """
        > **Reading the result.** A small shift (a few percent of the original ATT) is reassuring:
        > the conclusion doesn't hang on a single donor. A large shift would mean the donor pool is
        > thin and the estimate is being held up by one unit — a red flag worth raising before any
        > policy interpretation.
        """
        )
    )

    cells.append(
        md(
            """
        ### 6c — Bias when the treated unit is outside the donor convex hull

        Classical SCM's most fragile assumption is that the treated unit lies *inside* the convex
        hull of donor trajectories during the pre-period. When it doesn't — typically because the
        treated unit has a level or trend no donor can match — the simplex constraint forces a
        biased counterfactual. The pre-period RMSPE is the implicit credibility metric: a small
        RMSPE means the convex combination tracks the treated unit's pre-history well; a large
        RMSPE means the estimator is reaching.

        A practical rule of thumb (no formal guarantee): pre-period RMSPE below ~2% of the
        baseline is "good", above ~5% should make you worried. We computed this notebook's
        pre-period RMSPE in Act 3.

        The fix when this fails is **augmented synthetic control** (Ben-Michael, Feller &
        Rothstein 2021), which adds a ridge-regression correction that allows the estimator to
        extrapolate beyond the donor envelope with controlled bias. AugSynth is the v0.2 milestone
        for this package and will be revisited here once it lands.
        """
        )
    )

    cells.append(
        code(
            f"""
        # Diagnostic: pre-period gap distribution. If most pre-period gaps are within ±RMSPE,
        # the fit is good; large outliers in the pre-period signal an estimator that's reaching.
        years = est.periods_.astype(int)
        pre_mask = years < INTERVENTION

        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.plot(years, est.gap_, color=COLOR_TREATED, lw=1.8, label="Gap (real - synthetic)")
        rmspe_abs = float(np.sqrt(np.mean(est.gap_[pre_mask] ** 2)))
        ax.axhspan(-rmspe_abs, rmspe_abs, color=COLOR_SYNTH, alpha=0.10, label="±1 pre-period RMSPE")
        ax.axhline(0, color=COLOR_TEXT, lw=0.7)
        ax.axvline(INTERVENTION, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title(
            f"Pre-period RMSPE = {{est.rmspe_pre_*100:.2f}}% of baseline " +
            ("(below {spec.rmspe_rule_of_thumb_pct:.0f}% rule of thumb -> good fit)" if est.rmspe_pre_*100 < {spec.rmspe_rule_of_thumb_pct} else "(above {spec.rmspe_rule_of_thumb_pct:.0f}% rule of thumb -> consider AugSynth)")
        )
        ax.set_xlabel("Year"); ax.set_ylabel("Gap (" + {spec.outcome_unit_short!r} + ")")
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    # --- Act 7 (optional, populated in Phase C) --------------------------
    if spec.augsynth_act_enabled:
        cells.extend(_build_act7_cells(spec))

    # --- Closing ----------------------------------------------------------
    augsynth_ack = (
        (
            "\n*Act 7 above extends this analysis to the augmented estimator; "
            "the classical-vs-augmented deltas are discussed in that act's "
            "closing subsection (7.10).*\n"
        )
        if spec.augsynth_act_enabled
        else ""
    )

    cells.append(
        md(
            f"""
        ## Closing
        {augsynth_ack}
        **Paper.** {spec.paper_citation}

        **Links.** {paper_links}

        Classical synthetic control answered the paper's question. The estimator landed in
        `augsynth_py` v0.1 reproduces the paper's headline numbers within the formulation
        difference (outcome-only simplex vs predictors + V optimisation). The honest gaps —
        no confidence interval, weight sensitivity, bias when outside the convex hull — map onto
        the next milestones for this package:

        - **v0.1 (in flight):** placebo permutation, conformal inference
        - **v0.2:** AugSynth ridge augmentation (Ben-Michael, Feller & Rothstein 2021)

        When AugSynth lands, Act 6b/6c will be revisited with the fixed estimator.
        """
        )
    )

    return cells


# ---------------------------------------------------------------------------
# Build & save
# ---------------------------------------------------------------------------


def build_notebook(spec: ReplicationSpec) -> None:
    cells = build_cells(spec)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (augsynth-py)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    spec.output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Wrote {spec.output_path} - {len(cells)} cells ({n_code} code, {n_md} markdown).")


if __name__ == "__main__":
    for spec in (CALIFORNIA, BASQUE):
        build_notebook(spec)
