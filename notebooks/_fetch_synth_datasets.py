"""One-shot extractor for the canonical synthetic-control replication datasets.

Pulls two panels out of R and writes them as CSVs under ``notebooks/_data/``.
The CSVs are the source of truth checked into the repo; this script exists for
auditability ("here is exactly how we got these"), not as a runtime dependency.

Usage:

    .venv/bin/python notebooks/_fetch_synth_datasets.py

Re-running is idempotent — it overwrites the CSVs.

Datasets:
* ``smoking_adh2010.csv``: 39 US states × 31 years (1970-2000), from
  ``tidysynth::smoking``. Used by Abadie, Diamond & Hainmueller (2010) to
  estimate the effect of California's Proposition 99 on cigarette consumption.
* ``basque_ag2003.csv``: 18 Spanish regions × 43 years (1955-1997), from
  ``Synth::basque``. Used by Abadie & Gardeazabal (2003) to estimate the
  economic cost of the 1975 terrorism onset on the Basque Country.

R-side requirements: ``tidysynth`` and ``Synth`` installed. ``tidysynth`` is
installed with::

    Rscript -e 'install.packages("tidysynth")'
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "_data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r("suppressPackageStartupMessages(library(tidysynth))")
    ro.r("suppressPackageStartupMessages(library(Synth))")

    ro.r("data(smoking, package = 'tidysynth')")
    ro.r("data(basque, package = 'Synth')")

    with localconverter(ro.default_converter + pandas2ri.converter):
        smoking = ro.conversion.rpy2py(ro.r("as.data.frame(smoking)"))
        basque = ro.conversion.rpy2py(ro.r("as.data.frame(basque)"))

    smoking_path = DATA_DIR / "smoking_adh2010.csv"
    basque_path = DATA_DIR / "basque_ag2003.csv"
    smoking.to_csv(smoking_path, index=False)
    basque.to_csv(basque_path, index=False)

    print(f"Wrote {smoking_path} — shape {smoking.shape}")
    print(f"  columns: {list(smoking.columns)}")
    print(f"  states : {smoking['state'].nunique()}")
    print(f"  years  : {smoking['year'].min():.0f}-{smoking['year'].max():.0f}")
    print()
    print(f"Wrote {basque_path} — shape {basque.shape}")
    print(f"  columns: {list(basque.columns)[:6]}...")
    print(f"  regions: {basque['regionname'].nunique()}")
    print(f"  years  : {basque['year'].min():.0f}-{basque['year'].max():.0f}")


if __name__ == "__main__":
    main()
