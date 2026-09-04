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
from typing import Any

import polars as pl

DATA_DIR = Path(__file__).parent / "_data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import rpy2.robjects as ro

    ro.r("suppressPackageStartupMessages(library(tidysynth))")
    ro.r("suppressPackageStartupMessages(library(Synth))")

    ro.r("data(smoking, package = 'tidysynth')")
    ro.r("data(basque, package = 'Synth')")

    def from_r(frame: Any) -> pl.DataFrame:
        columns = {}
        for name in frame.names:
            column = frame.rx2(name)
            if "factor" in column.rclass:
                column = ro.r["as.character"](column)
            columns[str(name)] = list(column)
        return pl.DataFrame(columns)

    smoking = from_r(ro.r("as.data.frame(smoking)"))
    basque = from_r(ro.r("as.data.frame(basque)"))

    smoking_path = DATA_DIR / "smoking_adh2010.csv"
    basque_path = DATA_DIR / "basque_ag2003.csv"
    smoking.write_csv(smoking_path)
    basque.write_csv(basque_path)

    print(f"Wrote {smoking_path} — shape {smoking.shape}")
    print(f"  columns: {smoking.columns}")
    print(f"  states : {smoking.get_column('state').n_unique()}")
    print(
        f"  years  : {smoking.get_column('year').min():.0f}-{smoking.get_column('year').max():.0f}"
    )
    print()
    print(f"Wrote {basque_path} — shape {basque.shape}")
    print(f"  columns: {basque.columns[:6]}...")
    print(f"  regions: {basque.get_column('regionname').n_unique()}")
    print(f"  years  : {basque.get_column('year').min():.0f}-{basque.get_column('year').max():.0f}")


if __name__ == "__main__":
    main()
