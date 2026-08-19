"""One-shot extractor for the Jericoacoara airport panel (novel case study).

Builds ``notebooks/_data/jericoacoara_cempre.csv`` from two public IBGE APIs —
no API key required. The CSV checked into the repo is the source of truth;
this script exists for auditability ("here is exactly how we got it") and to
regenerate the panel from scratch. Re-running is idempotent — it overwrites
the CSV, so ``git diff`` after a re-run is the audit trail.

Usage::

    .venv/bin/python notebooks/_fetch_jericoacoara.py

Case study: the Comandante Ariston Pessoa regional airport opened in
Jijoca de Jericoacoara (CE) on 2017-06-24, replacing a 5-6 hour overland
trip from Fortaleza as the only way into the Jericoacoara beach village.
Notebook ``04_jericoacoara_airport.ipynb`` estimates the airport's effect on
the town's formal labor market with synthetic control methods.

Sources (all fetched at run time):

* **CEMPRE** (Cadastro Central de Empresas), SIDRA table 1685: salaried
  employment and local business units per municipality, annual 2006-2021
  (the series was discontinued after 2021). Section-level CNAE detail is
  only published for municipalities with 50k+ inhabitants (table 3421),
  which excludes Jijoca (~19k), so we use municipality totals — defensible
  here because tourism dominates the local formal economy.
* **Population estimates**, SIDRA table 6579: annual municipal population,
  used to express employment per 1,000 inhabitants. Census years are absent
  from the estimates table and are linearly interpolated (flagged at run
  time).

Variable and municipality identifiers are *discovered at run time by name*
(via the table metadata and the localidades API) rather than hard-coded, so
the script fails loudly with the available options if IBGE renames anything.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "_data"
OUT_CSV = DATA_DIR / "jericoacoara_cempre.csv"

LOCALIDADES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
METADATA_URL = "https://servicodados.ibge.gov.br/api/v3/agregados/{table}/metadados"
VALUES_URL = (
    "https://apisidra.ibge.gov.br/values/t/{table}/n6/{codes}/v/{variables}/p/all?formato=json"
)

CEMPRE_TABLE = 1685
POPULATION_TABLE = 6579

# Normalized (lowercase, accent-stripped) variable name -> CSV column.
CEMPRE_VARIABLES = {
    "pessoal ocupado assalariado": "salaried_workers",
    "numero de unidades locais": "local_units",
}
POPULATION_VARIABLES = {
    "populacao residente estimada": "population",
}

# (name, UF, role). Roles:
#   treated  — Jijoca de Jericoacoara, where the airport opened in 2017.
#   donor    — coastal tourism municipalities in CE/RN/PI/MA with no own
#              airport over 2006-2021 and outside the Fortaleza metro area.
#   neighbor — Cruz and Camocim border the Jericoacoara area and share its
#              tourist flow; they are fetched for spillover diagnostics but
#              MUST stay out of the donor pool (SUTVA).
# Deliberately excluded (documented, not fetched): Aracati (own airport from
# 2018), Parnaiba (own airport), Caucaia/Aquiraz (Fortaleza metro area).
ROSTER: list[tuple[str, str, str]] = [
    ("Jijoca de Jericoacoara", "CE", "treated"),
    ("Acaraú", "CE", "donor"),
    ("Itarema", "CE", "donor"),
    ("Amontada", "CE", "donor"),
    ("Trairi", "CE", "donor"),
    ("Paraipaba", "CE", "donor"),
    ("Paracuru", "CE", "donor"),
    ("Beberibe", "CE", "donor"),
    ("Icapuí", "CE", "donor"),
    ("Tibau do Sul", "RN", "donor"),
    ("São Miguel do Gostoso", "RN", "donor"),
    ("Touros", "RN", "donor"),
    ("Luís Correia", "PI", "donor"),
    ("Cajueiro da Praia", "PI", "donor"),
    ("Barreirinhas", "MA", "donor"),
    ("Tutóia", "MA", "donor"),
    ("Cruz", "CE", "neighbor"),
    ("Camocim", "CE", "neighbor"),
]

ROLE_ORDER = {"treated": 0, "donor": 1, "neighbor": 2}


def _normalize(text: str) -> str:
    """Lowercase and strip accents so name matching survives spelling drift."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def _get_json(url: str, *, tries: int = 4) -> Any:
    """GET a JSON document with exponential-backoff retries."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "augsynth-py/_fetch_jericoacoara.py"}
    )
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == tries:
                raise RuntimeError(f"GET {url} failed after {tries} tries: {exc}") from exc
            print(f"  retry {attempt}/{tries - 1} after error: {exc}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def resolve_municipality_codes() -> dict[str, tuple[str, str, str]]:
    """Resolve roster names to IBGE codes: code -> (name, uf, role)."""
    by_uf: dict[str, list[tuple[str, str, str]]] = {}
    for name, uf, role in ROSTER:
        by_uf.setdefault(uf, []).append((name, uf, role))

    resolved: dict[str, tuple[str, str, str]] = {}
    missing: list[str] = []
    for uf, entries in sorted(by_uf.items()):
        listing = _get_json(LOCALIDADES_URL.format(uf=uf))
        codes_by_name = {_normalize(m["nome"]): str(m["id"]) for m in listing}
        for name, _, role in entries:
            code = codes_by_name.get(_normalize(name))
            if code is None:
                missing.append(f"{name} ({uf})")
            else:
                resolved[code] = (name, uf, role)
    if missing:
        raise RuntimeError(
            "Municipalities not found in the IBGE localidades API "
            f"(check spelling against the API): {missing}"
        )
    return resolved


def resolve_variable_ids(table: int, wanted: dict[str, str]) -> dict[str, str]:
    """Map wanted variable names to their SIDRA IDs: variable_id -> CSV column.

    Matches on normalized equality first, then on a unique bidirectional
    substring (so "numero de unidades locais" still finds a variable named
    plain "Unidades locais"), and fails loudly listing the table's variables
    otherwise.
    """
    metadata = _get_json(METADATA_URL.format(table=table))
    available = {str(v["id"]): _normalize(str(v["nome"])) for v in metadata["variaveis"]}

    resolved: dict[str, str] = {}
    for wanted_name, column in wanted.items():
        exact = [vid for vid, name in available.items() if name == wanted_name]
        loose = [
            vid for vid, name in available.items() if wanted_name in name or name in wanted_name
        ]
        matches = exact or (loose if len(loose) == 1 else [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Table {table}: expected exactly one variable matching "
                f"{wanted_name!r}, got {matches or 'none'}. Available: "
                f"{sorted(available.values())}"
            )
        resolved[matches[0]] = column
    return resolved


def _parse_value(raw: str, *, context: str, warnings: list[str]) -> float | None:
    """Decode a SIDRA cell. '-' is an absolute zero; 'X'/'..'/'...' are masked."""
    text = raw.strip()
    if text == "-":
        return 0.0
    if text in {"X", "..", "..."}:
        warnings.append(f"{context}: masked value {text!r}")
        return None
    try:
        return float(text)
    except ValueError:
        warnings.append(f"{context}: unparseable value {raw!r}")
        return None


def fetch_values(
    table: int,
    codes: list[str],
    variables: dict[str, str],
    warnings: list[str],
) -> dict[tuple[str, int], dict[str, float | None]]:
    """Fetch table values: (municipality_code, year) -> {csv_column: value}."""
    url = VALUES_URL.format(table=table, codes=",".join(codes), variables=",".join(variables))
    rows = _get_json(url)
    if not isinstance(rows, list) or len(rows) < 2:
        raise RuntimeError(f"Unexpected apisidra response shape for table {table}: {rows!r}")

    # Row 0 maps short keys to header labels; find the dimension keys by label.
    header = rows[0]

    def key_for(label_prefix: str) -> str:
        matches = [
            key
            for key, label in header.items()
            if _normalize(str(label)).startswith(_normalize(label_prefix))
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Table {table}: expected one header matching {label_prefix!r}, "
                f"got {matches} in {header!r}"
            )
        return matches[0]

    municipality_key = key_for("municipio (codigo)")
    variable_key = key_for("variavel (codigo)")
    year_key = key_for("ano (codigo)")

    values: dict[tuple[str, int], dict[str, float | None]] = {}
    for row in rows[1:]:
        code = str(row[municipality_key])
        year = int(row[year_key])
        column = variables[str(row[variable_key])]
        context = f"table {table}, municipality {code}, {column}, {year}"
        cell = _parse_value(str(row["V"]), context=context, warnings=warnings)
        values.setdefault((code, year), {})[column] = cell
    return values


def interpolate_population(
    by_year: dict[int, float], years: list[int], *, context: str, warnings: list[str]
) -> dict[int, float | None]:
    """Fill missing years linearly (census years are absent from table 6579)."""
    filled: dict[int, float | None] = {}
    known = sorted(y for y, v in by_year.items() if v is not None)
    for year in years:
        if year in by_year and by_year[year] is not None:
            filled[year] = by_year[year]
            continue
        lower = max((y for y in known if y < year), default=None)
        upper = min((y for y in known if y > year), default=None)
        if lower is not None and upper is not None:
            weight = (year - lower) / (upper - lower)
            filled[year] = by_year[lower] + weight * (by_year[upper] - by_year[lower])
            warnings.append(f"{context}: population for {year} linearly interpolated")
        elif lower is not None or upper is not None:
            nearest = lower if lower is not None else upper
            filled[year] = by_year[nearest]  # type: ignore[index]
            warnings.append(f"{context}: population for {year} carried from {nearest}")
        else:
            filled[year] = None
            warnings.append(f"{context}: population for {year} unavailable")
    return filled


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    print("Resolving municipality codes (IBGE localidades API)...")
    municipalities = resolve_municipality_codes()
    codes = sorted(municipalities)

    print(f"Resolving variable IDs (SIDRA tables {CEMPRE_TABLE} and {POPULATION_TABLE})...")
    cempre_vars = resolve_variable_ids(CEMPRE_TABLE, CEMPRE_VARIABLES)
    population_vars = resolve_variable_ids(POPULATION_TABLE, POPULATION_VARIABLES)

    print("Fetching CEMPRE values...")
    cempre = fetch_values(CEMPRE_TABLE, codes, cempre_vars, warnings)
    print("Fetching population estimates...")
    population = fetch_values(POPULATION_TABLE, codes, population_vars, warnings)

    years = sorted({year for _, year in cempre})
    population_filled: dict[str, dict[int, float | None]] = {}
    for code in codes:
        name, uf, _ = municipalities[code]
        by_year = {
            year: cells.get("population") for (c, year), cells in population.items() if c == code
        }
        population_filled[code] = interpolate_population(
            by_year, years, context=f"{name} ({uf})", warnings=warnings
        )

    rows_out: list[dict[str, Any]] = []
    for code in codes:
        name, uf, role = municipalities[code]
        for year in years:
            cells = cempre.get((code, year), {})
            workers = cells.get("salaried_workers")
            units = cells.get("local_units")
            pop = population_filled[code][year]
            per_1k = round(workers / pop * 1000.0, 4) if workers is not None and pop else None
            rows_out.append(
                {
                    "municipality": name,
                    "uf": uf,
                    "ibge_code": code,
                    "role": role,
                    "year": year,
                    "salaried_workers": None if workers is None else int(workers),
                    "local_units": None if units is None else int(units),
                    "population": None if pop is None else round(pop),
                    "workers_per_1k": per_1k,
                }
            )
    rows_out.sort(key=lambda r: (ROLE_ORDER[r["role"]], r["municipality"], r["year"]))

    fieldnames = list(rows_out[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print()
    print(f"Wrote {OUT_CSV} — {len(rows_out)} rows")
    print(
        f"  municipalities: {len(codes)} ({sum(1 for m in municipalities.values() if m[2] == 'donor')} donors)"
    )
    print(f"  years         : {years[0]}-{years[-1]}")
    if warnings:
        print(f"  {len(warnings)} data warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  no masked or interpolated cells beyond census-year population")


if __name__ == "__main__":
    main()
