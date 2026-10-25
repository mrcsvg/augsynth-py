"""Fetch and cache the data behind ``04_tarifaco_eua_brasil.ipynb``.

Usage::

    .venv/bin/python notebooks/_fetch_tarifaco_data.py

Everything comes from public APIs that need no key. Both are rate-limited, so
a cold run takes 20-30 minutes; raw responses are cached under
``notebooks/scratch/tarifaco_cache/`` (gitignored) and a re-run only fetches
what is missing. The compact CSVs written to ``notebooks/_data/tarifaco/`` are
the checked-in source of truth the notebook reads; this script exists for
auditability ("here is exactly how we got these"), not as a runtime
dependency.

Sources
-------
* **UN Comtrade** (public preview API, ``comtradeapi.un.org/public/v1``):
  monthly exports of ~60 economies to the world and to the United States,
  total and HS chapter 27 (mineral fuels). Reporter-side, FOB, US$.
* **Comex Stat** (MDIC/Secex, ``api-comexstat.mdic.gov.br``): Brazilian
  exports, monthly, by HS6 to the US and by HS4 x destination country.
* **Executive Orders 14323 (Jul 30, 2025) and 14361 (Nov 20, 2025)**,
  Federal Register: Annex I, the HTSUS 8-digit lines exempted from the 40%
  surcharge. Parsed once with ``pdftotext -layout`` (poppler) into
  ``anexo_i_ordens_executivas.csv``; that CSV is checked in, so poppler is
  only needed to regenerate it.

Outputs (``notebooks/_data/tarifaco/``)
---------------------------------------
* ``comtrade_exportacoes.csv`` -- pais, mes, destino (mundo|eua),
  total_usd, combustiveis_usd.
* ``anexo_i_ordens_executivas.csv`` -- hts8, lista
  (eo14323_geral | eo14323_aeronaves | eo14361_inclusao).
* ``classificacao_sh6.csv`` -- sh6, descricao, categoria
  (tarifado | isento | isento_aeronaves | secao_232), alivio_nov25,
  fob_eua_2024.
* ``comex_eua_categoria.csv`` -- mes, categoria, alivio_nov25, fob_usd.
* ``comex_eua_isentos_sh4.csv`` -- mes, sh4, fob_usd: exports to the US of
  the HS6 lines with a general Annex I exemption, by HS4.
* ``comex_margens.csv`` -- referencia, grupo, mes, destino_tipo, fob_usd:
  non-US exports of each product group split into product-destination pairs
  that already existed before the reference date and pairs that did not.
* ``comex_novos_pares.csv`` -- referencia, grupo, pais, sh4, fob_usd: the
  new pairs of the tariffed group themselves, summed over 12 months.
* ``sh4_grupos.csv`` -- sh4, descricao, grupo, share_tarifado,
  exposicao_eua_2024, fob_total_2024, fob_eua_2024.
* ``sh4_janelas.csv`` -- sh4, grupo, destino (eua|outros), janela
  (antes|tarifaco), fob_usd: exports per heading over Aug-Feb before and
  during the surcharge.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl

HERE = Path(__file__).parent
OUT_DIR = HERE / "_data" / "tarifaco"
CACHE_DIR = HERE / "scratch" / "tarifaco_cache"

FIRST_MONTH = "2019-01"
LAST_MONTH = "2026-09"  # latest month Comex Stat had published on the fetch date

# ---------------------------------------------------------------------------
# UN Comtrade
# ---------------------------------------------------------------------------

COMTRADE_URL = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"

# M49 reporter code -> display name. Brazil plus a broad pool of candidate
# donors; the notebook decides which of them enter the synthetic control.
REPORTERS: dict[int, str] = {
    76: "Brasil",
    32: "Argentina",
    152: "Chile",
    170: "Colômbia",
    604: "Peru",
    218: "Equador",
    858: "Uruguai",
    600: "Paraguai",
    68: "Bolívia",
    188: "Costa Rica",
    320: "Guatemala",
    214: "Rep. Dominicana",
    484: "México",
    591: "Panamá",
    222: "El Salvador",
    340: "Honduras",
    558: "Nicarágua",
    36: "Austrália",
    554: "Nova Zelândia",
    826: "Reino Unido",
    702: "Singapura",
    682: "Arábia Saudita",
    784: "Emirados Árabes",
    792: "Turquia",
    818: "Egito",
    504: "Marrocos",
    376: "Israel",
    579: "Noruega",
    804: "Ucrânia",
    276: "Alemanha",
    251: "França",
    380: "Itália",
    724: "Espanha",
    528: "Países Baixos",
    56: "Bélgica",
    616: "Polônia",
    620: "Portugal",
    372: "Irlanda",
    40: "Áustria",
    752: "Suécia",
    208: "Dinamarca",
    246: "Finlândia",
    203: "Tchéquia",
    348: "Hungria",
    392: "Japão",
    410: "Coreia do Sul",
    360: "Indonésia",
    458: "Malásia",
    608: "Filipinas",
    764: "Tailândia",
    704: "Vietnã",
    490: "Taiwan",
    586: "Paquistão",
    50: "Bangladesh",
    144: "Sri Lanka",
    116: "Camboja",
    156: "China",
    699: "Índia",
    124: "Canadá",
    757: "Suíça",
    710: "África do Sul",
}


def _months(first: str, last: str) -> list[str]:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y}-{m:02d}" <= last:
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _get_json(url: str, data: bytes | None = None) -> Any:
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # rate limits (429) and timeouts: back off and retry
            wait = 5 * (attempt + 1)
            print(f"  retry in {wait}s after {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def _spread_cmd(long: pl.DataFrame) -> pl.DataFrame:
    """One column per commodity code (total and chapter 27)."""
    return long.pivot(
        on="cmd", index=["pais", "mes", "destino"], values="value", aggregate_function="sum"
    )


def fetch_comtrade() -> pl.DataFrame:
    cache = CACHE_DIR / "comtrade"
    cache.mkdir(parents=True, exist_ok=True)
    reporters = ",".join(map(str, REPORTERS))
    for ym in _months(FIRST_MONTH, LAST_MONTH):
        path = cache / f"{ym.replace('-', '')}.json"
        if path.exists():
            continue
        url = (
            f"{COMTRADE_URL}?reporterCode={reporters}&period={ym.replace('-', '')}"
            "&partnerCode=0,842&cmdCode=TOTAL,27&flowCode=X"
        )
        payload = _get_json(url)
        while "data" not in payload:  # {"statusCode": 429, ...} comes back as a 200 body
            time.sleep(10)
            payload = _get_json(url)
        path.write_text(json.dumps(payload["data"]))
        print(f"comtrade {ym}: {len(payload['data'])} rows", flush=True)
        time.sleep(4)

    rows = [
        (r["period"], r["reporterCode"], r["partnerCode"], r["cmdCode"], r["primaryValue"])
        for path in sorted(cache.glob("*.json"))
        for r in json.loads(path.read_text())
    ]
    long = pl.DataFrame(rows, schema=["period", "rep", "partner", "cmd", "value"], orient="row")
    labelled = long.filter(pl.col("rep").is_in(list(REPORTERS))).with_columns(
        pl.col("rep").replace_strict(REPORTERS, return_dtype=pl.Utf8).alias("pais"),
        pl.col("period").str.strptime(pl.Date, "%Y%m").alias("mes"),
        pl.col("partner").replace_strict({0: "mundo", 842: "eua"}).alias("destino"),
        pl.col("cmd").replace_strict({"TOTAL": "total_usd", "27": "combustiveis_usd"}),
    )
    wide = (
        _spread_cmd(labelled)
        # A reporter-month with no chapter-27 row exported no fuels; a missing
        # TOTAL row means the month is not reported and the row is dropped.
        .filter(pl.col("total_usd").is_not_null())
        .with_columns(pl.col("combustiveis_usd").fill_null(0.0))
        .select("pais", "mes", "destino", "total_usd", "combustiveis_usd")
        .sort("pais", "mes", "destino")
    )
    return wide


# ---------------------------------------------------------------------------
# Comex Stat
# ---------------------------------------------------------------------------

COMEX_URL = "https://api-comexstat.mdic.gov.br/general?language=pt"
US_COUNTRY_CODE = 249  # "Estados Unidos" in Comex Stat's country table
US_NAME = "Estados Unidos"


def _comex(body: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _get_json(COMEX_URL, json.dumps(body).encode())
    while "data" not in payload:  # {"error": {"code": 500, "message": "timeout..."}}
        print(f"  comex error {payload}; retrying", flush=True)
        time.sleep(10)
        payload = _get_json(COMEX_URL, json.dumps(body).encode())
    return list(payload["data"]["list"])


def fetch_comex_us_hs6() -> pl.DataFrame:
    """Monthly exports to the US by HS6, FIRST_MONTH..LAST_MONTH."""
    cache = CACHE_DIR / "comex"
    cache.mkdir(parents=True, exist_ok=True)
    for year in range(int(FIRST_MONTH[:4]), int(LAST_MONTH[:4]) + 1):
        path = cache / f"us_hs6_{year}.csv"
        if path.exists():
            continue
        last = LAST_MONTH[5:] if year == int(LAST_MONTH[:4]) else "12"
        rows: list[dict[str, Any]] = []
        # Half-years: a full year in one request times out server-side.
        for lo, hi in (("01", "06"), ("07", last)):
            if hi < lo:
                continue
            rows += _comex(
                {
                    "flow": "export",
                    "monthDetail": True,
                    "period": {"from": f"{year}-{lo}", "to": f"{year}-{min(hi, last)}"},
                    "filters": [{"filter": "country", "values": [US_COUNTRY_CODE]}],
                    "details": ["subHeading"],
                    "metrics": ["metricFOB"],
                }
            )
            time.sleep(1)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["year", "month", "hs6", "fob"])
            for r in rows:
                writer.writerow([r["year"], r["monthNumber"], r["subHeadingCode"], r["metricFOB"]])
        print(f"comex us_hs6 {year}: {len(rows)} rows", flush=True)

    frames = [
        pl.read_csv(p, schema_overrides={"hs6": pl.Utf8, "month": pl.Utf8})
        for p in sorted(cache.glob("us_hs6_*.csv"))
    ]
    return pl.concat(frames).select(
        pl.date(pl.col("year"), pl.col("month").cast(pl.Int32), 1).alias("mes"),
        pl.col("hs6").str.zfill(6).alias("sh6"),
        pl.col("fob").cast(pl.Float64),
    )


PAIRS_FIRST_MONTH = "2022-08"  # 24-month base window before the placebo reference date


def fetch_comex_pairs() -> pl.DataFrame:
    """Monthly exports by HS4 x destination country, PAIRS_FIRST_MONTH..LAST_MONTH."""
    cache = CACHE_DIR / "comex"
    cache.mkdir(parents=True, exist_ok=True)
    for ym in _months(PAIRS_FIRST_MONTH, LAST_MONTH):
        path = cache / f"pair_{ym}.csv"
        if path.exists():
            continue
        rows = _comex(
            {
                "flow": "export",
                "monthDetail": False,
                "period": {"from": ym, "to": ym},
                "details": ["country", "heading"],
                "metrics": ["metricFOB"],
            }
        )
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["country", "hs4", "fob"])
            for r in rows:
                writer.writerow([r["country"], r["headingCode"], r["metricFOB"]])
        print(f"comex pairs {ym}: {len(rows)} rows", flush=True)
        time.sleep(1)

    frames = []
    for path in sorted(cache.glob("pair_*.csv")):
        ym = path.stem.removeprefix("pair_")
        frames.append(
            pl.read_csv(path, schema_overrides={"hs4": pl.Utf8}).with_columns(
                pl.lit(dt.date(int(ym[:4]), int(ym[5:7]), 1)).alias("mes")
            )
        )
    return pl.concat(frames).select(
        "mes",
        pl.col("country").alias("pais"),
        pl.col("hs4").str.zfill(4).alias("sh4"),
        pl.col("fob").cast(pl.Float64),
    )


def fetch_comex_descriptions() -> tuple[dict[str, str], dict[str, str]]:
    """HS6 and HS4 descriptions (Portuguese), from one annual query each."""
    cache = CACHE_DIR / "comex"
    out = []
    for detail, code_key in (("subHeading", "subHeadingCode"), ("heading", "headingCode")):
        path = cache / f"names_{detail}.json"
        if not path.exists():
            rows = _comex(
                {
                    "flow": "export",
                    "monthDetail": False,
                    "period": {"from": "2024-01", "to": "2024-12"},
                    "details": [detail],
                    "metrics": ["metricFOB"],
                }
            )
            path.write_text(json.dumps({r[code_key]: r[detail] for r in rows}, ensure_ascii=False))
        out.append(json.loads(path.read_text()))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Annex I of EO 14323 / EO 14361
# ---------------------------------------------------------------------------

EO_PDFS = {
    # EO 14323 as published in the Federal Register, 90 FR 37739 (Aug 5, 2025).
    "eo14323": "https://www.govinfo.gov/content/pkg/FR-2025-08-05/pdf/2025-14896.pdf",
    # EO 14361 public-inspection copy (published 90 FR, Nov 26, 2025); its
    # Annex I is the full updated list, effective Nov 13, 2025.
    "eo14361": "https://public-inspection.federalregister.gov/2025-21417.pdf",
}
_HTS8 = re.compile(r"(?<![\d.])(\d{4}\.\d{2}\.\d{2})(?![\d.])(\s+\*)?")


def _annex_lines(name: str, start_marker: str, stop_marker: str) -> list[str]:
    cache = CACHE_DIR / "eo"
    cache.mkdir(parents=True, exist_ok=True)
    pdf, txt = cache / f"{name}.pdf", cache / f"{name}.txt"
    if not pdf.exists():
        with urllib.request.urlopen(EO_PDFS[name], timeout=120) as resp:
            pdf.write_bytes(resp.read())
    if not txt.exists():
        if shutil.which("pdftotext") is None:
            raise RuntimeError("pdftotext (poppler-utils) is needed to parse the annexes")
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
    lines = txt.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().replace(" ", "") == start_marker)
    stop = next(i for i, ln in enumerate(lines) if i > start and stop_marker in ln)
    return lines[start:stop]


def parse_annexes() -> pl.DataFrame:
    """HTS8 lines of Annex I. In EO 14323 an asterisk after the code marks the
    "civil aircraft only" lines; EO 14361 re-publishes the whole annex, so its
    inclusions are the lines it lists that EO 14323 did not."""
    general: set[str] = set()
    aircraft: set[str] = set()
    for ln in _annex_lines("eo14323", "ANNEXI", "ANNEX II"):
        for code, star in _HTS8.findall(ln):
            (aircraft if star else general).add(code)
    general -= aircraft
    updated = {
        code
        for ln in _annex_lines("eo14361", "ANNEXI", "ANNEX II")
        for code, _ in _HTS8.findall(ln)
    }
    added = updated - general - aircraft
    rows = (
        [(c, "eo14323_geral") for c in general]
        + [(c, "eo14323_aeronaves") for c in aircraft]
        + [(c, "eo14361_inclusao") for c in added]
    )
    return pl.DataFrame(rows, schema=["hts8", "lista"], orient="row").sort("hts8")


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------

# Section 232 products were carved out of the 40% (EO 14323, Sec. 3(a)):
# steel and aluminium (chapters 72, 73, 76), semi-finished copper and copper
# derivatives (7407-7419; cathodes and ores are outside the copper action),
# and autos / auto parts. Pig iron (7201) is not a 232 steel article; it is
# exempted by Annex I instead.
_S232_CHAPTERS = {"72", "73", "76"}
_S232_HEADINGS = {"8703", "8704", "8708"}
_S232_AUTO_PARTS_SH6 = {"840991", "840999", "401110", "401120", "848310", "850710"}
_PIG_IRON_SH6 = {"720110", "720120", "720150"}


def classify_sh6(sh6: str, annex_lists: set[str]) -> tuple[str, bool]:
    """Classify one HS6 line by how the Aug 6, 2025 surcharge treated it.

    ``annex_lists`` holds the Annex I lists (see :func:`parse_annexes`) that
    name at least one HTS8 line under ``sh6``. HTSUS and NCM share their first
    six digits, so an HS6 counts as ``isento`` when Annex I lists a general
    HTS8 line under it, and as ``isento_aeronaves`` when the only lines listed
    under it are "civil aircraft only" ones. The second group is a mix -- the
    Embraer supply chain was exempt, an electric motor bound for a factory was
    not -- so the notebook counts it as exempt when comparing with MDIC's
    split (§2) but never uses it as a control.
    """
    hs2, hs4 = sh6[:2], sh6[:4]
    is_232 = (
        hs2 in _S232_CHAPTERS or "7407" <= hs4 <= "7419" or hs4 in _S232_HEADINGS
    ) or sh6 in _S232_AUTO_PARTS_SH6
    if is_232 and sh6 not in _PIG_IRON_SH6:
        categoria = "secao_232"
    elif "eo14323_geral" in annex_lists:
        categoria = "isento"
    elif "eo14323_aeronaves" in annex_lists:
        categoria = "isento_aeronaves"
    else:
        categoria = "tarifado"
    alivio = categoria == "tarifado" and "eo14361_inclusao" in annex_lists
    return categoria, alivio


def build_classification(
    us_hs6: pl.DataFrame, annex: pl.DataFrame, names: dict[str, str]
) -> pl.DataFrame:
    by_sh6: dict[str, set[str]] = {}
    for hts8, lista in annex.iter_rows():
        by_sh6.setdefault(hts8.replace(".", "")[:6], set()).add(lista)
    fob_2024 = (
        us_hs6.filter(pl.col("mes").dt.year() == 2024).group_by("sh6").agg(pl.col("fob").sum())
    )
    rows = []
    for sh6 in sorted(set(us_hs6["sh6"].to_list())):
        categoria, alivio = classify_sh6(sh6, by_sh6.get(sh6, set()))
        rows.append((sh6, names.get(sh6, ""), categoria, alivio))
    out = pl.DataFrame(rows, schema=["sh6", "descricao", "categoria", "alivio_nov25"], orient="row")
    return (
        out.join(fob_2024, on="sh6", how="left")
        .with_columns(pl.col("fob").fill_null(0.0).alias("fob_eua_2024"))
        .drop("fob")
    )


def build_us_by_category(us_hs6: pl.DataFrame, classes: pl.DataFrame) -> pl.DataFrame:
    return (
        us_hs6.join(classes.select("sh6", "categoria", "alivio_nov25"), on="sh6", how="left")
        .group_by("mes", "categoria", "alivio_nov25")
        .agg(pl.col("fob").sum().alias("fob_usd"))
        .sort("mes", "categoria", "alivio_nov25")
    )


def build_us_exempt_sh4(us_hs6: pl.DataFrame, classes: pl.DataFrame) -> pl.DataFrame:
    """Monthly exports to the US of each HS4 heading, summed over its HS6
    lines with a general Annex I exemption -- the donor pool of the
    within-Brazil synthetic control."""
    exempt = classes.filter(pl.col("categoria") == "isento").select("sh6")
    return (
        us_hs6.join(exempt, on="sh6")
        .with_columns(pl.col("sh6").str.slice(0, 4).alias("sh4"))
        .group_by("mes", "sh4")
        .agg(pl.col("fob").sum().alias("fob_usd"))
        .sort("sh4", "mes")
    )


# Product groups for the extensive-margin analysis, at HS4. A heading is
# "tarifado" when most of its 2024 exports to the US fell under the 40% and
# the US mattered to it (>= 5% of its exports); "isento_exposto" is just as
# US-dependent but mostly escaped the 40% (general Annex I exemption or
# Section 232); "ambiguo" is US-dependent but mostly made of the "civil
# aircraft only" HS6 lines, whose real treatment cannot be told apart at HS6;
# everything else is "nao_exposto".
EXPOSURE_MIN = 0.05
TARIFF_SHARE_MIN = 0.5


def build_sh4_groups(
    pairs: pl.DataFrame, us_hs6: pl.DataFrame, classes: pl.DataFrame, names: dict[str, str]
) -> pl.DataFrame:
    us24 = (
        us_hs6.filter(pl.col("mes").dt.year() == 2024)
        .join(classes.select("sh6", "categoria"), on="sh6")
        .with_columns(pl.col("sh6").str.slice(0, 4).alias("sh4"))
        .group_by("sh4")
        .agg(
            pl.col("fob").sum().alias("fob_eua_2024"),
            pl.col("fob").filter(pl.col("categoria") == "tarifado").sum().alias("fob_tarifado"),
            pl.col("fob")
            .filter(pl.col("categoria") == "isento_aeronaves")
            .sum()
            .alias("fob_aeronaves"),
        )
    )
    tot24 = (
        pairs.filter(pl.col("mes").dt.year() == 2024)
        .group_by("sh4")
        .agg(pl.col("fob").sum().alias("fob_total_2024"))
    )
    g = tot24.join(us24, on="sh4", how="left").with_columns(
        pl.col("fob_eua_2024").fill_null(0.0),
        pl.col("fob_tarifado").fill_null(0.0),
        pl.col("fob_aeronaves").fill_null(0.0),
    )
    g = g.with_columns(
        (pl.col("fob_eua_2024") / pl.col("fob_total_2024"))
        .fill_nan(0.0)
        .alias("exposicao_eua_2024"),
        (pl.col("fob_tarifado") / pl.col("fob_eua_2024")).fill_nan(0.0).alias("share_tarifado"),
        (pl.col("fob_aeronaves") / pl.col("fob_eua_2024")).fill_nan(0.0).alias("share_aeronaves"),
    )
    exposed = pl.col("exposicao_eua_2024") >= EXPOSURE_MIN
    return (
        g.with_columns(
            pl.when(exposed & (pl.col("share_tarifado") >= TARIFF_SHARE_MIN))
            .then(pl.lit("tarifado"))
            .when(
                exposed & (pl.col("share_tarifado") + pl.col("share_aeronaves") >= TARIFF_SHARE_MIN)
            )
            .then(pl.lit("ambiguo"))
            .when(exposed)
            .then(pl.lit("isento_exposto"))
            .otherwise(pl.lit("nao_exposto"))
            .alias("grupo"),
            pl.col("sh4").replace_strict(names, default="").alias("descricao"),
        )
        .select(
            "sh4",
            "descricao",
            "grupo",
            "share_tarifado",
            "exposicao_eua_2024",
            "fob_total_2024",
            "fob_eua_2024",
        )
        .sort("sh4")
    )


# A product-destination pair (HS4 x country) is "existente" at a reference
# date when it shipped at least EXISTING_MIN_USD over a base window before
# it, and "novo" otherwise. Three references, as (reference date, base-window
# months, start of the 12-month window the new pairs are listed for):
#
# * "tarifaco": Aug 2025 (the 40% in force from Aug 6), 24-month base -- pairs
#   born after the surcharge.
# * "placebo": the same one year earlier, so that normal churn in new pairs
#   can be netted out.
# * "desde_2023": Aug 2023 with a 12-month base (Aug 2022 - Jul 2023, the
#   earliest the pair data goes) -- pairs born since the government's
#   market-opening drive began in 2023, which is the generous reading of
#   "new markets".
REFERENCES = {
    "tarifaco": (dt.date(2025, 8, 1), 24, dt.date(2025, 8, 1)),
    "placebo": (dt.date(2024, 8, 1), 24, dt.date(2024, 8, 1)),
    "desde_2023": (dt.date(2023, 8, 1), 12, dt.date(2025, 8, 1)),
}
FOLLOW_MONTHS = 12
EXISTING_MIN_USD = 100_000.0


def _shift(d: dt.date, months: int) -> dt.date:
    k = d.year * 12 + d.month - 1 + months
    return dt.date(k // 12, k % 12 + 1, 1)


def build_margins(pairs: pl.DataFrame, groups: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pairs = pairs.join(groups.select("sh4", "grupo"), on="sh4", how="left").with_columns(
        pl.col("grupo").fill_null("nao_exposto")
    )
    margins, new_pairs = [], []
    for ref, (tau, base_months, follow_start) in REFERENCES.items():
        base = pairs.filter(
            pl.col("mes").is_between(_shift(tau, -base_months), _shift(tau, -1))
            & (pl.col("pais") != US_NAME)
        )
        existing = (
            base.group_by("pais", "sh4")
            .agg(pl.col("fob").sum())
            .filter(pl.col("fob") >= EXISTING_MIN_USD)
            .select("pais", "sh4", pl.lit(True).alias("existente"))
        )
        after = pairs.filter(pl.col("mes") >= tau).join(existing, on=["pais", "sh4"], how="left")
        after = after.with_columns(
            pl.when(pl.col("pais") == US_NAME)
            .then(pl.lit("eua"))
            .when(pl.col("existente").fill_null(False))
            .then(pl.lit("existente"))
            .otherwise(pl.lit("novo"))
            .alias("destino_tipo")
        )
        margins.append(
            after.group_by("grupo", "mes", "destino_tipo")
            .agg(pl.col("fob").sum().alias("fob_usd"))
            .with_columns(pl.lit(ref).alias("referencia"))
        )
        follow = after.filter(
            (pl.col("destino_tipo") == "novo")
            & pl.col("mes").is_between(follow_start, _shift(follow_start, FOLLOW_MONTHS - 1))
        )
        new_pairs.append(
            follow.filter(pl.col("grupo") == "tarifado")
            .group_by("grupo", "pais", "sh4")
            .agg(pl.col("fob").sum().alias("fob_usd"))
            .filter(pl.col("fob_usd") >= 50_000)
            .with_columns(pl.lit(ref).alias("referencia"))
        )
    m = pl.concat(margins).select("referencia", "grupo", "mes", "destino_tipo", "fob_usd")
    n = pl.concat(new_pairs).select("referencia", "grupo", "pais", "sh4", "fob_usd")
    return (
        m.sort("referencia", "grupo", "mes", "destino_tipo"),
        n.sort("referencia", "grupo", "fob_usd", descending=[False, False, True]),
    )


# Same-length windows for the before/after comparisons of the redirection
# analysis: the seven months the 40% was collected (Aug 6, 2025 - Feb 23,
# 2026) and the same calendar months a year earlier.
WINDOWS = {
    "antes": (dt.date(2024, 8, 1), dt.date(2025, 2, 1)),
    "tarifaco": (dt.date(2025, 8, 1), dt.date(2026, 2, 1)),
}


def build_sh4_windows(pairs: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """Exports of each HS4 heading to the US and to everyone else, summed
    over each window of WINDOWS."""
    frames = []
    for name, (lo, hi) in WINDOWS.items():
        frames.append(
            pairs.filter(pl.col("mes").is_between(lo, hi))
            .with_columns(
                pl.when(pl.col("pais") == US_NAME)
                .then(pl.lit("eua"))
                .otherwise(pl.lit("outros"))
                .alias("destino")
            )
            .group_by("sh4", "destino")
            .agg(pl.col("fob").sum().alias("fob_usd"))
            .with_columns(pl.lit(name).alias("janela"))
        )
    return (
        pl.concat(frames)
        .join(groups.select("sh4", "grupo"), on="sh4", how="left")
        .with_columns(pl.col("grupo").fill_null("nao_exposto"))
        .select("sh4", "grupo", "destino", "janela", "fob_usd")
        .sort("sh4", "destino", "janela")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, name: str) -> None:
    path = OUT_DIR / name
    df.write_csv(path, float_precision=2)
    print(f"wrote {path.relative_to(HERE.parent)}: {df.height} rows")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    annex_path = OUT_DIR / "anexo_i_ordens_executivas.csv"
    if annex_path.exists():
        annex = pl.read_csv(annex_path)
    else:
        annex = parse_annexes()
        _write(annex, annex_path.name)

    sh6_names, sh4_names = fetch_comex_descriptions()
    us_hs6 = fetch_comex_us_hs6()
    pairs = fetch_comex_pairs()
    classes = build_classification(us_hs6, annex, sh6_names)
    groups = build_sh4_groups(pairs, us_hs6, classes, sh4_names)
    margins, new_pairs = build_margins(pairs, groups)

    _write(fetch_comtrade(), "comtrade_exportacoes.csv")
    _write(classes, "classificacao_sh6.csv")
    _write(build_us_by_category(us_hs6, classes), "comex_eua_categoria.csv")
    _write(build_us_exempt_sh4(us_hs6, classes), "comex_eua_isentos_sh4.csv")
    _write(groups, "sh4_grupos.csv")
    _write(margins, "comex_margens.csv")
    _write(new_pairs, "comex_novos_pares.csv")
    _write(build_sh4_windows(pairs, groups), "sh4_janelas.csv")


if __name__ == "__main__":
    main()
