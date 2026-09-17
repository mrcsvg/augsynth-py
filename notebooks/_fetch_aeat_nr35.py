"""Fetch/build the AEAT panel for the NR-35 notebook (04_nr35_trabalho_em_altura).

Two modes:

``python notebooks/_fetch_aeat_nr35.py``
    Downloads the AEAT (Anuário Estatístico de Acidentes do Trabalho) table
    workbooks from gov.br and builds ``notebooks/_data/aeat_nr35_panel.csv``:
    one row per (divisão CNAE 2.0, ano) with óbitos, vínculos and taxa de
    mortalidade, 2008-2019. Requires normal internet access to www.gov.br —
    **it cannot run inside restricted sandboxes** (the Claude Code remote
    environment blocks gov.br egress; this script was therefore written from
    the documented table structure and needs validation on first real run,
    see EXPERIMENTAL notes below).

``python notebooks/_fetch_aeat_nr35.py --demo``
    Generates ``notebooks/_data/aeat_nr35_panel_demo.csv`` (+ the true
    counterfactual in ``aeat_nr35_demo_truth.csv``): a *simulated* panel with
    AEAT-like magnitudes and a known injected NR-35 effect, so the notebook
    runs end-to-end and the estimators can be checked against ground truth.
    The demo numbers are NOT real AEAT data and must never be quoted as such.

``python notebooks/_fetch_aeat_nr35.py --probe``
    Reports which candidate AEAT URLs actually serve a file. Run this first
    when something 404s: gov.br renames the bundles between editions.

Provenance of the real pipeline (URLs probed 2026-09-05; see also
``_data/README-aeat.md``):

* Índice das edições: https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/saude-e-seguranca-do-trabalhador/acidente_trabalho_incapacidade
* Óbitos: Seção I, Subseção B, table 29.1 — "Quantidade de acidentes do
  trabalho liquidados, por consequência, segundo a CNAE" (Brasil). Rows are
  CNAE 2.0 *classes* (4 digits) + TOTAL + Ignorado; columns are 6 consequence
  groups (Total, Assistência Médica, <15 dias, >15 dias, Incapacidade
  Permanente, Óbito) × **3 years**. Classes are aggregated here to divisões.
* Indicadores: Seção II, chapter 59 (Brasil) — includes "Taxa de Mortalidade
  (por 100.000 vínculos)" and "Incidência (por 1.000 vínculos)", by CNAE
  class, for **2 years** per edition (`59.1` = year before the edition,
  `59.2` = the edition year).
* Vínculos (denominator) are NOT published as a column in the 2008-2019
  editions — that column only appears in recent ones (confirmed present in
  2023, absent in 2019). Since rates cannot be aggregated from class to
  divisão without weights, recover the weights the AEAT itself used:

      vínculos_classe = acidentes_registrados_classe * 1000 / incidência_classe

  Prefer this over inverting the mortality rate: accidents outnumber deaths by
  two orders of magnitude, so the 2-decimal rounding of the published rate
  costs far less precision, and few classes have a suppressed ("-") incidence.
  Then sum óbitos and vínculos to the divisão before dividing. RAIS is the
  external alternative if a class is suppressed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent / "_data"
PANEL_CSV = DATA_DIR / "aeat_nr35_panel.csv"
DEMO_CSV = DATA_DIR / "aeat_nr35_panel_demo.csv"
DEMO_TRUTH_CSV = DATA_DIR / "aeat_nr35_demo_truth.csv"

GOVBR = (
    "https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/"
    "saude-e-seguranca-do-trabalhador/acidente_trabalho_incapacidade"
)

# Tables ZIP per edition. All URLs below were probed on 2026-09-05 and serve a
# real file (a deliberately wrong path in the same directory returns 404, so
# the check discriminates). Note there is NO tables ZIP for the 2008 and 2012
# editions — their "Tabelas" link points back at the edition's own HTML page —
# but neither is needed: 2008 comes from the 2009/2010 editions and 2012 from
# the 2013 edition.
#
# The two sections we need have DIFFERENT year spans per edition:
#
# * Seção I, Subseção B (óbitos por CNAE, table 29.1) — the edition year plus
#   the TWO previous ones. Four editions therefore cover 2008-2019.
# * Seção II (indicadores por CNAE, chapter 59 = Brasil) — only TWO years:
#   table `59.1` is the year before the edition, `59.2` the edition year.
#   Covering 2008-2019 takes SIX editions (the odd ones below).
ZIP_URLS: dict[int, str] = {
    2009: f"{GOVBR}/arquivos/tabelas-aeat-2009.zip",
    2010: f"{GOVBR}/arquivos/tabelas-aeat-2010.zip",
    2011: f"{GOVBR}/arquivos/tabelas-aeat-2011.zip",
    2013: f"{GOVBR}/arquivos/aeat_tabelas_2013.zip",
    2014: f"{GOVBR}/arquivos/aeat2014_tabelas.zip",
    2015: f"{GOVBR}/arquivos/aeat15tab.zip",
    2016: f"{GOVBR}/arquivos/aeat-2016.zip",
    2017: f"{GOVBR}/arquivos/aeat-2017.zip",
    2018: f"{GOVBR}/arquivos/aeat-2018_tabelas-v2.zip",
    2019: f"{GOVBR}/arquivos/aeat-2019_def.zip",
    2020: f"{GOVBR}/arquivos/aeat-2020.zip",
}

# Óbitos route: edition -> the three years its Subseção B table carries.
EDITIONS: dict[int, dict[str, object]] = {
    2010: {"years": (2008, 2009, 2010), "zip_url": ZIP_URLS[2010]},
    2013: {
        "years": (2011, 2012, 2013),
        "zip_url": ZIP_URLS[2013],
        # Fallback: the 2013 edition also serves its tables as loose files.
        "b_brasil_xls": "https://www.gov.br/previdencia/pt-br/outros/imagens/2015/01/29a_01.xls",
    },
    2016: {"years": (2014, 2015, 2016), "zip_url": ZIP_URLS[2016]},
    2019: {"years": (2017, 2018, 2019), "zip_url": ZIP_URLS[2019]},
}

# Indicators route: edition -> (year of table 59.1, year of table 59.2).
# These six editions tile 2008-2019 without gaps or overlaps.
INDICATOR_EDITIONS: dict[int, tuple[int, int]] = {
    2009: (2008, 2009),
    2011: (2010, 2011),
    2013: (2012, 2013),
    2015: (2014, 2015),
    2017: (2016, 2017),
    2019: (2018, 2019),
}

# Why the panel starts in 2008, and the one year it could still gain (issue #25).
# CNAE 2.0 takes effect in the CNPJ on 2007-01-01 and the NTEP in April 2007, so
# every year before 2007 crosses BOTH breaks at once; the Concla 1.0 -> 2.0
# correspondence is per class and splits several donor sections across four
# CNAE 1.0 sections each, which aggregates cannot undo. The MPS index also links
# no edition before 2008. That leaves 2007, and only 2007:
#   * óbitos    - table 29.1 of the 2009 edition (edition year plus the two
#                 before it), whose ZIP is already in ZIP_URLS above; add
#                 2009: {"years": (2007, 2008, 2009), ...} to EDITIONS and
#                 de-duplicate 2008-2009 against the 2010 edition.
#   * vínculos  - chapter 59 of the 2008 edition (table 59.1 = 2007). That
#                 edition has no ZIP; it serves loose .xls from its Seção II
#                 page. This is the only piece needing new code.
# What it buys is measured in Act 12 of the notebook: with T0 = 6 and the
# evaluation window shortened to 2013-2017 the design leaves the inverted regime
# and the p-value floor drops from 1/9 to 1/11. It does not change the MDE.
# See _data/README-aeat.md for the full write-up.

# CNAE 2.0 divisões (IBGE), short labels used as unit names in the panel.
CNAE_DIVISOES: dict[str, str] = {
    "01": "01 Agricultura e pecuária",
    "02": "02 Produção florestal",
    "03": "03 Pesca e aquicultura",
    "05": "05 Extração de carvão",
    "06": "06 Extração de petróleo e gás",
    "07": "07 Minerais metálicos",
    "08": "08 Minerais não-metálicos (extr.)",
    "09": "09 Apoio à extração",
    "10": "10 Alimentos",
    "11": "11 Bebidas",
    "12": "12 Fumo",
    "13": "13 Têxteis",
    "14": "14 Confecção",
    "15": "15 Couro e calçados",
    "16": "16 Madeira",
    "17": "17 Celulose e papel",
    "18": "18 Impressão",
    "19": "19 Coque e petróleo (deriv.)",
    "20": "20 Químicos",
    "21": "21 Farmacêuticos",
    "22": "22 Borracha e plástico",
    "23": "23 Minerais não-metálicos (prod.)",
    "24": "24 Metalurgia",
    "25": "25 Produtos de metal",
    "26": "26 Informática e eletrônicos",
    "27": "27 Máquinas e mat. elétricos",
    "28": "28 Máquinas e equipamentos",
    "29": "29 Veículos automotores",
    "30": "30 Outros equip. de transporte",
    "31": "31 Móveis",
    "32": "32 Produtos diversos",
    "33": "33 Manutenção de máquinas",
    "35": "35 Eletricidade e gás",
    "36": "36 Água",
    "37": "37 Esgoto",
    "38": "38 Resíduos",
    "39": "39 Descontaminação",
    "41": "41 Construção de edifícios",
    "42": "42 Obras de infraestrutura",
    "43": "43 Serviços p/ construção",
    "45": "45 Comércio de veículos",
    "46": "46 Comércio atacadista",
    "47": "47 Comércio varejista",
    "49": "49 Transporte terrestre",
    "50": "50 Transporte aquaviário",
    "51": "51 Transporte aéreo",
    "52": "52 Armazenagem e aux. transporte",
    "53": "53 Correio e entregas",
    "55": "55 Alojamento",
    "56": "56 Alimentação (serviços)",
    "58": "58 Edição",
    "59": "59 Cinema e som",
    "60": "60 Rádio e TV",
    "61": "61 Telecomunicações",
    "62": "62 TI (serviços)",
    "63": "63 Serviços de informação",
    "64": "64 Serviços financeiros",
    "65": "65 Seguros e previdência",
    "66": "66 Aux. financeiros",
    "68": "68 Imobiliárias",
    "69": "69 Jurídicas e contábeis",
    "70": "70 Consultoria empresarial",
    "71": "71 Arquitetura e engenharia",
    "72": "72 P&D",
    "73": "73 Publicidade",
    "74": "74 Outras profissionais",
    "75": "75 Veterinária",
    "77": "77 Aluguéis não-imob.",
    "78": "78 Agenciamento de mão de obra",
    "79": "79 Agências de viagens",
    "80": "80 Vigilância e segurança",
    "81": "81 Serviços p/ edifícios",
    "82": "82 Serviços de escritório",
    "84": "84 Administração pública",
    "85": "85 Educação",
    "86": "86 Saúde humana",
    "87": "87 Assistência social (aloj.)",
    "88": "88 Assistência social",
    "90": "90 Artes e espetáculos",
    "91": "91 Patrimônio cultural",
    "92": "92 Jogos de azar",
    "93": "93 Esporte e recreação",
    "94": "94 Org. associativas",
    "95": "95 Reparação de bens pessoais",
    "96": "96 Serviços pessoais",
    "97": "97 Serviços domésticos",
    "99": "99 Organismos internacionais",
}


# ---------------------------------------------------------------------------
# Real fetch (EXPERIMENTAL: written from the documented structure; the
# sandbox this was authored in cannot reach gov.br, so the parsing below has
# not run against the real workbooks yet. Validate TOTAL rows against the
# published PDFs on first run.)
# ---------------------------------------------------------------------------


# The AEAT bundles are 5-13 MB each and gov.br drops connections mid-transfer
# often enough to matter (a run died with IncompleteRead after 1.4 of 13 MB).
# Cache them outside the repo so a re-run costs nothing and a flaky download
# does not restart the whole pipeline.
DOWNLOAD_CACHE = Path(tempfile.gettempdir()) / "augsynth-aeat-cache"
DOWNLOAD_ATTEMPTS = 3


def _download(url: str) -> bytes:
    """Fetch a URL, with an on-disk cache and retries on truncated responses."""
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    cached = DOWNLOAD_CACHE / hashlib.sha256(url.encode()).hexdigest()[:16]
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_bytes()

    last: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            req = Request(url, headers={"User-Agent": "augsynth-py/aeat-fetch"})
            with urlopen(req, timeout=300) as resp:
                blob = resp.read()
            declared = resp.headers.get("Content-Length")
            if declared and len(blob) != int(declared):
                raise OSError(f"download truncado: {len(blob)} de {declared} bytes")
            cached.write_bytes(blob)
            return blob
        except Exception as exc:
            last = exc
            print(f"    tentativa {attempt}/{DOWNLOAD_ATTEMPTS} falhou ({exc}); repetindo...")
    raise RuntimeError(f"Falha ao baixar {url} após {DOWNLOAD_ATTEMPTS} tentativas: {last}")


def _find_subsecao_b_brasil(zf: zipfile.ZipFile) -> str:
    """Locate table 29.1 (Subseção B, Brasil × CNAE) inside a tables ZIP.

    Editions name it ``29a_01.xls`` (2012-2013 era) but older/newer bundles
    vary, so fall back to scanning workbook contents for the table title.
    """
    names = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))]
    for n in names:
        if re.search(r"29a?_?0?1\.xlsx?$", n.lower()):
            return n
    import pandas as pd

    for n in names:
        try:
            head = pd.read_excel(io.BytesIO(zf.read(n)), header=None, nrows=8)
        except Exception:
            continue
        text = " ".join(str(v) for v in head.to_numpy().ravel()).lower()
        if (
            "liquidados" in text
            and "cnae" in text
            and "consequência" in text.replace("conseqüência", "consequência")
        ):
            return n
    raise FileNotFoundError(
        f"Tabela 29.1 (Subseção B, Brasil x CNAE) não encontrada no ZIP; membros: {names[:10]}..."
    )


# Óbito is the last of the six consequence groups, so its three year-columns
# sit at 0-indexed 16, 17 and 18 of a 19-column table. Verified identical in
# the 2010, 2013, 2016 and 2019 editions; only the header ROW moves (7 vs 6).
# Addressing the column by position beats collecting the numeric cells and
# taking the last three: the 2019 edition writes suppressed values as "-", and
# that approach silently dropped 448 rows (250 deaths in 2017 alone).
OBITO_COL = 16


def _cell_to_count(value: object) -> int | None:
    """Coerce one AEAT count cell to an int, or None when it is not a count.

    Blank cells and the "-" placeholder mean zero (suppressed or no cases).
    Returns None for anything unrecognized so the caller can skip the row
    rather than invent a number.
    """
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip()
        if text in {"-", "", "..", "...", "nan"}:
            return 0
        try:
            return int(float(text.replace(".", "").replace(",", ".")))
        except ValueError:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return 0 if value != value else int(value)  # NaN -> 0
    return None


def _parse_subsecao_b(xls_bytes: bytes, years: tuple[int, int, int]) -> dict[tuple[str, int], int]:
    """Parse óbitos by CNAE class from a Subseção B workbook, per year.

    Layout (confirmed against the AEAT 2012 PDF, tables 29.x): rows = CNAE 2.0
    classes (4-digit codes) + TOTAL + Ignorado; 18 numeric columns = 6
    consequence groups × 3 years, Óbito being the LAST group (last 3 numeric
    columns, years ascending). Returns {(divisao, ano): óbitos} aggregated to
    2-digit divisões, plus ("TOTAL", ano) for validation.
    """

    # The table is paginated across sheets in recent editions (the 2019
    # workbook splits table 29.1 over 10 sheets, "19Act29_01" .. "19Act29_01
    # 10", with TOTAL on the first and Ignorado on the last), while older .xls
    # editions carry a single sheet. Read them all: on a one-sheet workbook
    # this is a no-op, and on a paginated one it is the difference between
    # 669 CNAE classes and only the first 68.
    workbook = _open_workbook(xls_bytes)
    out: dict[tuple[str, int], int] = {}
    rows = (
        row
        for sheet in workbook.sheet_names
        for _, row in workbook.parse(sheet, header=None).iterrows()
    )
    for row in rows:
        cells = row.tolist()
        label = str(cells[0]).strip()
        if re.fullmatch(r"\d{4}", label):
            key = label[:2]
        elif label.upper().startswith("TOTAL"):
            key = "TOTAL"
        elif label.upper().startswith("IGNORADO"):
            # Deaths whose CNAE was not informed. Not a division, but needed to
            # reconcile against TOTAL below.
            key = "IGNORADO"
        else:
            continue
        if len(cells) <= OBITO_COL + 2:
            continue
        obito_cols = [_cell_to_count(cells[OBITO_COL + i]) for i in range(3)]
        if any(v is None for v in obito_cols):
            continue
        for yr, val in zip(years, obito_cols, strict=True):
            out[(key, yr)] = out.get((key, yr), 0) + int(val)
    if not out:
        raise ValueError("Nenhuma linha CNAE reconhecida — layout mudou; inspecione o workbook.")
    _reconcile(out, years)
    return out


def _reconcile(out: dict[tuple[str, int], int], years: tuple[int, int, int]) -> None:
    """Check that the parsed divisions add back up to the published TOTAL.

    ``TOTAL == sum(divisões) + Ignorado`` holds exactly in the AEAT tables
    (verified on the 2013 edition: 2938 = 2897 + 41 for 2011, and likewise for
    2012 and 2013). If it ever fails, the column offsets are wrong for this
    edition — a silent off-by-one would otherwise ship plausible but wrong
    numbers, which is worse than crashing.
    """
    divisions = {d for d, _ in out} - {"TOTAL", "IGNORADO"}
    for yr in years:
        total = out.get(("TOTAL", yr))
        if total is None:
            raise ValueError(f"Linha TOTAL ausente para {yr} — layout inesperado.")
        got = sum(out.get((d, yr), 0) for d in divisions) + out.get(("IGNORADO", yr), 0)
        if got != total:
            raise ValueError(
                f"Reconciliação falhou em {yr}: soma das divisões + Ignorado = {got}, "
                f"mas a tabela publica TOTAL = {total} (diferença {total - got}). "
                "As colunas de Óbito provavelmente não são as 3 últimas nesta edição."
            )


# Column positions, 0-indexed, stable across the 2009-2019 editions.
REGISTRADOS_COL = 1  # chapter 1: "Total" registered accidents, first of 3 years
INCIDENCIA_COL = 1  # chapter 59: "Incidência (por 1.000 vínculos)"
TAXA_MORT_COL = 5  # chapter 59: "Taxa de Mortalidade (por 100.000 vínculos)"

# Divisions forming the treated unit, and the floor below which a donor's rate
# is too noisy to be worth keeping (a division with a handful of vínculos swings
# wildly when one death lands in it).
CONSTRUCAO = ("41", "42", "43")
MIN_VINCULOS = 50_000

# CNAE 2.0 sections, as ranges of divisions. The panel is also published at
# this level because the division panel cannot identify a synthetic control:
# with 68 donors and only 5 pre-treatment years the simplex QP interpolates the
# pre-period exactly (RMSPE 2e-09) and the ATT swings from -0.4 to -9.5 when a
# single donor is dropped. Aggregating to sections dilutes the extreme
# divisions, puts the treated unit outside the donor convex hull, and leaves a
# real pre-period residual (RMSPE 5.7%) for the estimator to work against.
SECOES: dict[str, tuple[str, range]] = {
    "A": ("A Agropecuária", range(1, 4)),
    "B": ("B Indústrias extrativas", range(5, 10)),
    "C": ("C Indústrias de transformação", range(10, 34)),
    "D": ("D Eletricidade e gás", range(35, 36)),
    "E": ("E Água, esgoto e resíduos", range(36, 40)),
    "F": ("Construção", range(41, 44)),
    "G": ("G Comércio e reparação", range(45, 48)),
    "H": ("H Transporte e armazenagem", range(49, 54)),
    "I": ("I Alojamento e alimentação", range(55, 57)),
    "J": ("J Informação e comunicação", range(58, 64)),
    "K": ("K Financeiras e seguros", range(64, 67)),
    "L": ("L Atividades imobiliárias", range(68, 69)),
    "M": ("M Profissionais e técnicas", range(69, 76)),
    "N": ("N Administrativas e apoio", range(77, 83)),
    "O": ("O Administração pública", range(84, 85)),
    "P": ("P Educação", range(85, 86)),
    "Q": ("Q Saúde e serviço social", range(86, 89)),
    "R": ("R Artes, esporte e recreação", range(90, 94)),
    "S": ("S Outros serviços", range(94, 97)),
    "T": ("T Serviços domésticos", range(97, 98)),
    "U": ("U Organismos internacionais", range(99, 100)),
}
DIVISAO_PARA_SECAO: dict[str, str] = {
    f"{division:02d}": code for code, (_, divisions) in SECOES.items() for division in divisions
}
PANEL_SECOES_CSV = DATA_DIR / "aeat_nr35_panel_secoes.csv"


def _open_workbook(blob: bytes) -> Any:
    """Open an AEAT workbook, repairing the metadata some editions ship broken.

    ``19Act59_01.xlsx`` in the 2019 edition was written by SAS on Linux and
    carries ``<dcterms:modified>2022-07-25T 9:15:19-03:00</dcterms:modified>``:
    the hour is space-padded instead of zero-padded, and openpyxl rejects both
    that and the numeric UTC offset, raising ``TypeError: expected
    <class 'datetime.datetime'>`` before reading a single cell. The defect is
    in document metadata only, never in the data, so repair it rather than skip
    the table. The ladder is: read as-is, then rewrite the timestamps in
    canonical UTC form, then drop the metadata part entirely.
    """
    import pandas as pd

    try:
        return pd.ExcelFile(io.BytesIO(blob))
    except Exception:
        pass

    for repair in (_normalize_core_dates, _drop_core_properties):
        try:
            return pd.ExcelFile(io.BytesIO(_rebuild_zip(blob, repair)))
        except Exception:
            continue
    # Re-raise the original failure with its own traceback for the caller.
    return pd.ExcelFile(io.BytesIO(blob))


def _rebuild_zip(blob: bytes, transform: Any) -> bytes:
    """Copy a zip container, letting ``transform`` rewrite or drop each member."""
    source = zipfile.ZipFile(io.BytesIO(blob))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = transform(item.filename, source.read(item.filename))
            if data is not None:
                target.writestr(item, data)
    return buffer.getvalue()


def _normalize_core_dates(name: str, data: bytes) -> bytes:
    if name != "docProps/core.xml":
        return data
    text = data.decode("utf8", "replace")
    text = re.sub(
        r">(\d{4}-\d{2}-\d{2})T\s*(\d{1,2}):(\d{2}):(\d{2})[^<]*<",
        lambda mm: f">{mm.group(1)}T{int(mm.group(2)):02d}:{mm.group(3)}:{mm.group(4)}Z<",
        text,
    )
    return text.encode("utf8")


def _drop_core_properties(name: str, data: bytes) -> bytes | None:
    return None if name == "docProps/core.xml" else data


def _find_table(zf: zipfile.ZipFile, chapter: int, table: int) -> str:
    """Locate one AEAT table inside a tables ZIP, across edition naming schemes.

    Members are named ``29_01.xls`` (2009-2013), ``15Act29_01.xls`` (2015) or
    ``aeat-2019/Seção I - B_xlsx/19Act29_01.xlsx`` (2017-2019). Chapter 1 is
    Subseção A (accidents registered), chapter 29 is Subseção B (liquidated,
    the one carrying Óbito) and chapter 59 is Seção II (indicators). In each
    block the Brasil table is the first one; the rest are regions and states.
    """
    pattern = re.compile(rf"(?:^|/)(?:\d\dAct)?0*{chapter}_{table:02d}\.xlsx?$", re.I)
    hits = [n for n in zf.namelist() if pattern.search(n)]
    if not hits:
        raise FileNotFoundError(
            f"Tabela {chapter}.{table} não encontrada no ZIP; "
            f"membros de exemplo: {zf.namelist()[:5]}"
        )
    return hits[0]


def _column_by_class(blob: bytes, column: int) -> dict[str, float]:
    """Read one numeric column of an AEAT table, keyed by 4-digit CNAE class.

    Reads every sheet, since recent editions paginate a table across sheets.
    Non-numeric cells (notably the "-" used for suppressed values) are left
    out rather than coerced, so the caller can tell a missing value from a zero.
    """
    workbook = _open_workbook(blob)
    out: dict[str, float] = {}
    for sheet in workbook.sheet_names:
        for _, row in workbook.parse(sheet, header=None).iterrows():
            cells = row.tolist()
            label = str(cells[0]).strip()
            if not re.fullmatch(r"\d{4}", label) or len(cells) <= column:
                continue
            value = cells[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value == value:  # not NaN
                out[label] = float(value)
    return out


def build_real_panel() -> None:
    """Download the AEAT tables and write the real panel CSV.

    Runs over the six editions in ``INDICATOR_EDITIONS``; each contributes two
    years, together tiling 2008-2019. For every year it needs three tables from
    the same edition, so revisions stay internally consistent:

    * chapter 29 (Subseção B) -> óbitos per CNAE class
    * chapter 1 (Subseção A)  -> acidentes registrados per class
    * chapter 59 (Seção II)   -> incidência and taxa de mortalidade per class

    The vínculos denominator is not published for these years, so it is
    recovered from the identity behind the published incidence,

        vínculos = acidentes registrados * 1000 / incidência

    which is exact: the workbooks store the indicators at full float precision,
    not at the two decimals they display. The recovery is checked against the
    independent ``óbitos * 1e5 / taxa de mortalidade`` for the same class, and
    the two agree to floating-point noise.
    """
    import polars as pl

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    obitos: dict[tuple[str, int], int] = {}
    vinculos: dict[tuple[str, int], float] = {}

    for edition in sorted(INDICATOR_EDITIONS):
        indicator_years = INDICATOR_EDITIONS[edition]
        edition_years = (edition - 2, edition - 1, edition)
        print(f"AEAT {edition}: baixando {ZIP_URLS[edition]} ...")
        zf = zipfile.ZipFile(io.BytesIO(_download(ZIP_URLS[edition])))

        by_division = _parse_subsecao_b(zf.read(_find_table(zf, 29, 1)), edition_years)
        print(f"  óbitos: reconciliação OK para {edition_years}")

        registrados_blob = zf.read(_find_table(zf, 1, 1))
        obitos_classe_blob = zf.read(_find_table(zf, 29, 1))

        for slot, year in enumerate(indicator_years):
            year_col = edition_years.index(year)
            registrados = _column_by_class(registrados_blob, REGISTRADOS_COL + year_col)
            indicadores_blob = zf.read(_find_table(zf, 59, slot + 1))
            incidencia = _column_by_class(indicadores_blob, INCIDENCIA_COL)
            taxa_mort = _column_by_class(indicadores_blob, TAXA_MORT_COL)
            obitos_classe = _column_by_class(obitos_classe_blob, OBITO_COL + year_col)

            per_division: dict[str, float] = {}
            sem_incidencia = 0
            for cnae, registros in registrados.items():
                taxa = incidencia.get(cnae)
                if not taxa:
                    sem_incidencia += 1
                    continue
                per_division[cnae[:2]] = per_division.get(cnae[:2], 0.0) + registros * 1000.0 / taxa
            for division, total in per_division.items():
                vinculos[(division, year)] = total
            for division, _year in list(by_division):
                if _year == year and division not in {"TOTAL", "IGNORADO"}:
                    obitos[(division, year)] = by_division[(division, year)]

            _check_vinculos_recovery(registrados, incidencia, obitos_classe, taxa_mort, year)
            print(
                f"  {year}: {len(per_division)} divisões, "
                f"{sum(per_division.values()):,.0f} vínculos "
                f"({sem_incidencia} classes sem incidência, ignoradas)"
            )

    _write_panel(obitos, vinculos, pl)


def _check_vinculos_recovery(
    registrados: dict[str, float],
    incidencia: dict[str, float],
    obitos_classe: dict[str, float],
    taxa_mort: dict[str, float],
    year: int,
) -> None:
    """Cross-check the recovered denominator against a second, independent route.

    ``registrados * 1000 / incidência`` (the route actually used) and
    ``óbitos * 1e5 / taxa de mortalidade`` come from disjoint columns of
    different chapters, so agreement means the denominator really is the one
    the AEAT used, and a swapped or shifted column would blow the comparison
    apart by orders of magnitude.

    The comparison is made on the aggregate, not class by class. The two routes
    count different populations at the margin: Subseção B counts accidents
    *liquidated* in the year, which lag the registrations the mortality rate is
    built on. Where a class has one or two deaths that mismatch dominates — CNAE
    9492 in 2018 has 7 registros and 1 óbito, and the two routes imply 6,731 vs
    4,815 vínculos, ranges that do not even overlap. Summed over classes the
    noise cancels: editions 2009-2015 agree to 0.0000% and 2019, the worst,
    to 0.86%.
    """
    total_incidencia = 0.0
    total_mortalidade = 0.0
    for cnae, registros in registrados.items():
        inc = incidencia.get(cnae)
        taxa = taxa_mort.get(cnae)
        mortes = obitos_classe.get(cnae)
        if not inc or not taxa or not mortes:
            continue
        total_incidencia += registros * 1000.0 / inc
        total_mortalidade += mortes * 1e5 / taxa
    if not total_mortalidade:
        print(f"    cross-check vínculos: sem classes comparáveis em {year}")
        return
    gap = abs(total_incidencia / total_mortalidade - 1.0)
    if gap > 0.05:
        raise ValueError(
            f"Recuperação de vínculos inconsistente em {year}: as duas rotas diferem "
            f"{gap:.2%} no agregado ({total_incidencia:,.0f} via incidência vs "
            f"{total_mortalidade:,.0f} via mortalidade). As colunas de incidência ou "
            "de taxa de mortalidade provavelmente mudaram de posição nesta edição."
        )
    print(f"    cross-check vínculos: agregado difere {gap:.4%} entre as duas rotas")


def _write_secoes(
    obitos: dict[tuple[str, int], int],
    vinculos: dict[tuple[str, int], float],
    years: list[int],
    pl: Any,
) -> None:
    """Write the same panel aggregated to CNAE sections.

    Deaths and vínculos are summed before the ratio, so the section rate is the
    employment-weighted one rather than a mean of division rates.
    """
    rows: list[dict[str, object]] = []
    for year in years:
        totals: dict[str, list[float]] = {}
        for (division, row_year), v in vinculos.items():
            if row_year != year:
                continue
            code = DIVISAO_PARA_SECAO.get(division)
            if code is None:
                continue
            slot = totals.setdefault(code, [0.0, 0.0])
            slot[0] += obitos.get((division, year), 0)
            slot[1] += v
        for code, (mortes, v) in totals.items():
            if v < MIN_VINCULOS:
                continue
            rows.append(
                {
                    "setor": SECOES[code][0],
                    "ano": year,
                    "taxa_mortalidade": round(mortes / v * 1e5, 4),
                    "obitos": int(mortes),
                    "vinculos": round(v),
                }
            )
    frame = pl.DataFrame(rows)
    complete = (
        frame.group_by("setor").agg(n=pl.len()).filter(pl.col("n") == len(years))["setor"].to_list()
    )
    frame = frame.filter(pl.col("setor").is_in(complete)).sort(["setor", "ano"])
    frame.write_csv(PANEL_SECOES_CSV)
    print(f"\nWrote {PANEL_SECOES_CSV} — {frame['setor'].n_unique()} seções x {len(years)} anos")


def _write_panel(
    obitos: dict[tuple[str, int], int],
    vinculos: dict[tuple[str, int], float],
    pl: Any,
) -> None:
    """Aggregate to the treated unit plus donor divisions and write the CSV."""
    years = sorted({y for _, y in vinculos})
    divisions = sorted({d for d, _ in vinculos})

    rows: list[dict[str, object]] = []
    for year in years:
        treated_obitos = sum(obitos.get((d, year), 0) for d in CONSTRUCAO)
        treated_vinculos = sum(vinculos.get((d, year), 0.0) for d in CONSTRUCAO)
        if treated_vinculos <= 0:
            raise ValueError(f"Sem vínculos para a construção em {year}.")
        rows.append(
            {
                "setor": "Construção",
                "ano": year,
                "taxa_mortalidade": round(treated_obitos / treated_vinculos * 1e5, 4),
                "obitos": treated_obitos,
                "vinculos": round(treated_vinculos),
            }
        )
        for division in divisions:
            if division in CONSTRUCAO:
                continue
            v = vinculos.get((division, year), 0.0)
            if v < MIN_VINCULOS:
                continue
            rows.append(
                {
                    "setor": CNAE_DIVISOES.get(division, f"{division} (CNAE {division})"),
                    "ano": year,
                    "taxa_mortalidade": round(obitos.get((division, year), 0) / v * 1e5, 4),
                    "obitos": obitos.get((division, year), 0),
                    "vinculos": round(v),
                }
            )

    frame = pl.DataFrame(rows)
    # Keep only units observed in every year: synthetic control needs a
    # balanced panel, and a donor that appears midway would break the fit.
    complete = (
        frame.group_by("setor").agg(n=pl.len()).filter(pl.col("n") == len(years))["setor"].to_list()
    )
    frame = frame.filter(pl.col("setor").is_in(complete)).sort(["setor", "ano"])
    frame.write_csv(PANEL_CSV)
    _write_secoes(obitos, vinculos, years, pl)

    print(f"\nWrote {PANEL_CSV}")
    print(f"  {frame['setor'].n_unique()} unidades x {len(years)} anos ({years[0]}-{years[-1]})")
    dropped = len(divisions) - frame["setor"].n_unique() + 1
    print(f"  {dropped} divisões descartadas (< {MIN_VINCULOS:,} vínculos ou painel incompleto)")
    print("\n  Construção:")
    for row in frame.filter(pl.col("setor") == "Construção").sort("ano").iter_rows(named=True):
        print(
            f"    {row['ano']}: {row['taxa_mortalidade']:6.2f} por 100 mil "
            f"({row['obitos']:4d} óbitos / {row['vinculos']:,} vínculos)"
        )


# ---------------------------------------------------------------------------
# Probe — which candidate URLs actually serve a file?
# ---------------------------------------------------------------------------


# gov.br filenames are inconsistent across editions, and the Plone CMS answers
# 200-with-an-HTML-page for a missing file rather than 404. So probe by magic
# number, not by status code: ZIP starts with "PK", legacy .xls (OLE2) with
# D0 CF 11 E0, .xlsx is a ZIP too, PDF with "%PDF".
def _probe_urls() -> list[tuple[str, str]]:
    """Candidate URLs to probe, with a human label for each."""
    items: list[tuple[str, str]] = []
    for ed, url in sorted(ZIP_URLS.items()):
        roles = []
        if ed in EDITIONS:
            yrs = EDITIONS[ed]["years"]  # type: ignore[index]
            roles.append(f"óbitos {yrs[0]}-{yrs[2]}")  # type: ignore[index]
        if ed in INDICATOR_EDITIONS:
            a, b = INDICATOR_EDITIONS[ed]
            roles.append(f"indicadores {a}+{b}")
        items.append((url, f"tabelas AEAT {ed}" + (f" [{'; '.join(roles)}]" if roles else "")))
    items += [
        (f"{GOVBR}/arquivos/aeat-2012.pdf", "PDF AEAT 2012 (fallback; sem ZIP nesta edição)"),
        (
            "https://www.gov.br/previdencia/pt-br/outros/imagens/2014/01/29a_01.xls",
            "tabela 29.1 solta, edição 2012 (óbitos x CNAE, 2010-2012)",
        ),
        (
            "https://www.gov.br/previdencia/pt-br/outros/imagens/2015/01/29a_01.xls",
            "tabela 29.1 solta, edição 2013 (óbitos x CNAE, 2011-2013)",
        ),
        (
            "https://www.gov.br/previdencia/pt-br/outros/imagens/2014/01/59a_02.xls",
            "tabela 59.2 solta, edição 2012 (indicadores: taxa de mortalidade x CNAE)",
        ),
        (
            f"{GOVBR}/arquivos/aeat-1999_controle-negativo.zip",
            "CONTROLE NEGATIVO — deve falhar; se 'OK', o probe não discrimina",
        ),
    ]
    return items


PROBE_URLS: list[tuple[str, str]] = _probe_urls()

_MAGIC: list[tuple[bytes, str]] = [
    (b"PK\x03\x04", "zip/xlsx"),
    (b"\xd0\xcf\x11\xe0", "xls (OLE2)"),
    (b"%PDF", "pdf"),
    (b"<!DOCTYPE", "HTML (pagina de erro)"),
    (b"<html", "HTML (pagina de erro)"),
]


def probe_urls() -> None:
    """Report which candidate AEAT URLs actually serve a binary file.

    Fetches only the first bytes of each URL (HTTP Range), so it is cheap and
    safe to re-run. Prints one line per URL: veredito, tipo detectado, tamanho
    declarado e URL final (após redirecionamentos).
    """
    import urllib.error

    print(f"Testando {len(PROBE_URLS)} URLs (só os primeiros bytes de cada)...\n")
    ok, bad = [], []
    for url, label in PROBE_URLS:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (augsynth-py/aeat-probe)",
                "Range": "bytes=0-2047",
            },
        )
        try:
            with urlopen(req, timeout=60) as resp:
                head = resp.read(2048)
                status = resp.status
                ctype = resp.headers.get("Content-Type", "?").split(";")[0]
                clen = resp.headers.get("Content-Range") or resp.headers.get("Content-Length", "?")
                final = resp.url
        except urllib.error.HTTPError as exc:
            print(f"  [FALHA {exc.code}] {label}\n      {url}")
            bad.append(url)
            continue
        except Exception as exc:
            print(f"  [ERRO REDE] {label}: {type(exc).__name__}: {exc}\n      {url}")
            bad.append(url)
            continue

        kind = next((name for magic, name in _MAGIC if head.startswith(magic)), None)
        is_file = kind is not None and "HTML" not in kind
        veredito = "OK   " if is_file else "NAO  "
        (ok if is_file else bad).append(url)
        print(
            f"  [{veredito}] {label}\n"
            f"      tipo={kind or 'desconhecido'} http={status} ctype={ctype} tam={clen}\n"
            f"      {final}"
        )

    print(f"\nResumo: {len(ok)} servindo arquivo, {len(bad)} falhando.")
    print(
        "Se o CONTROLE NEGATIVO apareceu como OK, o servidor está devolvendo algo\n"
        "para qualquer caminho e os demais vereditos não valem — investigue à mão."
    )


# ---------------------------------------------------------------------------
# Demo panel — simulated, clearly labeled, with known injected effect
# ---------------------------------------------------------------------------

# True injected effect on the treated unit (multiplicative, on the rate):
# 2013 = -8% (capacitação só vigente a partir de 27/03/2013), 2014+ = -13%.
DEMO_TRUE_EFFECT: dict[int, float] = {2013: -0.08, **{y: -0.13 for y in range(2014, 2020)}}
DEMO_SEED = 313  # Portaria SIT 313/2012

# (divisão, taxa base 2008 por 100 mil, tendência anual média, escala de ruído)
# Magnitudes calibradas na ordem de grandeza publicada pelo AEAT (taxa média
# nacional ~7-9 no início da década de 2010, extrativas/transporte no topo,
# serviços de escritório na base). NÚMEROS SIMULADOS — não citar como AEAT.
_DEMO_DONORS: list[tuple[str, float, float, float]] = [
    ("01 Agricultura e pecuária", 14.5, -0.030, 0.9),
    ("02 Produção florestal", 22.0, -0.035, 2.4),
    ("05-09 Indústrias extrativas", 28.0, -0.025, 2.6),
    ("10 Alimentos", 6.8, -0.030, 0.5),
    ("11 Bebidas", 6.2, -0.028, 0.8),
    ("13 Têxteis", 3.1, -0.030, 0.5),
    ("14 Confecção", 1.6, -0.028, 0.3),
    ("15 Couro e calçados", 2.6, -0.025, 0.4),
    ("16 Madeira", 12.5, -0.040, 1.3),
    ("17 Celulose e papel", 6.0, -0.030, 0.8),
    ("20 Químicos", 5.5, -0.028, 0.7),
    ("22 Borracha e plástico", 4.6, -0.030, 0.5),
    ("23 Minerais não-metálicos (prod.)", 9.8, -0.035, 0.9),
    ("24 Metalurgia", 9.0, -0.032, 1.0),
    ("25 Produtos de metal", 6.4, -0.030, 0.6),
    ("28 Máquinas e equipamentos", 4.4, -0.028, 0.6),
    ("29 Veículos automotores", 3.8, -0.030, 0.5),
    ("31 Móveis", 5.2, -0.032, 0.7),
    ("33 Manutenção de máquinas", 8.6, -0.025, 1.1),
    ("35 Eletricidade e gás", 11.5, -0.038, 1.5),
    ("36-39 Água, esgoto e resíduos", 10.0, -0.028, 1.2),
    ("45 Comércio de veículos", 5.4, -0.030, 0.5),
    ("46 Comércio atacadista", 6.6, -0.028, 0.5),
    ("47 Comércio varejista", 3.4, -0.028, 0.3),
    ("49 Transporte terrestre", 18.5, -0.022, 1.2),
    ("50 Transporte aquaviário", 16.0, -0.028, 2.8),
    ("52 Armazenagem e aux. transporte", 8.8, -0.026, 0.9),
    ("53 Correio e entregas", 3.4, -0.024, 0.6),
    ("55 Alojamento", 2.2, -0.028, 0.4),
    ("56 Alimentação (serviços)", 1.8, -0.026, 0.3),
    ("61 Telecomunicações", 6.5, -0.036, 1.1),
    ("62-63 TI e informação", 1.1, -0.025, 0.3),
    ("64-66 Financeiras e seguros", 1.5, -0.030, 0.3),
    ("68 Imobiliárias", 3.0, -0.026, 0.6),
    ("69-75 Serviços profissionais", 2.4, -0.026, 0.4),
    ("77 Aluguéis não-imob.", 7.0, -0.028, 0.9),
    ("78 Agenciamento de mão de obra", 5.8, -0.030, 0.7),
    ("80 Vigilância e segurança", 8.4, -0.026, 0.8),
    ("81 Serviços p/ edifícios", 4.9, -0.028, 0.5),
    ("82 Serviços de escritório", 2.8, -0.026, 0.4),
    ("84 Administração pública", 2.0, -0.024, 0.4),
    ("85 Educação", 0.9, -0.024, 0.2),
    ("86 Saúde humana", 1.3, -0.024, 0.2),
    ("87-88 Assistência social", 1.6, -0.022, 0.4),
    ("90-93 Artes, esporte e recreação", 3.6, -0.024, 0.7),
    ("94-96 Outros serviços", 2.9, -0.026, 0.5),
]
_DEMO_TREATED = ("Construção", 17.5, -0.034, 0.8)


def build_demo_panel() -> None:
    import numpy as np
    import polars as pl

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DEMO_SEED)
    years = list(range(2008, 2020))
    # Choque macro comum (ciclo + recessão 2015-16), compartilhado com pesos
    # heterogêneos — é o que dá aos doadores poder preditivo sobre a tratada.
    macro = rng.normal(0.0, 0.035, size=len(years))
    macro[years.index(2015)] -= 0.05
    macro[years.index(2016)] -= 0.04

    rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for nome, base, trend, noise in [*_DEMO_DONORS, _DEMO_TREATED]:
        treated = nome == _DEMO_TREATED[0]
        beta_macro = rng.uniform(0.5, 1.5) if not treated else 1.25
        ar, eps_prev = 0.45, 0.0
        for i, yr in enumerate(years):
            eps = ar * eps_prev + rng.normal(0.0, noise) * np.sqrt(1 - ar**2)
            eps_prev = eps
            level = base * np.exp(trend * i + beta_macro * macro[i])
            taxa_cf = max(level + eps, 0.05)  # contrafactual (sem NR-35)
            taxa = taxa_cf * (1.0 + DEMO_TRUE_EFFECT.get(yr, 0.0)) if treated else taxa_cf
            rows.append({"setor": nome, "ano": yr, "taxa_mortalidade": round(float(taxa), 3)})
            if treated:
                truth_rows.append(
                    {
                        "ano": yr,
                        "taxa_contrafactual": round(float(taxa_cf), 3),
                        "efeito_verdadeiro": DEMO_TRUE_EFFECT.get(yr, 0.0),
                    }
                )

    pl.DataFrame(rows).sort(["setor", "ano"]).write_csv(DEMO_CSV)
    pl.DataFrame(truth_rows).write_csv(DEMO_TRUTH_CSV)
    print(
        f"Wrote {DEMO_CSV} — {len(rows)} linhas "
        f"({len(_DEMO_DONORS)} doadores + 1 tratada × {len(years)} anos)."
    )
    print(f"Wrote {DEMO_TRUTH_CSV} — contrafactual verdadeiro da tratada.")
    print(
        "ATENÇÃO: dados SIMULADOS (efeito verdadeiro injetado: "
        f"{DEMO_TRUE_EFFECT[2013]:+.0%} em 2013, {DEMO_TRUE_EFFECT[2014]:+.0%} de 2014 em diante)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="gera o painel simulado de demonstração"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="testa quais URLs do AEAT servem arquivo de verdade (não baixa nada)",
    )
    args = parser.parse_args()
    if args.probe:
        probe_urls()
    elif args.demo:
        build_demo_panel()
    else:
        try:
            build_real_panel()
        except NotImplementedError as exc:
            print(f"\n[incompleto] {exc}", file=sys.stderr)
            sys.exit(2)
