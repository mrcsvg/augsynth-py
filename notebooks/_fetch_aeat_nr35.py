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

Provenance of the real pipeline (all URLs verified 2026-08-24; see also
``_data/README-aeat.md``):

* Índice das edições: https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/saude-e-seguranca-do-trabalhador/acidente_trabalho_incapacidade
* Each edition publishes the reference year and the two previous ones, so the
  four editions 2010/2013/2016/2019 cover 2008-2019 with the most-revised
  figures for every year.
* Óbitos: Seção I, Subseção B — "Quantidade de acidentes do trabalho
  liquidados, por consequência, segundo a CNAE" (Brasil). Rows are CNAE 2.0
  *classes* (4 digits) + TOTAL + Ignorado; columns are 6 consequence groups
  (Total, Assistência Médica, <15 dias, >15 dias, Incapacidade Permanente,
  Óbito) × 3 years. Classes are aggregated here to divisões (2 digits).
* Vínculos (denominator): the AEAT publishes "número médio anual de vínculos"
  por CNAE from the 2009 edition onward; in editions where the workbook
  cannot be located this script raises with a pointer to RAIS as the
  alternative denominator.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent / "_data"
PANEL_CSV = DATA_DIR / "aeat_nr35_panel.csv"
DEMO_CSV = DATA_DIR / "aeat_nr35_panel_demo.csv"
DEMO_TRUTH_CSV = DATA_DIR / "aeat_nr35_demo_truth.csv"

GOVBR = (
    "https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/"
    "saude-e-seguranca-do-trabalhador/acidente_trabalho_incapacidade"
)

# Four editions cover 2008-2019; each brings its reference year plus the two
# previous ones, already revised. `zip_url` is the tables bundle; for editions
# distributed as loose .xls files (2012-era site), `b_brasil_xls` points
# straight at table 29.1 (Subseção B, Brasil × CNAE).
EDITIONS: dict[int, dict[str, object]] = {
    2010: {
        "years": (2008, 2009, 2010),
        "zip_url": f"{GOVBR}/arquivos/tabelas-aeat-2010.zip",
    },
    2013: {
        "years": (2011, 2012, 2013),
        "zip_url": f"{GOVBR}/arquivos/aeat_tabelas_2013.zip",
        # Fallback: loose files, schema confirmed for the 2013 edition site
        # (tabelas-b-2013 -> /pt-br/outros/imagens/2015/01/29a_01.xls).
        "b_brasil_xls": "https://www.gov.br/previdencia/pt-br/outros/imagens/2015/01/29a_01.xls",
    },
    2016: {
        "years": (2014, 2015, 2016),
        "zip_url": f"{GOVBR}/arquivos/aeat-2016.zip",
    },
    2019: {
        "years": (2017, 2018, 2019),
        "zip_url": f"{GOVBR}/arquivos/aeat-2019_def.zip",
    },
}

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


def _download(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "augsynth-py/aeat-fetch"})
    with urlopen(req, timeout=180) as resp:
        return resp.read()


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


def _parse_subsecao_b(xls_bytes: bytes, years: tuple[int, int, int]) -> dict[tuple[str, int], int]:
    """Parse óbitos by CNAE class from a Subseção B workbook, per year.

    Layout (confirmed against the AEAT 2012 PDF, tables 29.x): rows = CNAE 2.0
    classes (4-digit codes) + TOTAL + Ignorado; 18 numeric columns = 6
    consequence groups × 3 years, Óbito being the LAST group (last 3 numeric
    columns, years ascending). Returns {(divisao, ano): óbitos} aggregated to
    2-digit divisões, plus ("TOTAL", ano) for validation.
    """
    import pandas as pd

    df = pd.read_excel(io.BytesIO(xls_bytes), header=None)
    out: dict[tuple[str, int], int] = {}
    for _, row in df.iterrows():
        cells = row.tolist()
        label = str(cells[0]).strip()
        numeric = [c for c in cells[1:] if isinstance(c, (int, float)) and c == c]
        if len(numeric) < 18:
            continue
        obito_cols = numeric[-3:]
        if re.fullmatch(r"\d{4}", label):
            key = label[:2]
        elif label.upper().startswith("TOTAL"):
            key = "TOTAL"
        else:
            continue
        for yr, val in zip(years, obito_cols, strict=True):
            out[(key, yr)] = out.get((key, yr), 0) + int(val)
    if not out:
        raise ValueError("Nenhuma linha CNAE reconhecida — layout mudou; inspecione o workbook.")
    return out


def build_real_panel() -> None:

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    obitos: dict[tuple[str, int], int] = {}
    for ed, spec in sorted(EDITIONS.items()):
        years = spec["years"]  # type: ignore[assignment]
        print(f"AEAT {ed}: baixando {spec['zip_url']} ...")
        try:
            blob = _download(str(spec["zip_url"]))
            zf = zipfile.ZipFile(io.BytesIO(blob))
            member = _find_subsecao_b_brasil(zf)
            print(f"  tabela 29.1: {member}")
            parsed = _parse_subsecao_b(zf.read(member), years)  # type: ignore[arg-type]
        except Exception as exc:
            fallback = spec.get("b_brasil_xls")
            if not fallback:
                raise RuntimeError(f"AEAT {ed}: falhou e não há fallback: {exc}") from exc
            print(f"  ZIP falhou ({exc}); tentando {fallback}")
            parsed = _parse_subsecao_b(_download(str(fallback)), years)  # type: ignore[arg-type]
        for (div, yr), v in parsed.items():
            # Each year appears in up to two editions; keep the LATEST edition
            # (iteration is sorted, later editions overwrite = revised figures).
            obitos[(div, yr)] = v
        totals = {yr: parsed.get(("TOTAL", yr)) for yr in years}  # type: ignore[union-attr]
        print(f"  TOTAL Brasil (óbitos): {totals}  <- confira contra o PDF da edição")

    # Denominators: the AEAT "número médio anual de vínculos" workbooks are not
    # yet wired here (locations vary per edition). Building the CSV without
    # them would silently ship a count outcome where the notebook expects a
    # rate — refuse instead, pointing at the alternatives.
    raise NotImplementedError(
        "Óbitos extraídos, mas o denominador (vínculos por divisão CNAE) ainda "
        "não está plugado. Opções: (1) tabelas de vínculos do AEAT (edições "
        ">=2009), (2) RAIS/vínculos por divisão (basedosdados: br_me_rais). "
        "Complete _fetch_vinculos() e recalcule taxa = obitos/vinculos*1e5. "
        f"Óbitos por (divisão, ano) já disponíveis em memória: {len(obitos)} células."
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
    args = parser.parse_args()
    if args.demo:
        build_demo_panel()
    else:
        try:
            build_real_panel()
        except NotImplementedError as exc:
            print(f"\n[incompleto] {exc}", file=sys.stderr)
            sys.exit(2)
