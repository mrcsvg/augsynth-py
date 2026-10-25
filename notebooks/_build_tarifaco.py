"""Build ``04_tarifaco_eua_brasil.ipynb`` from a programmatic spec.

Run once to (re)generate the notebook, then execute it::

    .venv/bin/python notebooks/_build_tarifaco.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/04_tarifaco_eua_brasil.ipynb

The data it reads is produced by ``_fetch_tarifaco_data.py`` and checked in
under ``notebooks/_data/tarifaco/``. Same pattern as ``_build_showcase.py``:
the notebook is the artifact people open; this script is the source of truth
used to recreate it without diff noise from cell metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NOTEBOOK_PATH = Path(__file__).parent / "04_tarifaco_eua_brasil.ipynb"

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


CELLS: list[dict] = []

# ---------------------------------------------------------------------------
# 0. Header
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    # O tarifaço dos EUA e as exportações brasileiras

    **Quanto o Brasil perdeu, para onde foram as vendas — e quanto teria perdido sem novos mercados?**

    Em 9 de julho de 2025 Trump anunciou uma tarifa de 50% sobre produtos brasileiros. A
    sobretaxa de 40% (somada aos 10% "recíprocos" de abril) vigorou de 6 de agosto de 2025
    até 23 de fevereiro de 2026, quando a Suprema Corte dos EUA decidiu que a lei usada
    (IEEPA) não autoriza tarifas. No meio do caminho, o governo brasileiro lançou o
    *Plano Brasil Soberano* e acelerou a agenda de abertura de mercados.

    Este notebook é um **estudo** — o objetivo é tanto responder às perguntas quanto mostrar
    *como* respondê-las com controle sintético, usando o `augsynth-py` nativo (sem R):

    1. **Qual foi o efeito direto** da sobretaxa sobre as exportações aos EUA?
    2. **O total sentiu?** As exportações brasileiras para o mundo caíram?
    3. **Para onde foram as vendas** que deixaram de ir para os EUA?
    4. **E se não houvesse novos mercados?** Qual teria sido o impacto sem o redirecionamento?

    ## Roteiro

    | § | Seção | Pergunta |
    |---|---|---|
    | 1 | Linha do tempo | O que mudou, e quando |
    | 2 | Dados | De onde vêm os números; quais produtos foram tarifados |
    | 3 | Primeiro olhar | O que os dados brutos mostram — e por que não bastam |
    | 4 | Brasil × outros países | Efeito direto, controle sintético entre países |
    | 5 | Brasil × ele mesmo | Efeito direto, produtos tarifados × isentos |
    | 6 | O total | Exportações para o mundo e o limite do que o método enxerga |
    | 7 | Para onde foram as vendas | Redirecionamento, produto a produto |
    | 8 | E se não houvesse novos mercados? | Cenários contrafactuais |
    | 9 | Limitações | O que pode estar errado |
    | 10 | Conclusões | |

    > **Como rodar.** Os dados já estão em `notebooks/_data/tarifaco/` (gerados por
    > `notebooks/_fetch_tarifaco_data.py`, que baixa tudo de APIs públicas sem chave). O
    > notebook roda em ~1 minuto. Valores em US$ correntes (FOB); "bi" = bilhão.
""")
)

CELLS.append(
    code("""
    from __future__ import annotations

    import datetime as dt
    import warnings
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    import augsynth_py
    from augsynth_py import (
        AugSynth,
        Synth,
        conformal_interval,
        conformal_test,
        leave_one_out,
        placebo_in_space,
        simulate_power,
    )

    # Localiza notebooks/_data independentemente de onde o kernel iniciou.
    _here = Path.cwd()
    for _cand in (_here, _here / "notebooks", _here.parent / "notebooks"):
        if (_cand / "_data" / "tarifaco").exists():
            DATA = _cand / "_data" / "tarifaco"
            break
    else:
        raise RuntimeError(f"notebooks/_data/tarifaco não encontrado a partir de {_here}")

    # Visual: superfície clara, grade discreta, uma cor por papel (não por posição).
    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    C_BRASIL = "#2a78d6"   # Brasil observado
    C_SINT = "#eb6834"     # contrafactual (Brasil sintético)
    C_3, C_4 = "#1baf7a", "#eda100"
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
        "font.size": 10, "figure.dpi": 110,
    })
    pl.Config.set_tbl_rows(30)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    # Datas-chave (ver §1) e janelas usadas em todo o notebook.
    ANUNCIO = dt.date(2025, 7, 1)        # mês da carta de Trump (9/jul): início do pós-tratamento
    FIM_IEEPA = dt.date(2026, 2, 1)      # último mês com a sobretaxa de 40% cobrada
    INICIO = dt.date(2022, 7, 1)         # início do pré-período (36 meses; ver §1 e §5.1)
    EVENTOS = [
        (dt.date(2025, 7, 9), "carta (50%)"),
        (dt.date(2025, 8, 6), "+40% em vigor"),
        (dt.date(2025, 11, 13), "alívio agro"),
        (dt.date(2026, 2, 24), "Suprema Corte"),
        (dt.date(2026, 6, 1), "301 proposta"),
        (dt.date(2026, 7, 22), "Seção 301"),
    ]

    def marcar_eventos(ax, ate=None, rotulos=True):
        for d, rot in EVENTOS:
            if ate is not None and d > ate:
                continue
            ax.axvline(d, color=MUTED, lw=0.8, zorder=0)
            if rotulos:
                ax.text(d, 1.0, f" {rot}", transform=ax.get_xaxis_transform(), rotation=90,
                        va="top", ha="right", fontsize=7.5, color=INK2)

    def eixo_meses(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\\n%Y"))

    def bi(x: float) -> str:
        return f"US$ {x / 1e9:,.2f} bi".replace(",", "X").replace(".", ",").replace("X", ".")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print(f"augsynth-py {augsynth_py.__version__} | dados em {DATA}")
""")
)

# ---------------------------------------------------------------------------
# 1. Timeline
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 1. Linha do tempo

    | Data | Evento | Alcance |
    |---|---|---|
    | 5 abr 2025 | Tarifa "recíproca" de 10% (EO 14257) | Todos os produtos, quase todos os países |
    | 9 jul 2025 | Carta de Trump anuncia 50% para o Brasil a partir de 1º/ago | Anúncio — exportadores antecipam embarques |
    | 30 jul 2025 | **EO 14323**: +40% sobre o Brasil (IEEPA), com ~700 linhas isentas (Anexo I) | MDIC: 35,9% das exportações aos EUA atingidas |
    | 6 ago 2025 | Sobretaxa entra em vigor | Total de 50% nos produtos não isentos |
    | 13 ago 2025 | Plano Brasil Soberano (MP 1.309): crédito, Reintegra, compras públicas | Apoio a exportadores atingidos |
    | 20 nov 2025 | **EO 14361** retira os 40% de carne, café, frutas e outros agro (retroativo a 13/nov) | Alívio para o agro |
    | 20 fev 2026 | Suprema Corte: IEEPA não autoriza tarifas | Cobrança dos 40% para em 24/fev |
    | 24 fev 2026 | Tarifa global de 10% (Seção 122, 150 dias) | Igual para todos os países |
    | 1º jun 2026 | USTR conclui a investigação da Seção 301 e propõe 25% | Anúncio |
    | 22 jul 2026 | **Seção 301**: +25% sobre parte dos produtos brasileiros (+12,5% "trabalho forçado") | MDIC: 23,1% das exportações aos EUA |

    Daí saem as janelas usadas no notebook:

    * **Pré-período**: jul/2022 a jun/2025 (36 meses). Três anos dão três observações por mês para
      estimar a sazonalidade e deixam de fora a pandemia e o pico de preços de commodities do 1º
      semestre de 2022 (guerra na Ucrânia), quando a estrutura do comércio era outra. A §5.1 mostra
      como os resultados mudam com outras janelas.
    * **Tarifaço** (pós-tratamento principal): **jul/2025 a fev/2026** — 8 meses. Começa em julho,
      o mês do anúncio, e não em agosto: a antecipação de embarques em julho é parte do efeito, e
      deixá-la no pré-período contaminaria o ajuste (Abadie 2021, §3, recomenda "recuar" o
      tratamento para a data do anúncio quando há antecipação).
    * **Depois da Suprema Corte** (mar–mai/2026): o Brasil volta a pagar o mesmo que os outros. Se o
      efeito é da tarifa, a diferença deve diminuir. Em 1º/jun/2026 o USTR anuncia a Seção 301, em
      vigor desde 22/jul — um novo tratamento, que aparece no fim da série.

    Fontes: Federal Register (EO 14323, 90 FR 37739; EO 14361), notas do MDIC de 31/07/2025 e
    24/07/2026.
""")
)

# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 2. Dados

    | Arquivo | Fonte | Conteúdo |
    |---|---|---|
    | `comtrade_exportacoes.csv` | UN Comtrade (API pública) | Exportações mensais de 57 economias, para o mundo e para os EUA, total e capítulo 27 (combustíveis), 2019–2026 |
    | `comex_eua_categoria.csv` | Comex Stat (MDIC) | Exportações brasileiras mensais aos EUA por categoria tarifária |
    | `comex_eua_isentos_sh4.csv` | Comex Stat | Idem, por posição SH4, só produtos com isenção geral do Anexo I |
    | `classificacao_sh6.csv` | EO 14323/14361 + Comex Stat | Cada SH6 exportado aos EUA, com sua categoria tarifária |
    | `sh4_grupos.csv`, `sh4_janelas.csv` | Comex Stat | Grupos de produtos e exportações por destino antes/durante o tarifaço |
    | `comex_margens.csv`, `comex_novos_pares.csv` | Comex Stat | Exportações por par produto–destino "existente" ou "novo" (§8) |

    Duas escolhas atravessam o notebook:

    * **Sem combustíveis.** O petróleo bruto era ~14% das vendas aos EUA, estava isento e caiu em
      2026 por razões próprias (preço e redirecionamento da Petrobras). Nos países, a variável é
      exportações totais menos o capítulo 27.
    * **Valores, não volumes.** Comtrade e Comex Stat dão valor em dólar; preços (café alto, petróleo
      em queda) entram no resultado. O controle sintético compensa choques de preço comuns aos
      doadores, não os específicos de um produto.
""")
)

CELLS.append(
    code("""
    comtrade = pl.read_csv(DATA / "comtrade_exportacoes.csv", try_parse_dates=True)
    classif = pl.read_csv(DATA / "classificacao_sh6.csv", schema_overrides={"sh6": pl.Utf8})
    eua_cat = pl.read_csv(DATA / "comex_eua_categoria.csv", try_parse_dates=True)
    eua_isentos = pl.read_csv(DATA / "comex_eua_isentos_sh4.csv", try_parse_dates=True,
                              schema_overrides={"sh4": pl.Utf8})
    grupos = pl.read_csv(DATA / "sh4_grupos.csv", schema_overrides={"sh4": pl.Utf8})
    janelas = pl.read_csv(DATA / "sh4_janelas.csv", schema_overrides={"sh4": pl.Utf8})
    margens = pl.read_csv(DATA / "comex_margens.csv", try_parse_dates=True)
    novos_pares = pl.read_csv(DATA / "comex_novos_pares.csv", schema_overrides={"sh4": pl.Utf8})

    print(f"Comtrade : {comtrade['pais'].n_unique()} economias, {comtrade['mes'].min()} a {comtrade['mes'].max()}")
    print(f"Comex    : exportações aos EUA de {eua_cat['mes'].min()} a {eua_cat['mes'].max()}")
""")
)

CELLS.append(
    md("""
    ### 2.1 Quais produtos foram tarifados?

    O Anexo I da EO 14323 lista as linhas de 8 dígitos da tarifa americana (HTSUS) **isentas** dos
    40%. Os 6 primeiros dígitos da HTSUS e da NCM brasileira coincidem (são o Sistema Harmonizado),
    então cada SH6 exportado aos EUA cai em uma de quatro categorias:

    * **`tarifado`** — pagou os 40% (de 6/ago/2025 até a decisão da Suprema Corte; parte do agro só até 13/nov);
    * **`isento`** — tem linha com isenção geral no Anexo I (petróleo, celulose, suco de laranja, ferro-gusa, minérios, fertilizantes...);
    * **`isento_aeronaves`** — só tem linhas "apenas aeronaves civis" no Anexo I. É uma mistura: a peça que vai para a Embraer estava isenta, o mesmo motor elétrico vendido a uma fábrica não. Contamos como isento para comparar com o MDIC, mas **nunca** usamos como controle;
    * **`secao_232`** — aço, alumínio, cobre, autos e autopeças: fora dos 40% porque já pagavam as tarifas setoriais da Seção 232.

    Comparando com a conta oficial do MDIC (sobre as exportações de 2024):
""")
)

CELLS.append(
    code("""
    tot24 = classif["fob_eua_2024"].sum()
    nossa = (
        classif.with_columns(
            pl.col("categoria").replace({"isento_aeronaves": "isento"}).alias("cat_mdic")
        )
        .group_by("cat_mdic")
        .agg((pl.col("fob_eua_2024").sum() / tot24 * 100).round(1).alias("este notebook (%)"))
    )
    mdic = pl.DataFrame({"cat_mdic": ["tarifado", "isento", "secao_232"], "MDIC 31/07/2025 (%)": [35.9, 44.6, 19.5]})
    print(f"Exportações aos EUA em 2024: {bi(tot24)}")
    mdic.join(nossa, on="cat_mdic").rename({"cat_mdic": "categoria"})
""")
)

CELLS.append(
    md("""
    A classificação bate razoavelmente: ~40% tarifado contra 35,9% do MDIC. A diferença vem de
    linhas parciais ("Ex") e do uso do produto, que o MDIC conseguiu separar e o SH6 não consegue.
    Para o que vem a seguir isso basta: o grupo `tarifado` tem que ser majoritariamente tarifado,
    e o grupo de controle (`isento`), majoritariamente isento.
""")
)

# ---------------------------------------------------------------------------
# 3. First look
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 3. Primeiro olhar

    Exportações brasileiras aos EUA por categoria, em média móvel de 3 meses (o dado mensal é muito
    ruidoso):
""")
)

CELLS.append(
    code("""
    serie = (
        eua_cat.with_columns(pl.col("categoria").replace({"isento_aeronaves": "isento"}))
        .group_by("mes", "categoria").agg(pl.col("fob_usd").sum())
        .sort("categoria", "mes")
        .with_columns((pl.col("fob_usd").rolling_mean(3).over("categoria") / 1e9).alias("mm3"))
        .filter(pl.col("mes") >= dt.date(2023, 1, 1))
    )
    cores_cat = {"tarifado": C_BRASIL, "isento": C_SINT, "secao_232": C_3}
    rotulo_cat = {"tarifado": "Tarifados (40%)", "isento": "Isentos (Anexo I)", "secao_232": "Seção 232 (aço, autos...)"}

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for cat, cor in cores_cat.items():
        s = serie.filter(pl.col("categoria") == cat)
        ax.plot(s["mes"], s["mm3"], color=cor, label=rotulo_cat[cat])
        ax.annotate(rotulo_cat[cat], (s["mes"][-1], s["mm3"][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    marcar_eventos(ax)
    eixo_meses(ax)
    ax.set_ylabel("US$ bi por mês (média móvel 3m)")
    ax.set_title("Exportações brasileiras aos EUA por categoria tarifária")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(None, dt.date(2027, 2, 1))
    plt.show()
""")
)

CELLS.append(
    code("""
    def soma(cat, lo, hi):
        return eua_cat.filter(
            pl.col("categoria").replace({"isento_aeronaves": "isento"}).eq(cat)
            & pl.col("mes").is_between(lo, hi)
        )["fob_usd"].sum()

    antes = (dt.date(2024, 7, 1), dt.date(2025, 2, 1))
    durante = (ANUNCIO, FIM_IEEPA)
    pl.DataFrame([
        {"categoria": c,
         "jul/24–fev/25 (US$ bi)": round(soma(c, *antes) / 1e9, 2),
         "jul/25–fev/26 (US$ bi)": round(soma(c, *durante) / 1e9, 2),
         "variação (%)": round((soma(c, *durante) / soma(c, *antes) - 1) * 100, 1)}
        for c in ("tarifado", "isento", "secao_232")
    ])
""")
)

CELLS.append(
    md("""
    Os tarifados caem forte a partir de agosto e voltam a subir depois de fevereiro. Mas os isentos
    também caíram, e as vendas de aço (Seção 232, tarifa dobrada para 50% em junho/2025) recuaram 18%.
    Comparar "antes e depois" atribuiria à sobretaxa tudo o que mudou no período: preços de
    commodities, a demanda americana, as tarifas que *todo mundo* passou a pagar em 2025.

    O que precisamos é de um **contrafactual**: quanto o Brasil teria exportado *sem* a sobretaxa,
    com todo o resto igual. O controle sintético (Abadie, Diamond & Hainmueller 2010) constrói esse
    contrafactual como uma média ponderada de unidades não tratadas, com pesos escolhidos para
    reproduzir a trajetória do Brasil antes do tratamento. Vamos construí-lo de dois jeitos:

    * **§4 — com outros países** como doadores ("o que outros exportadores venderam aos EUA");
    * **§5 — com o próprio Brasil**: produtos isentos como doadores dos tarifados.

    Cada desenho tem uma fraqueza diferente; se os dois contam a mesma história, a confiança aumenta.
""")
)

# ---------------------------------------------------------------------------
# 4. Cross-country SC
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 4. Efeito direto — Brasil × outros países

    ### 4.1 Desenho

    * **Variável**: exportações não-combustíveis aos EUA (dado do país exportador, Comtrade).
    * **Normalização**: países têm escalas muito diferentes (a Alemanha exporta cinco vezes mais que o Brasil), e o
      controle sintético é uma média *convexa* — o Brasil precisa estar "dentro" dos doadores. Por
      isso cada série vira um **índice (100 = média do pré-período)**.
    * **Sazonalidade**: comércio mensal tem sazonalidade forte e específica (a safra de soja
      brasileira não é a de ninguém). Removemos um fator multiplicativo por mês-calendário,
      **estimado só no pré-período** (nada do pós-tratamento entra no ajuste).
    * **Painel**: jul/2022–fev/2026, só países com todos os meses.

    **Quem pode ser doador** é a decisão mais importante do controle sintético. Abadie (2021, §3)
    pede que o pool exclua (i) unidades atingidas pelo mesmo tratamento ou por intervenções
    parecidas, (ii) unidades que sofrem *spillover* do tratamento e (iii) unidades com choques
    idiossincráticos grandes no período. Traduzindo para o tarifaço:

    | Regra | Excluídos | Por quê |
    |---|---|---|
    | 1. Regime tarifário próprio | China, Índia, Canadá, México, Suíça, África do Sul | Índia também levou 50% em ago/2025; China teve tarifas e tréguas próprias; Canadá/México têm o regime do USMCA; Suíça levou 39%, África do Sul 30% |
    | 2. Concorrentes diretos (*spillover*) | Colômbia, Argentina, Paraguai, Uruguai, Austrália, Nova Zelândia | Grandes vendedores de café e carne aos EUA: o comprador americano que deixou o Brasil foi para eles. Como doadores, *somariam* o ganho deles à perda do Brasil |
    | 3. Choques próprios | Malásia, Tailândia, Singapura, Filipinas, Irlanda | Exportações aos EUA puxadas pelo boom de servidores de IA (Sudeste Asiático) e pela antecipação de fármacos (Irlanda) |

    Na §4.5 rodamos o mesmo modelo com e sem as regras 2 e 3, para ver quanto elas pesam.
""")
)

CELLS.append(
    code("""
    REGRA_1 = ["China", "Índia", "Canadá", "México", "Suíça", "África do Sul"]
    REGRA_2 = ["Colômbia", "Argentina", "Paraguai", "Uruguai", "Austrália", "Nova Zelândia"]
    REGRA_3 = ["Malásia", "Tailândia", "Singapura", "Filipinas", "Irlanda"]
    EXCLUIDOS = REGRA_1 + REGRA_2 + REGRA_3

    def variaveis_pais(df: pl.DataFrame) -> pl.DataFrame:
        \"\"\"Painel largo país × mês com as três variáveis não-combustíveis.\"\"\"
        w = df.with_columns((pl.col("total_usd") - pl.col("combustiveis_usd")).alias("sc")).pivot(
            on="destino", index=["pais", "mes"], values="sc"
        )
        return w.select(
            "pais", "mes",
            pl.col("eua").alias("eua"),
            pl.col("mundo").alias("mundo"),
            (pl.col("mundo") - pl.col("eua")).alias("resto"),
        )

    def preparar_painel(df, unidade, valor, inicio, fim, t0=ANUNCIO, excluir=(), dessazonalizar=True):
        \"\"\"Índice 100 = média pré-t0, dessazonalizado com fatores mensais do pré-período.

        Devolve também `escala`: multiplicando o índice por ela volta-se a US$ do mês
        (com a sazonalidade), o que permite converter gaps em dinheiro.
        \"\"\"
        meses = pl.date_range(inicio, fim, "1mo", eager=True)
        p = (df.filter(pl.col("mes").is_between(inicio, fim) & ~pl.col(unidade).is_in(list(excluir)))
               .select(pl.col(unidade).alias("unidade"), "mes", pl.col(valor).alias("usd"))
               .drop_nulls().filter(pl.col("usd") > 0))
        completos = p.group_by("unidade").agg(pl.len().alias("n")).filter(pl.col("n") == len(meses))
        p = p.join(completos.select("unidade"), on="unidade")
        p = p.with_columns(pl.col("usd").log().alias("ly"), pl.col("mes").dt.month().alias("m"))
        pre = pl.col("mes") < t0
        if dessazonalizar:
            fatores = (p.filter(pre)
                        .with_columns((pl.col("ly") - pl.col("ly").mean().over("unidade")).alias("s"))
                        .group_by("unidade", "m").agg(pl.col("s").mean()))
            p = p.join(fatores, on=["unidade", "m"])
        else:
            p = p.with_columns(pl.lit(0.0).alias("s"))
        p = p.with_columns((pl.col("ly") - pl.col("s")).exp().alias("v"))
        p = p.with_columns(pl.col("v").filter(pre).mean().over("unidade").alias("base"))
        return p.select(
            "unidade", "mes",
            (100 * pl.col("v") / pl.col("base")).alias("y"),
            "usd",
            (pl.col("base") * pl.col("s").exp() / 100).alias("escala"),
        ).sort("unidade", "mes")

    def ajustar(painel, tratado, est=None, t0=ANUNCIO):
        est = Synth() if est is None else est
        return est.fit(painel, unit="unidade", time="mes", outcome="y", treated=tratado, treatment_time=t0)

    def gap_usd(fit, painel, tratado):
        esc = painel.filter(pl.col("unidade") == tratado).sort("mes")["escala"].to_numpy()
        return fit.gap_ * esc

    paises = variaveis_pais(comtrade)
    painel_eua = preparar_painel(paises, "pais", "eua", INICIO, FIM_IEEPA, excluir=EXCLUIDOS)
    doadores = sorted(set(painel_eua["unidade"]) - {"Brasil"})
    print(f"{len(doadores)} doadores: {', '.join(doadores)}")
""")
)

CELLS.append(
    md("""
    ### 4.2 Ajuste

    Duas versões do estimador:

    * **`Synth`** — controle sintético clássico: pesos não negativos que somam 1 (ADH 2010, eq. 1),
      com efeito fixo de unidade (`fixedeff=True`, o padrão do `augsynth` em R).
    * **`AugSynth`** — controle sintético aumentado (Ben-Michael, Feller & Rothstein 2021): corrige o
      viés de um ajuste pré-período imperfeito com uma regressão ridge; λ por validação cruzada.
""")
)

CELLS.append(
    code("""
    fit_eua = ajustar(painel_eua, "Brasil")
    fit_eua_aug = ajustar(painel_eua, "Brasil", AugSynth())

    pesos = (pl.DataFrame({"doador": list(fit_eua.weights_), "peso": list(fit_eua.weights_.values())})
               .filter(pl.col("peso") > 0.005).sort("peso", descending=True)
               .with_columns(pl.col("peso").round(3)))
    print(f"Synth   : ATT = {fit_eua.att_:6.1f} pontos de índice | RMSPE pré = {fit_eua.rmspe_pre_:.1%}")
    print(f"AugSynth: ATT = {fit_eua_aug.att_:6.1f} pontos de índice | RMSPE pré = {fit_eua_aug.rmspe_pre_:.1%} | λ = {fit_eua_aug.lambda_:.3g}")
    pesos
""")
)

CELLS.append(
    code("""
    def grafico_sc(fit, titulo, ylabel="índice (100 = média pré)", fit_aug=None, ate=None):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1]})
        per = fit.periods_
        a1.plot(per, fit.actual_, color=C_BRASIL, label="Brasil (observado)")
        a1.plot(per, fit.synthetic_, color=C_SINT, ls="--", label="Brasil sintético (Synth)")
        if fit_aug is not None:
            a1.plot(per, fit_aug.synthetic_, color=C_SINT, lw=1.2, ls=":", label="Brasil sintético (AugSynth)")
        a1.axvspan(fit.treatment_time_, per[-1], color=GRID, alpha=0.35, zorder=0, lw=0)
        a1.set_ylabel(ylabel)
        a1.set_title(titulo)
        a1.legend(loc="upper left", fontsize=8.5)
        marcar_eventos(a1, ate=ate)
        a2.axhline(0, color=AXIS, lw=1)
        a2.bar(per, fit.gap_, width=20, color=np.where(fit.gap_ < 0, C_BRASIL, MUTED))
        a2.set_ylabel("gap (obs. − sint.)")
        a2.axvspan(fit.treatment_time_, per[-1], color=GRID, alpha=0.35, zorder=0, lw=0)
        marcar_eventos(a2, ate=ate, rotulos=False)
        eixo_meses(a2)
        fig.tight_layout()
        plt.show()

    grafico_sc(fit_eua, "Exportações não-combustíveis aos EUA: Brasil × Brasil sintético")
""")
)

CELLS.append(
    md("""
    O Brasil sintético acompanha o Brasil por três anos (erro médio de ~4% do nível, com a
    sazonalidade já removida) e os dois se separam em agosto/2025: o Brasil cai para o patamar de
    75–85 enquanto o sintético segue em 100–120. O gap fica entre −16 e −40 pontos de agosto a
    fevereiro e é menor em fevereiro/2026, o mês em que a sobretaxa deixou de ser cobrada.

    O `AugSynth` chega a um ATT parecido (veja acima), mas o seu contrafactual mês a mês oscila muito
    no pós-período, e por isso não aparece no gráfico. Com λ pequeno, o ridge praticamente interpola o
    pré-período e projeta o ruído dos doadores para frente. Quando o ajuste do `Synth` já é bom, a
    correção ridge tem pouco a acrescentar (BFR 2021, §3).

    ### 4.3 Em dólares

    O gap está em pontos de índice. Multiplicando pela escala de cada mês (média pré-período ×
    fator sazonal), volta-se a US$:
""")
)

CELLS.append(
    code("""
    def tabela_gaps(fit, painel, tratado):
        g = gap_usd(fit, painel, tratado)
        pos = ~fit.pre_mask_
        obs = painel.filter(pl.col("unidade") == tratado).sort("mes")["usd"].to_numpy()
        return pl.DataFrame({
            "mês": fit.periods_[pos],
            "observado (US$ bi)": np.round(obs[pos] / 1e9, 2),
            "contrafactual (US$ bi)": np.round((obs[pos] - g[pos]) / 1e9, 2),
            "gap (US$ bi)": np.round(g[pos] / 1e9, 2),
            "gap (pontos)": np.round(fit.gap_[pos], 1),
        })

    tab = tabela_gaps(fit_eua, painel_eua, "Brasil")
    PERDA_PAISES = float(tab["gap (US$ bi)"].sum()) * 1e9
    obs_total = float(tab["observado (US$ bi)"].sum()) * 1e9
    print(f"Perda acumulada jul/25–fev/26: {bi(PERDA_PAISES)} "
          f"({PERDA_PAISES / (obs_total - PERDA_PAISES):.1%} do contrafactual)")
    tab
""")
)

CELLS.append(
    md("""
    ### 4.4 É real? Inferência

    Três ferramentas, cada uma respondendo a uma pergunta diferente:

    * **Inferência conformal** (Chernozhukov, Wüthrich & Zhu 2021): testa a hipótese de efeito zero
      refazendo o ajuste *sob a nula* e comparando os resíduos do pós-período com permutações em
      blocos da série. Dá um p-valor exato sob a nula e um intervalo de confiança por inversão do teste.
    * **Placebo no espaço** (ADH 2010): finge que cada doador foi o tratado e compara a razão
      RMSPE pós/pré do Brasil com a deles. Pergunta: *o Brasil é incomum?*
    * **Leave-one-out** (Abadie 2021, §6): tira um doador de cada vez. Pergunta: *o resultado depende
      de um único país?*
""")
)

CELLS.append(
    code("""
    def inferencia(fit, alpha=0.10, nome=""):
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            ct = conformal_test(fit)
            ic = conformal_interval(fit, alpha=alpha)
        pis = placebo_in_space(fit)
        loo = leave_one_out(fit)
        rank = pis.ratios.filter(pl.col("is_treated"))["rank"][0]
        n = pis.ratios.filter(pl.col("rank").is_not_null()).height
        print(f"{nome}ATT = {fit.att_:.1f} pontos")
        print(f"  conformal : p = {ct.pvalue:.3f} | IC {1 - alpha:.0%} = [{ic[0]:.1f}, {ic[1]:.1f}] | "
              f"parcela absorvida pela reestimação sob a nula = {ct.absorbed_share:.0%}")
        print(f"  placebo   : Brasil é o {rank}º de {n} na razão RMSPE pós/pré -> p = {pis.pvalue:.3f}")
        print(f"  LOO       : ATT entre {loo.att_range[0]:.1f} e {loo.att_range[1]:.1f}; mais influente = "
              f"{loo.most_influential}; sinal estável = {loo.sign_stable}")
        for a in avisos:
            if issubclass(a.category, UserWarning):
                print(f"  aviso     : {str(a.message)[:160]}...")
        return ct, pis, loo

    ct_eua, pis_eua, loo_eua = inferencia(fit_eua)
""")
)

CELLS.append(
    code("""
    def grafico_placebos(pis, titulo):
        fig, ax = plt.subplots(figsize=(11, 3.8))
        for u, g in pis.gaps.group_by("unit"):
            if u[0] != "Brasil":
                ax.plot(g["period"], g["gap"], color=MUTED, lw=0.8, alpha=0.45)
        b = pis.gaps.filter(pl.col("unit") == "Brasil")
        ax.plot(b["period"], b["gap"], color=C_BRASIL, lw=2.2, label="Brasil")
        ax.plot([], [], color=MUTED, lw=0.8, label="placebos (cada doador como tratado)")
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(ANUNCIO, color=MUTED, lw=0.8)
        lim = max(60.0, 2.0 * float(np.abs(b["gap"]).max()))
        ax.set_ylim(-lim, lim)  # alguns placebos mal ajustados saem da escala
        ax.set_title(titulo)
        ax.set_ylabel("gap (pontos)")
        ax.legend(loc="lower left", fontsize=8.5)
        eixo_meses(ax)
        plt.show()

    grafico_placebos(pis_eua, "Placebos no espaço — exportações aos EUA")
""")
)

CELLS.append(
    md("""
    **Como ler.**

    * **Conformal**: rejeita o efeito zero (p ≈ 0,02), e o intervalo de 90% fica inteiro abaixo de
      zero. O aviso sobre a "parcela absorvida" diz que, ao reestimar sob a nula, o controle sintético
      explica boa parte do próprio gap que está testando (efeito fixo + troca de pesos). Isso tira
      poder do teste e entorta o intervalo (`docs/methodology.md` §5.5 e §5.10) — mas o viés é contra
      encontrar efeito, então rejeitar mesmo assim é um bom sinal.
    * **Placebo**: o Brasil é o 2º mais extremo entre 26 unidades. Com 26 unidades, o menor p possível é
      1/26 ≈ 0,04, e 2025 foi um ano turbulento para *todos* os exportadores aos EUA (antecipação no
      1º trimestre, tarifas recíprocas, tréguas) — os fios cinzas mostram isso.
    * **Leave-one-out**: nenhum doador sozinho muda o sinal ou a ordem de grandeza.

    ### 4.5 Robustez — e a lição do *spillover*
""")
)

CELLS.append(
    code("""
    especificacoes = {
        "principal (regras 1+2+3)": dict(),
        "pré-período desde 2021": dict(inicio=dt.date(2021, 1, 1)),
        "sem dessazonalizar": dict(dessazonalizar=False),
        "tratamento em ago/2025": dict(t0=dt.date(2025, 8, 1)),
        "inclui combustíveis": dict(valor="total"),
        "só regra 1 (mantém concorrentes e choques)": dict(excluir=REGRA_1),
        "regras 1+3 (mantém concorrentes)": dict(excluir=REGRA_1 + REGRA_3),
        "regras 1+2 (mantém choques)": dict(excluir=REGRA_1 + REGRA_2),
    }
    paises_total = comtrade.filter(pl.col("destino") == "eua").select("pais", "mes", pl.col("total_usd").alias("total"))
    linhas = []
    for nome, kw in especificacoes.items():
        kw = dict(kw)
        valor = kw.pop("valor", "eua")
        base = paises_total if valor == "total" else paises
        t0 = kw.get("t0", ANUNCIO)
        kw.setdefault("excluir", EXCLUIDOS)
        p = preparar_painel(base, "pais", valor, kw.pop("inicio", INICIO), FIM_IEEPA, **kw)
        f = ajustar(p, "Brasil", t0=t0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # λ na borda da grade em algumas variantes
            fa = ajustar(p, "Brasil", AugSynth(), t0=t0)
        g = gap_usd(f, p, "Brasil")
        linhas.append({"especificação": nome, "doadores": len(f.weights_),
                       "ATT Synth (pts)": round(f.att_, 1), "ATT AugSynth (pts)": round(fa.att_, 1),
                       "RMSPE pré": round(f.rmspe_pre_, 3),
                       "perda Synth (US$ bi)": round(g[~f.pre_mask_].sum() / 1e9, 2)})
    robustez = pl.DataFrame(linhas)
    robustez
""")
)

CELLS.append(
    md("""
    O sinal é negativo em todas as especificações, e nas variantes sem combustíveis a perda fica entre
    US$ 4,8 e 5,3 bi (com combustíveis, US$ 6,4 bi: o petróleo também caiu, por outras razões). Duas
    exceções ensinam algo:

    * **Com os concorrentes no pool** (sem a regra 2), a perda sobe ~35%. Isso não é ruído, é o
      *spillover* funcionando como previsto: Colômbia, Argentina e Paraguai venderam mais café e carne
      aos EUA justamente porque o Brasil ficou mais caro. Usados como doadores, eles inflam o Brasil
      sintético, e o "efeito" passa a somar a perda do Brasil ao ganho deles — a violação de
      "não-interferência" (SUTVA) que Abadie (2021) manda evitar.
    * **Com o pré-período desde 2021**, a perda cai para ~US$ 2 bi. Para reproduzir o superciclo de
      2021–22, os pesos migram para Alemanha e Países Baixos, cujas vendas aos EUA caíram no 2º
      semestre de 2025 por razões próprias (fármacos, semicondutores). A §5.1 olha isso com calma.
""")
)

# ---------------------------------------------------------------------------
# 5. Within-Brazil SC
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 5. Efeito direto — Brasil × ele mesmo

    O desenho entre países mede o efeito sobre *todas* as exportações aos EUA, das quais só ~40% foram
    tarifadas, e sofre com doadores heterogêneos. O segundo desenho troca os países por **produtos**:

    * **Tratado**: a cesta de produtos `tarifado` exportada aos EUA (soma, índice 100 = média pré).
    * **Doadores**: posições SH4 com **isenção geral** do Anexo I (celulose, ferro-gusa, petróleo,
      suco de laranja...), cada uma uma série. Ficam de fora as `isento_aeronaves` (tratamento
      ambíguo) e as da Seção 232 (que tiveram choque próprio: o aço foi a 50% em junho/2025).
    * Mesma dessazonalização e mesmo pré-período. O painel vai até **set/2026**, último mês do Comex
      Stat: os pesos dependem só do pré-período, então estender a série não muda o ajuste e permite
      ver o que aconteceu depois da Suprema Corte.

    Vantagem: mesmo exportador, mesmo câmbio, mesmos portos, mesma logística — o contrafactual é
    "o Brasil vendendo aos EUA o que não foi tarifado". Fraqueza: os isentos são commodities com
    ciclos de preço próprios e são poucos (~12 séries com pelo menos US$ 1 mi/mês), o que limita a
    inferência por placebo.
""")
)

CELLS.append(
    code("""
    FIM_COMEX = eua_cat["mes"].max()

    def painel_produtos(filtro, fim=FIM_COMEX, minimo=1e6, inicio=INICIO):
        tratado = (eua_cat.filter(filtro).group_by("mes").agg(pl.col("fob_usd").sum())
                          .with_columns(pl.lit("Tarifados").alias("unidade")))
        df = pl.concat([tratado.select("unidade", "mes", "fob_usd"),
                        eua_isentos.select(pl.col("sh4").alias("unidade"), "mes", "fob_usd")])
        p = preparar_painel(df, "unidade", "fob_usd", inicio, fim)
        # Doadores pequenos (< US$ 1 mi/mês no pré) são só ruído.
        grandes = (p.filter(pl.col("mes") < ANUNCIO).group_by("unidade")
                     .agg((pl.col("usd").mean()).alias("m")).filter(pl.col("m") >= minimo)["unidade"])
        return p.filter(pl.col("unidade").is_in(grandes.implode()))

    TARIFADO = pl.col("categoria") == "tarifado"
    painel_prod = painel_produtos(TARIFADO)
    fit_prod = ajustar(painel_prod, "Tarifados")
    descr_sh4 = dict(zip(grupos["sh4"], grupos["descricao"]))
    print(f"{len(fit_prod.weights_)} doadores | RMSPE pré = {fit_prod.rmspe_pre_:.1%}")
    (pl.DataFrame({"sh4": list(fit_prod.weights_), "peso": list(fit_prod.weights_.values())})
       .filter(pl.col("peso") > 0.005).sort("peso", descending=True)
       .with_columns(pl.col("peso").round(3),
                     pl.col("sh4").replace_strict(descr_sh4, default="").str.slice(0, 55).alias("descrição")))
""")
)

CELLS.append(
    code("""
    grafico_sc(fit_prod, "Produtos tarifados × sintético feito de produtos isentos (exportações aos EUA)")
""")
)

CELLS.append(
    code("""
    def media_janela(fit, lo, hi):
        m = (fit.periods_ >= np.datetime64(lo)) & (fit.periods_ <= np.datetime64(hi))
        return float(fit.gap_[m].mean())

    pl.DataFrame([
        {"janela": "40% para todos os tarifados (jul/25–out/25)", "gap médio (pts)": media_janela(fit_prod, ANUNCIO, dt.date(2025, 10, 1))},
        {"janela": "40% sem o agro aliviado (nov/25–fev/26)", "gap médio (pts)": media_janela(fit_prod, dt.date(2025, 11, 1), FIM_IEEPA)},
        {"janela": "depois da Suprema Corte (mar/26–mai/26)", "gap médio (pts)": media_janela(fit_prod, dt.date(2026, 3, 1), dt.date(2026, 5, 1))},
        {"janela": "Seção 301 anunciada e em vigor (jun/26–set/26)", "gap médio (pts)": media_janela(fit_prod, dt.date(2026, 6, 1), FIM_COMEX)},
    ]).with_columns(pl.col("gap médio (pts)").round(1))
""")
)

CELLS.append(
    md("""
    O ajuste pré-período é pior que o do desenho entre países (as commodities isentas são mais
    voláteis que a cesta tarifada), mas a **forma** do gap ao longo do tempo é o resultado mais
    informativo do notebook, porque acompanha cada mudança de política:

    * abre em agosto/2025, com os 40% em vigor, e chega ao fundo entre outubro e dezembro (−55 a −75 pontos);
    * começa a estreitar em janeiro, depois do alívio de novembro para carne, café e frutas;
    * cai para cerca de −9 pontos depois da decisão da Suprema Corte (mar–mai/2026), contra −40
      durante a cobrança;
    * volta a abrir em junho/2026, quando o USTR anuncia a Seção 301 — que atinge de novo máquinas,
      madeira, calçados e móveis —, e segue aberto com ela em vigor.

    Um choque que não fosse a tarifa (preço, demanda americana) dificilmente abriria, fecharia e
    reabriria exatamente nessas datas. O gráfico abaixo separa a cesta em duas: os produtos que
    ganharam o alívio de novembro (agro) e os que não ganharam.
""")
)

CELLS.append(
    code("""
    ALIVIO = pl.col("alivio_nov25")
    fit_agro = ajustar(painel_produtos(TARIFADO & ALIVIO), "Tarifados")
    fit_resto = ajustar(painel_produtos(TARIFADO & ~ALIVIO), "Tarifados")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for f, cor, rot in [(fit_agro, C_3, "aliviados em nov/25 (carne, café, frutas...)"),
                        (fit_resto, C_BRASIL, "não aliviados (máquinas, madeira, calçados, móveis...)")]:
        m = f.periods_ >= np.datetime64(dt.date(2025, 1, 1))
        ax.plot(f.periods_[m], f.gap_[m], color=cor, marker="o", ms=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=rot)
    ax.axhline(0, color=AXIS, lw=1)
    marcar_eventos(ax)
    eixo_meses(ax)
    ax.set_ylabel("gap (pontos)")
    ax.set_title("Gap por tipo de produto tarifado: o alívio de novembro aparece")
    ax.legend(loc="lower left", fontsize=8.5)
    plt.show()
""")
)

CELLS.append(
    md("""
    A série do agro é ruidosa (repare nas oscilações do 1º semestre de 2025), mas o desenho é claro:

    * os **aliviados** (carne, café, frutas) caem até −100 pontos em dezembro e saltam para acima do
      contrafactual em janeiro — o primeiro mês cheio de embarques sem os 40%;
    * os **não aliviados** seguem 30–35 pontos abaixo até março e só convergem em abril–maio, depois
      da Suprema Corte;
    * em junho/2026, quando o USTR anuncia a Seção 301 — que mira manufaturados e poupa carne, café e
      frutas —, só os não aliviados voltam a cair.

    É o padrão que as tarifas preveem, produto a produto e mês a mês — difícil de produzir por acaso.

    Para a inferência e para a conta em dólares, usamos só a janela do tarifaço (jul/25–fev/26),
    reestimando no painel truncado com os mesmos doadores:
""")
)

CELLS.append(
    code("""
    doadores_prod = list(fit_prod.weights_)
    painel_prod_ieepa = painel_prod.filter(pl.col("mes") <= FIM_IEEPA)
    fit_prod_ieepa = ajustar(painel_prod_ieepa, "Tarifados")
    ct_prod, pis_prod, loo_prod = inferencia(fit_prod_ieepa, nome="Tarifados × isentos | ")

    PERDA_PRODUTOS = gap_usd(fit_prod_ieepa, painel_prod_ieepa, "Tarifados")[~fit_prod_ieepa.pre_mask_].sum()

    # Mesma coisa só com os produtos que NÃO ganharam o alívio de novembro (sobretudo manufaturados).
    painel_sem_alivio = painel_produtos(TARIFADO & ~pl.col("alivio_nov25"), fim=FIM_IEEPA)
    fit_sem_alivio = ajustar(painel_sem_alivio, "Tarifados")
    PERDA_SEM_ALIVIO = gap_usd(fit_sem_alivio, painel_sem_alivio, "Tarifados")[~fit_sem_alivio.pre_mask_].sum()

    print()
    print(f"Perda jul/25–fev/26, cesta tarifada inteira      : {bi(PERDA_PRODUTOS)} (ATT {fit_prod_ieepa.att_:.1f} pts)")
    print(f"Perda jul/25–fev/26, tarifados sem alívio em nov : {bi(PERDA_SEM_ALIVIO)} (ATT {fit_sem_alivio.att_:.1f} pts)")
""")
)

CELLS.append(
    md("""
    A cesta tarifada vendeu cerca de **40% menos** aos EUA do que teria vendido, uma perda de
    **~US$ 4,4 bi** em 8 meses. Só os produtos que não ganharam o alívio de novembro — majoritariamente
    manufaturados — perderam ~US$ 3,3 bi.

    Sobre a inferência: o teste conformal não rejeita aqui (a reestimação sob a nula absorve três
    quartos do gap — o mesmo limite da §4.4, agravado pela volatilidade dos doadores). O placebo é mais
    útil: a cesta tarifada tem a maior razão RMSPE pós/pré entre as 13 séries, o resultado mais extremo
    possível. Com 13 unidades, isso dá p = 1/13 ≈ 0,08.

    ### 4 + 5 juntos

    | Desenho | O que mede | Perda jul/25–fev/26 |
    |---|---|---|
    | Brasil × países (§4) | todas as exportações não-combustíveis aos EUA | ~US$ 5,3 bi (−22% do contrafactual) |
    | Brasil × ele mesmo (§5) | só a cesta tarifada | ~US$ 4,4 bi (−40%) |

    Os dois desenhos têm fraquezas diferentes e chegam à mesma ordem de grandeza. Que o primeiro seja
    um pouco maior faz sentido: ele inclui os produtos `isento_aeronaves` (parte deles foi tarifada) e
    o aço semiacabado, que pesa muito mais na pauta brasileira do que na dos doadores e caiu com a
    Seção 232 a 50% — um choque que não é o tarifaço. Daqui em diante, a **perda direta nos EUA** é
    tratada como a faixa entre as duas estimativas.

    ### 5.1 Sensibilidade à janela do pré-período

    O controle sintético não tem uma regra fechada para o tamanho do pré-período: mais meses dão mais
    informação, mas só se a estrutura for estável (Abadie 2021, §3). Reestimando os dois desenhos com
    o pré-período começando em datas diferentes:
""")
)

CELLS.append(
    code("""
    inicios = [dt.date(2021, 1, 1), dt.date(2021, 7, 1), dt.date(2022, 1, 1), dt.date(2022, 7, 1),
               dt.date(2023, 1, 1), dt.date(2023, 7, 1), dt.date(2024, 1, 1)]
    linhas = []
    for ini in inicios:
        pp = preparar_painel(paises, "pais", "eua", ini, FIM_IEEPA, excluir=EXCLUIDOS)
        fp = ajustar(pp, "Brasil")
        pq = painel_produtos(TARIFADO, fim=FIM_IEEPA, inicio=ini)
        fq = ajustar(pq, "Tarifados")
        linhas.append({
            "início do pré": ini,
            "países: ATT (pts)": round(fp.att_, 1),
            "países: perda (US$ bi)": round(gap_usd(fp, pp, "Brasil")[~fp.pre_mask_].sum() / 1e9, 2),
            "países: RMSPE pré": round(fp.rmspe_pre_, 3),
            "produtos: ATT (pts)": round(fq.att_, 1),
            "produtos: perda (US$ bi)": round(gap_usd(fq, pq, "Tarifados")[~fq.pre_mask_].sum() / 1e9, 2),
            "produtos: RMSPE pré": round(fq.rmspe_pre_, 3),
        })
    sens = pl.DataFrame(linhas)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(sens["início do pré"], sens["países: perda (US$ bi)"], color=C_BRASIL, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Brasil × países (§4)")
    ax.plot(sens["início do pré"], sens["produtos: perda (US$ bi)"], color=C_SINT, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="tarifados × isentos (§5)")
    ax.axvline(INICIO, color=MUTED, lw=0.8)
    ax.text(INICIO, 0.98, " especificação principal", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("perda jul/25–fev/26 (US$ bi)")
    ax.set_xlabel("início do pré-período")
    ax.set_title("Perda estimada conforme o início do pré-período")
    ax.legend(loc="upper right", fontsize=8.5)
    plt.show()
    sens
""")
)

CELLS.append(
    md("""
    O desenho **dentro do Brasil é estável**: a perda fica entre US$ 3,5 e 4,5 bi em qualquer janela.
    O desenho **entre países não é**: com pré-períodos que começam antes de meados de 2022, a perda cai
    para US$ 1–2 bi. Nesses casos o ajuste pré-período é pior (RMSPE maior) e os pesos se concentram
    em Alemanha e Países Baixos, cujas vendas aos EUA caíram no fim de 2025 por motivos próprios — o
    tipo de doador com choque idiossincrático que a regra 3 tentava excluir. Ficamos com a janela de
    36 meses; a leitura honesta é que **a perda direta foi de US$ 4–5 bi**, com o desenho entre países
    admitindo valores bem menores se a janela incluir o superciclo de 2021–22.
""")
)

# ---------------------------------------------------------------------------
# 6. Total exports
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 6. O total sentiu?

    O Brasil fechou 2025 com exportações recorde. Se a perda nos EUA foi de US$ 4–5 bi, onde ela
    aparece no total? Mesmo desenho da §4, mesmo pool de doadores, agora para as exportações
    não-combustíveis **para o resto do mundo** (sem os EUA) e **para o mundo**:
""")
)

CELLS.append(
    code("""
    resultados_total = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (var, titulo) in zip(axes, [("resto", "Resto do mundo (sem EUA)"), ("mundo", "Mundo")]):
        p = preparar_painel(paises, "pais", var, INICIO, FIM_IEEPA, excluir=EXCLUIDOS)
        f = ajustar(p, "Brasil")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ct = conformal_test(f)
            ic = conformal_interval(f, alpha=0.10)
        pis = placebo_in_space(f)
        g = gap_usd(f, p, "Brasil")
        resultados_total[var] = dict(fit=f, painel=p, usd=g[~f.pre_mask_].sum(), p=ct.pvalue, ic=ic, placebo=pis.pvalue)
        ax.plot(f.periods_, f.actual_, color=C_BRASIL, label="Brasil")
        ax.plot(f.periods_, f.synthetic_, color=C_SINT, ls="--", label="Brasil sintético")
        ax.axvspan(ANUNCIO, f.periods_[-1], color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.set_title(titulo)
        eixo_meses(ax)
    axes[0].set_ylabel("índice (100 = média pré)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    plt.show()

    pl.DataFrame([
        {"destino": v, "ATT (pts)": round(r["fit"].att_, 1), "gap acumulado (US$ bi)": round(r["usd"] / 1e9, 2),
         "IC 90% conformal (pts)": f"[{r['ic'][0]:.1f}, {r['ic'][1]:.1f}]",
         "p conformal": round(r["p"], 3), "p placebo": round(r["placebo"], 3)}
        for v, r in resultados_total.items()
    ])
""")
)

CELLS.append(
    md("""
    Nada. O Brasil segue o seu sintético para o resto do mundo e para o mundo; o placebo põe o Brasil
    entre as unidades *menos* incomuns, e os intervalos cobrem o zero com folga (e também quedas de
    10% ou mais). Duas leituras possíveis:

    1. a perda nos EUA foi compensada por vendas em outros lugares; ou
    2. a perda existe no total, mas é pequena demais para este desenho enxergar.

    Para separar uma da outra, a pergunta certa é: **qual o menor efeito que esse desenho detectaria?**
    É exatamente o que a análise de poder do `augsynth-py` responde: injeta efeitos conhecidos no
    pré-período, em janelas que terminam em meses diferentes, reestima tudo e conta quantas vezes o
    teste conformal os detecta (`simulate_power`, no estilo do `GeoLiftPower`).
""")
)

CELLS.append(
    code("""
    pre_mundo = resultados_total["mundo"]["painel"].filter(pl.col("mes") < ANUNCIO)
    poder = simulate_power(
        pre_mundo, estimator=Synth(), unit="unidade", time="mes", outcome="y", treated="Brasil",
        durations=8, effect_sizes=(-0.30, -0.20, -0.15, -0.10, -0.05, -0.02, 0.0),
        lookback_window=12, alpha=0.10, n_jobs=-1,
    )
    curva = poder.power_curve().sort("effect_size")
    exp_total = comtrade.filter((pl.col("pais") == "Brasil") & (pl.col("destino") == "mundo")
                                & pl.col("mes").is_between(ANUNCIO, FIM_IEEPA))
    EXPORT_TOTAL = float((exp_total["total_usd"]).sum())
    perda_pct = sorted([abs(PERDA_PAISES) / EXPORT_TOTAL, abs(PERDA_PRODUTOS) / EXPORT_TOTAL])

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(-curva["effect_size"] * 100, curva["power"], color=C_BRASIL, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(0.8, color=MUTED, lw=0.8)
    ax.text(29.5, 0.82, "poder de 80%", ha="right", fontsize=8, color=INK2)
    ax.axvspan(perda_pct[0] * 100, perda_pct[1] * 100, color=C_SINT, alpha=0.18, lw=0)
    ax.text(perda_pct[1] * 100 + 0.4, 0.45, "perda nos EUA, como %\\ndas exportações totais", fontsize=8, color=INK2)
    ax.set_xlabel("queda injetada nas exportações para o mundo (%)")
    ax.set_ylabel("poder (taxa de detecção, α = 10%)")
    ax.set_title("Quanto o desenho entre países consegue enxergar no total")
    ax.set_ylim(-0.03, 1.03)
    plt.show()
    print(f"Exportações totais do Brasil jul/25–fev/26: {bi(EXPORT_TOTAL)}")
    print(f"Perda nos EUA = {perda_pct[0]:.1%} a {perda_pct[1]:.1%} das exportações totais no período")
    curva.select("effect_size", "n_simulations", "power")
""")
)

CELLS.append(
    md("""
    Mesmo quedas de 15–30% no total são detectadas em pouco mais da metade das janelas, e abaixo de
    5% o teste não detecta nada. A perda nos EUA, mesmo que **nada** tivesse sido compensado, equivale
    a ~2% das exportações totais do período — muito abaixo do que o método enxerga. Ou seja: o "nada"
    da tabela anterior **não prova** que houve compensação. Para saber para onde as vendas foram, é
    preciso descer ao microdado.

    > Essa é uma lição geral sobre controle sintético: um resultado nulo só é informativo quando o
    > efeito plausível está acima do efeito mínimo detectável. Rode a análise de poder *antes* de
    > concluir que "não houve efeito".
""")
)

# ---------------------------------------------------------------------------
# 7. Redirection
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 7. Para onde foram as vendas?

    Com o Comex Stat dá para seguir cada produto (SH4) até cada destino. Os produtos foram divididos
    em grupos pela exposição aos EUA em 2024 (`sh4_grupos.csv`):

    * **`tarifado`** — mais da metade das vendas aos EUA pagou os 40%, e os EUA eram ≥ 5% das exportações do produto (559 posições; US$ 62 bi exportados em 2024);
    * **`isento_exposto`** — tão dependente dos EUA quanto, mas isento ou na Seção 232 (petróleo, celulose, aço, aviões...);
    * **`nao_exposto`** — os EUA pesavam menos de 5% (soja, minério de ferro, milho...).

    Comparamos os 7 meses de cobrança (ago/25–fev/26) com os mesmos meses do ano anterior. Se houve
    redirecionamento, as vendas dos tarifados **para outros destinos** devem crescer mais que as dos
    grupos de controle:
""")
)

CELLS.append(
    code("""
    larga = janelas.pivot(on="janela", index=["sh4", "grupo", "destino"], values="fob_usd").fill_null(0)

    def crescimento(excluir=()):
        return (larga.filter(~pl.col("sh4").is_in(list(excluir)) & (pl.col("grupo") != "ambiguo"))
                     .group_by("grupo", "destino").agg(pl.col("antes").sum(), pl.col("tarifaco").sum())
                     .with_columns(((pl.col("tarifaco") / pl.col("antes") - 1) * 100).alias("cresc_%"))
                     .sort("destino", "grupo"))

    def excesso(tab, controle):
        t = tab.filter((pl.col("grupo") == "tarifado") & (pl.col("destino") == "outros")).row(0, named=True)
        c = tab.filter((pl.col("grupo") == controle) & (pl.col("destino") == "outros")).row(0, named=True)
        return (t["cresc_%"] - c["cresc_%"]) / 100 * t["antes"]

    cresc = crescimento()
    REDIR = {c: excesso(cresc, c) for c in ("nao_exposto", "isento_exposto")}
    print("Redirecionamento (crescimento das vendas dos tarifados para outros destinos acima do controle):")
    for c, v in REDIR.items():
        print(f"  vs {c:15s}: {bi(v)}")
    cresc.with_columns((pl.col("antes") / 1e9).round(2).alias("ago/24–fev/25 (US$ bi)"),
                       (pl.col("tarifaco") / 1e9).round(2).alias("ago/25–fev/26 (US$ bi)"),
                       pl.col("cresc_%").round(1)).select("grupo", "destino", "ago/24–fev/25 (US$ bi)",
                                                         "ago/25–fev/26 (US$ bi)", "cresc_%")
""")
)

CELLS.append(
    md("""
    As vendas dos tarifados para outros destinos cresceram ~20%, duas a três vezes o ritmo dos
    controles — um excesso de US$ 3–4 bi, da ordem da perda nos EUA. À primeira vista, compensação
    quase total. Mas de quais produtos veio esse crescimento?
""")
)

CELLS.append(
    code("""
    contrib = (larga.filter((pl.col("grupo") == "tarifado") & (pl.col("destino") == "outros"))
                    .with_columns((pl.col("tarifaco") - pl.col("antes")).alias("delta"))
                    .sort("delta", descending=True))
    total_delta = contrib["delta"].sum()
    top = contrib.head(8).with_columns(
        pl.col("sh4").replace_strict(descr_sh4, default="").str.slice(0, 50).alias("descrição"),
        (pl.col("delta") / 1e9).round(2).alias("Δ outros destinos (US$ bi)"),
        (pl.col("delta") / total_delta * 100).round(0).alias("% do Δ total"),
    ).select("sh4", "descrição", "Δ outros destinos (US$ bi)", "% do Δ total")
    print(f"Δ total das vendas dos tarifados para outros destinos: {bi(total_delta)}")
    top
""")
)

CELLS.append(
    code("""
    CARNE_CAFE = ("0201", "0202", "0901")
    cresc_sem = crescimento(excluir=CARNE_CAFE)
    REDIR_SEM = {c: excesso(cresc_sem, c) for c in ("nao_exposto", "isento_exposto")}
    print("Sem carne bovina (0201, 0202) e café (0901):")
    for c, v in REDIR_SEM.items():
        print(f"  redirecionamento vs {c:15s}: {bi(v)}")
    cresc_sem.filter(pl.col("destino") == "outros").with_columns(pl.col("cresc_%").round(1)).select("grupo", "cresc_%")
""")
)

CELLS.append(
    md("""
    Aqui está o achado central do notebook. **Carne bovina congelada e café explicam todo o
    "redirecionamento"**: tirando os dois, as vendas dos demais tarifados para outros destinos
    cresceram 8% — dentro da faixa dos controles (6–10%). O resto da cesta — máquinas, madeira,
    móveis, calçados, óleos essenciais, motores — perdeu vendas nos EUA e **não as recuperou em lugar
    nenhum**.

    E mesmo a carne e o café pedem cautela antes de chamar tudo de "redirecionamento causado pela
    tarifa":

    * a **carne** foi para a China, que comprou recordes no 2º semestre de 2025 — em parte antecipando
      as salvaguardas que Pequim impôs à carne importada em janeiro de 2026, algo que teria acontecido
      com ou sem Trump;
    * o **café** estava com preços historicamente altos, o que infla qualquer comparação em valor.

    Os US$ 3–4 bi de "redirecionamento" são, portanto, um **teto**. A parte causada pela tarifa está
    em algum lugar entre zero e esse teto.
""")
)

# ---------------------------------------------------------------------------
# 8. What-if: no new markets
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 8. E se não houvesse novos mercados?

    O governo destacou a abertura de mercados como resposta ao tarifaço; o painel do MAPA registra
    cerca de 600 aberturas sanitárias para o agro entre 2023 e 2026 (78 em 2023, 222 em 2024, 225 em
    2025). Para medir o papel dos **novos mercados**, definimos mercado como um par
    **produto (SH4) × país de destino**:

    * um par é **existente** se exportou pelo menos US$ 100 mil numa janela-base antes da data de
      referência, e **novo** caso contrário;
    * **`tarifaco`**: data de referência ago/2025, base de 24 meses — mercados que surgiram *depois*
      da sobretaxa;
    * **`placebo`**: a mesma coisa um ano antes (ago/2024), para descontar a rotatividade normal de
      novos pares;
    * **`desde_2023`**: base ago/2022–jul/2023 — todos os mercados surgidos desde o início do
      esforço de aberturas. É a leitura mais generosa de "novos mercados".

    Uma ressalva: aberturas sanitárias costumam levar de 6 a 12 meses até virar embarque
    (habilitação de plantas, registro). Mercados abertos *depois* de agosto/2025 dificilmente
    pesariam dentro da janela do tarifaço — daí a importância da referência `desde_2023`.
""")
)

CELLS.append(
    code("""
    def janela_ref(ref, lo, hi):
        return (margens.filter((pl.col("referencia") == ref) & pl.col("mes").is_between(lo, hi))
                       .group_by("grupo", "destino_tipo").agg(pl.col("fob_usd").sum())
                       .pivot(on="destino_tipo", index="grupo", values="fob_usd")
                       .select("grupo", "eua", "existente", "novo"))

    P0 = (dt.date(2024, 8, 1), dt.date(2025, 2, 1))   # ago/24–fev/25
    P1 = (dt.date(2025, 8, 1), FIM_IEEPA)              # ago/25–fev/26

    pos = janela_ref("tarifaco", *P1).with_columns(pl.lit("novos pós-tarifa (ago/25–fev/26)").alias("medida"))
    pla = janela_ref("placebo", *P0).with_columns(pl.lit("placebo: novos um ano antes (ago/24–fev/25)").alias("medida"))
    d23_0 = janela_ref("desde_2023", *P0).with_columns(pl.lit("novos desde 2023, em ago/24–fev/25").alias("medida"))
    d23_1 = janela_ref("desde_2023", *P1).with_columns(pl.lit("novos desde 2023, em ago/25–fev/26").alias("medida"))
    tab_novos = (pl.concat([pla, pos, d23_0, d23_1])
                   .filter(pl.col("grupo").is_in(["tarifado", "nao_exposto"]))
                   .with_columns((pl.col("novo") / (pl.col("novo") + pl.col("existente")) * 100).alias("% novos"),
                                 (pl.col("novo") / 1e9).alias("novos (US$ bi)"),
                                 (pl.col("existente") / 1e9).alias("existentes (US$ bi)")))
    tab_novos.select("medida", "grupo", pl.col("novos (US$ bi)").round(3), pl.col("existentes (US$ bi)").round(2),
                     pl.col("% novos").round(2)).sort("grupo", "medida")
""")
)

CELLS.append(
    code("""
    def linha(tab, medida, grupo):
        return tab.filter((pl.col("medida") == medida) & (pl.col("grupo") == grupo)).row(0, named=True)

    # (a) Mercados surgidos depois da tarifa: participação dos novos pares nas vendas
    #     não-EUA, contra o mesmo indicador um ano antes, descontada a variação do controle.
    m_pos, m_pla = "novos pós-tarifa (ago/25–fev/26)", "placebo: novos um ano antes (ago/24–fev/25)"
    d_t = linha(tab_novos, m_pos, "tarifado")["% novos"] - linha(tab_novos, m_pla, "tarifado")["% novos"]
    d_c = linha(tab_novos, m_pos, "nao_exposto")["% novos"] - linha(tab_novos, m_pla, "nao_exposto")["% novos"]
    t1 = linha(tab_novos, m_pos, "tarifado")
    NOVOS_POS = (d_t - d_c) / 100 * (t1["novo"] + t1["existente"])

    # (b) Mercados surgidos desde 2023: crescimento das vendas a esses pares acima do
    #     crescimento que os mesmos pares tiveram no grupo de controle.
    m0, m1 = "novos desde 2023, em ago/24–fev/25", "novos desde 2023, em ago/25–fev/26"
    g_c = linha(tab_novos, m1, "nao_exposto")["novo"] / linha(tab_novos, m0, "nao_exposto")["novo"] - 1
    NOVOS_2023 = linha(tab_novos, m1, "tarifado")["novo"] - linha(tab_novos, m0, "tarifado")["novo"] * (1 + g_c)
    NOVOS_2023_BRUTO = linha(tab_novos, m1, "tarifado")["novo"]

    print(f"(a) excesso em mercados surgidos após a tarifa          : {bi(NOVOS_POS)}")
    print(f"(b) excesso em mercados surgidos desde 2023             : {bi(NOVOS_2023)}")
    print(f"    vendas totais dos tarifados a esses mercados (bruto): {bi(NOVOS_2023_BRUTO)}")
    print(f"Comparação: 'redirecionamento' total (teto, §7)          : {bi(min(REDIR.values()))} a {bi(max(REDIR.values()))}")
""")
)

CELLS.append(
    code("""
    top_destinos = (novos_pares.filter((pl.col("referencia") == "desde_2023") & (pl.col("grupo") == "tarifado"))
                               .group_by("pais").agg((pl.col("fob_usd").sum() / 1e6).round(1).alias("US$ mi (ago/25–jul/26)"),
                                                     pl.len().alias("pares"))
                               .sort("US$ mi (ago/25–jul/26)", descending=True).head(10))
    top_pares = (novos_pares.filter((pl.col("referencia") == "desde_2023") & (pl.col("grupo") == "tarifado"))
                            .sort("fob_usd", descending=True).head(10)
                            .with_columns(pl.col("sh4").replace_strict(descr_sh4, default="").str.slice(0, 45).alias("produto"),
                                          (pl.col("fob_usd") / 1e6).round(1).alias("US$ mi"))
                            .select("pais", "sh4", "produto", "US$ mi"))
    print("Principais destinos 'novos desde 2023' dos produtos tarifados:")
    display(top_destinos)
    top_pares
""")
)

CELLS.append(
    md("""
    Os novos mercados existem — motores para a Argélia, centros de usinagem para o Canadá, farinha de
    carne para Paquistão e Indonésia, café para a Índia, sebo bovino e manteiga de cacau para a
    Holanda —, mas são **pequenos diante do choque**: menos de US$ 1 bi em vendas, e um excesso (o
    que cresceu além do normal) de US$ 0,1–0,2 bi, contra uma perda de US$ 4–5 bi nos EUA.

    ### 8.1 Os cenários

    Juntando as peças (todos os valores para os 8 meses do tarifaço):

    * **Perda direta nos EUA** (*L*): §4 e §5.
    * **Efeito líquido observado** sobre o total: estimado na §6 — próximo de zero, com intervalo largo.
    * **Redirecionamento** implícito: o que fecha a conta, *R = efeito líquido + L*.
    * **Novos mercados** (*N*): o excesso medido acima.

    | Cenário | Efeito líquido nas exportações |
    |---|---|
    | A. Observado | efeito líquido estimado (§6) |
    | B. Sem os mercados surgidos após a tarifa | A − *N* (pós-tarifa) |
    | C. Sem os mercados abertos desde 2023 | A − *N* (desde 2023) |
    | D. Sem nenhum redirecionamento | −*L* |
""")
)

CELLS.append(
    code("""
    L = sorted([abs(PERDA_PAISES), abs(PERDA_PRODUTOS)])          # faixa da perda direta
    LIQ = resultados_total["mundo"]["usd"]                          # efeito líquido observado (§6)
    cenarios = pl.DataFrame([
        {"cenário": "A. Observado (com redirecionamento)", "baixo": LIQ, "alto": LIQ},
        {"cenário": "B. Sem mercados surgidos após a tarifa", "baixo": LIQ - max(NOVOS_POS, 0), "alto": LIQ - max(NOVOS_POS, 0)},
        {"cenário": "C. Sem mercados abertos desde 2023", "baixo": LIQ - NOVOS_2023_BRUTO, "alto": LIQ - max(NOVOS_2023, 0)},
        {"cenário": "D. Sem nenhum redirecionamento", "baixo": -L[1], "alto": -L[0]},
    ]).with_columns(
        (pl.col("baixo") / 1e9).round(2).alias("pior (US$ bi)"),
        (pl.col("alto") / 1e9).round(2).alias("melhor (US$ bi)"),
        (pl.col("baixo") / EXPORT_TOTAL * 100).round(2).alias("pior (% export.)"),
        (pl.col("alto") / EXPORT_TOTAL * 100).round(2).alias("melhor (% export.)"),
    )

    def brl(x):
        return f"{x:.1f}".replace(".", ",").replace("-", "−")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    nomes = cenarios["cenário"].to_list()
    for i, r in enumerate(cenarios.iter_rows(named=True)):
        y = len(cenarios) - 1 - i
        lo, hi = sorted([r["baixo"] / 1e9, r["alto"] / 1e9])
        if hi - lo < 0.05:
            ax.plot([lo], [y], "o", color=C_BRASIL, ms=9, markeredgecolor=SURFACE, markeredgewidth=2)
            rot = brl(lo)
        else:
            ax.plot([lo, hi], [y, y], color=C_BRASIL, lw=8, solid_capstyle="round")
            rot = f"{brl(hi)} a {brl(lo)}"
        ax.text(hi + 0.15, y, rot, va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(nomes))[::-1], nomes, fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(-6.0, 1.2)
    ax.set_xlabel("efeito líquido nas exportações brasileiras, jul/25–fev/26 (US$ bi)")
    ax.set_title("O tarifaço em quatro cenários")
    ax.grid(axis="y", visible=False)
    plt.show()
    print(f"Faixa da perda direta nos EUA (L): {bi(L[0])} a {bi(L[1])}")
    print(f"Redirecionamento implícito (R = líquido + L): {bi(LIQ + L[0])} a {bi(LIQ + L[1])}")
    cenarios.select("cenário", "pior (US$ bi)", "melhor (US$ bi)", "pior (% export.)", "melhor (% export.)")
""")
)

CELLS.append(
    md("""
    **A resposta à pergunta do "e se".** Sem os novos mercados, o impacto do tarifaço teria sido
    praticamente o mesmo — algumas dezenas a poucas centenas de milhões de dólares a mais de perda.
    Mesmo na leitura mais generosa (apagar *todas* as vendas dos tarifados a mercados surgidos desde
    2023), a perda extra fica abaixo de US$ 1 bi.

    O que fez diferença foi o redirecionamento para **parceiros que já existiam**, quase todo em
    **carne e café**. Sem esse redirecionamento (cenário D), o Brasil teria perdido **US$ 4–5 bi em
    8 meses, ~2% de tudo o que exportou no período** — pouco para o total do país, mas concentrado
    em setores e regiões específicos.

    O redirecionamento implícito na conta (R ≈ US$ 3,8–4,7 bi) se sobrepõe ao teto medido no
    microdado (US$ 3,0–4,2 bi, §7): duas peças estimadas de jeitos independentes chegam ao mesmo
    lugar. Se o teto do microdado estiver certo, o efeito líquido verdadeiro é um pouco mais negativo
    que o de A — mais um motivo para ler A com o intervalo largo da §6.

    Duas observações para não tirar a conclusão errada:

    * Isso **não** quer dizer que a abertura de mercados é inútil: ela opera em horizonte de anos, e
      o tarifaço durou 7 meses. A pergunta aqui é estreita — quanto ela amorteceu *este* choque.
    * O efeito líquido observado (A) tem intervalo largo (§6). Os cenários são contas sobre
      estimativas pontuais; a incerteza de cada peça se soma.
""")
)

# ---------------------------------------------------------------------------
# 9. Limitations
# ---------------------------------------------------------------------------
CELLS.append(
    md("""
    ## 9. Limitações

    1. **Doadores também foram tratados.** Em 2025 todos os países passaram a pagar tarifas nos EUA. O
       que medimos é o efeito dos 40% *adicionais* do Brasil em relação a países que pagavam 10–20% —
       que é a pergunta certa, mas depende de os doadores não terem tido choques próprios grandes
       (regra 3).
    2. **Spillover.** As regras 2 e 3 são escolhas. A §4.5 mostra que elas importam muito; com outra
       lista de concorrentes, a perda entre países muda de escala.
    3. **Valores, não volumes.** Preços de café, carne e petróleo mexeram muito no período. Um estudo
       com quilogramas (o Comex Stat tem `metricKG`) separaria preço de quantidade.
    4. **Classificação no nível SH6.** As linhas "Ex" e "apenas aeronaves" não podem ser separadas sem
       dados de 10 dígitos; ~4 p.p. de diferença em relação ao MDIC.
    5. **Inferência limitada.** O teste conformal perde poder quando a reestimação sob a nula absorve o
       efeito (§4.4, §5, §6), e o placebo dentro do Brasil tem só 13 unidades (p mínimo ≈ 0,08). A
       evidência mais forte é a **concordância** entre os dois desenhos e a **sincronia** do gap com
       as datas de política (entrada em vigor, alívio agro, Suprema Corte, Seção 301).
    6. **Janela do pré-período.** O desenho entre países é sensível a ela (§5.1): começando antes de
       meados de 2022, a perda cai para US$ 1–2 bi.
    7. **Novos mercados ≠ política de abertura.** Um par produto–destino novo pode vir de uma abertura
       sanitária do MAPA, de uma missão da ApexBrasil ou da iniciativa de uma empresa. O exercício mede
       o canal "novos mercados" como um todo — um teto para o papel da política.
    8. **Dados preliminares.** Os dados mensais do Comtrade e do Comex Stat são revisados; os meses mais
       recentes podem mudar.
""")
)

# ---------------------------------------------------------------------------
# 10. Conclusions
# ---------------------------------------------------------------------------
CELLS.append(
    code("""
    resumo = pl.DataFrame([
        {"quantidade": "Perda direta nos EUA — Brasil × países (§4)", "valor": bi(PERDA_PAISES)},
        {"quantidade": "Perda direta nos EUA — tarifados × isentos (§5)", "valor": bi(PERDA_PRODUTOS)},
        {"quantidade": "   ...dos quais tarifados sem alívio em nov (§5)", "valor": bi(PERDA_SEM_ALIVIO)},
        {"quantidade": "Efeito líquido sobre as exportações para o mundo (§6)", "valor": bi(LIQ)},
        {"quantidade": "Redirecionamento observado, teto (§7)", "valor": f"{bi(min(REDIR.values()))} a {bi(max(REDIR.values()))}"},
        {"quantidade": "   ...sem carne e café (§7)", "valor": f"{bi(min(REDIR_SEM.values()))} a {bi(max(REDIR_SEM.values()))}"},
        {"quantidade": "Novos mercados surgidos após a tarifa, excesso (§8)", "valor": bi(NOVOS_POS)},
        {"quantidade": "Novos mercados desde 2023, excesso (§8)", "valor": bi(NOVOS_2023)},
        {"quantidade": "Exportações totais do Brasil no período", "valor": bi(EXPORT_TOTAL)},
    ])
    resumo
""")
)

CELLS.append(
    md("""
    ## 10. Conclusões

    1. **O tarifaço machucou as vendas aos EUA.** A cesta tarifada vendeu ~40% menos do que teria
       vendido, e as exportações não-combustíveis aos EUA ficaram ~22% abaixo do contrafactual: uma
       perda de **US$ 4–5 bi em 8 meses** (jul/25–fev/26). O desenho dentro do Brasil dá US$ 3,5–4,5 bi
       em qualquer janela; o entre países cai para US$ 1–2 bi só com pré-períodos iniciados antes de
       meados de 2022. O gap abre na entrada em vigor, fecha para o agro depois do alívio de novembro,
       fecha para os manufaturados depois da Suprema Corte e reabre para eles com a Seção 301.
    2. **O total não sentiu — mas o método também não sentiria.** As exportações para o mundo seguem o
       contrafactual; a análise de poder mostra que nem quedas de 15–30% seriam detectadas com
       segurança, e a perda nos EUA equivale a ~2% do total.
    3. **A compensação veio de carne e café para parceiros antigos.** Tirando esses dois produtos, os
       demais tarifados — sobretudo manufaturados — perderam nos EUA e não recuperaram em lugar nenhum.
       E parte do avanço da carne (China) teria acontecido de qualquer forma.
    4. **Sem novos mercados, o impacto teria sido quase o mesmo.** Mercados novos renderam US$ 0,1–0,2
       bi de vendas além do normal (e menos de US$ 1 bi no total). **Sem redirecionamento algum, a perda líquida
       teria sido de US$ 4–5 bi** (~2% das exportações do período), concentrada na indústria.
    5. **Sobre o método.** O pool de doadores e a janela pré-período mudam o resultado: concorrentes
       que ganham com a tarifa do tratado inflam o efeito, e doadores com choques próprios o diluem.
       Resultados nulos precisam de análise de poder antes de virar conclusão. E dois desenhos com
       fraquezas diferentes valem mais que um desenho com p-valor bonito.

    ### Para ir além

    * Refazer em **quilogramas** para separar preço de quantidade.
    * Usar o tratamento da **Seção 301** (jul/2026) como uma segunda "rodada" do mesmo experimento.
    * Olhar a dimensão **regional** (o Comex Stat tem UF): o choque concentrou-se em SC, SP, ES, MG, RS e CE.
    * Ligar os novos pares produto–destino ao **painel de aberturas do MAPA**, para separar abertura
      sanitária de iniciativa privada.

    ## Referências

    * Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic Control Methods for Comparative Case Studies. *JASA*, 105(490).
    * Abadie, A. (2021). Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects. *JEL*, 59(2).
    * Ben-Michael, E., Feller, A. & Rothstein, J. (2021). The Augmented Synthetic Control Method. *JASA*, 116(536).
    * Chernozhukov, V., Wüthrich, K. & Zhu, Y. (2021). An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls. *JASA*, 116(536).
    * Executive Order 14323 (30/07/2025), 90 FR 37739; Executive Order 14361 (20/11/2025).
    * MDIC — Nota sobre a Ordem Executiva dos EUA (31/07/2025); Alcance das Medidas Tarifárias dos Estados Unidos sobre Exportações Brasileiras (24/07/2026).
    * UN Comtrade (comtradeapi.un.org) e Comex Stat (comexstat.mdic.gov.br), dados consultados em out/2026.
""")
)


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"Wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
