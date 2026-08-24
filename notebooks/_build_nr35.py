"""Build `04_nr35_trabalho_em_altura.ipynb` from a programmatic spec.

Run this to (re)generate the notebook:

    .venv/bin/python notebooks/_build_nr35.py

The notebook itself is the artifact users open and edit. This script is the
source of truth used to (re)create it cleanly when the design evolves — handy
when iterating on multiple acts without diff-noise from cell metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NOTEBOOK_PATH = Path(__file__).parent / "04_nr35_trabalho_em_altura.ipynb"


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
# Cell content
# ---------------------------------------------------------------------------

CELLS: list[dict] = []

# --- Header ----------------------------------------------------------------
CELLS.append(
    md("""
    # NR-35 (trabalho em altura) e a mortalidade na construção civil

    > **Desenho.** Controle sintético (clássico e aumentado por ridge) sobre o painel
    > anual de mortalidade por acidente de trabalho do AEAT, com a **construção** como
    > unidade tratada e as **demais divisões CNAE** como pool de doadores.

    A NR-35 — Norma Regulamentadora de **Trabalho em Altura** — é a resposta normativa
    brasileira à principal causa isolada de morte na construção civil: a **queda de
    altura**. A norma impõe análise de risco, permissão de trabalho, sistemas de
    ancoragem e capacitação obrigatória para qualquer atividade acima de 2 m.

    A pergunta deste notebook: **a entrada em vigor da NR-35 reduziu a taxa de
    mortalidade por acidente de trabalho na construção, relativamente ao que se
    esperaria sem a norma?** O contrafactual "sem a norma" é construído como uma
    combinação ponderada das demais divisões da CNAE — setores expostos ao mesmo
    ambiente macro (ciclo econômico, formalização, fiscalização geral), mas não ao
    tratamento específico da NR-35.

    ## Roteiro

    1. **Ato 1 — O relógio institucional.** Publicação ≠ vigência: por que T₀ = 2013.
    2. **Ato 2 — Os dados.** Painel AEAT de óbitos e taxa de mortalidade por divisão CNAE.
    3. **Ato 3 — SCM clássico** (`progfunc = "none"`): a construção sintética.
    4. **Ato 4 — AugSynth** (`progfunc = "ridge"`): o que a augmentação compra com 5 anos de pré.
    5. **Ato 5 — É real?** Inferência conformal (CWZ 2021) + placebos in-space.
    6. **Ato 6 — T₀ importa.** Vigência (2013) vs publicação (2012), e o ano híbrido.
    7. **Ato 7 — Robustez do pool.** Doadores com exposição própria a trabalho em altura.
    8. **Ato 8 — Que efeito este desenho detecta?** MDE via `simulate_power` (v0.4).
    9. **Ato 9 — Conclusões e limitações.**
""")
)

# --- Setup cell ------------------------------------------------------------
CELLS.append(
    code("""
    # Imports e configuração visual.
    from __future__ import annotations

    import sys
    import warnings
    from pathlib import Path

    # Torna `augsynth_py` importável rodando o notebook de qualquer cwd.
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

    from augsynth_py import AugSynth, Synth, conformal_interval, conformal_pvalue, simulate_power

    # Paleta — um destaque + cinzas. Sem rainbow.
    COLOR_TREATED   = "#D7263D"  # vermelho — construção
    COLOR_SYNTH     = "#1B4965"  # azul escuro — contrafactual SCM
    COLOR_AUGMENTED = "#7A5195"  # roxo — contrafactual AugSynth
    COLOR_DONOR     = "#B0B0B0"  # cinza — doadores
    COLOR_GRID      = "#EAEAEA"
    COLOR_TEXT      = "#333333"
    COLOR_ALT       = "#F18F01"  # laranja — cenários alternativos (T0=2012 etc.)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.edgecolor": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "legend.frameon": False,
    })
""")
)

# --- Ato 1 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 1 — O relógio institucional: publicação ≠ vigência

    A NR-35 foi aprovada pela **Portaria SIT n.º 313, de 23/03/2012** (DOU de
    27/03/2012). Mas a portaria **não entrou em vigor na publicação**: o seu art. 2.º
    escalonou a vigência em duas ondas contadas da publicação —

    | Marco | Data | O que passa a valer |
    |---|---|---|
    | Publicação (DOU) | 27/03/2012 | nada ainda — prazo começa a correr |
    | Vigência geral (6 meses) | 27/09/2012 | corpo da norma (análise de risco, PT, EPI, ancoragem) |
    | Vigência plena (12 meses) | 27/03/2013 | itens de capacitação (treinamento obrigatório) |

    Com dados **anuais**, isso significa:

    - **2012 é um ano híbrido** — 9 meses sem norma nenhuma, 3 meses de vigência
      parcial. Não é pré limpo nem pós limpo.
    - **2013 é o primeiro ano-calendário inteiramente sob a norma** (e, a partir de
      março, sob a norma *completa*, capacitação inclusa).

    Por isso o desenho principal usa **T₀ = 2013** — a vigência, não a publicação.
    O Ato 6 quantifica o quanto o deslocamento de um ano muda o resultado, e trata
    o ano híbrido de 2012 explicitamente.

    ### A outra ponta da janela: o NTEP

    O início do painel também é uma escolha institucional. Em **abril de 2007** entrou
    em operação o **NTEP** (Nexo Técnico Epidemiológico Previdenciário, Lei
    11.430/2006 + Decreto 6.042/2007), que passou a reconhecer acidentes/doenças de
    trabalho **sem CAT registrada** — um salto de nível em toda a série de acidentes
    do AEAT. Começar em **2008** deixa a quebra fora da janela, ao custo de um
    pré-período de apenas **5 anos (2008–2012)**. Esse pré curto é a fragilidade
    central do desenho — e é exatamente onde a augmentação por ridge (Ato 4) deixa
    de ser decorativa.

    Fechamos em **2019** para não misturar o choque da COVID-19 no pós-período.
""")
)

CELLS.append(
    code("""
    # Constantes do desenho.
    TREATED       = "Construção"   # seção F (divisões CNAE 41+42+43 agregadas)
    T0_VIGENCIA   = 2013           # primeiro ano-calendário sob vigência
    T0_PUBLICACAO = 2012           # ano da publicação (cenário alternativo, Ato 6)
    ANO_MIN       = 2008           # pós-NTEP (e já sob CNAE 2.0 no AEAT)
    ANO_MAX       = 2019           # pré-COVID

    UNIT, TIME, OUT = "setor", "ano", "taxa_mortalidade"
    OUT_LABEL = "Óbitos por 100 mil vínculos"
""")
)

# --- Ato 2 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 2 — Os dados

    O painel vem do **AEAT** (Anuário Estatístico de Acidentes do Trabalho, MPS/MTE):
    óbitos por acidente de trabalho e vínculos empregatícios por **divisão CNAE 2.0**
    e ano. A taxa de mortalidade é `óbitos / vínculos × 100.000` — a métrica que o
    próprio AEAT publica no capítulo de taxas.

    Três decisões de preparo (ver `_data/README-aeat.md` para a proveniência):

    - As divisões da construção (**41 Construção de edifícios, 42 Obras de
      infraestrutura, 43 Serviços especializados**) são agregadas em uma única
      unidade tratada, somando óbitos e vínculos antes de calcular a taxa —
      média ponderada, não média de taxas.
    - Doadores são as demais divisões CNAE com vínculos suficientes para uma taxa
      estável (piso de vínculos documentado no preparo); divisões minúsculas geram
      taxas erráticas (0 ou 2 óbitos num ano dobram a taxa) e só adicionariam ruído
      ao pool.
    - O desfecho é a **taxa**, não a contagem: sem normalizar por vínculos, o ciclo
      de emprego da própria construção (que despenca depois de 2014) se disfarçaria
      de efeito da norma.
""")
)

CELLS.append(
    code("""
    panel = (
        pl.read_csv(DATA_DIR / "aeat_nr35_panel.csv")
        .filter(pl.col("ano").is_between(ANO_MIN, ANO_MAX))
        .sort([UNIT, TIME])
    )

    n_units = panel[UNIT].n_unique()
    n_years = panel[TIME].n_unique()
    assert panel.height == n_units * n_years, "painel desbalanceado"

    print(f"unidades      : {n_units} (1 tratada + {n_units - 1} doadoras)")
    print(f"anos          : {panel[TIME].min()}–{panel[TIME].max()} ({n_years})")
    print(f"pré-período   : {ANO_MIN}–{T0_VIGENCIA - 1} ({T0_VIGENCIA - ANO_MIN} anos)")
    print(f"pós-período   : {T0_VIGENCIA}–{ANO_MAX} ({ANO_MAX - T0_VIGENCIA + 1} anos)")
    panel.filter(pl.col(UNIT) == TREATED)
""")
)

CELLS.append(
    code("""
    # Trajetórias: construção em vermelho, doadores em cinza.
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for unit, group in panel.partition_by(UNIT, as_dict=True).items():
        is_treated = unit[0] == TREATED
        ax.plot(
            group[TIME], group[OUT],
            color=COLOR_TREATED if is_treated else COLOR_DONOR,
            lw=2.4 if is_treated else 0.9,
            alpha=1.0 if is_treated else 0.5,
            zorder=3 if is_treated else 1,
            label=TREATED if is_treated else None,
        )
    ax.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.text(T0_VIGENCIA - 0.4, ax.get_ylim()[1] * 0.98, "  vigência NR-35",
            color=COLOR_TEXT, fontsize=10, va="top")
    ax.set_title("Taxa de mortalidade por acidente de trabalho — construção vs demais divisões CNAE")
    ax.set_xlabel("Ano"); ax.set_ylabel(OUT_LABEL)
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    md("""
    > **Leitura.** A construção não é um setor mediano: a taxa dela roda bem acima da
    > maioria das divisões. Isso importa metodologicamente — se a tratada estiver
    > perto (ou fora) da borda do envelope convexo dos doadores, o SCM clássico não
    > consegue ajustar o pré-período, e é aí que a augmentação por ridge tem função
    > real, não decorativa.
""")
)

# --- Ato 3 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 3 — SCM clássico (`progfunc = "none"`)

    O estimador de v0.1: pesos $w$ no simplex ($w \\ge 0$, $\\sum_j w_j = 1$) que
    minimizam o erro quadrático do pré-período,

    $$
    w^* \\;=\\; \\arg\\min_{w \\ge 0,\\ \\sum_j w_j = 1}\\;
    \\big\\|\\, y_1^{\\text{pre}} \\,-\\, Y_0^{\\text{pre}}\\, w \\,\\big\\|^2,
    $$

    equivalente ao `augsynth(progfunc = "None", fixedeff = TRUE)` do R. O
    `fixedeff=True` desconta o nível médio pré de cada unidade — necessário aqui,
    porque a construção tem *nível* de taxa que quase nenhum doador alcança; o que
    pedimos aos doadores é que reproduzam a *trajetória*.

    Com **5 anos de pré e dezenas de doadores**, atenção ao risco simétrico: o QP
    tem graus de liberdade de sobra e pode interpolar o pré exatamente mesmo sem
    nenhum doador individualmente parecido (overfitting de interpolação). RMSPE pré
    ≈ 0 aqui é sinal de alerta, não de sucesso.
""")
)

CELLS.append(
    code("""
    scm = Synth().fit(
        panel,
        unit=UNIT, time=TIME, outcome=OUT,
        treated=TREATED, treatment_time=T0_VIGENCIA,
    )

    print(f"ATT (média pós {T0_VIGENCIA}–{ANO_MAX})  : {scm.att_:+.3f} óbitos/100 mil vínculos")
    print(f"ATT relativo ao nível pré           : {scm.att_pct_ * 100:+.1f}%")
    print(f"RMSPE pré (fração do nível pré)     : {scm.rmspe_pre_ * 100:.2f}%")
    print(f"L2 imbalance (escalonado vs 1/J)    : {scm.scaled_l2_imbalance_:.3f}")
    print(f"# doadores com peso > 1%            : {sum(1 for w in scm.weights_.values() if w > 0.01)}")
""")
)

CELLS.append(
    code("""
    # Quem compõe a construção sintética?
    sorted_w = sorted(scm.weights_.items(), key=lambda x: -x[1])
    top = [(u, w) for u, w in sorted_w if w > 0.01][:12]
    others = sum(w for _, w in sorted_w) - sum(w for _, w in top)
    labels = [u for u, _ in top] + (["(demais)"] if others > 1e-4 else [])
    values = [w for _, w in top] + ([others] if others > 1e-4 else [])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.5))
    ax.barh(labels, values, color=COLOR_SYNTH)
    for i, v in enumerate(values):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", color=COLOR_TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel("Peso"); ax.set_title("Pesos SCM — a construção sintética")
    plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    code("""
    # Real vs sintética + gap.
    years = scm.periods_.astype(int)
    fig, (ax, axg) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True, height_ratios=[3, 2]
    )

    ax.axvspan(years[0], T0_VIGENCIA - 0.5, color=COLOR_GRID, alpha=0.5, zorder=0)
    ax.plot(years, scm.actual_, color=COLOR_TREATED, lw=2.4, label="Construção (real)")
    ax.plot(years, scm.synthetic_, color=COLOR_SYNTH, lw=2.0, ls="--", label="Construção sintética (SCM)")
    ax.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.set_ylabel(OUT_LABEL)
    ax.set_title("Real vs sintética — SCM clássico, T₀ = vigência (2013)")
    ax.legend(loc="best")

    axg.plot(years, scm.gap_, color=COLOR_TREATED, lw=2.0)
    axg.fill_between(years, 0, scm.gap_, where=(years >= T0_VIGENCIA),
                     color=COLOR_TREATED, alpha=0.25)
    axg.axhline(0, color=COLOR_TEXT, lw=0.7)
    axg.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    axg.set_xlabel("Ano"); axg.set_ylabel("Gap (real − sintética)")
    axg.set_title(f"ATT = {scm.att_:+.3f} óbitos/100 mil ({scm.att_pct_ * 100:+.1f}% do nível pré)")
    plt.tight_layout(); plt.show()
""")
)

# --- Ato 4 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 4 — AugSynth (`progfunc = "ridge"`): a augmentação com função real

    A augmentação de Ben-Michael, Feller & Rothstein (2021) ajusta uma ridge sobre o
    resíduo pré do SCM e corrige o contrafactual com ela; os pesos efetivos viram
    $\\omega + \\gamma$, podem ser negativos e extrapolam para fora do envelope
    convexo — com viés controlado pelo $\\lambda$ escolhido por validação cruzada
    leave-one-out sobre os anos do pré.

    É aqui que o pré curto vira argumento de artigo em vez de nota de rodapé:

    - Se o SCM **não** fecha o pré (tratada fora do envelope), a ridge fecha — e a
      comparação de RMSPE pré `none` × `ridge` mede quanto.
    - Se o SCM fecha o pré *bem demais* (interpolação com 5 pontos), o LOO-CV da
      ridge é o único freio explícito de overfitting no pipeline — o $\\lambda$
      escolhido diz se a correção generaliza fora do ano deixado de fora.

    Com $T_0 = 5$, o LOO-CV tem só 5 dobras: a curva de CV abaixo merece ser olhada
    ponto a ponto, não tratada como caixa-preta.
""")
)

CELLS.append(
    code("""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="CV-selected lambda")
        aug = AugSynth().fit(
            panel,
            unit=UNIT, time=TIME, outcome=OUT,
            treated=TREATED, treatment_time=T0_VIGENCIA,
        )

    grid = aug.lambda_cv_path_[:, 0]
    losses = aug.lambda_cv_path_[:, 1]
    best_idx = int(losses.argmin())

    print(f"λ escolhido por LOO-CV              : {aug.lambda_:.4g} (posição {best_idx}/{len(grid) - 1} da grade)")
    print(f"ATT AugSynth                        : {aug.att_:+.3f} óbitos/100 mil ({aug.att_pct_ * 100:+.1f}%)")
    print(f"RMSPE pré  SCM  (progfunc='none')   : {scm.rmspe_pre_ * 100:.2f}%")
    print(f"RMSPE pré  AugSynth ('ridge')       : {aug.rmspe_pre_ * 100:.2f}%")
    print(f"L2 imbalance escalonado SCM         : {scm.scaled_l2_imbalance_:.3f}")
    print(f"L2 imbalance escalonado AugSynth    : {aug.scaled_l2_imbalance_:.3f}")
    n_neg = sum(1 for w in aug.weights_.values() if w < -1e-6)
    print(f"# doadores com peso efetivo negativo: {n_neg}")
""")
)

CELLS.append(
    code("""
    # O contraste central do notebook: none vs ridge, contrafactual e correção.
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, height_ratios=[3, 2])

    ax.axvspan(years[0], T0_VIGENCIA - 0.5, color=COLOR_GRID, alpha=0.5, zorder=0)
    ax.plot(years, aug.actual_, color=COLOR_TREATED, lw=2.4, label="Construção (real)")
    ax.plot(years, scm.synthetic_, color=COLOR_SYNTH, lw=1.8, ls="--", label="SCM (progfunc='none')")
    ax.plot(years, aug.synthetic_, color=COLOR_AUGMENTED, lw=1.8, label="AugSynth (progfunc='ridge')")
    ax.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.set_ylabel(OUT_LABEL)
    ax.set_title("Contrafactual SCM vs AugSynth")
    ax.legend(loc="best")

    axr.axhline(0, color=COLOR_TEXT, lw=0.8)
    axr.fill_between(years, 0, aug.ridge_correction_, color=COLOR_AUGMENTED, alpha=0.35)
    axr.plot(years, aug.ridge_correction_, color=COLOR_AUGMENTED, lw=1.5)
    axr.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    axr.set_xlabel("Ano"); axr.set_ylabel("Correção ridge")
    axr.set_title("O que a ridge adiciona ao SCM, ano a ano")
    plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    code("""
    # Curva de LOO-CV do lambda — com T0=5, olhar ponto a ponto.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.semilogx(grid, losses, color=COLOR_AUGMENTED, marker="o", markersize=3, lw=1.2)
    ax.axvline(aug.lambda_, color=COLOR_SYNTH, ls="--", lw=1.0,
               label=f"λ escolhido = {aug.lambda_:.3g}")
    ax.set_xlabel("λ (escala log)"); ax.set_ylabel("Perda LOO-CV")
    ax.set_title("Caminho de validação cruzada do λ (5 dobras)")
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()

    if best_idx in (0, len(grid) - 1):
        print("⚠️  λ na borda da grade — alargar lambda_grid antes de confiar no CV.")
""")
)

CELLS.append(
    code("""
    # Pesos efetivos (ω+γ) vs pesos simplex (ω): quem a ridge passou a (des)confiar.
    eff = aug.weights_
    scm_w = aug.scm_weights_
    order = sorted(eff, key=lambda u: -abs(eff[u]))[:15]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - 0.2, [scm_w[u] for u in order], width=0.4, color=COLOR_SYNTH, label="ω (SCM, simplex)")
    ax.bar(x + 0.2, [eff[u] for u in order], width=0.4, color=COLOR_AUGMENTED, label="ω+γ (AugSynth)")
    ax.axhline(0, color=COLOR_TEXT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peso")
    ax.set_title("15 maiores |pesos efetivos|: simplex vs ridge (BFR 2021, §2.4)")
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()
""")
)

CELLS.append(
    md("""
    > **O contraste que interessa para o artigo.** A tabela-resumo do Ato 9 fecha os
    > números, mas a mecânica está acima: com o pré curto, `none` × `ridge` não é uma
    > escolha estética — é a diferença entre um contrafactual preso ao envelope dos
    > doadores e um que extrapola com penalização explícita, validada fora da amostra
    > (ainda que numa amostra de 5).
""")
)

# --- Ato 5 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 5 — É real? Conformal (CWZ 2021) + placebos in-space

    Dois instrumentos independentes:

    1. **Inferência conformal** (Chernozhukov, Wüthrich & Zhu 2021): testa
       $H_0\\colon$ efeito pós constante $= h_0$ refazendo o ajuste sob o nulo e
       permutando resíduos. Com $T = 12$ períodos, a permutação em **bloco** (a
       recomendada para séries) só tem 12 rotações distintas — o p-valor é
       quantizado em múltiplos de $1/12 \\approx 0{,}083$ e **nunca chega a 0,05**.
       Reportamos block e iid lado a lado por isso; a versão iid assume
       intercambialidade mais forte.
    2. **Placebos in-space** (Abadie et al. 2010): reajustar o SCM fingindo que cada
       doador foi tratado em 2013 e comparar o gap da construção com a distribuição
       placebo, via razão RMSPE pós/pré.
""")
)

CELLS.append(
    code("""
    rng = np.random.default_rng(2013)

    for nome, est in (("SCM", scm), ("AugSynth", aug)):
        p_block = conformal_pvalue(est, permutation_type="block")
        p_iid = conformal_pvalue(est, permutation_type="iid", ns=2000, rng=rng)
        lo, hi = conformal_interval(est, permutation_type="block", grid_size=60)
        print(f"{nome:8s}: p(block) = {p_block:.3f} | p(iid) = {p_iid:.3f} | "
              f"IC 95% (block) = [{lo:+.2f}, {hi:+.2f}] óbitos/100 mil")
""")
)

CELLS.append(
    code("""
    # Placebos in-space: refit em cada doador; razão RMSPE pós/pré (Abadie 2010).
    def rmspe_ratio(est) -> float:
        pre, post = est.pre_mask_, ~est.pre_mask_
        rmspe = lambda m: float(np.sqrt(np.mean(est.gap_[m] ** 2)))
        return rmspe(post) / max(rmspe(pre), 1e-12)

    donors = [u for u in panel[UNIT].unique().sort().to_list() if u != TREATED]
    placebo_fits = {}
    for u in donors:
        placebo_fits[u] = Synth().fit(
            panel.filter(pl.col(UNIT) != TREATED),  # tratada fora do pool placebo
            unit=UNIT, time=TIME, outcome=OUT,
            treated=u, treatment_time=T0_VIGENCIA,
        )

    ratios = {u: rmspe_ratio(f) for u, f in placebo_fits.items()}
    ratio_treated = rmspe_ratio(scm)
    rank = 1 + sum(1 for r in ratios.values() if r >= ratio_treated)
    p_placebo = rank / (len(ratios) + 1)
    print(f"Razão RMSPE pós/pré da construção : {ratio_treated:.2f}")
    print(f"Posição entre {len(ratios) + 1} unidades       : {rank}ª")
    print(f"p-valor placebo (in-space)        : {p_placebo:.3f}")
""")
)

CELLS.append(
    code("""
    # Gaps placebo (cinza) vs gap da construção (vermelho).
    fig, ax = plt.subplots(figsize=(11, 5))
    for u, f in placebo_fits.items():
        ax.plot(f.periods_.astype(int), f.gap_, color=COLOR_DONOR, lw=0.8, alpha=0.5)
    ax.plot(years, scm.gap_, color=COLOR_TREATED, lw=2.4, label="Construção")
    ax.axhline(0, color=COLOR_TEXT, lw=0.7)
    ax.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.set_title("Gaps placebo (cada doador tratado em 2013) vs gap observado da construção")
    ax.set_xlabel("Ano"); ax.set_ylabel("Gap (real − sintética)")
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()
""")
)

# --- Ato 6 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 6 — T₀ importa: vigência (2013) vs publicação (2012)

    A tentação de datar o tratamento na **publicação** (março/2012) tem duas
    consequências mecânicas: o pré encolhe de 5 para 4 anos, e o ano híbrido de 2012
    — 9 meses ainda sem norma — entra no pós, diluindo qualquer efeito real.
    No desenho principal, o híbrido fica no *pré*, o que também não é neutro: se a
    vigência parcial de out–dez/2012 já reduziu óbitos, o contrafactual absorve
    parte do efeito e o ATT estimado fica **conservador** (viés contra encontrar
    efeito).

    Três cenários, mesmo pipeline:

    - **A (principal)** — T₀ = 2013, 2012 no pré.
    - **B (publicação)** — T₀ = 2012.
    - **C (híbrido fora)** — T₀ = 2013, ano de 2012 removido do painel.
""")
)

CELLS.append(
    code("""
    cenarios = {
        "A: T0=2013 (vigência)": (panel, T0_VIGENCIA),
        "B: T0=2012 (publicação)": (panel, T0_PUBLICACAO),
        "C: T0=2013, sem 2012": (panel.filter(pl.col(TIME) != 2012), T0_VIGENCIA),
    }

    linhas = []
    fits_cen = {}
    for nome, (pnl, t0) in cenarios.items():
        f_scm = Synth().fit(pnl, unit=UNIT, time=TIME, outcome=OUT,
                            treated=TREATED, treatment_time=t0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="CV-selected lambda")
            f_aug = AugSynth().fit(pnl, unit=UNIT, time=TIME, outcome=OUT,
                                   treated=TREATED, treatment_time=t0)
        fits_cen[nome] = (f_scm, f_aug)
        linhas.append({
            "cenário": nome,
            "anos pré": int(sum(f_scm.pre_mask_)),
            "ATT SCM": round(f_scm.att_, 3),
            "ATT AugSynth": round(f_aug.att_, 3),
            "RMSPE pré SCM (%)": round(f_scm.rmspe_pre_ * 100, 2),
            "RMSPE pré Aug (%)": round(f_aug.rmspe_pre_ * 100, 2),
            "p conformal (block, Aug)": round(conformal_pvalue(f_aug, permutation_type="block"), 3),
        })
    pl.DataFrame(linhas)
""")
)

CELLS.append(
    code("""
    # Os três gaps AugSynth sobrepostos.
    fig, ax = plt.subplots(figsize=(11, 5))
    estilos = {
        "A: T0=2013 (vigência)": (COLOR_AUGMENTED, "-", 2.4),
        "B: T0=2012 (publicação)": (COLOR_ALT, "--", 1.8),
        "C: T0=2013, sem 2012": (COLOR_SYNTH, ":", 1.8),
    }
    for nome, (_, f_aug) in fits_cen.items():
        cor, ls, lw = estilos[nome]
        ax.plot(f_aug.periods_.astype(int), f_aug.gap_, color=cor, ls=ls, lw=lw,
                label=f"{nome} (ATT {f_aug.att_:+.2f})")
    ax.axhline(0, color=COLOR_TEXT, lw=0.7)
    ax.axvline(T0_VIGENCIA - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.set_title("O deslocamento de um ano muda o quê? Gaps AugSynth nos três cenários")
    ax.set_xlabel("Ano"); ax.set_ylabel("Gap (real − sintética)")
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()
""")
)

# --- Ato 7 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 7 — Robustez do pool: doadores com exposição própria a altura

    SUTVA aqui exige que os doadores **não tenham sido tratados** pela NR-35. A norma
    vale para *qualquer* setor com trabalho acima de 2 m — o que contamina doadores
    com exposição relevante: **eletricidade** (linhas e redes), **telecomunicações**
    (torres) e **transporte/armazenagem** (carga em altura). Se a NR-35 também
    reduziu óbitos nesses doadores, o contrafactual cai junto com a tratada e o ATT
    estimado é **subestimado** (viés contra o efeito).

    Refazemos SCM e AugSynth sem esses doadores. Aproveitamos e rodamos o
    leave-one-out clássico: derrubar o doador de maior peso e ver se o ATT se move.
""")
)

CELLS.append(
    code("""
    # Divisões com exposição própria relevante a trabalho em altura.
    EXPOSTOS = [u for u in donors if any(
        chave in u.lower() for chave in ("eletricidade", "telecomunica", "transporte")
    )]
    print("Doadores removidos por exposição à NR-35:", EXPOSTOS or "(nenhum encontrado)")

    panel_limpo = panel.filter(~pl.col(UNIT).is_in(EXPOSTOS))
    scm_limpo = Synth().fit(panel_limpo, unit=UNIT, time=TIME, outcome=OUT,
                            treated=TREATED, treatment_time=T0_VIGENCIA)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="CV-selected lambda")
        aug_limpo = AugSynth().fit(panel_limpo, unit=UNIT, time=TIME, outcome=OUT,
                                   treated=TREATED, treatment_time=T0_VIGENCIA)

    # Leave-one-out do doador de maior peso efetivo.
    top_donor = max(aug.weights_, key=lambda u: abs(aug.weights_[u]))
    panel_loo = panel.filter(pl.col(UNIT) != top_donor)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="CV-selected lambda")
        aug_loo = AugSynth().fit(panel_loo, unit=UNIT, time=TIME, outcome=OUT,
                                 treated=TREATED, treatment_time=T0_VIGENCIA)

    print(f"ATT AugSynth — pool completo        : {aug.att_:+.3f}")
    print(f"ATT AugSynth — sem expostos         : {aug_limpo.att_:+.3f}")
    print(f"ATT SCM      — sem expostos         : {scm_limpo.att_:+.3f}")
    print(f"ATT AugSynth — sem '{top_donor}' (maior |peso|): {aug_loo.att_:+.3f}")
""")
)

# --- Ato 8 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 8 — Que efeito este desenho conseguiria detectar? (MDE, v0.4)

    Antes de interpretar qualquer p-valor, a pergunta de desenho: **com 5 anos de
    pré e inferência conformal, que tamanho de efeito este painel detecta?**
    `simulate_power` responde por simulação: trunca o painel no fim do pré
    (2012), injeta efeitos multiplicativos na construção em janelas
    placebo-no-tempo e mede a taxa de detecção.

    Duas ressalvas de leitura: dentro da janela de simulação o pré efetivo encolhe
    ainda mais (3–4 anos), e usamos permutação iid porque com $T = 5$ a block só
    tem 5 rotações (p mínimo 0,2 — poder zero por construção em α = 0,10).
    O resultado é a fotografia honesta do que 5 anos de pré compram.
""")
)

CELLS.append(
    code("""
    panel_pre = panel.filter(pl.col(TIME) < T0_VIGENCIA)
    pwr = simulate_power(
        panel_pre,
        estimator=Synth(),
        unit=UNIT, time=TIME, outcome=OUT,
        treated=TREATED,
        durations=2,
        effect_sizes=(0.0, -0.05, -0.10, -0.15, -0.20, -0.30, -0.40),
        effect_type="multiplicative",
        lookback_window=2,
        alpha=0.10,
        permutation_type="iid",
        side="left",
        ns=1000,
        rng=np.random.default_rng(35),
        on_error="record",
    )
    curva = pwr.power_curve()
    mde = pwr.mde(target_power=0.8)
    print(curva)
    print(f"\\nMDE (poder ≥ 0,8, α = 0,10): "
          f"{'não alcançado na grade simulada' if mde is None else f'{mde:+.0%} na taxa de mortalidade'}")
""")
)

CELLS.append(
    code("""
    # Curva de poder.
    cd = curva.filter(pl.col("effect_size") != 0).sort("effect_size")
    fp = curva.filter(pl.col("effect_size") == 0)["power"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot([abs(e) for e in cd["effect_size"]], cd["power"],
            color=COLOR_SYNTH, marker="o", lw=1.8)
    ax.axhline(0.8, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.6)
    ax.text(0.005, 0.815, "poder alvo 0,8", color=COLOR_TEXT, fontsize=9)
    if len(fp) and fp[0] is not None:
        ax.axhline(fp[0], color=COLOR_ALT, ls=":", lw=1.2)
        ax.text(0.005, float(fp[0]) + 0.015, f"taxa de falso positivo = {fp[0]:.2f}",
                color=COLOR_ALT, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Redução injetada na taxa de mortalidade (|efeito multiplicativo|)")
    ax.set_ylabel("Poder (taxa de detecção)")
    ax.set_title("Poder do desenho com 5 anos de pré (conformal iid, α = 0,10)")
    plt.tight_layout(); plt.show()
""")
)

# --- Ato 9 -----------------------------------------------------------------
CELLS.append(
    md("""
    ## Ato 9 — Conclusões e limitações

    A tabela abaixo fecha os números dos atos anteriores em um só lugar.
""")
)

CELLS.append(
    code("""
    resumo = pl.DataFrame([
        {"estimador": "SCM (progfunc='none')", "ATT": round(scm.att_, 3),
         "ATT %": round(scm.att_pct_ * 100, 1), "RMSPE pré %": round(scm.rmspe_pre_ * 100, 2),
         "p conformal (block)": round(conformal_pvalue(scm, permutation_type="block"), 3)},
        {"estimador": "AugSynth (progfunc='ridge')", "ATT": round(aug.att_, 3),
         "ATT %": round(aug.att_pct_ * 100, 1), "RMSPE pré %": round(aug.rmspe_pre_ * 100, 2),
         "p conformal (block)": round(conformal_pvalue(aug, permutation_type="block"), 3)},
    ])
    resumo
""")
)

CELLS.append(
    md("""
    ### O que este desenho sustenta — e o que não sustenta

    **Sustenta.**

    - A escolha de **T₀ = vigência (2013)**, documentada no relógio institucional do
      Ato 1 e estressada no Ato 6 — publicação vs vigência não é detalhe: muda o
      pré, muda o pós e muda a leitura do ano híbrido de 2012.
    - O contraste **`none` × `ridge`** com pré curto como objeto de interesse por si
      só (Ato 4): quanto do ajuste pré vem do envelope convexo e quanto vem da
      extrapolação penalizada.
    - Inferência que respeita o tamanho do painel: p-valor conformal em bloco
      quantizado em 1/12, placebos in-space e a curva de poder do Ato 8 dizem
      *antes* o que o desenho pode afirmar.

    **Não sustenta (limitações honestas).**

    - **5 anos de pré** — a restrição que organiza o notebook inteiro. Alongar para
      trás de 2008 cruza a quebra do NTEP (e a transição CNAE 1.0→2.0 no AEAT).
    - **Desfecho agregado**: mortalidade por *todas* as causas de acidente, não só
      quedas de altura. O AEAT público não cruza CID × CNAE nesse nível; um recorte
      por quedas (CID W10–W19) exigiria microdados. Efeito da NR-35 diluído ⇒ viés
      conservador.
    - **Cobertura previdenciária**: o AEAT enxerga o mercado formal
      celetista (e, na taxa, o denominador de vínculos) — a informalidade da
      construção fica fora, e é plausivelmente onde a norma menos pega.
    - **Contaminação de doadores** (Ato 7) e o ano híbrido no pré (Ato 6) puxam o
      ATT na direção conservadora; nenhum dos dois inverte sinal nas checagens.

    ### Referências

    - Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic Control Methods
      for Comparative Case Studies*. JASA.
    - Ben-Michael, E., Feller, A., & Rothstein, J. (2021). *The Augmented Synthetic
      Control Method*. JASA.
    - Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). *An Exact and Robust
      Conformal Inference Method for Counterfactual and Synthetic Controls*. JASA.
    - Brasil, MTE. **Portaria SIT n.º 313, de 23/03/2012** (NR-35 — Trabalho em
      Altura), DOU 27/03/2012.
    - AEAT — Anuário Estatístico de Acidentes do Trabalho (vários anos).
""")
)

# ---------------------------------------------------------------------------

NOTEBOOK = {
    "cells": CELLS,
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


if __name__ == "__main__":
    NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False))
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    print(f"Wrote {NOTEBOOK_PATH} — {len(CELLS)} cells ({n_code} code, {n_md} markdown).")
