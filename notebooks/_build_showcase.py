"""Build `01_showcase_sp_campaign.ipynb` from a programmatic spec.

Run this once to (re)generate the notebook:

    .venv/bin/python notebooks/_build_showcase.py

The notebook itself is the artifact users open and edit. This script is the
source of truth used to (re)create it cleanly when the design evolves — handy
when iterating on multiple acts without diff-noise from cell metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NOTEBOOK_PATH = Path(__file__).parent / "01_showcase_sp_campaign.ipynb"


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
CELLS.append(md("""
    # `augsynth-py` — Showcase

    > **Status.** Este notebook usa **R** (`Synth`, `augsynth`) via `rpy2` como motor de cálculo.
    > A versão nativa em Python ainda está sendo construída — veja o **Ato 6** para o roadmap de paridade.

    ## Roteiro

    1. **Ato 1 — A pergunta de negócio.** Lançamos uma campanha em SP. Funcionou?
    2. **Ato 2 — Por que comparações ingênuas falham.** DiD não basta.
    3. **Ato 3 — Synthetic Control clássico.** Construindo uma "SP sintética".
    4. **Ato 4 — AugSynth.** O que fazer quando o pré-período é difícil de ajustar.
    5. **Ato 5 — É real?** Inferência via placebo + conformal.
    6. **Ato 6 — Migração pra Python.** O que falta no `augsynth-py`.

    Os dados são sintéticos, com efeito conhecido (`true ATT = +8%`), pra você poder *ver* se o método recuperou o efeito.
"""))

# --- Setup cell ------------------------------------------------------------
CELLS.append(code("""
    # Imports e configuração visual.
    from __future__ import annotations

    import json
    import sys
    from pathlib import Path

    # Localiza notebooks/_r_oracles.py e o data dir independentemente de onde o kernel iniciou.
    _here = Path.cwd()
    for cand in (_here, _here / "notebooks", _here.parent / "notebooks", _here.parent):
        if (cand / "_r_oracles.py").exists():
            sys.path.insert(0, str(cand))
            DATA_DIR = cand / "_data"
            break
    else:
        raise RuntimeError(f"_r_oracles.py não encontrado a partir de {_here}")

    import numpy as np
    import polars as pl
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    from _r_oracles import (
        make_panel,
        naive_did,
        fit_scm_r,
        fit_augsynth_r,
        placebo_test,
        CITIES_BR,
        CITY_COORDS,
    )

    # Paleta — um destaque + cinzas. Sem rainbow.
    COLOR_TREATED = "#D7263D"   # vermelho coral pra SP
    COLOR_SYNTH   = "#1B4965"   # azul escuro pra contrafactual
    COLOR_DONOR   = "#B0B0B0"   # cinza claro pros donors
    COLOR_GRID    = "#EAEAEA"
    COLOR_TEXT    = "#333333"
    COLOR_AUG     = "#F18F01"   # laranja pro AugSynth

    mpl.rcParams.update({
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
    })

    def fmt_pct(x: float) -> str:
        return f"{x*100:+.2f}%"

    def fmt_brl(x: float) -> str:
        return f"R$ {x/1_000:.0f}k"

    def draw_brazil_basemap(ax, color="#F5F4EE", edgecolor="#D4D4D4", lw=0.5):
        # Desenha o contorno simplificado dos estados do BR como pano de fundo.
        geo = json.loads((DATA_DIR / "brazil_states.geojson").read_text())
        patches = []
        for feat in geo["features"]:
            for poly in feat["geometry"]["coordinates"]:
                for ring in poly:
                    patches.append(MplPolygon(ring, closed=True))
        ax.add_collection(PatchCollection(
            patches, facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=1,
        ))
        ax.set_aspect("equal")
        ax.set_xlim(-75, -32)
        ax.set_ylim(-34, 6)
        ax.axis("off")
"""))

# --- Ato 1 -----------------------------------------------------------------
CELLS.append(md("""
    ## Ato 1 — A pergunta de negócio

    No dia 60, o time de marketing lançou uma campanha de **influencer marketing apenas em São Paulo**.
    No mês seguinte, as vendas em SP subiram bem mais do que vinham subindo. O time vibra.

    Mas: **foi a campanha, ou seria assim de qualquer jeito?** Vendas crescem por mil motivos —
    sazonalidade, tendência orgânica, choques regionais. Antes de atribuir o lift à campanha,
    precisamos de um **contrafactual**: o que teria acontecido em SP *sem* a campanha?
"""))

CELLS.append(code("""
    # Gera o painel sintético: 27 capitais BR × 90 dias, com ATT verdadeiro de +8% em SP a partir do dia 60.
    panel = make_panel(seed=42)

    print(f"Painel: {panel.height:,} linhas, {panel['city'].n_unique()} cidades, {panel['day'].n_unique()} dias")
    print(f"Cidade tratada: {panel.filter(pl.col('is_treated_unit'))['city'][0]}")
    print(f"Dia do tratamento: {panel.filter(pl.col('post'))['day'].min()}")
    panel.head(5)
"""))

CELLS.append(code("""
    # Plot 1 — Todas as cidades normalizadas (índice 100 = média do pré-período de cada cidade).
    # Sem normalizar, SP (R$ 1M/dia) eclipsa o resto e o gráfico vira espaguete.

    TREATMENT_DAY = 60

    def normalize(panel: pl.DataFrame) -> pl.DataFrame:
        pre_mean = (
            panel.filter(pl.col("day") < TREATMENT_DAY)
            .group_by("city").agg(pre=pl.col("sales").mean())
        )
        return panel.join(pre_mean, on="city").with_columns(
            sales_idx=100 * pl.col("sales") / pl.col("pre"),
        )

    panel_n = normalize(panel)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for city, group in panel_n.partition_by("city", as_dict=True).items():
        is_sp = city[0] == "São Paulo"
        ax.plot(
            group["day"], group["sales_idx"],
            color=COLOR_TREATED if is_sp else COLOR_DONOR,
            lw=2.2 if is_sp else 0.8,
            alpha=1.0 if is_sp else 0.45,
            zorder=3 if is_sp else 1,
            label="São Paulo" if is_sp else None,
        )

    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.text(TREATMENT_DAY + 0.6, ax.get_ylim()[1] * 0.985, " Lançamento da campanha",
            color=COLOR_TEXT, fontsize=10, va="top")

    sp_post_idx = panel_n.filter((pl.col("city")=="São Paulo") & (pl.col("day")>=TREATMENT_DAY))["sales_idx"].mean()
    ax.annotate(
        f"São Paulo no pós-período: índice ≈ {sp_post_idx:.0f}",
        xy=(85, sp_post_idx), xytext=(63, sp_post_idx + 6),
        color=COLOR_TREATED, fontsize=10,
        arrowprops=dict(arrowstyle="->", color=COLOR_TREATED, lw=1.0),
    )

    ax.set_title("Vendas diárias por cidade (índice 100 = média pré-tratamento)")
    ax.set_xlabel("Dia"); ax.set_ylabel("Índice (pré = 100)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Plot 2 — SP vs MÉDIA das outras cidades.
    # Spoiler: a média também sobe (sazonalidade + trend orgânico). A questão é: SP subiu MAIS por causa
    # da campanha, ou os outros estão indo bem também?

    sp = panel_n.filter(pl.col("city")=="São Paulo").sort("day")
    donors_avg = (
        panel_n.filter(pl.col("city")!="São Paulo")
        .group_by("day").agg(sales_idx=pl.col("sales_idx").mean())
        .sort("day")
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(sp["day"], sp["sales_idx"], color=COLOR_TREATED, lw=2.5, label="São Paulo")
    ax.plot(donors_avg["day"], donors_avg["sales_idx"], color=COLOR_DONOR, lw=2.0,
            ls="-.", label="Média das demais 26 capitais")

    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.fill_between(sp["day"], sp["sales_idx"], donors_avg["sales_idx"],
                    where=(sp["day"] >= TREATMENT_DAY).to_numpy(),
                    color=COLOR_TREATED, alpha=0.10, label="Gap pós-tratamento")

    ax.set_title("SP vs média do resto — qual fração do gap é a campanha?")
    ax.set_xlabel("Dia"); ax.set_ylabel("Índice (pré = 100)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""
    > **Takeaway.** A média também subiu — então parte do crescimento de SP é "ambiente", não a campanha.
    > Precisamos isolar o efeito causal. É aí que entra o synthetic control.
"""))

# --- Ato 2 -----------------------------------------------------------------
CELLS.append(md("""
    ## Ato 2 — Por que comparações ingênuas falham

    A primeira tentação é fazer **diff-in-diff** clássico: comparar SP com a média dos outros estados,
    antes vs depois.

    O problema: DiD assume **trends paralelos no pré-período**. Se SP já vinha crescendo num ritmo
    diferente das outras cidades, parte do "efeito" estimado é só o drift pré-existente.
"""))

CELLS.append(code("""
    did = naive_did(panel, treated_city="São Paulo", treatment_day=TREATMENT_DAY)
    print(f"DiD ATT (absoluto): R$ {did['att_abs']:,.0f}/dia")
    print(f"DiD ATT (% sobre baseline): {fmt_pct(did['att_pct'])}")
    print(f"True ATT injetado:           +8.00%")
    print(f"Diferença DiD vs verdadeiro: {fmt_pct(did['att_pct'] - 0.08)}")
"""))

CELLS.append(code("""
    # Diagnóstico visual: as trends foram paralelas no pré-período?

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                              gridspec_kw=dict(height_ratios=[2.2, 1]))

    ax = axes[0]
    ax.plot(sp["day"], sp["sales_idx"], color=COLOR_TREATED, lw=2.5, label="São Paulo")
    ax.plot(donors_avg["day"], donors_avg["sales_idx"], color=COLOR_DONOR, lw=2.0, ls="-.",
            label="Média dos donors")

    pre_mask = sp["day"] < TREATMENT_DAY
    ax.fill_between(sp["day"], sp["sales_idx"], donors_avg["sales_idx"],
                    where=pre_mask.to_numpy(),
                    color=COLOR_TREATED, alpha=0.13,
                    hatch="///", edgecolor=COLOR_TREATED, lw=0.0,
                    label="Gap pré-tratamento (deveria ser ~0)")
    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.set_title("Trends paralelas? Não exatamente.")
    ax.set_ylabel("Índice (pré = 100)")
    ax.legend(loc="upper left", frameon=False)

    # Painel inferior: o gap em si
    ax2 = axes[1]
    gap = sp["sales_idx"].to_numpy() - donors_avg["sales_idx"].to_numpy()
    ax2.fill_between(sp["day"], 0, gap, where=(sp["day"] < TREATMENT_DAY).to_numpy(),
                     color=COLOR_TREATED, alpha=0.25)
    ax2.fill_between(sp["day"], 0, gap, where=(sp["day"] >= TREATMENT_DAY).to_numpy(),
                     color=COLOR_TREATED, alpha=0.55)
    ax2.axhline(0, color=COLOR_TEXT, lw=0.7)
    ax2.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax2.set_xlabel("Dia"); ax2.set_ylabel("Gap (SP − média)")
    ax2.set_title("O gap já era positivo antes do dia 60 — DiD vai superestimar")
    plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""
    > **Takeaway.** O gap pré-tratamento não é zero — SP já estava acima da média antes da campanha.
    > DiD pega esse drift e atribui à campanha. Precisamos de um contrafactual que **case com SP no pré-período por construção**.
"""))

# --- Ato 3 -----------------------------------------------------------------
CELLS.append(md("""
    ## Ato 3 — Synthetic Control clássico (Abadie 2010)

    A ideia: em vez de comparar SP com a *média* dos donors, encontre uma **combinação ponderada** dos donors
    cujo trajeto pré-tratamento *imite* SP. Esse "SP sintético" vira o contrafactual.

    Restrições do SCM clássico:
    - Pesos não-negativos (`w ≥ 0`).
    - Soma dos pesos = 1.

    Por que assim: a **combinação convexa** força o sintético a ficar dentro do "envelope" dos donors —
    sem extrapolar pra fora dos dados observados.

    > Aqui chamamos `augsynth(progfunc='None', scm=TRUE, fixedeff=TRUE)` em R via `rpy2`.
    > Em `augsynth-py` v0.1, isso será `Synth().fit(panel, treated="São Paulo", treatment_day=60)`.
"""))

CELLS.append(code("""
    scm = fit_scm_r(panel, treated_city="São Paulo", treatment_day=TREATMENT_DAY)

    print(f"Soma dos pesos: {sum(scm.weights.values()):.3f}")
    print(f"Pesos não-negativos? {min(scm.weights.values()):.3f} ≥ 0")
    print(f"RMSPE pré-tratamento: {scm.rmspe_pre*100:.2f}% do nível médio")
    print(f"ATT estimado (médio):   {fmt_pct(scm.att_avg_pct)} (true = +8.00%)")
"""))

CELLS.append(code("""
    # Plot 1 — Quem é a SP sintética? Top 10 doadores por peso.

    sorted_w = sorted(scm.weights.items(), key=lambda x: -x[1])
    top = sorted_w[:10]
    others_w = sum(w for _, w in sorted_w[10:])
    labels = [c for c, _ in top] + ["Outras"]
    values = [w for _, w in top] + [others_w]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(labels, values, color=[COLOR_SYNTH if v > 0.01 else COLOR_DONOR for v in values])
    for bar, v in zip(bars, values):
        if v > 0.005:
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", color=COLOR_TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Peso"); ax.set_title("Pesos do SCM clássico — quem compõe a SP sintética")
    ax.grid(axis="y", alpha=0.0)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Mesmos pesos, agora no mapa — útil pra ver se a SP sintética tem alguma estrutura geográfica
    # (com fixed effects, o SCM escolhe por similaridade de TRAJETÓRIA após demean, não por proximidade).

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_brazil_basemap(ax)

    for city, (lat, lon) in CITY_COORDS.items():
        is_treated = city == "São Paulo"
        weight = scm.weights.get(city, 0.0)
        if is_treated:
            ax.scatter(lon, lat, s=320, color=COLOR_TREATED,
                        edgecolor="white", linewidth=2.5, zorder=10)
            ax.annotate("São Paulo", (lon, lat), xytext=(9, 9),
                        textcoords="offset points",
                        color=COLOR_TREATED, fontsize=11, fontweight="bold", zorder=11)
        elif weight > 0.01:
            size = 80 + weight * 1500
            ax.scatter(lon, lat, s=size, color=COLOR_SYNTH, alpha=0.85,
                        edgecolor="white", linewidth=1.2, zorder=8)
            ax.annotate(f"{city}\\nw={weight:.2f}", (lon, lat), xytext=(8, -4),
                        textcoords="offset points",
                        fontsize=9, color=COLOR_SYNTH, zorder=9)
        else:
            ax.scatter(lon, lat, s=22, color=COLOR_DONOR, alpha=0.55,
                        edgecolor="white", linewidth=0.4, zorder=4)

    ax.set_title("Pesos do SCM no mapa — onde mora a 'SP sintética'?",
                  fontsize=12, pad=12, color=COLOR_TEXT)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Plot 2 — SP real vs SP sintética

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.arange(len(scm.actual)), scm.actual, color=COLOR_TREATED, lw=2.4, label="SP real")
    ax.plot(np.arange(len(scm.synthetic)), scm.synthetic, color=COLOR_SYNTH, lw=2.0, ls="--",
            label="SP sintética (SCM)")

    ax.axvspan(0, TREATMENT_DAY, color=COLOR_GRID, alpha=0.5, zorder=0)
    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)

    ax.text(TREATMENT_DAY/2, ax.get_ylim()[1]*0.97, "Pré (calibração)",
            color=COLOR_TEXT, ha="center", va="top", fontsize=10)
    ax.text(TREATMENT_DAY + (90-TREATMENT_DAY)/2, ax.get_ylim()[1]*0.97, "Pós (efeito)",
            color=COLOR_TEXT, ha="center", va="top", fontsize=10)

    ax.text(2, ax.get_ylim()[0]*1.02, f"RMSPE pré = {scm.rmspe_pre*100:.2f}%",
            color=COLOR_TEXT, fontsize=10, va="bottom",
            bbox=dict(facecolor="white", edgecolor=COLOR_GRID, boxstyle="round,pad=0.3"))

    ax.set_title("São Paulo real vs sintético — o ajuste do pré-período é a moeda de credibilidade do SCM")
    ax.set_xlabel("Dia"); ax.set_ylabel("Vendas (R$)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Plot 3 — O gap é o efeito.

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(np.arange(len(scm.gap)), scm.gap, color=COLOR_TREATED, lw=2.0)
    ax.fill_between(np.arange(len(scm.gap)), 0, scm.gap,
                    where=np.arange(len(scm.gap)) >= TREATMENT_DAY,
                    color=COLOR_TREATED, alpha=0.30)
    ax.axhline(0, color=COLOR_TEXT, lw=0.7)
    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)

    ax.annotate(
        f"ATT médio pós = {fmt_pct(scm.att_avg_pct)}\\n(R$ {scm.att_avg:,.0f}/dia)",
        xy=(75, scm.gap[75]), xytext=(45, scm.gap[75:].max()*1.15),
        fontsize=11, color=COLOR_TEXT,
        bbox=dict(facecolor="white", edgecolor=COLOR_TREATED, boxstyle="round,pad=0.4"),
        arrowprops=dict(arrowstyle="->", color=COLOR_TREATED, lw=1.0),
    )

    ax.set_title("Gap (real − sintético) — o efeito da campanha")
    ax.set_xlabel("Dia"); ax.set_ylabel("Gap (R$/dia)")
    plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""
    > **Takeaway.** O SCM clássico recuperou um ATT próximo dos +8% verdadeiros
    > (com pequeno viés residual por não absorver totalmente o trend heterogêneo).
    > A "SP sintética" não é uma média — é uma combinação cirúrgica que reproduz
    > a trajetória pré-período de SP a partir de poucos donors com perfil parecido.
"""))

# --- Ato 4 -----------------------------------------------------------------
CELLS.append(md("""
    ## Ato 4 — AugSynth (Ben-Michael, Feller & Rothstein 2021)

    O SCM clássico tem um ponto cego: a restrição de **combinação convexa** assume que
    o tratado fica *dentro* do envelope dos donors. E quando não fica?

    Cenário típico: o tratado tem um **trend mais íngreme** ou um **nível** que nenhum donor
    atinge — por exemplo, SP cresce mais rápido que qualquer outra capital. A combinação convexa
    fica *abaixo* da SP real no pré-período → RMSPE pré ruim → viés contaminando o pós.

    A solução do AugSynth: pegue o resíduo do SCM e **corrija com uma regressão ridge** dos outcomes
    pré-tratamento. O resultado é uma combinação ponderada dos donors **mais** uma correção de viés
    que pode usar pesos negativos pra extrapolar.

    Em fórmula (Ben-Michael 2021, eq. 2.4): `Y_synth_aug = Y_synth_SCM + ridge(Y_pre_treated − Y_pre_synth)`.

    > Aqui chamamos `augsynth(progfunc='Ridge', scm=TRUE, fixedeff=TRUE)` em R.
    > Em `augsynth-py` v0.1: `AugSynth().fit(panel, treated="São Paulo", treatment_day=60)`.
"""))

CELLS.append(code("""
    # Painel "difícil": SP recebe um boost de trend que a coloca FORA do convex hull dos donors.
    panel_hard = make_panel(seed=42, treated_trend_boost=0.0025)

    scm_h = fit_scm_r(panel_hard, treated_city="São Paulo", treatment_day=TREATMENT_DAY)
    aug_h = fit_augsynth_r(panel_hard, treated_city="São Paulo", treatment_day=TREATMENT_DAY,
                           conformal=True)

    print(f"Cenário com SP fora do convex hull dos donors:")
    print(f"  SCM      : RMSPE pré = {scm_h.rmspe_pre*100:5.2f}%   ATT = {fmt_pct(scm_h.att_avg_pct)}   "
          f"#pesos não-zero = {sum(1 for w in scm_h.weights.values() if abs(w)>1e-3)}/{len(scm_h.weights)}")
    print(f"  AugSynth : RMSPE pré = {aug_h.rmspe_pre*100:5.2f}%   ATT = {fmt_pct(aug_h.att_avg_pct)}   "
          f"peso mín = {min(aug_h.weights.values()):+.3f} (pode ser negativo)")
"""))

CELLS.append(code("""
    # Plot 1 — Ajuste pré-período: SCM vs AugSynth lado a lado.

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    for ax, fit, title, color in [
        (axes[0], scm_h, "SCM clássico", COLOR_SYNTH),
        (axes[1], aug_h, "AugSynth (ridge)", COLOR_AUG),
    ]:
        ax.plot(np.arange(len(fit.actual)), fit.actual, color=COLOR_TREATED, lw=2.3, label="SP real")
        ax.plot(np.arange(len(fit.synthetic)), fit.synthetic, color=color, lw=2.0, ls="--",
                label=f"SP sintética ({fit.method})")
        ax.axvspan(0, TREATMENT_DAY, color=COLOR_GRID, alpha=0.5, zorder=0)
        ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title(f"{title} — RMSPE pré = {fit.rmspe_pre*100:.2f}%")
        ax.set_xlabel("Dia")
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Vendas (R$)")
    fig.suptitle("Quando SP está fora do convex hull, AugSynth ajusta melhor o pré-período",
                 fontsize=13, fontweight="semibold", y=1.02)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Plot 2 — Pesos: SCM vs AugSynth (note os negativos no AugSynth).

    common = sorted(set(scm_h.weights) | set(aug_h.weights),
                    key=lambda c: -(abs(scm_h.weights.get(c, 0)) + abs(aug_h.weights.get(c, 0))))[:12]
    scm_vals = [scm_h.weights.get(c, 0) for c in common]
    aug_vals = [aug_h.weights.get(c, 0) for c in common]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(common))
    bar_h = 0.38
    ax.barh(y - bar_h/2, scm_vals, height=bar_h, color=COLOR_SYNTH, label="SCM")
    ax.barh(y + bar_h/2, aug_vals, height=bar_h, color=COLOR_AUG, label="AugSynth")
    ax.axvline(0, color=COLOR_TEXT, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(common); ax.invert_yaxis()
    ax.set_xlabel("Peso")
    ax.set_title("Pesos: AugSynth permite negativos (extrapolação) — SCM não")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.0)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Plot 3 — Os dois gaps sobrepostos.

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.arange(len(scm_h.gap)), scm_h.gap, color=COLOR_SYNTH, lw=2.0,
            label=f"SCM        — ATT médio = {fmt_pct(scm_h.att_avg_pct)}")
    ax.plot(np.arange(len(aug_h.gap)), aug_h.gap, color=COLOR_AUG, lw=2.0,
            label=f"AugSynth — ATT médio = {fmt_pct(aug_h.att_avg_pct)}")
    ax.axhline(0, color=COLOR_TEXT, lw=0.7)
    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.fill_between(np.arange(len(scm_h.gap)), 0, scm_h.gap,
                    where=np.arange(len(scm_h.gap)) >= TREATMENT_DAY,
                    color=COLOR_SYNTH, alpha=0.10)
    ax.set_title("Gap pós-tratamento: AugSynth corrige parte do viés do SCM")
    ax.set_xlabel("Dia"); ax.set_ylabel("Gap (R$/dia)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(md("""
    > **Takeaway.** Quando o pré-período é difícil de ajustar (tratado fora do envelope dos donors),
    > o SCM clássico fica enviesado e os pesos colapsam em poucos donors. O AugSynth usa ridge pra
    > **corrigir o resíduo pré-período**, atinge melhor RMSPE, e produz ATT mais próximo do verdadeiro.
"""))

# --- Ato 5 -----------------------------------------------------------------
CELLS.append(md("""
    ## Ato 5 — É real? Inferência

    Estimar o efeito é metade do trabalho. **Quão confiantes** estamos de que ele não é ruído?

    Duas técnicas complementares:

    **5a. Placebo permutation** (Abadie). Re-rodamos o SCM tratando *cada donor* como se fosse o tratado.
    Isso gera uma distribuição de "efeitos placebo" sob a hipótese nula. Se o ATT de SP for extremo
    nessa distribuição, é evidência contra `H₀: efeito zero`.

    **5b. Conformal inference** (Chernozhukov, Wuthrich & Zhu 2021). Constrói intervalos de confiança
    *por período* sem assumir distribuição — disponível nativamente em `augsynth(inference="conformal")`.
"""))

CELLS.append(code("""
    # 5a. Placebo permutation — ⚠️ leva ~3 min, refita SCM 26x (uma por donor).

    placebo_res = placebo_test(panel, treated_city="São Paulo",
                                treatment_day=TREATMENT_DAY, fitter=fit_scm_r)
    print(f"Placebos retidos (RMSPE pré ≤ 5× SP): {len(placebo_res['placebos'])}/26")
    print(f"P-valor empírico (rank de |ATT_SP| entre placebos): {placebo_res['p_value']:.3f}")
"""))

CELLS.append(code("""
    # Plot — distribuição de gaps placebos vs SP

    fig, ax = plt.subplots(figsize=(11, 5.4))
    days = np.arange(len(placebo_res["treated"].gap))

    for city, fit in placebo_res["placebos"].items():
        ax.plot(days, fit.gap, color=COLOR_DONOR, lw=0.8, alpha=0.55)

    ax.plot(days, placebo_res["treated"].gap, color=COLOR_TREATED, lw=2.5, label="São Paulo (tratado)")

    ax.axhline(0, color=COLOR_TEXT, lw=0.7)
    ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
    ax.set_title("Gap de SP vs gaps placebos — quão extremo é SP?")
    ax.set_xlabel("Dia"); ax.set_ylabel("Gap (R$/dia)")
    ax.legend(handles=[
        plt.Line2D([], [], color=COLOR_TREATED, lw=2.5, label="São Paulo"),
        plt.Line2D([], [], color=COLOR_DONOR, lw=1.0, alpha=0.7,
                   label=f"Placebos ({len(placebo_res['placebos'])} cidades)"),
    ], loc="upper left", frameon=False)

    ax.text(0.98, 0.04,
            f"p-valor empírico = {placebo_res['p_value']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="semibold", color=COLOR_TEXT,
            bbox=dict(facecolor="white", edgecolor=COLOR_GRID, boxstyle="round,pad=0.4"))
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # Distribuição geográfica dos ATTs placebos — visualiza a extremidade de SP em escala nacional.

    all_atts = {city: fit.att_avg_pct for city, fit in placebo_res["placebos"].items()}
    all_atts["São Paulo"] = placebo_res["treated"].att_avg_pct
    vmax = max(abs(v) for v in all_atts.values())
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_brazil_basemap(ax)

    for city, (lat, lon) in CITY_COORDS.items():
        if city not in all_atts:
            ax.scatter(lon, lat, s=45, color="#DDDDDD",
                        edgecolor="white", linewidth=0.5, alpha=0.7, zorder=5)
            continue
        att = all_atts[city]
        is_treated = city == "São Paulo"
        color = cmap(norm(att))
        if is_treated:
            ax.scatter(lon, lat, s=340, color=color,
                        edgecolor=COLOR_TREATED, linewidth=2.5, zorder=10)
            ax.annotate(f"São Paulo\\n{att*100:+.1f}%", (lon, lat), xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=10, fontweight="bold",
                        color=COLOR_TREATED, zorder=11)
        else:
            ax.scatter(lon, lat, s=130, color=color,
                        edgecolor="white", linewidth=0.6, zorder=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("ATT placebo (% sobre baseline pré)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("ATTs placebos no mapa — SP é o ponto extremo",
                  fontsize=12, pad=12, color=COLOR_TEXT)
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
    # 5b. Conformal inference — refita AugSynth com inference="conformal" pra obter CIs por período.

    aug_conf = fit_augsynth_r(panel, treated_city="São Paulo", treatment_day=TREATMENT_DAY,
                                conformal=True)
    print(f"ATT por período disponível: {aug_conf.att_by_period is not None}")
    print(f"CIs disponíveis (lower/upper): {aug_conf.ci_lower is not None}/{aug_conf.ci_upper is not None}")
"""))

CELLS.append(code("""
    # Plot — ATT trajetória + banda de confiança 90% (conformal)

    if aug_conf.att_by_period is not None and aug_conf.ci_lower is not None:
        att_t = aug_conf.att_by_period
        lo, hi = aug_conf.ci_lower, aug_conf.ci_upper
        periods = aug_conf.att_periods if aug_conf.att_periods is not None else np.arange(len(att_t))

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(periods, lo, hi, color=COLOR_AUG, alpha=0.20, label="CI 90% (conformal)")
        ax.plot(periods, att_t, color=COLOR_AUG, lw=2.2, label="ATT estimado")
        ax.axhline(0, color=COLOR_TEXT, lw=0.7)
        ax.axvline(TREATMENT_DAY, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title("ATT por período com CI conformal — a banda atravessa zero?")
        ax.set_xlabel("Dia"); ax.set_ylabel("ATT (R$/dia)")
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
    else:
        print("CIs conformal não disponíveis nesta versão do augsynth — pulando plot.")
"""))

CELLS.append(md("""
    > **Takeaway.** Placebo dá um teste global ("o ATT é extremo?"); conformal dá leitura
    > **período a período** ("em qual dia a banda deixa de cruzar zero?"). Use os dois.
    > No nosso caso ambos concordam: o efeito é detectável e robusto.
"""))

# --- Ato 6 -----------------------------------------------------------------
CELLS.append(md("""
    ## Ato 6 — Migração pra Python: o que falta

    Tudo nesse notebook hoje passa por R via `rpy2`. O objetivo do `augsynth-py` é trazer essas
    funcionalidades pro lado Python, com paridade numérica testada contra o R como oracle.
"""))

CELLS.append(code("""
    # Introspecção do estado atual de src/augsynth_py/synth/.

    SYNTH_DIR = Path.cwd() / "src" / "augsynth_py" / "synth"
    implemented = sorted(p.stem for p in SYNTH_DIR.glob("*.py") if p.stem != "__init__")
    print("Estimadores nativos implementados em src/augsynth_py/synth/:")
    print(f"  {implemented if implemented else '(nenhum ainda — só skeleton)'}")
"""))

CELLS.append(code("""
    # Tabela de paridade: o que cada ato chamou hoje vs como vai ficar em augsynth-py.

    parity = [
        ("Ato 2", "DiD ingênuo",          "puro Python (já feito)",                "—",                                   "✅"),
        ("Ato 3", "Synthetic Control",    "augsynth(progfunc='None') via rpy2",   "Synth().fit(...).predict()",          "⏳ v0.1"),
        ("Ato 4", "AugSynth (ridge)",     "augsynth(progfunc='Ridge') via rpy2",  "AugSynth().fit(...).predict()",       "⏳ v0.1"),
        ("Ato 5a","Placebo permutation",  "loop manual + fit_scm_r",              ".placebo_test(n_jobs=-1)",            "⏳ v0.1"),
        ("Ato 5b","Conformal inference",  "augsynth(inference='conformal')",      ".confidence_intervals(alpha=0.10)",   "⏳ v0.1"),
        ("—",     "Generalized SCM (Xu)", "gsynth via rpy2 (não usado aqui)",     "GSC().fit(...)",                       "🔜 v0.2"),
        ("—",     "Multi-cell ASCM",      "—",                                     "MultiAugSynth().fit(...)",            "🔜 v0.3"),
        ("—",     "Power analysis (geo)", "—",                                     "PowerAnalysis().run(...)",            "🔜 v0.4"),
    ]

    col_widths = [max(len(str(row[i])) for row in parity) for i in range(5)]
    headers = ["Ato", "Funcionalidade", "Hoje (R via rpy2)", "augsynth-py", "Status"]
    col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]
    fmt_row = lambda row: "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths))
    sep = "-" * (sum(col_widths) + 2*(len(col_widths)-1))
    print(fmt_row(headers)); print(sep)
    for row in parity: print(fmt_row(row))
"""))

CELLS.append(md("""
    > **Takeaway.** Nada nesse notebook é metodologicamente novo — o objetivo do `augsynth-py`
    > é tornar essa pipeline **nativamente Python**, sem dependência de R, com testes de paridade
    > numérica contra os pacotes R do `augsynth`/`Synth` como oracle.
    >
    > **Próximos passos no backlog**: replicar o caso canônico do paper Ben-Michael et al. 2021
    > (corte de impostos do Kansas em 2012) num notebook irmão, e implementar o primeiro estimador
    > nativo (`Synth`) em `src/augsynth_py/synth/synth.py`.
"""))


# ---------------------------------------------------------------------------
# Build & save
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
