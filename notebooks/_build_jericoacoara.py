"""Programmatic builder for the Jericoacoara airport case-study notebook.

Run once to (re)generate the notebook::

    .venv/bin/python notebooks/_build_jericoacoara.py

The notebook is the artifact users open and edit. This script is the source of
truth used to recreate it cleanly when the design evolves — same pattern as
``_build_showcase.py`` and ``_build_replications.py``.

Unlike notebooks 02/03 this is not a replication: there is no published paper
to check the numbers against. It is an original application of the package to
a novel Brazilian natural experiment, written in Portuguese for the audience
the case speaks to. The data pipeline lives in ``_fetch_jericoacoara.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NOTEBOOK_PATH = Path(__file__).parent / "04_jericoacoara_airport.ipynb"

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
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    # --- Title & framing ----------------------------------------------------
    cells.append(
        md(
            """
        # O aeroporto de Jericoacoara — um estudo de caso inédito

        Até 2017, chegar à vila de Jericoacoara (CE) exigia 5 a 6 horas de
        estrada desde Fortaleza, com o trecho final feito em veículo 4x4 por
        cima das dunas. Em **24 de junho de 2017** foi inaugurado o aeroporto
        regional Comandante Ariston Pessoa, no município de **Jijoca de
        Jericoacoara** (~19 mil habitantes), com voos comerciais regulares a
        partir do segundo semestre. De um dia para o outro, um dos destinos
        turísticos mais isolados do Nordeste ganhou acesso direto aos hubs do
        país.

        **Pergunta causal:** qual foi o efeito do aeroporto sobre o mercado de
        trabalho formal do município?

        Este notebook é diferente dos notebooks 02 e 03: **não é uma
        replicação**. Não existe (até onde sabemos) estudo publicado de
        controle sintético sobre este evento — os números que saem daqui não
        têm gabarito na literatura. O que há de mais próximo é a evidência de
        que aeroportos regionais elevam turismo local, obtida com controle
        sintético em regiões alemãs (Regional Studies, 2020). Tratem os
        resultados como uma análise original, com as ressalvas do ato final.

        **Desenho do estudo**

        | Elemento | Escolha |
        |---|---|
        | Unidade tratada | Jijoca de Jericoacoara (CE) |
        | Intervenção | 2017 (primeiro ano com aeroporto; exposição parcial) |
        | Desfecho | Pessoal ocupado assalariado por 1.000 habitantes (CEMPRE/IBGE) |
        | Painel | 2006–2021, anual (a série CEMPRE foi encerrada em 2021) |
        | Doadores | 15 municípios litorâneos de CE/RN/PI/MA sem aeroporto próprio |
        | Excluídos do pool | Cruz e Camocim (vizinhos — spillover), Aracati e Parnaíba (aeroportos próprios), municípios da RM de Fortaleza |

        Os dados vêm de duas APIs públicas do IBGE (sem chave): a tabela 1685
        do SIDRA (CEMPRE — Cadastro Central de Empresas) e a tabela 6579
        (estimativas de população). O detalhamento setorial por seção CNAE só
        é publicado para municípios com 50 mil+ habitantes, o que exclui
        Jijoca; usamos o total municipal, defensável aqui porque o turismo
        domina a economia formal local. O extrator auditável está em
        `_fetch_jericoacoara.py`.

        **Roteiro:**
        1. **Ato 1** — os dados e o pool de doadores
        2. **Ato 2** — por que a comparação ingênua falha
        3. **Ato 3** — controle sintético clássico (`Synth`)
        4. **Ato 4** — ridge augmentation (`AugSynth`)
        5. **Ato 5** — inferência conformal (CWZ 2021)
        6. **Ato 6** — placebos in-space e o teste dos vizinhos (spillover)
        7. **Ato 7** — poder ex-ante do desenho (`simulate_power`, v0.4)
        """
        )
    )

    # --- Setup ---------------------------------------------------------------
    cells.append(
        code(
            """
        # Imports e estilo. Espelha os notebooks 01-03.
        from __future__ import annotations

        import subprocess
        import sys
        from pathlib import Path

        # Torna `augsynth_py` importável rodando o notebook de qualquer lugar.
        _here = Path.cwd()
        for cand in (_here, _here.parent, _here.parent.parent):
            if (cand / "src" / "augsynth_py").exists():
                sys.path.insert(0, str(cand / "src"))
                NB_DIR = (cand / "notebooks") if (cand / "notebooks").exists() else _here
                break
        else:
            NB_DIR = _here
        DATA_DIR = NB_DIR / "_data"

        import numpy as np
        import polars as pl
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        from augsynth_py import AugSynth, Synth, conformal_interval, conformal_pvalue, simulate_power

        COLOR_TREATED = "#D7263D"
        COLOR_SYNTH   = "#1B4965"
        COLOR_DONOR   = "#B0B0B0"
        COLOR_GRID    = "#EAEAEA"
        COLOR_TEXT    = "#333333"
        COLOR_AUGMENTED = "#7A5195"

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

        UNIT  = "municipality"
        TIME  = "year"
        OUT   = "workers_per_1k"
        TREATED = "Jijoca de Jericoacoara"
        INTERVENTION = 2017  # primeiro ano pós: aeroporto inaugurado em jun/2017
        OUT_LABEL = "Empregos formais por 1.000 hab. (CEMPRE)"
        """
        )
    )

    # --- Ato 1 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 1 — Os dados e o pool de doadores

        A célula abaixo carrega o painel; se o CSV ainda não existir, ela roda
        o extrator uma vez (requer internet, ~30 s). O CSV traz três papéis:
        `treated`, `donor` e `neighbor` — os vizinhos Cruz e Camocim ficam
        **fora** do pool de doadores (compartilham o fluxo turístico de Jeri e
        contaminariam o contrafactual), mas voltam no Ato 6 como diagnóstico
        de spillover.

        Doadores com célula mascarada pelo sigilo do CEMPRE em qualquer ano
        são descartados do pool, com aviso — um painel balanceado é requisito
        do estimador.
        """
        )
    )

    cells.append(
        code(
            """
        CSV_PATH = DATA_DIR / "jericoacoara_cempre.csv"
        if not CSV_PATH.exists():
            print("CSV ausente — rodando notebooks/_fetch_jericoacoara.py (requer internet)...")
            subprocess.run([sys.executable, str(NB_DIR / "_fetch_jericoacoara.py")], check=True)

        raw = pl.read_csv(CSV_PATH)
        study = raw.filter(pl.col("role").is_in(["treated", "donor"]))

        incomplete = (
            study.filter(pl.col(OUT).is_null())[UNIT].unique().sort().to_list()
        )
        if TREATED in incomplete:
            raise ValueError("A série da unidade tratada tem células mascaradas — sem análise possível.")
        if incomplete:
            print(f"Doadores descartados por células mascaradas: {incomplete}")
        panel = study.filter(~pl.col(UNIT).is_in(incomplete))

        n_donors = panel.filter(pl.col("role") == "donor")[UNIT].n_unique()
        print(f"painel      : {panel.shape[0]} linhas, {panel[UNIT].n_unique()} municípios ({n_donors} doadores)")
        print(f"anos        : {panel[TIME].min()}-{panel[TIME].max()}  (pré: {panel[TIME].min()}-{INTERVENTION - 1})")
        panel.filter(pl.col(UNIT) == TREATED).head(5)
        """
        )
    )

    cells.append(
        code(
            """
        # Trajetórias: Jijoca em vermelho, doadores em cinza.
        fig, ax = plt.subplots(figsize=(11, 5.2))
        for unit, group in panel.sort(TIME).partition_by(UNIT, as_dict=True).items():
            is_treated = unit[0] == TREATED
            ax.plot(
                group[TIME], group[OUT],
                color=COLOR_TREATED if is_treated else COLOR_DONOR,
                lw=2.4 if is_treated else 0.9,
                alpha=1.0 if is_treated else 0.55,
                zorder=3 if is_treated else 1,
                label="Jijoca de Jericoacoara" if is_treated else None,
            )
        ax.axvline(INTERVENTION - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.text(INTERVENTION - 0.4, ax.get_ylim()[1] * 0.98, "  Aeroporto (jun/2017)",
                color=COLOR_TEXT, fontsize=10, va="top")
        ax.set_title("Emprego formal por 1.000 habitantes — Jijoca vs doadores")
        ax.set_xlabel("Ano"); ax.set_ylabel(OUT_LABEL)
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    # --- Ato 2 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 2 — Por que a comparação ingênua falha

        O reflexo natural é comparar Jijoca com a média dos demais municípios,
        antes e depois de 2017 — o DiD 2×2 de livro-texto. Isso assume que,
        sem o aeroporto, Jijoca teria evoluído **em paralelo** à média do
        litoral. Municípios turísticos pequenos têm dinâmicas idiossincráticas
        (uma pousada grande abre, uma temporada de kite excepcional), então a
        hipótese merece desconfiança — e é exatamente o que o controle
        sintético substitui por uma combinação ponderada que reproduz a
        trajetória pré-2017 por construção.
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

        pre  = pl.col(TIME) <  INTERVENTION
        post = pl.col(TIME) >= INTERVENTION
        did_att = (
            (comparison.filter(post)["treated"].mean() - comparison.filter(pre)["treated"].mean())
            - (comparison.filter(post)["donors_mean"].mean() - comparison.filter(pre)["donors_mean"].mean())
        )
        print(f"ATT do DiD 2x2 (variação de Jijoca - variação da média dos doadores): {did_att:+.2f} empregos/1.000 hab.")

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(comparison[TIME], comparison["treated"], color=COLOR_TREATED, lw=2.4, label="Jijoca")
        ax.plot(comparison[TIME], comparison["donors_mean"], color=COLOR_DONOR, lw=2.0,
                ls="-.", label="Média simples dos doadores")
        ax.axvline(INTERVENTION - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title("Jijoca vs média crua dos doadores — as tendências pré eram mesmo paralelas?")
        ax.set_xlabel("Ano"); ax.set_ylabel(OUT_LABEL)
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    # --- Ato 3 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 3 — Controle sintético clássico

        `Synth` resolve o QP no simplex (Abadie, Diamond & Hainmueller 2010):
        pesos não-negativos que somam 1, minimizando o erro quadrático da
        trajetória pré-2017. A "Jijoca sintética" é a combinação convexa dos
        doadores que melhor imita a Jijoca real antes do aeroporto.
        """
        )
    )

    cells.append(
        code(
            """
        scm = Synth().fit(
            panel, unit=UNIT, time=TIME, outcome=OUT,
            treated=TREATED, treatment_time=INTERVENTION,
        )

        weights = (
            pl.DataFrame({"município": list(scm.weights_), "peso": list(scm.weights_.values())})
            .sort("peso", descending=True)
        )
        print(f"RMSPE pré (normalizado): {scm.rmspe_pre_:.4f}")
        print(f"ATT 2017-2021          : {scm.att_:+.2f} empregos/1.000 hab. ({scm.att_pct_:+.1%})")
        print()
        print("Doadores com peso > 1%:")
        print(weights.filter(pl.col("peso") > 0.01))
        """
        )
    )

    cells.append(
        code(
            """
        def plot_fit(fit, title, color_synth=COLOR_SYNTH, synth_label="Jijoca sintética"):
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), width_ratios=[3, 2])
            pre_mask = np.asarray(fit.pre_mask_)
            axes[0].plot(fit.periods_, fit.actual_, color=COLOR_TREATED, lw=2.4, label="Jijoca real")
            axes[0].plot(fit.periods_, fit.synthetic_, color=color_synth, lw=2.0, ls="--", label=synth_label)
            axes[0].axvline(INTERVENTION - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
            axes[0].set_title(title)
            axes[0].set_xlabel("Ano"); axes[0].set_ylabel(OUT_LABEL)
            axes[0].legend(loc="upper left", frameon=False)
            axes[1].axhline(0, color=COLOR_TEXT, lw=0.8)
            axes[1].fill_between(fit.periods_, fit.gap_, 0, where=~pre_mask, color=color_synth, alpha=0.25)
            axes[1].plot(fit.periods_, fit.gap_, color=color_synth, lw=2.0)
            axes[1].axvline(INTERVENTION - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
            axes[1].set_title("Gap (real - sintético)")
            axes[1].set_xlabel("Ano"); axes[1].set_ylabel("Empregos/1.000 hab.")
            plt.tight_layout(); plt.show()

        plot_fit(scm, "Synth — o ajuste pré-2017 é a moeda de credibilidade do SCM")
        """
        )
    )

    cells.append(
        md(
            """
        > **Como ler.** O painel esquerdo compara a Jijoca real com a
        > sintética; a credibilidade do exercício está no ajuste **antes** da
        > linha tracejada. O painel direito mostra o gap: o efeito estimado é
        > a área pós-2017. Se o RMSPE pré for alto (regra de bolso: acima de
        > ~5% da média pré), o contrafactual não reproduz a Jijoca pré-aeroporto
        > e o ATT deve ser lido com ceticismo — veja também o Ato 6.
        """
        )
    )

    # --- Ato 4 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 4 — Ridge augmentation

        Com poucos doadores e uma tratada na borda do casco convexo (Jijoca é
        um dos municípios mais turísticos do pool), o SCM clássico pode não
        fechar o ajuste pré. `AugSynth` (Ben-Michael, Feller & Rothstein 2021)
        corrige o viés residual com uma regressão ridge — os pesos deixam de
        ser convexos (podem ser negativos), em troca de um ajuste pré melhor.
        O λ é escolhido por validação cruzada leave-one-out no pré-período.
        """
        )
    )

    cells.append(
        code(
            """
        aug = AugSynth().fit(
            panel, unit=UNIT, time=TIME, outcome=OUT,
            treated=TREATED, treatment_time=INTERVENTION,
        )

        print(f"lambda (LOO-CV)         : {aug.lambda_:.4g}")
        print(f"RMSPE pré  Synth        : {scm.rmspe_pre_:.4f}")
        print(f"RMSPE pré  AugSynth     : {aug.rmspe_pre_:.4f}")
        print(f"ATT  Synth              : {scm.att_:+.2f} empregos/1.000 hab. ({scm.att_pct_:+.1%})")
        print(f"ATT  AugSynth           : {aug.att_:+.2f} empregos/1.000 hab. ({aug.att_pct_:+.1%})")
        neg = sum(1 for w in aug.weights_.values() if w < -1e-8)
        print(f"pesos negativos (ridge) : {neg} de {len(aug.weights_)}")

        plot_fit(aug, "AugSynth — contrafactual com correção ridge",
                 color_synth=COLOR_AUGMENTED, synth_label="Jijoca sintética (aug.)")
        """
        )
    )

    cells.append(
        md(
            """
        > **Como ler.** Se Synth e AugSynth contam a mesma história (ATTs
        > próximos), o resultado não depende da escolha do estimador — o
        > cenário confortável. Se divergem, o ajuste pré de cada um é o
        > desempate: o AugSynth existe exatamente para os casos em que o
        > simplex não alcança a tratada.
        """
        )
    )

    # --- Ato 5 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 5 — Inferência conformal (CWZ 2021)

        Com uma única unidade tratada, erro-padrão clássico não existe. A
        inferência conformal de Chernozhukov, Wüthrich & Zhu (2021) testa a
        hipótese nula de efeito zero permutando os resíduos do modelo
        reajustado sob a nula — exata em amostra finita, sem assintótica.

        **Granularidade importa aqui:** com 16 anos de painel, o esquema de
        blocos (permutações cíclicas) gera 16 permutações — o menor p-valor
        possível é 1/16 ≈ 0,06. Significância a 5% é *inalcançável por
        construção* neste painel anual; por isso trabalhamos com α = 0,10 (o
        padrão de geo-experimentos) e reportamos o IC de 90%.
        """
        )
    )

    cells.append(
        code(
            """
        for name, fit in [("Synth", scm), ("AugSynth", aug)]:
            p_two = conformal_pvalue(fit, side="two-sided")
            lo, hi = conformal_interval(fit, alpha=0.10)
            print(f"{name:9s}  p (bicaudal) = {p_two:.3f}   IC 90% do efeito constante: [{lo:+.2f}, {hi:+.2f}] empregos/1.000 hab.")
        """
        )
    )

    # --- Ato 6 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 6 — Placebos in-space e o teste dos vizinhos

        Dois exercícios de falsificação:

        1. **Placebos in-space** (Abadie et al.): reatribuímos o "tratamento"
           a cada doador e comparamos a razão RMSPE pós/pré de Jijoca com a
           distribuição placebo. Se Jijoca não estiver na cauda, o gap
           estimado é indistinguível de ruído idiossincrático.
        2. **Vizinhos como pseudo-tratados**: Cruz e Camocim ficaram fora do
           pool por compartilharem o fluxo turístico de Jeri. Ajustando um SCM
           para cada um (contra o mesmo pool de doadores), um gap positivo
           pós-2017 é evidência direta de spillover — reforça a exclusão e
           lembra que o nosso ATT mede o efeito *local em Jijoca*, não o
           efeito regional do aeroporto.
        """
        )
    )

    cells.append(
        code(
            """
        def rmspe_ratio(fit):
            pre_mask = np.asarray(fit.pre_mask_)
            pre_rmspe = np.sqrt(np.mean(fit.gap_[pre_mask] ** 2))
            post_rmspe = np.sqrt(np.mean(fit.gap_[~pre_mask] ** 2))
            return post_rmspe / pre_rmspe

        donor_names = sorted(u for u in panel[UNIT].unique().to_list() if u != TREATED)
        placebo_fits = {}
        for donor in donor_names:
            pool = panel.filter(pl.col(UNIT) != TREATED)  # a tratada de verdade sai do pool placebo
            placebo_fits[donor] = Synth().fit(
                pool, unit=UNIT, time=TIME, outcome=OUT,
                treated=donor, treatment_time=INTERVENTION,
            )

        ratios = (
            pl.DataFrame({
                "município": [TREATED, *placebo_fits],
                "razão RMSPE pós/pré": [rmspe_ratio(scm), *(rmspe_ratio(f) for f in placebo_fits.values())],
                "papel": ["tratado", *(["placebo"] * len(placebo_fits))],
            })
            .sort("razão RMSPE pós/pré", descending=True)
        )
        rank = 1 + ratios["município"].to_list().index(TREATED)
        n_units = ratios.height
        print(ratios)
        print()
        print(f"Jijoca é a {rank}ª maior razão entre {n_units} unidades "
              f"=> p-valor de permutação in-space ~ {rank / n_units:.3f}")
        """
        )
    )

    cells.append(
        code(
            """
        # Espaguete de gaps: placebos em cinza, Jijoca em vermelho.
        fig, ax = plt.subplots(figsize=(11, 5.0))
        for donor, fit in placebo_fits.items():
            ax.plot(fit.periods_, fit.gap_, color=COLOR_DONOR, lw=0.9, alpha=0.6)
        ax.plot(scm.periods_, scm.gap_, color=COLOR_TREATED, lw=2.5, label="Jijoca (tratado)")
        ax.axhline(0, color=COLOR_TEXT, lw=0.8)
        ax.axvline(INTERVENTION - 0.5, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title("Gaps placebo — o gap de Jijoca se destaca do ruído dos doadores?")
        ax.set_xlabel("Ano"); ax.set_ylabel("Gap (real - sintético), empregos/1.000 hab.")
        ax.legend(loc="upper left", frameon=False)
        plt.tight_layout(); plt.show()
        """
        )
    )

    cells.append(
        code(
            """
        # Vizinhos como pseudo-tratados: spillover vira gap positivo pós-2017.
        neighbors = raw.filter(
            (pl.col("role") == "neighbor") & pl.col(OUT).is_not_null()
        )
        donor_pool = panel.filter(pl.col(UNIT) != TREATED)
        for neighbor in neighbors[UNIT].unique().sort().to_list():
            neighbor_rows = neighbors.filter(pl.col(UNIT) == neighbor)
            if neighbor_rows.height < panel[TIME].n_unique():
                print(f"{neighbor:10s}  série incompleta (sigilo CEMPRE) — diagnóstico pulado")
                continue
            pseudo = pl.concat([donor_pool, neighbor_rows.select(donor_pool.columns)])
            fit = Synth().fit(
                pseudo, unit=UNIT, time=TIME, outcome=OUT,
                treated=neighbor, treatment_time=INTERVENTION,
            )
            print(f"{neighbor:10s}  ATT pós-2017 = {fit.att_:+.2f} empregos/1.000 hab. "
                  f"({fit.att_pct_:+.1%}), RMSPE pré = {fit.rmspe_pre_:.4f}")
        """
        )
    )

    # --- Ato 7 ---------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ato 7 — Poder ex-ante: este desenho enxergaria o quê?

        Antes de acreditar em qualquer ATT, vale perguntar o que este desenho
        *conseguiria* detectar. Usamos `simulate_power` (v0.4) **apenas com os
        dados pré-2017**: injetamos efeitos multiplicativos simulados de 3
        anos na Jijoca ainda-não-tratada e medimos a taxa de detecção
        conformal. É o mesmo protocolo do `GeoLiftPower`, aplicado como
        auditoria ex-ante do desenho — se o MDE sair maior que qualquer efeito
        plausível de um aeroporto, o painel anual simplesmente não tem
        resolução para a pergunta, e o Ato 3 vira ilustração, não evidência.

        (No painel pré há 11 anos; com blocos, o menor p-valor é 1/11 ≈ 0,09,
        então α = 0,10 é de novo o limiar operacional.)
        """
        )
    )

    cells.append(
        code(
            """
        pre_panel = panel.filter(pl.col(TIME) < INTERVENTION)
        power = simulate_power(
            pre_panel,
            estimator=Synth(),
            unit=UNIT, time=TIME, outcome=OUT,
            treated=TREATED,
            durations=3,
            effect_sizes=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
            alpha=0.10,
        )
        curve = power.power_curve()
        mde = power.mde(target_power=0.8)
        print(curve)
        print()
        if mde is None:
            print("MDE (poder 80%): nenhum efeito da grade atinge 80% de detecção neste desenho.")
        else:
            print(f"MDE (poder 80%): lift de {mde:.0%} no emprego formal por 3 anos.")

        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.plot(curve["effect_size"], curve["power"], marker="o", color=COLOR_SYNTH, lw=2.0)
        ax.axhline(0.8, color=COLOR_TEXT, ls="--", lw=0.9, alpha=0.55)
        ax.set_title("Curva de poder ex-ante — efeito simulado de 3 anos, painel pré-2017")
        ax.set_xlabel("Efeito injetado (lift multiplicativo)"); ax.set_ylabel("Poder (taxa de detecção)")
        ax.set_ylim(-0.02, 1.02)
        plt.tight_layout(); plt.show()
        """
        )
    )

    # --- Closing -------------------------------------------------------------
    cells.append(
        md(
            """
        ## Ressalvas e próximos passos

        O que este notebook **não** estabelece, e como avançar:

        - **Frequência anual.** O CEMPRE dá 11 anos de pré e 5 de pós — o
          conformal opera no limite da granularidade. A extensão natural é o
          CAGED mensal (microdados do Ministério do Trabalho), que
          multiplicaria os períodos por 12 ao custo de um pipeline de dados
          bem mais pesado.
        - **2017 é um ano de exposição parcial** (aeroporto aberto em junho,
          voos regulares no 2º semestre): o efeito estimado em 2017 dilui o
          efeito de regime permanente.
        - **A pandemia atravessa o pós-período.** 2020-21 derrubou o turismo
          em todo o pool; o contrafactual absorve o choque comum, mas efeitos
          heterogêneos da COVID entre municípios ficam confundidos com o
          efeito do aeroporto. Reportar o ATT também na janela 2017-2019 é a
          checagem honesta.
        - **Spillover.** Se o Ato 6 mostrar gaps positivos em Cruz/Camocim, o
          efeito regional do aeroporto é maior que o ATT local de Jijoca; se
          mostrar gaps em doadores próximos, o ATT local está *subestimado*
          (o contrafactual foi puxado para cima).
        - **Desfecho agregado.** Sem o corte setorial (sigilo para municípios
          <50 mil hab.), não separamos alojamento/alimentação do resto — o
          canal é inferido, não observado.

        Se os resultados sobreviverem a essas ressalvas, este caso tem
        tamanho de working paper — e serve de modelo para os outros
        candidatos inéditos mapeados no repositório (banimento do amianto em
        Minaçu, zona livre de aftosa em SC, afundamento de Maceió).
        """
        )
    )

    return cells


def build_notebook() -> None:
    cells = build_cells()
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Wrote {NOTEBOOK_PATH} - {len(cells)} cells ({n_code} code, {n_md} markdown).")


if __name__ == "__main__":
    build_notebook()
