# Painel AEAT para o notebook NR-35 — proveniência

O notebook `04_nr35_trabalho_em_altura.ipynb` procura, nesta ordem:

1. `aeat_nr35_panel.csv` — **dados reais** do AEAT (Anuário Estatístico de
   Acidentes do Trabalho, MPS): óbitos e taxa de mortalidade por divisão
   CNAE 2.0, 2008–2019. Gerado por `notebooks/_fetch_aeat_nr35.py`
   (requer internet com acesso a `www.gov.br` — o pipeline está documentado
   e parcialmente implementado; ver docstring do script).
2. `aeat_nr35_panel_demo.csv` — **dados simulados** de demonstração, com
   efeito verdadeiro conhecido injetado na tratada (documentado em
   `aeat_nr35_demo_truth.csv`). Gerado por
   `python notebooks/_fetch_aeat_nr35.py --demo` (semente fixa, reproduzível).
   **Não são dados do AEAT** — magnitudes apenas calibradas na ordem de
   grandeza publicada. O notebook sinaliza em destaque quando roda neste modo.

## Por que o CSV real não está commitado

Esta sessão de desenvolvimento roda atrás de um proxy com allowlist estrita
(registries de pacotes + GitHub); `www.gov.br` e espelhos (dados.gov.br,
web.archive.org, basedosdados) são bloqueados, e as tabelas do AEAT são
arquivos `.xls` binários — inacessíveis também pelas ferramentas de fetch
textuais. A extração alternativa via PDF (993 p./edição) foi descartada por
risco de atribuição: há 34–56 tabelas de layout idêntico por edição
(Brasil/regiões/UFs) e os extratores de trechos não preservam o título da
tabela adjacente.

## Fontes reais (verificadas em 2026-08-24)

- Índice das edições do AEAT:
  <https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/saude-e-seguranca-do-trabalhador/acidente_trabalho_incapacidade>
- Cada edição publica o ano de referência + 2 anteriores (revisados); as
  edições 2010, 2013, 2016 e 2019 cobrem 2008–2019 por inteiro.
  ZIPs de tabelas: `arquivos/tabelas-aeat-2010.zip`,
  `arquivos/aeat_tabelas_2013.zip`, `arquivos/aeat-2016.zip`,
  `arquivos/aeat-2019_def.zip` (relativos ao índice acima).
- Óbitos: Seção I, Subseção B, tabela 29.1 — "Quantidade de acidentes do
  trabalho liquidados, por consequência, segundo a CNAE, no Brasil".
  Linhas = classes CNAE 2.0 (4 dígitos) + TOTAL; colunas = 6 grupos de
  consequência × 3 anos; Óbito é o último grupo. Ex. (edição 2012, arquivos
  soltos): `https://www.gov.br/previdencia/pt-br/outros/imagens/2014/01/29a_01.xls`.
- Taxas/indicadores: Seção II, tabelas 59.x — "Indicadores de acidentes do
  trabalho, segundo a CNAE" (inclui "Taxa de Mortalidade (por 100.000
  vínculos)"); classes CNAE + TOTAL + "Outros (<100 vínculos)".
- Denominador (vínculos por divisão): tabelas de vínculos do AEAT
  (edições ≥ 2009) ou RAIS (`br_me_rais` na Base dos Dados).
- Microdados de CAT (INSS, ~2018+; agente causador × CNAE — para
  caracterizar o canal "quedas de altura" no pós-período):
  <https://dadosabertos.inss.gov.br/>

## Decisões de preparo (valem para o CSV real)

- Divisões 41, 42 e 43 agregadas na unidade tratada "Construção"
  (soma de óbitos e vínculos antes da taxa — média ponderada).
- Doadores: demais divisões com vínculos suficientes para taxa estável
  (piso documentado no script); divisões minúsculas só adicionam ruído.
- Desfecho: taxa de mortalidade (óbitos por 100 mil vínculos), não contagem —
  sem normalizar, o ciclo de emprego da construção se disfarça de efeito.
