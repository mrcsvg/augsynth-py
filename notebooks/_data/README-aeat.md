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

## Fontes reais (URLs testadas em 2026-09-05)

Índice das edições:
<https://www.gov.br/previdencia/pt-br/assuntos/previdencia-social/saude-e-seguranca-do-trabalhador/acidente_trabalho_incapacidade>

Todas as URLs abaixo são relativas a esse índice e foram testadas: elas
**servem arquivo**. O teste discrimina de verdade — um caminho inventado no
mesmo diretório devolve 404. Rode `python notebooks/_fetch_aeat_nr35.py --probe`
para reconferir (o gov.br renomeia os pacotes entre edições).

| Edição | ZIP de tabelas |
|---|---|
| 2009 | `arquivos/tabelas-aeat-2009.zip` |
| 2010 | `arquivos/tabelas-aeat-2010.zip` |
| 2011 | `arquivos/tabelas-aeat-2011.zip` |
| 2013 | `arquivos/aeat_tabelas_2013.zip` |
| 2014 | `arquivos/aeat2014_tabelas.zip` |
| 2015 | `arquivos/aeat15tab.zip` |
| 2016 | `arquivos/aeat-2016.zip` |
| 2017 | `arquivos/aeat-2017.zip` |
| 2018 | `arquivos/aeat-2018_tabelas-v2.zip` |
| 2019 | `arquivos/aeat-2019_def.zip` |
| 2020 | `arquivos/aeat-2020.zip` |

**Não existe ZIP para as edições 2008 e 2012** — o link "Tabelas" delas aponta
para a própria página HTML da edição. Nenhuma das duas é necessária: 2008 vem
das edições 2009/2010 e 2012 da edição 2013. A edição 2012 também serve as
tabelas soltas, ex. `https://www.gov.br/previdencia/pt-br/outros/imagens/2014/01/29a_01.xls`
(a de 2013 usa a pasta `/2015/01/`).

### As duas seções cobrem spans de anos DIFERENTES

- **Óbitos** — Seção I, Subseção B, tabela 29.1 ("acidentes do trabalho
  liquidados, por consequência, segundo a CNAE, no Brasil"): o ano da edição
  **mais os 2 anteriores**. Linhas = classes CNAE 2.0 (4 dígitos) + TOTAL +
  Ignorado; colunas = 6 grupos de consequência × 3 anos, sendo *Óbito* o
  último grupo. Quatro edições (2010, 2013, 2016, 2019) cobrem 2008–2019.
- **Indicadores** — Seção II, capítulo 59 (Brasil): apenas **2 anos** por
  edição — tabela `59.1` = ano anterior, `59.2` = ano da edição. Inclui
  "Taxa de Mortalidade (por 100.000 vínculos)" e "Incidência (por 1.000
  vínculos)", por classe CNAE, com linha `Outros (1)` = atividades com menos
  de 100 vínculos médios. Cobrir 2008–2019 exige **seis** edições: 2009,
  2011, 2013, 2015, 2017, 2019.

### Denominador de vínculos — não é publicado em 2008–2019

A coluna "quantidade média de vínculos" só aparece nas edições recentes
(presente em 2023, **ausente em 2019**). Como taxas não se agregam de classe
para divisão sem peso, recupere o próprio denominador que o AEAT usou:

```
vínculos_classe = acidentes_registrados_classe × 1000 / incidência_classe
```

cruzando a Subseção A (registrados por CNAE) com a Seção II (incidência), e só
então some óbitos e vínculos por divisão antes de dividir. Prefira essa via à
inversão da taxa de mortalidade: acidentes superam óbitos em duas ordens de
grandeza, então o arredondamento de 2 casas da taxa publicada custa muito menos
precisão, e poucas classes têm incidência suprimida. Para classes suprimidas
(`-`), caia para a RAIS por divisão.

### Outras fontes

- **Versão online (HTML)**, sem `.xls`: só das edições 2019 em diante — cobre
  os anos 2018 e 2019. A edição 2012 tem uma página `tabela-dos-indicadores`
  com links `.xls` diretos (anos 2011 e 2012).
- **AEAT Infologo** (`http://www3.dataprev.gov.br/aeat/inicio.htm`) — o
  tabulador que resolveria vínculos de uma vez. Ainda linkado pelo MPS, mas
  **não respondeu** (timeout) nas verificações de 2026-08-24 e 2026-09-05.
  Inconclusivo: pode ser instabilidade ou bloqueio a IPs estrangeiros; vale
  tentar de uma rede brasileira.
- **Microdados de CAT** (INSS, CSV a partir de jul/2018; agente causador ×
  CNAE — para caracterizar o canal "quedas de altura" no pós-período):
  <https://dadosabertos.inss.gov.br/dataset/inss-comunicacao-de-acidente-de-trabalho-cat>
  Não cobre 2008–2017 e não traz denominador.
- **Base dos Dados**: a ficha do AEAT declara "Possui dados estruturados: Não"
  e "Tem API: Não" — é catálogo apontando para a fonte original, não atalho.

## Decisões de preparo (valem para o CSV real)

- Divisões 41, 42 e 43 agregadas na unidade tratada "Construção"
  (soma de óbitos e vínculos antes da taxa — média ponderada).
- Doadores: demais divisões com vínculos suficientes para taxa estável
  (piso documentado no script); divisões minúsculas só adicionam ruído.
- Desfecho: taxa de mortalidade (óbitos por 100 mil vínculos), não contagem —
  sem normalizar, o ciclo de emprego da construção se disfarça de efeito.
