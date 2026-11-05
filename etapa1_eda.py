# -*- coding: utf-8 -*-
"""
Trabalho de Aprendizado Supervisionado - Inteligência Artificial - UFSM
ETAPA 1 - Análise Exploratória do Dataset (RodoFauna / IBRAM-DF)

Uso:
    python etapa1_eda.py                       (lê data/rodofauna.xlsx)
    python etapa1_eda.py caminho/da/base.xlsx

Saídas (pasta resultados_etapa1/):
    resultados_etapa1.md    relatório com texto, tabelas e gráficos
    fig01 ... fig10 .png    gráficos (usados pelo .md)

Itens do enunciado cobertos (na ordem das seções do terminal):
    1. Descrição do problema
    2. Quantidade de registros e de atributos, tipos dos atributos
    3. Valores ausentes e 'nao_identificado'
    4. Estatísticas descritivas das variáveis numéricas e outliers
    5. Variável-alvo de cada experimento e distribuição das classes
    6. Distribuição espacial
    7. Distribuição temporal
    8. Atributos redundantes e correlação
    9. Resumo dos problemas encontrados
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
ARQUIVO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/rodofauna.xlsx")
SAIDA = Path("resultados_etapa1")
SAIDA.mkdir(exist_ok=True)

NI = "nao_identificado"
TAM_CELULA = 500  # metros - unidade espacial usada para os alvos binário e de regressão

# Campos que o enunciado proíbe como atributos preditores
PROIBIDOS = ["objectid", "id", "silvestre_", "hora", "sexo", "posicao", "ordem",
             "familia", "nome_popul", "campanha", "periodo", "mes"]

CORES_CLASSE = {"birds": "#2a78d6", "mammals": "#eb6834", "reptiles": "#1baf7a",
                "amphibians": "#eda100"}
NOME_CLASSE = {"birds": "Aves", "mammals": "Mamíferos", "reptiles": "Répteis",
               "amphibians": "Anfíbios"}
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
DIAS = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]
AZUL = "#2a78d6"
LARANJA = "#eb6834"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})

# Tudo que é impresso no terminal também vai para resultados_etapa1.md
md = ["# Etapa 1 - Análise Exploratória do Dataset (RodoFauna / IBRAM-DF)", ""]


def log(texto=""):
    """Imprime uma linha e a adiciona ao .md (linhas recuadas viram itens de lista)."""
    print(texto)
    texto = str(texto)
    if texto.startswith("   "):
        md.append("- " + texto.strip())
    elif texto == "":
        md.append("")
    else:
        md.append(texto + "  ")  # dois espaços = quebra de linha no Markdown


def paragrafo(texto):
    """Texto corrido: quebrado em linhas no terminal, um parágrafo só no .md."""
    print(textwrap.fill(texto, 78))
    md.extend([texto, ""])


def _fmt(v):
    if pd.isna(v):
        return "-"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def tabela(obj, nome_valor="registros"):
    """Imprime a tabela no terminal e a escreve como tabela Markdown no .md."""
    if isinstance(obj, pd.Series):
        obj = obj.rename(nome_valor).to_frame()
    print(obj.to_string())
    d = obj.reset_index()
    d.columns = ["" if str(c) == "index" else str(c) for c in d.columns]
    md.append("")
    md.append("| " + " | ".join(d.columns) + " |")
    md.append("|" + "---|" * len(d.columns))
    for linha in d.itertuples(index=False):
        md.append("| " + " | ".join(_fmt(v) for v in linha) + " |")
    md.append("")


def titulo(texto):
    print()
    print("=" * 78)
    print(texto.upper())
    print("=" * 78)
    md.extend(["", f"## {texto}", ""])


def subtitulo(texto):
    print()
    print(texto)
    md.extend(["", f"### {texto}", ""])


def salvar(fig, nome):
    fig.savefig(SAIDA / nome)
    plt.close(fig)
    print(f"   [figura salva: {nome}]")
    legenda = LEGENDAS.get(nome, nome)
    md.extend(["", f"![{legenda}]({nome})", f"*{legenda}*", ""])


LEGENDAS = {
    "fig01_ausentes.png": "Figura 1 - Percentual de valores ausentes/nao_identificado por atributo",
    "fig02_boxplot_coordenadas.png": "Figura 2 - Boxplots das coordenadas x e y",
    "fig03_distribuicao_classe.png": "Figura 3 - Distribuição da variável-alvo classe",
    "fig04_top_especies.png": "Figura 4 - 15 espécies mais atropeladas",
    "fig05_atropelamentos_por_celula.png": "Figura 5 - Atropelamentos por célula de 500 m",
    "fig06_mapa_por_classe.png": "Figura 6 - Distribuição espacial por classe",
    "fig07_densidade_espacial.png": "Figura 7 - Densidade espacial dos atropelamentos",
    "fig08_serie_mensal.png": "Figura 8 - Série mensal de atropelamentos",
    "fig09_mes_e_dia_semana.png": "Figura 9 - Atropelamentos por mês do ano e dia da semana",
    "fig10_correlacoes.png": "Figura 10 - Correlação (Spearman) e associação (V de Cramér)",
}


problemas = []  # lista de problemas encontrados, impressa no final


# ---------------------------------------------------------------------------
# 1. Descrição do problema
# ---------------------------------------------------------------------------
titulo("1. Descrição do problema")
paragrafo("Base do Projeto RodoFauna (IBRAM/DF): animais encontrados atropelados em "
          "percursos sistemáticos nas rodovias do DF entre abr/2010 e mar/2015. "
          "Cada linha é uma ocorrência (um animal), com data, classificação taxonômica, "
          "unidade de conservação (uc) e coordenadas UTM (x, y).")
paragrafo("Objetivo: prever o tipo (classe) e a intensidade/ocorrência de atropelamentos "
          "de fauna SILVESTRE e identificar trechos críticos (hotspots).")

# ---------------------------------------------------------------------------
# 2. Registros, atributos e tipos
# ---------------------------------------------------------------------------
titulo("2. Registros, atributos e tipos")

bruto = pd.read_excel(ARQUIVO)
bruto.columns = [c.strip().lower() for c in bruto.columns]
for c in bruto.columns:  # padroniza os textos (minúsculas, sem espaços nas pontas)
    if not pd.api.types.is_string_dtype(bruto[c]):
        continue
    bruto[c] = bruto[c].astype("string").str.strip().str.lower()

log(f"Arquivo: {ARQUIVO}")
log(f"Base bruta: {len(bruto)} registros x {bruto.shape[1]} atributos")
log()
log("Tipos dos atributos:")
tipos = pd.DataFrame({
    "tipo": bruto.dtypes.astype(str),
    "valores_distintos": bruto.nunique(),
    "pode_ser_preditor": ["não" if c in PROIBIDOS else
                          ("alvo" if c in ("classe", "especie_nm") else "sim")
                          for c in bruto.columns],
})
tabela(tipos)

log()
log("silvestre_ (base bruta):")
tabela(bruto["silvestre_"].value_counts())

# Filtra apenas fauna silvestre (exigência do enunciado)
df = bruto[bruto["silvestre_"] == "silvestre"].copy()
log()
log(f"Após filtrar silvestre_ == 'silvestre': {len(df)} registros")

# Duplicatas: mesma linha repetida (ignorando o objectid, que é só o nº da linha)
colunas_sem_objectid = [c for c in df.columns if c != "objectid"]
n_dup = df.duplicated(subset=colunas_sem_objectid).sum()
log(f"Registros duplicados (todas as colunas iguais, exceto objectid): {n_dup}")
if "id" in df.columns:
    log(f"Valores distintos de 'id' na base bruta: {bruto['id'].nunique()} "
        f"(em {len(bruto)} linhas)")
if n_dup > 0:
    problemas.append(f"{n_dup} registros silvestres duplicados ({n_dup / len(df):.1%}). "
                     "Sem removê-los, todas as contagens ficam infladas.")
    df = df.drop_duplicates(subset=colunas_sem_objectid)
    log(f"-> Duplicatas removidas. Base analisada daqui em diante: {len(df)} registros")

# Converte a data e cria os atributos temporais usados nas análises abaixo
if pd.api.types.is_datetime64_any_dtype(df["data_dt"]):  # Excel já trouxe como data
    df["data"] = df["data_dt"]
else:  # texto no formato dd/mm/aa
    df["data"] = pd.to_datetime(df["data_dt"].astype(str), format="%d/%m/%y", errors="coerce")
df["mes_num"] = df["data"].dt.month
df["dia_do_ano"] = df["data"].dt.dayofyear
df["dia_semana_num"] = df["data"].dt.dayofweek  # 0 = segunda
df["periodo_climatico"] = np.where(df["mes_num"].between(5, 9), "seco", "chuvoso")
log(f"Datas que não puderam ser convertidas: {df['data'].isna().sum()}")

# ---------------------------------------------------------------------------
# 3. Valores ausentes e 'nao_identificado'
# ---------------------------------------------------------------------------
titulo("3. Valores ausentes e 'nao_identificado'")

originais = [c for c in bruto.columns]
ausentes = pd.DataFrame({
    "ausentes": df[originais].isna().sum(),
    "nao_identificado": (df[originais] == NI).sum(),
})
ausentes["%_sem_info"] = 100 * (ausentes["ausentes"] + ausentes["nao_identificado"]) / len(df)
ausentes = ausentes.sort_values("%_sem_info", ascending=False)
ausentes = ausentes[ausentes["%_sem_info"] > 0]
tabela(ausentes.round(2)) if len(ausentes) else log("Nenhum valor ausente.")

fig, ax = plt.subplots(figsize=(8, 0.45 * len(ausentes) + 1.2))
cores = [LARANJA if c in PROIBIDOS else AZUL for c in ausentes.index]
ax.barh(ausentes.index, ausentes["%_sem_info"], color=cores)
ax.invert_yaxis()
for i, v in enumerate(ausentes["%_sem_info"]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlim(0, 105)
ax.set_xlabel("% dos registros silvestres (ausente + nao_identificado)")
ax.set_title("Atributos com informação faltante")
salvar(fig, "fig01_ausentes.png")

for c, v in ausentes["%_sem_info"].items():
    if v >= 30:
        obs = " (já excluído como preditor pelo enunciado)" if c in PROIBIDOS else ""
        problemas.append(f"'{c}' sem informação em {v:.0f}% dos registros{obs}.")

# ---------------------------------------------------------------------------
# 4. Estatísticas descritivas e outliers
# ---------------------------------------------------------------------------
titulo("4. Estatísticas descritivas e outliers")

numericas = ["x", "y", "ano", "mes_num", "dia_do_ano"]
desc = df[numericas].describe().T.round(2)
desc.columns = ["n", "média", "desvio", "mín", "25%", "mediana", "75%", "máx"]
tabela(desc)

log()
log("Outliers em x e y (critério de Tukey: fora de [Q1 - 1,5*IQR, Q3 + 1,5*IQR]):")
for c in ["x", "y"]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    log(f"   {c}: {n_out} outliers")
# Limites aproximados do DF em UTM fuso 23S (com folga de ~10 km)
fora_df = ~(df["x"].between(135_000, 265_000) & df["y"].between(8_210_000, 8_300_000))
log(f"   Pontos fora dos limites do DF: {fora_df.sum()}")
if fora_df.sum() > 0:
    problemas.append(f"{fora_df.sum()} pontos com coordenadas fora do DF (provável erro).")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, c in zip(axs, ["x", "y"]):
    sns.boxplot(y=df[c], ax=ax, color=AZUL, width=0.4)
    ax.set_title(f"Boxplot de {c} (UTM, m)")
    ax.set_ylabel("")
    ax.ticklabel_format(style="plain", axis="y")
salvar(fig, "fig02_boxplot_coordenadas.png")

# ---------------------------------------------------------------------------
# 5. Variáveis-alvo e distribuição das classes
# ---------------------------------------------------------------------------
titulo("5. Variáveis-alvo e distribuição das classes")

# 5.1 Multiclasse: classe
subtitulo("5.1 Classificação multiclasse -> alvo: 'classe'")
cont = df["classe"].value_counts()
tabela(pd.DataFrame({"registros": cont, "%": (100 * cont / cont.sum()).round(2)}))
validas = cont.drop(NI, errors="ignore")
razao = validas.max() / validas.min()
log(f"Razão maior/menor classe: {razao:.1f} : 1")
log(f"Acurácia de quem sempre chuta '{validas.idxmax()}': {validas.max() / validas.sum():.1%}")
if razao > 3:
    problemas.append(f"'classe' desbalanceada ({razao:.1f}:1): '{validas.idxmax()}' tem "
                     f"{validas.max() / validas.sum():.0%} dos registros. Acurácia sozinha é enganosa.")
if NI in cont.index:
    problemas.append(f"{cont[NI]} registro(s) com classe = nao_identificado (remover do multiclasse).")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([NOME_CLASSE.get(c, c) for c in validas.index], validas.values,
       color=[CORES_CLASSE.get(c, "gray") for c in validas.index])
for i, v in enumerate(validas.values):
    ax.text(i, v, f"{v}\n({v / validas.sum():.1%})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, validas.max() * 1.2)
ax.set_ylabel("registros")
ax.set_title("Distribuição da variável-alvo 'classe'")
ax.grid(axis="x", visible=False)
salvar(fig, "fig03_distribuicao_classe.png")

# 5.2 Espécie (alvo alternativo) - verifica classes com poucos exemplos
subtitulo("5.2 Alvo alternativo 'especie_nm' (classes com poucos exemplos)")
esp = df.loc[df["especie_nm"] != NI, "especie_nm"].value_counts()
log(f"Espécies distintas: {len(esp)} | registros sem espécie identificada: "
    f"{(df['especie_nm'] == NI).sum()}")
for k in (10, 30, 100):
    log(f"   espécies com >= {k:>3} registros: {(esp >= k).sum()}")
log(f"   espécies com < 10 registros: {(esp < 10).sum()} ({(esp < 10).mean():.0%} das espécies)")
problemas.append(f"Espécies: {(esp < 10).sum()} das {len(esp)} têm menos de 10 registros -> "
                 "classificar por espécie não é viável sem agrupar; por isso o multiclasse usa 'classe'.")

top = df[df["especie_nm"] != NI].groupby("especie_nm")["classe"].agg(["size", "first"])
top = top.sort_values("size", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.index.str.replace("_", " "), top["size"],
        color=[CORES_CLASSE.get(c, "gray") for c in top["first"]])
ax.invert_yaxis()
ax.set_xlabel("registros")
ax.set_title("15 espécies mais atropeladas (cor = classe)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v, label=NOME_CLASSE[k])
                   for k, v in CORES_CLASSE.items()], loc="lower right")
salvar(fig, "fig04_top_especies.png")

# 5.3 e 5.4 Alvos binário e de regressão (dependem de agregação espacial)
subtitulo(f"5.3 e 5.4 Alvos binário e de regressão (células de {TAM_CELULA} m)")
log(f"Classificação binária -> alvo: hotspot (1/0) por célula de {TAM_CELULA} m")
log(f"Regressão -> alvo: nº de atropelamentos por célula de {TAM_CELULA} m por mês")
df["celula"] = ((df["x"] // TAM_CELULA).astype(int).astype(str) + "_" +
                (df["y"] // TAM_CELULA).astype(int).astype(str))
por_celula = df.groupby("celula").size()
log(f"Células de {TAM_CELULA} m com pelo menos 1 atropelamento: {len(por_celula)}")
log("Atropelamentos por célula: " + ", ".join(
    f"{k} = {v:.1f}" for k, v in por_celula.describe()[["mean", "50%", "max"]]
    .rename({"mean": "média", "50%": "mediana", "max": "máximo"}).items()))

meses_periodo = df["data"].dt.to_period("M").nunique()
painel = df.groupby(["celula", df["data"].dt.to_period("M")]).size()
n_obs = len(por_celula) * meses_periodo
pct_zeros = 1 - len(painel) / n_obs
log(f"Painel célula x mês: {len(por_celula)} células x {meses_periodo} meses = {n_obs} "
    f"observações, {pct_zeros:.0%} delas com zero atropelamentos")
log("Obs.: a base só tem ocorrências; a classe 0 (sem/baixo risco) do problema binário")
log("      precisa ser construída na Etapa 2 e justificada.")
if pct_zeros > 0.7:
    problemas.append(f"Alvo de regressão célula x mês tem {pct_zeros:.0%} de zeros "
                     "(muitos zeros; avaliar agregação maior, ex.: trimestre).")

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(por_celula, bins=30, color=AZUL, edgecolor="white")
ax.set_xlabel(f"atropelamentos na célula de {TAM_CELULA} m (2010-2015)")
ax.set_ylabel("nº de células")
ax.set_title("Concentração dos atropelamentos por célula")
salvar(fig, "fig05_atropelamentos_por_celula.png")

# ---------------------------------------------------------------------------
# 6. Distribuição espacial
# ---------------------------------------------------------------------------
titulo("6. Distribuição espacial")

log("Registros por unidade de conservação (uc):")
tabela(df["uc"].value_counts())
log()
log("10 células mais críticas:")
tabela(por_celula.sort_values(ascending=False).head(10).rename_axis("célula (x_y)"))
top10 = por_celula.sort_values(ascending=False).head(10).sum()
log(f"-> concentram {top10 / len(df):.1%} dos atropelamentos em "
    f"{10 / len(por_celula):.1%} das células")

mapa = df[~fora_df]  # mapas sem os pontos fora do DF, para não distorcer a escala
classes = [c for c in CORES_CLASSE if c in validas.index]
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, c in zip(axs.ravel(), classes):
    ax.scatter(mapa["x"], mapa["y"], s=3, color="#dddddd")
    sub = mapa[mapa["classe"] == c]
    ax.scatter(sub["x"], sub["y"], s=6, color=CORES_CLASSE[c])
    ax.set_title(f"{NOME_CLASSE[c]} (n = {len(sub)})")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
fig.suptitle("Ocorrências por classe (cinza = todas)", fontweight="bold")
salvar(fig, "fig06_mapa_por_classe.png")

fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(mapa["x"], mapa["y"], gridsize=50, cmap="Blues", mincnt=1)
fig.colorbar(hb, ax=ax, label="atropelamentos")
ax.set_aspect("equal")
ax.ticklabel_format(style="plain")
ax.tick_params(axis="x", rotation=30)
ax.set_xlabel("x (UTM)")
ax.set_ylabel("y (UTM)")
ax.set_title("Densidade espacial dos atropelamentos")
salvar(fig, "fig07_densidade_espacial.png")

# ---------------------------------------------------------------------------
# 7. Distribuição temporal
# ---------------------------------------------------------------------------
titulo("7. Distribuição temporal")

log(f"Período: {df['data'].min():%d/%m/%Y} a {df['data'].max():%d/%m/%Y}")
log("(2010 começa em abril e 2015 termina em março: anos incompletos)")
log()
log("Registros por ano:")
tabela(df["data"].dt.year.value_counts().sort_index().rename_axis("ano"))
log()
log("Registros por mês do ano:")
por_mes = df["mes_num"].value_counts().reindex(range(1, 13), fill_value=0)
por_mes.index = MESES
tabela(por_mes.rename_axis("mês"))
log()
log("Registros por dia da semana:")
por_dia = df["dia_semana_num"].value_counts().reindex(range(7), fill_value=0)
por_dia.index = DIAS
tabela(por_dia.rename_axis("dia"))
if por_dia[["sab", "dom"]].sum() == 0:
    problemas.append("Nenhum registro em sábado/domingo: as coletas só ocorreram em dias úteis, "
                     "então 'dia_da_sem' reflete a agenda de coleta, não o risco.")
log()
log("Período climático (seco = mai-set, chuvoso = out-abr):")
pc = df["periodo_climatico"].value_counts()
log(f"   chuvoso: {pc.get('chuvoso', 0)} ({pc.get('chuvoso', 0) / 7:.0f} por mês)")
log(f"   seco:    {pc.get('seco', 0)} ({pc.get('seco', 0) / 5:.0f} por mês)")

serie = df.groupby(df["data"].dt.to_period("M")).size()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(serie.index.to_timestamp(), serie.values, color=AZUL, marker="o", ms=3)
ax.set_ylabel("registros")
ax.set_title("Atropelamentos por mês (abr/2010 - mar/2015)")
salvar(fig, "fig08_serie_mensal.png")

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})
axs[0].bar(MESES, por_mes.values,
           color=[LARANJA if 5 <= m <= 9 else AZUL for m in range(1, 13)])
axs[0].set_title("Por mês do ano (laranja = estação seca)")
axs[0].set_ylabel("registros")
axs[1].bar(DIAS, por_dia.values, color=AZUL)
axs[1].set_title("Por dia da semana")
for ax in axs:
    ax.grid(axis="x", visible=False)
salvar(fig, "fig09_mes_e_dia_semana.png")

# ---------------------------------------------------------------------------
# 8. Redundância e correlação
# ---------------------------------------------------------------------------
titulo("8. Atributos redundantes e correlação")

# Campos que repetem a informação de data_dt
log("Concordância dos campos originais com data_dt:")
ok_ano = (df["ano"] == df["data"].dt.year).mean()
log(f"   ano        x data_dt: {ok_ano:.1%}")
dia_orig = df["dia_da_sem"].str[:3].map({d: i for i, d in enumerate(DIAS)})
ok_dia = (dia_orig == df["dia_semana_num"]).mean()
log(f"   dia_da_sem x data_dt: {ok_dia:.1%}  ({(dia_orig != df['dia_semana_num']).sum()} divergências)")
log(f"   valores de dia_da_sem: {df['dia_da_sem'].value_counts().to_dict()}")
log("-> ano, mes e dia_da_sem são deriváveis de data_dt (redundantes).")
if ok_dia < 1:
    problemas.append(f"'dia_da_sem' diverge da data em {(dia_orig != df['dia_semana_num']).sum()} "
                     "registros (erro de digitação) - derivar o dia da semana de data_dt.")

# Hierarquia taxonômica: espécie determina a classe -> vazamento se usada como entrada
esp_classe = df[(df["especie_nm"] != NI) & (df["classe"] != NI)].groupby("especie_nm")["classe"].nunique()
log(f"Espécies associadas a mais de uma classe: {(esp_classe > 1).sum()} "
    "-> especie_nm/ordem/familia determinam a classe (não podem ser entrada).")

# uc praticamente definida pela posição
pureza = df.groupby("celula")["uc"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
log(f"Células de {TAM_CELULA} m com uma única uc: {(pureza == 1).mean():.1%} "
    "-> uc é quase redundante com (x, y).")

# Correlação entre numéricas (Spearman)
corr = df[["x", "y", "ano", "mes_num", "dia_do_ano", "dia_semana_num"]].corr(method="spearman")
log()
log("Correlação de Spearman (numéricas):")
tabela(corr.round(2))


# Associação entre categóricas (V de Cramér: 0 = nenhuma, 1 = total)
def cramers_v(a, b):
    tabela = pd.crosstab(a, b)
    chi2 = chi2_contingency(tabela)[0]
    return np.sqrt(chi2 / (tabela.values.sum() * (min(tabela.shape) - 1)))


categoricas = ["classe", "uc", "periodo_climatico", "ano", "mes_num"]
V = pd.DataFrame(index=categoricas, columns=categoricas, dtype=float)
for a in categoricas:
    for b in categoricas:
        V.loc[a, b] = 1.0 if a == b else cramers_v(df[a].astype(str), df[b].astype(str))
log()
log("V de Cramér (categóricas):")
tabela(V.round(2))

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
fig.subplots_adjust(wspace=0.4)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axs[0])
axs[0].set_title("Spearman (numéricas)")
sns.heatmap(V, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axs[1])
axs[1].set_title("V de Cramér (categóricas)")
salvar(fig, "fig10_correlacoes.png")

fortes = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns)
          for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.7]
fortes += [(a, b, V.loc[a, b]) for i, a in enumerate(categoricas)
           for b in categoricas[i + 1:] if V.loc[a, b] >= 0.7]
if fortes:
    problemas.append("Pares fortemente associados (>= 0,7): " +
                     "; ".join(f"{a} x {b} ({v:.2f})" for a, b, v in fortes) +
                     " -> manter só um de cada par.")

# ---------------------------------------------------------------------------
# 9. Resumo dos problemas
# ---------------------------------------------------------------------------
titulo("9. Resumo dos problemas encontrados")
for i, p in enumerate(problemas, 1):
    log(f"{i}. {p}")

(SAIDA / "resultados_etapa1.md").write_text("\n".join(md), encoding="utf-8")
print(f"\nResultados salvos em: {SAIDA.resolve()}")