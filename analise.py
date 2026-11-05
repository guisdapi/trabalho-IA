"""
ETAPA 1 - ANALISE EXPLORATORIA - PROJETO RODOFAUNA

Como rodar:   python etapa1_eda_rodofauna.py > resultados.txt
  -> os numeros vao para o arquivo resultados.txt
  -> os graficos sao salvos na pasta "graficos"

Pre-requisito: ter o arquivo "rodofauna_bruto.csv" na mesma pasta do script
(baixado do link do trabalho). Se nao tiver, o script baixa sozinho.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # salva graficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("graficos", exist_ok=True)     # cria a pasta dos graficos
sns.set_style("whitegrid")


def salvar_grafico(nome_arquivo):
    """Salva o grafico atual na pasta 'graficos' e fecha."""
    plt.tight_layout()
    plt.savefig("graficos/" + nome_arquivo, dpi=150)
    plt.close()


# =====================================================================
# PASSO 1 - CARREGAR OS DADOS
# =====================================================================
NOME_ARQUIVO = "rodofauna_bruto.csv"

if os.path.exists(NOME_ARQUIVO):
    tabela_completa = pd.read_csv(NOME_ARQUIVO)
else:
    # Baixa os dados do servidor do IBRAM (500 ou 1000 registros por vez)
    import requests
    endereco = ("https://onda.ibram.df.gov.br/server/rest/services/"
                "Monitoramento_Ambiental/Programa_Rodofauna_Atropelada/"
                "FeatureServer/2/query")
    parametros = {"where": "1=1", "outFields": "*", "f": "json",
                  "returnGeometry": "false", "resultOffset": 0,
                  "resultRecordCount": 1000}
    lista_registros = []
    while True:
        resposta = requests.get(endereco, params=parametros, timeout=60).json()
        pagina = resposta.get("features", [])
        if len(pagina) == 0:
            break
        for item in pagina:
            lista_registros.append(item["attributes"])
        parametros["resultOffset"] += len(pagina)
    tabela_completa = pd.DataFrame(lista_registros)
    tabela_completa.to_csv(NOME_ARQUIVO, index=False)

# nomes das colunas em minusculo, para evitar erro de digitacao
tabela_completa.columns = tabela_completa.columns.str.strip().str.lower()


# =====================================================================
# PASSO 2 - FICAR SO COM FAUNA SILVESTRE (exigido pelo enunciado)
# =====================================================================
tabela_completa["silvestre_"] = tabela_completa["silvestre_"].astype(str).str.strip().str.lower()
tabela = tabela_completa[tabela_completa["silvestre_"] == "silvestre"].copy()

# A partir daqui, "tabela" = somente registros silvestres.

# Converter a coluna de data para o tipo data
if pd.api.types.is_numeric_dtype(tabela["data_dt"]):
    tabela["data_dt"] = pd.to_datetime(tabela["data_dt"], unit="ms", errors="coerce")
else:
    tabela["data_dt"] = pd.to_datetime(tabela["data_dt"], dayfirst=True, errors="coerce")

# Criar colunas auxiliares de tempo (ano, mes, dia da semana)
tabela["ano_da_data"] = tabela["data_dt"].dt.year
tabela["mes_da_data"] = tabela["data_dt"].dt.month
tabela["dia_da_semana"] = tabela["data_dt"].dt.dayofweek    # 0 = segunda, 6 = domingo


# =====================================================================
# PASSO 3 - TAMANHO E TIPOS DOS DADOS
# =====================================================================
print("=== QUANTIDADE DE REGISTROS E ATRIBUTOS ===")
print("Registros (todos):     ", len(tabela_completa))
print("Registros (silvestres):", len(tabela))
print("Numero de atributos:   ", tabela.shape[1])

print("\n=== TIPO DE CADA ATRIBUTO ===")
print(tabela.dtypes.to_string())


# =====================================================================
# PASSO 4 - VALORES AUSENTES
# Existem dois tipos de "vazio" neste dataset:
#   a) celula realmente vazia (NaN)
#   b) texto "nao_identificado" (o pandas NAO considera isso como vazio)
# =====================================================================
print("\n=== VALORES AUSENTES POR ATRIBUTO ===")
resumo_ausentes = pd.DataFrame(index=tabela.columns)
resumo_ausentes["vazios_nan"] = tabela.isna().sum()

quantidade_nao_identificado = []
for coluna in tabela.columns:
    if tabela[coluna].dtype == "object":    # so colunas de texto podem ter "nao_identificado"
        total = (tabela[coluna].astype(str).str.lower() == "nao_identificado").sum()
    else:
        total = 0
    quantidade_nao_identificado.append(total)
resumo_ausentes["nao_identificado"] = quantidade_nao_identificado

soma_vazios = resumo_ausentes["vazios_nan"] + resumo_ausentes["nao_identificado"]
resumo_ausentes["porcentagem_vazia"] = (soma_vazios / len(tabela) * 100).round(2)
print(resumo_ausentes.sort_values("porcentagem_vazia", ascending=False).to_string())


# =====================================================================
# PASSO 5 - DISTRIBUICAO DAS CLASSES (aves, mamiferos, repteis, anfibios)
# Serve para ver se as classes sao desbalanceadas.
# =====================================================================
contagem_por_classe = tabela["classe"].value_counts()
porcentagem_por_classe = (contagem_por_classe / len(tabela) * 100).round(2)

print("\n=== REGISTROS POR CLASSE ===")
print(pd.DataFrame({"quantidade": contagem_por_classe,
                    "porcentagem": porcentagem_por_classe}).to_string())

razao = contagem_por_classe.max() / contagem_por_classe.min()
print("\nRazao de desbalanceamento (maior classe / menor classe):", round(razao, 2))

plt.figure(figsize=(7, 4))
sns.countplot(data=tabela, y="classe", order=contagem_por_classe.index, color="steelblue")
plt.title("Registros por classe taxonomica")
salvar_grafico("classes.png")

# Especies: ha exemplos suficientes de cada uma?
MINIMO_EXEMPLOS = 30
contagem_por_especie = tabela["especie_nm"].value_counts()
print("\n=== ESPECIES ===")
print("Numero de especies:", len(contagem_por_especie))
print("Especies com menos de", MINIMO_EXEMPLOS, "registros:",
      (contagem_por_especie < MINIMO_EXEMPLOS).sum())
print("\n10 especies mais frequentes:")
print(contagem_por_especie.head(10).to_string())

plt.figure(figsize=(8, 6))
contagem_por_especie.head(20).sort_values().plot.barh()
plt.title("20 especies mais frequentes")
salvar_grafico("top20_especies.png")


# =====================================================================
# PASSO 6 - ESTATISTICAS DESCRITIVAS (media, desvio, minimo, maximo...)
# =====================================================================
print("\n=== ESTATISTICAS DESCRITIVAS ===")
print(tabela[["ano", "x", "y"]].describe().T.round(2).to_string())


# =====================================================================
# PASSO 7 - DISTRIBUICAO ESPACIAL (onde ocorrem os atropelamentos)
# x e y sao as coordenadas em metros.
# =====================================================================
plt.figure(figsize=(8, 8))
plt.scatter(tabela["x"], tabela["y"], s=3, alpha=0.3)
plt.axis("equal")
plt.title("Localizacao dos atropelamentos")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
salvar_grafico("mapa_pontos.png")

# Mapa de calor: regioes mais escuras/vermelhas = mais atropelamentos
plt.figure(figsize=(9, 8))
plt.hexbin(tabela["x"], tabela["y"], gridsize=60, cmap="YlOrRd", mincnt=1)
plt.colorbar(label="Numero de atropelamentos")
plt.axis("equal")
plt.title("Densidade de atropelamentos")
salvar_grafico("mapa_calor.png")

print("\n=== REGISTROS POR UNIDADE DE CONSERVACAO (uc) ===")
print(tabela["uc"].value_counts(dropna=False).to_string())


# =====================================================================
# PASSO 8 - DISTRIBUICAO TEMPORAL (quando ocorrem)
# =====================================================================
print("\n=== TEMPORAL ===")
print("Primeira data:", tabela["data_dt"].min().date())
print("Ultima data:  ", tabela["data_dt"].max().date())
print("\nRegistros por ano:")
print(tabela["ano_da_data"].value_counts().sort_index().to_string())
print("\nRegistros por mes:")
print(tabela["mes_da_data"].value_counts().sort_index().to_string())

figura, eixos = plt.subplots(1, 3, figsize=(15, 4))
tabela["ano_da_data"].value_counts().sort_index().plot.bar(ax=eixos[0], title="Por ano")
tabela["mes_da_data"].value_counts().sort_index().plot.bar(ax=eixos[1], title="Por mes")
tabela["dia_da_semana"].value_counts().sort_index().plot.bar(ax=eixos[2], title="Por dia da semana (0 = segunda)")
salvar_grafico("temporal.png")


# =====================================================================
# PASSO 9 - OUTLIERS (valores muito fora do padrao)
# Regra do IQR: e outlier quem esta muito abaixo do 1o quartil
# ou muito acima do 3o quartil.
# =====================================================================
print("\n=== OUTLIERS EM x E y ===")
for coluna in ["x", "y"]:
    primeiro_quartil = tabela[coluna].quantile(0.25)
    terceiro_quartil = tabela[coluna].quantile(0.75)
    amplitude = terceiro_quartil - primeiro_quartil
    limite_inferior = primeiro_quartil - 1.5 * amplitude
    limite_superior = terceiro_quartil + 1.5 * amplitude
    quantidade = ((tabela[coluna] < limite_inferior) | (tabela[coluna] > limite_superior)).sum()
    print(coluna, ":", quantidade, "outliers")

figura, eixos = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(y=tabela["x"], ax=eixos[0])
sns.boxplot(y=tabela["y"], ax=eixos[1])
salvar_grafico("boxplots.png")


# =====================================================================
# PASSO 10 - ATRIBUTOS REDUNDANTES (que repetem a mesma informacao)
# =====================================================================
print("\n=== ATRIBUTOS COM VALOR UNICO (constantes) ===")
constantes = [c for c in tabela.columns if tabela[c].nunique() <= 1]
print(constantes)

print("\n=== QUANTOS VALORES DIFERENTES TEM CADA ATRIBUTO TAXONOMICO ===")
colunas_taxonomia = [c for c in ["classe", "ordem", "familia", "especie_nm", "nome_popul"]
                     if c in tabela.columns]
print(tabela[colunas_taxonomia].nunique().to_string())


# =====================================================================
# PASSO 11 - CORRELACAO ENTRE ATRIBUTOS NUMERICOS
# Valores perto de 1 ou -1 = atributos muito parecidos.
# =====================================================================
print("\n=== CORRELACAO ===")
matriz_correlacao = tabela[["ano_da_data", "mes_da_data", "dia_da_semana", "x", "y"]].corr()
print(matriz_correlacao.round(3).to_string())

plt.figure(figsize=(6, 5))
sns.heatmap(matriz_correlacao, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlacao entre atributos")
salvar_grafico("correlacao.png")

print("\nFim. Graficos salvos na pasta 'graficos'.")