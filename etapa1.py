import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


caminho_arquivo = "data/rodofauna.xlsx"
tamanho_segmento = 500

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("graficosE1", exist_ok=True)





# Descobre o trecho da rodovia a partir das coordenadas do registro
def descobrir_trecho_pela_localizacao(linha):
    if linha["y"] < 8245000:
        return "jbb_fal_ibge"
    if linha["x"] > 205000:
        return "esecae"
    return "pnb"


tabela = pd.read_excel(caminho_arquivo)

quantidade_linhas, quantidade_colunas = tabela.shape
print("Quantidade de registros:", quantidade_linhas)
print("Quantidade de atributos:", quantidade_colunas)

print("\nTipos de cada atributo:")
print(tabela.dtypes)

colunas_para_comparar = [coluna for coluna in tabela.columns if coluna != "OBJECTID"]
quantidade_duplicados = tabela.duplicated(subset=colunas_para_comparar).sum()
print("\nLinhas duplicadas:", quantidade_duplicados)

tabela = tabela.drop_duplicates(subset=colunas_para_comparar)
print("Registros depois de remover duplicados:", len(tabela))


print("\nValores da coluna silvestre_:")
print(tabela["silvestre_"].value_counts())

tabela_silvestre = tabela[tabela["silvestre_"] == "silvestre"]

print("\nRegistros antes do filtro:", len(tabela))
print("Registros depois do filtro:", len(tabela_silvestre))


total_celulas_vazia = tabela_silvestre.isna().sum().sum()
print("\nCélulas realmente vazias:", total_celulas_vazia)

quantidade_nao_identificado = (tabela_silvestre == "nao_identificado").sum()
percentual_nao_identificado = quantidade_nao_identificado / len(tabela_silvestre) * 100

resumo_ausentes = pd.DataFrame({
    "quantidade": quantidade_nao_identificado,
    "percentual": percentual_nao_identificado.round(1),
})

resumo_ausentes = resumo_ausentes[resumo_ausentes["quantidade"] > 0]
resumo_ausentes = resumo_ausentes.sort_values("quantidade", ascending=False)

print("\nValores 'nao_identificado' por coluna:")
print(resumo_ausentes)


quantidade_por_classe = tabela_silvestre["classe"].value_counts()
percentual_por_classe = (quantidade_por_classe / len(tabela_silvestre) * 100).round(1)

print("\nRegistros por classe:")
print(pd.DataFrame({"quantidade": quantidade_por_classe, "percentual": percentual_por_classe}))

quantidade_por_classe.plot(kind="bar", color="steelblue")
plt.title("Atropelamentos de fauna silvestre por classe")
plt.xlabel("Classe")
plt.ylabel("Quantidade de registros")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_classes.png", dpi=120)
plt.close()


colunas_numericas = ["x", "y", "ano"]
estatisticas = tabela_silvestre[colunas_numericas].describe().round(1)

print("\nEstatísticas descritivas:")
print(estatisticas)


tabela_silvestre = tabela_silvestre.copy()
tabela_silvestre["data"] = pd.to_datetime(tabela_silvestre["data_dt"], format="%d/%m/%y")
tabela_silvestre["mes_numero"] = tabela_silvestre["data"].dt.month
tabela_silvestre["dia_semana_numero"] = tabela_silvestre["data"].dt.dayofweek

registros_por_ano = tabela_silvestre["ano"].value_counts().sort_index()
registros_por_mes = tabela_silvestre["mes_numero"].value_counts().sort_index()
registros_por_dia_semana = tabela_silvestre["dia_semana_numero"].value_counts().sort_index()

print("\nPeríodo coberto:", tabela_silvestre["data"].min().date(), "até", tabela_silvestre["data"].max().date())

print("\nRegistros por ano:")
print(registros_por_ano)
print("\nRegistros por mês:")
print(registros_por_mes)
print("\nRegistros por dia da semana (0 = segunda):")
print(registros_por_dia_semana)

figura, eixos = plt.subplots(1, 3, figsize=(15, 4))
registros_por_ano.plot(kind="bar", ax=eixos[0], color="steelblue", title="Por ano")
registros_por_mes.plot(kind="bar", ax=eixos[1], color="steelblue", title="Por mês")
registros_por_dia_semana.plot(kind="bar", ax=eixos[2], color="steelblue", title="Por dia da semana (0 = segunda)")
for eixo in eixos:
    eixo.set_xlabel("")
    eixo.set_ylabel("Quantidade de registros")
    eixo.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_temporal.png", dpi=120)
plt.close()


registros_por_uc = tabela_silvestre["uc"].value_counts()
print("\nRegistros por unidade de conservação (uc):")
print(registros_por_uc)

plt.figure(figsize=(8, 7))
for nome_uc, registros_da_uc in tabela_silvestre.groupby("uc"):
    plt.scatter(registros_da_uc["x"], registros_da_uc["y"], s=4, label=nome_uc)
plt.title("Localização dos atropelamentos de fauna silvestre")
plt.xlabel("x (UTM, metros)")
plt.ylabel("y (UTM, metros)")
plt.legend(markerscale=3)
plt.axis("equal")
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_espacial.png", dpi=120)
plt.close()


tabela_silvestre["trecho_pela_localizacao"] = tabela_silvestre.apply(descobrir_trecho_pela_localizacao, axis=1)

print("\nuc registrada (linhas) x trecho pela localização (colunas):")
print(pd.crosstab(tabela_silvestre["uc"], tabela_silvestre["trecho_pela_localizacao"]))

uc_conhecida = tabela_silvestre["uc"] != "nao_identificado"
uc_diferente = tabela_silvestre["uc"] != tabela_silvestre["trecho_pela_localizacao"]
registros_inconsistentes = tabela_silvestre[uc_conhecida & uc_diferente]

print("\nRegistros com uc diferente do trecho onde estão:", len(registros_inconsistentes))
print(registros_inconsistentes["uc"].value_counts())


tabela_silvestre["celula_x"] = tabela_silvestre["x"] // tamanho_segmento
tabela_silvestre["celula_y"] = tabela_silvestre["y"] // tamanho_segmento
tabela_silvestre["segmento"] = tabela_silvestre["celula_x"].astype(str) + "_" + tabela_silvestre["celula_y"].astype(str)

atropelamentos_por_segmento = tabela_silvestre["segmento"].value_counts()

print("\nQuantidade de segmentos com pelo menos 1 atropelamento:", len(atropelamentos_por_segmento))
print("\nEstatísticas da contagem por segmento:")
print(atropelamentos_por_segmento.describe().round(1))
print("\nPercentil 75 (candidato a limiar do hotspot):", atropelamentos_por_segmento.quantile(0.75))

atropelamentos_por_segmento.plot(kind="hist", bins=30, color="steelblue")
plt.title("Atropelamentos por segmento de 500 m")
plt.xlabel("Atropelamentos no segmento")
plt.ylabel("Quantidade de segmentos")
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_contagem_segmentos.png", dpi=120)
plt.close()


primeiro_quartil = atropelamentos_por_segmento.quantile(0.25)
terceiro_quartil = atropelamentos_por_segmento.quantile(0.75)
intervalo_interquartil = terceiro_quartil - primeiro_quartil

limite_inferior = primeiro_quartil - 1.5 * intervalo_interquartil
limite_superior = terceiro_quartil + 1.5 * intervalo_interquartil

segmentos_outliers = atropelamentos_por_segmento[
    (atropelamentos_por_segmento < limite_inferior) | (atropelamentos_por_segmento > limite_superior)
]

print("\nQ1:", primeiro_quartil, "| Q3:", terceiro_quartil, "| Intervalo interquartil:", intervalo_interquartil)
print("Limites aceitos: de", limite_inferior, "até", limite_superior)
print("Segmentos outliers:", len(segmentos_outliers), "de", len(atropelamentos_por_segmento))
print("\nOs maiores segmentos:")
print(segmentos_outliers.sort_values(ascending=False).head(10))

plt.figure(figsize=(4, 6))
plt.boxplot(atropelamentos_por_segmento)
plt.title("Boxplot: atropelamentos por segmento de 500 m")
plt.ylabel("Atropelamentos no segmento")
plt.xticks([])
plt.tight_layout()
plt.savefig("graficosE1/outliers_segmentos.png", dpi=120)
plt.close()


colunas_para_correlacao = ["x", "y", "ano", "campanha", "periodo", "mes_numero", "dia_semana_numero"]
matriz_correlacao = tabela_silvestre[colunas_para_correlacao].corr().round(2)
print("\nMatriz de correlação de Pearson:")
print(matriz_correlacao.to_string())

print("\nQuantidade de valores distintos por coluna (colunas com poucos valores ou que repetem outra são candidatas a redundantes):")
print(tabela_silvestre.nunique().sort_values())

print("\nPares com correlação alta (|r| >= 0.8), candidatos a atributos redundantes:")
for i, coluna_a in enumerate(colunas_para_correlacao):
    for coluna_b in colunas_para_correlacao[i + 1:]:
        if abs(matriz_correlacao.loc[coluna_a, coluna_b]) >= 0.8:
            print(" ", coluna_a, "x", coluna_b, "=", matriz_correlacao.loc[coluna_a, coluna_b])


tabela_classe_trecho = tabela_silvestre[tabela_silvestre["classe"] != "nao_identificado"]

contagem_classe_trecho = pd.crosstab(tabela_classe_trecho["trecho_pela_localizacao"], tabela_classe_trecho["classe"])
percentual_classe_trecho = (pd.crosstab(tabela_classe_trecho["trecho_pela_localizacao"], tabela_classe_trecho["classe"], normalize="index") * 100).round(1)

qui_quadrado, valor_p, graus_liberdade, esperado = chi2_contingency(contagem_classe_trecho)
total_registros = contagem_classe_trecho.to_numpy().sum()
menor_dimensao = min(contagem_classe_trecho.shape) - 1
v_de_cramer = (qui_quadrado / (total_registros * menor_dimensao)) ** 0.5

print("\nQuantidade de registros por trecho e classe:")
print(contagem_classe_trecho)
print("\nPercentual de cada classe dentro do trecho (%):")
print(percentual_classe_trecho)
print("\nTeste qui-quadrado (classe x trecho):")
print("Qui-quadrado:", round(qui_quadrado, 1), "| graus de liberdade:", graus_liberdade, "| p-valor:", valor_p)
print("V de Cramér:", round(v_de_cramer, 2))


registros_por_classe_valida = tabela_silvestre[tabela_silvestre["classe"] != "nao_identificado"]["classe"].value_counts()
print("\nMenor classe do multiclasse:", registros_por_classe_valida.idxmin(), "com", registros_por_classe_valida.min(), "registros")

especies_identificadas = tabela_silvestre[tabela_silvestre["especie_nm"] != "nao_identificado"]
registros_por_especie = especies_identificadas["especie_nm"].value_counts()

for minimo_registros in [5, 10, 30, 50]:
    especies_com_minimo = (registros_por_especie >= minimo_registros).sum()
    registros_cobertos = registros_por_especie[registros_por_especie >= minimo_registros].sum()
    print("Espécies com pelo menos", minimo_registros, "registros:", especies_com_minimo,
          "(cobrem", registros_cobertos, "registros)")


caminho_saida = "data/rodofauna_etapa1.csv"
os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

tabela_saida = tabela_silvestre.copy()
tabela_saida["data"] = tabela_saida["data"].dt.strftime("%Y-%m-%d")
tabela_saida.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

print("\nTabela salva em", caminho_saida, "com", len(tabela_saida), "registros e", tabela_saida.shape[1], "colunas")
print("Colunas:", list(tabela_saida.columns))