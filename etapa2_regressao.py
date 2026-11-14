import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


pasta_graficos = "graficosE2"
pasta_saida = "data/etapa2"
caminho_registros = os.path.join(pasta_saida, "registros_com_periodo.csv")
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")
semente = 42
proporcao_validacao = 0.15
proporcao_teste = 0.15
periodos_treino = [1, 2, 3]
periodos_validacao = [4]
periodos_teste = [5]
tamanho_segmento = 500

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_saida, exist_ok=True)





# Divide os dados aleatoriamente em treino, validação e teste
def dividir_aleatorio(tabela_problema):
    indices = np.arange(len(tabela_problema))
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, random_state=semente)

    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, random_state=semente)

    particao = np.empty(len(tabela_problema), dtype=object)
    particao[indices_treino] = "treino"
    particao[indices_validacao] = "validacao"
    particao[indices_teste] = "teste"
    return particao


# Divide os dados pelo tempo: primeiros períodos no treino, o seguinte na validação e o último no teste
def dividir_temporal(tabela_problema):
    return np.select(
        [tabela_problema["periodo_monitoramento"].isin(periodos_treino),
         tabela_problema["periodo_monitoramento"].isin(periodos_validacao),
         tabela_problema["periodo_monitoramento"].isin(periodos_teste)],
        ["treino", "validacao", "teste"],
        default="fora",
    )


# Mostra quantas linhas ficaram em cada parte, a média do alvo e a porcentagem de zeros
def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo):
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    resumo["média do alvo"] = tabela_problema.groupby(coluna_particao)[coluna_alvo].mean().round(2)
    resumo["% de zeros"] = (tabela_problema.groupby(coluna_particao)[coluna_alvo].apply(lambda valores: (valores == 0).mean()) * 100).round(1)
    print(resumo.to_string())


# Calcula a distância de cada ponto do teste até o ponto de treino mais próximo
def distancias_teste_ao_treino(tabela_problema, coluna_particao, coluna_x, coluna_y):
    pontos_treino = tabela_problema[tabela_problema[coluna_particao] == "treino"][[coluna_x, coluna_y]].to_numpy()
    pontos_teste = tabela_problema[tabela_problema[coluna_particao] == "teste"][[coluna_x, coluna_y]].to_numpy()
    arvore = cKDTree(pontos_treino)
    distancias, _ = arvore.query(pontos_teste)
    return distancias


# Mostra o quanto o teste fica perto do treino em cada divisão (vazamento espacial)
def resumir_vazamento(tabela_problema, colunas_particao, coluna_x, coluna_y):
    linhas_resumo = []
    for coluna_particao in colunas_particao:
        distancias = distancias_teste_ao_treino(tabela_problema, coluna_particao, coluna_x, coluna_y)
        linhas_resumo.append({
            "divisão": coluna_particao.replace("particao_", ""),
            "distância mediana (m)": round(np.median(distancias)),
            "% no mesmo ponto (0 m)": round((distancias == 0).mean() * 100, 1),
            f"% a até {tamanho_segmento} m": round((distancias <= tamanho_segmento).mean() * 100, 1),
        })
    print(pd.DataFrame(linhas_resumo).to_string(index=False))


# Monta o pré-processamento: padroniza as numéricas, aplica one-hot nas categóricas e mantém as binárias
def montar_preprocessador(colunas_numericas, colunas_categoricas, colunas_binarias):
    return ColumnTransformer([
        ("numericas", StandardScaler(), colunas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colunas_categoricas),
        ("binarias", "passthrough", colunas_binarias),
    ])


# Ajusta o pré-processamento só no treino, transforma as três partes e salva o conjunto da divisão
def preparar_e_salvar(tabela_problema, nome_problema, coluna_particao, coluna_alvo, colunas_numericas, colunas_categoricas, colunas_binarias, classes=None):
    colunas_entrada = colunas_numericas + colunas_categoricas + colunas_binarias
    preprocessador = montar_preprocessador(colunas_numericas, colunas_categoricas, colunas_binarias)

    treino = tabela_problema[tabela_problema[coluna_particao] == "treino"]
    validacao = tabela_problema[tabela_problema[coluna_particao] == "validacao"]
    teste = tabela_problema[tabela_problema[coluna_particao] == "teste"]

    conjunto = {
        "X_treino": preprocessador.fit_transform(treino[colunas_entrada]),
        "X_validacao": preprocessador.transform(validacao[colunas_entrada]),
        "X_teste": preprocessador.transform(teste[colunas_entrada]),
        "y_treino": treino[coluna_alvo].to_numpy(),
        "y_validacao": validacao[coluna_alvo].to_numpy(),
        "y_teste": teste[coluna_alvo].to_numpy(),
        "nomes_atributos": list(preprocessador.get_feature_names_out()),
        "preprocessador": preprocessador,
        "classes": classes,
    }

    nome_divisao = coluna_particao.replace("particao_", "")
    nome_arquivo = f"{nome_problema}_{nome_divisao}.joblib"
    joblib.dump(conjunto, os.path.join(pasta_saida, nome_arquivo))
    print(f"  {nome_arquivo}: treino {conjunto['X_treino'].shape} | validação {conjunto['X_validacao'].shape} | teste {conjunto['X_teste'].shape}")
    return conjunto


tabela = pd.read_csv(caminho_registros, parse_dates=["data"], dtype={"segmento": str})
segmentos = pd.read_csv(caminho_segmentos, dtype={"segmento": str})
print("Registros carregados:", len(tabela))
print("Segmentos carregados:", len(segmentos))


print("\nRegistros por trecho e período (todos precisam ter registros nos 5 períodos):")
print(pd.crosstab(tabela["trecho_pela_localizacao"], tabela["periodo_monitoramento"]))

lista_segmentos = segmentos["segmento"]
lista_periodos = sorted(tabela["periodo_monitoramento"].unique())
todas_combinacoes = pd.MultiIndex.from_product([lista_segmentos, lista_periodos], names=["segmento", "periodo_monitoramento"])

contagem = tabela.groupby(["segmento", "periodo_monitoramento"]).size()
segmento_periodo = contagem.reindex(todas_combinacoes, fill_value=0).rename("atropelamentos").reset_index()

segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "centro_x", "centro_y", "trecho"]], on="segmento")

print("\nLinhas da tabela segmento x período:", len(segmento_periodo), "(", len(lista_segmentos), "segmentos x", len(lista_periodos), "períodos )")
print("Total de atropelamentos conferido:", segmento_periodo["atropelamentos"].sum(), "de", len(tabela))

print("\nEstatísticas do alvo da regressão (atropelamentos por segmento e período):")
print(segmento_periodo["atropelamentos"].describe().round(2))

quantidade_zeros = (segmento_periodo["atropelamentos"] == 0).sum()
print("\nLinhas com 0 atropelamentos:", quantidade_zeros, f"({quantidade_zeros / len(segmento_periodo) * 100:.1f}%)")

print("\nMédia de atropelamentos por segmento em cada período:")
print(segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().round(2))

figura, eixos = plt.subplots(1, 2, figsize=(13, 4))
segmento_periodo["atropelamentos"].plot(kind="hist", bins=30, ax=eixos[0], color="steelblue")
eixos[0].set_title("Atropelamentos por segmento e período")
eixos[0].set_xlabel("Atropelamentos no segmento no período")
eixos[0].set_ylabel("Quantidade de linhas")
segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().plot(kind="bar", ax=eixos[1], color="steelblue")
eixos[1].set_title("Média por segmento em cada período")
eixos[1].set_xlabel("Período de monitoramento (abril a março)")
eixos[1].set_ylabel("Atropelamentos por segmento")
eixos[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "regressao_segmento_periodo.png"), dpi=120)
plt.close()


segmento_periodo["particao_aleatoria"] = dividir_aleatorio(segmento_periodo)
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "particao_espacial"]], on="segmento")
segmento_periodo["particao_temporal"] = dividir_temporal(segmento_periodo)

print("\nLinhas, média de atropelamentos e % de zeros em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(segmento_periodo, coluna_particao, "atropelamentos")


print("\nDistância de cada linha de teste até a linha de treino mais próxima (centro dos segmentos):")
resumir_vazamento(segmento_periodo, ["particao_aleatoria", "particao_espacial", "particao_temporal"], "centro_x", "centro_y")

for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    segmentos_treino = set(segmento_periodo.loc[segmento_periodo[coluna_particao] == "treino", "segmento"])
    linhas_teste = segmento_periodo[segmento_periodo[coluna_particao] == "teste"]
    percentual_repetido = linhas_teste["segmento"].isin(segmentos_treino).mean() * 100
    print(f"  {coluna_particao}: {percentual_repetido:.1f}% das linhas de teste são de segmentos que também estão no treino")


print("\nConjuntos salvos em", pasta_saida, "(linhas, colunas):")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    preparar_e_salvar(segmento_periodo, "regressao", coluna_particao, "atropelamentos",
                      colunas_numericas=["centro_x", "centro_y"], colunas_categoricas=["trecho"], colunas_binarias=[])

segmento_periodo.to_csv(os.path.join(pasta_saida, "base_regressao.csv"), index=False, encoding="utf-8-sig")
print("\nBase legível salva: base_regressao.csv")

