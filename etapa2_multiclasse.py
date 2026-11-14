import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
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





# Divide os dados aleatoriamente em treino, validação e teste, mantendo a proporção de cada classe
def dividir_aleatorio(tabela_problema, coluna_estratificar):
    indices = np.arange(len(tabela_problema))
    estratos = tabela_problema[coluna_estratificar].to_numpy()
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, stratify=estratos, random_state=semente)

    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, stratify=estratos[indices_resto], random_state=semente)

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


# Mostra quantas linhas ficaram em cada parte e a porcentagem de cada classe
def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo):
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    percentual_alvo = (pd.crosstab(tabela_problema[coluna_particao], tabela_problema[coluna_alvo], normalize="index") * 100).round(1)
    print(resumo.join(percentual_alvo).to_string())


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
print("Registros carregados:", len(tabela))

base_multiclasse = tabela[tabela["classe"] != "nao_identificado"].copy().reset_index(drop=True)
print("\nRegistros removidos por classe nao_identificado:", len(tabela) - len(base_multiclasse))
print("Registros na base do multiclasse:", len(base_multiclasse))


base_multiclasse["mes_numero"] = base_multiclasse["data"].dt.month
base_multiclasse["mes_sin"] = np.sin(2 * np.pi * base_multiclasse["mes_numero"] / 12)
base_multiclasse["mes_cos"] = np.cos(2 * np.pi * base_multiclasse["mes_numero"] / 12)

base_multiclasse["estacao_chuvosa"] = base_multiclasse["mes_numero"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

print("\nRegistros por dia da semana do percurso (0 = segunda, 6 = domingo):")
print(base_multiclasse["data"].dt.dayofweek.value_counts().sort_index())

codificador_classe = LabelEncoder()
base_multiclasse["classe_codigo"] = codificador_classe.fit_transform(base_multiclasse["classe"])
print("\nCódigo de cada classe:", dict(zip(codificador_classe.classes_, range(len(codificador_classe.classes_)))))

print("\nDistribuição do alvo do multiclasse:")
print(pd.DataFrame({
    "quantidade": base_multiclasse["classe"].value_counts(),
    "percentual": (base_multiclasse["classe"].value_counts(normalize=True) * 100).round(1),
}))

atributos_multiclasse = ["x", "y", "trecho_pela_localizacao", "mes_sin", "mes_cos", "estacao_chuvosa"]
print("\nAtributos de entrada do multiclasse:", atributos_multiclasse)

percentual_classe_mes = pd.crosstab(base_multiclasse["mes_numero"], base_multiclasse["classe"], normalize="index") * 100
percentual_classe_mes.plot(kind="bar", stacked=True, figsize=(10, 5), width=0.85)
plt.title("Composição das classes em cada mês")
plt.xlabel("Mês")
plt.ylabel("Percentual dos atropelamentos do mês")
plt.xticks(rotation=0)
plt.legend(title="classe", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "classes_por_mes.png"), dpi=120)
plt.close()


base_multiclasse["particao_aleatoria"] = dividir_aleatorio(base_multiclasse, "classe")

segmentos = pd.read_csv(caminho_segmentos, dtype={"segmento": str}, usecols=["segmento", "particao_espacial"])
base_multiclasse = base_multiclasse.merge(segmentos, on="segmento")

base_multiclasse["particao_temporal"] = dividir_temporal(base_multiclasse)

print("\nLinhas e % de cada classe em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(base_multiclasse, coluna_particao, "classe")


print("\nDistância de cada registro de teste até o registro de treino mais próximo (local de cada atropelamento):")
resumir_vazamento(base_multiclasse, ["particao_aleatoria", "particao_espacial", "particao_temporal"], "x", "y")

figura, eixos = plt.subplots(1, 3, figsize=(18, 6))
cores = {"treino": "steelblue", "validacao": "orange", "teste": "crimson"}
for eixo, coluna_particao in zip(eixos, ["particao_aleatoria", "particao_espacial", "particao_temporal"]):
    for parte, cor in cores.items():
        pontos = base_multiclasse[base_multiclasse[coluna_particao] == parte]
        eixo.scatter(pontos["x"], pontos["y"], s=3, color=cor, label=parte)
    eixo.set_title("Divisão " + coluna_particao.replace("particao_", ""))
    eixo.set_xlabel("x (UTM, metros)")
    eixo.set_ylabel("y (UTM, metros)")
    eixo.set_aspect("equal")
    eixo.legend(markerscale=4)
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "divisoes_mapa.png"), dpi=120)
plt.close()


print("\nConjuntos salvos em", pasta_saida, "(linhas, colunas):")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    conjunto_exemplo = preparar_e_salvar(base_multiclasse, "multiclasse", coluna_particao, "classe_codigo",
                      colunas_numericas=["x", "y", "mes_sin", "mes_cos"], colunas_categoricas=["trecho_pela_localizacao"], colunas_binarias=["estacao_chuvosa"],
                      classes=list(codificador_classe.classes_))

print("\nConferência da padronização (multiclasse temporal):")
nomes = conjunto_exemplo["nomes_atributos"]
for parte in ["treino", "validacao", "teste"]:
    matriz = pd.DataFrame(conjunto_exemplo["X_" + parte], columns=nomes)
    print(f"\n  {parte} - média e desvio de cada coluna:")
    print(pd.DataFrame({"média": matriz.mean().round(2), "desvio": matriz.std().round(2)}).T.to_string())

base_multiclasse.to_csv(os.path.join(pasta_saida, "base_multiclasse.csv"), index=False, encoding="utf-8-sig")
print("\nBase legível salva: base_multiclasse.csv")

