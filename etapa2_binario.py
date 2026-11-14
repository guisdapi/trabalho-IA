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
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")
semente = 42
proporcao_validacao = 0.15
proporcao_teste = 0.15
tamanho_segmento = 500

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_saida, exist_ok=True)





# Divide os dados aleatoriamente em treino, validação e teste, mantendo a proporção de hotspots
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


segmentos = pd.read_csv(caminho_segmentos, dtype={"segmento": str})
print("Segmentos carregados:", len(segmentos))

limiar_hotspot = segmentos.loc[segmentos["hotspot"] == 1, "atropelamentos"].min()
print("\nHotspot = segmento com", limiar_hotspot, "ou mais atropelamentos (percentil 75, definido no etapa2_comum.py)")

print("\nSegmentos por classe do hotspot:")
print(pd.DataFrame({
    "quantidade": segmentos["hotspot"].value_counts(),
    "percentual": (segmentos["hotspot"].value_counts(normalize=True) * 100).round(1),
}))

print("\nHotspots por trecho:")
print(pd.crosstab(segmentos["trecho"], segmentos["hotspot"], margins=True))

plt.figure(figsize=(8, 7))
baixo_risco = segmentos[segmentos["hotspot"] == 0]
hotspots = segmentos[segmentos["hotspot"] == 1]
plt.scatter(baixo_risco["centro_x"], baixo_risco["centro_y"], s=10, color="lightgray", label="baixo risco (0)")
plt.scatter(hotspots["centro_x"], hotspots["centro_y"], s=10, color="crimson", label=f"hotspot (1): >= {limiar_hotspot:.0f} atropelamentos")
plt.title("Segmentos de 500 m: hotspots de atropelamento (P75)")
plt.xlabel("x (UTM, metros)")
plt.ylabel("y (UTM, metros)")
plt.legend(markerscale=2)
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "hotspots_segmentos.png"), dpi=120)
plt.close()


segmentos["particao_aleatoria"] = dividir_aleatorio(segmentos, "hotspot")

print("\nLinhas e % de cada classe (0 = baixo risco, 1 = hotspot) em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial"]:
    resumir_divisao(segmentos, coluna_particao, "hotspot")


print("\nDistância de cada segmento de teste até o segmento de treino mais próximo (centro dos segmentos):")
resumir_vazamento(segmentos, ["particao_aleatoria", "particao_espacial"], "centro_x", "centro_y")


print("\nConjuntos salvos em", pasta_saida, "(linhas, colunas):")
for coluna_particao in ["particao_aleatoria", "particao_espacial"]:
    preparar_e_salvar(segmentos, "hotspot", coluna_particao, "hotspot",
                      colunas_numericas=["centro_x", "centro_y"], colunas_categoricas=["trecho"], colunas_binarias=[],
                      classes=["baixo_risco", "hotspot"])

segmentos.to_csv(os.path.join(pasta_saida, "base_hotspot.csv"), index=False, encoding="utf-8-sig")
print("\nBase legível salva: base_hotspot.csv")
