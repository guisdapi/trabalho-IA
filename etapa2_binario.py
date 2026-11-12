# 2 etapa IA - Pré-processamento do problema BINÁRIO (hotspot)
# cada linha é um segmento de 500 m; o alvo diz se ele é hotspot (1) ou de baixo risco (0)
# roda DEPOIS do etapa2_comum.py, que monta a tabela de segmentos, o alvo e a divisão espacial:
#   python etapa2_comum.py
#   python etapa2_binario.py

# ===============================================================
# IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # salva os conjuntos prontos em arquivo, para a etapa 3 carregar
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
# semente e proporções precisam ser as mesmas nos três arquivos da etapa 2
pasta_graficos = "graficosE2"
pasta_saida = "data/etapa2"      # onde ficam os conjuntos prontos para a etapa 3
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")  # salvo pelo etapa2_comum.py
semente = 42
proporcao_validacao = 0.15
proporcao_teste = 0.15
tamanho_segmento = 500           # metros, usado na análise de vazamento (passo 4)

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_saida, exist_ok=True)


# ===============================================================
# FUNÇÕES
# ===============================================================
def dividir_aleatorio(tabela_problema, coluna_estratificar):
    # sorteia as linhas em 70/15/15; stratify mantém a proporção do alvo igual nas três partes
    indices = np.arange(len(tabela_problema))
    estratos = tabela_problema[coluna_estratificar].to_numpy()
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, stratify=estratos, random_state=semente)

    # a validação é 15% do total, mas o sorteio agora é feito sobre os 85% que sobraram: 0.15 / 0.85 = 0.176
    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, stratify=estratos[indices_resto], random_state=semente)

    particao = np.empty(len(tabela_problema), dtype=object) # cria uma coluna vazia e preenche cada parte
    particao[indices_treino] = "treino"
    particao[indices_validacao] = "validacao"
    particao[indices_teste] = "teste"
    return particao


def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo):
    # linhas de cada parte e percentual de cada classe dentro dela: o ideal é ficar parecido nas três
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    percentual_alvo = (pd.crosstab(tabela_problema[coluna_particao], tabela_problema[coluna_alvo], normalize="index") * 100).round(1)
    print(resumo.join(percentual_alvo).to_string())


def distancias_teste_ao_treino(tabela_problema, coluna_particao, coluna_x, coluna_y):
    pontos_treino = tabela_problema[tabela_problema[coluna_particao] == "treino"][[coluna_x, coluna_y]].to_numpy()
    pontos_teste = tabela_problema[tabela_problema[coluna_particao] == "teste"][[coluna_x, coluna_y]].to_numpy()
    arvore = cKDTree(pontos_treino) # organiza os pontos de treino numa estrutura que acha o mais próximo rapidamente
    distancias, _ = arvore.query(pontos_teste) # para cada ponto de teste, devolve a distância até o ponto de treino mais próximo
    return distancias


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


def montar_preprocessador(colunas_numericas, colunas_categoricas, colunas_binarias):
    return ColumnTransformer([
        ("numericas", StandardScaler(), colunas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colunas_categoricas), # handle_unknown="ignore": se aparecer um trecho que não estava no treino, as colunas ficam 0 em vez de dar erro
        ("binarias", "passthrough", colunas_binarias),
    ])


def preparar_e_salvar(tabela_problema, nome_problema, coluna_particao, coluna_alvo, colunas_numericas, colunas_categoricas, colunas_binarias, classes=None):
    colunas_entrada = colunas_numericas + colunas_categoricas + colunas_binarias
    preprocessador = montar_preprocessador(colunas_numericas, colunas_categoricas, colunas_binarias)

    treino = tabela_problema[tabela_problema[coluna_particao] == "treino"]
    validacao = tabela_problema[tabela_problema[coluna_particao] == "validacao"]
    teste = tabela_problema[tabela_problema[coluna_particao] == "teste"]

    conjunto = {
        "X_treino": preprocessador.fit_transform(treino[colunas_entrada]), # fit_transform: aprende média/desvio/categorias no treino e já transforma
        "X_validacao": preprocessador.transform(validacao[colunas_entrada]), # transform: só aplica o que foi aprendido no treino
        "X_teste": preprocessador.transform(teste[colunas_entrada]),
        "y_treino": treino[coluna_alvo].to_numpy(),
        "y_validacao": validacao[coluna_alvo].to_numpy(),
        "y_teste": teste[coluna_alvo].to_numpy(),
        "nomes_atributos": list(preprocessador.get_feature_names_out()), # nomes das colunas depois da codificação, úteis para interpretar o modelo
        "preprocessador": preprocessador, # guardado para transformar dados novos do mesmo jeito
        "classes": classes, # nomes das classes na ordem dos códigos
    }

    nome_divisao = coluna_particao.replace("particao_", "")
    nome_arquivo = f"{nome_problema}_{nome_divisao}.joblib"
    joblib.dump(conjunto, os.path.join(pasta_saida, nome_arquivo)) # salva tudo num arquivo só
    print(f"  {nome_arquivo}: treino {conjunto['X_treino'].shape} | validação {conjunto['X_validacao'].shape} | teste {conjunto['X_teste'].shape}")
    return conjunto


# ===============================================================
# PASSO 1: CARREGAR A TABELA DE SEGMENTOS E OLHAR O ALVO
# ===============================================================
# a tabela já vem com o alvo (hotspot) e a divisão espacial, calculados no etapa2_comum.py
# dtype={"segmento": str}: o código do segmento ("438_16547") precisa continuar como texto
segmentos = pd.read_csv(caminho_segmentos, dtype={"segmento": str})
print("Segmentos carregados:", len(segmentos))

# o limiar mostrado é a menor contagem que virou hotspot (o P75 pode ser um número quebrado, como 4,5)
limiar_hotspot = segmentos.loc[segmentos["hotspot"] == 1, "atropelamentos"].min()
print("\nHotspot = segmento com", limiar_hotspot, "ou mais atropelamentos (percentil 75, definido no etapa2_comum.py)")

print("\nSegmentos por classe do hotspot:")
print(pd.DataFrame({
    "quantidade": segmentos["hotspot"].value_counts(),
    "percentual": (segmentos["hotspot"].value_counts(normalize=True) * 100).round(1), # normalize=True devolve a proporção em vez da contagem
}))

print("\nHotspots por trecho:")
print(pd.crosstab(segmentos["trecho"], segmentos["hotspot"], margins=True)) # margins=True adiciona a linha e a coluna de total

# mapa dos segmentos: cinza = baixo risco, vermelho = hotspot
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


# ===============================================================
# PASSO 2: DIVISÃO EM TREINO, VALIDAÇÃO E TESTE
# ===============================================================
# treino = onde o modelo aprende | validação = onde escolhemos os hiperparâmetros | teste = avaliação final, só usado no fim
# - aleatória: sorteio dos segmentos, mantendo a proporção de hotspots nas três partes
# - espacial: já veio pronta do etapa2_comum.py (blocos de 3 km inteiros em cada parte)
# - o hotspot NÃO tem divisão temporal: cada segmento soma os 5 períodos, então não existe um "período" para separar
segmentos["particao_aleatoria"] = dividir_aleatorio(segmentos, "hotspot")

print("\nLinhas e % de cada classe (0 = baixo risco, 1 = hotspot) em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial"]:
    resumir_divisao(segmentos, coluna_particao, "hotspot")


# ===============================================================
# PASSO 3: VAZAMENTO ESPACIAL (O TESTE ESTÁ PERTO DO TREINO?)
# ===============================================================
# se um segmento de teste está colado a um segmento de treino, o modelo pode acertar só por "lembrar" daquele lugar
# isso deixa a avaliação otimista; aqui medimos, para cada segmento de teste, a distância até o segmento de treino mais próximo
print("\nDistância de cada segmento de teste até o segmento de treino mais próximo (centro dos segmentos):")
resumir_vazamento(segmentos, ["particao_aleatoria", "particao_espacial"], "centro_x", "centro_y")


# ===============================================================
# PASSO 4: CODIFICAÇÃO, PADRONIZAÇÃO E SALVAR OS CONJUNTOS PRONTOS
# ===============================================================
# One-Hot no trecho (nominal: um número 0, 1, 2 criaria uma ordem falsa) e StandardScaler nas coordenadas
# REGRA IMPORTANTE: o preprocessador é ajustado (fit) SÓ com o treino; validação e teste só são transformados (transform)
# atributos: só localização; o total de atropelamentos NÃO entra, porque é ele que define o alvo
print("\nConjuntos salvos em", pasta_saida, "(linhas, colunas):")
for coluna_particao in ["particao_aleatoria", "particao_espacial"]:
    preparar_e_salvar(segmentos, "hotspot", coluna_particao, "hotspot",
                      colunas_numericas=["centro_x", "centro_y"], colunas_categoricas=["trecho"], colunas_binarias=[],
                      classes=["baixo_risco", "hotspot"])

# base legível (antes da codificação), com as colunas de partição, para consulta e para o relatório
segmentos.to_csv(os.path.join(pasta_saida, "base_hotspot.csv"), index=False, encoding="utf-8-sig")
print("\nBase legível salva: base_hotspot.csv")
