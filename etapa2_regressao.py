# 2 etapa IA - Pré-processamento do problema de REGRESSÃO
# cada linha é um segmento de 500 m em um período de monitoramento (abril a março);
# o alvo é quantos atropelamentos o segmento teve naquele período
# roda DEPOIS do etapa2_comum.py, que cria o período, a tabela de segmentos e a divisão espacial:
#   python etapa2_comum.py
#   python etapa2_regressao.py

# ===============================================================
# IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # salva os conjuntos prontos em arquivo, para a etapa 4 carregar
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
# semente e proporções precisam ser as mesmas nos três arquivos da etapa 2
pasta_graficos = "graficosE2"
pasta_saida = "data/etapa2"      # onde ficam os conjuntos prontos para a etapa 4
caminho_registros = os.path.join(pasta_saida, "registros_com_periodo.csv")  # salvo pelo etapa2_comum.py
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")              # salvo pelo etapa2_comum.py
semente = 42
proporcao_validacao = 0.15
proporcao_teste = 0.15
periodos_treino = [1, 2, 3]      # divisão temporal: abr/2010 a mar/2013
periodos_validacao = [4]         # abr/2013 a mar/2014
periodos_teste = [5]             # abr/2014 a mar/2015
tamanho_segmento = 500           # metros, usado na análise de vazamento (passo 4)

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_saida, exist_ok=True)


# ===============================================================
# FUNÇÕES
# ===============================================================
def dividir_aleatorio(tabela_problema):
    # sorteia as linhas em 70/15/15 (sem estratificar: o alvo é um número, não uma classe)
    indices = np.arange(len(tabela_problema))
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, random_state=semente)

    # a validação é 15% do total, mas o sorteio agora é feito sobre os 85% que sobraram: 0.15 / 0.85 = 0.176
    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, random_state=semente)

    particao = np.empty(len(tabela_problema), dtype=object) # cria uma coluna vazia e preenche cada parte
    particao[indices_treino] = "treino"
    particao[indices_validacao] = "validacao"
    particao[indices_teste] = "teste"
    return particao


def dividir_temporal(tabela_problema):
    # a parte depende só do período de monitoramento
    return np.select(
        [tabela_problema["periodo_monitoramento"].isin(periodos_treino),
         tabela_problema["periodo_monitoramento"].isin(periodos_validacao),
         tabela_problema["periodo_monitoramento"].isin(periodos_teste)],
        ["treino", "validacao", "teste"],
        default="fora", # nenhuma linha cai aqui, porque todas estão nos períodos 1 a 5
    ) # np.select escolhe o valor da primeira condição verdadeira de cada linha


def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo):
    # linhas de cada parte, média do alvo e % de zeros: se mudarem muito entre as partes, o teste fica mais difícil
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    resumo["média do alvo"] = tabela_problema.groupby(coluna_particao)[coluna_alvo].mean().round(2)
    resumo["% de zeros"] = (tabela_problema.groupby(coluna_particao)[coluna_alvo].apply(lambda valores: (valores == 0).mean()) * 100).round(1)
    print(resumo.to_string())


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
        "classes": classes, # None na regressão (não há classes)
    }

    nome_divisao = coluna_particao.replace("particao_", "")
    nome_arquivo = f"{nome_problema}_{nome_divisao}.joblib"
    joblib.dump(conjunto, os.path.join(pasta_saida, nome_arquivo)) # salva tudo num arquivo só
    print(f"  {nome_arquivo}: treino {conjunto['X_treino'].shape} | validação {conjunto['X_validacao'].shape} | teste {conjunto['X_teste'].shape}")
    return conjunto


# ===============================================================
# PASSO 1: CARREGAR OS REGISTROS E OS SEGMENTOS
# ===============================================================
# os dois vêm do etapa2_comum.py: os registros já têm o período, e os segmentos já têm o centro, o trecho e a divisão espacial
# dtype={"segmento": str}: o código do segmento ("438_16547") precisa continuar como texto
tabela = pd.read_csv(caminho_registros, parse_dates=["data"], dtype={"segmento": str})
segmentos = pd.read_csv(caminho_segmentos, dtype={"segmento": str})
print("Registros carregados:", len(tabela))
print("Segmentos carregados:", len(segmentos))


# ===============================================================
# PASSO 2: TABELA SEGMENTO x PERÍODO
# ===============================================================
# cada linha passa a ser um segmento em um período de monitoramento (abril a março)
# o alvo da regressão é quantos atropelamentos o segmento teve naquele período
# o período só organiza as linhas e depois divide treino/validação/teste; ele não entra como atributo de entrada

# conferência: os zeros só fazem sentido se todos os trechos foram percorridos em todos os períodos
print("\nRegistros por trecho e período (todos precisam ter registros nos 5 períodos):")
print(pd.crosstab(tabela["trecho_pela_localizacao"], tabela["periodo_monitoramento"]))

# monta todas as combinações segmento x período: 293 segmentos x 5 períodos
lista_segmentos = segmentos["segmento"]
lista_periodos = sorted(tabela["periodo_monitoramento"].unique())
todas_combinacoes = pd.MultiIndex.from_product([lista_segmentos, lista_periodos], names=["segmento", "periodo_monitoramento"]) # from_product cria todos os pares possíveis entre as duas listas

# conta os atropelamentos de cada par que aparece na tabela, e reindex completa os pares que não aparecem com 0
contagem = tabela.groupby(["segmento", "periodo_monitoramento"]).size()
segmento_periodo = contagem.reindex(todas_combinacoes, fill_value=0).rename("atropelamentos").reset_index()

# traz as informações fixas de cada segmento (centro e trecho)
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "centro_x", "centro_y", "trecho"]], on="segmento") # merge junta as duas tabelas pela coluna segmento

print("\nLinhas da tabela segmento x período:", len(segmento_periodo), "(", len(lista_segmentos), "segmentos x", len(lista_periodos), "períodos )")
print("Total de atropelamentos conferido:", segmento_periodo["atropelamentos"].sum(), "de", len(tabela)) # precisa bater com o total de registros

print("\nEstatísticas do alvo da regressão (atropelamentos por segmento e período):")
print(segmento_periodo["atropelamentos"].describe().round(2))

quantidade_zeros = (segmento_periodo["atropelamentos"] == 0).sum()
print("\nLinhas com 0 atropelamentos:", quantidade_zeros, f"({quantidade_zeros / len(segmento_periodo) * 100:.1f}%)")

print("\nMédia de atropelamentos por segmento em cada período:")
print(segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().round(2))

# gráficos: distribuição do alvo e média por período
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


# ===============================================================
# PASSO 3: DIVISÃO EM TREINO, VALIDAÇÃO E TESTE
# ===============================================================
# treino = onde o modelo aprende | validação = onde escolhemos os hiperparâmetros | teste = avaliação final, só usado no fim
# - aleatória: sorteio das linhas segmento x período
#   (o mesmo segmento aparece em períodos diferentes no treino e no teste: é o vazamento que vamos discutir)
# - espacial: cada linha recebe a parte do seu segmento, sorteada no etapa2_comum.py (as mesmas regiões dos outros problemas)
# - temporal: períodos 1 a 3 no treino, 4 na validação e 5 no teste
segmento_periodo["particao_aleatoria"] = dividir_aleatorio(segmento_periodo)
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "particao_espacial"]], on="segmento")
segmento_periodo["particao_temporal"] = dividir_temporal(segmento_periodo)

print("\nLinhas, média de atropelamentos e % de zeros em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(segmento_periodo, coluna_particao, "atropelamentos")


# ===============================================================
# PASSO 4: VAZAMENTO ESPACIAL (O TESTE ESTÁ PERTO DO TREINO?)
# ===============================================================
# se uma linha de teste está a poucos metros de uma linha de treino, o modelo pode acertar só por "lembrar" daquele lugar
# aqui medimos, para cada linha de teste, a distância até a linha de treino mais próxima
print("\nDistância de cada linha de teste até a linha de treino mais próxima (centro dos segmentos):")
resumir_vazamento(segmento_periodo, ["particao_aleatoria", "particao_espacial", "particao_temporal"], "centro_x", "centro_y")

# na regressão, distância 0 significa que o MESMO segmento está no treino em outro período
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    segmentos_treino = set(segmento_periodo.loc[segmento_periodo[coluna_particao] == "treino", "segmento"])
    linhas_teste = segmento_periodo[segmento_periodo[coluna_particao] == "teste"]
    percentual_repetido = linhas_teste["segmento"].isin(segmentos_treino).mean() * 100 # isin marca as linhas cujo segmento também aparece no treino
    print(f"  {coluna_particao}: {percentual_repetido:.1f}% das linhas de teste são de segmentos que também estão no treino")


# ===============================================================
# PASSO 5: CODIFICAÇÃO, PADRONIZAÇÃO E SALVAR OS CONJUNTOS PRONTOS
# ===============================================================
# One-Hot no trecho (nominal: um número 0, 1, 2 criaria uma ordem falsa) e StandardScaler nas coordenadas
# REGRA IMPORTANTE: o preprocessador é ajustado (fit) SÓ com o treino; validação e teste só são transformados (transform)
# atributos: só localização (o período organiza as linhas e a divisão, mas não entra como atributo)
print("\nConjuntos salvos em", pasta_saida, "(linhas, colunas):")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    preparar_e_salvar(segmento_periodo, "regressao", coluna_particao, "atropelamentos",
                      colunas_numericas=["centro_x", "centro_y"], colunas_categoricas=["trecho"], colunas_binarias=[])

# base legível (antes da codificação), com as colunas de partição, para consulta e para o relatório
segmento_periodo.to_csv(os.path.join(pasta_saida, "base_regressao.csv"), index=False, encoding="utf-8-sig")
print("\nBase legível salva: base_regressao.csv")

# como carregar na etapa 4:
#   conjunto = joblib.load("data/etapa2/regressao_aleatoria.joblib")
#   X_treino, y_treino = conjunto["X_treino"], conjunto["y_treino"]
