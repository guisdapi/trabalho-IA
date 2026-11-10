# 3 etapa IA - Classificação com MLP

# ===============================================================
# IMPORTS
# ===============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # esconde as mensagens internas do TensorFlow (avisos de CPU/GPU) que poluem o terminal; precisa vir ANTES do import do tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # desliga a otimização oneDNN: os resultados ficam iguais em toda execução (com a mesma semente), o que é importante para o grupo comparar resultados; com a base pequena, a perda de velocidade é insignificante

import time  # cronometra o treino de cada modelo, para discutir o custo computacional
import numpy as np  # transforma as probabilidades da rede na classe prevista (a maior probabilidade, ou o corte de 0,5 no hotspot)
import pandas as pd  # monta as tabelas de resultado (MLP x Random Forest em cada divisão)
import matplotlib.pyplot as plt  # gráficos: curvas de loss/accuracy por época, matrizes de confusão e curva ROC
import joblib  # abre os conjuntos prontos salvos na etapa 2 (data/etapa2/*.joblib)

# rede neural (MLP)
import tensorflow as tf  # o "motor" que faz as contas do treino; também usado para fixar a semente
from tensorflow import keras  # a parte do TensorFlow usada para montar, treinar e parar a rede (camadas, otimizador, parada antecipada)

# modelo tradicional, linha de base e pesos das classes
from sklearn.dummy import DummyClassifier  # linha de base: "modelo" que sempre chuta a classe mais comum
from sklearn.ensemble import RandomForestClassifier  # modelo tradicional usado na comparação com a MLP
from sklearn.utils.class_weight import compute_class_weight  # calcula o peso de cada classe a partir do treino (classes raras pesam mais)

# métricas (as mesmas funções avaliam a MLP e o Random Forest, para a comparação ser justa)
from sklearn.metrics import (
    accuracy_score,          # % de acertos
    precision_score,         # dos que o modelo disse que são da classe, quantos realmente são
    recall_score,            # dos que realmente são da classe, quantos o modelo encontrou
    f1_score,                # equilíbrio entre precision e recall
    roc_auc_score,           # ROC-AUC do hotspot (avalia todos os cortes de probabilidade)
    confusion_matrix,        # tabela de quais classes se confundem com quais
    ConfusionMatrixDisplay,  # desenha a matriz de confusão
    RocCurveDisplay,         # desenha a curva ROC do hotspot
    classification_report,   # precision, recall e F1 de cada classe de uma vez
)


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_conjuntos = "data/etapa2"  # onde a etapa 2 salvou os conjuntos prontos (.joblib)
partes = ["treino", "validacao", "teste"]  # as três partes de cada conjunto, na ordem em que serão mostradas

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script, igual às etapas 1 e 2


# ===============================================================
# FUNÇÕES
# ===============================================================
def carregar_conjunto(nome_arquivo):
    # abre um dos arquivos .joblib da etapa 2; ele é um dicionário com X_treino, y_treino, ..., nomes_atributos, classes e preprocessador
    caminho = os.path.join(pasta_conjuntos, nome_arquivo)
    conjunto = joblib.load(caminho)
    print("\nConjunto carregado:", caminho)
    return conjunto


def conferir_conjunto(conjunto):
    # serve para qualquer um dos 5 conjuntos (hotspot ou multiclasse), porque todos têm as mesmas chaves

    # --- conferência 1: tamanhos ---
    # X é a tabela de entrada (linhas x colunas) e y é o alvo (uma resposta por linha); os dois precisam ter o mesmo número de linhas
    total_linhas = sum(len(conjunto["y_" + parte]) for parte in partes)
    print("\nTamanho de cada parte:")
    for parte in partes:
        linhas_X, colunas_X = conjunto["X_" + parte].shape # .shape devolve (linhas, colunas)
        linhas_y = len(conjunto["y_" + parte])
        situacao = "ok" if linhas_X == linhas_y else "ERRO: X e y com tamanhos diferentes"
        print(f"  {parte}: {linhas_X} linhas ({linhas_X / total_linhas * 100:.1f}%), {colunas_X} colunas | y: {linhas_y} respostas -> {situacao}")

    # --- conferência 2: atributos de entrada ---
    # os nomes vêm do ColumnTransformer da etapa 2: "numericas__" = padronizada, "categoricas__" = coluna do One-Hot, "binarias__" = passou sem alteração
    print("\nAtributos de entrada:", len(conjunto["nomes_atributos"]))
    for nome in conjunto["nomes_atributos"]:
        print("  ", nome)

    # --- conferência 3: valores faltando ---
    # a rede não aceita células vazias (NaN); se aparecer algum, o treino dá erro ou a loss vira "nan"
    print("\nCélulas vazias (NaN) em cada parte:")
    for parte in partes:
        print(f"  {parte}:", int(np.isnan(conjunto["X_" + parte]).sum())) # isnan marca as células vazias e sum conta quantas são

    # --- conferência 4: códigos das classes ---
    # o alvo está em números; a lista "classes" diz qual nome corresponde a cada número (posição 0 = código 0, ...)
    print("\nCódigo de cada classe:", {codigo: nome for codigo, nome in enumerate(conjunto["classes"])})
    print("Códigos que aparecem no alvo de cada parte:")
    for parte in partes:
        print(f"  {parte}:", sorted(np.unique(conjunto["y_" + parte]).tolist())) # np.unique devolve os valores distintos

    # --- conferência 5: distribuição das classes em cada parte ---
    # o ideal é que a proporção de cada classe fique parecida no treino, na validação e no teste
    quantidade = pd.DataFrame({parte: pd.Series(conjunto["y_" + parte]).value_counts() for parte in partes}).fillna(0).astype(int) # fillna(0): classe que não aparece numa parte fica com 0
    quantidade = quantidade.sort_index() # ordena pelo código da classe
    quantidade.index = [conjunto["classes"][codigo] for codigo in quantidade.index] # troca o código pelo nome da classe
    percentual = (quantidade / quantidade.sum() * 100).round(1) # divide cada coluna pelo total da própria parte

    print("\nQuantidade de cada classe em cada parte:")
    print(quantidade.to_string())
    print("\nPercentual de cada classe em cada parte (%):")
    print(percentual.to_string())


# ===============================================================
# PASSO 1: CARREGAR O CONJUNTO DA ETAPA 2 E CONFERIR
# ===============================================================
# começamos pelo multiclasse com divisão aleatória (mais dados e divisão mais simples)
# os outros conjuntos (multiclasse espacial/temporal e hotspot aleatória/espacial) usam as mesmas funções depois
conjunto = carregar_conjunto("multiclasse_aleatoria.joblib")
conferir_conjunto(conjunto)

# separa as peças em variáveis com nomes curtos, para usar nos próximos passos
X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]