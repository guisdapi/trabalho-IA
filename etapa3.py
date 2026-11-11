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
def avaliar_multiclasse(y_real, y_previsto):
    # calcula as métricas do multiclasse pedidas no enunciado; a mesma função vai avaliar a linha de base, a MLP e o Random Forest
    # average="macro": calcula a métrica de cada classe e tira a média simples, então anfíbios (classe rara) pesam igual a aves
    # zero_division=0: se o modelo nunca prevê uma classe, a precision dela fica 0 em vez de dar aviso (acontece com a linha de base)
    return {
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision_macro": precision_score(y_real, y_previsto, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_previsto, average="macro", zero_division=0),
        "f1_macro": f1_score(y_real, y_previsto, average="macro", zero_division=0),
    }


# ===============================================================
# PASSO 1: CARREGAR O CONJUNTO DA ETAPA 2
# ===============================================================
# começamos pelo multiclasse com divisão aleatória (mais dados e divisão mais simples)
# o .joblib é um dicionário com X_treino, y_treino, ..., nomes_atributos, classes e preprocessador, salvo no passo 7 da etapa 2
caminho_conjunto = os.path.join(pasta_conjuntos, "multiclasse_aleatoria.joblib")
conjunto = joblib.load(caminho_conjunto)

# separa as peças em variáveis com nomes curtos, para usar nos próximos passos
X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]

# conferência rápida: os tamanhos precisam bater com o final da etapa 2, e a rede vai receber X_treino.shape[1] entradas
print("Conjunto carregado:", caminho_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")
print("Código de cada classe:", dict(enumerate(nomes_classes))) # enumerate numera a lista: 0 = primeira classe, 1 = segunda, ...


# ===============================================================
# PASSO 2: LINHA DE BASE (CHUTAR SEMPRE A CLASSE MAIS COMUM)
# ===============================================================
# a linha de base é a nota mínima: qualquer modelo que aprende alguma coisa precisa ficar acima dela
# o DummyClassifier com strategy="most_frequent" olha só o y_treino, descobre a classe mais comum e passa a prever ela para TODA linha
# ele recebe o X só por obrigação do scikit-learn, mas ignora completamente os atributos
linha_de_base = DummyClassifier(strategy="most_frequent") # linha de base = sempre chutar a classe mais comum do treino, que seria a classe 0 (aves) nesse conjunto
linha_de_base.fit(X_treino, y_treino) # .fit() significa "treinar", mas a linha de base não aprende nada, só descobre a classe mais comum do treino

classe_chutada = nomes_classes[linha_de_base.predict(X_treino[:1])[0]] # prevê uma linha qualquer só para descobrir qual classe ele sempre chuta
print("\nA linha de base prevê sempre:", classe_chutada)

# avalia na validação (referência para ajustar a MLP) e no teste (referência para a tabela final de comparação)
# não há vazamento em avaliar no teste aqui, porque a linha de base não tem nada para ajustar
resultados_linha_de_base = pd.DataFrame({
    "validacao": avaliar_multiclasse(y_validacao, linha_de_base.predict(X_validacao)),
    "teste": avaliar_multiclasse(y_teste, linha_de_base.predict(X_teste)),
}).round(3)

print("\nMétricas da linha de base:")
print(resultados_linha_de_base.to_string())
# leitura esperada: a accuracy fica igual ao % da classe mais comum, mas o recall macro fica em 0,25 (acerta 100% de uma classe e 0% das outras 3)
# e o F1 macro fica bem baixo; isso mostra por que a accuracy sozinha engana quando as classes são desbalanceadas