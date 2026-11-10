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