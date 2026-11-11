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

semente = 42  # a mesma da etapa 2: fixa os sorteios (pesos iniciais da rede, ordem dos batches), para todo o grupo obter o mesmo resultado

# hiperparâmetros da MLP (ponto de partida; a otimização com Optuna fica para a etapa 5)
neuronios_camadas_ocultas = [32, 16]  # 2 camadas ocultas, com 32 e 16 neurônios; cada número da lista é uma camada
taxa_aprendizado = 0.001              # tamanho do passo de cada ajuste dos pesos (valor padrão do Adam)
tamanho_batch = 32                    # quantas linhas a rede vê antes de cada ajuste dos pesos (usado no treino, passo 5)
epocas_maximas = 300                  # limite de épocas; na prática o treino para antes, pela parada antecipada (passo 5)
paciencia = 20                        # quantas épocas seguidas sem melhora na validação a parada antecipada tolera antes de parar

pasta_graficos = "graficosE3"

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script, igual às etapas 1 e 2
os.makedirs(pasta_graficos, exist_ok=True) # cria a pasta dos gráficos da etapa 3 se ela não existir


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


def preparar_alvo_para_rede(y, numero_classes):
    # a rede compara a sua saída com o alvo, então o alvo precisa ter o mesmo formato da saída
    if numero_classes == 2:
        # binária: a saída é 1 número (probabilidade de hotspot), então o alvo vira uma coluna de 0.0/1.0
        return y.reshape(-1, 1).astype("float32") # reshape(-1, 1) transforma a lista em uma coluna
    # multiclasse: a saída são 4 probabilidades, então o alvo vira 4 colunas com um único 1 (One-Hot)
    # exemplo: código 2 (mammals) -> [0, 0, 1, 0]
    return keras.utils.to_categorical(y, numero_classes)


def montar_mlp(numero_entradas, numero_classes):
    # monta e configura a rede; serve para o multiclasse e para o hotspot (binária), muda só a camada de saída e a loss
    modelo = keras.Sequential(name="mlp")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada")) # camada de entrada: 1 neurônio por atributo

    # camadas ocultas: Dense = cada neurônio se liga a todos os neurônios da camada anterior
    for numero_camada, neuronios in enumerate(neuronios_camadas_ocultas, start=1):
        modelo.add(keras.layers.Dense(neuronios, activation="relu", name=f"oculta_{numero_camada}"))

    if numero_classes == 2:
        # binária: 1 neurônio com sigmoide, que devolve um número entre 0 e 1 (probabilidade de ser hotspot)
        modelo.add(keras.layers.Dense(1, activation="sigmoid", name="saida"))
        funcao_perda = "binary_crossentropy"
        f1 = keras.metrics.F1Score(average="micro", threshold=0.5, name="f1") # F1 da classe hotspot, com o corte de 0,5
    else:
        # multiclasse: 1 neurônio por classe com softmax, que devolve uma probabilidade para cada classe (somando 1)
        modelo.add(keras.layers.Dense(numero_classes, activation="softmax", name="saida"))
        funcao_perda = "categorical_crossentropy"
        f1 = keras.metrics.F1Score(average="macro", name="f1_macro") # F1 macro, a métrica principal do multiclasse

    # compile junta as peças do treino: quem ajusta os pesos (otimizador), o que ele tenta diminuir (loss)
    # e o que é acompanhado a cada época (métricas, que viram as curvas do relatório)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss=funcao_perda,
        metrics=["accuracy", f1],
    )
    return modelo


# ===============================================================
# PASSO 1: CARREGAR O CONJUNTO DA ETAPA 2
# ===============================================================
# começamos pelo multiclasse com divisão aleatória (mais dados e divisão mais simples)
# o .joblib é um dicionário com X_treino, y_treino, ..., nomes_atributos, classes e preprocessador, salvo no passo 7 da etapa 2
nome_conjunto = "multiclasse_aleatoria" # também usado no nome dos gráficos, para não misturar quando rodarmos as outras divisões
caminho_conjunto = os.path.join(pasta_conjuntos, nome_conjunto + ".joblib")
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
linha_de_base = DummyClassifier(strategy="most_frequent")
linha_de_base.fit(X_treino, y_treino)

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


# ===============================================================
# PASSO 3: PESOS DAS CLASSES (TRATAR O DESBALANCEAMENTO)
# ===============================================================
# sem pesos, errar um anfíbio custa o mesmo que errar uma ave; como há poucos anfíbios, a rede aprende que ignorá-los quase não custa nada
# com pesos, o erro em uma classe rara "dói" mais na loss, e a rede passa a prestar atenção nela
# class_weight="balanced" calcula: peso = total de linhas / (número de classes x linhas daquela classe)
#   classe com exatamente 1/4 do treino -> peso 1 | classe mais rara -> peso maior que 1 | classe mais comum -> peso menor que 1
# calculado SÓ com o y_treino: a validação e o teste não podem influenciar nenhuma decisão do treino
classes_do_treino = np.unique(y_treino)
pesos = compute_class_weight(class_weight="balanced", classes=classes_do_treino, y=y_treino)
pesos_classes = dict(zip(classes_do_treino.tolist(), pesos.tolist())) # o Keras recebe os pesos como dicionário {código da classe: peso}

print("\nPeso de cada classe no treino:")
print(pd.DataFrame({
    "classe": [nomes_classes[codigo] for codigo in classes_do_treino],
    "linhas no treino": pd.Series(y_treino).value_counts().sort_index().to_numpy(),
    "peso": np.round(pesos, 2),
}).to_string(index=False))


# ===============================================================
# PASSO 4: MONTAR E CONFIGURAR A MLP
# ===============================================================
# fixa a semente do Python, do NumPy e do TensorFlow de uma vez; precisa vir ANTES de montar a rede,
# porque os pesos iniciais dos neurônios são sorteados no momento em que as camadas são criadas
keras.utils.set_random_seed(semente)

numero_entradas = X_treino.shape[1]  # 8 atributos -> 8 entradas
numero_classes = len(nomes_classes)  # 4 classes -> 4 saídas

# o alvo de treino e de validação no formato da saída da rede (One-Hot no multiclasse)
# o y_teste continua em código (0 a 3), porque as métricas do scikit-learn comparam códigos
y_treino_rede = preparar_alvo_para_rede(y_treino, numero_classes)
y_validacao_rede = preparar_alvo_para_rede(y_validacao, numero_classes)

mlp = montar_mlp(numero_entradas, numero_classes)
print()
mlp.summary() # mostra as camadas, o formato da saída de cada uma e quantos pesos (parâmetros) a rede tem para ajustar

# parada antecipada: a cada época olha a loss da validação; se ela passar "paciencia" épocas sem melhorar, o treino para
# restore_best_weights=True: no fim, a rede volta aos pesos da época com a menor loss de validação (e não aos da última época)
# monitoramos a loss (e não o F1) porque ela varia de forma mais suave e mede também a confiança das respostas
parada_antecipada = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=paciencia,
    restore_best_weights=True,
    verbose=1, # avisa no terminal em qual época parou
)


# ===============================================================
# PASSO 5: TREINAR A MLP E GERAR AS CURVAS DE TREINAMENTO
# ===============================================================
# o class_weight do Keras só é aplicado na loss do TREINO; a loss da validação seria calculada sem pesos
# aí as duas curvas ficariam em escalas diferentes e não daria para compará-las no gráfico
# por isso a validação recebe os mesmos pesos, linha a linha (sample_weight): cada linha ganha o peso da sua classe
# assim as duas losses medem a mesma coisa, e a parada antecipada escolhe a época pensando em todas as classes, inclusive as raras
# (as métricas accuracy e F1 não usam esses pesos: elas continuam sendo a accuracy e o F1 de verdade)
pesos_linhas_validacao = np.array([pesos_classes[codigo] for codigo in y_validacao])

inicio = time.perf_counter() # marca o horário de início, para medir o custo computacional
historico = mlp.fit(
    X_treino, y_treino_rede,
    validation_data=(X_validacao, y_validacao_rede, pesos_linhas_validacao), # avaliada no fim de cada época; NÃO é usada para ajustar os pesos
    epochs=epocas_maximas,
    batch_size=tamanho_batch,
    class_weight=pesos_classes,
    callbacks=[parada_antecipada],
    shuffle=True, # embaralha a ordem das linhas a cada época, para os batches não serem sempre os mesmos
    verbose=0,    # 0 = não imprime nada durante o treino; troque para 2 se quiser ver uma linha por época
)
tempo_treino_mlp = time.perf_counter() - inicio

# historico.history é um dicionário com uma lista por medida, um valor por época:
# loss, accuracy, f1_macro (treino) e val_loss, val_accuracy, val_f1_macro (validação)
curvas = historico.history
epocas_treinadas = len(curvas["loss"])
melhor_epoca = int(np.argmin(curvas["val_loss"])) + 1 # argmin devolve a posição da menor loss de validação; +1 porque as épocas começam em 1

print(f"\nTreino da MLP: {epocas_treinadas} épocas em {tempo_treino_mlp:.1f} s")
print("Melhor época (menor loss de validação):", melhor_epoca, "-> a rede ficou com os pesos dessa época")
print("\nMedidas na melhor época:")
print(pd.DataFrame({
    "treino": [curvas["loss"][melhor_epoca - 1], curvas["accuracy"][melhor_epoca - 1], curvas["f1_macro"][melhor_epoca - 1]],
    "validacao": [curvas["val_loss"][melhor_epoca - 1], curvas["val_accuracy"][melhor_epoca - 1], curvas["val_f1_macro"][melhor_epoca - 1]],
}, index=["loss", "accuracy", "f1_macro"]).round(3).to_string())

# gráfico: à esquerda a loss, à direita o F1 macro (a métrica principal); treino x validação em cada um
# a linha tracejada marca a melhor época; depois dela, o treino continuou só para a parada antecipada confirmar que não havia melhora
epocas = range(1, epocas_treinadas + 1)
figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
for eixo, medida, titulo in [(eixos[0], "loss", "Loss (entropia cruzada com pesos)"), (eixos[1], "f1_macro", "F1 macro")]:
    eixo.plot(epocas, curvas[medida], color="steelblue", label="treino")
    eixo.plot(epocas, curvas["val_" + medida], color="orange", label="validação")
    eixo.axvline(melhor_epoca, color="gray", linestyle="--", label=f"melhor época ({melhor_epoca})") # axvline desenha uma linha vertical
    eixo.set_title(titulo)
    eixo.set_xlabel("Época")
    eixo.legend()
figura.suptitle(f"Curvas de treinamento da MLP - {nome_conjunto}") # título geral acima dos dois gráficos
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, f"curvas_mlp_{nome_conjunto}.png"), dpi=120)
plt.close()
print("\nGráfico salvo:", os.path.join(pasta_graficos, f"curvas_mlp_{nome_conjunto}.png"))