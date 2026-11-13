# 5 etapa IA - Otimização de hiperparâmetros da MLP com Optuna (problema MULTICLASSE: classe taxonômica)
#
# por que o multiclasse: é o problema com mais dados (~3 mil linhas no treino e ~660 na validação),
# então o F1 macro da validação é estável o bastante para comparar dezenas de configurações
# (no hotspot, com 44 segmentos na validação, o Optuna acabaria "decorando" a validação)
#
# a divisão vem do terminal (padrão: aleatória); o 2º número, opcional, é a quantidade de trials:
#   python etapa5_optuna.py aleatoria
#   python etapa5_optuna.py aleatoria 5     (teste rápido com 5 trials, para ver se tudo roda antes da busca completa)
#   python etapa5_optuna.py temporal
# roda DEPOIS do etapa3_multiclasse.py da mesma divisão, porque compara com o modelo original salvo em resultadosE3

# ===============================================================
# IMPORTS
# ===============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # esconde as mensagens internas do TensorFlow; precisa vir ANTES do import do tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # resultados iguais em toda execução (com a mesma semente), igual às etapas 3 e 4

import sys   # lê a divisão e a quantidade de trials digitadas no terminal
import time  # cronometra a busca e o treino, para discutir o custo da otimização
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # abre os conjuntos prontos salvos na etapa 2 (data/etapa2/*.joblib)
import optuna  # biblioteca de otimização de hiperparâmetros

from tensorflow import keras  # a mesma biblioteca das etapas 3 e 4
from sklearn.utils.class_weight import compute_class_weight  # pesos das classes, igual à etapa 3
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report,
)


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_conjuntos = "data/etapa2"      # conjuntos prontos da etapa 2
pasta_resultados_e3 = "resultadosE3" # onde está o modelo original (etapa 3), usado na comparação
pasta_graficos = "graficosE5"
pasta_resultados = "resultadosE5"
divisoes_possiveis = ["aleatoria", "espacial", "temporal"]

semente = 42            # a mesma das outras etapas
numero_trials = 50      # quantas configurações o Optuna testa (cada trial = treinar uma rede e medir na validação)
epocas_maximas = 300    # limite de épocas por trial; na etapa 3 o multiclasse nunca passou de ~130, então 300 dá folga
paciencia = 20          # parada antecipada igual à etapa 3

# --- espaço de busca (os limites e a justificativa vão para o relatório; ver passo 3) ---
camadas_min, camadas_max = 1, 3            # na etapa 3 só testamos 2 camadas; aqui vemos se 1 basta ou se 3 ajuda
neuronios_min, neuronios_max = 8, 128      # neurônios da 1ª camada; cobre abaixo e acima do que já testamos (16 a 64)
ativacoes = ["relu", "tanh"]               # as duas ativações mais usadas em camadas ocultas de MLP
otimizadores = ["adam", "rmsprop", "sgd"]  # os três citados no enunciado
momentum_sgd = 0.9                         # momentum fixo do SGD (valor mais usado); fixo para todo trial ter os mesmos hiperparâmetros
taxa_min, taxa_max = 1e-4, 1e-2            # em volta do 0,001 usado na etapa 3 (10x menor até 10x maior)
batches = [16, 32, 64, 128]                # em volta do 32 usado na etapa 3
dropout_max = 0.5                          # de 0 (sem dropout) a 0,5; acima disso a rede costuma "esquecer" demais
l2_min, l2_max = 1e-6, 1e-2                # do quase nada até uma penalização forte dos pesos

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING) # esconde o log padrão do Optuna; o progresso é mostrado pela função mostrar_progresso


# ===============================================================
# FUNÇÕES
# ===============================================================
def avaliar(y_real, y_previsto):
    # as mesmas métricas da etapa 3 (macro: cada classe pesa igual, inclusive a rara)
    return {
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision_macro": precision_score(y_real, y_previsto, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_previsto, average="macro", zero_division=0),
        "f1_macro": f1_score(y_real, y_previsto, average="macro", zero_division=0),
    }


def sortear_hiperparametros(trial):
    # o Optuna escolhe um valor para cada hiperparâmetro dentro dos limites do espaço de busca
    # log=True: sorteia na escala logarítmica (0,0001 a 0,001 tem a mesma chance que 0,001 a 0,01),
    # o certo para valores que variam em ordens de grandeza, como taxa de aprendizado e L2
    return {
        "camadas": trial.suggest_int("camadas", camadas_min, camadas_max),
        "neuronios_primeira_camada": trial.suggest_int("neuronios_primeira_camada", neuronios_min, neuronios_max, log=True),
        "ativacao": trial.suggest_categorical("ativacao", ativacoes),
        "otimizador": trial.suggest_categorical("otimizador", otimizadores),
        "taxa_aprendizado": trial.suggest_float("taxa_aprendizado", taxa_min, taxa_max, log=True),
        "tamanho_batch": trial.suggest_categorical("tamanho_batch", batches),
        "dropout": trial.suggest_float("dropout", 0.0, dropout_max),
        "l2": trial.suggest_float("l2", l2_min, l2_max, log=True),
    }


def neuronios_das_camadas(hiperparametros):
    # formato de pirâmide, igual às redes da etapa 3 (32-16, 64-32): cada camada tem metade dos neurônios da anterior
    # (mínimo de 4); assim só a 1ª camada é sorteada, e todo trial tem os mesmos hiperparâmetros,
    # o que permite ao Optuna calcular a importância de cada um no passo 5
    primeira = hiperparametros["neuronios_primeira_camada"]
    return [max(primeira // (2 ** posicao), 4) for posicao in range(hiperparametros["camadas"])]


def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas) # [64, 32] -> "64-32"


def montar_otimizador(hiperparametros):
    taxa = hiperparametros["taxa_aprendizado"]
    if hiperparametros["otimizador"] == "adam":
        return keras.optimizers.Adam(learning_rate=taxa)
    if hiperparametros["otimizador"] == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=taxa)
    return keras.optimizers.SGD(learning_rate=taxa, momentum=momentum_sgd)


def montar_mlp(hiperparametros):
    modelo = keras.Sequential(name="mlp_optuna")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada"))
    for numero_camada, neuronios in enumerate(neuronios_das_camadas(hiperparametros), start=1):
        # kernel_regularizer = L2 (weight decay): soma na loss uma penalidade proporcional ao tamanho dos pesos,
        # o que empurra a rede para pesos menores e uma fronteira mais "suave" (decora menos o treino)
        modelo.add(keras.layers.Dense(
            neuronios, activation=hiperparametros["ativacao"],
            kernel_regularizer=keras.regularizers.l2(hiperparametros["l2"]), name=f"oculta_{numero_camada}",
        ))
        # Dropout: a cada batch do treino, desliga ao acaso essa fração dos neurônios da camada
        # a rede não pode depender de um neurônio específico; na previsão nada é desligado
        if hiperparametros["dropout"] > 0:
            modelo.add(keras.layers.Dropout(hiperparametros["dropout"], name=f"dropout_{numero_camada}"))
    modelo.add(keras.layers.Dense(numero_classes, activation="softmax", name="saida"))
    modelo.compile(
        optimizer=montar_otimizador(hiperparametros),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro", name="f1_macro")],
    )
    return modelo


def treinar_mlp(hiperparametros):
    # mesmo treino da etapa 3: pesos das classes, validação com os mesmos pesos e parada antecipada pela loss da validação
    keras.utils.set_random_seed(semente) # a mesma semente em todo trial: as diferenças vêm dos hiperparâmetros, não do sorteio
    modelo = montar_mlp(hiperparametros)
    parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)
    inicio = time.perf_counter()
    historico = modelo.fit(
        X_treino, y_treino_rede,
        validation_data=(X_validacao, y_validacao_rede, pesos_linhas_validacao),
        epochs=epocas_maximas,
        batch_size=hiperparametros["tamanho_batch"],
        class_weight=pesos_classes,
        callbacks=[parada_antecipada],
        shuffle=True,
        verbose=0,
    )
    tempo = time.perf_counter() - inicio
    curvas = historico.history
    return {
        "modelo": modelo,
        "curvas": curvas,
        "epocas_treinadas": len(curvas["loss"]),
        "melhor_epoca": int(np.argmin(curvas["val_loss"])) + 1,
        "tempo": tempo,
    }


def prever_classes(modelo, X):
    # a classe com a maior probabilidade (igual à etapa 3)
    return np.asarray(modelo(X, training=False)).argmax(axis=1)


def objetivo(trial):
    # função que o Optuna chama a cada trial: sorteia os hiperparâmetros, treina e devolve a nota (F1 macro na VALIDAÇÃO)
    # o teste não aparece aqui: ele só é usado uma vez, no passo 6, depois que a busca terminou
    hiperparametros = sortear_hiperparametros(trial)
    resultado = treinar_mlp(hiperparametros)
    f1_validacao = avaliar(y_validacao, prever_classes(resultado["modelo"], X_validacao))["f1_macro"]

    # informações extras guardadas no trial, para a tabela de resultados
    trial.set_user_attr("arquitetura", nome_arquitetura(neuronios_das_camadas(hiperparametros)))
    trial.set_user_attr("parametros", resultado["modelo"].count_params())
    trial.set_user_attr("epocas", resultado["epocas_treinadas"])
    trial.set_user_attr("tempo_s", round(resultado["tempo"], 1))
    # True = a rede usou todas as épocas sem a parada antecipada parar (acontece com taxa de aprendizado muito baixa, típico do SGD):
    # ela ainda estava melhorando, então a nota dela é um pouco pessimista
    trial.set_user_attr("parou_no_limite", resultado["epocas_treinadas"] == epocas_maximas)

    keras.backend.clear_session() # libera a memória da rede que acabou de ser avaliada; sem isso, cada trial fica mais lento que o anterior
    return f1_validacao


def mostrar_progresso(estudo, trial):
    # chamada pelo Optuna ao fim de cada trial
    print(f"  trial {trial.number + 1:>3}/{numero_trials}: F1 macro val = {trial.value:.3f}"
          f" | melhor até agora = {estudo.best_value:.3f}"
          f" | rede {trial.user_attrs['arquitetura']}, {trial.params['otimizador']}, {trial.user_attrs['tempo_s']} s")


def desenhar_curvas(resultado, titulo, caminho):
    curvas, melhor_epoca = resultado["curvas"], resultado["melhor_epoca"]
    epocas = range(1, resultado["epocas_treinadas"] + 1)
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    for eixo, medida, titulo_eixo in [(eixos[0], "loss", "Loss (entropia cruzada com pesos + L2)"), (eixos[1], "f1_macro", "F1 macro")]:
        eixo.plot(epocas, curvas[medida], color="steelblue", label="treino")
        eixo.plot(epocas, curvas["val_" + medida], color="orange", label="validação")
        eixo.axvline(melhor_epoca, color="gray", linestyle="--", label=f"melhor época ({melhor_epoca})")
        eixo.set_title(titulo_eixo)
        eixo.set_xlabel("Época")
        eixo.legend()
    figura.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo:", caminho)


# ===============================================================
# PASSO 1: CARREGAR O CONJUNTO E OS RESULTADOS DO MODELO ORIGINAL (ETAPA 3)
# ===============================================================
divisao = sys.argv[1].replace("multiclasse_", "") if len(sys.argv) > 1 else "aleatoria"
if divisao not in divisoes_possiveis:
    raise SystemExit(f"Divisão desconhecida: {divisao} (use {', '.join(divisoes_possiveis)})")
if len(sys.argv) > 2:
    numero_trials = int(sys.argv[2]) # permite um teste rápido: python etapa5_optuna.py aleatoria 5
nome_conjunto = "multiclasse_" + divisao

conjunto = joblib.load(os.path.join(pasta_conjuntos, nome_conjunto + ".joblib"))
X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]
numero_classes = len(nomes_classes)
numero_entradas = X_treino.shape[1]
print("Conjunto carregado:", nome_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")

# modelo original = a MLP escolhida na etapa 3 para esta divisão (as tabelas salvas no passo 9 do etapa3_multiclasse.py)
caminhos_e3 = {nome: os.path.join(pasta_resultados_e3, f"{nome}_{nome_conjunto}.csv") for nome in ["metricas_teste", "comparacao_final", "arquiteturas_mlp"]}
if not all(os.path.exists(caminho) for caminho in caminhos_e3.values()):
    raise SystemExit(f"Faltam os resultados da etapa 3 -> rode antes: python etapa3_multiclasse.py {divisao}")
metricas_e3 = pd.read_csv(caminhos_e3["metricas_teste"], index_col=0)
comparacao_e3 = pd.read_csv(caminhos_e3["comparacao_final"], index_col=0)
arquiteturas_e3 = pd.read_csv(caminhos_e3["arquiteturas_mlp"], dtype={"arquitetura": str})
coluna_original = [coluna for coluna in metricas_e3.columns if coluna.startswith("mlp_")][0] # ex.: "mlp_32-16"
arquitetura_original = coluna_original.replace("mlp_", "")
linha_original = arquiteturas_e3[arquiteturas_e3["arquitetura"] == arquitetura_original].iloc[0]
print(f"Modelo original (etapa 3): MLP {arquitetura_original} | F1 macro na validação = {linha_original['val_f1_macro']:.3f}")


# ===============================================================
# PASSO 2: PESOS DAS CLASSES E DADOS PARA A REDE (IGUAL À ETAPA 3)
# ===============================================================
# o tratamento do desbalanceamento é o mesmo da etapa 3, para que a única diferença entre os modelos sejam os hiperparâmetros
classes_do_treino = np.unique(y_treino)
pesos = compute_class_weight(class_weight="balanced", classes=classes_do_treino, y=y_treino)
pesos_classes = dict(zip(classes_do_treino.tolist(), pesos.tolist()))
pesos_linhas_validacao = np.array([pesos_classes.get(codigo, 1.0) for codigo in y_validacao])
y_treino_rede = keras.utils.to_categorical(y_treino, numero_classes)
y_validacao_rede = keras.utils.to_categorical(y_validacao, numero_classes)


# ===============================================================
# PASSO 3: ESPAÇO DE BUSCA
# ===============================================================
# tabela para o relatório: o que o Optuna pode variar, os limites e por quê
espaco_busca = pd.DataFrame([
    ("camadas", f"{camadas_min} a {camadas_max}", "na etapa 3 só foram testadas 2 camadas; verifica se 1 basta ou se 3 ajuda"),
    ("neuronios_primeira_camada", f"{neuronios_min} a {neuronios_max} (escala log)", "cobre abaixo e acima das redes da etapa 3 (16 a 64); as camadas seguintes têm metade da anterior"),
    ("ativacao", ", ".join(ativacoes), "as duas ativações mais usadas em camadas ocultas"),
    ("otimizador", ", ".join(otimizadores), f"os três citados no enunciado (SGD com momentum fixo de {momentum_sgd})"),
    ("taxa_aprendizado", f"{taxa_min:g} a {taxa_max:g} (escala log)", "de 10x menor a 10x maior que o 0,001 da etapa 3"),
    ("tamanho_batch", ", ".join(str(valor) for valor in batches), "em volta do 32 da etapa 3"),
    ("dropout", f"0 a {dropout_max}", "de nenhum dropout até desligar metade dos neurônios"),
    ("l2", f"{l2_min:g} a {l2_max:g} (escala log)", "de uma penalização quase nula até uma forte"),
], columns=["hiperparametro", "faixa", "justificativa"])
print("\nEspaço de busca:")
print(espaco_busca[["hiperparametro", "faixa"]].to_string(index=False))
print(f"\nÉpocas: até {epocas_maximas}, com parada antecipada (paciência {paciencia}) - não entram na busca, a parada antecipada decide")


# ===============================================================
# PASSO 4: BUSCA COM O OPTUNA
# ===============================================================
# o sampler TPE é o padrão do Optuna: os 10 primeiros trials são sorteados ao acaso, para conhecer o espaço;
# depois ele passa a sortear mais perto das combinações que deram as melhores notas (busca "inteligente", não exaustiva)
# direction="maximize": quanto maior o F1 macro na validação, melhor
# seed: com a mesma semente, o Optuna sorteia as mesmas combinações e a busca pode ser repetida
estudo = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=semente),
    study_name=f"mlp_{nome_conjunto}",
)

print(f"\nIniciando a busca: {numero_trials} trials (cada trial treina uma rede e mede o F1 macro na validação)")
inicio_busca = time.perf_counter()
estudo.optimize(objetivo, n_trials=numero_trials, callbacks=[mostrar_progresso])
tempo_busca = time.perf_counter() - inicio_busca
print(f"\nBusca concluída em {tempo_busca / 60:.1f} minutos")


# ===============================================================
# PASSO 5: RESULTADOS DA BUSCA
# ===============================================================
melhor_trial = estudo.best_trial
melhores_hiperparametros = melhor_trial.params
print(f"\nMelhor trial: nº {melhor_trial.number + 1} | F1 macro na validação = {melhor_trial.value:.3f}"
      f" (original: {linha_original['val_f1_macro']:.3f})")
print("Melhores hiperparâmetros:")
for nome, valor in melhores_hiperparametros.items():
    print(f"  {nome}: {valor:.6g}" if isinstance(valor, float) else f"  {nome}: {valor}")
print("  arquitetura resultante:", melhor_trial.user_attrs["arquitetura"])

# --- tabela com todos os trials ---
tabela_trials = pd.DataFrame([{
    "trial": trial.number + 1,
    "f1_macro_validacao": round(trial.value, 4),
    **trial.params,
    **trial.user_attrs,
} for trial in estudo.trials]).sort_values("f1_macro_validacao", ascending=False)
print(f"Trials que usaram todas as {epocas_maximas} épocas (ainda melhorando quando pararam):", int(tabela_trials["parou_no_limite"].sum()), "de", len(tabela_trials))
print("\nOs 10 melhores trials:")
print(tabela_trials.head(10).to_string(index=False, float_format=lambda valor: f"{valor:.4g}"))

# --- histórico da otimização ---
# cada ponto é um trial; a linha mostra a melhor nota encontrada até aquele trial
# a linha tracejada é o modelo original da etapa 3: o que fica acima dela foi melhor que ele na validação
notas = [trial.value for trial in estudo.trials]
numeros = [trial.number + 1 for trial in estudo.trials]
plt.figure(figsize=(10, 5))
plt.scatter(numeros, notas, s=25, color="steelblue", alpha=0.7, label="F1 macro de cada trial")
plt.plot(numeros, np.maximum.accumulate(notas), color="crimson", label="melhor até o trial") # maximum.accumulate: o maior valor visto até cada posição
plt.axhline(linha_original["val_f1_macro"], color="gray", linestyle="--", label=f"original da etapa 3 ({arquitetura_original})")
if numero_trials > 10:
    plt.axvline(10.5, color="lightgray", linestyle=":", label="fim dos 10 trials aleatórios iniciais")
plt.title(f"Histórico da otimização - {nome_conjunto}")
plt.xlabel("Trial")
plt.ylabel("F1 macro na validação")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # só números inteiros no eixo dos trials
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1)) # legenda fora do gráfico, para não cobrir nenhum ponto
plt.tight_layout()
caminho_historico = os.path.join(pasta_graficos, f"historico_optuna_{nome_conjunto}.png")
plt.savefig(caminho_historico, dpi=120, bbox_inches="tight")
plt.close()
print("\nGráfico salvo:", caminho_historico)

# --- importância dos hiperparâmetros ---
# o Optuna treina um modelo (fANOVA) que tenta prever a nota do trial a partir dos hiperparâmetros,
# e mede quanto da variação das notas é explicada por cada um (a soma dá 1)
# leitura: alto = mudar esse hiperparâmetro muda bastante o F1; baixo = tanto faz o valor escolhido (dentro da faixa testada)
importancias = None
if len(estudo.trials) >= 2:
    importancias = pd.Series(optuna.importance.get_param_importances(
        estudo, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=semente),
    )).sort_values()
    print("\nImportância dos hiperparâmetros:")
    print(importancias.sort_values(ascending=False).round(3).to_string())
    importancias.plot(kind="barh", color="steelblue", figsize=(8, 4.5))
    plt.title(f"Importância dos hiperparâmetros (fANOVA) - {nome_conjunto}")
    plt.xlabel("Importância")
    plt.tight_layout()
    caminho_importancia = os.path.join(pasta_graficos, f"importancia_hiperparametros_{nome_conjunto}.png")
    plt.savefig(caminho_importancia, dpi=120, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo:", caminho_importancia)


# ===============================================================
# PASSO 6: TREINAR A MELHOR CONFIGURAÇÃO E AVALIAR NO TESTE (UMA ÚNICA VEZ)
# ===============================================================
# treina de novo com os melhores hiperparâmetros (mesma semente, então é a mesma rede do melhor trial)
# só agora o teste é usado, uma única vez; nada mais é ajustado depois disso
print("\nTreinando a melhor configuração...")
resultado_otimizado = treinar_mlp(melhores_hiperparametros)
mlp_otimizada = resultado_otimizado["modelo"]
arquitetura_otimizada = melhor_trial.user_attrs["arquitetura"]
print(f"Treino: {resultado_otimizado['epocas_treinadas']} épocas em {resultado_otimizado['tempo']:.1f} s (melhor época: {resultado_otimizado['melhor_epoca']})")
# conferência: com a mesma semente, a rede treinada de novo tem que dar o mesmo F1 macro na validação do melhor trial
f1_validacao_conferencia = avaliar(y_validacao, prever_classes(mlp_otimizada, X_validacao))["f1_macro"]
print(f"Conferência: F1 macro na validação = {f1_validacao_conferencia:.3f} (no melhor trial: {melhor_trial.value:.3f})")
desenhar_curvas(resultado_otimizado, f"Curvas de treinamento da MLP otimizada ({arquitetura_otimizada}) - {nome_conjunto}",
                os.path.join(pasta_graficos, f"curvas_mlp_otimizada_{nome_conjunto}.png"))

y_previsto_teste = prever_classes(mlp_otimizada, X_teste)
metricas_otimizada_teste = avaliar(y_teste, y_previsto_teste)

print("\nMétricas por classe da MLP otimizada no teste:")
print(classification_report(y_teste, y_previsto_teste, labels=list(range(numero_classes)), target_names=nomes_classes, digits=3, zero_division=0))

# matriz de confusão no mesmo formato da etapa 3 (quantidade e % de cada classe real)
figura, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
ConfusionMatrixDisplay.from_predictions(y_teste, y_previsto_teste, labels=list(range(numero_classes)), display_labels=nomes_classes,
                                        cmap="Purples", ax=eixos[0], colorbar=False)
eixos[0].set_title("Quantidade")
ConfusionMatrixDisplay.from_predictions(y_teste, y_previsto_teste, labels=list(range(numero_classes)), display_labels=nomes_classes,
                                        normalize="true", values_format=".0%", cmap="Purples", ax=eixos[1], colorbar=False)
eixos[1].set_title("% de cada classe real (diagonal = recall)")
for eixo in eixos:
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
figura.suptitle(f"Matriz de confusão da MLP otimizada no teste - {nome_conjunto}")
plt.tight_layout()
caminho_matriz = os.path.join(pasta_graficos, f"matriz_confusao_mlp_otimizada_{nome_conjunto}.png")
plt.savefig(caminho_matriz, dpi=120, bbox_inches="tight")
plt.close()
print("Gráfico salvo:", caminho_matriz)


# ===============================================================
# PASSO 7: COMPARAÇÃO MODELO ORIGINAL x MODELO OTIMIZADO
# ===============================================================
# --- métricas no teste: linha de base, original, otimizada e Random Forest (os da etapa 3 vêm das tabelas salvas) ---
metricas_teste = pd.DataFrame({
    "linha_de_base": metricas_e3["linha_de_base"],
    f"mlp_original_{arquitetura_original}": metricas_e3[coluna_original],
    f"mlp_otimizada_{arquitetura_otimizada}": pd.Series(metricas_otimizada_teste),
    "random_forest": metricas_e3["random_forest"],
}).round(3)
print("\nMétricas no TESTE:")
print(metricas_teste.to_string())

# --- original x otimizada: desempenho, generalização e custo ---
f1_treino_otimizada = avaliar(y_treino, prever_classes(mlp_otimizada, X_treino))["f1_macro"]
comparacao = pd.DataFrame({
    f"mlp_original_{arquitetura_original}": {
        "f1_macro_validacao": linha_original["val_f1_macro"],
        "f1_macro_teste": metricas_e3.loc["f1_macro", coluna_original],
        "f1_macro_treino": comparacao_e3.loc["f1_macro_treino", coluna_original],
        "parametros": linha_original["parametros"],
        "epocas": linha_original["epocas"],
        "tempo_treino_s": comparacao_e3.loc["tempo_treino_s", coluna_original],
        "tempo_busca_min": np.nan, # o original não passou por busca
    },
    f"mlp_otimizada_{arquitetura_otimizada}": {
        "f1_macro_validacao": melhor_trial.value,
        "f1_macro_teste": metricas_otimizada_teste["f1_macro"],
        "f1_macro_treino": f1_treino_otimizada,
        "parametros": mlp_otimizada.count_params(),
        "epocas": resultado_otimizado["epocas_treinadas"],
        "tempo_treino_s": resultado_otimizado["tempo"],
        "tempo_busca_min": tempo_busca / 60,
    },
})
comparacao.loc["diferenca_treino_teste"] = comparacao.loc["f1_macro_treino"] - comparacao.loc["f1_macro_teste"] # quanto maior, mais decorou o treino
comparacao = comparacao.round(3)
print("\nModelo original x modelo otimizado:")
print(comparacao.to_string())
ganho = comparacao.iloc[:, 1]["f1_macro_teste"] - comparacao.iloc[:, 0]["f1_macro_teste"]
print(f"\nGanho de F1 macro no teste: {ganho:+.3f}")

# --- gráfico: F1 macro no teste dos quatro modelos ---
grafico = metricas_teste.loc["f1_macro"]
eixo = grafico.plot(kind="bar", figsize=(9, 5), color=["lightgray", "steelblue", "rebeccapurple", "seagreen"], width=0.7)
eixo.bar_label(eixo.containers[0], fmt="%.3f", fontsize=9)
plt.title(f"F1 macro no teste - original x otimizada - {nome_conjunto}")
plt.ylabel("F1 macro")
plt.xticks(rotation=0)
eixo.set_xticklabels(["linha de base", f"MLP original\n{arquitetura_original}", f"MLP otimizada\n{arquitetura_otimizada}", "Random Forest"])
eixo.set_ylim(0, grafico.max() * 1.2)
plt.tight_layout()
caminho_comparacao = os.path.join(pasta_graficos, f"comparacao_original_otimizada_{nome_conjunto}.png")
plt.savefig(caminho_comparacao, dpi=120, bbox_inches="tight")
plt.close()
print("Gráfico salvo:", caminho_comparacao)

# --- salvar as tabelas para o relatório ---
espaco_busca.to_csv(os.path.join(pasta_resultados, f"espaco_busca_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
tabela_trials.to_csv(os.path.join(pasta_resultados, f"trials_optuna_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
pd.Series(melhores_hiperparametros, name="valor").to_csv(os.path.join(pasta_resultados, f"melhores_hiperparametros_{nome_conjunto}.csv"), encoding="utf-8-sig")
if importancias is not None:
    importancias.sort_values(ascending=False).round(4).rename("importancia").to_csv(
        os.path.join(pasta_resultados, f"importancia_hiperparametros_{nome_conjunto}.csv"), encoding="utf-8-sig")
metricas_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao.to_csv(os.path.join(pasta_resultados, f"comparacao_original_otimizada_{nome_conjunto}.csv"), encoding="utf-8-sig")
print("\nTabelas salvas em", pasta_resultados)
