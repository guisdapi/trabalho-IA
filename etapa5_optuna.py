import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna

from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report,
)


pasta_conjuntos = "data/etapa2"
pasta_resultados_e3 = "resultadosE3"
pasta_graficos = "graficosE5"
pasta_resultados = "resultadosE5"
divisoes_possiveis = ["aleatoria", "espacial", "temporal"]

semente = 42
numero_trials = 50
epocas_maximas = 300
paciencia = 20

camadas_min, camadas_max = 1, 3
neuronios_min, neuronios_max = 8, 128
ativacoes = ["relu", "tanh"]
otimizadores = ["adam", "rmsprop", "sgd"]
momentum_sgd = 0.9
taxa_min, taxa_max = 1e-4, 1e-2
batches = [16, 32, 64, 128]
dropout_max = 0.5
l2_min, l2_max = 1e-6, 1e-2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


                                                                 
         
                                                                 
def avaliar(y_real, y_previsto):
    return {
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision_macro": precision_score(y_real, y_previsto, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_previsto, average="macro", zero_division=0),
        "f1_macro": f1_score(y_real, y_previsto, average="macro", zero_division=0),
    }


def sortear_hiperparametros(trial):
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
    primeira = hiperparametros["neuronios_primeira_camada"]
    return [max(primeira // (2 ** posicao), 4) for posicao in range(hiperparametros["camadas"])]


def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas)


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
        modelo.add(keras.layers.Dense(
            neuronios, activation=hiperparametros["ativacao"],
            kernel_regularizer=keras.regularizers.l2(hiperparametros["l2"]), name=f"oculta_{numero_camada}",
        ))
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
    keras.utils.set_random_seed(semente)
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
    return np.asarray(modelo(X, training=False)).argmax(axis=1)


def objetivo(trial):
    hiperparametros = sortear_hiperparametros(trial)
    resultado = treinar_mlp(hiperparametros)
    f1_validacao = avaliar(y_validacao, prever_classes(resultado["modelo"], X_validacao))["f1_macro"]

    trial.set_user_attr("arquitetura", nome_arquitetura(neuronios_das_camadas(hiperparametros)))
    trial.set_user_attr("parametros", resultado["modelo"].count_params())
    trial.set_user_attr("epocas", resultado["epocas_treinadas"])
    trial.set_user_attr("tempo_s", round(resultado["tempo"], 1))
    trial.set_user_attr("parou_no_limite", resultado["epocas_treinadas"] == epocas_maximas)

    keras.backend.clear_session()
    return f1_validacao


def mostrar_progresso(estudo, trial):
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


divisao = sys.argv[1].replace("multiclasse_", "") if len(sys.argv) > 1 else "aleatoria"
if divisao not in divisoes_possiveis:
    raise SystemExit(f"Divisão desconhecida: {divisao} (use {', '.join(divisoes_possiveis)})")
if len(sys.argv) > 2:
    numero_trials = int(sys.argv[2])
nome_conjunto = "multiclasse_" + divisao

conjunto = joblib.load(os.path.join(pasta_conjuntos, nome_conjunto + ".joblib"))
X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]
numero_classes = len(nomes_classes)
numero_entradas = X_treino.shape[1]
print("Conjunto carregado:", nome_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")

caminhos_e3 = {nome: os.path.join(pasta_resultados_e3, f"{nome}_{nome_conjunto}.csv") for nome in ["metricas_teste", "comparacao_final", "arquiteturas_mlp"]}
if not all(os.path.exists(caminho) for caminho in caminhos_e3.values()):
    raise SystemExit(f"Faltam os resultados da etapa 3 -> rode antes: python etapa3_multiclasse.py {divisao}")
metricas_e3 = pd.read_csv(caminhos_e3["metricas_teste"], index_col=0)
comparacao_e3 = pd.read_csv(caminhos_e3["comparacao_final"], index_col=0)
arquiteturas_e3 = pd.read_csv(caminhos_e3["arquiteturas_mlp"], dtype={"arquitetura": str})
coluna_original = [coluna for coluna in metricas_e3.columns if coluna.startswith("mlp_")][0]
arquitetura_original = coluna_original.replace("mlp_", "")
linha_original = arquiteturas_e3[arquiteturas_e3["arquitetura"] == arquitetura_original].iloc[0]
print(f"Modelo original (etapa 3): MLP {arquitetura_original} | F1 macro na validação = {linha_original['val_f1_macro']:.3f}")


classes_do_treino = np.unique(y_treino)
pesos = compute_class_weight(class_weight="balanced", classes=classes_do_treino, y=y_treino)
pesos_classes = dict(zip(classes_do_treino.tolist(), pesos.tolist()))
pesos_linhas_validacao = np.array([pesos_classes.get(codigo, 1.0) for codigo in y_validacao])
y_treino_rede = keras.utils.to_categorical(y_treino, numero_classes)
y_validacao_rede = keras.utils.to_categorical(y_validacao, numero_classes)


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


melhor_trial = estudo.best_trial
melhores_hiperparametros = melhor_trial.params
print(f"\nMelhor trial: nº {melhor_trial.number + 1} | F1 macro na validação = {melhor_trial.value:.3f}"
      f" (original: {linha_original['val_f1_macro']:.3f})")
print("Melhores hiperparâmetros:")
for nome, valor in melhores_hiperparametros.items():
    print(f"  {nome}: {valor:.6g}" if isinstance(valor, float) else f"  {nome}: {valor}")
print("  arquitetura resultante:", melhor_trial.user_attrs["arquitetura"])

tabela_trials = pd.DataFrame([{
    "trial": trial.number + 1,
    "f1_macro_validacao": round(trial.value, 4),
    **trial.params,
    **trial.user_attrs,
} for trial in estudo.trials]).sort_values("f1_macro_validacao", ascending=False)
print(f"Trials que usaram todas as {epocas_maximas} épocas (ainda melhorando quando pararam):", int(tabela_trials["parou_no_limite"].sum()), "de", len(tabela_trials))
print("\nOs 10 melhores trials:")
print(tabela_trials.head(10).to_string(index=False, float_format=lambda valor: f"{valor:.4g}"))

notas = [trial.value for trial in estudo.trials]
numeros = [trial.number + 1 for trial in estudo.trials]
plt.figure(figsize=(10, 5))
plt.scatter(numeros, notas, s=25, color="steelblue", alpha=0.7, label="F1 macro de cada trial")
plt.plot(numeros, np.maximum.accumulate(notas), color="crimson", label="melhor até o trial")
plt.axhline(linha_original["val_f1_macro"], color="gray", linestyle="--", label=f"original da etapa 3 ({arquitetura_original})")
if numero_trials > 10:
    plt.axvline(10.5, color="lightgray", linestyle=":", label="fim dos 10 trials aleatórios iniciais")
plt.title(f"Histórico da otimização - {nome_conjunto}")
plt.xlabel("Trial")
plt.ylabel("F1 macro na validação")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
caminho_historico = os.path.join(pasta_graficos, f"historico_optuna_{nome_conjunto}.png")
plt.savefig(caminho_historico, dpi=120, bbox_inches="tight")
plt.close()
print("\nGráfico salvo:", caminho_historico)

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


print("\nTreinando a melhor configuração...")
resultado_otimizado = treinar_mlp(melhores_hiperparametros)
mlp_otimizada = resultado_otimizado["modelo"]
arquitetura_otimizada = melhor_trial.user_attrs["arquitetura"]
print(f"Treino: {resultado_otimizado['epocas_treinadas']} épocas em {resultado_otimizado['tempo']:.1f} s (melhor época: {resultado_otimizado['melhor_epoca']})")
f1_validacao_conferencia = avaliar(y_validacao, prever_classes(mlp_otimizada, X_validacao))["f1_macro"]
print(f"Conferência: F1 macro na validação = {f1_validacao_conferencia:.3f} (no melhor trial: {melhor_trial.value:.3f})")
desenhar_curvas(resultado_otimizado, f"Curvas de treinamento da MLP otimizada ({arquitetura_otimizada}) - {nome_conjunto}",
                os.path.join(pasta_graficos, f"curvas_mlp_otimizada_{nome_conjunto}.png"))

y_previsto_teste = prever_classes(mlp_otimizada, X_teste)
metricas_otimizada_teste = avaliar(y_teste, y_previsto_teste)

print("\nMétricas por classe da MLP otimizada no teste:")
print(classification_report(y_teste, y_previsto_teste, labels=list(range(numero_classes)), target_names=nomes_classes, digits=3, zero_division=0))

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


metricas_teste = pd.DataFrame({
    "linha_de_base": metricas_e3["linha_de_base"],
    f"mlp_original_{arquitetura_original}": metricas_e3[coluna_original],
    f"mlp_otimizada_{arquitetura_otimizada}": pd.Series(metricas_otimizada_teste),
    "random_forest": metricas_e3["random_forest"],
}).round(3)
print("\nMétricas no TESTE:")
print(metricas_teste.to_string())

f1_treino_otimizada = avaliar(y_treino, prever_classes(mlp_otimizada, X_treino))["f1_macro"]
comparacao = pd.DataFrame({
    f"mlp_original_{arquitetura_original}": {
        "f1_macro_validacao": linha_original["val_f1_macro"],
        "f1_macro_teste": metricas_e3.loc["f1_macro", coluna_original],
        "f1_macro_treino": comparacao_e3.loc["f1_macro_treino", coluna_original],
        "parametros": linha_original["parametros"],
        "epocas": linha_original["epocas"],
        "tempo_treino_s": comparacao_e3.loc["tempo_treino_s", coluna_original],
        "tempo_busca_min": np.nan,
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
comparacao.loc["diferenca_treino_teste"] = comparacao.loc["f1_macro_treino"] - comparacao.loc["f1_macro_teste"]
comparacao = comparacao.round(3)
print("\nModelo original x modelo otimizado:")
print(comparacao.to_string())
ganho = comparacao.iloc[:, 1]["f1_macro_teste"] - comparacao.iloc[:, 0]["f1_macro_teste"]
print(f"\nGanho de F1 macro no teste: {ganho:+.3f}")

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

espaco_busca.to_csv(os.path.join(pasta_resultados, f"espaco_busca_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
tabela_trials.to_csv(os.path.join(pasta_resultados, f"trials_optuna_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
pd.Series(melhores_hiperparametros, name="valor").to_csv(os.path.join(pasta_resultados, f"melhores_hiperparametros_{nome_conjunto}.csv"), encoding="utf-8-sig")
if importancias is not None:
    importancias.sort_values(ascending=False).round(4).rename("importancia").to_csv(
        os.path.join(pasta_resultados, f"importancia_hiperparametros_{nome_conjunto}.csv"), encoding="utf-8-sig")
metricas_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao.to_csv(os.path.join(pasta_resultados, f"comparacao_original_otimizada_{nome_conjunto}.csv"), encoding="utf-8-sig")
print("\nTabelas salvas em", pasta_resultados)
