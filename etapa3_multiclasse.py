import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow import keras

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


pasta_conjuntos = "data/etapa2"
pasta_graficos = "graficosE3"
pasta_resultados = "resultadosE3"
divisoes_possiveis = ["aleatoria", "espacial", "temporal"]

semente = 42

neuronios_camadas_ocultas = [32, 16]
arquiteturas_candidatas = [[16, 8], [32, 16], [64, 32]]
tolerancia_desempate = 0.01
taxa_aprendizado = 0.001
tamanho_batch = 32
epocas_maximas = 300
paciencia = 20

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)





# Calcula accuracy, precision, recall e F1 macro
def avaliar(y_real, y_previsto):
    return {
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision_macro": precision_score(y_real, y_previsto, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_previsto, average="macro", zero_division=0),
        "f1_macro": f1_score(y_real, y_previsto, average="macro", zero_division=0),
    }


# Monta a MLP com as camadas ocultas ReLU e a saída softmax com um neurônio por classe
def montar_mlp(numero_entradas, neuronios_camadas):
    modelo = keras.Sequential(name="mlp")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada"))

    for numero_camada, neuronios in enumerate(neuronios_camadas, start=1):
        modelo.add(keras.layers.Dense(neuronios, activation="relu", name=f"oculta_{numero_camada}"))

    modelo.add(keras.layers.Dense(numero_classes, activation="softmax", name="saida"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro", name="f1_macro")],
    )
    return modelo


# Treina a MLP com pesos de classe e parada antecipada, guardando as curvas e o tempo
def treinar_mlp(neuronios_camadas, X_treino_rede, X_validacao_rede):
    keras.utils.set_random_seed(semente)
    modelo = montar_mlp(X_treino_rede.shape[1], neuronios_camadas)

    parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)

    inicio = time.perf_counter()
    historico = modelo.fit(
        X_treino_rede, y_treino_rede,
        validation_data=(X_validacao_rede, y_validacao_rede, pesos_linhas_validacao),
        epochs=epocas_maximas,
        batch_size=tamanho_batch,
        class_weight=pesos_classes,
        callbacks=[parada_antecipada],
        shuffle=True,
        verbose=0,
    )
    tempo = time.perf_counter() - inicio

    curvas = historico.history
    epocas_treinadas = len(curvas["loss"])
    if epocas_treinadas == epocas_maximas:
        print(f"  AVISO: a MLP {nome_arquitetura(neuronios_camadas)} usou todas as {epocas_maximas} épocas; considere aumentar epocas_maximas")
    return {
        "neuronios": neuronios_camadas,
        "modelo": modelo,
        "curvas": curvas,
        "epocas_treinadas": epocas_treinadas,
        "melhor_epoca": int(np.argmin(curvas["val_loss"])) + 1,
        "tempo": tempo,
    }


# Escolhe a classe com a maior probabilidade
def prever_classes(modelo, X):
    probabilidades = np.asarray(modelo(X, training=False))
    return probabilidades.argmax(axis=1)


# Gera o nome da arquitetura a partir dos neurônios das camadas (ex.: 32-16)
def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas)


# Desenha as curvas de erro e de F1 no treino e na validação ao longo das épocas
def desenhar_curvas(resultado):
    curvas, melhor_epoca = resultado["curvas"], resultado["melhor_epoca"]
    epocas = range(1, resultado["epocas_treinadas"] + 1)
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    for eixo, medida, titulo in [(eixos[0], "loss", "Loss (entropia cruzada com pesos)"), (eixos[1], "f1_macro", "F1 macro")]:
        eixo.plot(epocas, curvas[medida], color="steelblue", label="treino")
        eixo.plot(epocas, curvas["val_" + medida], color="orange", label="validação")
        eixo.axvline(melhor_epoca, color="gray", linestyle="--", label=f"melhor época ({melhor_epoca})")
        eixo.set_title(titulo)
        eixo.set_xlabel("Época")
        eixo.legend()
    arquitetura = nome_arquitetura(resultado["neuronios"])
    figura.suptitle(f"Curvas de treinamento da MLP {arquitetura} - {nome_conjunto}")
    plt.tight_layout()
    caminho = os.path.join(pasta_graficos, f"curvas_mlp_{nome_conjunto}_{arquitetura}.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    print("Gráfico salvo:", caminho)


# Desenha e salva a matriz de confusão do modelo
def desenhar_matriz_confusao(y_previsto, nome_modelo, mapa_de_cores, nome_arquivo):
    figura, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_previsto, labels=list(range(numero_classes)), display_labels=nomes_classes,
        cmap=mapa_de_cores, ax=eixos[0], colorbar=False,
    )
    eixos[0].set_title("Quantidade")
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_previsto, labels=list(range(numero_classes)), display_labels=nomes_classes,
        normalize="true", values_format=".0%", cmap=mapa_de_cores, ax=eixos[1], colorbar=False,
    )
    eixos[1].set_title("% de cada classe real (diagonal = recall)")
    for eixo in eixos:
        eixo.set_xlabel("Classe prevista")
        eixo.set_ylabel("Classe real")
    figura.suptitle(f"Matriz de confusão {nome_modelo} no teste - {nome_conjunto}")
    plt.tight_layout()
    caminho = os.path.join(pasta_graficos, nome_arquivo)
    plt.savefig(caminho, dpi=120)
    plt.close()
    print("Gráfico salvo:", caminho)


divisao = sys.argv[1].replace("multiclasse_", "") if len(sys.argv) > 1 else "aleatoria"
if divisao not in divisoes_possiveis:
    raise SystemExit(f"Divisão desconhecida: {divisao} (use {', '.join(divisoes_possiveis)})")
nome_conjunto = "multiclasse_" + divisao
caminho_conjunto = os.path.join(pasta_conjuntos, nome_conjunto + ".joblib")
conjunto = joblib.load(caminho_conjunto)

X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]
numero_classes = len(nomes_classes)

print("Conjunto carregado:", caminho_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")
print("Código de cada classe:", dict(enumerate(nomes_classes)))


linha_de_base = DummyClassifier(strategy="most_frequent")
linha_de_base.fit(X_treino, y_treino)

classe_chutada = nomes_classes[linha_de_base.predict(X_treino[:1])[0]]
print("\nA linha de base prevê sempre:", classe_chutada)

resultados_linha_de_base = pd.DataFrame({
    "validacao": avaliar(y_validacao, linha_de_base.predict(X_validacao)),
    "teste": avaliar(y_teste, linha_de_base.predict(X_teste)),
}).round(3)

print("\nMétricas da linha de base:")
print(resultados_linha_de_base.to_string())
classes_do_treino = np.unique(y_treino)
pesos = compute_class_weight(class_weight="balanced", classes=classes_do_treino, y=y_treino)
pesos_classes = dict(zip(classes_do_treino.tolist(), pesos.tolist()))

print("\nPeso de cada classe no treino:")
print(pd.DataFrame({
    "classe": [nomes_classes[codigo] for codigo in classes_do_treino],
    "linhas no treino": pd.Series(y_treino).value_counts().sort_index().to_numpy(),
    "peso": np.round(pesos, 2),
}).to_string(index=False))


numero_entradas = X_treino.shape[1]

y_treino_rede = keras.utils.to_categorical(y_treino, numero_classes)
y_validacao_rede = keras.utils.to_categorical(y_validacao, numero_classes)

pesos_linhas_validacao = np.array([pesos_classes.get(codigo, 1.0) for codigo in y_validacao])

print()
montar_mlp(numero_entradas, neuronios_camadas_ocultas).summary()


resultado_inicial = treinar_mlp(neuronios_camadas_ocultas, X_treino, X_validacao)
curvas, melhor_epoca = resultado_inicial["curvas"], resultado_inicial["melhor_epoca"]

print(f"\nTreino da MLP {nome_arquitetura(neuronios_camadas_ocultas)}: {resultado_inicial['epocas_treinadas']} épocas em {resultado_inicial['tempo']:.1f} s")
print("Melhor época (menor loss de validação):", melhor_epoca, "-> a rede ficou com os pesos dessa época")
print("\nMedidas na melhor época:")
print(pd.DataFrame({
    "treino": [curvas["loss"][melhor_epoca - 1], curvas["accuracy"][melhor_epoca - 1], curvas["f1_macro"][melhor_epoca - 1]],
    "validacao": [curvas["val_loss"][melhor_epoca - 1], curvas["val_accuracy"][melhor_epoca - 1], curvas["val_f1_macro"][melhor_epoca - 1]],
}, index=["loss", "accuracy", "f1_macro"]).round(3).to_string())
desenhar_curvas(resultado_inicial)


resultados_arquiteturas = {nome_arquitetura(neuronios_camadas_ocultas): resultado_inicial}
for neuronios_camadas in arquiteturas_candidatas:
    arquitetura = nome_arquitetura(neuronios_camadas)
    if arquitetura not in resultados_arquiteturas:
        print(f"\nTreinando a MLP {arquitetura}...")
        resultados_arquiteturas[arquitetura] = treinar_mlp(neuronios_camadas, X_treino, X_validacao)
        desenhar_curvas(resultados_arquiteturas[arquitetura])

linhas_comparacao = []
for arquitetura, resultado in resultados_arquiteturas.items():
    metricas_validacao = avaliar(y_validacao, prever_classes(resultado["modelo"], X_validacao))
    linhas_comparacao.append({
        "arquitetura": arquitetura,
        "parametros": resultado["modelo"].count_params(),
        "epocas": resultado["epocas_treinadas"],
        "melhor_epoca": resultado["melhor_epoca"],
        "tempo_s": round(resultado["tempo"], 1),
        "val_loss": round(min(resultado["curvas"]["val_loss"]), 3),
        "val_accuracy": round(metricas_validacao["accuracy"], 3),
        "val_f1_macro": round(metricas_validacao["f1_macro"], 3),
    })
comparacao_arquiteturas = pd.DataFrame(linhas_comparacao).sort_values("parametros")
print("\nComparação das arquiteturas na validação:")
print(comparacao_arquiteturas.to_string(index=False))

melhor_f1 = comparacao_arquiteturas["val_f1_macro"].max()
quase_empatadas = comparacao_arquiteturas[comparacao_arquiteturas["val_f1_macro"] >= melhor_f1 - tolerancia_desempate]
arquitetura_escolhida = quase_empatadas.iloc[0]["arquitetura"]

resultado_escolhido = resultados_arquiteturas[arquitetura_escolhida]
mlp = resultado_escolhido["modelo"]
tempo_treino_mlp = resultado_escolhido["tempo"]
coluna_mlp = "mlp_" + arquitetura_escolhida
print(f"\nArquitetura escolhida: {arquitetura_escolhida} (melhor F1 macro na validação: {melhor_f1:.3f}; tolerância de desempate: {tolerancia_desempate})")


y_previsto_teste = prever_classes(mlp, X_teste)

resultados_teste = pd.DataFrame({
    "linha_de_base": resultados_linha_de_base["teste"],
    coluna_mlp: avaliar(y_teste, y_previsto_teste),
}).round(3)
print("\nMétricas no TESTE:")
print(resultados_teste.to_string())
print("\nMétricas por classe no teste:")
print(classification_report(y_teste, y_previsto_teste, labels=list(range(numero_classes)), target_names=nomes_classes, digits=3, zero_division=0))

matriz = confusion_matrix(y_teste, y_previsto_teste, labels=list(range(numero_classes)))
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(pd.DataFrame(matriz, index=["real_" + nome for nome in nomes_classes], columns=["prev_" + nome for nome in nomes_classes]).to_string())
print()
desenhar_matriz_confusao(y_previsto_teste, f"da MLP {arquitetura_escolhida}", "Blues", f"matriz_confusao_mlp_{nome_conjunto}.png")


candidatos_rf = [
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": 15, "min_samples_leaf": 1},
    {"max_depth": 15, "min_samples_leaf": 5},
]

linhas_rf = []
modelos_rf = []
for parametros in candidatos_rf:
    floresta = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=semente, n_jobs=-1, **parametros)
    inicio = time.perf_counter()
    floresta.fit(X_treino, y_treino)
    tempo = time.perf_counter() - inicio
    metricas_validacao = avaliar(y_validacao, floresta.predict(X_validacao))
    modelos_rf.append((floresta, tempo))
    linhas_rf.append({
        "max_depth": "sem limite" if parametros["max_depth"] is None else parametros["max_depth"],
        "min_samples_leaf": parametros["min_samples_leaf"],
        "tempo_s": round(tempo, 2),
        "treino_f1_macro": round(avaliar(y_treino, floresta.predict(X_treino))["f1_macro"], 3),
        "val_accuracy": round(metricas_validacao["accuracy"], 3),
        "val_f1_macro": round(metricas_validacao["f1_macro"], 3),
    })

comparacao_rf = pd.DataFrame(linhas_rf)
print("\nRandom Forest - comparação dos parâmetros na validação:")
print(comparacao_rf.to_string(index=False))
posicao_escolhida = int(comparacao_rf["val_f1_macro"].idxmax())
rf, tempo_treino_rf = modelos_rf[posicao_escolhida]
parametros_rf_escolhidos = candidatos_rf[posicao_escolhida]
print("\nParâmetros escolhidos para o Random Forest:", parametros_rf_escolhidos)

y_previsto_teste_rf = rf.predict(X_teste)
resultados_teste["random_forest"] = pd.Series(avaliar(y_teste, y_previsto_teste_rf)).round(3)
print("\nMétricas no TESTE (linha de base x MLP x Random Forest):")
print(resultados_teste.to_string())

print("\nMétricas por classe do Random Forest no teste:")
print(classification_report(y_teste, y_previsto_teste_rf, labels=list(range(numero_classes)), target_names=nomes_classes, digits=3, zero_division=0))

desenhar_matriz_confusao(y_previsto_teste_rf, "do Random Forest", "Greens", f"matriz_confusao_rf_{nome_conjunto}.png")

nomes_atributos_curtos = [nome.split("__")[1] for nome in conjunto["nomes_atributos"]]
importancias = pd.Series(rf.feature_importances_, index=nomes_atributos_curtos).sort_values()
print("\nImportância dos atributos no Random Forest:")
print(importancias.sort_values(ascending=False).round(3).to_string())

importancias.plot(kind="barh", color="seagreen", figsize=(8, 4.5))
plt.title(f"Importância dos atributos - Random Forest ({nome_conjunto})")
plt.xlabel("Importância")
plt.tight_layout()
caminho_importancia = os.path.join(pasta_graficos, f"importancia_atributos_rf_{nome_conjunto}.png")
plt.savefig(caminho_importancia, dpi=120)
plt.close()
print("Gráfico salvo:", caminho_importancia)


padronizador = conjunto["preprocessador"].named_transformers_["numericas"]
quantidade_numericas = len(padronizador.mean_)

# Volta os atributos numéricos para a escala original, para testar os modelos sem padronização
def desfazer_padronizacao(X):
    X_original = X.copy()
    X_original[:, :quantidade_numericas] = padronizador.inverse_transform(X[:, :quantidade_numericas])
    return X_original

X_treino_sem, X_validacao_sem, X_teste_sem = desfazer_padronizacao(X_treino), desfazer_padronizacao(X_validacao), desfazer_padronizacao(X_teste)
print("\nExemplo de x e y sem padronização (primeira linha do treino):", X_treino_sem[0, :2].round(0))

print(f"\nTreinando a MLP {arquitetura_escolhida} SEM padronização...")
mlp_sem = treinar_mlp(resultado_escolhido["neuronios"], X_treino_sem, X_validacao_sem)["modelo"]
f1_mlp_sem = avaliar(y_teste, prever_classes(mlp_sem, X_teste_sem))["f1_macro"]

rf_sem = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=semente, n_jobs=-1, **parametros_rf_escolhidos)
rf_sem.fit(X_treino_sem, y_treino)
f1_rf_sem = avaliar(y_teste, rf_sem.predict(X_teste_sem))["f1_macro"]
inicio = time.perf_counter()
prever_classes(mlp, X_teste)
tempo_previsao_mlp = time.perf_counter() - inicio
inicio = time.perf_counter()
rf.predict(X_teste)
tempo_previsao_rf = time.perf_counter() - inicio

comparacao_final = pd.DataFrame({
    coluna_mlp: {
        "accuracy_teste": resultados_teste.loc["accuracy", coluna_mlp],
        "f1_macro_teste": resultados_teste.loc["f1_macro", coluna_mlp],
        "f1_macro_treino": avaliar(y_treino, prever_classes(mlp, X_treino))["f1_macro"],
        "f1_macro_teste_sem_padronizacao": f1_mlp_sem,
        "tempo_treino_s": tempo_treino_mlp,
        "tempo_previsao_teste_ms": tempo_previsao_mlp * 1000,
    },
    "random_forest": {
        "accuracy_teste": resultados_teste.loc["accuracy", "random_forest"],
        "f1_macro_teste": resultados_teste.loc["f1_macro", "random_forest"],
        "f1_macro_treino": avaliar(y_treino, rf.predict(X_treino))["f1_macro"],
        "f1_macro_teste_sem_padronizacao": f1_rf_sem,
        "tempo_treino_s": tempo_treino_rf,
        "tempo_previsao_teste_ms": tempo_previsao_rf * 1000,
    },
}).round(3)
comparacao_final.loc["diferenca_treino_teste"] = (comparacao_final.loc["f1_macro_treino"] - comparacao_final.loc["f1_macro_teste"]).round(3)

print(f"\nComparação final MLP x Random Forest ({nome_conjunto}):")
print(comparacao_final.to_string())

resultados_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_final.to_csv(os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_arquiteturas.to_csv(os.path.join(pasta_resultados, f"arquiteturas_mlp_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
comparacao_rf.to_csv(os.path.join(pasta_resultados, f"parametros_rf_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
print("\nTabelas salvas em", pasta_resultados)
