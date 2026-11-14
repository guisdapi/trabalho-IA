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
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)


pasta_conjuntos = "data/etapa2"
pasta_graficos = "graficosE3"
pasta_resultados = "resultadosE3"
divisoes_possiveis = ["aleatoria", "espacial"]

semente = 42

neuronios_camadas_ocultas = [32, 16]
arquiteturas_candidatas = [[16, 8], [32, 16], [64, 32]]
tolerancia_desempate = 0.01
taxa_aprendizado = 0.001
tamanho_batch = 32
epocas_maximas = 1000
paciencia = 20
corte_hotspot = 0.5
cortes_testados = np.arange(0.20, 0.81, 0.05)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)





# Calcula as métricas de classificação (accuracy, precision, recall, F1 e ROC-AUC)
def avaliar(y_real, y_previsto, probabilidade_hotspot=None):
    metricas = {
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision": precision_score(y_real, y_previsto, zero_division=0),
        "recall": recall_score(y_real, y_previsto, zero_division=0),
        "f1": f1_score(y_real, y_previsto, zero_division=0),
    }
    if probabilidade_hotspot is not None:
        metricas["roc_auc"] = roc_auc_score(y_real, probabilidade_hotspot)
    return metricas


# Escolhe, na validação, o corte de probabilidade com o melhor F1
def escolher_corte(y_real, probabilidade_hotspot):
    f1_de_cada_corte = [f1_score(y_real, (probabilidade_hotspot >= corte).astype(int), zero_division=0) for corte in cortes_testados]
    return float(round(cortes_testados[int(np.argmax(f1_de_cada_corte))], 2))


# Monta a MLP com as camadas ocultas ReLU e um neurônio de saída com sigmoide
def montar_mlp(numero_entradas, neuronios_camadas):
    modelo = keras.Sequential(name="mlp")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada"))

    for numero_camada, neuronios in enumerate(neuronios_camadas, start=1):
        modelo.add(keras.layers.Dense(neuronios, activation="relu", name=f"oculta_{numero_camada}"))

    modelo.add(keras.layers.Dense(1, activation="sigmoid", name="saida"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="micro", threshold=corte_hotspot, name="f1")],
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


# Retorna a probabilidade de cada segmento ser hotspot
def prever_probabilidade(modelo, X):
    return np.asarray(modelo(X, training=False))[:, 0]


# Transforma a probabilidade em 0 ou 1 usando o corte escolhido para a MLP
def prever_classes(modelo, X):
    return (prever_probabilidade(modelo, X) >= corte_hotspot).astype(int)


# Gera o nome da arquitetura a partir dos neurônios das camadas (ex.: 32-16)
def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas)


# Desenha as curvas de erro e de F1 no treino e na validação ao longo das épocas
def desenhar_curvas(resultado):
    curvas, melhor_epoca = resultado["curvas"], resultado["melhor_epoca"]
    epocas = range(1, resultado["epocas_treinadas"] + 1)
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    for eixo, medida, titulo in [(eixos[0], "loss", "Loss (entropia cruzada com pesos)"), (eixos[1], "f1", "F1 da classe hotspot")]:
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
        y_teste, y_previsto, labels=[0, 1], display_labels=nomes_classes,
        cmap=mapa_de_cores, ax=eixos[0], colorbar=False,
    )
    eixos[0].set_title("Quantidade")
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_previsto, labels=[0, 1], display_labels=nomes_classes,
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


divisao = sys.argv[1].replace("hotspot_", "") if len(sys.argv) > 1 else "aleatoria"
if divisao not in divisoes_possiveis:
    raise SystemExit(f"Divisão desconhecida: {divisao} (use {', '.join(divisoes_possiveis)})")
nome_conjunto = "hotspot_" + divisao
caminho_conjunto = os.path.join(pasta_conjuntos, nome_conjunto + ".joblib")
conjunto = joblib.load(caminho_conjunto)

X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]

print("Conjunto carregado:", caminho_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")
print("Código de cada classe:", dict(enumerate(nomes_classes)))


linha_de_base = DummyClassifier(strategy="most_frequent")
linha_de_base.fit(X_treino, y_treino)

classe_chutada = nomes_classes[linha_de_base.predict(X_treino[:1])[0]]
print("\nA linha de base prevê sempre:", classe_chutada)

resultados_linha_de_base = pd.DataFrame({
    "validacao": avaliar(y_validacao, linha_de_base.predict(X_validacao), linha_de_base.predict_proba(X_validacao)[:, 1]),
    "teste": avaliar(y_teste, linha_de_base.predict(X_teste), linha_de_base.predict_proba(X_teste)[:, 1]),
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

y_treino_rede = y_treino.reshape(-1, 1).astype("float32")
y_validacao_rede = y_validacao.reshape(-1, 1).astype("float32")

pesos_linhas_validacao = np.array([pesos_classes.get(codigo, 1.0) for codigo in y_validacao])

print()
montar_mlp(numero_entradas, neuronios_camadas_ocultas).summary()


resultado_inicial = treinar_mlp(neuronios_camadas_ocultas, X_treino, X_validacao)
curvas, melhor_epoca = resultado_inicial["curvas"], resultado_inicial["melhor_epoca"]

print(f"\nTreino da MLP {nome_arquitetura(neuronios_camadas_ocultas)}: {resultado_inicial['epocas_treinadas']} épocas em {resultado_inicial['tempo']:.1f} s")
print("Melhor época (menor loss de validação):", melhor_epoca, "-> a rede ficou com os pesos dessa época")
print("\nMedidas na melhor época:")
print(pd.DataFrame({
    "treino": [curvas["loss"][melhor_epoca - 1], curvas["accuracy"][melhor_epoca - 1], curvas["f1"][melhor_epoca - 1]],
    "validacao": [curvas["val_loss"][melhor_epoca - 1], curvas["val_accuracy"][melhor_epoca - 1], curvas["val_f1"][melhor_epoca - 1]],
}, index=["loss", "accuracy", "f1"]).round(3).to_string())
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
    modelo_candidato = resultado["modelo"]
    metricas_validacao = avaliar(y_validacao, prever_classes(modelo_candidato, X_validacao), prever_probabilidade(modelo_candidato, X_validacao))
    linhas_comparacao.append({
        "arquitetura": arquitetura,
        "parametros": modelo_candidato.count_params(),
        "epocas": resultado["epocas_treinadas"],
        "melhor_epoca": resultado["melhor_epoca"],
        "tempo_s": round(resultado["tempo"], 1),
        "val_loss": round(min(resultado["curvas"]["val_loss"]), 3),
        "val_accuracy": round(metricas_validacao["accuracy"], 3),
        "val_f1": round(metricas_validacao["f1"], 3),
        "val_roc_auc": round(metricas_validacao["roc_auc"], 3),
    })
comparacao_arquiteturas = pd.DataFrame(linhas_comparacao).sort_values("parametros")
print("\nComparação das arquiteturas na validação:")
print(comparacao_arquiteturas.to_string(index=False))

melhor_auc = comparacao_arquiteturas["val_roc_auc"].max()
quase_empatadas = comparacao_arquiteturas[comparacao_arquiteturas["val_roc_auc"] >= melhor_auc - tolerancia_desempate]
arquitetura_escolhida = quase_empatadas.iloc[0]["arquitetura"]

resultado_escolhido = resultados_arquiteturas[arquitetura_escolhida]
mlp = resultado_escolhido["modelo"]
tempo_treino_mlp = resultado_escolhido["tempo"]
coluna_mlp = "mlp_" + arquitetura_escolhida
print(f"\nArquitetura escolhida: {arquitetura_escolhida} (melhor ROC-AUC na validação: {melhor_auc:.3f}; tolerância de desempate: {tolerancia_desempate})")

corte_hotspot = escolher_corte(y_validacao, prever_probabilidade(mlp, X_validacao))
f1_validacao_corte_05 = f1_score(y_validacao, (prever_probabilidade(mlp, X_validacao) >= 0.5).astype(int), zero_division=0)
f1_validacao_corte_novo = f1_score(y_validacao, prever_classes(mlp, X_validacao), zero_division=0)
print(f"Corte da MLP escolhido na validação: {corte_hotspot:.2f} (F1 na validação: {f1_validacao_corte_05:.3f} com 0,5 -> {f1_validacao_corte_novo:.3f} com {corte_hotspot:.2f})")


probabilidade_teste_mlp = prever_probabilidade(mlp, X_teste)
y_previsto_teste = (probabilidade_teste_mlp >= corte_hotspot).astype(int)

resultados_teste = pd.DataFrame({
    "linha_de_base": resultados_linha_de_base["teste"],
    coluna_mlp: avaliar(y_teste, y_previsto_teste, probabilidade_teste_mlp),
}).round(3)
print("\nMétricas no TESTE:")
print(resultados_teste.to_string())
print("\nMétricas por classe no teste:")
print(classification_report(y_teste, y_previsto_teste, labels=[0, 1], target_names=nomes_classes, digits=3, zero_division=0))

matriz = confusion_matrix(y_teste, y_previsto_teste, labels=[0, 1])
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
    metricas_validacao = avaliar(y_validacao, floresta.predict(X_validacao), floresta.predict_proba(X_validacao)[:, 1])
    modelos_rf.append((floresta, tempo))
    linhas_rf.append({
        "max_depth": "sem limite" if parametros["max_depth"] is None else parametros["max_depth"],
        "min_samples_leaf": parametros["min_samples_leaf"],
        "tempo_s": round(tempo, 2),
        "treino_f1": round(avaliar(y_treino, floresta.predict(X_treino))["f1"], 3),
        "val_accuracy": round(metricas_validacao["accuracy"], 3),
        "val_f1": round(metricas_validacao["f1"], 3),
        "val_roc_auc": round(metricas_validacao["roc_auc"], 3),
    })

comparacao_rf = pd.DataFrame(linhas_rf)
print("\nRandom Forest - comparação dos parâmetros na validação:")
print(comparacao_rf.to_string(index=False))
posicao_escolhida = int(comparacao_rf["val_roc_auc"].idxmax())
rf, tempo_treino_rf = modelos_rf[posicao_escolhida]
parametros_rf_escolhidos = candidatos_rf[posicao_escolhida]
print("\nParâmetros escolhidos para o Random Forest:", parametros_rf_escolhidos)

corte_rf = escolher_corte(y_validacao, rf.predict_proba(X_validacao)[:, 1])
print(f"Corte do Random Forest escolhido na validação: {corte_rf:.2f}")

# Transforma a probabilidade do Random Forest em 0 ou 1 usando o corte dele
def prever_classes_rf(floresta, X):
    return (floresta.predict_proba(X)[:, 1] >= corte_rf).astype(int)

y_previsto_teste_rf = prever_classes_rf(rf, X_teste)
probabilidade_teste_rf = rf.predict_proba(X_teste)[:, 1]
resultados_teste["random_forest"] = pd.Series(avaliar(y_teste, y_previsto_teste_rf, probabilidade_teste_rf)).round(3)
print("\nMétricas no TESTE (linha de base x MLP x Random Forest):")
print(resultados_teste.to_string())

print("\nMétricas por classe do Random Forest no teste:")
print(classification_report(y_teste, y_previsto_teste_rf, labels=[0, 1], target_names=nomes_classes, digits=3, zero_division=0))

desenhar_matriz_confusao(y_previsto_teste_rf, "do Random Forest", "Greens", f"matriz_confusao_rf_{nome_conjunto}.png")

figura, eixo = plt.subplots(figsize=(6.5, 6))
for nome_modelo, probabilidade, cor in [(f"MLP {arquitetura_escolhida}", probabilidade_teste_mlp, "steelblue"),
                                        ("Random Forest", probabilidade_teste_rf, "seagreen")]:
    taxa_falso_positivo, taxa_verdadeiro_positivo, _ = roc_curve(y_teste, probabilidade)
    area = roc_auc_score(y_teste, probabilidade)
    eixo.plot(taxa_falso_positivo, taxa_verdadeiro_positivo, color=cor, label=f"{nome_modelo} (AUC = {area:.2f})")
eixo.plot([0, 1], [0, 1], color="gray", linestyle="--", label="sorteio (AUC = 0.50)")
eixo.set_xlabel("Taxa de falso positivo (baixo risco marcado como hotspot)")
eixo.set_ylabel("Recall do hotspot (taxa de verdadeiro positivo)")
eixo.set_title(f"Curva ROC no teste - {nome_conjunto}")
eixo.legend(loc="lower right")
plt.tight_layout()
caminho_roc = os.path.join(pasta_graficos, f"curva_roc_{nome_conjunto}.png")
plt.savefig(caminho_roc, dpi=120)
plt.close()
print("Gráfico salvo:", caminho_roc)

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
print("\nExemplo de centro_x e centro_y sem padronização (primeira linha do treino):", X_treino_sem[0, :2].round(0))

print(f"\nTreinando a MLP {arquitetura_escolhida} SEM padronização...")
mlp_sem = treinar_mlp(resultado_escolhido["neuronios"], X_treino_sem, X_validacao_sem)["modelo"]
f1_mlp_sem = avaliar(y_teste, prever_classes(mlp_sem, X_teste_sem))["f1"]

rf_sem = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=semente, n_jobs=-1, **parametros_rf_escolhidos)
rf_sem.fit(X_treino_sem, y_treino)
f1_rf_sem = avaliar(y_teste, prever_classes_rf(rf_sem, X_teste_sem))["f1"]
inicio = time.perf_counter()
prever_classes(mlp, X_teste)
tempo_previsao_mlp = time.perf_counter() - inicio
inicio = time.perf_counter()
prever_classes_rf(rf, X_teste)
tempo_previsao_rf = time.perf_counter() - inicio

comparacao_final = pd.DataFrame({
    coluna_mlp: {
        "accuracy_teste": resultados_teste.loc["accuracy", coluna_mlp],
        "f1_teste": resultados_teste.loc["f1", coluna_mlp],
        "f1_treino": avaliar(y_treino, prever_classes(mlp, X_treino))["f1"],
        "f1_teste_sem_padronizacao": f1_mlp_sem,
        "tempo_treino_s": tempo_treino_mlp,
        "tempo_previsao_teste_ms": tempo_previsao_mlp * 1000,
        "roc_auc_teste": resultados_teste.loc["roc_auc", coluna_mlp],
        "corte_decisao": corte_hotspot,
    },
    "random_forest": {
        "accuracy_teste": resultados_teste.loc["accuracy", "random_forest"],
        "f1_teste": resultados_teste.loc["f1", "random_forest"],
        "f1_treino": avaliar(y_treino, prever_classes_rf(rf, X_treino))["f1"],
        "f1_teste_sem_padronizacao": f1_rf_sem,
        "tempo_treino_s": tempo_treino_rf,
        "tempo_previsao_teste_ms": tempo_previsao_rf * 1000,
        "roc_auc_teste": resultados_teste.loc["roc_auc", "random_forest"],
        "corte_decisao": corte_rf,
    },
}).round(3)
comparacao_final.loc["diferenca_treino_teste"] = (comparacao_final.loc["f1_treino"] - comparacao_final.loc["f1_teste"]).round(3)

print(f"\nComparação final MLP x Random Forest ({nome_conjunto}):")
print(comparacao_final.to_string())

resultados_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_final.to_csv(os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_arquiteturas.to_csv(os.path.join(pasta_resultados, f"arquiteturas_mlp_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
comparacao_rf.to_csv(os.path.join(pasta_resultados, f"parametros_rf_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
print("\nTabelas salvas em", pasta_resultados)
