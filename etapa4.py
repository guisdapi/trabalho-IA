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

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


pasta_conjuntos = "data/etapa2"
pasta_graficos = "graficosE4"
pasta_resultados = "resultadosE4"

semente = 42

neuronios_camadas_ocultas = [32, 16]
arquiteturas_candidatas = [[16, 8], [32, 16], [64, 32]]
tolerancia_desempate = 0.01
taxa_aprendizado = 0.001
tamanho_batch = 32
epocas_maximas = 1000
paciencia = 20

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)





# Calcula as métricas de regressão (MAE, MSE, RMSE e R²)
def avaliar_regressao(y_real, y_previsto):
    mse = mean_squared_error(y_real, y_previsto)
    return {
        "mae": mean_absolute_error(y_real, y_previsto),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_real, y_previsto),
    }


# Faz a previsão com qualquer modelo e troca valores negativos por zero
def prever(modelo, X):
    if isinstance(modelo, keras.Model):
        previsao = np.asarray(modelo(X, training=False))[:, 0]
    else:
        previsao = modelo.predict(X)
    return np.maximum(previsao, 0)


# Monta a MLP com as camadas ocultas ReLU e um neurônio de saída linear
def montar_mlp(numero_entradas, neuronios_camadas):
    modelo = keras.Sequential(name="mlp_regressao")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada"))

    for numero_camada, neuronios in enumerate(neuronios_camadas, start=1):
        modelo.add(keras.layers.Dense(neuronios, activation="relu", name=f"oculta_{numero_camada}"))

    modelo.add(keras.layers.Dense(1, activation="linear", name="saida"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss="mse",
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return modelo


# Treina a MLP com parada antecipada, guardando as curvas e o tempo
def treinar_mlp(neuronios_camadas, X_treino_rede, X_validacao_rede):
    keras.utils.set_random_seed(semente)
    modelo = montar_mlp(X_treino_rede.shape[1], neuronios_camadas)

    parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)

    inicio = time.perf_counter()
    historico = modelo.fit(
        X_treino_rede, y_treino_rede,
        validation_data=(X_validacao_rede, y_validacao_rede),
        epochs=epocas_maximas,
        batch_size=tamanho_batch,
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


# Gera o nome da arquitetura a partir dos neurônios das camadas (ex.: 32-16)
def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas)


# Desenha as curvas de MSE e MAE no treino e na validação ao longo das épocas
def desenhar_curvas(resultado):
    curvas, melhor_epoca = resultado["curvas"], resultado["melhor_epoca"]
    epocas = range(1, resultado["epocas_treinadas"] + 1)
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    for eixo, medida, titulo in [(eixos[0], "loss", "Loss (MSE)"), (eixos[1], "mae", "MAE (atropelamentos)")]:
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


nome_conjunto = sys.argv[1] if len(sys.argv) > 1 else "regressao_aleatoria"
caminho_conjunto = os.path.join(pasta_conjuntos, nome_conjunto + ".joblib")
conjunto = joblib.load(caminho_conjunto)

X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_atributos_curtos = [nome.split("__")[1] for nome in conjunto["nomes_atributos"]]

print("Conjunto carregado:", caminho_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")
print("Atributos de entrada:", nomes_atributos_curtos)

print("\nAlvo (atropelamentos por segmento e período) em cada parte:")
print(pd.DataFrame({
    parte: {
        "linhas": len(y),
        "média": y.mean(),
        "desvio": y.std(),
        "máximo": y.max(),
        "% de zeros": (y == 0).mean() * 100,
    }
    for parte, y in [("treino", y_treino), ("validacao", y_validacao), ("teste", y_teste)]
}).round(2).to_string())


linha_de_base = DummyRegressor(strategy="mean")
linha_de_base.fit(X_treino, y_treino)
print(f"\nA linha de base prevê sempre: {linha_de_base.predict(X_treino[:1])[0]:.2f} atropelamentos (média do treino)")

resultados_linha_de_base = pd.DataFrame({
    "validacao": avaliar_regressao(y_validacao, prever(linha_de_base, X_validacao)),
    "teste": avaliar_regressao(y_teste, prever(linha_de_base, X_teste)),
}).round(3)
print("\nMétricas da linha de base:")
print(resultados_linha_de_base.to_string())


y_treino_rede = y_treino.reshape(-1, 1).astype("float32")
y_validacao_rede = y_validacao.reshape(-1, 1).astype("float32")

numero_entradas = X_treino.shape[1]
print()
montar_mlp(numero_entradas, neuronios_camadas_ocultas).summary()


resultado_inicial = treinar_mlp(neuronios_camadas_ocultas, X_treino, X_validacao)
curvas, melhor_epoca = resultado_inicial["curvas"], resultado_inicial["melhor_epoca"]

print(f"\nTreino da MLP {nome_arquitetura(neuronios_camadas_ocultas)}: {resultado_inicial['epocas_treinadas']} épocas em {resultado_inicial['tempo']:.1f} s")
print("Melhor época (menor loss de validação):", melhor_epoca, "-> a rede ficou com os pesos dessa época")
print("\nMedidas na melhor época:")
print(pd.DataFrame({
    "treino": [curvas[medida][melhor_epoca - 1] for medida in ["loss", "mae", "rmse"]],
    "validacao": [curvas["val_" + medida][melhor_epoca - 1] for medida in ["loss", "mae", "rmse"]],
}, index=["loss (mse)", "mae", "rmse"]).round(3).to_string())
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
    metricas_validacao = avaliar_regressao(y_validacao, prever(resultado["modelo"], X_validacao))
    linhas_comparacao.append({
        "arquitetura": arquitetura,
        "parametros": resultado["modelo"].count_params(),
        "epocas": resultado["epocas_treinadas"],
        "melhor_epoca": resultado["melhor_epoca"],
        "tempo_s": round(resultado["tempo"], 1),
        "val_mae": round(metricas_validacao["mae"], 3),
        "val_rmse": round(metricas_validacao["rmse"], 3),
        "val_r2": round(metricas_validacao["r2"], 3),
    })
comparacao_arquiteturas = pd.DataFrame(linhas_comparacao).sort_values("parametros")
print("\nComparação das arquiteturas na validação:")
print(comparacao_arquiteturas.to_string(index=False))

melhor_rmse = comparacao_arquiteturas["val_rmse"].min()
quase_empatadas = comparacao_arquiteturas[comparacao_arquiteturas["val_rmse"] <= melhor_rmse * (1 + tolerancia_desempate)]
arquitetura_escolhida = quase_empatadas.iloc[0]["arquitetura"]

resultado_escolhido = resultados_arquiteturas[arquitetura_escolhida]
mlp = resultado_escolhido["modelo"]
tempo_treino_mlp = resultado_escolhido["tempo"]
coluna_mlp = "mlp_" + arquitetura_escolhida
print(f"\nArquitetura escolhida: {arquitetura_escolhida} (melhor RMSE na validação: {melhor_rmse:.3f}; tolerância de desempate: {tolerancia_desempate:.0%})")


previsao_teste_mlp = prever(mlp, X_teste)

resultados_teste = pd.DataFrame({
    "linha_de_base": resultados_linha_de_base["teste"],
    coluna_mlp: avaliar_regressao(y_teste, previsao_teste_mlp),
}).round(3)
print("\nMétricas no TESTE:")
print(resultados_teste.to_string())
linear = LinearRegression()
inicio = time.perf_counter()
linear.fit(X_treino, y_treino)
tempo_treino_linear = time.perf_counter() - inicio
resultados_teste["regressao_linear"] = pd.Series(avaliar_regressao(y_teste, prever(linear, X_teste))).round(3)

print("\nPesos da regressão linear (atributos numéricos padronizados):")
print(pd.Series(linear.coef_, index=nomes_atributos_curtos).round(3).to_string())

candidatos_rf = [
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": None, "min_samples_leaf": 10},
    {"max_depth": 10, "min_samples_leaf": 5},
]

linhas_rf = []
modelos_rf = []
for parametros in candidatos_rf:
    floresta = RandomForestRegressor(n_estimators=300, random_state=semente, n_jobs=-1, **parametros)
    inicio = time.perf_counter()
    floresta.fit(X_treino, y_treino)
    tempo = time.perf_counter() - inicio
    metricas_validacao = avaliar_regressao(y_validacao, prever(floresta, X_validacao))
    modelos_rf.append((floresta, tempo))
    linhas_rf.append({
        "max_depth": "sem limite" if parametros["max_depth"] is None else parametros["max_depth"],
        "min_samples_leaf": parametros["min_samples_leaf"],
        "tempo_s": round(tempo, 2),
        "treino_rmse": round(avaliar_regressao(y_treino, prever(floresta, X_treino))["rmse"], 3),
        "val_mae": round(metricas_validacao["mae"], 3),
        "val_rmse": round(metricas_validacao["rmse"], 3),
        "val_r2": round(metricas_validacao["r2"], 3),
    })

comparacao_rf = pd.DataFrame(linhas_rf)
print("\nRandom Forest - comparação dos parâmetros na validação:")
print(comparacao_rf.to_string(index=False))
posicao_escolhida = int(comparacao_rf["val_rmse"].idxmin())
rf, tempo_treino_rf = modelos_rf[posicao_escolhida]
parametros_rf_escolhidos = candidatos_rf[posicao_escolhida]
print("\nParâmetros escolhidos para o Random Forest:", parametros_rf_escolhidos)

previsao_teste_rf = prever(rf, X_teste)
resultados_teste["random_forest"] = pd.Series(avaliar_regressao(y_teste, previsao_teste_rf)).round(3)
print("\nMétricas no TESTE (linha de base x MLP x regressão linear x Random Forest):")
print(resultados_teste.to_string())

importancias = pd.Series(rf.feature_importances_, index=nomes_atributos_curtos).sort_values()
print("\nImportância dos atributos no Random Forest:")
print(importancias.sort_values(ascending=False).round(3).to_string())

importancias.plot(kind="barh", color="seagreen", figsize=(8, 4))
plt.title(f"Importância dos atributos - Random Forest ({nome_conjunto})")
plt.xlabel("Importância")
plt.tight_layout()
caminho_importancia = os.path.join(pasta_graficos, f"importancia_atributos_rf_{nome_conjunto}.png")
plt.savefig(caminho_importancia, dpi=120)
plt.close()
print("Gráfico salvo:", caminho_importancia)


previsao_teste_linear = prever(linear, X_teste)
modelos_para_grafico = [
    (f"MLP {arquitetura_escolhida}", previsao_teste_mlp, "steelblue"),
    ("Regressão linear", previsao_teste_linear, "darkorange"),
    ("Random Forest", previsao_teste_rf, "seagreen"),
]
limite = max(y_teste.max(), max(previsao.max() for _, previsao, _ in modelos_para_grafico)) * 1.05

figura, eixos = plt.subplots(2, 3, figsize=(16, 9.5))
for coluna, (nome_modelo, previsao, cor) in enumerate(modelos_para_grafico):
    metricas = avaliar_regressao(y_teste, previsao)

    eixo = eixos[0, coluna]
    eixo.scatter(y_teste, previsao, s=14, alpha=0.4, color=cor)
    eixo.plot([0, limite], [0, limite], color="gray", linestyle="--", label="previsão perfeita")
    eixo.set_xlim(0, limite)
    eixo.set_ylim(0, limite)
    eixo.set_aspect("equal")
    eixo.set_title(f"{nome_modelo}\nR² = {metricas['r2']:.2f} | RMSE = {metricas['rmse']:.2f} | MAE = {metricas['mae']:.2f}")
    eixo.set_xlabel("Atropelamentos reais")
    eixo.set_ylabel("Atropelamentos preditos")
    eixo.legend(loc="upper left")

    eixo = eixos[1, coluna]
    residuos = y_teste - previsao
    eixo.scatter(previsao, residuos, s=14, alpha=0.4, color=cor)
    eixo.axhline(0, color="gray", linestyle="--")
    eixo.set_title(f"Resíduos - {nome_modelo}")
    eixo.set_xlabel("Atropelamentos preditos")
    eixo.set_ylabel("Resíduo (real - predito)")

figura.suptitle(f"Valores reais x preditos e resíduos no teste - {nome_conjunto}")
plt.tight_layout()
caminho_real_predito = os.path.join(pasta_graficos, f"real_predito_residuos_{nome_conjunto}.png")
plt.savefig(caminho_real_predito, dpi=120)
plt.close()
print("Gráfico salvo:", caminho_real_predito)


padronizador = conjunto["preprocessador"].named_transformers_["numericas"]
quantidade_numericas = len(padronizador.mean_)

# Volta os atributos numéricos para a escala original, para testar os modelos sem padronização
def desfazer_padronizacao(X):
    X_original = X.copy()
    X_original[:, :quantidade_numericas] = padronizador.inverse_transform(X[:, :quantidade_numericas])
    return X_original

X_treino_sem, X_validacao_sem, X_teste_sem = desfazer_padronizacao(X_treino), desfazer_padronizacao(X_validacao), desfazer_padronizacao(X_teste)

print(f"\nTreinando a MLP {arquitetura_escolhida} SEM padronização...")
mlp_sem = treinar_mlp(resultado_escolhido["neuronios"], X_treino_sem, X_validacao_sem)["modelo"]
rmse_mlp_sem = avaliar_regressao(y_teste, prever(mlp_sem, X_teste_sem))["rmse"]

rf_sem = RandomForestRegressor(n_estimators=300, random_state=semente, n_jobs=-1, **parametros_rf_escolhidos).fit(X_treino_sem, y_treino)
rmse_rf_sem = avaliar_regressao(y_teste, prever(rf_sem, X_teste_sem))["rmse"]

linear_sem = LinearRegression().fit(X_treino_sem, y_treino)
rmse_linear_sem = avaliar_regressao(y_teste, prever(linear_sem, X_teste_sem))["rmse"]
tempos_previsao = {}
for nome_coluna, modelo in [(coluna_mlp, mlp), ("regressao_linear", linear), ("random_forest", rf)]:
    inicio = time.perf_counter()
    prever(modelo, X_teste)
    tempos_previsao[nome_coluna] = time.perf_counter() - inicio



# Monta a linha da tabela final com as métricas de um modelo
def linha_final(nome_coluna, modelo, rmse_sem_padronizacao, tempo_treino):
    return {
        "mae_teste": resultados_teste.loc["mae", nome_coluna],
        "rmse_teste": resultados_teste.loc["rmse", nome_coluna],
        "r2_teste": resultados_teste.loc["r2", nome_coluna],
        "rmse_treino": avaliar_regressao(y_treino, prever(modelo, X_treino))["rmse"],
        "rmse_teste_sem_padronizacao": rmse_sem_padronizacao,
        "tempo_treino_s": tempo_treino,
        "tempo_previsao_teste_ms": tempos_previsao[nome_coluna] * 1000,
    }

comparacao_final = pd.DataFrame({
    coluna_mlp: linha_final(coluna_mlp, mlp, rmse_mlp_sem, tempo_treino_mlp),
    "regressao_linear": linha_final("regressao_linear", linear, rmse_linear_sem, tempo_treino_linear),
    "random_forest": linha_final("random_forest", rf, rmse_rf_sem, tempo_treino_rf),
}).round(3)
comparacao_final.loc["diferenca_teste_treino_rmse"] = (comparacao_final.loc["rmse_teste"] - comparacao_final.loc["rmse_treino"]).round(3)

print(f"\nComparação final MLP x modelos tradicionais ({nome_conjunto}):")
print(comparacao_final.to_string())

resultados_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_final.to_csv(os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_arquiteturas.to_csv(os.path.join(pasta_resultados, f"arquiteturas_mlp_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
comparacao_rf.to_csv(os.path.join(pasta_resultados, f"parametros_rf_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame({
    "centro_x": X_teste_sem[:, 0],
    "centro_y": X_teste_sem[:, 1],
    "atropelamentos_reais": y_teste,
    "previsto_linha_de_base": prever(linha_de_base, X_teste),
    "previsto_" + coluna_mlp: previsao_teste_mlp,
    "previsto_regressao_linear": previsao_teste_linear,
    "previsto_random_forest": previsao_teste_rf,
}).round(3).to_csv(os.path.join(pasta_resultados, f"previsoes_teste_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
print("\nTabelas salvas em", pasta_resultados)