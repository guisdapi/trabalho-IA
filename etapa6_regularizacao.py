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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score


nome_conjunto = "multiclasse_aleatoria"
pasta_conjuntos = "data/etapa2"
pasta_resultados_e5 = "resultadosE5"
pasta_graficos = "graficosE6"
pasta_resultados = "resultadosE6"

semente = 42
epocas_maximas = 300
paciencia = 20
neuronios_rede_grande = [256, 128]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)


                                                                 
         
                                                                 
def montar_mlp(neuronios_camadas, dropout, l2):
    modelo = keras.Sequential(name="mlp_regularizacao")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada"))
    for numero_camada, neuronios in enumerate(neuronios_camadas, start=1):
        modelo.add(keras.layers.Dense(
            neuronios, activation=melhores["ativacao"],
            kernel_regularizer=keras.regularizers.l2(l2) if l2 > 0 else None, name=f"oculta_{numero_camada}",
        ))
        if dropout > 0:
            modelo.add(keras.layers.Dropout(dropout, name=f"dropout_{numero_camada}"))
    modelo.add(keras.layers.Dense(numero_classes, activation="softmax", name="saida"))

    taxa = melhores["taxa_aprendizado"]
    otimizador = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=taxa),
        "rmsprop": lambda: keras.optimizers.RMSprop(learning_rate=taxa),
        "sgd": lambda: keras.optimizers.SGD(learning_rate=taxa, momentum=0.9),
    }[melhores["otimizador"]]()

    modelo.compile(
        optimizer=otimizador,
        loss="categorical_crossentropy",
        metrics=[keras.metrics.F1Score(average="macro", name="f1_macro")],
        weighted_metrics=[keras.metrics.CategoricalCrossentropy(name="entropia")],
    )
    return modelo


def treinar(configuracao):
    keras.utils.set_random_seed(semente)
    modelo = montar_mlp(configuracao["neuronios"], configuracao["dropout"], configuracao["l2"])

    callbacks = []
    if configuracao["parada_antecipada"]:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True))

    inicio = time.perf_counter()
    historico = modelo.fit(
        X_treino, y_treino_rede,
        validation_data=(X_validacao, y_validacao_rede, pesos_linhas_validacao),
        epochs=epocas_maximas,
        batch_size=melhores["tamanho_batch"],
        class_weight=pesos_classes,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )
    tempo = time.perf_counter() - inicio
    curvas = historico.history
    return modelo, curvas, tempo


def prever_classes(modelo, X):
    return np.asarray(modelo(X, training=False)).argmax(axis=1)


def f1_macro(y_real, y_previsto):
    return f1_score(y_real, y_previsto, average="macro", zero_division=0)


if len(sys.argv) > 1:
    epocas_maximas = int(sys.argv[1])

conjunto = joblib.load(os.path.join(pasta_conjuntos, nome_conjunto + ".joblib"))
X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]
numero_classes = len(nomes_classes)
numero_entradas = X_treino.shape[1]
print("Conjunto carregado:", nome_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")

caminho_melhores = os.path.join(pasta_resultados_e5, f"melhores_hiperparametros_{nome_conjunto}.csv")
if not os.path.exists(caminho_melhores):
    raise SystemExit("Faltam os resultados da etapa 5 -> rode antes: python etapa5_optuna.py aleatoria")
valores = pd.read_csv(caminho_melhores, index_col=0)["valor"]
melhores = {
    "camadas": int(valores["camadas"]),
    "neuronios_primeira_camada": int(valores["neuronios_primeira_camada"]),
    "ativacao": str(valores["ativacao"]),
    "otimizador": str(valores["otimizador"]),
    "taxa_aprendizado": float(valores["taxa_aprendizado"]),
    "tamanho_batch": int(valores["tamanho_batch"]),
    "dropout": float(valores["dropout"]),
    "l2": float(valores["l2"]),
}
neuronios_otimizada = [max(melhores["neuronios_primeira_camada"] // (2 ** posicao), 4) for posicao in range(melhores["camadas"])]
nome_otimizada = "-".join(str(neuronios) for neuronios in neuronios_otimizada)
nome_grande = "-".join(str(neuronios) for neuronios in neuronios_rede_grande)
print("\nMelhores hiperparâmetros da etapa 5:")
for nome, valor in melhores.items():
    print(f"  {nome}: {valor:.4g}" if isinstance(valor, float) else f"  {nome}: {valor}")


classes_do_treino = np.unique(y_treino)
pesos = compute_class_weight(class_weight="balanced", classes=classes_do_treino, y=y_treino)
pesos_classes = dict(zip(classes_do_treino.tolist(), pesos.tolist()))
pesos_linhas_validacao = np.array([pesos_classes.get(codigo, 1.0) for codigo in y_validacao])
y_treino_rede = keras.utils.to_categorical(y_treino, numero_classes)
y_validacao_rede = keras.utils.to_categorical(y_validacao, numero_classes)


dropout_otimo, l2_otimo = melhores["dropout"], melhores["l2"]
configuracoes = [
    {"nome": f"{nome_otimizada}_sem_regularizacao", "rotulo": f"{nome_otimizada}: sem regularização",
     "neuronios": neuronios_otimizada, "dropout": 0.0, "l2": 0.0, "parada_antecipada": False},
    {"nome": f"{nome_otimizada}_parada_antecipada", "rotulo": f"{nome_otimizada}: só parada antecipada",
     "neuronios": neuronios_otimizada, "dropout": 0.0, "l2": 0.0, "parada_antecipada": True},
    {"nome": f"{nome_otimizada}_dropout_l2", "rotulo": f"{nome_otimizada}: só Dropout + L2",
     "neuronios": neuronios_otimizada, "dropout": dropout_otimo, "l2": l2_otimo, "parada_antecipada": False},
    {"nome": f"{nome_otimizada}_completa", "rotulo": f"{nome_otimizada}: Dropout + L2 + parada (Optuna)",
     "neuronios": neuronios_otimizada, "dropout": dropout_otimo, "l2": l2_otimo, "parada_antecipada": True},
    {"nome": f"{nome_grande}_sem_regularizacao", "rotulo": f"{nome_grande}: sem regularização",
     "neuronios": neuronios_rede_grande, "dropout": 0.0, "l2": 0.0, "parada_antecipada": False},
    {"nome": f"{nome_grande}_completa", "rotulo": f"{nome_grande}: Dropout + L2 + parada",
     "neuronios": neuronios_rede_grande, "dropout": dropout_otimo, "l2": l2_otimo, "parada_antecipada": True},
]

print(f"\nConfigurações comparadas (até {epocas_maximas} épocas; sem parada antecipada a rede treina todas):")
print(pd.DataFrame(configuracoes)[["rotulo", "dropout", "l2", "parada_antecipada"]].round(5).to_string(index=False))


linhas = []
curvas_por_configuracao = {}
for configuracao in configuracoes:
    print(f"\nTreinando: {configuracao['rotulo']}...")
    modelo, curvas, tempo = treinar(configuracao)
    curvas_por_configuracao[configuracao["nome"]] = curvas

    entropia_validacao = np.array(curvas["val_entropia"])
    epoca_menor_erro = int(entropia_validacao.argmin()) + 1
    y_previsto_teste = prever_classes(modelo, X_teste)
    linha = {
        "configuracao": configuracao["rotulo"],
        "dropout": configuracao["dropout"],
        "l2": configuracao["l2"],
        "parada_antecipada": configuracao["parada_antecipada"],
        "parametros": modelo.count_params(),
        "epocas": len(curvas["loss"]),
        "epoca_menor_erro_validacao": epoca_menor_erro,
        "tempo_s": tempo,
        "f1_macro_treino": f1_macro(y_treino, prever_classes(modelo, X_treino)),
        "f1_macro_validacao": f1_macro(y_validacao, prever_classes(modelo, X_validacao)),
        "f1_macro_teste": f1_macro(y_teste, y_previsto_teste),
        "accuracy_teste": accuracy_score(y_teste, y_previsto_teste),
        "entropia_treino_final": curvas["entropia"][-1],
        "entropia_validacao_minima": entropia_validacao.min(),
        "entropia_validacao_final": entropia_validacao[-1],
    }
    linha["diferenca_treino_validacao"] = linha["f1_macro_treino"] - linha["f1_macro_validacao"]
    linha["subida_entropia_validacao"] = linha["entropia_validacao_final"] - linha["entropia_validacao_minima"]
    linhas.append(linha)
    print(f"  {linha['epocas']} épocas em {tempo:.1f} s | F1 macro: treino {linha['f1_macro_treino']:.3f}, "
          f"validação {linha['f1_macro_validacao']:.3f}, teste {linha['f1_macro_teste']:.3f}")
    keras.backend.clear_session()

comparacao = pd.DataFrame(linhas)
colunas_principais = ["configuracao", "epocas", "f1_macro_treino", "f1_macro_validacao", "f1_macro_teste",
                      "diferenca_treino_validacao", "subida_entropia_validacao"]
print("\nComparação das configurações:")
print(comparacao[colunas_principais].round(3).to_string(index=False))


figura, eixos = plt.subplots(2, len(configuracoes), figsize=(4.2 * len(configuracoes), 8), sharey="row")
limite_superior_entropia = max(max(curvas["entropia"][1:] + curvas["val_entropia"][1:]) for curvas in curvas_por_configuracao.values())
limite_inferior_entropia = min(min(curvas["entropia"] + curvas["val_entropia"]) for curvas in curvas_por_configuracao.values())
for coluna, configuracao in enumerate(configuracoes):
    curvas = curvas_por_configuracao[configuracao["nome"]]
    epocas = range(1, len(curvas["loss"]) + 1)
    epoca_menor_erro = int(np.argmin(curvas["val_entropia"])) + 1
    for linha_grafico, (medida, titulo_medida) in enumerate([("entropia", "Erro (entropia cruzada)"), ("f1_macro", "F1 macro")]):
        eixo = eixos[linha_grafico, coluna]
        eixo.plot(epocas, curvas[medida], color="steelblue", label="treino")
        eixo.plot(epocas, curvas["val_" + medida], color="orange", label="validação")
        eixo.axvline(epoca_menor_erro, color="gray", linestyle="--", linewidth=1, label=f"menor erro na validação ({epoca_menor_erro})")
        eixo.set_xlabel("Época")
        if linha_grafico == 0:
            eixo.set_title(configuracao["rotulo"], fontsize=10)
            eixo.set_ylim(limite_inferior_entropia * 0.98, limite_superior_entropia * 1.02)
        if coluna == 0:
            eixo.set_ylabel(titulo_medida)
        eixo.legend(fontsize=7)
figura.suptitle(f"Curvas de treino x validação com e sem regularização - {nome_conjunto}\n"
                "(com Dropout, a curva de treino é medida com neurônios desligados e por isso fica abaixo da validação)")
plt.tight_layout()
caminho_curvas = os.path.join(pasta_graficos, f"curvas_regularizacao_{nome_conjunto}.png")
plt.savefig(caminho_curvas, dpi=120, bbox_inches="tight")
plt.close()
print("\nGráfico salvo:", caminho_curvas)

grafico = comparacao.set_index("configuracao")[["f1_macro_treino", "f1_macro_validacao", "f1_macro_teste"]]
grafico.columns = ["treino", "validação", "teste"]
eixo = grafico.plot(kind="bar", figsize=(13, 5.5), color=["steelblue", "orange", "crimson"], width=0.8)
for barras in eixo.containers:
    eixo.bar_label(barras, fmt="%.2f", fontsize=7)
plt.title(f"F1 macro no treino, na validação e no teste - {nome_conjunto}")
plt.ylabel("F1 macro (medido sem dropout)")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
eixo.set_ylim(0, grafico.to_numpy().max() * 1.15)
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
caminho_barras = os.path.join(pasta_graficos, f"f1_treino_validacao_teste_{nome_conjunto}.png")
plt.savefig(caminho_barras, dpi=120, bbox_inches="tight")
plt.close()
print("Gráfico salvo:", caminho_barras)


sem_regularizacao = comparacao.iloc[0]
rede_grande_sem = comparacao.iloc[4]
melhor = comparacao.loc[comparacao["f1_macro_validacao"].idxmax()]
print("\nResumo:")
print(f"  {sem_regularizacao['configuracao']}: diferença treino-validação = {sem_regularizacao['diferenca_treino_validacao']:.3f}, "
      f"subida do erro na validação = {sem_regularizacao['subida_entropia_validacao']:.3f}")
print(f"  {rede_grande_sem['configuracao']}: diferença treino-validação = {rede_grande_sem['diferenca_treino_validacao']:.3f}, "
      f"subida do erro na validação = {rede_grande_sem['subida_entropia_validacao']:.3f}")
print(f"  Melhor F1 macro na validação: {melhor['configuracao']} ({melhor['f1_macro_validacao']:.3f}; teste {melhor['f1_macro_teste']:.3f})")

comparacao.round(4).to_csv(os.path.join(pasta_resultados, f"comparacao_regularizacao_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
print("\nTabela salva em", pasta_resultados)
