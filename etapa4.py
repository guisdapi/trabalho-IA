# 4 etapa IA - Regressão com MLP
# prevê QUANTOS atropelamentos um segmento de 500 m teve em um período de monitoramento (abril a março)
# cada linha do conjunto é um par segmento x período (tabela montada no passo 3 da etapa 2)
# o nome do conjunto vem do terminal:
#   python etapa4.py regressao_aleatoria
#   python etapa4.py regressao_espacial
#   python etapa4.py regressao_temporal
# depois das 3 divisões: python comparar_divisoes_regressao.py

# ===============================================================
# IMPORTS
# ===============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # esconde as mensagens internas do TensorFlow; precisa vir ANTES do import do tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # resultados iguais em toda execução (com a mesma semente), igual à etapa 3

import sys   # lê o nome do conjunto digitado no terminal (python etapa4.py regressao_espacial)
import time  # cronometra treino e previsão, para discutir o custo computacional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # abre os conjuntos prontos salvos na etapa 2 (data/etapa2/*.joblib)

# rede neural (MLP)
from tensorflow import keras  # a mesma biblioteca da etapa 3

# linha de base e modelos tradicionais
from sklearn.dummy import DummyRegressor  # linha de base: "modelo" que sempre chuta a média do treino
from sklearn.linear_model import LinearRegression  # modelo tradicional mais simples: uma soma ponderada dos atributos
from sklearn.ensemble import RandomForestRegressor  # modelo tradicional não linear: o mesmo da etapa 3, na versão de regressão

# métricas pedidas no enunciado (as mesmas funções avaliam todos os modelos, para a comparação ser justa)
from sklearn.metrics import (
    mean_absolute_error,  # MAE: erro médio, em atropelamentos ("erra em média 0,8 atropelamento por segmento")
    mean_squared_error,   # MSE: média dos erros ao quadrado (erros grandes pesam muito mais)
    r2_score,             # R²: quanto da variação do alvo o modelo explica (1 = perfeito, 0 = igual a chutar a média)
)


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_conjuntos = "data/etapa2"  # onde a etapa 2 salvou os conjuntos prontos (.joblib)
pasta_graficos = "graficosE4"
pasta_resultados = "resultadosE4"

semente = 42  # a mesma das etapas 2 e 3

# hiperparâmetros da MLP: o mesmo ponto de partida da etapa 3, para as duas redes serem comparáveis
# (a otimização com Optuna fica para a etapa 5)
neuronios_camadas_ocultas = [32, 16]  # 2 camadas ocultas, com 32 e 16 neurônios
arquiteturas_candidatas = [[16, 8], [32, 16], [64, 32]]  # rede menor, a inicial e uma maior, comparadas na validação
tolerancia_desempate = 0.01  # se uma rede menor ficar a menos de 1% do melhor RMSE da validação, preferimos a menor
taxa_aprendizado = 0.001     # valor padrão do Adam
tamanho_batch = 32           # linhas vistas antes de cada ajuste dos pesos
epocas_maximas = 1000        # limite; na prática a parada antecipada termina antes
                             # (maior que o da etapa 3: com 300, a rede de regressão ainda estava melhorando na última época)
paciencia = 20               # épocas seguidas sem melhora na validação antes de parar

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_resultados, exist_ok=True)


# ===============================================================
# FUNÇÕES
# ===============================================================
def avaliar_regressao(y_real, y_previsto):
    # as 4 métricas pedidas no enunciado; a mesma função avalia a linha de base, a MLP, a regressão linear e o Random Forest
    mse = mean_squared_error(y_real, y_previsto)
    return {
        "mae": mean_absolute_error(y_real, y_previsto),
        "mse": mse,
        "rmse": np.sqrt(mse), # raiz do MSE: volta para a unidade do alvo (atropelamentos), mas continua pesando mais os erros grandes
        "r2": r2_score(y_real, y_previsto),
    }


def prever(modelo, X):
    # devolve a previsão de qualquer modelo (rede do Keras ou modelo do scikit-learn) como uma lista de números
    # a rede é chamada direto (em vez de modelo.predict), o que evita o aviso de "retracing" do TensorFlow, igual à etapa 3
    if isinstance(modelo, keras.Model):
        previsao = np.asarray(modelo(X, training=False))[:, 0] # a saída tem 1 coluna; [:, 0] transforma em lista
    else:
        previsao = modelo.predict(X)
    # o alvo é uma contagem, então não existe "-0,3 atropelamento"
    # a saída linear da MLP e a regressão linear podem dar valores negativos; aqui eles viram 0
    # o corte vale para TODOS os modelos (o RF e a linha de base nunca dão negativo, então para eles nada muda)
    return np.maximum(previsao, 0)


def montar_mlp(numero_entradas, neuronios_camadas):
    # mesma estrutura da etapa 3; mudam só a camada de saída e a loss
    modelo = keras.Sequential(name="mlp_regressao")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada")) # 1 neurônio por atributo

    for numero_camada, neuronios in enumerate(neuronios_camadas, start=1):
        modelo.add(keras.layers.Dense(neuronios, activation="relu", name=f"oculta_{numero_camada}"))

    # saída: 1 neurônio com ativação LINEAR (devolve o número "cru", sem limite)
    # na classificação a saída era sigmoide/softmax porque precisava ser probabilidade; aqui precisa ser uma contagem
    modelo.add(keras.layers.Dense(1, activation="linear", name="saida"))

    # loss MSE: a rede tenta diminuir a média dos erros ao quadrado; é a loss padrão de regressão e combina com o RMSE avaliado no fim
    # (efeito colateral: os poucos segmentos com MUITOS atropelamentos pesam bastante, porque o erro deles é elevado ao quadrado)
    # métricas acompanhadas por época: MAE (erro médio em atropelamentos) e RMSE
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss="mse",
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return modelo


def treinar_mlp(neuronios_camadas, X_treino_rede, X_validacao_rede):
    # os X entram como parâmetro (e não como variável global, como na etapa 3): assim o passo 9 treina sem padronização sem trocar variáveis
    keras.utils.set_random_seed(semente) # a mesma semente para toda rede: as diferenças vêm da arquitetura, não do sorteio
    modelo = montar_mlp(X_treino_rede.shape[1], neuronios_camadas)

    # parada antecipada igual à etapa 3: olha a loss da validação e volta aos pesos da melhor época
    parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)

    inicio = time.perf_counter()
    historico = modelo.fit(
        X_treino_rede, y_treino_rede,
        validation_data=(X_validacao_rede, y_validacao_rede), # sem pesos: na regressão não há classes para balancear
        epochs=epocas_maximas,
        batch_size=tamanho_batch,
        callbacks=[parada_antecipada],
        shuffle=True,
        verbose=0, # troque para 2 se quiser ver uma linha por época
    )
    tempo = time.perf_counter() - inicio

    curvas = historico.history # loss, mae, rmse (treino) e val_loss, val_mae, val_rmse (validação), um valor por época
    epocas_treinadas = len(curvas["loss"])
    if epocas_treinadas == epocas_maximas:
        # a parada antecipada não chegou a parar: a validação ainda melhorava na última época
        print(f"  AVISO: a MLP {nome_arquitetura(neuronios_camadas)} usou todas as {epocas_maximas} épocas; considere aumentar epocas_maximas")
    return {
        "neuronios": neuronios_camadas,
        "modelo": modelo,
        "curvas": curvas,
        "epocas_treinadas": epocas_treinadas,
        "melhor_epoca": int(np.argmin(curvas["val_loss"])) + 1, # +1 porque as épocas começam em 1
        "tempo": tempo,
    }


def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas) # [32, 16] -> "32-16"


def desenhar_curvas(resultado):
    # à esquerda a loss (MSE), à direita o MAE; treino x validação em cada um
    # o RMSE não é desenhado porque é só a raiz da loss: a curva teria exatamente o mesmo formato
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


# ===============================================================
# PASSO 1: CARREGAR O CONJUNTO DA ETAPA 2
# ===============================================================
# sem nada depois do nome do script, usa a divisão aleatória
nome_conjunto = sys.argv[1] if len(sys.argv) > 1 else "regressao_aleatoria"
caminho_conjunto = os.path.join(pasta_conjuntos, nome_conjunto + ".joblib")
conjunto = joblib.load(caminho_conjunto)

X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_atributos_curtos = [nome.split("__")[1] for nome in conjunto["nomes_atributos"]] # "numericas__centro_x" -> "centro_x"

print("Conjunto carregado:", caminho_conjunto)
print(f"Treino {X_treino.shape} | validação {X_validacao.shape} | teste {X_teste.shape}  (linhas, colunas)")
print("Atributos de entrada:", nomes_atributos_curtos)

# o alvo em cada parte: se a média ou o % de zeros mudarem muito entre treino e teste, o teste fica mais difícil
# (acontece na divisão temporal se o período 5 tiver mais ou menos atropelamentos que os períodos 1 a 3)
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


# ===============================================================
# PASSO 2: LINHA DE BASE (CHUTAR SEMPRE A MÉDIA DO TREINO)
# ===============================================================
# a nota mínima da regressão: o DummyRegressor ignora os atributos e prevê a mesma média para todo segmento
# por definição, o R² dele fica perto de 0 (no teste fica levemente negativo, porque a média do teste não é exatamente a do treino)
# qualquer modelo que aprendeu alguma coisa sobre a localização precisa ter MAE e RMSE menores e R² maior que 0
linha_de_base = DummyRegressor(strategy="mean")
linha_de_base.fit(X_treino, y_treino)
print(f"\nA linha de base prevê sempre: {linha_de_base.predict(X_treino[:1])[0]:.2f} atropelamentos (média do treino)") # prevê uma linha qualquer só para mostrar o valor

resultados_linha_de_base = pd.DataFrame({
    "validacao": avaliar_regressao(y_validacao, prever(linha_de_base, X_validacao)),
    "teste": avaliar_regressao(y_teste, prever(linha_de_base, X_teste)),
}).round(3)
print("\nMétricas da linha de base:")
print(resultados_linha_de_base.to_string())


# ===============================================================
# PASSO 3: PREPARAR O ALVO PARA A REDE E MOSTRAR A ARQUITETURA
# ===============================================================
# a saída da rede é uma coluna, então o alvo também vira uma coluna de números decimais
# o alvo NÃO é padronizado: ele já é pequeno (poucos atropelamentos por segmento) e assim as métricas saem direto em atropelamentos
# diferente da etapa 3, não há pesos: na regressão não existem classes; os muitos zeros do alvo são discutidos no relatório
y_treino_rede = y_treino.reshape(-1, 1).astype("float32")
y_validacao_rede = y_validacao.reshape(-1, 1).astype("float32")

numero_entradas = X_treino.shape[1] # centro_x, centro_y e as colunas do trecho (One-Hot)
print()
montar_mlp(numero_entradas, neuronios_camadas_ocultas).summary() # só para o resumo; quem treina é a função treinar_mlp


# ===============================================================
# PASSO 4: TREINAR A MLP INICIAL E GERAR AS CURVAS DE TREINAMENTO
# ===============================================================
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


# ===============================================================
# PASSO 5: AJUSTE LEVE - COMPARAR ARQUITETURAS NA VALIDAÇÃO
# ===============================================================
# mesma ideia da etapa 3: uma rede menor e uma maior que a inicial; a escolha usa SÓ a validação
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
comparacao_arquiteturas = pd.DataFrame(linhas_comparacao).sort_values("parametros") # da menor para a maior rede
print("\nComparação das arquiteturas na validação:")
print(comparacao_arquiteturas.to_string(index=False))

# regra de escolha: MENOR RMSE na validação (na regressão, menor erro = melhor)
# desempate: se uma rede menor ficar a menos de 1% do melhor RMSE, ela é escolhida (mesmo raciocínio da etapa 3)
# a tolerância é relativa (1%) porque o RMSE depende da escala do alvo, ao contrário do F1, que vai sempre de 0 a 1
melhor_rmse = comparacao_arquiteturas["val_rmse"].min()
quase_empatadas = comparacao_arquiteturas[comparacao_arquiteturas["val_rmse"] <= melhor_rmse * (1 + tolerancia_desempate)]
arquitetura_escolhida = quase_empatadas.iloc[0]["arquitetura"] # tabela ordenada por parâmetros: a primeira é a menor

resultado_escolhido = resultados_arquiteturas[arquitetura_escolhida]
mlp = resultado_escolhido["modelo"]
tempo_treino_mlp = resultado_escolhido["tempo"]
coluna_mlp = "mlp_" + arquitetura_escolhida
print(f"\nArquitetura escolhida: {arquitetura_escolhida} (melhor RMSE na validação: {melhor_rmse:.3f}; tolerância de desempate: {tolerancia_desempate:.0%})")


# ===============================================================
# PASSO 6: AVALIAR A MLP ESCOLHIDA NO TESTE (UMA ÚNICA VEZ)
# ===============================================================
# todas as escolhas já foram feitas com treino e validação; o teste só MEDE
previsao_teste_mlp = prever(mlp, X_teste)

resultados_teste = pd.DataFrame({
    "linha_de_base": resultados_linha_de_base["teste"], # calculada no passo 2
    coluna_mlp: avaliar_regressao(y_teste, previsao_teste_mlp),
}).round(3)
print("\nMétricas no TESTE:")
print(resultados_teste.to_string())
# leitura: MAE = erro médio em atropelamentos; RMSE >= MAE, e quanto maior a diferença entre eles, mais o erro está concentrado
# em poucos segmentos com erro grande (normalmente os segmentos com muitos atropelamentos)


# ===============================================================
# PASSO 7: MODELOS TRADICIONAIS (REGRESSÃO LINEAR E RANDOM FOREST)
# ===============================================================
# os dois usam EXATAMENTE os mesmos X_treino, X_validacao e X_teste da MLP (o enunciado exige as mesmas divisões)

# --- regressão linear ---
# prevê o alvo como uma soma: peso_1 x centro_x + peso_2 x centro_y + ... + constante
# não tem hiperparâmetros para ajustar, então não usa a validação
# limitação esperada: ela só consegue representar "mais atropelamentos para o leste/norte" (uma rampa),
# e não manchas de hotspots espalhadas pelo mapa; serve para mostrar por que um modelo não linear é necessário
linear = LinearRegression()
inicio = time.perf_counter()
linear.fit(X_treino, y_treino)
tempo_treino_linear = time.perf_counter() - inicio
resultados_teste["regressao_linear"] = pd.Series(avaliar_regressao(y_teste, prever(linear, X_teste))).round(3)

# os pesos da regressão linear são a sua interpretação: como os numéricos estão padronizados, o peso diz
# quantos atropelamentos a previsão muda quando o atributo aumenta 1 desvio padrão
# (as colunas do trecho são 0/1 e somam sempre 1, então os pesos delas só fazem sentido comparados entre si)
print("\nPesos da regressão linear (atributos numéricos padronizados):")
print(pd.Series(linear.coef_, index=nomes_atributos_curtos).round(3).to_string())

# --- Random Forest: ajuste leve pela validação (mesma ideia da etapa 3) ---
# max_depth: quantas perguntas seguidas cada árvore pode fazer | min_samples_leaf: linhas mínimas em cada folha
# na regressão, a folha prevê a MÉDIA das suas linhas; folha com 1 linha "decora" a contagem exata daquele segmento e período,
# que tem muito de acaso (um ano com 3 atropelamentos e o seguinte com 0 no mesmo segmento), por isso testamos folhas maiores
candidatos_rf = [
    {"max_depth": None, "min_samples_leaf": 1},   # padrão do scikit-learn (árvores totalmente livres)
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
        "treino_rmse": round(avaliar_regressao(y_treino, prever(floresta, X_treino))["rmse"], 3), # mostra o quanto ele decorou o treino
        "val_mae": round(metricas_validacao["mae"], 3),
        "val_rmse": round(metricas_validacao["rmse"], 3),
        "val_r2": round(metricas_validacao["r2"], 3),
    })

comparacao_rf = pd.DataFrame(linhas_rf)
print("\nRandom Forest - comparação dos parâmetros na validação:")
print(comparacao_rf.to_string(index=False))
# treino_rmse bem menor que val_rmse = a floresta decorou o treino (overfitting)

posicao_escolhida = int(comparacao_rf["val_rmse"].idxmin()) # idxmin: a linha com o MENOR RMSE na validação
rf, tempo_treino_rf = modelos_rf[posicao_escolhida]
parametros_rf_escolhidos = candidatos_rf[posicao_escolhida]
print("\nParâmetros escolhidos para o Random Forest:", parametros_rf_escolhidos)

previsao_teste_rf = prever(rf, X_teste)
resultados_teste["random_forest"] = pd.Series(avaliar_regressao(y_teste, previsao_teste_rf)).round(3)
print("\nMétricas no TESTE (linha de base x MLP x regressão linear x Random Forest):")
print(resultados_teste.to_string())

# --- importância dos atributos no RF (mesma leitura e mesmo cuidado da etapa 3) ---
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


# ===============================================================
# PASSO 8: GRÁFICOS DE VALORES REAIS x PREDITOS E DE RESÍDUOS
# ===============================================================
# uma coluna por modelo (MLP, regressão linear, Random Forest), todos no mesmo teste
# linha de cima - real x predito: cada ponto é um segmento em um período; no modelo perfeito todos ficariam na diagonal tracejada
#   pontos ABAIXO da diagonal = o modelo previu menos do que aconteceu (subestimou); ACIMA = superestimou
#   como o alvo é uma contagem, os pontos formam "faixas" horizontais (0, 1, 2, ... atropelamentos reais)
# linha de baixo - resíduos: resíduo = real - predito, contra o valor predito
#   o ideal é uma nuvem sem formato em volta do 0; um "funil" que abre para a direita indica que o erro cresce com o valor previsto
previsao_teste_linear = prever(linear, X_teste)
modelos_para_grafico = [
    (f"MLP {arquitetura_escolhida}", previsao_teste_mlp, "steelblue"),
    ("Regressão linear", previsao_teste_linear, "darkorange"),
    ("Random Forest", previsao_teste_rf, "seagreen"),
]
limite = max(y_teste.max(), max(previsao.max() for _, previsao, _ in modelos_para_grafico)) * 1.05 # mesmo limite nos 3 gráficos de cima

figura, eixos = plt.subplots(2, 3, figsize=(16, 9.5))
for coluna, (nome_modelo, previsao, cor) in enumerate(modelos_para_grafico):
    metricas = avaliar_regressao(y_teste, previsao)

    eixo = eixos[0, coluna]
    eixo.scatter(y_teste, previsao, s=14, alpha=0.4, color=cor) # alpha deixa os pontos transparentes: onde há muitos sobrepostos, fica mais escuro
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


# ===============================================================
# PASSO 9: COMPARAÇÃO FINAL MLP x MODELOS TRADICIONAIS
# ===============================================================
# o enunciado pede vantagens e desvantagens das abordagens; aqui ficam os números para essa discussão:
# desempenho (já no passo 7), generalização (treino x teste), sensibilidade ao pré-processamento e custo computacional

# --- sensibilidade ao pré-processamento: treinar os três SEM a padronização ---
# mesma técnica da etapa 3: o StandardScaler da etapa 2 desfaz a padronização das colunas numéricas (as primeiras do X)
padronizador = conjunto["preprocessador"].named_transformers_["numericas"]
quantidade_numericas = len(padronizador.mean_) # centro_x e centro_y

def desfazer_padronizacao(X):
    X_original = X.copy() # copia para não estragar o X padronizado
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
# esperado: RF igual (árvores só comparam valores) e regressão linear igual (os pesos apenas mudam de escala para compensar)
# já a MLP piora: com coordenadas UTM na casa dos milhões, as contas da rede ficam desequilibradas e o ajuste dos pesos se perde

# --- custo computacional: tempo de previsão do teste ---
tempos_previsao = {}
for nome_coluna, modelo in [(coluna_mlp, mlp), ("regressao_linear", linear), ("random_forest", rf)]:
    inicio = time.perf_counter()
    prever(modelo, X_teste)
    tempos_previsao[nome_coluna] = time.perf_counter() - inicio

# --- tabela final ---
# rmse_treino x rmse_teste: quanto maior a diferença (teste - treino), mais o modelo decorou o treino
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
# na regressão o erro de teste fica ACIMA do de treino quando há overfitting, então a diferença é teste - treino
comparacao_final.loc["diferenca_teste_treino_rmse"] = (comparacao_final.loc["rmse_teste"] - comparacao_final.loc["rmse_treino"]).round(3)

print(f"\nComparação final MLP x modelos tradicionais ({nome_conjunto}):")
print(comparacao_final.to_string())

# --- salvar tabelas e previsões ---
resultados_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_final.to_csv(os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_arquiteturas.to_csv(os.path.join(pasta_resultados, f"arquiteturas_mlp_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
comparacao_rf.to_csv(os.path.join(pasta_resultados, f"parametros_rf_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")

# previsões de cada linha do teste, com o centro do segmento em metros: servem para mapear onde os modelos erram
# e para a discussão da etapa 7 sobre usar o modelo para apontar hotspots
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