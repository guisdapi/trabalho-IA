# 3 etapa IA - Classificação MULTICLASSE com MLP (classe taxonômica: amphibians, birds, mammals, reptiles)
# a divisão dos dados vem do terminal:
#   python etapa3_multiclasse.py aleatoria
#   python etapa3_multiclasse.py espacial
#   python etapa3_multiclasse.py temporal
# depois das 3 divisões: python comparar_divisoes.py multiclasse

# ===============================================================
# IMPORTS
# ===============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # esconde as mensagens internas do TensorFlow (avisos de CPU/GPU) que poluem o terminal; precisa vir ANTES do import do tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # desliga a otimização oneDNN: os resultados ficam iguais em toda execução (com a mesma semente), o que é importante para o grupo comparar resultados; com a base pequena, a perda de velocidade é insignificante

import sys   # lê a divisão digitada no terminal (python etapa3_multiclasse.py espacial)
import time  # cronometra o treino de cada modelo, para discutir o custo computacional
import numpy as np  # transforma as probabilidades da rede na classe prevista (a maior probabilidade)
import pandas as pd  # monta as tabelas de resultado (MLP x Random Forest em cada divisão)
import matplotlib.pyplot as plt  # gráficos: curvas de loss/F1 por época e matrizes de confusão
import joblib  # abre os conjuntos prontos salvos na etapa 2 (data/etapa2/*.joblib)

# rede neural (MLP)
from tensorflow import keras  # a parte do TensorFlow usada para montar, treinar e parar a rede (camadas, otimizador, parada antecipada); o TensorFlow é o "motor" que faz as contas por baixo

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
    confusion_matrix,        # tabela de quais classes se confundem com quais
    ConfusionMatrixDisplay,  # desenha a matriz de confusão
    classification_report,   # precision, recall e F1 de cada classe de uma vez
)


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_conjuntos = "data/etapa2"  # onde a etapa 2 salvou os conjuntos prontos (.joblib)
pasta_graficos = "graficosE3"
pasta_resultados = "resultadosE3"
divisoes_possiveis = ["aleatoria", "espacial", "temporal"]

semente = 42  # a mesma da etapa 2: fixa os sorteios (pesos iniciais da rede, ordem dos batches), para todo o grupo obter o mesmo resultado

# hiperparâmetros da MLP (ponto de partida; a otimização com Optuna fica para a etapa 5)
neuronios_camadas_ocultas = [32, 16]  # 2 camadas ocultas, com 32 e 16 neurônios; cada número da lista é uma camada
arquiteturas_candidatas = [[16, 8], [32, 16], [64, 32]]  # passo 6: rede menor, a inicial e uma maior, comparadas na validação
tolerancia_desempate = 0.01          # passo 6: se outra rede ficar a menos de 0,01 do melhor F1 macro, preferimos a que tem menos parâmetros
taxa_aprendizado = 0.001              # tamanho do passo de cada ajuste dos pesos (valor padrão do Adam)
tamanho_batch = 32                    # quantas linhas a rede vê antes de cada ajuste dos pesos (usado no treino, passo 5)
epocas_maximas = 300                  # limite de épocas; na prática o treino para antes, pela parada antecipada (passo 5)
paciencia = 20                        # quantas épocas seguidas sem melhora na validação a parada antecipada tolera antes de parar

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script, igual às etapas 1 e 2
os.makedirs(pasta_graficos, exist_ok=True) # cria a pasta dos gráficos da etapa 3 se ela não existir
os.makedirs(pasta_resultados, exist_ok=True)


# ===============================================================
# FUNÇÕES
# ===============================================================
def avaliar(y_real, y_previsto):
    # calcula as métricas do multiclasse pedidas no enunciado; a mesma função avalia a linha de base, a MLP e o Random Forest
    # average="macro": calcula a métrica de cada classe e tira a média simples, então anfíbios (classe rara) pesam igual a aves
    # zero_division=0: se o modelo nunca prevê uma classe, a precision dela fica 0 em vez de dar aviso (acontece com a linha de base)
    return {
        "accuracy": accuracy_score(y_real, y_previsto),
        "precision_macro": precision_score(y_real, y_previsto, average="macro", zero_division=0),
        "recall_macro": recall_score(y_real, y_previsto, average="macro", zero_division=0),
        "f1_macro": f1_score(y_real, y_previsto, average="macro", zero_division=0),
    }


def montar_mlp(numero_entradas, neuronios_camadas):
    # monta e configura a rede
    modelo = keras.Sequential(name="mlp")
    modelo.add(keras.Input(shape=(numero_entradas,), name="entrada")) # camada de entrada: 1 neurônio por atributo

    # camadas ocultas: Dense = cada neurônio se liga a todos os neurônios da camada anterior
    for numero_camada, neuronios in enumerate(neuronios_camadas, start=1):
        modelo.add(keras.layers.Dense(neuronios, activation="relu", name=f"oculta_{numero_camada}"))

    # saída: 1 neurônio por classe com softmax, que devolve uma probabilidade para cada classe (somando 1)
    modelo.add(keras.layers.Dense(numero_classes, activation="softmax", name="saida"))

    # compile junta as peças do treino: quem ajusta os pesos (otimizador), o que ele tenta diminuir (loss)
    # e o que é acompanhado a cada época (métricas, que viram as curvas do relatório)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="macro", name="f1_macro")], # F1 macro, a métrica principal do multiclasse
    )
    return modelo


def treinar_mlp(neuronios_camadas, X_treino_rede, X_validacao_rede):
    # monta uma rede com os neurônios pedidos e treina com os X recebidos
    # os X entram como parâmetro: assim o passo 9 treina sem padronização sem precisar trocar variáveis
    # usa as variáveis criadas nos passos 3 e 4 (y_treino_rede, pesos_classes, pesos_linhas_validacao)
    keras.utils.set_random_seed(semente) # a mesma semente para toda rede: as diferenças entre elas vêm da arquitetura, não do sorteio
    modelo = montar_mlp(X_treino_rede.shape[1], neuronios_camadas)

    # parada antecipada: a cada época olha a loss da validação; se ela passar "paciencia" épocas sem melhorar, o treino para
    # restore_best_weights=True: no fim, a rede volta aos pesos da época com a menor loss de validação (e não aos da última época)
    # monitoramos a loss (e não o F1) porque ela varia de forma mais suave e mede também a confiança das respostas
    parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)

    inicio = time.perf_counter() # marca o horário de início, para medir o custo computacional
    historico = modelo.fit(
        X_treino_rede, y_treino_rede,
        validation_data=(X_validacao_rede, y_validacao_rede, pesos_linhas_validacao), # avaliada no fim de cada época; NÃO é usada para ajustar os pesos
        epochs=epocas_maximas,
        batch_size=tamanho_batch,
        class_weight=pesos_classes,
        callbacks=[parada_antecipada],
        shuffle=True, # embaralha a ordem das linhas a cada época, para os batches não serem sempre os mesmos
        verbose=0,    # 0 = não imprime nada durante o treino; troque para 2 se quiser ver uma linha por época
    )
    tempo = time.perf_counter() - inicio

    # historico.history é um dicionário com uma lista por medida, um valor por época:
    # loss, accuracy, f1_macro (treino) e val_loss, val_accuracy, val_f1_macro (validação)
    curvas = historico.history
    epocas_treinadas = len(curvas["loss"])
    if epocas_treinadas == epocas_maximas:
        # a parada antecipada não chegou a parar: a validação ainda melhorava na última época
        print(f"  AVISO: a MLP {nome_arquitetura(neuronios_camadas)} usou todas as {epocas_maximas} épocas; considere aumentar epocas_maximas")
    return {
        "neuronios": neuronios_camadas,
        "modelo": modelo,
        "curvas": curvas,
        "epocas_treinadas": epocas_treinadas,
        "melhor_epoca": int(np.argmin(curvas["val_loss"])) + 1, # posição da menor loss de validação; +1 porque as épocas começam em 1
        "tempo": tempo,
    }


def prever_classes(modelo, X):
    # a rede devolve 4 probabilidades por linha (uma por classe); argmax devolve a posição (código) da maior
    # chama a rede direto (em vez de modelo.predict), o que evita um aviso de "retracing" do TensorFlow quando várias redes são avaliadas seguidas
    probabilidades = np.asarray(modelo(X, training=False)) # training=False: modo de previsão
    return probabilidades.argmax(axis=1)


def nome_arquitetura(neuronios_camadas):
    return "-".join(str(neuronios) for neuronios in neuronios_camadas) # [32, 16] -> "32-16", usado em tabelas e nomes de arquivo


def desenhar_curvas(resultado):
    # à esquerda a loss, à direita o F1 macro (a métrica principal); treino x validação em cada um
    # a linha tracejada marca a melhor época; depois dela, o treino continuou só para a parada antecipada confirmar que não havia melhora
    curvas, melhor_epoca = resultado["curvas"], resultado["melhor_epoca"]
    epocas = range(1, resultado["epocas_treinadas"] + 1)
    figura, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    for eixo, medida, titulo in [(eixos[0], "loss", "Loss (entropia cruzada com pesos)"), (eixos[1], "f1_macro", "F1 macro")]:
        eixo.plot(epocas, curvas[medida], color="steelblue", label="treino")
        eixo.plot(epocas, curvas["val_" + medida], color="orange", label="validação")
        eixo.axvline(melhor_epoca, color="gray", linestyle="--", label=f"melhor época ({melhor_epoca})") # axvline desenha uma linha vertical
        eixo.set_title(titulo)
        eixo.set_xlabel("Época")
        eixo.legend()
    arquitetura = nome_arquitetura(resultado["neuronios"])
    figura.suptitle(f"Curvas de treinamento da MLP {arquitetura} - {nome_conjunto}") # título geral acima dos dois gráficos
    plt.tight_layout()
    caminho = os.path.join(pasta_graficos, f"curvas_mlp_{nome_conjunto}_{arquitetura}.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    print("Gráfico salvo:", caminho)


def desenhar_matriz_confusao(y_previsto, nome_modelo, mapa_de_cores, nome_arquivo):
    # duas versões da matriz lado a lado:
    # esquerda: quantidade de animais | direita: % de cada classe REAL (cada linha soma 100%; a diagonal é o recall de cada classe)
    # a versão em % é a mais justa para comparar classes de tamanhos diferentes (10 anfíbios x 300 aves)
    figura, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_previsto, labels=list(range(numero_classes)), display_labels=nomes_classes,
        cmap=mapa_de_cores, ax=eixos[0], colorbar=False,
    )
    eixos[0].set_title("Quantidade")
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_previsto, labels=list(range(numero_classes)), display_labels=nomes_classes,
        normalize="true", values_format=".0%", cmap=mapa_de_cores, ax=eixos[1], colorbar=False, # normalize="true": divide cada linha pelo total da classe real
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


# ===============================================================
# PASSO 1: CARREGAR O CONJUNTO DA ETAPA 2
# ===============================================================
# o .joblib é um dicionário com X_treino, y_treino, ..., nomes_atributos, classes e preprocessador, salvo pelo etapa2_multiclasse.py
# sys.argv é a lista do que foi digitado no comando: [0] = nome do script, [1] = o que vem depois dele
# sem nada depois do nome do script, usa a divisão aleatória
divisao = sys.argv[1].replace("multiclasse_", "") if len(sys.argv) > 1 else "aleatoria" # aceita "espacial" ou "multiclasse_espacial"
if divisao not in divisoes_possiveis:
    raise SystemExit(f"Divisão desconhecida: {divisao} (use {', '.join(divisoes_possiveis)})")
nome_conjunto = "multiclasse_" + divisao # também vai no nome dos gráficos e tabelas, para as divisões não se misturarem
caminho_conjunto = os.path.join(pasta_conjuntos, nome_conjunto + ".joblib")
conjunto = joblib.load(caminho_conjunto)

# separa as peças em variáveis com nomes curtos, para usar nos próximos passos
X_treino, X_validacao, X_teste = conjunto["X_treino"], conjunto["X_validacao"], conjunto["X_teste"]
y_treino, y_validacao, y_teste = conjunto["y_treino"], conjunto["y_validacao"], conjunto["y_teste"]
nomes_classes = conjunto["classes"]
numero_classes = len(nomes_classes)  # 4 classes -> 4 saídas na rede

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
    "validacao": avaliar(y_validacao, linha_de_base.predict(X_validacao)),
    "teste": avaliar(y_teste, linha_de_base.predict(X_teste)),
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
# PASSO 4: PREPARAR OS DADOS PARA A REDE E MOSTRAR A ARQUITETURA
# ===============================================================
numero_entradas = X_treino.shape[1]  # 8 atributos -> 8 entradas

# a saída da rede são 4 probabilidades, então o alvo de treino e de validação vira 4 colunas com um único 1 (One-Hot)
# exemplo: código 2 (mammals) -> [0, 0, 1, 0]
# o y_teste continua em código (0 a 3), porque as métricas do scikit-learn comparam códigos
y_treino_rede = keras.utils.to_categorical(y_treino, numero_classes)
y_validacao_rede = keras.utils.to_categorical(y_validacao, numero_classes)

# o class_weight do Keras só é aplicado na loss do TREINO; a loss da validação seria calculada sem pesos
# aí as duas curvas ficariam em escalas diferentes e não daria para compará-las no gráfico
# por isso a validação recebe os mesmos pesos, linha a linha (sample_weight): cada linha ganha o peso da sua classe
# assim as duas losses medem a mesma coisa, e a parada antecipada escolhe a época pensando em todas as classes, inclusive as raras
# (as métricas accuracy e F1 não usam esses pesos: elas continuam sendo a accuracy e o F1 de verdade)
# .get(codigo, 1.0): se uma classe aparecer na validação mas não no treino (pode acontecer nas divisões espacial/temporal), ela recebe peso 1 em vez de dar erro
pesos_linhas_validacao = np.array([pesos_classes.get(codigo, 1.0) for codigo in y_validacao])

# mostra as camadas da arquitetura inicial e quantos pesos (parâmetros) ela tem para ajustar
# (essa rede é só para o resumo; quem monta e treina de verdade é a função treinar_mlp, nos passos 5 e 6)
print()
montar_mlp(numero_entradas, neuronios_camadas_ocultas).summary()


# ===============================================================
# PASSO 5: TREINAR A MLP INICIAL E GERAR AS CURVAS DE TREINAMENTO
# ===============================================================
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


# ===============================================================
# PASSO 6: AJUSTE LEVE - COMPARAR ARQUITETURAS NA VALIDAÇÃO
# ===============================================================
# testa uma rede menor e uma maior que a inicial; a escolha usa SÓ a validação (o teste continua guardado para o passo 7)
# a busca completa de hiperparâmetros fica para a etapa 5 (Optuna); aqui o objetivo é justificar o número de neurônios
resultados_arquiteturas = {nome_arquitetura(neuronios_camadas_ocultas): resultado_inicial} # a inicial já foi treinada no passo 5
for neuronios_camadas in arquiteturas_candidatas:
    arquitetura = nome_arquitetura(neuronios_camadas)
    if arquitetura not in resultados_arquiteturas: # não treina de novo a que já existe
        print(f"\nTreinando a MLP {arquitetura}...")
        resultados_arquiteturas[arquitetura] = treinar_mlp(neuronios_camadas, X_treino, X_validacao)
        desenhar_curvas(resultados_arquiteturas[arquitetura])

# tabela de comparação: o F1 macro é calculado com o scikit-learn, igual será no teste e no Random Forest
linhas_comparacao = []
for arquitetura, resultado in resultados_arquiteturas.items():
    metricas_validacao = avaliar(y_validacao, prever_classes(resultado["modelo"], X_validacao))
    linhas_comparacao.append({
        "arquitetura": arquitetura,
        "parametros": resultado["modelo"].count_params(), # quantos pesos a rede ajusta
        "epocas": resultado["epocas_treinadas"],
        "melhor_epoca": resultado["melhor_epoca"],
        "tempo_s": round(resultado["tempo"], 1),
        "val_loss": round(min(resultado["curvas"]["val_loss"]), 3),
        "val_accuracy": round(metricas_validacao["accuracy"], 3),
        "val_f1_macro": round(metricas_validacao["f1_macro"], 3),
    })
comparacao_arquiteturas = pd.DataFrame(linhas_comparacao).sort_values("parametros") # da menor para a maior rede
print("\nComparação das arquiteturas na validação:")
print(comparacao_arquiteturas.to_string(index=False))

# regra de escolha: maior F1 macro na validação
# desempate: se uma rede menor ficar a menos de "tolerancia_desempate" do melhor F1, ela é escolhida
# (diferença tão pequena pode ser só sorte do sorteio; com desempenho igual, a rede mais simples é preferível e decora menos)
melhor_f1 = comparacao_arquiteturas["val_f1_macro"].max()
quase_empatadas = comparacao_arquiteturas[comparacao_arquiteturas["val_f1_macro"] >= melhor_f1 - tolerancia_desempate]
arquitetura_escolhida = quase_empatadas.iloc[0]["arquitetura"] # a tabela está ordenada por parâmetros, então a primeira é a menor

resultado_escolhido = resultados_arquiteturas[arquitetura_escolhida]
mlp = resultado_escolhido["modelo"]  # a rede que segue para a avaliação no teste (passo 7)
tempo_treino_mlp = resultado_escolhido["tempo"]
coluna_mlp = "mlp_" + arquitetura_escolhida
print(f"\nArquitetura escolhida: {arquitetura_escolhida} (melhor F1 macro na validação: {melhor_f1:.3f}; tolerância de desempate: {tolerancia_desempate})")


# ===============================================================
# PASSO 7: AVALIAR A MLP ESCOLHIDA NO TESTE (UMA ÚNICA VEZ)
# ===============================================================
# todas as escolhas (pesos, arquitetura, época) já foram feitas com treino e validação
# a partir daqui nada mais é ajustado: o teste só MEDE o desempenho em dados que a rede nunca viu
# se mudássemos a rede depois de olhar o teste, ele viraria uma segunda validação e o resultado ficaria otimista
y_previsto_teste = prever_classes(mlp, X_teste)

# --- métricas gerais: MLP x linha de base ---
resultados_teste = pd.DataFrame({
    "linha_de_base": resultados_linha_de_base["teste"], # calculada no passo 2
    coluna_mlp: avaliar(y_teste, y_previsto_teste),
}).round(3)
print("\nMétricas no TESTE:")
print(resultados_teste.to_string())
# a comparação principal é pelo F1 macro: a accuracy da linha de base já parece razoável sem ela aprender nada (passo 2)

# --- métricas por classe ---
# precision: das vezes que a rede disse "essa classe", quantas estavam certas
# recall: dos animais que realmente eram dessa classe, quantos a rede encontrou
# support: quantos animais de cada classe existem no teste (classes com support pequeno têm métricas instáveis)
# macro avg = média simples das classes | weighted avg = média ponderada pelo support (favorece as classes grandes)
print("\nMétricas por classe no teste:")
print(classification_report(y_teste, y_previsto_teste, labels=list(range(numero_classes)), target_names=nomes_classes, digits=3, zero_division=0))

# --- matriz de confusão ---
# linha = classe real, coluna = classe prevista; a diagonal são os acertos, fora dela são as confusões
matriz = confusion_matrix(y_teste, y_previsto_teste, labels=list(range(numero_classes)))
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(pd.DataFrame(matriz, index=["real_" + nome for nome in nomes_classes], columns=["prev_" + nome for nome in nomes_classes]).to_string())
print()
desenhar_matriz_confusao(y_previsto_teste, f"da MLP {arquitetura_escolhida}", "Blues", f"matriz_confusao_mlp_{nome_conjunto}.png")


# ===============================================================
# PASSO 8: RANDOM FOREST (MODELO TRADICIONAL PARA COMPARAÇÃO)
# ===============================================================
# o Random Forest usa EXATAMENTE os mesmos X_treino, X_validacao e X_teste da MLP (o enunciado exige as mesmas divisões)
# os dados estão padronizados por causa da MLP, mas isso não muda nada para as árvores:
# uma árvore só pergunta "x é maior que tal valor?", e a resposta é a mesma com ou sem padronização
#
# como funciona: o RF cria muitas árvores de decisão, cada uma treinada com um sorteio diferente das linhas e dos atributos;
# na previsão, cada árvore vota em uma classe e vence a mais votada
#
# class_weight="balanced": o mesmo tratamento do desbalanceamento usado na MLP (passo 3), para a comparação ser justa
# n_jobs=-1: usa todos os núcleos do processador (o TensorFlow também usa), então os tempos de treino ficam comparáveis

# --- ajuste leve pela validação (mesma ideia do passo 6) ---
# max_depth: quantas perguntas seguidas cada árvore pode fazer (None = sem limite, até separar tudo)
# min_samples_leaf: quantos animais, no mínimo, cada "folha" (resposta final da árvore) precisa ter
# os dois controlam o quanto a árvore pode decorar o treino: árvore funda e folha pequena decoram mais
candidatos_rf = [
    {"max_depth": None, "min_samples_leaf": 1},  # padrão do scikit-learn (árvores totalmente livres)
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": 15, "min_samples_leaf": 1},
    {"max_depth": 15, "min_samples_leaf": 5},
]

linhas_rf = []
modelos_rf = []
for parametros in candidatos_rf:
    floresta = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=semente, n_jobs=-1, **parametros) # **parametros "abre" o dicionário como argumentos
    inicio = time.perf_counter()
    floresta.fit(X_treino, y_treino)
    tempo = time.perf_counter() - inicio
    metricas_validacao = avaliar(y_validacao, floresta.predict(X_validacao))
    modelos_rf.append((floresta, tempo))
    linhas_rf.append({
        "max_depth": "sem limite" if parametros["max_depth"] is None else parametros["max_depth"],
        "min_samples_leaf": parametros["min_samples_leaf"],
        "tempo_s": round(tempo, 2),
        "treino_f1_macro": round(avaliar(y_treino, floresta.predict(X_treino))["f1_macro"], 3), # mostra o quanto ele decorou o treino
        "val_accuracy": round(metricas_validacao["accuracy"], 3),
        "val_f1_macro": round(metricas_validacao["f1_macro"], 3),
    })

comparacao_rf = pd.DataFrame(linhas_rf)
print("\nRandom Forest - comparação dos parâmetros na validação:")
print(comparacao_rf.to_string(index=False))
# treino_f1_macro perto de 1 com val_f1_macro bem menor = a floresta decorou o treino (overfitting)

posicao_escolhida = int(comparacao_rf["val_f1_macro"].idxmax()) # idxmax devolve a linha com o maior F1 macro na validação
rf, tempo_treino_rf = modelos_rf[posicao_escolhida]
parametros_rf_escolhidos = candidatos_rf[posicao_escolhida]
print("\nParâmetros escolhidos para o Random Forest:", parametros_rf_escolhidos)

# --- avaliação no teste (uma única vez, igual à MLP) ---
y_previsto_teste_rf = rf.predict(X_teste) # o RF já devolve o código da classe, sem precisar converter probabilidades
resultados_teste["random_forest"] = pd.Series(avaliar(y_teste, y_previsto_teste_rf)).round(3) # nova coluna na tabela do passo 7
print("\nMétricas no TESTE (linha de base x MLP x Random Forest):")
print(resultados_teste.to_string())

print("\nMétricas por classe do Random Forest no teste:")
print(classification_report(y_teste, y_previsto_teste_rf, labels=list(range(numero_classes)), target_names=nomes_classes, digits=3, zero_division=0))

# matriz de confusão do RF, no mesmo formato da MLP (passo 7), para comparar as duas lado a lado no relatório
desenhar_matriz_confusao(y_previsto_teste_rf, "do Random Forest", "Greens", f"matriz_confusao_rf_{nome_conjunto}.png")

# --- importância dos atributos (a vantagem de interpretação do RF) ---
# mede o quanto cada atributo ajudou as árvores a separar as classes; a soma de todos dá 1
# cuidado na leitura: esse cálculo tende a favorecer atributos com muitos valores diferentes (x e y)
# em relação a atributos 0/1 (trecho, estação chuvosa); vale citar essa limitação no relatório
nomes_atributos_curtos = [nome.split("__")[1] for nome in conjunto["nomes_atributos"]] # "numericas__x" -> "x"
importancias = pd.Series(rf.feature_importances_, index=nomes_atributos_curtos).sort_values()
print("\nImportância dos atributos no Random Forest:")
print(importancias.sort_values(ascending=False).round(3).to_string())

importancias.plot(kind="barh", color="seagreen", figsize=(8, 4.5)) # barh = barras horizontais, mais fácil de ler nomes longos
plt.title(f"Importância dos atributos - Random Forest ({nome_conjunto})")
plt.xlabel("Importância")
plt.tight_layout()
caminho_importancia = os.path.join(pasta_graficos, f"importancia_atributos_rf_{nome_conjunto}.png")
plt.savefig(caminho_importancia, dpi=120)
plt.close()
print("Gráfico salvo:", caminho_importancia)


# ===============================================================
# PASSO 9: COMPARAÇÃO FINAL MLP x RANDOM FOREST
# ===============================================================
# o enunciado pede para discutir: desempenho, capacidade de generalização, sensibilidade ao pré-processamento,
# custo computacional e facilidade de interpretação; este passo junta os números que faltavam numa tabela só
# (a interpretação já foi coberta no passo 8, com a importância dos atributos)

# --- sensibilidade ao pré-processamento: treinar os dois modelos SEM a padronização ---
# os conjuntos da etapa 2 já vêm padronizados, então desfazemos a padronização só das colunas numéricas
# o StandardScaler ajustado na etapa 2 está guardado dentro do preprocessador, e inverse_transform faz a conta ao contrário
# as colunas numéricas são as primeiras do X (ordem do ColumnTransformer: numéricas, categóricas, binárias)
padronizador = conjunto["preprocessador"].named_transformers_["numericas"] # pega o StandardScaler pelo nome dado na etapa 2
quantidade_numericas = len(padronizador.mean_) # uma média guardada por coluna numérica (x, y, mes_sin, mes_cos)

def desfazer_padronizacao(X):
    X_original = X.copy() # copia para não estragar o X padronizado, que ainda é usado
    X_original[:, :quantidade_numericas] = padronizador.inverse_transform(X[:, :quantidade_numericas]) # [:, :4] = todas as linhas, 4 primeiras colunas
    return X_original

X_treino_sem, X_validacao_sem, X_teste_sem = desfazer_padronizacao(X_treino), desfazer_padronizacao(X_validacao), desfazer_padronizacao(X_teste)
print("\nExemplo de x e y sem padronização (primeira linha do treino):", X_treino_sem[0, :2].round(0)) # devem voltar às coordenadas UTM em metros

# MLP sem padronização: mesma arquitetura e mesmo treino, só muda a entrada
print(f"\nTreinando a MLP {arquitetura_escolhida} SEM padronização...")
mlp_sem = treinar_mlp(resultado_escolhido["neuronios"], X_treino_sem, X_validacao_sem)["modelo"]
f1_mlp_sem = avaliar(y_teste, prever_classes(mlp_sem, X_teste_sem))["f1_macro"]

# Random Forest sem padronização: mesmos parâmetros escolhidos no passo 8
rf_sem = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=semente, n_jobs=-1, **parametros_rf_escolhidos)
rf_sem.fit(X_treino_sem, y_treino)
f1_rf_sem = avaliar(y_teste, rf_sem.predict(X_teste_sem))["f1_macro"]
# esperado: o RF dá praticamente o mesmo resultado (as árvores só comparam valores, a escala não importa)
# e a MLP piora bastante (x e y na casa das centenas de milhares dominam as contas e atrapalham o ajuste dos pesos)

# --- custo computacional: tempo de previsão (o tempo de treino já foi medido nos passos 6 e 8) ---
inicio = time.perf_counter()
prever_classes(mlp, X_teste)
tempo_previsao_mlp = time.perf_counter() - inicio
inicio = time.perf_counter()
rf.predict(X_teste)
tempo_previsao_rf = time.perf_counter() - inicio

# --- tabela final ---
# f1_macro_treino x f1_macro_teste: quanto maior a diferença, mais o modelo decorou o treino (pior generalização)
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

# salva as tabelas do teste e da comparação, para o relatório não depender da saída do terminal
resultados_teste.to_csv(os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_final.to_csv(os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv"), encoding="utf-8-sig")
comparacao_arquiteturas.to_csv(os.path.join(pasta_resultados, f"arquiteturas_mlp_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
comparacao_rf.to_csv(os.path.join(pasta_resultados, f"parametros_rf_{nome_conjunto}.csv"), index=False, encoding="utf-8-sig")
print("\nTabelas salvas em", pasta_resultados)
