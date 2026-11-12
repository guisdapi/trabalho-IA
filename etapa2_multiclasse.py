# 2 etapa IA - Pré-processamento do problema MULTICLASSE (classe taxonômica)
# cada linha é um atropelamento; o alvo é a classe do animal (amphibians, birds, mammals, reptiles)
# roda DEPOIS do etapa2_comum.py, que cria o período de monitoramento e a divisão espacial:
#   python etapa2_comum.py
#   python etapa2_multiclasse.py

# ===============================================================
# IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # salva os conjuntos prontos em arquivo, para a etapa 3 carregar
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
# semente e proporções precisam ser as mesmas nos três arquivos da etapa 2
pasta_graficos = "graficosE2"
pasta_saida = "data/etapa2"      # onde ficam os conjuntos prontos para a etapa 3
caminho_registros = os.path.join(pasta_saida, "registros_com_periodo.csv")  # salvo pelo etapa2_comum.py
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")              # salvo pelo etapa2_comum.py
semente = 42
proporcao_validacao = 0.15
proporcao_teste = 0.15
periodos_treino = [1, 2, 3]      # divisão temporal: abr/2010 a mar/2013
periodos_validacao = [4]         # abr/2013 a mar/2014
periodos_teste = [5]             # abr/2014 a mar/2015
tamanho_segmento = 500           # metros, usado na análise de vazamento (passo 4)

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_saida, exist_ok=True)


# ===============================================================
# FUNÇÕES
# ===============================================================
def dividir_aleatorio(tabela_problema, coluna_estratificar):
    # sorteia as linhas em 70/15/15; stratify mantém a proporção do alvo igual nas três partes
    indices = np.arange(len(tabela_problema))
    estratos = tabela_problema[coluna_estratificar].to_numpy()
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, stratify=estratos, random_state=semente)

    # a validação é 15% do total, mas o sorteio agora é feito sobre os 85% que sobraram: 0.15 / 0.85 = 0.176
    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, stratify=estratos[indices_resto], random_state=semente)

    particao = np.empty(len(tabela_problema), dtype=object) # cria uma coluna vazia e preenche cada parte
    particao[indices_treino] = "treino"
    particao[indices_validacao] = "validacao"
    particao[indices_teste] = "teste"
    return particao


def dividir_temporal(tabela_problema):
    # a parte depende só do período de monitoramento
    return np.select(
        [tabela_problema["periodo_monitoramento"].isin(periodos_treino),
         tabela_problema["periodo_monitoramento"].isin(periodos_validacao),
         tabela_problema["periodo_monitoramento"].isin(periodos_teste)],
        ["treino", "validacao", "teste"],
        default="fora", # nenhum registro cai aqui, porque todos estão nos períodos 1 a 5
    ) # np.select escolhe o valor da primeira condição verdadeira de cada linha


def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo):
    # linhas de cada parte e percentual de cada classe dentro dela: o ideal é ficar parecido nas três
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    percentual_alvo = (pd.crosstab(tabela_problema[coluna_particao], tabela_problema[coluna_alvo], normalize="index") * 100).round(1)
    print(resumo.join(percentual_alvo).to_string())


def distancias_teste_ao_treino(tabela_problema, coluna_particao, coluna_x, coluna_y):
    pontos_treino = tabela_problema[tabela_problema[coluna_particao] == "treino"][[coluna_x, coluna_y]].to_numpy()
    pontos_teste = tabela_problema[tabela_problema[coluna_particao] == "teste"][[coluna_x, coluna_y]].to_numpy()
    arvore = cKDTree(pontos_treino) # organiza os pontos de treino numa estrutura que acha o mais próximo rapidamente
    distancias, _ = arvore.query(pontos_teste) # para cada ponto de teste, devolve a distância até o ponto de treino mais próximo
    return distancias


def resumir_vazamento(tabela_problema, colunas_particao, coluna_x, coluna_y):
    linhas_resumo = []
    for coluna_particao in colunas_particao:
        distancias = distancias_teste_ao_treino(tabela_problema, coluna_particao, coluna_x, coluna_y)
        linhas_resumo.append({
            "divisão": coluna_particao.replace("particao_", ""),
            "distância mediana (m)": round(np.median(distancias)),
            "% no mesmo ponto (0 m)": round((distancias == 0).mean() * 100, 1),
            f"% a até {tamanho_segmento} m": round((distancias <= tamanho_segmento).mean() * 100, 1),
        })
    print(pd.DataFrame(linhas_resumo).to_string(index=False))


def montar_preprocessador(colunas_numericas, colunas_categoricas, colunas_binarias):
    return ColumnTransformer([
        ("numericas", StandardScaler(), colunas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colunas_categoricas), # handle_unknown="ignore": se aparecer um trecho que não estava no treino, as colunas ficam 0 em vez de dar erro
        ("binarias", "passthrough", colunas_binarias),
    ])


def preparar_e_salvar(tabela_problema, nome_problema, coluna_particao, coluna_alvo, colunas_numericas, colunas_categoricas, colunas_binarias, classes=None):
    colunas_entrada = colunas_numericas + colunas_categoricas + colunas_binarias
    preprocessador = montar_preprocessador(colunas_numericas, colunas_categoricas, colunas_binarias)

    treino = tabela_problema[tabela_problema[coluna_particao] == "treino"]
    validacao = tabela_problema[tabela_problema[coluna_particao] == "validacao"]
    teste = tabela_problema[tabela_problema[coluna_particao] == "teste"]

    conjunto = {
        "X_treino": preprocessador.fit_transform(treino[colunas_entrada]), # fit_transform: aprende média/desvio/categorias no treino e já transforma
        "X_validacao": preprocessador.transform(validacao[colunas_entrada]), # transform: só aplica o que foi aprendido no treino
        "X_teste": preprocessador.transform(teste[colunas_entrada]),
        "y_treino": treino[coluna_alvo].to_numpy(),
        "y_validacao": validacao[coluna_alvo].to_numpy(),
        "y_teste": teste[coluna_alvo].to_numpy(),
        "nomes_atributos": list(preprocessador.get_feature_names_out()), # nomes das colunas depois da codificação, úteis para interpretar o modelo
        "preprocessador": preprocessador, # guardado para transformar dados novos do mesmo jeito
        "classes": classes, # nomes das classes na ordem dos códigos
    }

    nome_divisao = coluna_particao.replace("particao_", "")
    nome_arquivo = f"{nome_problema}_{nome_divisao}.joblib"
    joblib.dump(conjunto, os.path.join(pasta_saida, nome_arquivo)) # salva tudo num arquivo só
    print(f"  {nome_arquivo}: treino {conjunto['X_treino'].shape} | validação {conjunto['X_validacao'].shape} | teste {conjunto['X_teste'].shape}")
    return conjunto


# ===============================================================
# PASSO 1: CARREGAR OS REGISTROS E MONTAR A BASE DO MULTICLASSE
# ===============================================================
# os registros já vêm com o período de monitoramento, calculado no etapa2_comum.py
# dtype={"segmento": str}: o código do segmento ("438_16547") precisa continuar como texto
tabela = pd.read_csv(caminho_registros, parse_dates=["data"], dtype={"segmento": str})
print("Registros carregados:", len(tabela))

# aqui cada linha continua sendo um atropelamento, e o alvo é a classe (birds, mammals, reptiles, amphibians)
# registros sem classe identificada não têm alvo, então saem da base
base_multiclasse = tabela[tabela["classe"] != "nao_identificado"].copy().reset_index(drop=True) # reset_index(drop=True) renumera as linhas de 0 em diante
print("\nRegistros removidos por classe nao_identificado:", len(tabela) - len(base_multiclasse))
print("Registros na base do multiclasse:", len(base_multiclasse))


# ===============================================================
# PASSO 2: ATRIBUTOS E ALVO
# ===============================================================
# --- atributos de mês ---
# codificação cíclica: o mês vira um ponto num círculo (seno e cosseno), assim dezembro (12) fica perto de janeiro (1)
# se o mês entrasse como número de 1 a 12, o modelo entenderia que dezembro e janeiro são os mais distantes
base_multiclasse["mes_numero"] = base_multiclasse["data"].dt.month
base_multiclasse["mes_sin"] = np.sin(2 * np.pi * base_multiclasse["mes_numero"] / 12)
base_multiclasse["mes_cos"] = np.cos(2 * np.pi * base_multiclasse["mes_numero"] / 12)

# estação chuvosa do Cerrado (DF): outubro a abril = 1, seca de maio a setembro = 0
base_multiclasse["estacao_chuvosa"] = base_multiclasse["mes_numero"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

# --- por que o dia da semana não entra ---
# os percursos só aconteceram de segunda a sexta (não há registro no fim de semana), então fim_de_semana seria sempre 0
# e o dia da semana mostra a agenda dos percursos, não quando o animal morreu (o percurso de segunda recolhe os atropelamentos do fim de semana)
print("\nRegistros por dia da semana do percurso (0 = segunda, 6 = domingo):")
print(base_multiclasse["data"].dt.dayofweek.value_counts().sort_index())

# --- por que o ano não entra ---
# na divisão temporal o teste é 2014-2015, anos que o modelo nunca viu no treino; o ano funcionaria como o próprio corte da divisão
# e não ajudaria a prever a classe de um atropelamento futuro

# --- alvo ---
# o LabelEncoder transforma o texto da classe em número (0, 1, 2, 3)
# pode ser usado aqui porque é a SAÍDA do modelo; nos atributos de entrada ele criaria uma ordem falsa entre categorias
codificador_classe = LabelEncoder()
base_multiclasse["classe_codigo"] = codificador_classe.fit_transform(base_multiclasse["classe"])
print("\nCódigo de cada classe:", dict(zip(codificador_classe.classes_, range(len(codificador_classe.classes_)))))

print("\nDistribuição do alvo do multiclasse:")
print(pd.DataFrame({
    "quantidade": base_multiclasse["classe"].value_counts(),
    "percentual": (base_multiclasse["classe"].value_counts(normalize=True) * 100).round(1),
}))

atributos_multiclasse = ["x", "y", "trecho_pela_localizacao", "mes_sin", "mes_cos", "estacao_chuvosa"]
print("\nAtributos de entrada do multiclasse:", atributos_multiclasse)

# gráfico: percentual de cada classe em cada mês, mostra se o mês ajuda a distinguir as classes
percentual_classe_mes = pd.crosstab(base_multiclasse["mes_numero"], base_multiclasse["classe"], normalize="index") * 100
percentual_classe_mes.plot(kind="bar", stacked=True, figsize=(10, 5), width=0.85)
plt.title("Composição das classes em cada mês")
plt.xlabel("Mês")
plt.ylabel("Percentual dos atropelamentos do mês")
plt.xticks(rotation=0)
plt.legend(title="classe", bbox_to_anchor=(1.01, 1), loc="upper left") # bbox_to_anchor coloca a legenda fora do gráfico, à direita
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "classes_por_mes.png"), dpi=120)
plt.close()


# ===============================================================
# PASSO 3: DIVISÃO EM TREINO, VALIDAÇÃO E TESTE
# ===============================================================
# treino = onde o modelo aprende | validação = onde escolhemos os hiperparâmetros | teste = avaliação final, só usado no fim
# - aleatória: sorteio dos registros, mantendo a proporção de cada classe nas três partes
# - espacial: cada registro recebe a parte do seu segmento, sorteada no etapa2_comum.py (as mesmas regiões dos outros problemas)
# - temporal: períodos 1 a 3 no treino, 4 na validação e 5 no teste
base_multiclasse["particao_aleatoria"] = dividir_aleatorio(base_multiclasse, "classe")

segmentos = pd.read_csv(caminho_segmentos, dtype={"segmento": str}, usecols=["segmento", "particao_espacial"])
base_multiclasse = base_multiclasse.merge(segmentos, on="segmento") # copia a parte espacial do segmento de cada registro

base_multiclasse["particao_temporal"] = dividir_temporal(base_multiclasse)

print("\nLinhas e % de cada classe em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(base_multiclasse, coluna_particao, "classe")


# ===============================================================
# PASSO 4: VAZAMENTO ESPACIAL (O TESTE ESTÁ PERTO DO TREINO?)
# ===============================================================
# se um ponto de teste está a poucos metros de um ponto de treino, o modelo pode acertar só por "lembrar" daquele lugar
# isso deixa a avaliação otimista: parece que o modelo generaliza, mas ele está sendo testado em lugares que já conhece
# aqui medimos, para cada atropelamento de teste, a distância até o atropelamento de treino mais próximo
print("\nDistância de cada registro de teste até o registro de treino mais próximo (local de cada atropelamento):")
resumir_vazamento(base_multiclasse, ["particao_aleatoria", "particao_espacial", "particao_temporal"], "x", "y")

# mapa das divisões: mostra visualmente onde ficou cada parte
figura, eixos = plt.subplots(1, 3, figsize=(18, 6))
cores = {"treino": "steelblue", "validacao": "orange", "teste": "crimson"}
for eixo, coluna_particao in zip(eixos, ["particao_aleatoria", "particao_espacial", "particao_temporal"]):
    for parte, cor in cores.items():
        pontos = base_multiclasse[base_multiclasse[coluna_particao] == parte]
        eixo.scatter(pontos["x"], pontos["y"], s=3, color=cor, label=parte)
    eixo.set_title("Divisão " + coluna_particao.replace("particao_", ""))
    eixo.set_xlabel("x (UTM, metros)")
    eixo.set_ylabel("y (UTM, metros)")
    eixo.set_aspect("equal") # mesma escala nos dois eixos, para não distorcer o mapa
    eixo.legend(markerscale=4)
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "divisoes_mapa.png"), dpi=120)
plt.close()


# ===============================================================
# PASSO 5: CODIFICAÇÃO, PADRONIZAÇÃO E SALVAR OS CONJUNTOS PRONTOS
# ===============================================================
# One-Hot no trecho: cria uma coluna 0/1 para cada trecho (trecho_esecae, trecho_pnb, ...)
#   o trecho é nominal (não existe trecho "maior" que outro); um número 0, 1, 2 criaria uma ordem falsa
# StandardScaler nos numéricos: deixa cada coluna com média 0 e desvio padrão 1
#   sem isso, x e y (na casa das centenas de milhares) dominariam o mes_sin (entre -1 e 1); a MLP é muito sensível a isso
# estacao_chuvosa já é 0/1, então passa sem alteração (passthrough)
#
# REGRA IMPORTANTE: o preprocessador é ajustado (fit) SÓ com o treino; validação e teste só são transformados (transform)
#   se a média e o desvio fossem calculados com o teste junto, informação do teste "vazaria" para o treino
print("\nConjuntos salvos em", pasta_saida, "(linhas, colunas):")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    conjunto_exemplo = preparar_e_salvar(base_multiclasse, "multiclasse", coluna_particao, "classe_codigo",
                      colunas_numericas=["x", "y", "mes_sin", "mes_cos"], colunas_categoricas=["trecho_pela_localizacao"], colunas_binarias=["estacao_chuvosa"],
                      classes=list(codificador_classe.classes_))

# conferência da padronização no último conjunto (temporal):
# no treino a média tem que ser ~0 e o desvio ~1; na validação e no teste ficam próximos, mas não exatos (o que é esperado)
print("\nConferência da padronização (multiclasse temporal):")
nomes = conjunto_exemplo["nomes_atributos"]
for parte in ["treino", "validacao", "teste"]:
    matriz = pd.DataFrame(conjunto_exemplo["X_" + parte], columns=nomes)
    print(f"\n  {parte} - média e desvio de cada coluna:")
    print(pd.DataFrame({"média": matriz.mean().round(2), "desvio": matriz.std().round(2)}).T.to_string())

# base legível (antes da codificação), com as colunas de partição, para consulta e para o relatório
base_multiclasse.to_csv(os.path.join(pasta_saida, "base_multiclasse.csv"), index=False, encoding="utf-8-sig")
print("\nBase legível salva: base_multiclasse.csv")

# como carregar na etapa 3:
#   conjunto = joblib.load("data/etapa2/multiclasse_aleatoria.joblib")
#   X_treino, y_treino = conjunto["X_treino"], conjunto["y_treino"]
