# 1 etapa IA

# ===============================================================
# IMPORTS
# ===============================================================
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency  # o teste de qui-quadrado é usado para verificar se há associação entre duas variáveis categóricas. Ele compara a distribuição observada com a distribuição esperada se as variáveis fossem independentes. O resultado inclui o valor do qui-quadrado, o valor-p, os graus de liberdade e a tabela de frequências esperadas.


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
caminho_arquivo = "data/rodofauna.xlsx"
tamanho_segmento = 500  # metros


# ===============================================================
# FUNÇÕES
# ===============================================================
def descobrir_trecho_pela_localizacao(linha):
    if linha["y"] < 8245000:
        return "jbb_fal_ibge"
    if linha["x"] > 205000:
        return "esecae"
    return "pnb"


# ===============================================================
# PASSO 1: CARREGAR A PLANILHA E OLHAR O QUE ELA CONTÉM
# ===============================================================
tabela = pd.read_excel(caminho_arquivo) # lê o xlsx e guarda na variavel tabela

quantidade_linhas, quantidade_colunas = tabela.shape # só devolve as linha e colunas
print("Quantidade de registros:", quantidade_linhas)
print("Quantidade de atributos:", quantidade_colunas)

print("\nTipos de cada atributo:")
print(tabela.dtypes) # devolve o tipo de cada coluna

# print("\nPrimeiras 3 linhas:")
# print(tabela.head(3).T)  # apenas teste para ver se a leitura ta funcionando


# ===============================================================
# PASSO 1.1: REMOVER REGISTROS DUPLICADOS
# ===============================================================
# decidi verificar as duplicatas e remover antes da filtragem e analise estatistica, percebi que com as duplicatas os valores ficavam distorcidos demais, em questão de quantidade
colunas_para_comparar = [coluna for coluna in tabela.columns if coluna != "OBJECTID"] # monta uma lista de todas colunas, menos objectid
quantidade_duplicados = tabela.duplicated(subset=colunas_para_comparar).sum() # marca as linhas que repetem, uma anterior, e sum conta quantas são
print("\nLinhas duplicadas:", quantidade_duplicados)

tabela = tabela.drop_duplicates(subset=colunas_para_comparar) # mantém somente a primeira copia de cada linha duplicada, e remove as demais
print("Registros depois de remover duplicados:", len(tabela))


# ===============================================================
# PASSO 2: FICAR SÓ COM A FAUNA SILVESTRE
# ===============================================================
print("\nValores da coluna silvestre_:")
print(tabela["silvestre_"].value_counts()) # value.counts conta quantas vezes aparece cada valor na coluna. Serve pra verificar se não há variações como "Silvestres" etc

tabela_silvestre = tabela[tabela["silvestre_"] == "silvestre"] #tabela[silvestre_"]... compara cada linha e gera uma lista de verdadeiro/falso, e tabela[...] mantém e deolve apenas as linhas verdadeiras.

print("\nRegistros antes do filtro:", len(tabela)) # com doméstico
print("Registros depois do filtro:", len(tabela_silvestre)) # somente silvestres


# ===============================================================
# PASSO 3: VALORES AUSENTES
# ===============================================================
total_celulas_vazia = tabela_silvestre.isna().sum().sum() # isna marca verdadeiro as celular realmente vazias, o 1 sum soma por coluna e o 2 soma tudo.
print("\nCélulas realmente vazias:", total_celulas_vazia)

quantidade_nao_identificado = (tabela_silvestre == "nao_identificado").sum()
percentual_nao_identificado = quantidade_nao_identificado / len(tabela_silvestre) * 100 # só transforma o valor em percentual

resumo_ausentes = pd.DataFrame({ # junta qntd + percentual numa tabela e . round(1) limita uma casa decimal
    "quantidade": quantidade_nao_identificado,
    "percentual": percentual_nao_identificado.round(1),
})

resumo_ausentes = resumo_ausentes[resumo_ausentes["quantidade"] > 0] # filtra apenas as colunas que tem algum valor "nao_identificado"
resumo_ausentes = resumo_ausentes.sort_values("quantidade", ascending=False) # sort_value oderna de maior para menor e o ascending indica a direção, nesse caso de decrescente

print("\nValores 'nao_identificado' por coluna:")
print(resumo_ausentes)


# ===============================================================
# PASSO 4: DISTRIBUIÇÃO DAS CLASSES
# ===============================================================
# Distribuição das classes / é o alvo do algoritmo multiclasse, é preciso ver a distribuição e desbalanceamento
quantidade_por_classe = tabela_silvestre["classe"].value_counts() # value conta os registros de cada classe
percentual_por_classe = (quantidade_por_classe / len(tabela_silvestre) * 100).round(1)

print("\nRegistros por classe:")
print(pd.DataFrame({"quantidade": quantidade_por_classe, "percentual": percentual_por_classe}))

quantidade_por_classe.plot(kind="bar", color="steelblue") #plot(kind) desenha o grafico de barra
plt.title("Atropelamentos de fauna silvestre por classe") # serve pra colocar titulos
plt.xlabel("Classe") # titulo
plt.ylabel("Quantidade de registros") # titulo
plt.xticks(rotation=0) # deixa os nomes em horizontal
plt.tight_layout() # ajusta as margens do grafico para não cortar nada
plt.savefig("graficosE1/distribuicao_classes.png", dpi=120) # salva a img em um arquivo
plt.close() # fecha a imagem, antes era plt.show() mas como o show trava a execução do programa, troquei por close, que fecha a imagem e continua a execução do programa


# ===============================================================
# PASSO 5: ESTATÍSTICAS DESCRITIVAS
# ===============================================================
# funções para a parte de estatistica descritiva
colunas_numericas = ["x", "y", "ano"]
estatisticas = tabela_silvestre[colunas_numericas].describe().round(1) # .describe calcula de uma vez só a contagem, media, desvio padrao, minimo, maximo e quartis. .round(1) limita a 1 casa decimal

print("\nEstatísticas descritivas:")
print(estatisticas)


# ===============================================================
# PASSO 6: DISTRIBUIÇÃO TEMPORAL
# ===============================================================
# distribuição temporal
tabela_silvestre = tabela_silvestre.copy() # cria uma copia, pra não correr o risco de sobrescrever a tabela original, isso pq nessa etapa adicionamos colunas novas de data, mes, e dia da semana
tabela_silvestre["data"] = pd.to_datetime(tabela_silvestre["data_dt"], format="%d/%m/%y") # converte o texto em data de verdade
tabela_silvestre["mes_numero"] = tabela_silvestre["data"].dt.month # extrai o mes da data e cria uma nova coluna
tabela_silvestre["dia_semana_numero"] = tabela_silvestre["data"].dt.dayofweek # estrai o dia da semana, seg=0 até domingo=6

registros_por_ano = tabela_silvestre["ano"].value_counts().sort_index() # essas linhas serve para contar quantos registros tem de cada ano, e o sort_index ordena em ano, mes dia
registros_por_mes = tabela_silvestre["mes_numero"].value_counts().sort_index()
registros_por_dia_semana = tabela_silvestre["dia_semana_numero"].value_counts().sort_index()

# as proximas funções servem para os graficos da distribuição temporal
print("\nPeríodo coberto:", tabela_silvestre["data"].min().date(), "até", tabela_silvestre["data"].max().date()) # min = data mais antiga, max = data mais recente

print("\nRegistros por ano:")
print(registros_por_ano)
print("\nRegistros por mês:")
print(registros_por_mes)
print("\nRegistros por dia da semana (0 = segunda):")
print(registros_por_dia_semana)

figura, eixos = plt.subplots(1, 3, figsize=(15, 4))
registros_por_ano.plot(kind="bar", ax=eixos[0], color="steelblue", title="Por ano")
registros_por_mes.plot(kind="bar", ax=eixos[1], color="steelblue", title="Por mês")
registros_por_dia_semana.plot(kind="bar", ax=eixos[2], color="steelblue", title="Por dia da semana (0 = segunda)")
for eixo in eixos:
    eixo.set_xlabel("")
    eixo.set_ylabel("Quantidade de registros")
    eixo.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_temporal.png", dpi=120)
plt.close()


# ===============================================================
# PASSO 7: DISTRIBUIÇÃO ESPACIAL
# ===============================================================
# funções para a parte de distribuição espacial
registros_por_uc = tabela_silvestre["uc"].value_counts()
print("\nRegistros por unidade de conservação (uc):")
print(registros_por_uc)

plt.figure(figsize=(8, 7))
for nome_uc, registros_da_uc in tabela_silvestre.groupby("uc"): # grouphy separa a tabela em um grupo por UC e o for percorre os grupos, cada um recebe seu nome e suas linhas
    plt.scatter(registros_da_uc["x"], registros_da_uc["y"], s=4, label=nome_uc) # desenha um ponto por atropelamento
plt.title("Localização dos atropelamentos de fauna silvestre")
plt.xlabel("x (UTM, metros)")
plt.ylabel("y (UTM, metros)")
plt.legend(markerscale=3)
plt.axis("equal") # deixar o eixo x e y com a mesma escala, pra não distorcer o mapa
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_espacial.png", dpi=120)
plt.close()


# ===============================================================
# PASSO 7.1: CONFERIR SE A UC COMBINA COM A LOCALIZAÇÃO (X, Y)
# ===============================================================
# funções para conferir se a coluna uc combina com a localização (x, y), isso pq ocorreu de alguns registros estarem com a uc errada, mas a localização estava correta, então é preciso corrigir a uc desses registros, e para isso é preciso saber quais estão errados
# isso consegue perceber no grafico gerado da distribuicao espacial, onde tem cores sobrepostas
tabela_silvestre["trecho_pela_localizacao"] = tabela_silvestre.apply(descobrir_trecho_pela_localizacao, axis=1) # .apply(função, axis=1) roda a função em cada linha e guarda o resultado na nova coluna

print("\nuc registrada (linhas) x trecho pela localização (colunas):")
print(pd.crosstab(tabela_silvestre["uc"], tabela_silvestre["trecho_pela_localizacao"])) # cruza duas colunas: as linhas mostram a uc registrada e as colunas o trecho pela localização. O que está fora da diagonal é inconsistente.

uc_conhecida = tabela_silvestre["uc"] != "nao_identificado" # nao ha uc para comparar, então não faz sentido marcar como inconsistente
uc_diferente = tabela_silvestre["uc"] != tabela_silvestre["trecho_pela_localizacao"]
registros_inconsistentes = tabela_silvestre[uc_conhecida & uc_diferente]

print("\nRegistros com uc diferente do trecho onde estão:", len(registros_inconsistentes))
print(registros_inconsistentes["uc"].value_counts())


# ===============================================================
# PASSO 8: SEGMENTOS DE 500 M E CONTAGEM POR SEGMENTO
# ===============================================================
#analises dos segmentos de 500m, e a contagem de atropelamentos por segmento é o alvo da regressão e a base dos pontos com mais atropelamento (hotspost)
tabela_silvestre["celula_x"] = tabela_silvestre["x"] // tamanho_segmento # faz divisao inteira e descarta o resto
tabela_silvestre["celula_y"] = tabela_silvestre["y"] // tamanho_segmento
tabela_silvestre["segmento"] = tabela_silvestre["celula_x"].astype(str) + "_" + tabela_silvestre["celula_y"].astype(str) # cria uma coluna com a coordenada do segmento, juntando as duas celulas em uma string (covertida pela .astype(str)), separadas por "_"

atropelamentos_por_segmento = tabela_silvestre["segmento"].value_counts()

print("\nQuantidade de segmentos com pelo menos 1 atropelamento:", len(atropelamentos_por_segmento))
print("\nEstatísticas da contagem por segmento:")
print(atropelamentos_por_segmento.describe().round(1))
print("\nPercentil 75 (candidato a limiar do hotspot):", atropelamentos_por_segmento.quantile(0.75)) # pontos de corte para definir o hotspot, ou seja, os segmentos com mais atropelamentos que 75% dos outros segmentos

atropelamentos_por_segmento.plot(kind="hist", bins=30, color="steelblue") # desenha um histograma
plt.title("Atropelamentos por segmento de 500 m")
plt.xlabel("Atropelamentos no segmento")
plt.ylabel("Quantidade de segmentos")
plt.tight_layout()
plt.savefig("graficosE1/distribuicao_contagem_segmentos.png", dpi=120)
plt.close()


# ===============================================================
# PASSO 9: OUTLIERS
# ===============================================================
# funções pra outliers, ou seja, segmentos com mais atropelamentos que 75% dos outros segmentos, que são os candidatos a hotspot
primeiro_quartil = atropelamentos_por_segmento.quantile(0.25) # mais baixo em atropelamento
terceiro_quartil = atropelamentos_por_segmento.quantile(0.75) # mais alto
intervalo_interquartil = terceiro_quartil - primeiro_quartil # os "50%" do meio

limite_inferior = primeiro_quartil - 1.5 * intervalo_interquartil
limite_superior = terceiro_quartil + 1.5 * intervalo_interquartil

segmentos_outliers = atropelamentos_por_segmento[
    (atropelamentos_por_segmento < limite_inferior) | (atropelamentos_por_segmento > limite_superior) # os valores abaixo de 25% ou acima de 75% são as outliers, ou seja, os segmentos com menos atropelamentos que 25% dos outros segmentos e os segmentos com mais atropelamentos que 75% dos outros segmentos
]

print("\nQ1:", primeiro_quartil, "| Q3:", terceiro_quartil, "| Intervalo interquartil:", intervalo_interquartil)
print("Limites aceitos: de", limite_inferior, "até", limite_superior)
print("Segmentos outliers:", len(segmentos_outliers), "de", len(atropelamentos_por_segmento))
print("\nOs maiores segmentos:")
print(segmentos_outliers.sort_values(ascending=False).head(10))

plt.figure(figsize=(4, 6))
plt.boxplot(atropelamentos_por_segmento) # desenha a caixa de Q1 a Q3, a linha da mediana, os “bigodes” até o último valor dentro dos limites, e cada outlier como um círculo isolado.
plt.title("Boxplot: atropelamentos por segmento de 500 m")
plt.ylabel("Atropelamentos no segmento")
plt.xticks([])
plt.tight_layout()
plt.savefig("graficosE1/outliers_segmentos.png", dpi=120)
plt.close()


# ===============================================================
# PASSO 10: CORRELAÇÃO E ATRIBUTOS REDUNDANTES
# ===============================================================
# correlação e atributos redundantes
colunas_para_correlacao = ["x", "y", "ano", "campanha", "periodo", "mes_numero", "dia_semana_numero"]
matriz_correlacao = tabela_silvestre[colunas_para_correlacao].corr().round(2)
print(matriz_correlacao.to_string())

# o coeficiente de Pearson vai de −1 a 1. Perto de 1, as duas variáveis crescem juntas.
# Perto de −1, uma cresce quando a outra cai. Perto de 0, não há relação linear.
# O .corr() calcula isso para todos os pares. O .to_string() evita que o terminal esconda colunas com “...”

print(tabela_silvestre.nunique().sort_values())


# ===============================================================
# PASSO 10.4: ASSOCIAÇÃO ENTRE VARIÁVEIS CATEGÓRICAS (CLASSE X TRECHO)
# ===============================================================
# teste de associação entre variaveis categóricas, como classe e uc, por exemplo se a classe depende do trecho do atropelamento, se a uc influencia.
tabela_classe_trecho = tabela_silvestre[tabela_silvestre["classe"] != "nao_identificado"] # pega somente as classes identificadas

contagem_classe_trecho = pd.crosstab(tabela_classe_trecho["trecho_pela_localizacao"], tabela_classe_trecho["classe"]) # o pd monta uma tabela de cruzamento, linhas = trecho, colunas = classes, e cada celula conta quantos registros tem aquela combinação de trecho e classe. Serve pra ver se a classe depende do trecho, ou seja, se a distribuição das classes é diferente entre os trechos.
percentual_classe_trecho = (pd.crosstab(tabela_classe_trecho["trecho_pela_localizacao"], tabela_classe_trecho["classe"], normalize="index") * 100).round(1) # faz a mesma coisa que a linha anterior, mas o normalize="index" faz com que cada linha some 100%, ou seja, mostra o percentual de cada classe dentro do trecho, e não a quantidade absoluta. O * 100 transforma em percentual e o .round(1) limita a 1 casa decimal.

qui_quadrado, valor_p, graus_liberdade, esperado = chi2_contingency(contagem_classe_trecho) # o graus_liberdade e esperado não são usadas, mas é necessário colocar elas pq a função recebe 4 valores
total_registros = contagem_classe_trecho.to_numpy().sum() # to_numpy e sum somam todas as celulas da tabela
menor_dimensao = min(contagem_classe_trecho.shape) - 1 # o min(contagem_classe_trecho.shape) pega a menor dimensão da tabela, ou seja, o menor número de linhas ou colunas. O -1 é necessário pq o coeficiente de Cramer é calculado com base no menor número de categorias menos 1.
v_de_cramer = (qui_quadrado / (total_registros * menor_dimensao)) ** 0.5 # formula do coeficiente de Cramer, que é uma medida de associação entre duas variáveis categóricas. Ele varia de 0 a 1, onde 0 indica nenhuma associação e 1 indica associação perfeita. No nosso caso deu 0,15 logo a diferença é pequena mas existente.


# ===============================================================
# PASSO 11: CLASSES COM POUCOS EXEMPLOS
# ===============================================================
# função para verificar classes com poucos exemplos
registros_por_classe_valida = tabela_silvestre[tabela_silvestre["classe"] != "nao_identificado"]["classe"].value_counts()
print("\nMenor classe do multiclasse:", registros_por_classe_valida.idxmin(), "com", registros_por_classe_valida.min(), "registros")

especies_identificadas = tabela_silvestre[tabela_silvestre["especie_nm"] != "nao_identificado"]
registros_por_especie = especies_identificadas["especie_nm"].value_counts()

for minimo_registros in [5, 10, 30, 50]:
    especies_com_minimo = (registros_por_especie >= minimo_registros).sum()
    registros_cobertos = registros_por_especie[registros_por_especie >= minimo_registros].sum()
    print("Espécies com pelo menos", minimo_registros, "registros:", especies_com_minimo,
          "(cobrem", registros_cobertos, "registros)")