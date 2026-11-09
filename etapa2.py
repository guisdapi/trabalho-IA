# 2 etapa IA - Pré-processamento

# ===============================================================
# IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
caminho_entrada = "data/rodofauna_etapa1.csv"  # tabela salva no passo 12 da etapa 1
ano_inicio_monitoramento = 2010  # o monitoramento começou em abril de 2010
mes_inicio_periodo = 4           # cada período de monitoramento vai de abril até março do ano seguinte
tamanho_segmento = 500           # metros, o mesmo da etapa 1
percentil_hotspot = 0.75         # segmentos com contagem >= P75 são hotspot
pasta_graficos = "graficosE2"
semente = 42                     # fixa os sorteios, para todo o grupo obter as mesmas divisões
proporcao_validacao = 0.15
proporcao_teste = 0.15
tamanho_bloco_espacial = 3000    # metros, blocos de 3 km (6 segmentos) usados na divisão espacial
periodos_treino = [1, 2, 3]      # divisão temporal: abr/2010 a mar/2013
periodos_validacao = [4]         # abr/2013 a mar/2014
periodos_teste = [5]             # abr/2014 a mar/2015

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True) # cria a pasta dos gráficos da etapa 2 se ela não existir


# ===============================================================
# PASSO 1: CARREGAR A TABELA DA ETAPA 1 E CRIAR O PERÍODO DE MONITORAMENTO
# ===============================================================
tabela = pd.read_csv(caminho_entrada, parse_dates=["data"]) # parse_dates faz a coluna data ser lida como data de verdade, e não como texto
print("Registros carregados:", len(tabela))
print("Atributos:", tabela.shape[1])
print("Período coberto:", tabela["data"].min().date(), "até", tabela["data"].max().date())

# ano em que o período do registro começou: de abril a dezembro é o próprio ano, de janeiro a março é o ano anterior
mes = tabela["data"].dt.month
ano = tabela["data"].dt.year
ano_inicio_do_periodo = ano.where(mes >= mes_inicio_periodo, ano - 1) # .where mantém o valor onde a condição é verdadeira e troca pelo segundo valor onde é falsa

# transforma o ano de início em um número de período: abr/2010-mar/2011 = 1, abr/2011-mar/2012 = 2, ... abr/2014-mar/2015 = 5
tabela["periodo_monitoramento"] = ano_inicio_do_periodo - ano_inicio_monitoramento + 1

print("\nRegistros por período de monitoramento (abril a março):")
print(tabela["periodo_monitoramento"].value_counts().sort_index())

# conferência 1: todo registro precisa cair entre o período 1 e o 5
fora_dos_periodos = tabela[(tabela["periodo_monitoramento"] < 1) | (tabela["periodo_monitoramento"] > 5)]
print("\nRegistros fora dos períodos 1 a 5:", len(fora_dos_periodos))

# conferência 2: mostra o primeiro e o último dia com registro em cada período, para ver se os cortes estão em abril/março
print("\nPrimeira e última data de cada período:")
print(tabela.groupby("periodo_monitoramento")["data"].agg(["min", "max"]))

# conferência 3: compara com a coluna periodo original da planilha (se ela já for o ciclo abril-março, as duas vão coincidir)
if "periodo" in tabela.columns:
    print("\nperiodo da planilha (linhas) x periodo_monitoramento calculado (colunas):")
    print(pd.crosstab(tabela["periodo"], tabela["periodo_monitoramento"]))


# ===============================================================
# PASSO 2: TABELA DE SEGMENTOS E ALVO DO HOTSPOT (CLASSIFICAÇÃO BINÁRIA)
# ===============================================================
# cada linha passa a ser um segmento de 500 m, e não mais um atropelamento
# groupby junta todos os registros do mesmo segmento, e o agg diz o que calcular em cada grupo:
# celula_x e celula_y são iguais em todo o segmento (pega a primeira), trecho também, e size conta quantos registros o segmento tem
segmentos = tabela.groupby("segmento").agg(
    celula_x=("celula_x", "first"),
    celula_y=("celula_y", "first"),
    trecho=("trecho_pela_localizacao", "first"),
    atropelamentos=("segmento", "size"),
).reset_index() # reset_index transforma o segmento de índice em coluna normal

# conferência: um segmento não pode ter registros em dois trechos diferentes
# (os limites do trecho, y = 8245000 e x = 205000, são múltiplos de 500, então caem exatamente na borda dos segmentos)
trechos_por_segmento = tabela.groupby("segmento")["trecho_pela_localizacao"].nunique()
print("\nSegmentos com registros em mais de um trecho:", (trechos_por_segmento > 1).sum())

# centro do segmento: a célula começa em celula * 500 e termina 500 m depois, então o meio fica a 250 m do início
segmentos["centro_x"] = segmentos["celula_x"] * tamanho_segmento + tamanho_segmento / 2
segmentos["centro_y"] = segmentos["celula_y"] * tamanho_segmento + tamanho_segmento / 2

print("\nQuantidade de segmentos:", len(segmentos))
print("\nEstatísticas dos atropelamentos por segmento (2010-2015):")
print(segmentos["atropelamentos"].describe().round(1))

# alvo binário: hotspot = 1 quando o segmento tem tantos ou mais atropelamentos que o percentil 75
limiar_hotspot = segmentos["atropelamentos"].quantile(percentil_hotspot)
segmentos["hotspot"] = (segmentos["atropelamentos"] >= limiar_hotspot).astype(int) # astype(int) troca verdadeiro/falso por 1/0

print("\nLimiar do hotspot (P75):", limiar_hotspot, "atropelamentos")
print("\nSegmentos por classe do hotspot:")
print(pd.DataFrame({
    "quantidade": segmentos["hotspot"].value_counts(),
    "percentual": (segmentos["hotspot"].value_counts(normalize=True) * 100).round(1), # normalize=True devolve a proporção em vez da contagem
}))

print("\nHotspots por trecho:")
print(pd.crosstab(segmentos["trecho"], segmentos["hotspot"], margins=True)) # margins=True adiciona a linha e a coluna de total

# mapa dos segmentos: cinza = baixo risco, vermelho = hotspot
plt.figure(figsize=(8, 7))
baixo_risco = segmentos[segmentos["hotspot"] == 0]
hotspots = segmentos[segmentos["hotspot"] == 1]
plt.scatter(baixo_risco["centro_x"], baixo_risco["centro_y"], s=10, color="lightgray", label="baixo risco (0)")
plt.scatter(hotspots["centro_x"], hotspots["centro_y"], s=10, color="crimson", label=f"hotspot (1): >= {limiar_hotspot:.0f} atropelamentos")
plt.title("Segmentos de 500 m: hotspots de atropelamento (P75)")
plt.xlabel("x (UTM, metros)")
plt.ylabel("y (UTM, metros)")
plt.legend(markerscale=2)
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "hotspots_segmentos.png"), dpi=120)
plt.close()


# ===============================================================
# PASSO 3: TABELA SEGMENTO x PERÍODO (REGRESSÃO)
# ===============================================================
# cada linha passa a ser um segmento em um período de monitoramento (abril a março)
# o alvo da regressão é quantos atropelamentos o segmento teve naquele período
# o período só organiza as linhas e depois divide treino/validação/teste; ele não entra como atributo de entrada

# conferência: os zeros só fazem sentido se todos os trechos foram percorridos em todos os períodos
print("\nRegistros por trecho e período (todos precisam ter registros nos 5 períodos):")
print(pd.crosstab(tabela["trecho_pela_localizacao"], tabela["periodo_monitoramento"]))

# monta todas as combinações segmento x período: 293 segmentos x 5 períodos
lista_segmentos = segmentos["segmento"]
lista_periodos = sorted(tabela["periodo_monitoramento"].unique())
todas_combinacoes = pd.MultiIndex.from_product([lista_segmentos, lista_periodos], names=["segmento", "periodo_monitoramento"]) # from_product cria todos os pares possíveis entre as duas listas

# conta os atropelamentos de cada par que aparece na tabela, e reindex completa os pares que não aparecem com 0
contagem = tabela.groupby(["segmento", "periodo_monitoramento"]).size()
segmento_periodo = contagem.reindex(todas_combinacoes, fill_value=0).rename("atropelamentos").reset_index()

# traz as informações fixas de cada segmento (centro e trecho) da tabela do passo 2
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "centro_x", "centro_y", "trecho"]], on="segmento") # merge junta as duas tabelas pela coluna segmento

print("\nLinhas da tabela segmento x período:", len(segmento_periodo), "(", len(lista_segmentos), "segmentos x", len(lista_periodos), "períodos )")
print("Total de atropelamentos conferido:", segmento_periodo["atropelamentos"].sum(), "de", len(tabela)) # precisa bater com o total de registros

print("\nEstatísticas do alvo da regressão (atropelamentos por segmento e período):")
print(segmento_periodo["atropelamentos"].describe().round(2))

quantidade_zeros = (segmento_periodo["atropelamentos"] == 0).sum()
print("\nLinhas com 0 atropelamentos:", quantidade_zeros, f"({quantidade_zeros / len(segmento_periodo) * 100:.1f}%)")

print("\nMédia de atropelamentos por segmento em cada período:")
print(segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().round(2))

# gráficos: distribuição do alvo e média por período
figura, eixos = plt.subplots(1, 2, figsize=(13, 4))
segmento_periodo["atropelamentos"].plot(kind="hist", bins=30, ax=eixos[0], color="steelblue")
eixos[0].set_title("Atropelamentos por segmento e período")
eixos[0].set_xlabel("Atropelamentos no segmento no período")
eixos[0].set_ylabel("Quantidade de linhas")
segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().plot(kind="bar", ax=eixos[1], color="steelblue")
eixos[1].set_title("Média por segmento em cada período")
eixos[1].set_xlabel("Período de monitoramento (abril a março)")
eixos[1].set_ylabel("Atropelamentos por segmento")
eixos[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "regressao_segmento_periodo.png"), dpi=120)
plt.close()


# ===============================================================
# PASSO 4: BASE DO MULTICLASSE (CLASSE TAXONÔMICA)
# ===============================================================
# aqui cada linha continua sendo um atropelamento, e o alvo é a classe (birds, mammals, reptiles, amphibians)
# registros sem classe identificada não têm alvo, então saem da base
base_multiclasse = tabela[tabela["classe"] != "nao_identificado"].copy().reset_index(drop=True) # reset_index(drop=True) renumera as linhas de 0 em diante
print("\nRegistros removidos por classe nao_identificado:", len(tabela) - len(base_multiclasse))
print("Registros na base do multiclasse:", len(base_multiclasse))

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
# PASSO 5: DIVISÃO EM TREINO, VALIDAÇÃO E TESTE
# ===============================================================
# treino = onde o modelo aprende | validação = onde escolhemos os hiperparâmetros | teste = avaliação final, só usado no fim
# cada problema recebe uma coluna por tipo de divisão, com o texto "treino", "validacao" ou "teste" em cada linha

def dividir_aleatorio(tabela_problema, coluna_estratificar=None):
    # sorteia as linhas em 70/15/15; stratify mantém a proporção do alvo igual nas três partes
    indices = np.arange(len(tabela_problema))
    estratos = tabela_problema[coluna_estratificar].to_numpy() if coluna_estratificar else None
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, stratify=estratos, random_state=semente)

    # a validação é 15% do total, mas o sorteio agora é feito sobre os 85% que sobraram: 0.15 / 0.85 = 0.176
    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    estratos_resto = estratos[indices_resto] if coluna_estratificar else None
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, stratify=estratos_resto, random_state=semente)

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


def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo, alvo_e_classe=True):
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    if alvo_e_classe:
        # percentual de cada classe dentro de cada parte: o ideal é ficar parecido nas três
        percentual_alvo = (pd.crosstab(tabela_problema[coluna_particao], tabela_problema[coluna_alvo], normalize="index") * 100).round(1)
        resumo = resumo.join(percentual_alvo)
    else:
        resumo["média do alvo"] = tabela_problema.groupby(coluna_particao)[coluna_alvo].mean().round(2)
        resumo["% de zeros"] = (tabela_problema.groupby(coluna_particao)[coluna_alvo].apply(lambda valores: (valores == 0).mean()) * 100).round(1)
    print(resumo.to_string())


# --- divisão espacial (a mesma para os três problemas) ---
# o mapa é dividido em blocos de 3 km, e cada bloco vai INTEIRO para treino, validação ou teste
# assim o teste fica em trechos de rodovia que o modelo nunca viu, e um segmento de teste não tem o vizinho do lado no treino
# o sorteio é feito uma vez na tabela de segmentos e depois copiado para os outros problemas pelo segmento,
# então as regiões de teste são as mesmas no hotspot, no multiclasse e na regressão
segmentos["bloco_espacial"] = (
    (segmentos["centro_x"] // tamanho_bloco_espacial).astype(int).astype(str) + "_"
    + (segmentos["centro_y"] // tamanho_bloco_espacial).astype(int).astype(str)
)
print("\nBlocos espaciais de", tamanho_bloco_espacial, "m:", segmentos["bloco_espacial"].nunique())

# StratifiedGroupKFold separa os blocos em 7 grupos (cada um com ~1/7 = 14% dos segmentos),
# tentando manter em cada grupo a mesma proporção de hotspots; o grupo 0 vira teste, o 1 validação e o resto treino
separador_espacial = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=semente)
grupo_do_segmento = np.empty(len(segmentos), dtype=int)
for numero_grupo, (_, indices_grupo) in enumerate(separador_espacial.split(segmentos, segmentos["hotspot"], groups=segmentos["bloco_espacial"])):
    grupo_do_segmento[indices_grupo] = numero_grupo
segmentos["particao_espacial"] = np.select([grupo_do_segmento == 0, grupo_do_segmento == 1], ["teste", "validacao"], default="treino")

# --- aplica as divisões em cada problema ---
segmentos["particao_aleatoria"] = dividir_aleatorio(segmentos, "hotspot")
# o hotspot não tem divisão temporal: cada segmento soma os 5 períodos, então não existe um "período" para separar

base_multiclasse["particao_aleatoria"] = dividir_aleatorio(base_multiclasse, "classe")
base_multiclasse = base_multiclasse.merge(segmentos[["segmento", "particao_espacial"]], on="segmento") # copia a parte espacial do segmento de cada registro
base_multiclasse["particao_temporal"] = dividir_temporal(base_multiclasse)

segmento_periodo["particao_aleatoria"] = dividir_aleatorio(segmento_periodo)
# na aleatória o mesmo segmento aparece em períodos diferentes no treino e no teste: é o vazamento que vamos discutir
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "particao_espacial"]], on="segmento")
segmento_periodo["particao_temporal"] = dividir_temporal(segmento_periodo)

# --- conferência das divisões ---
print("\nHOTSPOT (binária) - linhas e % de cada classe (0 = baixo risco, 1 = hotspot) em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial"]:
    resumir_divisao(segmentos, coluna_particao, "hotspot")

print("\nMULTICLASSE - linhas e % de cada classe em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(base_multiclasse, coluna_particao, "classe")

print("\nREGRESSÃO - linhas, média de atropelamentos e % de zeros em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(segmento_periodo, coluna_particao, "atropelamentos", alvo_e_classe=False)

# 2 etapa IA - Pré-processamento

# ===============================================================
# IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
caminho_entrada = "data/rodofauna_etapa1.csv"  # tabela salva no passo 12 da etapa 1
ano_inicio_monitoramento = 2010  # o monitoramento começou em abril de 2010
mes_inicio_periodo = 4           # cada período de monitoramento vai de abril até março do ano seguinte
tamanho_segmento = 500           # metros, o mesmo da etapa 1
percentil_hotspot = 0.75         # segmentos com contagem >= P75 são hotspot
pasta_graficos = "graficosE2"
semente = 42                     # fixa os sorteios, para todo o grupo obter as mesmas divisões
proporcao_validacao = 0.15
proporcao_teste = 0.15
tamanho_bloco_espacial = 3000    # metros, blocos de 3 km (6 segmentos) usados na divisão espacial
periodos_treino = [1, 2, 3]      # divisão temporal: abr/2010 a mar/2013
periodos_validacao = [4]         # abr/2013 a mar/2014
periodos_teste = [5]             # abr/2014 a mar/2015

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True) # cria a pasta dos gráficos da etapa 2 se ela não existir


# ===============================================================
# PASSO 1: CARREGAR A TABELA DA ETAPA 1 E CRIAR O PERÍODO DE MONITORAMENTO
# ===============================================================
tabela = pd.read_csv(caminho_entrada, parse_dates=["data"]) # parse_dates faz a coluna data ser lida como data de verdade, e não como texto
print("Registros carregados:", len(tabela))
print("Atributos:", tabela.shape[1])
print("Período coberto:", tabela["data"].min().date(), "até", tabela["data"].max().date())

# ano em que o período do registro começou: de abril a dezembro é o próprio ano, de janeiro a março é o ano anterior
mes = tabela["data"].dt.month
ano = tabela["data"].dt.year
ano_inicio_do_periodo = ano.where(mes >= mes_inicio_periodo, ano - 1) # .where mantém o valor onde a condição é verdadeira e troca pelo segundo valor onde é falsa

# transforma o ano de início em um número de período: abr/2010-mar/2011 = 1, abr/2011-mar/2012 = 2, ... abr/2014-mar/2015 = 5
tabela["periodo_monitoramento"] = ano_inicio_do_periodo - ano_inicio_monitoramento + 1

print("\nRegistros por período de monitoramento (abril a março):")
print(tabela["periodo_monitoramento"].value_counts().sort_index())

# conferência 1: todo registro precisa cair entre o período 1 e o 5
fora_dos_periodos = tabela[(tabela["periodo_monitoramento"] < 1) | (tabela["periodo_monitoramento"] > 5)]
print("\nRegistros fora dos períodos 1 a 5:", len(fora_dos_periodos))

# conferência 2: mostra o primeiro e o último dia com registro em cada período, para ver se os cortes estão em abril/março
print("\nPrimeira e última data de cada período:")
print(tabela.groupby("periodo_monitoramento")["data"].agg(["min", "max"]))

# conferência 3: compara com a coluna periodo original da planilha (se ela já for o ciclo abril-março, as duas vão coincidir)
if "periodo" in tabela.columns:
    print("\nperiodo da planilha (linhas) x periodo_monitoramento calculado (colunas):")
    print(pd.crosstab(tabela["periodo"], tabela["periodo_monitoramento"]))


# ===============================================================
# PASSO 2: TABELA DE SEGMENTOS E ALVO DO HOTSPOT (CLASSIFICAÇÃO BINÁRIA)
# ===============================================================
# cada linha passa a ser um segmento de 500 m, e não mais um atropelamento
# groupby junta todos os registros do mesmo segmento, e o agg diz o que calcular em cada grupo:
# celula_x e celula_y são iguais em todo o segmento (pega a primeira), trecho também, e size conta quantos registros o segmento tem
segmentos = tabela.groupby("segmento").agg(
    celula_x=("celula_x", "first"),
    celula_y=("celula_y", "first"),
    trecho=("trecho_pela_localizacao", "first"),
    atropelamentos=("segmento", "size"),
).reset_index() # reset_index transforma o segmento de índice em coluna normal

# conferência: um segmento não pode ter registros em dois trechos diferentes
# (os limites do trecho, y = 8245000 e x = 205000, são múltiplos de 500, então caem exatamente na borda dos segmentos)
trechos_por_segmento = tabela.groupby("segmento")["trecho_pela_localizacao"].nunique()
print("\nSegmentos com registros em mais de um trecho:", (trechos_por_segmento > 1).sum())

# centro do segmento: a célula começa em celula * 500 e termina 500 m depois, então o meio fica a 250 m do início
segmentos["centro_x"] = segmentos["celula_x"] * tamanho_segmento + tamanho_segmento / 2
segmentos["centro_y"] = segmentos["celula_y"] * tamanho_segmento + tamanho_segmento / 2

print("\nQuantidade de segmentos:", len(segmentos))
print("\nEstatísticas dos atropelamentos por segmento (2010-2015):")
print(segmentos["atropelamentos"].describe().round(1))

# alvo binário: hotspot = 1 quando o segmento tem tantos ou mais atropelamentos que o percentil 75
limiar_hotspot = segmentos["atropelamentos"].quantile(percentil_hotspot)
segmentos["hotspot"] = (segmentos["atropelamentos"] >= limiar_hotspot).astype(int) # astype(int) troca verdadeiro/falso por 1/0

print("\nLimiar do hotspot (P75):", limiar_hotspot, "atropelamentos")
print("\nSegmentos por classe do hotspot:")
print(pd.DataFrame({
    "quantidade": segmentos["hotspot"].value_counts(),
    "percentual": (segmentos["hotspot"].value_counts(normalize=True) * 100).round(1), # normalize=True devolve a proporção em vez da contagem
}))

print("\nHotspots por trecho:")
print(pd.crosstab(segmentos["trecho"], segmentos["hotspot"], margins=True)) # margins=True adiciona a linha e a coluna de total

# mapa dos segmentos: cinza = baixo risco, vermelho = hotspot
plt.figure(figsize=(8, 7))
baixo_risco = segmentos[segmentos["hotspot"] == 0]
hotspots = segmentos[segmentos["hotspot"] == 1]
plt.scatter(baixo_risco["centro_x"], baixo_risco["centro_y"], s=10, color="lightgray", label="baixo risco (0)")
plt.scatter(hotspots["centro_x"], hotspots["centro_y"], s=10, color="crimson", label=f"hotspot (1): >= {limiar_hotspot:.0f} atropelamentos")
plt.title("Segmentos de 500 m: hotspots de atropelamento (P75)")
plt.xlabel("x (UTM, metros)")
plt.ylabel("y (UTM, metros)")
plt.legend(markerscale=2)
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "hotspots_segmentos.png"), dpi=120)
plt.close()


# ===============================================================
# PASSO 3: TABELA SEGMENTO x PERÍODO (REGRESSÃO)
# ===============================================================
# cada linha passa a ser um segmento em um período de monitoramento (abril a março)
# o alvo da regressão é quantos atropelamentos o segmento teve naquele período
# o período só organiza as linhas e depois divide treino/validação/teste; ele não entra como atributo de entrada

# conferência: os zeros só fazem sentido se todos os trechos foram percorridos em todos os períodos
print("\nRegistros por trecho e período (todos precisam ter registros nos 5 períodos):")
print(pd.crosstab(tabela["trecho_pela_localizacao"], tabela["periodo_monitoramento"]))

# monta todas as combinações segmento x período: 293 segmentos x 5 períodos
lista_segmentos = segmentos["segmento"]
lista_periodos = sorted(tabela["periodo_monitoramento"].unique())
todas_combinacoes = pd.MultiIndex.from_product([lista_segmentos, lista_periodos], names=["segmento", "periodo_monitoramento"]) # from_product cria todos os pares possíveis entre as duas listas

# conta os atropelamentos de cada par que aparece na tabela, e reindex completa os pares que não aparecem com 0
contagem = tabela.groupby(["segmento", "periodo_monitoramento"]).size()
segmento_periodo = contagem.reindex(todas_combinacoes, fill_value=0).rename("atropelamentos").reset_index()

# traz as informações fixas de cada segmento (centro e trecho) da tabela do passo 2
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "centro_x", "centro_y", "trecho"]], on="segmento") # merge junta as duas tabelas pela coluna segmento

print("\nLinhas da tabela segmento x período:", len(segmento_periodo), "(", len(lista_segmentos), "segmentos x", len(lista_periodos), "períodos )")
print("Total de atropelamentos conferido:", segmento_periodo["atropelamentos"].sum(), "de", len(tabela)) # precisa bater com o total de registros

print("\nEstatísticas do alvo da regressão (atropelamentos por segmento e período):")
print(segmento_periodo["atropelamentos"].describe().round(2))

quantidade_zeros = (segmento_periodo["atropelamentos"] == 0).sum()
print("\nLinhas com 0 atropelamentos:", quantidade_zeros, f"({quantidade_zeros / len(segmento_periodo) * 100:.1f}%)")

print("\nMédia de atropelamentos por segmento em cada período:")
print(segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().round(2))

# gráficos: distribuição do alvo e média por período
figura, eixos = plt.subplots(1, 2, figsize=(13, 4))
segmento_periodo["atropelamentos"].plot(kind="hist", bins=30, ax=eixos[0], color="steelblue")
eixos[0].set_title("Atropelamentos por segmento e período")
eixos[0].set_xlabel("Atropelamentos no segmento no período")
eixos[0].set_ylabel("Quantidade de linhas")
segmento_periodo.groupby("periodo_monitoramento")["atropelamentos"].mean().plot(kind="bar", ax=eixos[1], color="steelblue")
eixos[1].set_title("Média por segmento em cada período")
eixos[1].set_xlabel("Período de monitoramento (abril a março)")
eixos[1].set_ylabel("Atropelamentos por segmento")
eixos[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(pasta_graficos, "regressao_segmento_periodo.png"), dpi=120)
plt.close()


# ===============================================================
# PASSO 4: BASE DO MULTICLASSE (CLASSE TAXONÔMICA)
# ===============================================================
# aqui cada linha continua sendo um atropelamento, e o alvo é a classe (birds, mammals, reptiles, amphibians)
# registros sem classe identificada não têm alvo, então saem da base
base_multiclasse = tabela[tabela["classe"] != "nao_identificado"].copy().reset_index(drop=True) # reset_index(drop=True) renumera as linhas de 0 em diante
print("\nRegistros removidos por classe nao_identificado:", len(tabela) - len(base_multiclasse))
print("Registros na base do multiclasse:", len(base_multiclasse))

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
# PASSO 5: DIVISÃO EM TREINO, VALIDAÇÃO E TESTE
# ===============================================================
# treino = onde o modelo aprende | validação = onde escolhemos os hiperparâmetros | teste = avaliação final, só usado no fim
# cada problema recebe uma coluna por tipo de divisão, com o texto "treino", "validacao" ou "teste" em cada linha

def dividir_aleatorio(tabela_problema, coluna_estratificar=None):
    # sorteia as linhas em 70/15/15; stratify mantém a proporção do alvo igual nas três partes
    indices = np.arange(len(tabela_problema))
    estratos = tabela_problema[coluna_estratificar].to_numpy() if coluna_estratificar else None
    indices_resto, indices_teste = train_test_split(indices, test_size=proporcao_teste, stratify=estratos, random_state=semente)

    # a validação é 15% do total, mas o sorteio agora é feito sobre os 85% que sobraram: 0.15 / 0.85 = 0.176
    proporcao_val_no_resto = proporcao_validacao / (1 - proporcao_teste)
    estratos_resto = estratos[indices_resto] if coluna_estratificar else None
    indices_treino, indices_validacao = train_test_split(indices_resto, test_size=proporcao_val_no_resto, stratify=estratos_resto, random_state=semente)

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


def resumir_divisao(tabela_problema, coluna_particao, coluna_alvo, alvo_e_classe=True):
    print(f"\n  {coluna_particao}:")
    quantidade = tabela_problema[coluna_particao].value_counts().reindex(["treino", "validacao", "teste"])
    resumo = pd.DataFrame({"linhas": quantidade, "% das linhas": (quantidade / len(tabela_problema) * 100).round(1)})
    if alvo_e_classe:
        # percentual de cada classe dentro de cada parte: o ideal é ficar parecido nas três
        percentual_alvo = (pd.crosstab(tabela_problema[coluna_particao], tabela_problema[coluna_alvo], normalize="index") * 100).round(1)
        resumo = resumo.join(percentual_alvo)
    else:
        resumo["média do alvo"] = tabela_problema.groupby(coluna_particao)[coluna_alvo].mean().round(2)
        resumo["% de zeros"] = (tabela_problema.groupby(coluna_particao)[coluna_alvo].apply(lambda valores: (valores == 0).mean()) * 100).round(1)
    print(resumo.to_string())


# --- divisão espacial (a mesma para os três problemas) ---
# o mapa é dividido em blocos de 3 km, e cada bloco vai INTEIRO para treino, validação ou teste
# assim o teste fica em trechos de rodovia que o modelo nunca viu, e um segmento de teste não tem o vizinho do lado no treino
# o sorteio é feito uma vez na tabela de segmentos e depois copiado para os outros problemas pelo segmento,
# então as regiões de teste são as mesmas no hotspot, no multiclasse e na regressão
segmentos["bloco_espacial"] = (
    (segmentos["centro_x"] // tamanho_bloco_espacial).astype(int).astype(str) + "_"
    + (segmentos["centro_y"] // tamanho_bloco_espacial).astype(int).astype(str)
)
print("\nBlocos espaciais de", tamanho_bloco_espacial, "m:", segmentos["bloco_espacial"].nunique())

# StratifiedGroupKFold separa os blocos em 7 grupos (cada um com ~1/7 = 14% dos segmentos),
# tentando manter em cada grupo a mesma proporção de hotspots; o grupo 0 vira teste, o 1 validação e o resto treino
separador_espacial = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=semente)
grupo_do_segmento = np.empty(len(segmentos), dtype=int)
for numero_grupo, (_, indices_grupo) in enumerate(separador_espacial.split(segmentos, segmentos["hotspot"], groups=segmentos["bloco_espacial"])):
    grupo_do_segmento[indices_grupo] = numero_grupo
segmentos["particao_espacial"] = np.select([grupo_do_segmento == 0, grupo_do_segmento == 1], ["teste", "validacao"], default="treino")

# --- aplica as divisões em cada problema ---
segmentos["particao_aleatoria"] = dividir_aleatorio(segmentos, "hotspot")
# o hotspot não tem divisão temporal: cada segmento soma os 5 períodos, então não existe um "período" para separar

base_multiclasse["particao_aleatoria"] = dividir_aleatorio(base_multiclasse, "classe")
base_multiclasse = base_multiclasse.merge(segmentos[["segmento", "particao_espacial"]], on="segmento") # copia a parte espacial do segmento de cada registro
base_multiclasse["particao_temporal"] = dividir_temporal(base_multiclasse)

segmento_periodo["particao_aleatoria"] = dividir_aleatorio(segmento_periodo)
# na aleatória o mesmo segmento aparece em períodos diferentes no treino e no teste: é o vazamento que vamos discutir
segmento_periodo = segmento_periodo.merge(segmentos[["segmento", "particao_espacial"]], on="segmento")
segmento_periodo["particao_temporal"] = dividir_temporal(segmento_periodo)

# --- conferência das divisões ---
print("\nHOTSPOT (binária) - linhas e % de cada classe (0 = baixo risco, 1 = hotspot) em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial"]:
    resumir_divisao(segmentos, coluna_particao, "hotspot")

print("\nMULTICLASSE - linhas e % de cada classe em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(base_multiclasse, coluna_particao, "classe")

print("\nREGRESSÃO - linhas, média de atropelamentos e % de zeros em cada parte:")
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    resumir_divisao(segmento_periodo, coluna_particao, "atropelamentos", alvo_e_classe=False)


# ===============================================================
# PASSO 6: VAZAMENTO ESPACIAL (O TESTE ESTÁ PERTO DO TREINO?)
# ===============================================================
# se um ponto de teste está a poucos metros de um ponto de treino, o modelo pode acertar só por "lembrar" daquele lugar
# isso deixa a avaliação otimista: parece que o modelo generaliza, mas ele está sendo testado em lugares que já conhece
# aqui medimos, para cada linha de teste, a distância até a linha de treino mais próxima

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


print("\nDistância de cada linha de teste até a linha de treino mais próxima:")
print("\nHOTSPOT (centro dos segmentos):")
resumir_vazamento(segmentos, ["particao_aleatoria", "particao_espacial"], "centro_x", "centro_y")
print("\nMULTICLASSE (local de cada atropelamento):")
resumir_vazamento(base_multiclasse, ["particao_aleatoria", "particao_espacial", "particao_temporal"], "x", "y")
print("\nREGRESSÃO (centro dos segmentos):")
resumir_vazamento(segmento_periodo, ["particao_aleatoria", "particao_espacial", "particao_temporal"], "centro_x", "centro_y")

# na regressão, distância 0 significa que o MESMO segmento está no treino em outro período
for coluna_particao in ["particao_aleatoria", "particao_espacial", "particao_temporal"]:
    segmentos_treino = set(segmento_periodo.loc[segmento_periodo[coluna_particao] == "treino", "segmento"])
    linhas_teste = segmento_periodo[segmento_periodo[coluna_particao] == "teste"]
    percentual_repetido = linhas_teste["segmento"].isin(segmentos_treino).mean() * 100 # isin marca as linhas cujo segmento também aparece no treino
    print(f"  {coluna_particao}: {percentual_repetido:.1f}% das linhas de teste são de segmentos que também estão no treino")

# mapa das divisões do multiclasse: mostra visualmente onde ficou cada parte
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