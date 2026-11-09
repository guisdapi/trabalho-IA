# 2 etapa IA - Pré-processamento

# ===============================================================
# IMPORTS
# ===============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
caminho_entrada = "data/rodofauna_etapa1.csv"  # tabela salva no passo 12 da etapa 1
ano_inicio_monitoramento = 2010  # o monitoramento começou em abril de 2010
mes_inicio_periodo = 4           # cada período de monitoramento vai de abril até março do ano seguinte
tamanho_segmento = 500           # metros, o mesmo da etapa 1
percentil_hotspot = 0.75         # segmentos com contagem >= P75 são hotspot
pasta_graficos = "graficosE2"

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