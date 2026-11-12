# 2 etapa IA - Pré-processamento: parte COMUM aos três problemas
# roda ANTES dos arquivos de cada problema, porque eles usam o que é salvo aqui:
#   python etapa2_comum.py
#   python etapa2_binario.py      (hotspot)
#   python etapa2_multiclasse.py  (classe taxonômica)
#   python etapa2_regressao.py    (atropelamentos por segmento e período)
#
# o que fica aqui (e não em cada arquivo) é o que precisa ser IGUAL nos três problemas:
#   - o período de monitoramento (abril a março), usado pela divisão temporal e pela regressão
#   - a tabela de segmentos de 500 m, com o alvo do hotspot
#   - a divisão espacial: sorteada UMA vez nos segmentos, para as regiões de teste serem as mesmas nos três problemas

# ===============================================================
# IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
caminho_entrada = "data/rodofauna_etapa1.csv"  # tabela salva no passo 12 da etapa 1
ano_inicio_monitoramento = 2010  # o monitoramento começou em abril de 2010
mes_inicio_periodo = 4           # cada período de monitoramento vai de abril até março do ano seguinte
tamanho_segmento = 500           # metros, o mesmo da etapa 1
percentil_hotspot = 0.75         # segmentos com contagem >= P75 são hotspot
tamanho_bloco_espacial = 3000    # metros, blocos de 3 km (6 segmentos) usados na divisão espacial
semente = 42                     # fixa os sorteios, para todo o grupo obter as mesmas divisões
pasta_saida = "data/etapa2"      # onde ficam as tabelas comuns e os conjuntos prontos para as etapas 3 e 4

# arquivos que os três problemas vão ler
caminho_registros = os.path.join(pasta_saida, "registros_com_periodo.csv")  # um atropelamento por linha, com o período
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")              # um segmento por linha, com o hotspot e a divisão espacial

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_saida, exist_ok=True)


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
# PASSO 2: TABELA DE SEGMENTOS E ALVO DO HOTSPOT
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
# ele é calculado AQUI (e não só no etapa2_binario.py) porque a divisão espacial do passo 3 usa o hotspot para equilibrar as partes;
# a análise do hotspot (distribuição, mapa, hotspots por trecho) fica no etapa2_binario.py
limiar_hotspot = segmentos["atropelamentos"].quantile(percentil_hotspot)
segmentos["hotspot"] = (segmentos["atropelamentos"] >= limiar_hotspot).astype(int) # astype(int) troca verdadeiro/falso por 1/0
print("\nLimiar do hotspot (P75):", limiar_hotspot, "atropelamentos")


# ===============================================================
# PASSO 3: DIVISÃO ESPACIAL (A MESMA PARA OS TRÊS PROBLEMAS)
# ===============================================================
# o mapa é dividido em blocos de 3 km, e cada bloco vai INTEIRO para treino, validação ou teste
# assim o teste fica em trechos de rodovia que o modelo nunca viu, e um segmento de teste não tem o vizinho do lado no treino
# o sorteio é feito uma vez aqui, na tabela de segmentos, e cada arquivo de problema copia a parte pelo segmento,
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

print("\nSegmentos e blocos em cada parte da divisão espacial:")
print(pd.DataFrame({
    "segmentos": segmentos["particao_espacial"].value_counts(),
    "blocos": segmentos.groupby("particao_espacial")["bloco_espacial"].nunique(),
}).reindex(["treino", "validacao", "teste"]).to_string())


# ===============================================================
# PASSO 4: SALVAR AS TABELAS COMUNS
# ===============================================================
# os arquivos de cada problema leem estas duas tabelas em vez de refazer os passos 1 a 3
tabela.to_csv(caminho_registros, index=False, encoding="utf-8-sig")
segmentos.to_csv(caminho_segmentos, index=False, encoding="utf-8-sig")
print("\nTabelas comuns salvas:")
print(" ", caminho_registros, "->", len(tabela), "registros")
print(" ", caminho_segmentos, "->", len(segmentos), "segmentos")
