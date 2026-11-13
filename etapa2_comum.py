import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


caminho_entrada = "data/rodofauna_etapa1.csv"
ano_inicio_monitoramento = 2010
mes_inicio_periodo = 4
tamanho_segmento = 500
percentil_hotspot = 0.75
tamanho_bloco_espacial = 3000
semente = 42
pasta_saida = "data/etapa2"

caminho_registros = os.path.join(pasta_saida, "registros_com_periodo.csv")
caminho_segmentos = os.path.join(pasta_saida, "segmentos.csv")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_saida, exist_ok=True)


tabela = pd.read_csv(caminho_entrada, parse_dates=["data"])
print("Registros carregados:", len(tabela))
print("Atributos:", tabela.shape[1])
print("Período coberto:", tabela["data"].min().date(), "até", tabela["data"].max().date())

mes = tabela["data"].dt.month
ano = tabela["data"].dt.year
ano_inicio_do_periodo = ano.where(mes >= mes_inicio_periodo, ano - 1)

tabela["periodo_monitoramento"] = ano_inicio_do_periodo - ano_inicio_monitoramento + 1

print("\nRegistros por período de monitoramento (abril a março):")
print(tabela["periodo_monitoramento"].value_counts().sort_index())

fora_dos_periodos = tabela[(tabela["periodo_monitoramento"] < 1) | (tabela["periodo_monitoramento"] > 5)]
print("\nRegistros fora dos períodos 1 a 5:", len(fora_dos_periodos))

print("\nPrimeira e última data de cada período:")
print(tabela.groupby("periodo_monitoramento")["data"].agg(["min", "max"]))

if "periodo" in tabela.columns:
    print("\nperiodo da planilha (linhas) x periodo_monitoramento calculado (colunas):")
    print(pd.crosstab(tabela["periodo"], tabela["periodo_monitoramento"]))


segmentos = tabela.groupby("segmento").agg(
    celula_x=("celula_x", "first"),
    celula_y=("celula_y", "first"),
    trecho=("trecho_pela_localizacao", "first"),
    atropelamentos=("segmento", "size"),
).reset_index()

trechos_por_segmento = tabela.groupby("segmento")["trecho_pela_localizacao"].nunique()
print("\nSegmentos com registros em mais de um trecho:", (trechos_por_segmento > 1).sum())

segmentos["centro_x"] = segmentos["celula_x"] * tamanho_segmento + tamanho_segmento / 2
segmentos["centro_y"] = segmentos["celula_y"] * tamanho_segmento + tamanho_segmento / 2

print("\nQuantidade de segmentos:", len(segmentos))
print("\nEstatísticas dos atropelamentos por segmento (2010-2015):")
print(segmentos["atropelamentos"].describe().round(1))

limiar_hotspot = segmentos["atropelamentos"].quantile(percentil_hotspot)
segmentos["hotspot"] = (segmentos["atropelamentos"] >= limiar_hotspot).astype(int)
print("\nLimiar do hotspot (P75):", limiar_hotspot, "atropelamentos")


segmentos["bloco_espacial"] = (
    (segmentos["centro_x"] // tamanho_bloco_espacial).astype(int).astype(str) + "_"
    + (segmentos["centro_y"] // tamanho_bloco_espacial).astype(int).astype(str)
)
print("\nBlocos espaciais de", tamanho_bloco_espacial, "m:", segmentos["bloco_espacial"].nunique())

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


tabela.to_csv(caminho_registros, index=False, encoding="utf-8-sig")
segmentos.to_csv(caminho_segmentos, index=False, encoding="utf-8-sig")
print("\nTabelas comuns salvas:")
print(" ", caminho_registros, "->", len(tabela), "registros")
print(" ", caminho_segmentos, "->", len(segmentos), "segmentos")
