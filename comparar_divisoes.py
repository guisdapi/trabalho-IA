# 3 etapa IA - Comparação das divisões (generalização)
# roda DEPOIS do etapa3.py ter sido executado nas 3 divisões do multiclasse:
#   python etapa3.py multiclasse_aleatoria
#   python etapa3.py multiclasse_espacial
#   python etapa3.py multiclasse_temporal
#   python comparar_divisoes.py

# ===============================================================
# IMPORTS
# ===============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_resultados = "resultadosE3"  # onde o etapa3.py salvou as tabelas (passo 9)
pasta_graficos = "graficosE3"
problema = "multiclasse"
divisoes = ["aleatoria", "espacial", "temporal"]

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script


# ===============================================================
# PASSO 1: JUNTAR OS RESULTADOS DAS 3 DIVISÕES
# ===============================================================
# cada divisão tem o seu metricas_teste_*.csv e comparacao_final_*.csv; aqui viram uma linha cada numa tabela só
linhas = []
for divisao in divisoes:
    nome_conjunto = f"{problema}_{divisao}"
    caminho_metricas = os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv")
    if not os.path.exists(caminho_metricas):
        print("Faltando:", caminho_metricas, "-> rode antes: python etapa3.py", nome_conjunto)
        continue # pula essa divisão e segue com as outras

    metricas = pd.read_csv(caminho_metricas, index_col=0) # index_col=0: a primeira coluna (accuracy, f1_macro, ...) vira o índice
    comparacao = pd.read_csv(os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv"), index_col=0)
    coluna_mlp = [coluna for coluna in metricas.columns if coluna.startswith("mlp_")][0] # a arquitetura escolhida pode mudar de uma divisão para outra

    linhas.append({
        "divisao": divisao,
        "arquitetura_mlp": coluna_mlp.replace("mlp_", ""),
        "f1_linha_de_base": metricas.loc["f1_macro", "linha_de_base"],
        "f1_mlp": metricas.loc["f1_macro", coluna_mlp],
        "f1_rf": metricas.loc["f1_macro", "random_forest"],
        "accuracy_mlp": metricas.loc["accuracy", coluna_mlp],
        "accuracy_rf": metricas.loc["accuracy", "random_forest"],
        "dif_treino_teste_mlp": comparacao.loc["diferenca_treino_teste", coluna_mlp],
        "dif_treino_teste_rf": comparacao.loc["diferenca_treino_teste", "random_forest"],
    })

if not linhas: # nenhuma divisão foi encontrada: não há o que comparar
    raise SystemExit("Nenhum resultado encontrado em " + pasta_resultados + ". Rode o etapa3.py antes.")
tabela = pd.DataFrame(linhas).set_index("divisao")

# queda em relação à divisão aleatória: quanto o F1 macro cai quando o teste é numa região ou num período que o modelo nunca viu
# valor negativo = piorou; é a medida da "otimização" da divisão aleatória discutida na etapa 2
if "aleatoria" in tabela.index:
    tabela["queda_f1_mlp"] = (tabela["f1_mlp"] - tabela.loc["aleatoria", "f1_mlp"]).round(3)
    tabela["queda_f1_rf"] = (tabela["f1_rf"] - tabela.loc["aleatoria", "f1_rf"]).round(3)

print(f"Generalização do {problema} nas divisões (métricas no teste):")
print(tabela.round(3).to_string())

tabela.round(3).to_csv(os.path.join(pasta_resultados, f"generalizacao_divisoes_{problema}.csv"), encoding="utf-8-sig")


# ===============================================================
# PASSO 2: GRÁFICO DO F1 MACRO POR DIVISÃO
# ===============================================================
# barras agrupadas: para cada divisão, linha de base x MLP x Random Forest
grafico = tabela[["f1_linha_de_base", "f1_mlp", "f1_rf"]].rename(columns={
    "f1_linha_de_base": "linha de base", "f1_mlp": "MLP", "f1_rf": "Random Forest",
})
eixo = grafico.plot(kind="bar", figsize=(9, 5), color=["lightgray", "steelblue", "seagreen"], width=0.75)
for barras in eixo.containers: # escreve o valor em cima de cada barra
    eixo.bar_label(barras, fmt="%.2f", fontsize=9)
plt.title(f"F1 macro no teste por divisão - {problema}")
plt.xlabel("Divisão dos dados")
plt.ylabel("F1 macro")
plt.xticks(rotation=0)
eixo.set_ylim(0, grafico.to_numpy().max() * 1.15) # deixa 15% de espaço acima da barra mais alta, para os números não encostarem no topo
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3) # legenda embaixo do gráfico, em 3 colunas, para não cobrir as barras
plt.tight_layout()
caminho_grafico = os.path.join(pasta_graficos, f"generalizacao_divisoes_{problema}.png")
plt.savefig(caminho_grafico, dpi=120)
plt.close()
print("\nGráfico salvo:", caminho_grafico)
print("Tabela salva:", os.path.join(pasta_resultados, f"generalizacao_divisoes_{problema}.csv"))