# 3 etapa IA - Comparação das divisões (generalização)
# roda DEPOIS da etapa 3 ter sido executada em todas as divisões do problema escolhido:
#   multiclasse (3 divisões):
#     python etapa3_multiclasse.py aleatoria
#     python etapa3_multiclasse.py espacial
#     python etapa3_multiclasse.py temporal
#     python comparar_divisoes.py multiclasse
#   hotspot (2 divisões; não tem temporal, porque cada segmento soma os 5 períodos):
#     python etapa3_binario.py aleatoria
#     python etapa3_binario.py espacial
#     python comparar_divisoes.py hotspot

# ===============================================================
# IMPORTS
# ===============================================================
import os
import sys  # lê o nome do problema digitado no terminal (python comparar_divisoes.py hotspot)
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_resultados = "resultadosE3"  # onde a etapa 3 salvou as tabelas (passo 9)
pasta_graficos = "graficosE3"

# sem nada depois do nome do script, compara o multiclasse (igual era antes)
problema = sys.argv[1] if len(sys.argv) > 1 else "multiclasse"
divisoes_de_cada_problema = {
    "multiclasse": ["aleatoria", "espacial", "temporal"],
    "hotspot": ["aleatoria", "espacial"],
}
if problema not in divisoes_de_cada_problema:
    raise SystemExit("Problema desconhecido: " + problema + " (use multiclasse ou hotspot)")
divisoes = divisoes_de_cada_problema[problema]

# a mesma métrica principal da etapa 3: F1 macro no multiclasse e F1 da classe hotspot na binária
metrica = "f1_macro" if problema == "multiclasse" else "f1"
titulo_metrica = "F1 macro" if problema == "multiclasse" else "F1 da classe hotspot"

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True) # garante a pasta do gráfico, mesmo que ela tenha sido apagada


# ===============================================================
# PASSO 1: JUNTAR OS RESULTADOS DAS DIVISÕES
# ===============================================================
# cada divisão tem o seu metricas_teste_*.csv e comparacao_final_*.csv; aqui viram uma linha cada numa tabela só
linhas = []
for divisao in divisoes:
    nome_conjunto = f"{problema}_{divisao}"
    caminho_metricas = os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv")
    caminho_comparacao = os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv")
    if not (os.path.exists(caminho_metricas) and os.path.exists(caminho_comparacao)):
        script = "etapa3_multiclasse.py" if problema == "multiclasse" else "etapa3_binario.py"
        print("Faltando resultados de", nome_conjunto, "-> rode antes: python", script, divisao)
        continue # pula essa divisão e segue com as outras

    metricas = pd.read_csv(caminho_metricas, index_col=0) # index_col=0: a primeira coluna (accuracy, f1_macro, ...) vira o índice
    comparacao = pd.read_csv(caminho_comparacao, index_col=0)
    coluna_mlp = [coluna for coluna in metricas.columns if coluna.startswith("mlp_")][0] # a arquitetura escolhida pode mudar de uma divisão para outra

    linha = {
        "divisao": divisao,
        "arquitetura_mlp": coluna_mlp.replace("mlp_", ""),
        "f1_linha_de_base": metricas.loc[metrica, "linha_de_base"],
        "f1_mlp": metricas.loc[metrica, coluna_mlp],
        "f1_rf": metricas.loc[metrica, "random_forest"],
        "accuracy_mlp": metricas.loc["accuracy", coluna_mlp],
        "accuracy_rf": metricas.loc["accuracy", "random_forest"],
        "dif_treino_teste_mlp": comparacao.loc["diferenca_treino_teste", coluna_mlp],
        "dif_treino_teste_rf": comparacao.loc["diferenca_treino_teste", "random_forest"],
    }
    if problema == "hotspot": # o ROC-AUC só existe na binária
        linha["roc_auc_mlp"] = metricas.loc["roc_auc", coluna_mlp]
        linha["roc_auc_rf"] = metricas.loc["roc_auc", "random_forest"]
    linhas.append(linha)

if not linhas: # nenhuma divisão foi encontrada: não há o que comparar
    raise SystemExit("Nenhum resultado encontrado em " + pasta_resultados + ". Rode a etapa 3 antes.")
tabela = pd.DataFrame(linhas).set_index("divisao")

# queda em relação à divisão aleatória: quanto o F1 cai quando o teste é numa região ou num período que o modelo nunca viu
# valor negativo = piorou; é a medida da "otimização" da divisão aleatória discutida na etapa 2
if "aleatoria" in tabela.index:
    tabela["queda_f1_mlp"] = (tabela["f1_mlp"] - tabela.loc["aleatoria", "f1_mlp"]).round(3)
    tabela["queda_f1_rf"] = (tabela["f1_rf"] - tabela.loc["aleatoria", "f1_rf"]).round(3)

print(f"Generalização do {problema} nas divisões (métricas no teste; f1 = {titulo_metrica}):")
print(tabela.round(3).to_string())

caminho_tabela = os.path.join(pasta_resultados, f"generalizacao_divisoes_{problema}.csv")
tabela.round(3).to_csv(caminho_tabela, encoding="utf-8-sig")


# ===============================================================
# PASSO 2: GRÁFICO DA MÉTRICA PRINCIPAL POR DIVISÃO
# ===============================================================
# barras agrupadas: para cada divisão, linha de base x MLP x Random Forest
grafico = tabela[["f1_linha_de_base", "f1_mlp", "f1_rf"]].rename(columns={
    "f1_linha_de_base": "linha de base", "f1_mlp": "MLP", "f1_rf": "Random Forest",
})
eixo = grafico.plot(kind="bar", figsize=(9, 5), color=["lightgray", "steelblue", "seagreen"], width=0.75)
for barras in eixo.containers: # escreve o valor em cima de cada barra
    eixo.bar_label(barras, fmt="%.2f", fontsize=9)
plt.title(f"{titulo_metrica} no teste por divisão - {problema}")
plt.xlabel("Divisão dos dados")
plt.ylabel(titulo_metrica)
plt.xticks(rotation=0)
eixo.set_ylim(0, max(grafico.to_numpy().max(), 0.1) * 1.15) # deixa 15% de espaço acima da barra mais alta; o max(..., 0.1) evita limite 0 se todas as barras forem 0
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3) # legenda embaixo do gráfico, em 3 colunas, para não cobrir as barras
plt.tight_layout()
caminho_grafico = os.path.join(pasta_graficos, f"generalizacao_divisoes_{problema}.png")
plt.savefig(caminho_grafico, dpi=120)
plt.close()
print("\nGráfico salvo:", caminho_grafico)
print("Tabela salva:", caminho_tabela)