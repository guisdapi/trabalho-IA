import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


pasta_resultados = "resultadosE3"
pasta_graficos = "graficosE3"

problema = sys.argv[1] if len(sys.argv) > 1 else "multiclasse"
divisoes_de_cada_problema = {
    "multiclasse": ["aleatoria", "espacial", "temporal"],
    "hotspot": ["aleatoria", "espacial"],
}
if problema not in divisoes_de_cada_problema:
    raise SystemExit("Problema desconhecido: " + problema + " (use multiclasse ou hotspot)")
divisoes = divisoes_de_cada_problema[problema]

metrica = "f1_macro" if problema == "multiclasse" else "f1"
titulo_metrica = "F1 macro" if problema == "multiclasse" else "F1 da classe hotspot"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)


linhas = []
for divisao in divisoes:
    nome_conjunto = f"{problema}_{divisao}"
    caminho_metricas = os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv")
    caminho_comparacao = os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv")
    if not (os.path.exists(caminho_metricas) and os.path.exists(caminho_comparacao)):
        script = "etapa3_multiclasse.py" if problema == "multiclasse" else "etapa3_binario.py"
        print("Faltando resultados de", nome_conjunto, "-> rode antes: python", script, divisao)
        continue

    metricas = pd.read_csv(caminho_metricas, index_col=0)
    comparacao = pd.read_csv(caminho_comparacao, index_col=0)
    coluna_mlp = [coluna for coluna in metricas.columns if coluna.startswith("mlp_")][0]

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
    if problema == "hotspot":
        linha["roc_auc_mlp"] = metricas.loc["roc_auc", coluna_mlp]
        linha["roc_auc_rf"] = metricas.loc["roc_auc", "random_forest"]
    linhas.append(linha)

if not linhas:
    raise SystemExit("Nenhum resultado encontrado em " + pasta_resultados + ". Rode a etapa 3 antes.")
tabela = pd.DataFrame(linhas).set_index("divisao")

if "aleatoria" in tabela.index:
    tabela["queda_f1_mlp"] = (tabela["f1_mlp"] - tabela.loc["aleatoria", "f1_mlp"]).round(3)
    tabela["queda_f1_rf"] = (tabela["f1_rf"] - tabela.loc["aleatoria", "f1_rf"]).round(3)

print(f"Generalização do {problema} nas divisões (métricas no teste; f1 = {titulo_metrica}):")
print(tabela.round(3).to_string())

caminho_tabela = os.path.join(pasta_resultados, f"generalizacao_divisoes_{problema}.csv")
tabela.round(3).to_csv(caminho_tabela, encoding="utf-8-sig")


grafico = tabela[["f1_linha_de_base", "f1_mlp", "f1_rf"]].rename(columns={
    "f1_linha_de_base": "linha de base", "f1_mlp": "MLP", "f1_rf": "Random Forest",
})
eixo = grafico.plot(kind="bar", figsize=(9, 5), color=["lightgray", "steelblue", "seagreen"], width=0.75)
for barras in eixo.containers:
    eixo.bar_label(barras, fmt="%.2f", fontsize=9)
plt.title(f"{titulo_metrica} no teste por divisão - {problema}")
plt.xlabel("Divisão dos dados")
plt.ylabel(titulo_metrica)
plt.xticks(rotation=0)
eixo.set_ylim(0, max(grafico.to_numpy().max(), 0.1) * 1.15)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()
caminho_grafico = os.path.join(pasta_graficos, f"generalizacao_divisoes_{problema}.png")
plt.savefig(caminho_grafico, dpi=120)
plt.close()
print("\nGráfico salvo:", caminho_grafico)
print("Tabela salva:", caminho_tabela)