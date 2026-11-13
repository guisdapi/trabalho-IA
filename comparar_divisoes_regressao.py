import os
import pandas as pd
import matplotlib.pyplot as plt


pasta_resultados = "resultadosE4"
pasta_graficos = "graficosE4"
problema = "regressao"
divisoes = ["aleatoria", "espacial", "temporal"]
modelos = {"linha_de_base": "linha de base", "mlp": "MLP", "regressao_linear": "Regressão linear", "random_forest": "Random Forest"}

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(pasta_graficos, exist_ok=True)


linhas = []
for divisao in divisoes:
    nome_conjunto = f"{problema}_{divisao}"
    caminho_metricas = os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv")
    caminho_comparacao = os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv")
    if not (os.path.exists(caminho_metricas) and os.path.exists(caminho_comparacao)):
        print("Faltando resultados de", nome_conjunto, "-> rode antes: python etapa4.py", nome_conjunto)
        continue

    metricas = pd.read_csv(caminho_metricas, index_col=0)
    comparacao = pd.read_csv(caminho_comparacao, index_col=0)
    coluna_mlp = [coluna for coluna in metricas.columns if coluna.startswith("mlp_")][0]
    metricas = metricas.rename(columns={coluna_mlp: "mlp"})
    comparacao = comparacao.rename(columns={coluna_mlp: "mlp"})

    linha = {"divisao": divisao, "arquitetura_mlp": coluna_mlp.replace("mlp_", "")}
    for modelo in modelos:
        linha[f"rmse_{modelo}"] = metricas.loc["rmse", modelo]
        linha[f"r2_{modelo}"] = metricas.loc["r2", modelo]
    for modelo in ["mlp", "regressao_linear", "random_forest"]:
        linha[f"dif_teste_treino_{modelo}"] = comparacao.loc["diferenca_teste_treino_rmse", modelo]
    linhas.append(linha)

if not linhas:
    raise SystemExit("Nenhum resultado encontrado em " + pasta_resultados + ". Rode o etapa4.py antes.")
tabela = pd.DataFrame(linhas).set_index("divisao")

if "aleatoria" in tabela.index:
    for modelo in ["mlp", "regressao_linear", "random_forest"]:
        tabela[f"queda_r2_{modelo}"] = (tabela[f"r2_{modelo}"] - tabela.loc["aleatoria", f"r2_{modelo}"]).round(3)

print("Generalização da regressão nas divisões (métricas no teste):")
print(tabela.round(3).T.to_string())

caminho_tabela = os.path.join(pasta_resultados, f"generalizacao_divisoes_{problema}.csv")
tabela.round(3).to_csv(caminho_tabela, encoding="utf-8-sig")


cores = ["lightgray", "steelblue", "darkorange", "seagreen"]
figura, eixos = plt.subplots(1, 2, figsize=(15, 5))
for eixo, metrica, titulo in [(eixos[0], "r2", "R² no teste (maior = melhor)"), (eixos[1], "rmse", "RMSE no teste (menor = melhor)")]:
    grafico = tabela[[f"{metrica}_{modelo}" for modelo in modelos]]
    grafico.columns = list(modelos.values())
    grafico.plot(kind="bar", ax=eixo, color=cores, width=0.8, legend=False)
    for barras in eixo.containers:
        eixo.bar_label(barras, fmt="%.2f", fontsize=8)
    eixo.axhline(0, color="black", linewidth=0.8)
    eixo.set_title(titulo)
    eixo.set_xlabel("Divisão dos dados")
    eixo.tick_params(axis="x", rotation=0)
    minimo, maximo = grafico.to_numpy().min(), grafico.to_numpy().max()
    eixo.set_ylim(min(minimo * 1.2, 0), max(maximo, 0.1) * 1.15)
handles, rotulos = eixos[0].get_legend_handles_labels()
figura.legend(handles, rotulos, loc="lower center", ncol=4)
figura.suptitle("Generalização da regressão nas divisões")
plt.tight_layout(rect=[0, 0.07, 1, 1])
caminho_grafico = os.path.join(pasta_graficos, f"generalizacao_divisoes_{problema}.png")
plt.savefig(caminho_grafico, dpi=120)
plt.close()
print("\nGráfico salvo:", caminho_grafico)
print("Tabela salva:", caminho_tabela)