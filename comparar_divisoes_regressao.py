# 4 etapa IA - Comparação das divisões da regressão (generalização)
# roda DEPOIS do etapa4.py ter sido executado nas 3 divisões:
#   python etapa4.py regressao_aleatoria
#   python etapa4.py regressao_espacial
#   python etapa4.py regressao_temporal
#   python comparar_divisoes_regressao.py

# ===============================================================
# IMPORTS
# ===============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
pasta_resultados = "resultadosE4"  # onde o etapa4.py salvou as tabelas (passo 9)
pasta_graficos = "graficosE4"
problema = "regressao"
divisoes = ["aleatoria", "espacial", "temporal"]
modelos = {"linha_de_base": "linha de base", "mlp": "MLP", "regressao_linear": "Regressão linear", "random_forest": "Random Forest"}

os.chdir(os.path.dirname(os.path.abspath(__file__))) # faz os caminhos relativos partirem da pasta do script
os.makedirs(pasta_graficos, exist_ok=True)


# ===============================================================
# PASSO 1: JUNTAR OS RESULTADOS DAS 3 DIVISÕES
# ===============================================================
linhas = []
for divisao in divisoes:
    nome_conjunto = f"{problema}_{divisao}"
    caminho_metricas = os.path.join(pasta_resultados, f"metricas_teste_{nome_conjunto}.csv")
    caminho_comparacao = os.path.join(pasta_resultados, f"comparacao_final_{nome_conjunto}.csv")
    if not (os.path.exists(caminho_metricas) and os.path.exists(caminho_comparacao)):
        print("Faltando resultados de", nome_conjunto, "-> rode antes: python etapa4.py", nome_conjunto)
        continue

    metricas = pd.read_csv(caminho_metricas, index_col=0) # linhas: mae, mse, rmse, r2 | colunas: um modelo cada
    comparacao = pd.read_csv(caminho_comparacao, index_col=0)
    coluna_mlp = [coluna for coluna in metricas.columns if coluna.startswith("mlp_")][0] # a arquitetura pode mudar de uma divisão para outra
    metricas = metricas.rename(columns={coluna_mlp: "mlp"}) # nome fixo, para as colunas da tabela final não dependerem da arquitetura
    comparacao = comparacao.rename(columns={coluna_mlp: "mlp"})

    linha = {"divisao": divisao, "arquitetura_mlp": coluna_mlp.replace("mlp_", "")}
    for modelo in modelos: # rmse e r2 de cada modelo no teste
        linha[f"rmse_{modelo}"] = metricas.loc["rmse", modelo]
        linha[f"r2_{modelo}"] = metricas.loc["r2", modelo]
    for modelo in ["mlp", "regressao_linear", "random_forest"]: # o quanto cada modelo decorou o treino
        linha[f"dif_teste_treino_{modelo}"] = comparacao.loc["diferenca_teste_treino_rmse", modelo]
    linhas.append(linha)

if not linhas:
    raise SystemExit("Nenhum resultado encontrado em " + pasta_resultados + ". Rode o etapa4.py antes.")
tabela = pd.DataFrame(linhas).set_index("divisao")

# variação do R² em relação à divisão aleatória: quanto o modelo perde quando o teste é numa região
# ou num período que ele nunca viu (valor negativo = piorou)
# usamos o R² (e não o RMSE) porque ele não depende da escala do alvo: o teste de cada divisão tem médias diferentes,
# então um RMSE menor numa divisão pode ser só porque aquele teste tem menos atropelamentos
if "aleatoria" in tabela.index:
    for modelo in ["mlp", "regressao_linear", "random_forest"]:
        tabela[f"queda_r2_{modelo}"] = (tabela[f"r2_{modelo}"] - tabela.loc["aleatoria", f"r2_{modelo}"]).round(3)

print("Generalização da regressão nas divisões (métricas no teste):")
print(tabela.round(3).T.to_string()) # .T: uma coluna por divisão, fica mais fácil de ler com tantas métricas

caminho_tabela = os.path.join(pasta_resultados, f"generalizacao_divisoes_{problema}.csv")
tabela.round(3).to_csv(caminho_tabela, encoding="utf-8-sig")


# ===============================================================
# PASSO 2: GRÁFICO DO R² E DO RMSE POR DIVISÃO
# ===============================================================
# à esquerda o R² (comparável entre divisões); à direita o RMSE (o erro em atropelamentos)
# a linha de base tem R² ~0 por definição, então no gráfico da esquerda ela aparece como uma barra quase vazia
cores = ["lightgray", "steelblue", "darkorange", "seagreen"]
figura, eixos = plt.subplots(1, 2, figsize=(15, 5))
for eixo, metrica, titulo in [(eixos[0], "r2", "R² no teste (maior = melhor)"), (eixos[1], "rmse", "RMSE no teste (menor = melhor)")]:
    grafico = tabela[[f"{metrica}_{modelo}" for modelo in modelos]]
    grafico.columns = list(modelos.values())
    grafico.plot(kind="bar", ax=eixo, color=cores, width=0.8, legend=False)
    for barras in eixo.containers: # escreve o valor em cima (ou embaixo, se negativo) de cada barra
        eixo.bar_label(barras, fmt="%.2f", fontsize=8)
    eixo.axhline(0, color="black", linewidth=0.8) # referência do zero (o R² pode ficar negativo)
    eixo.set_title(titulo)
    eixo.set_xlabel("Divisão dos dados")
    eixo.tick_params(axis="x", rotation=0)
    minimo, maximo = grafico.to_numpy().min(), grafico.to_numpy().max()
    eixo.set_ylim(min(minimo * 1.2, 0), max(maximo, 0.1) * 1.15) # espaço para os números, inclusive abaixo do zero
handles, rotulos = eixos[0].get_legend_handles_labels() # pega as cores e nomes de um dos gráficos (são iguais nos dois)
figura.legend(handles, rotulos, loc="lower center", ncol=4) # uma legenda só, embaixo dos dois gráficos
figura.suptitle("Generalização da regressão nas divisões")
plt.tight_layout(rect=[0, 0.07, 1, 1]) # rect deixa 7% livre embaixo para a legenda
caminho_grafico = os.path.join(pasta_graficos, f"generalizacao_divisoes_{problema}.png")
plt.savefig(caminho_grafico, dpi=120)
plt.close()
print("\nGráfico salvo:", caminho_grafico)
print("Tabela salva:", caminho_tabela)