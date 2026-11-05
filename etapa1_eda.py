# -*- coding: utf-8 -*-
"""
Trabalho de Aprendizado Supervisionado - Inteligência Artificial - UFSM
ETAPA 1 - Análise Exploratória do Dataset (RodoFauna / IBRAM-DF)

Como executar (no terminal, dentro da pasta do projeto):
    python etapa1_eda.py                       (lê o arquivo data/rodofauna.xlsx)
    python etapa1_eda.py caminho/da/base.xlsx  (lê outro arquivo)

O que o programa gera (dentro da pasta resultados_etapa1/):
    resultados_etapa1.md    relatório com texto, tabelas e gráficos
    fig01 ... fig09 .png    gráficos (usados pelo relatório .md)

Seções (seguem a lista da Etapa 1 do enunciado):
    1. Descrição do problema
    2. Quantidade de registros, quantidade de atributos e tipos dos atributos
    3. Quantidade de valores ausentes (e de 'nao_identificado')
    4. Variável-alvo de cada experimento e distribuição das classes
    5. Estatísticas descritivas das variáveis numéricas e outliers
    6. Distribuição espacial dos atropelamentos
    7. Distribuição temporal dos atropelamentos
    8. Atributos redundantes e correlação entre atributos
    9. Resumo dos possíveis problemas encontrados
"""

# =============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================
import sys  # dá acesso aos argumentos digitados no terminal (sys.argv)
from pathlib import Path  # manipula caminhos de arquivos e pastas (funciona no Windows e no Linux)

import numpy as np  # cálculos numéricos (raiz quadrada)
import pandas as pd  # leitura da planilha e manipulação de tabelas (DataFrames)
import matplotlib  # biblioteca base de gráficos

matplotlib.use("Agg")  # faz os gráficos serem salvos em arquivo, sem abrir janelas na tela
import matplotlib.pyplot as plt  # interface para criar figuras e eixos dos gráficos
import seaborn as sns  # gráficos estatísticos prontos (boxplot, mapa de calor)
from scipy.stats import chi2_contingency  # teste qui-quadrado, usado no cálculo do V de Cramér

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
# Se o usuário passou um caminho no terminal (sys.argv[1]), usa ele; senão usa o padrão
if len(sys.argv) > 1:  # sys.argv[0] é o nome do script; sys.argv[1] seria o caminho informado
    CAMINHO_DO_ARQUIVO = Path(sys.argv[1])  # caminho informado pelo usuário
else:
    CAMINHO_DO_ARQUIVO = Path("data/rodofauna.xlsx")  # caminho padrão da base

PASTA_DE_SAIDA = Path("resultados_etapa1")  # pasta onde ficarão o relatório e os gráficos
PASTA_DE_SAIDA.mkdir(exist_ok=True)  # cria a pasta (se já existir, não dá erro)

NAO_IDENTIFICADO = "nao_identificado"  # texto que a base usa quando a informação não foi registrada

# Cor fixa de cada classe taxonômica, para todos os gráficos usarem a mesma cor
CORES_POR_CLASSE = {
    "birds": "#2a78d6",  # aves = azul
    "mammals": "#eb6834",  # mamíferos = laranja
    "reptiles": "#1baf7a",  # répteis = verde-água
    "amphibians": "#eda100",  # anfíbios = amarelo
}
# A base usa os nomes das classes em inglês; este dicionário traduz para os gráficos
NOME_DA_CLASSE_EM_PORTUGUES = {
    "birds": "Aves",
    "mammals": "Mamíferos",
    "reptiles": "Répteis",
    "amphibians": "Anfíbios",
}
NOMES_DOS_MESES = ["jan", "fev", "mar", "abr", "mai", "jun",
                   "jul", "ago", "set", "out", "nov", "dez"]  # rótulos do eixo dos meses
NOMES_DOS_DIAS_DA_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]  # 0 = segunda ... 6 = domingo
COR_AZUL = "#2a78d6"  # cor padrão das barras

# Estilo visual padrão de todos os gráficos
sns.set_theme(style="whitegrid")  # fundo branco com linhas de grade claras
plt.rcParams.update({
    "savefig.dpi": 150,  # resolução das imagens salvas
    "savefig.bbox": "tight",  # corta as bordas brancas sobrando
    "axes.spines.top": False,  # remove a borda de cima do gráfico
    "axes.spines.right": False,  # remove a borda da direita do gráfico
    "axes.titleweight": "bold",  # títulos em negrito
})

# =============================================================================
# FUNÇÕES AUXILIARES (escrevem no terminal e no relatório .md ao mesmo tempo)
# =============================================================================
# Lista com as linhas do relatório em Markdown; no final do programa vira o arquivo .md
linhas_do_relatorio = ["# Etapa 1 - Análise Exploratória do Dataset (RodoFauna / IBRAM-DF)", ""]

# Lista dos possíveis problemas encontrados ao longo da análise; é impressa na seção 9
problemas_encontrados = []


def registrar(texto=""):
    """Mostra uma linha no terminal e a adiciona ao relatório .md."""
    print(texto)  # mostra no terminal
    texto = str(texto)  # garante que é texto (caso venha um número)
    if texto.startswith("   "):  # linhas que começam com 3 espaços são sub-itens...
        linhas_do_relatorio.append("- " + texto.strip())  # ...e viram itens de lista no .md
    elif texto == "":  # linha vazia...
        linhas_do_relatorio.append("")  # ...separa parágrafos no .md
    else:  # linha comum
        linhas_do_relatorio.append(texto + "  ")  # dois espaços no fim = quebra de linha no Markdown


def formatar_valor(valor):
    """Formata um valor para aparecer bonito na tabela do .md."""
    if pd.isna(valor):  # valor ausente (NaN)...
        return "-"  # ...aparece como traço
    if isinstance(valor, float) and valor.is_integer():  # número decimal sem casas (ex.: 2863.0)...
        return str(int(valor))  # ...aparece sem o ".0"
    return str(valor)  # qualquer outro valor aparece como está


def registrar_tabela(tabela, nome_da_coluna_de_valores="registros"):
    """Mostra uma tabela no terminal e a escreve como tabela Markdown no .md."""
    if isinstance(tabela, pd.Series):  # se for uma série (uma coluna só)...
        tabela = tabela.rename(nome_da_coluna_de_valores).to_frame()  # ...vira tabela com nome na coluna
    print(tabela.to_string())  # mostra a tabela completa no terminal
    tabela_com_indice = tabela.reset_index()  # transforma o índice (rótulos das linhas) em coluna
    tabela_com_indice.columns = [  # nomes das colunas como texto; o índice sem nome fica em branco
        "" if str(coluna) == "index" else str(coluna) for coluna in tabela_com_indice.columns
    ]
    linhas_do_relatorio.append("")  # linha vazia antes da tabela (exigência do Markdown)
    linhas_do_relatorio.append("| " + " | ".join(tabela_com_indice.columns) + " |")  # cabeçalho
    linhas_do_relatorio.append("|" + "---|" * len(tabela_com_indice.columns))  # linha separadora
    for linha in tabela_com_indice.itertuples(index=False):  # percorre cada linha da tabela
        linhas_do_relatorio.append("| " + " | ".join(formatar_valor(valor) for valor in linha) + " |")
    linhas_do_relatorio.append("")  # linha vazia depois da tabela


def registrar_titulo(texto):
    """Título de seção: em maiúsculas entre linhas de '=' no terminal; '##' no .md."""
    print()  # linha em branco antes
    print("=" * 78)  # linha de separação
    print(texto.upper())  # título em maiúsculas
    print("=" * 78)  # linha de separação
    linhas_do_relatorio.extend(["", f"## {texto}", ""])  # '##' = título de nível 2 no Markdown


# Legenda de cada figura no relatório (nome do arquivo -> texto da legenda)
LEGENDAS_DAS_FIGURAS = {
    "fig01_ausentes.png": "Figura 1 - Percentual de valores ausentes/nao_identificado por atributo",
    "fig02_distribuicao_classe.png": "Figura 2 - Distribuição da variável-alvo classe",
    "fig03_top_especies.png": "Figura 3 - 15 espécies mais atropeladas",
    "fig04_boxplot_coordenadas.png": "Figura 4 - Boxplots das coordenadas x e y",
    "fig05_mapa_por_classe.png": "Figura 5 - Distribuição espacial por classe",
    "fig06_densidade_espacial.png": "Figura 6 - Densidade espacial dos atropelamentos",
    "fig07_serie_mensal.png": "Figura 7 - Série mensal de atropelamentos",
    "fig08_mes_e_dia_semana.png": "Figura 8 - Atropelamentos por mês do ano e dia da semana",
    "fig09_correlacoes.png": "Figura 9 - Correlação (Spearman) e associação (V de Cramér)",
}


def salvar_figura(figura, nome_do_arquivo):
    """Salva o gráfico em .png, fecha a figura e insere a imagem com legenda no .md."""
    figura.savefig(PASTA_DE_SAIDA / nome_do_arquivo)  # grava a imagem na pasta de saída
    plt.close(figura)  # fecha a figura para liberar memória
    print(f"   [figura salva: {nome_do_arquivo}]")  # avisa no terminal
    legenda = LEGENDAS_DAS_FIGURAS.get(nome_do_arquivo, nome_do_arquivo)  # busca a legenda da figura
    linhas_do_relatorio.extend([
        "",
        f"![{legenda}]({nome_do_arquivo})",  # sintaxe Markdown para inserir imagem
        f"*{legenda}*",  # legenda em itálico abaixo da imagem
        "",
    ])


# =============================================================================
# 1. DESCRIÇÃO DO PROBLEMA
# =============================================================================
registrar_titulo("1. Descrição do problema")
registrar("Base do Projeto RodoFauna (IBRAM/DF): animais encontrados atropelados em percursos "
          "sistemáticos nas rodovias do DF entre abr/2010 e mar/2015.")
registrar("Cada linha é uma ocorrência (um animal), com data, classificação taxonômica, "
          "unidade de conservação (uc) e coordenadas UTM (x, y).")
registrar("Objetivo: prever o tipo e a intensidade/ocorrência de atropelamentos de fauna "
          "SILVESTRE e identificar trechos críticos (hotspots).")

# =============================================================================
# 2. REGISTROS, ATRIBUTOS E TIPOS
# =============================================================================
registrar_titulo("2. Registros, atributos e tipos")

base_bruta = pd.read_excel(CAMINHO_DO_ARQUIVO)  # lê a primeira aba da planilha Excel
# Padroniza os nomes das colunas: sem espaços nas pontas e em minúsculas
base_bruta.columns = [coluna.strip().lower() for coluna in base_bruta.columns]
# Padroniza os textos das colunas de texto (ex.: " Birds " vira "birds")
for coluna in base_bruta.columns:  # percorre todas as colunas
    if not pd.api.types.is_string_dtype(base_bruta[coluna]):  # se a coluna não for de texto...
        continue  # ...pula para a próxima
    base_bruta[coluna] = base_bruta[coluna].astype("string").str.strip().str.lower()

registrar(f"Arquivo: {CAMINHO_DO_ARQUIVO}")
# len() = número de linhas (registros); shape[1] = número de colunas (atributos)
registrar(f"Quantidade de registros: {len(base_bruta)}")
registrar(f"Quantidade de atributos: {base_bruta.shape[1]}")
registrar()
registrar("Tipos dos atributos:")
tabela_de_tipos = pd.DataFrame({
    "tipo": base_bruta.dtypes.astype(str),  # tipo de dado de cada coluna (int64, string...)
    "valores_distintos": base_bruta.nunique(),  # quantos valores diferentes cada coluna tem
})
registrar_tabela(tabela_de_tipos)

registrar()
registrar("Registros silvestres x domésticos:")
registrar_tabela(base_bruta["silvestre_"].value_counts())  # quantos são silvestres x domésticos

# O enunciado pede para usar apenas fauna silvestre; .copy() cria uma tabela independente
base_silvestre = base_bruta[base_bruta["silvestre_"] == "silvestre"].copy()
registrar()
registrar(f"Registros de fauna silvestre (usados no restante da análise): {len(base_silvestre)}")

# Duplicatas: linhas com todas as colunas iguais, ignorando o objectid (que é só o nº da linha)
colunas_sem_objectid = [coluna for coluna in base_silvestre.columns if coluna != "objectid"]
quantidade_de_duplicatas = base_silvestre.duplicated(subset=colunas_sem_objectid).sum()
registrar(f"Registros duplicados (todas as colunas iguais, exceto objectid): {quantidade_de_duplicatas}")
if quantidade_de_duplicatas > 0:  # se houver duplicatas...
    problemas_encontrados.append(
        f"Registros duplicados: {quantidade_de_duplicatas} "
        f"({quantidade_de_duplicatas / len(base_silvestre):.1%} dos registros silvestres)."
    )
    # ...as cópias são retiradas desta análise, senão todas as contagens ficariam dobradas
    base_silvestre = base_silvestre.drop_duplicates(subset=colunas_sem_objectid)
    registrar(f"-> Sem as duplicatas, a análise segue com {len(base_silvestre)} registros")

# Converte a coluna data_dt em data, para contar atropelamentos por ano, mês e dia da semana
if pd.api.types.is_datetime64_any_dtype(base_silvestre["data_dt"]):  # o Excel já trouxe como data
    base_silvestre["data"] = base_silvestre["data_dt"]
else:  # veio como texto no formato dia/mês/ano com 2 dígitos (ex.: 05/04/10)
    base_silvestre["data"] = pd.to_datetime(
        base_silvestre["data_dt"].astype(str),
        format="%d/%m/%y",  # formato dd/mm/aa
        errors="coerce",  # datas inválidas viram NaT (data ausente) em vez de dar erro
    )
base_silvestre["numero_do_mes"] = base_silvestre["data"].dt.month  # 1 = janeiro ... 12 = dezembro
base_silvestre["numero_do_dia_da_semana"] = base_silvestre["data"].dt.dayofweek  # 0 = segunda ... 6 = domingo

# =============================================================================
# 3. VALORES AUSENTES E 'nao_identificado'
# =============================================================================
registrar_titulo("3. Valores ausentes e 'nao_identificado'")

colunas_originais = list(base_bruta.columns)  # só as colunas da planilha (sem as que criamos)
tabela_de_ausentes = pd.DataFrame({
    "ausentes": base_silvestre[colunas_originais].isna().sum(),  # células vazias por coluna
    "nao_identificado": (base_silvestre[colunas_originais] == NAO_IDENTIFICADO).sum(),  # 'nao_identificado' por coluna
})
# Percentual de registros sem informação = (vazios + nao_identificado) / total
tabela_de_ausentes["percentual_sem_informacao"] = (
    100 * (tabela_de_ausentes["ausentes"] + tabela_de_ausentes["nao_identificado"]) / len(base_silvestre)
)
tabela_de_ausentes = tabela_de_ausentes.sort_values("percentual_sem_informacao", ascending=False)  # maior primeiro
tabela_de_ausentes = tabela_de_ausentes[tabela_de_ausentes["percentual_sem_informacao"] > 0]  # só os que têm falta
if len(tabela_de_ausentes) > 0:
    registrar_tabela(tabela_de_ausentes.round(2))  # arredonda para 2 casas decimais
else:
    registrar("Nenhum valor ausente.")

# Gráfico de barras horizontais com o percentual sem informação de cada atributo
figura, eixo = plt.subplots(figsize=(8, 0.45 * len(tabela_de_ausentes) + 1.2))  # altura cresce com o nº de barras
eixo.barh(tabela_de_ausentes.index, tabela_de_ausentes["percentual_sem_informacao"], color=COR_AZUL)
eixo.invert_yaxis()  # maior valor no topo
for posicao, percentual in enumerate(tabela_de_ausentes["percentual_sem_informacao"]):
    eixo.text(percentual + 0.5, posicao, f"{percentual:.1f}%", va="center", fontsize=9)  # rótulo no fim da barra
eixo.set_xlim(0, 105)  # eixo x de 0% a 105% (folga para o rótulo)
eixo.set_xlabel("% dos registros silvestres (ausente + nao_identificado)")
eixo.set_title("Atributos com informação faltante")
salvar_figura(figura, "fig01_ausentes.png")

# Considera "grande quantidade" quando 30% ou mais dos registros não têm a informação
for atributo, percentual in tabela_de_ausentes["percentual_sem_informacao"].items():
    if percentual >= 30:
        problemas_encontrados.append(f"'{atributo}' sem informação em {percentual:.0f}% dos registros.")

# =============================================================================
# 4. VARIÁVEL-ALVO DE CADA EXPERIMENTO E DISTRIBUIÇÃO DAS CLASSES
# =============================================================================
registrar_titulo("4. Variável-alvo de cada experimento e distribuição das classes")

registrar("Variável-alvo de cada experimento:")
registrar("   Classificação multiclasse: 'classe' (aves, mamíferos, répteis, anfíbios)")
registrar("   Classificação binária: alto/baixo risco de atropelamento por segmento da rodovia")
registrar("   Regressão: número de atropelamentos por segmento da rodovia (e por período)")
registrar("Os alvos binário e de regressão não existem prontos na base: são construídos "
          "agregando os registros por segmento, o que é feito nas próximas etapas.")
registrar()

# ---- Distribuição das classes da variável 'classe' ----
registrar("Distribuição da variável 'classe':")
contagem_por_classe = base_silvestre["classe"].value_counts()  # registros de cada classe
registrar_tabela(pd.DataFrame({
    "registros": contagem_por_classe,
    "percentual": (100 * contagem_por_classe / contagem_por_classe.sum()).round(2),
}))
# Classes válidas = todas menos 'nao_identificado'
classes_validas = contagem_por_classe.drop(NAO_IDENTIFICADO, errors="ignore")
razao_de_desbalanceamento = classes_validas.max() / classes_validas.min()  # maior classe / menor classe
registrar(f"Razão entre a maior e a menor classe: {razao_de_desbalanceamento:.1f} : 1")
if razao_de_desbalanceamento > 3:  # consideramos desbalanceado se a maior for mais de 3x a menor
    problemas_encontrados.append(
        f"Classes desbalanceadas: '{classes_validas.idxmax()}' tem "
        f"{classes_validas.max() / classes_validas.sum():.0%} dos registros e a maior classe é "
        f"{razao_de_desbalanceamento:.1f} vezes a menor."
    )

# Gráfico de barras da distribuição das classes
figura, eixo = plt.subplots(figsize=(7, 4))
eixo.bar(
    [NOME_DA_CLASSE_EM_PORTUGUES.get(classe, classe) for classe in classes_validas.index],  # nomes em português
    classes_validas.values,  # altura das barras
    color=[CORES_POR_CLASSE.get(classe, "gray") for classe in classes_validas.index],  # cor de cada classe
)
for posicao, quantidade in enumerate(classes_validas.values):  # escreve o valor em cima de cada barra
    eixo.text(posicao, quantidade, f"{quantidade}\n({quantidade / classes_validas.sum():.1%})",
              ha="center", va="bottom", fontsize=9)
eixo.set_ylim(0, classes_validas.max() * 1.2)  # folga no topo para os rótulos
eixo.set_ylabel("registros")
eixo.set_title("Distribuição da variável-alvo 'classe'")
eixo.grid(axis="x", visible=False)  # remove as linhas de grade verticais
salvar_figura(figura, "fig02_distribuicao_classe.png")

# ---- Espécies: verifica classes com quantidade insuficiente de exemplos ----
registrar()
registrar("Distribuição da variável 'especie_nm' (possível alvo alternativo):")
especie_identificada = base_silvestre["especie_nm"] != NAO_IDENTIFICADO  # True = espécie conhecida
contagem_por_especie = base_silvestre.loc[especie_identificada, "especie_nm"].value_counts()
registrar(f"   espécies distintas: {len(contagem_por_especie)}")
for minimo_de_registros in (10, 30, 100):  # quantas espécies têm pelo menos 10, 30 e 100 registros
    registrar(f"   espécies com pelo menos {minimo_de_registros} registros: "
              f"{(contagem_por_especie >= minimo_de_registros).sum()}")
especies_com_poucos_exemplos = contagem_por_especie < 10  # True = espécie com menos de 10 registros
registrar(f"   espécies com menos de 10 registros: {especies_com_poucos_exemplos.sum()}")
problemas_encontrados.append(
    f"Classes com poucos exemplos: {especies_com_poucos_exemplos.sum()} das {len(contagem_por_especie)} "
    "espécies têm menos de 10 registros."
)

# 15 espécies mais frequentes, guardando também a classe de cada uma (para colorir as barras)
especies_mais_frequentes = (
    base_silvestre[especie_identificada]
    .groupby("especie_nm")["classe"]  # agrupa por espécie, olhando a coluna classe
    .agg(quantidade="size", classe="first")  # conta os registros e pega a classe da espécie
    .sort_values("quantidade", ascending=False)  # ordena da mais para a menos frequente
    .head(15)  # fica com as 15 primeiras
)
figura, eixo = plt.subplots(figsize=(8, 5))
eixo.barh(
    especies_mais_frequentes.index.str.replace("_", " "),  # nomes sem underline
    especies_mais_frequentes["quantidade"],
    color=[CORES_POR_CLASSE.get(classe, "gray") for classe in especies_mais_frequentes["classe"]],
)
eixo.invert_yaxis()  # mais frequente no topo
eixo.set_xlabel("registros")
eixo.set_title("15 espécies mais atropeladas (cor = classe)")
eixo.legend(  # legenda manual: um quadradinho colorido para cada classe
    handles=[plt.Rectangle((0, 0), 1, 1, color=cor, label=NOME_DA_CLASSE_EM_PORTUGUES[classe])
             for classe, cor in CORES_POR_CLASSE.items()],
    loc="lower right",
)
salvar_figura(figura, "fig03_top_especies.png")

# =============================================================================
# 5. ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS NUMÉRICAS E OUTLIERS
# =============================================================================
registrar_titulo("5. Estatísticas descritivas das variáveis numéricas")

variaveis_numericas = ["x", "y", "ano"]  # atributos numéricos com significado (id e objectid são só códigos)
# describe() calcula contagem, média, desvio-padrão, mínimo, quartis e máximo; .T vira a tabela de lado
estatisticas_descritivas = base_silvestre[variaveis_numericas].describe().T.round(2)
estatisticas_descritivas.columns = ["quantidade", "média", "desvio_padrão", "mínimo",
                                    "1º_quartil", "mediana", "3º_quartil", "máximo"]
registrar_tabela(estatisticas_descritivas)

registrar()
registrar("Outliers em x e y (critério de Tukey: fora de [Q1 - 1,5*IQR, Q3 + 1,5*IQR]):")
for coordenada in ["x", "y"]:
    primeiro_quartil, terceiro_quartil = base_silvestre[coordenada].quantile([0.25, 0.75])  # Q1 e Q3
    intervalo_interquartil = terceiro_quartil - primeiro_quartil  # IQR = Q3 - Q1
    limite_inferior = primeiro_quartil - 1.5 * intervalo_interquartil  # abaixo disso é outlier
    limite_superior = terceiro_quartil + 1.5 * intervalo_interquartil  # acima disso é outlier
    quantidade_de_outliers = (
        (base_silvestre[coordenada] < limite_inferior) | (base_silvestre[coordenada] > limite_superior)
    ).sum()  # conta os registros fora dos limites
    registrar(f"   {coordenada}: {quantidade_de_outliers} outliers")
    if quantidade_de_outliers > 0:
        problemas_encontrados.append(f"{quantidade_de_outliers} outliers na coordenada {coordenada}.")

# Boxplots de x e y lado a lado: os pontos isolados fora dos "bigodes" são os outliers
figura, eixos = plt.subplots(1, 2, figsize=(9, 3.5))  # 1 linha, 2 colunas de gráficos
for eixo, coordenada in zip(eixos, ["x", "y"]):  # um boxplot em cada eixo
    sns.boxplot(y=base_silvestre[coordenada], ax=eixo, color=COR_AZUL, width=0.4)
    eixo.set_title(f"Boxplot de {coordenada} (UTM, m)")
    eixo.set_ylabel("")
    eixo.ticklabel_format(style="plain", axis="y")  # mostra números inteiros (sem notação científica)
salvar_figura(figura, "fig04_boxplot_coordenadas.png")

# =============================================================================
# 6. DISTRIBUIÇÃO ESPACIAL DOS ATROPELAMENTOS
# =============================================================================
registrar_titulo("6. Distribuição espacial dos atropelamentos")

registrar("Registros por unidade de conservação (uc):")
registrar_tabela(base_silvestre["uc"].value_counts())

# Um mapa para cada classe (grade 2 x 2); em cinza ficam todas as ocorrências como referência
classes_no_grafico = [classe for classe in CORES_POR_CLASSE if classe in classes_validas.index]
figura, eixos = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)  # mesma escala nos 4
for eixo, classe in zip(eixos.ravel(), classes_no_grafico):  # ravel() transforma a grade 2x2 em lista
    eixo.scatter(base_silvestre["x"], base_silvestre["y"], s=3, color="#dddddd")  # todas em cinza
    registros_da_classe = base_silvestre[base_silvestre["classe"] == classe]
    eixo.scatter(registros_da_classe["x"], registros_da_classe["y"], s=6, color=CORES_POR_CLASSE[classe])
    eixo.set_title(f"{NOME_DA_CLASSE_EM_PORTUGUES[classe]} (n = {len(registros_da_classe)})")
    eixo.set_aspect("equal")  # 1 metro em x = 1 metro em y (mapa sem distorção)
    eixo.ticklabel_format(style="plain")  # coordenadas sem notação científica
    eixo.tick_params(labelsize=7)  # números dos eixos menores
figura.suptitle("Ocorrências por classe (cinza = todas)", fontweight="bold")
salvar_figura(figura, "fig05_mapa_por_classe.png")

# Mapa de densidade: hexágonos mais escuros = mais atropelamentos naquela área
figura, eixo = plt.subplots(figsize=(8, 6))
hexagonos = eixo.hexbin(base_silvestre["x"], base_silvestre["y"],
                        gridsize=50,  # 50 hexágonos na largura
                        cmap="Blues",  # escala de azul
                        mincnt=1)  # só desenha hexágonos com pelo menos 1 registro
figura.colorbar(hexagonos, ax=eixo, label="atropelamentos")  # barra de cores ao lado
eixo.set_aspect("equal")
eixo.ticklabel_format(style="plain")
eixo.tick_params(axis="x", rotation=30)  # gira os números do eixo x para não sobreporem
eixo.set_xlabel("x (UTM)")
eixo.set_ylabel("y (UTM)")
eixo.set_title("Densidade espacial dos atropelamentos")
salvar_figura(figura, "fig06_densidade_espacial.png")

# =============================================================================
# 7. DISTRIBUIÇÃO TEMPORAL DOS ATROPELAMENTOS
# =============================================================================
registrar_titulo("7. Distribuição temporal dos atropelamentos")

registrar(f"Período: {base_silvestre['data'].min():%d/%m/%Y} a {base_silvestre['data'].max():%d/%m/%Y}")
registrar("(2010 começa em abril e 2015 termina em março: anos incompletos)")
registrar()
registrar("Registros por ano:")
registrar_tabela(base_silvestre["data"].dt.year.value_counts().sort_index().rename_axis("ano"))
registrar()
registrar("Registros por mês do ano:")
# reindex garante que os 12 meses apareçam, na ordem, mesmo que algum tenha zero
atropelamentos_por_mes = base_silvestre["numero_do_mes"].value_counts().reindex(range(1, 13), fill_value=0)
atropelamentos_por_mes.index = NOMES_DOS_MESES  # troca 1..12 por jan..dez
registrar_tabela(atropelamentos_por_mes.rename_axis("mês"))
registrar()
registrar("Registros por dia da semana:")
atropelamentos_por_dia_da_semana = (
    base_silvestre["numero_do_dia_da_semana"].value_counts().reindex(range(7), fill_value=0)
)
atropelamentos_por_dia_da_semana.index = NOMES_DOS_DIAS_DA_SEMANA  # troca 0..6 por seg..dom
registrar_tabela(atropelamentos_por_dia_da_semana.rename_axis("dia"))

# Série temporal: total de atropelamentos em cada mês do período
mes_de_cada_registro = base_silvestre["data"].dt.to_period("M")  # ex.: 2011-03
serie_mensal = base_silvestre.groupby(mes_de_cada_registro).size()
figura, eixo = plt.subplots(figsize=(11, 3.8))
eixo.plot(serie_mensal.index.to_timestamp(), serie_mensal.values, color=COR_AZUL, marker="o", markersize=3)
eixo.set_ylabel("registros")
eixo.set_title("Atropelamentos por mês (abr/2010 - mar/2015)")
salvar_figura(figura, "fig07_serie_mensal.png")

# Dois gráficos de barras lado a lado: por mês do ano e por dia da semana
figura, eixos = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})  # esquerdo 2x mais largo
eixos[0].bar(NOMES_DOS_MESES, atropelamentos_por_mes.values, color=COR_AZUL)
eixos[0].set_title("Por mês do ano")
eixos[0].set_ylabel("registros")
eixos[1].bar(NOMES_DOS_DIAS_DA_SEMANA, atropelamentos_por_dia_da_semana.values, color=COR_AZUL)
eixos[1].set_title("Por dia da semana")
for eixo in eixos:
    eixo.grid(axis="x", visible=False)  # remove as linhas de grade verticais
salvar_figura(figura, "fig08_mes_e_dia_semana.png")

# =============================================================================
# 8. ATRIBUTOS REDUNDANTES E CORRELAÇÃO ENTRE ATRIBUTOS
# =============================================================================
registrar_titulo("8. Atributos redundantes e correlação entre atributos")

# ---- Atributos redundantes: ano, mes e dia_da_sem repetem a informação de data_dt ----
registrar("Atributos que repetem a informação de data_dt:")
ano_igual_ao_da_data = base_silvestre["ano"] == base_silvestre["data"].dt.year  # True = mesmo ano
registrar(f"   ano igual ao ano de data_dt em {ano_igual_ao_da_data.mean():.1%} dos registros")
# Converte 'seg', 'ter'... da coluna original em 0, 1... para comparar com o dia calculado pela data
numero_do_dia_por_nome = {nome: numero for numero, nome in enumerate(NOMES_DOS_DIAS_DA_SEMANA)}
dia_da_semana_original = base_silvestre["dia_da_sem"].str[:3].map(numero_do_dia_por_nome)
dia_igual_ao_da_data = dia_da_semana_original == base_silvestre["numero_do_dia_da_semana"]
registrar(f"   dia_da_sem igual ao dia de data_dt em {dia_igual_ao_da_data.mean():.1%} dos registros "
          f"({(~dia_igual_ao_da_data).sum()} divergências)")
problemas_encontrados.append("Atributos redundantes: 'ano', 'mes' e 'dia_da_sem' repetem a informação de 'data_dt'.")
if (~dia_igual_ao_da_data).sum() > 0:  # se houver qualquer divergência
    problemas_encontrados.append(
        f"'dia_da_sem' não confere com 'data_dt' em {(~dia_igual_ao_da_data).sum()} registros."
    )

# ---- Atributos redundantes: hierarquia taxonômica (espécie -> família -> ordem -> classe) ----
registros_com_especie_e_classe = base_silvestre[
    (base_silvestre["especie_nm"] != NAO_IDENTIFICADO) & (base_silvestre["classe"] != NAO_IDENTIFICADO)
]
classes_por_especie = registros_com_especie_e_classe.groupby("especie_nm")["classe"].nunique()
registrar()
registrar("Hierarquia taxonômica (espécie -> família -> ordem -> classe):")
registrar(f"   espécies associadas a mais de uma classe: {(classes_por_especie > 1).sum()}")
if (classes_por_especie > 1).sum() == 0:  # cada espécie pertence a uma única classe
    problemas_encontrados.append(
        "Atributos redundantes: 'especie_nm', 'nome_popul', 'familia' e 'ordem' determinam a 'classe'."
    )

# ---- Correlação de Spearman entre variáveis numéricas ----
# Spearman mede se duas variáveis crescem juntas (+1), em sentidos opostos (-1) ou não se relacionam (0)
variaveis_para_correlacao = ["x", "y", "ano", "numero_do_mes", "numero_do_dia_da_semana"]
matriz_de_correlacao = base_silvestre[variaveis_para_correlacao].corr(method="spearman")
registrar()
registrar("Correlação de Spearman (numéricas):")
registrar_tabela(matriz_de_correlacao.round(2))


# ---- V de Cramér entre variáveis categóricas ----
def calcular_v_de_cramer(variavel_a, variavel_b):
    """Associação entre duas variáveis categóricas: 0 = nenhuma, 1 = uma determina a outra."""
    tabela_de_contingencia = pd.crosstab(variavel_a, variavel_b)  # contagem de cada combinação de valores
    estatistica_qui_quadrado = chi2_contingency(tabela_de_contingencia)[0]  # [0] = valor do qui-quadrado
    total_de_registros = tabela_de_contingencia.values.sum()
    menor_dimensao = min(tabela_de_contingencia.shape) - 1  # min(linhas, colunas) - 1
    return np.sqrt(estatistica_qui_quadrado / (total_de_registros * menor_dimensao))  # fórmula do V de Cramér


variaveis_categoricas = ["classe", "uc", "ano", "mes", "dia_da_sem"]  # atributos originais da base
matriz_v_de_cramer = pd.DataFrame(index=variaveis_categoricas, columns=variaveis_categoricas, dtype=float)
for variavel_linha in variaveis_categoricas:  # calcula o V para cada par de variáveis
    for variavel_coluna in variaveis_categoricas:
        if variavel_linha == variavel_coluna:
            matriz_v_de_cramer.loc[variavel_linha, variavel_coluna] = 1.0  # variável com ela mesma = 1
        else:
            matriz_v_de_cramer.loc[variavel_linha, variavel_coluna] = calcular_v_de_cramer(
                base_silvestre[variavel_linha].astype(str), base_silvestre[variavel_coluna].astype(str)
            )
registrar()
registrar("V de Cramér (categóricas):")
registrar_tabela(matriz_v_de_cramer.round(2))

# Mapas de calor das duas matrizes, lado a lado
figura, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
figura.subplots_adjust(wspace=0.4)  # espaço entre os dois gráficos
sns.heatmap(matriz_de_correlacao, annot=True, fmt=".2f",  # annot = escreve o valor em cada quadrado
            cmap="RdBu_r", vmin=-1, vmax=1, ax=eixos[0])  # vermelho = positiva, azul = negativa
eixos[0].set_title("Spearman (numéricas)")
sns.heatmap(matriz_v_de_cramer, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=eixos[1])
eixos[1].set_title("V de Cramér (categóricas)")
salvar_figura(figura, "fig09_correlacoes.png")

# Lista os pares com associação elevada (>= 0,7), olhando só a metade de cima de cada matriz
pares_com_associacao_elevada = []
for posicao, variavel_a in enumerate(variaveis_para_correlacao):
    for variavel_b in variaveis_para_correlacao[posicao + 1:]:  # só pares ainda não vistos
        valor = matriz_de_correlacao.loc[variavel_a, variavel_b]
        if abs(valor) >= 0.7:  # abs = valor absoluto (vale para correlação positiva ou negativa)
            pares_com_associacao_elevada.append((variavel_a, variavel_b, valor))
for posicao, variavel_a in enumerate(variaveis_categoricas):
    for variavel_b in variaveis_categoricas[posicao + 1:]:
        valor = matriz_v_de_cramer.loc[variavel_a, variavel_b]
        if valor >= 0.7:
            pares_com_associacao_elevada.append((variavel_a, variavel_b, valor))
if pares_com_associacao_elevada:
    descricao_dos_pares = "; ".join(
        f"{variavel_a} x {variavel_b} ({valor:.2f})"
        for variavel_a, variavel_b, valor in pares_com_associacao_elevada
    )
    problemas_encontrados.append(f"Correlação elevada (>= 0,7) entre: {descricao_dos_pares}.")

# =============================================================================
# 9. RESUMO DOS POSSÍVEIS PROBLEMAS ENCONTRADOS
# =============================================================================
registrar_titulo("9. Resumo dos possíveis problemas encontrados")
for numero, problema in enumerate(problemas_encontrados, start=1):  # numera a partir de 1
    registrar(f"{numero}. {problema}")

# Junta todas as linhas do relatório e grava o arquivo .md (UTF-8 para manter os acentos)
caminho_do_relatorio = PASTA_DE_SAIDA / "resultados_etapa1.md"
caminho_do_relatorio.write_text("\n".join(linhas_do_relatorio), encoding="utf-8")
print(f"\nResultados salvos em: {PASTA_DE_SAIDA.resolve()}")