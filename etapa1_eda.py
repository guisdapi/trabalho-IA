# -*- coding: utf-8 -*-
"""
Trabalho de Aprendizado Supervisionado - Inteligência Artificial - UFSM
ETAPA 1 - Análise Exploratória do Dataset (RodoFauna / IBRAM-DF)

Como executar (no terminal, dentro da pasta do projeto):
    python etapa1_eda.py                       (lê o arquivo data/rodofauna.xlsx)
    python etapa1_eda.py caminho/da/base.xlsx  (lê outro arquivo)

O que o programa gera (dentro da pasta resultados_etapa1/):
    resultados_etapa1.md    relatório com texto, tabelas e gráficos
    fig01 ... fig10 .png    gráficos (usados pelo relatório .md)

Itens do enunciado cobertos, na ordem das seções:
    1. Descrição do problema
    2. Quantidade de registros e de atributos, tipos dos atributos
    3. Valores ausentes e 'nao_identificado'
    4. Estatísticas descritivas das variáveis numéricas e outliers
    5. Variável-alvo de cada experimento e distribuição das classes
    6. Distribuição espacial
    7. Distribuição temporal
    8. Atributos redundantes e correlação
    9. Resumo dos problemas encontrados
"""

# =============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================
import sys  # dá acesso aos argumentos digitados no terminal (sys.argv)
import textwrap  # quebra textos longos em várias linhas para o terminal
from pathlib import Path  # manipula caminhos de arquivos e pastas (funciona no Windows e no Linux)

import numpy as np  # cálculos numéricos (raiz quadrada, condições vetorizadas etc.)
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
TAMANHO_DA_CELULA_EM_METROS = 500  # lado da célula espacial (quadrado de 500 m x 500 m)

# Atributos que o enunciado proíbe de usar como entrada (preditores) dos modelos
ATRIBUTOS_PROIBIDOS_COMO_PREDITORES = [
    "objectid", "id", "silvestre_", "hora", "sexo", "posicao", "ordem",
    "familia", "nome_popul", "campanha", "periodo", "mes",
]

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
COR_LARANJA = "#eb6834"  # cor de destaque

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

# Lista dos problemas encontrados ao longo da análise; é impressa na seção 9
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


def registrar_paragrafo(texto):
    """Texto corrido: quebrado em várias linhas no terminal, um parágrafo só no .md."""
    print(textwrap.fill(texto, 78))  # quebra o texto a cada 78 caracteres no terminal
    linhas_do_relatorio.extend([texto, ""])  # no .md fica um parágrafo único seguido de linha vazia


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


def registrar_subtitulo(texto):
    """Subtítulo: texto simples no terminal; '###' no .md."""
    print()  # linha em branco antes
    print(texto)  # subtítulo
    linhas_do_relatorio.extend(["", f"### {texto}", ""])  # '###' = título de nível 3 no Markdown


# Legenda de cada figura no relatório (nome do arquivo -> texto da legenda)
LEGENDAS_DAS_FIGURAS = {
    "fig01_ausentes.png": "Figura 1 - Percentual de valores ausentes/nao_identificado por atributo",
    "fig02_boxplot_coordenadas.png": "Figura 2 - Boxplots das coordenadas x e y",
    "fig03_distribuicao_classe.png": "Figura 3 - Distribuição da variável-alvo classe",
    "fig04_top_especies.png": "Figura 4 - 15 espécies mais atropeladas",
    "fig05_atropelamentos_por_celula.png": "Figura 5 - Atropelamentos por célula de 500 m",
    "fig06_mapa_por_classe.png": "Figura 6 - Distribuição espacial por classe",
    "fig07_densidade_espacial.png": "Figura 7 - Densidade espacial dos atropelamentos",
    "fig08_serie_mensal.png": "Figura 8 - Série mensal de atropelamentos",
    "fig09_mes_e_dia_semana.png": "Figura 9 - Atropelamentos por mês do ano e dia da semana",
    "fig10_correlacoes.png": "Figura 10 - Correlação (Spearman) e associação (V de Cramér)",
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
registrar_paragrafo(
    "Base do Projeto RodoFauna (IBRAM/DF): animais encontrados atropelados em "
    "percursos sistemáticos nas rodovias do DF entre abr/2010 e mar/2015. "
    "Cada linha é uma ocorrência (um animal), com data, classificação taxonômica, "
    "unidade de conservação (uc) e coordenadas UTM (x, y)."
)
registrar_paragrafo(
    "Objetivo: prever o tipo (classe) e a intensidade/ocorrência de atropelamentos "
    "de fauna SILVESTRE e identificar trechos críticos (hotspots)."
)

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
registrar(f"Base bruta: {len(base_bruta)} registros x {base_bruta.shape[1]} atributos")
registrar()
registrar("Tipos dos atributos:")


def classificar_uso_do_atributo(nome_do_atributo):
    """Diz se o atributo pode ser preditor, se é alvo, ou se é proibido pelo enunciado."""
    if nome_do_atributo in ATRIBUTOS_PROIBIDOS_COMO_PREDITORES:
        return "não"  # proibido pelo enunciado
    if nome_do_atributo in ("classe", "especie_nm"):
        return "alvo"  # usado como variável-alvo
    return "sim"  # pode ser usado como entrada


tabela_de_tipos = pd.DataFrame({
    "tipo": base_bruta.dtypes.astype(str),  # tipo de dado de cada coluna (int64, string...)
    "valores_distintos": base_bruta.nunique(),  # quantos valores diferentes cada coluna tem
    "pode_ser_preditor": [classificar_uso_do_atributo(coluna) for coluna in base_bruta.columns],
})
registrar_tabela(tabela_de_tipos)

registrar()
registrar("silvestre_ (base bruta):")
registrar_tabela(base_bruta["silvestre_"].value_counts())  # quantos são silvestres x domésticos

# Mantém só a fauna silvestre (exigência do enunciado); .copy() cria uma tabela independente
base_silvestre = base_bruta[base_bruta["silvestre_"] == "silvestre"].copy()
registrar()
registrar(f"Após filtrar silvestre_ == 'silvestre': {len(base_silvestre)} registros")

# Duplicatas: linhas com todas as colunas iguais, ignorando o objectid (que é só o nº da linha)
colunas_sem_objectid = [coluna for coluna in base_silvestre.columns if coluna != "objectid"]
quantidade_de_duplicatas = base_silvestre.duplicated(subset=colunas_sem_objectid).sum()
registrar(f"Registros duplicados (todas as colunas iguais, exceto objectid): {quantidade_de_duplicatas}")
if "id" in base_silvestre.columns:  # se a coluna id existe, mostra quantos ids diferentes há
    registrar(f"Valores distintos de 'id' na base bruta: {base_bruta['id'].nunique()} "
              f"(em {len(base_bruta)} linhas)")
if quantidade_de_duplicatas > 0:  # se houver duplicatas...
    percentual_de_duplicatas = quantidade_de_duplicatas / len(base_silvestre)
    problemas_encontrados.append(
        f"{quantidade_de_duplicatas} registros silvestres duplicados ({percentual_de_duplicatas:.1%}). "
        "Sem removê-los, todas as contagens ficam infladas."
    )
    # ...remove as cópias, mantendo a primeira ocorrência de cada linha
    base_silvestre = base_silvestre.drop_duplicates(subset=colunas_sem_objectid)
    registrar(f"-> Duplicatas removidas. Base analisada daqui em diante: {len(base_silvestre)} registros")

# Converte a coluna data_dt em data de verdade, para extrair mês, dia do ano etc.
if pd.api.types.is_datetime64_any_dtype(base_silvestre["data_dt"]):  # o Excel já trouxe como data
    base_silvestre["data"] = base_silvestre["data_dt"]
else:  # veio como texto no formato dia/mês/ano com 2 dígitos (ex.: 05/04/10)
    base_silvestre["data"] = pd.to_datetime(
        base_silvestre["data_dt"].astype(str),
        format="%d/%m/%y",  # formato dd/mm/aa
        errors="coerce",  # datas inválidas viram NaT (data ausente) em vez de dar erro
    )
# Atributos temporais derivados da data (usados nas seções 4, 7 e 8)
base_silvestre["numero_do_mes"] = base_silvestre["data"].dt.month  # 1 = janeiro ... 12 = dezembro
base_silvestre["dia_do_ano"] = base_silvestre["data"].dt.dayofyear  # 1 a 366
base_silvestre["numero_do_dia_da_semana"] = base_silvestre["data"].dt.dayofweek  # 0 = segunda ... 6 = domingo
# Período climático do DF: seco de maio a setembro, chuvoso de outubro a abril
base_silvestre["periodo_climatico"] = np.where(
    base_silvestre["numero_do_mes"].between(5, 9),  # condição: mês entre 5 e 9
    "seco",  # valor se a condição for verdadeira
    "chuvoso",  # valor se for falsa
)
registrar(f"Datas que não puderam ser convertidas: {base_silvestre['data'].isna().sum()}")

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
cores_das_barras = [  # laranja = atributo proibido pelo enunciado; azul = permitido
    COR_LARANJA if atributo in ATRIBUTOS_PROIBIDOS_COMO_PREDITORES else COR_AZUL
    for atributo in tabela_de_ausentes.index
]
eixo.barh(tabela_de_ausentes.index, tabela_de_ausentes["percentual_sem_informacao"], color=cores_das_barras)
eixo.invert_yaxis()  # maior valor no topo
for posicao, percentual in enumerate(tabela_de_ausentes["percentual_sem_informacao"]):
    eixo.text(percentual + 0.5, posicao, f"{percentual:.1f}%", va="center", fontsize=9)  # rótulo no fim da barra
eixo.set_xlim(0, 105)  # eixo x de 0% a 105% (folga para o rótulo)
eixo.set_xlabel("% dos registros silvestres (ausente + nao_identificado)")
eixo.set_title("Atributos com informação faltante")
salvar_figura(figura, "fig01_ausentes.png")

# Registra como problema todo atributo com 30% ou mais de informação faltante
for atributo, percentual in tabela_de_ausentes["percentual_sem_informacao"].items():
    if percentual >= 30:
        if atributo in ATRIBUTOS_PROIBIDOS_COMO_PREDITORES:
            observacao = " (já excluído como preditor pelo enunciado)"
        else:
            observacao = ""
        problemas_encontrados.append(f"'{atributo}' sem informação em {percentual:.0f}% dos registros{observacao}.")

# =============================================================================
# 4. ESTATÍSTICAS DESCRITIVAS E OUTLIERS
# =============================================================================
registrar_titulo("4. Estatísticas descritivas e outliers")

variaveis_numericas = ["x", "y", "ano", "numero_do_mes", "dia_do_ano"]
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

# Limites aproximados do DF em coordenadas UTM (fuso 23S), com folga de ~10 km
x_dentro_do_df = base_silvestre["x"].between(135_000, 265_000)
y_dentro_do_df = base_silvestre["y"].between(8_210_000, 8_300_000)
fora_do_distrito_federal = ~(x_dentro_do_df & y_dentro_do_df)  # '~' inverte: True = fora do DF
registrar(f"   Pontos fora dos limites do DF: {fora_do_distrito_federal.sum()}")
if fora_do_distrito_federal.sum() > 0:
    problemas_encontrados.append(
        f"{fora_do_distrito_federal.sum()} pontos com coordenadas fora do DF (provável erro)."
    )

# Boxplots de x e y lado a lado: os pontos isolados fora dos "bigodes" são os outliers
figura, eixos = plt.subplots(1, 2, figsize=(9, 3.5))  # 1 linha, 2 colunas de gráficos
for eixo, coordenada in zip(eixos, ["x", "y"]):  # um boxplot em cada eixo
    sns.boxplot(y=base_silvestre[coordenada], ax=eixo, color=COR_AZUL, width=0.4)
    eixo.set_title(f"Boxplot de {coordenada} (UTM, m)")
    eixo.set_ylabel("")
    eixo.ticklabel_format(style="plain", axis="y")  # mostra números inteiros (sem notação científica)
salvar_figura(figura, "fig02_boxplot_coordenadas.png")

# =============================================================================
# 5. VARIÁVEIS-ALVO E DISTRIBUIÇÃO DAS CLASSES
# =============================================================================
registrar_titulo("5. Variáveis-alvo e distribuição das classes")

# ---- 5.1 Classificação multiclasse: alvo = classe ----
registrar_subtitulo("5.1 Classificação multiclasse -> alvo: 'classe'")
contagem_por_classe = base_silvestre["classe"].value_counts()  # registros de cada classe
registrar_tabela(pd.DataFrame({
    "registros": contagem_por_classe,
    "percentual": (100 * contagem_por_classe / contagem_por_classe.sum()).round(2),
}))
# Classes válidas = todas menos 'nao_identificado'
classes_validas = contagem_por_classe.drop(NAO_IDENTIFICADO, errors="ignore")
razao_de_desbalanceamento = classes_validas.max() / classes_validas.min()  # maior classe / menor classe
classe_majoritaria = classes_validas.idxmax()  # nome da classe com mais registros
proporcao_da_classe_majoritaria = classes_validas.max() / classes_validas.sum()
registrar(f"Razão maior/menor classe: {razao_de_desbalanceamento:.1f} : 1")
# Um "modelo" que sempre responde a classe majoritária já acerta essa proporção:
registrar(f"Acurácia de quem sempre chuta '{classe_majoritaria}': {proporcao_da_classe_majoritaria:.1%}")
if razao_de_desbalanceamento > 3:  # consideramos desbalanceado se a maior for 3x a menor
    problemas_encontrados.append(
        f"'classe' desbalanceada ({razao_de_desbalanceamento:.1f}:1): '{classe_majoritaria}' tem "
        f"{proporcao_da_classe_majoritaria:.0%} dos registros. Acurácia sozinha é enganosa."
    )
if NAO_IDENTIFICADO in contagem_por_classe.index:
    problemas_encontrados.append(
        f"{contagem_por_classe[NAO_IDENTIFICADO]} registro(s) com classe = nao_identificado "
        "(remover do multiclasse)."
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
salvar_figura(figura, "fig03_distribuicao_classe.png")

# ---- 5.2 Espécie como alvo alternativo: verifica classes com poucos exemplos ----
registrar_subtitulo("5.2 Alvo alternativo 'especie_nm' (classes com poucos exemplos)")
especie_identificada = base_silvestre["especie_nm"] != NAO_IDENTIFICADO  # True = espécie conhecida
contagem_por_especie = base_silvestre.loc[especie_identificada, "especie_nm"].value_counts()
registrar(f"Espécies distintas: {len(contagem_por_especie)} | registros sem espécie identificada: "
          f"{(~especie_identificada).sum()}")
for minimo_de_registros in (10, 30, 100):  # quantas espécies têm pelo menos 10, 30 e 100 registros
    registrar(f"   espécies com >= {minimo_de_registros:>3} registros: "
              f"{(contagem_por_especie >= minimo_de_registros).sum()}")
especies_raras = contagem_por_especie < 10  # True = espécie com menos de 10 registros
registrar(f"   espécies com < 10 registros: {especies_raras.sum()} ({especies_raras.mean():.0%} das espécies)")
problemas_encontrados.append(
    f"Espécies: {especies_raras.sum()} das {len(contagem_por_especie)} têm menos de 10 registros -> "
    "classificar por espécie não é viável sem agrupar; por isso o multiclasse usa 'classe'."
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
salvar_figura(figura, "fig04_top_especies.png")

# ---- 5.3 e 5.4 Alvos binário e de regressão (precisam agregar os registros em células) ----
registrar_subtitulo(f"5.3 e 5.4 Alvos binário e de regressão (células de {TAMANHO_DA_CELULA_EM_METROS} m)")
registrar(f"Classificação binária -> alvo: hotspot (1/0) por célula de {TAMANHO_DA_CELULA_EM_METROS} m")
registrar(f"Regressão -> alvo: nº de atropelamentos por célula de {TAMANHO_DA_CELULA_EM_METROS} m por mês")
# Cada ponto recebe o código da célula de 500 m em que cai: divisão inteira (//) da coordenada por 500
coluna_da_celula = (base_silvestre["x"] // TAMANHO_DA_CELULA_EM_METROS).astype(int).astype(str)
linha_da_celula = (base_silvestre["y"] // TAMANHO_DA_CELULA_EM_METROS).astype(int).astype(str)
base_silvestre["celula"] = coluna_da_celula + "_" + linha_da_celula  # ex.: "437_16547"
atropelamentos_por_celula = base_silvestre.groupby("celula").size()  # registros em cada célula
registrar(f"Células de {TAMANHO_DA_CELULA_EM_METROS} m com pelo menos 1 atropelamento: "
          f"{len(atropelamentos_por_celula)}")
registrar(f"Atropelamentos por célula: média = {atropelamentos_por_celula.mean():.1f}, "
          f"mediana = {atropelamentos_por_celula.median():.1f}, "
          f"máximo = {atropelamentos_por_celula.max()}")

# Painel célula x mês: quantos pares (célula, mês) existem e quantos têm zero atropelamentos
mes_de_cada_registro = base_silvestre["data"].dt.to_period("M")  # ex.: 2011-03
quantidade_de_meses = mes_de_cada_registro.nunique()  # meses diferentes no período
pares_celula_mes_com_registro = base_silvestre.groupby(["celula", mes_de_cada_registro]).size()
total_de_pares_celula_mes = len(atropelamentos_por_celula) * quantidade_de_meses  # todas as combinações
percentual_de_zeros = 1 - len(pares_celula_mes_com_registro) / total_de_pares_celula_mes
registrar(f"Painel célula x mês: {len(atropelamentos_por_celula)} células x {quantidade_de_meses} meses = "
          f"{total_de_pares_celula_mes} observações, {percentual_de_zeros:.0%} delas com zero atropelamentos")
registrar("Obs.: a base só tem ocorrências; a classe 0 (sem/baixo risco) do problema binário")
registrar("      precisa ser construída na Etapa 2 e justificada.")
if percentual_de_zeros > 0.7:
    problemas_encontrados.append(
        f"Alvo de regressão célula x mês tem {percentual_de_zeros:.0%} de zeros "
        "(muitos zeros; avaliar agregação maior, ex.: trimestre)."
    )

# Histograma: quantas células têm poucos e quantas têm muitos atropelamentos
figura, eixo = plt.subplots(figsize=(7, 3.8))
eixo.hist(atropelamentos_por_celula, bins=30, color=COR_AZUL, edgecolor="white")  # 30 faixas
eixo.set_xlabel(f"atropelamentos na célula de {TAMANHO_DA_CELULA_EM_METROS} m (2010-2015)")
eixo.set_ylabel("nº de células")
eixo.set_title("Concentração dos atropelamentos por célula")
salvar_figura(figura, "fig05_atropelamentos_por_celula.png")

# =============================================================================
# 6. DISTRIBUIÇÃO ESPACIAL
# =============================================================================
registrar_titulo("6. Distribuição espacial")

registrar("Registros por unidade de conservação (uc):")
registrar_tabela(base_silvestre["uc"].value_counts())
registrar()
registrar("10 células mais críticas:")
dez_celulas_mais_criticas = atropelamentos_por_celula.sort_values(ascending=False).head(10)
registrar_tabela(dez_celulas_mais_criticas.rename_axis("célula (x_y)"))
registrar(f"-> concentram {dez_celulas_mais_criticas.sum() / len(base_silvestre):.1%} dos atropelamentos em "
          f"{10 / len(atropelamentos_por_celula):.1%} das células")

# Base usada nos mapas: sem os pontos fora do DF, para não distorcer a escala
base_para_mapas = base_silvestre[~fora_do_distrito_federal]
classes_no_grafico = [classe for classe in CORES_POR_CLASSE if classe in classes_validas.index]

# Um mapa para cada classe (grade 2 x 2); em cinza ficam todas as ocorrências como referência
figura, eixos = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)  # mesma escala nos 4
for eixo, classe in zip(eixos.ravel(), classes_no_grafico):  # ravel() transforma a grade 2x2 em lista
    eixo.scatter(base_para_mapas["x"], base_para_mapas["y"], s=3, color="#dddddd")  # todas em cinza
    registros_da_classe = base_para_mapas[base_para_mapas["classe"] == classe]
    eixo.scatter(registros_da_classe["x"], registros_da_classe["y"], s=6, color=CORES_POR_CLASSE[classe])
    eixo.set_title(f"{NOME_DA_CLASSE_EM_PORTUGUES[classe]} (n = {len(registros_da_classe)})")
    eixo.set_aspect("equal")  # 1 metro em x = 1 metro em y (mapa sem distorção)
    eixo.ticklabel_format(style="plain")  # coordenadas sem notação científica
    eixo.tick_params(labelsize=7)  # números dos eixos menores
figura.suptitle("Ocorrências por classe (cinza = todas)", fontweight="bold")
salvar_figura(figura, "fig06_mapa_por_classe.png")

# Mapa de densidade: hexágonos mais escuros = mais atropelamentos naquela área
figura, eixo = plt.subplots(figsize=(8, 6))
hexagonos = eixo.hexbin(base_para_mapas["x"], base_para_mapas["y"],
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
salvar_figura(figura, "fig07_densidade_espacial.png")

# =============================================================================
# 7. DISTRIBUIÇÃO TEMPORAL
# =============================================================================
registrar_titulo("7. Distribuição temporal")

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
if atropelamentos_por_dia_da_semana[["sab", "dom"]].sum() == 0:  # nenhum registro no fim de semana
    problemas_encontrados.append(
        "Nenhum registro em sábado/domingo: as coletas só ocorreram em dias úteis, "
        "então 'dia_da_sem' reflete a agenda de coleta, não o risco."
    )
registrar()
registrar("Período climático (seco = mai-set, chuvoso = out-abr):")
contagem_por_periodo_climatico = base_silvestre["periodo_climatico"].value_counts()
registros_no_chuvoso = contagem_por_periodo_climatico.get("chuvoso", 0)
registros_no_seco = contagem_por_periodo_climatico.get("seco", 0)
# Divide pelo nº de meses de cada período (7 chuvosos e 5 secos) para comparar de forma justa
registrar(f"   chuvoso: {registros_no_chuvoso} ({registros_no_chuvoso / 7:.0f} por mês)")
registrar(f"   seco:    {registros_no_seco} ({registros_no_seco / 5:.0f} por mês)")

# Série temporal: total de atropelamentos em cada mês do período
serie_mensal = base_silvestre.groupby(mes_de_cada_registro).size()
figura, eixo = plt.subplots(figsize=(11, 3.8))
eixo.plot(serie_mensal.index.to_timestamp(), serie_mensal.values, color=COR_AZUL, marker="o", markersize=3)
eixo.set_ylabel("registros")
eixo.set_title("Atropelamentos por mês (abr/2010 - mar/2015)")
salvar_figura(figura, "fig08_serie_mensal.png")

# Dois gráficos de barras lado a lado: por mês do ano e por dia da semana
figura, eixos = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})  # esquerdo 2x mais largo
eixos[0].bar(NOMES_DOS_MESES, atropelamentos_por_mes.values,
             color=[COR_LARANJA if 5 <= mes <= 9 else COR_AZUL for mes in range(1, 13)])  # laranja = seca
eixos[0].set_title("Por mês do ano (laranja = estação seca)")
eixos[0].set_ylabel("registros")
eixos[1].bar(NOMES_DOS_DIAS_DA_SEMANA, atropelamentos_por_dia_da_semana.values, color=COR_AZUL)
eixos[1].set_title("Por dia da semana")
for eixo in eixos:
    eixo.grid(axis="x", visible=False)  # remove as linhas de grade verticais
salvar_figura(figura, "fig09_mes_e_dia_semana.png")

# =============================================================================
# 8. ATRIBUTOS REDUNDANTES E CORRELAÇÃO
# =============================================================================
registrar_titulo("8. Atributos redundantes e correlação")

# ---- Campos originais que repetem a informação de data_dt ----
registrar("Concordância dos campos originais com data_dt:")
concordancia_do_ano = (base_silvestre["ano"] == base_silvestre["data"].dt.year).mean()  # % de linhas iguais
registrar(f"   ano        x data_dt: {concordancia_do_ano:.1%}")
# Converte 'seg', 'ter'... da coluna original em 0, 1... para comparar com o dia calculado pela data
numero_do_dia_por_nome = {nome: numero for numero, nome in enumerate(NOMES_DOS_DIAS_DA_SEMANA)}
dia_da_semana_original = base_silvestre["dia_da_sem"].str[:3].map(numero_do_dia_por_nome)
dia_igual_ao_da_data = dia_da_semana_original == base_silvestre["numero_do_dia_da_semana"]
registrar(f"   dia_da_sem x data_dt: {dia_igual_ao_da_data.mean():.1%}  "
          f"({(~dia_igual_ao_da_data).sum()} divergências)")
registrar(f"   valores de dia_da_sem: {base_silvestre['dia_da_sem'].value_counts().to_dict()}")
registrar("-> ano, mes e dia_da_sem são deriváveis de data_dt (redundantes).")
if dia_igual_ao_da_data.mean() < 1:  # se houver qualquer divergência
    problemas_encontrados.append(
        f"'dia_da_sem' diverge da data em {(~dia_igual_ao_da_data).sum()} registros "
        "(erro de digitação) - derivar o dia da semana de data_dt."
    )

# ---- Hierarquia taxonômica: a espécie determina a classe ----
# Se cada espécie tem uma única classe, usar espécie (ou ordem/família) como entrada seria "colar" a resposta
registros_com_especie_e_classe = base_silvestre[
    (base_silvestre["especie_nm"] != NAO_IDENTIFICADO) & (base_silvestre["classe"] != NAO_IDENTIFICADO)
]
classes_por_especie = registros_com_especie_e_classe.groupby("especie_nm")["classe"].nunique()
registrar(f"Espécies associadas a mais de uma classe: {(classes_por_especie > 1).sum()} "
          "-> especie_nm/ordem/familia determinam a classe (não podem ser entrada).")


# ---- uc é praticamente definida pela posição (x, y) ----
def proporcao_da_uc_mais_comum(ucs_da_celula):
    """Fração dos registros da célula que pertencem à uc mais comum dela (1.0 = célula com uma única uc)."""
    return ucs_da_celula.value_counts(normalize=True).iloc[0]


pureza_da_uc_por_celula = base_silvestre.groupby("celula")["uc"].agg(proporcao_da_uc_mais_comum)
registrar(f"Células de {TAMANHO_DA_CELULA_EM_METROS} m com uma única uc: "
          f"{(pureza_da_uc_por_celula == 1).mean():.1%} -> uc é quase redundante com (x, y).")

# ---- Correlação de Spearman entre variáveis numéricas ----
# Spearman mede se duas variáveis crescem juntas (+1), em sentidos opostos (-1) ou não se relacionam (0)
variaveis_para_correlacao = ["x", "y", "ano", "numero_do_mes", "dia_do_ano", "numero_do_dia_da_semana"]
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


variaveis_categoricas = ["classe", "uc", "periodo_climatico", "ano", "numero_do_mes"]
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
salvar_figura(figura, "fig10_correlacoes.png")

# Lista os pares com associação forte (>= 0,7), olhando só a metade de cima de cada matriz
pares_fortemente_associados = []
for posicao, variavel_a in enumerate(variaveis_para_correlacao):
    for variavel_b in variaveis_para_correlacao[posicao + 1:]:  # só pares ainda não vistos
        valor = matriz_de_correlacao.loc[variavel_a, variavel_b]
        if abs(valor) >= 0.7:  # abs = valor absoluto (vale para correlação positiva ou negativa)
            pares_fortemente_associados.append((variavel_a, variavel_b, valor))
for posicao, variavel_a in enumerate(variaveis_categoricas):
    for variavel_b in variaveis_categoricas[posicao + 1:]:
        valor = matriz_v_de_cramer.loc[variavel_a, variavel_b]
        if valor >= 0.7:
            pares_fortemente_associados.append((variavel_a, variavel_b, valor))
if pares_fortemente_associados:
    descricao_dos_pares = "; ".join(
        f"{variavel_a} x {variavel_b} ({valor:.2f})"
        for variavel_a, variavel_b, valor in pares_fortemente_associados
    )
    problemas_encontrados.append(
        f"Pares fortemente associados (>= 0,7): {descricao_dos_pares} -> manter só um de cada par."
    )

# =============================================================================
# 9. RESUMO DOS PROBLEMAS ENCONTRADOS
# =============================================================================
registrar_titulo("9. Resumo dos problemas encontrados")
for numero, problema in enumerate(problemas_encontrados, start=1):  # numera a partir de 1
    registrar(f"{numero}. {problema}")

# Junta todas as linhas do relatório e grava o arquivo .md (UTF-8 para manter os acentos)
caminho_do_relatorio = PASTA_DE_SAIDA / "resultados_etapa1.md"
caminho_do_relatorio.write_text("\n".join(linhas_do_relatorio), encoding="utf-8")
print(f"\nResultados salvos em: {PASTA_DE_SAIDA.resolve()}")