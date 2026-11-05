"""
Etapa 1 - Análise Exploratória do Dataset RodoFauna (IBRAM/DF)

Lê o arquivo CSV exportado do serviço GIS, analisa apenas os registros de
fauna silvestre e gera um relatório em PDF com tabelas, gráficos e textos.

Como usar:
    1. Coloque o CSV na mesma pasta deste script (ou ajuste o caminho abaixo).
    2. Execute:  python etapa1_analise_exploratoria.py
    3. O relatório será salvo em "relatorio_etapa1_analise_exploratoria.pdf".

Bibliotecas necessárias: pandas, numpy, matplotlib
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
caminho_do_arquivo_de_dados = "rodofauna.csv"
caminho_do_relatorio_pdf = "relatorio_etapa1_analise_exploratoria.pdf"
nomes_dos_integrantes_do_grupo = "Nome do integrante 1, Nome do integrante 2, Nome do integrante 3"

valor_nao_identificado = "nao_identificado"
minimo_de_ocorrencias_por_especie = 30
limite_de_correlacao_alta = 0.8
percentual_para_considerar_atributo_pouco_informativo = 50

tamanho_da_pagina_em_retrato = (8.27, 11.69)
tamanho_da_pagina_em_paisagem = (11.69, 8.27)

nomes_dos_meses = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]
nomes_dos_dias_da_semana = [
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
    "sexta-feira", "sábado", "domingo",
]

# Papel de cada atributo, conforme o enunciado do trabalho
papel_de_cada_atributo = {
    "id": "não utilizar",
    "campanha": "não utilizar",
    "ano": "preditor",
    "periodo": "não utilizar",
    "mes": "não utilizar (derivar de data_dt)",
    "data_dt": "preditor (derivar atributos temporais)",
    "dia_da_sem": "preditor",
    "silvestre_": "apenas filtro de registros",
    "classe": "alvo (multiclasse)",
    "ordem": "não utilizar",
    "familia": "não utilizar",
    "especie_nm": "alvo possível (espécie)",
    "nome_popul": "não utilizar",
    "uc": "preditor",
    "x": "preditor",
    "y": "preditor",
    "posicao": "não utilizar",
    "hora": "não utilizar",
    "sexo": "não utilizar",
}


# ----------------------------------------------------------------------------
# Leitura e preparação dos dados
# ----------------------------------------------------------------------------
def carregar_dados():
    tabela = pd.read_csv(caminho_do_arquivo_de_dados, sep=None, engine="python")
    tabela.columns = tabela.columns.str.strip().str.lower()

    # Padroniza os textos (remove espaços e coloca em minúsculas)
    for nome_da_coluna in tabela.columns:
        if pd.api.types.is_string_dtype(tabela[nome_da_coluna]):
            tabela[nome_da_coluna] = tabela[nome_da_coluna].str.strip().str.lower()

    # Garante que as coordenadas UTM sejam numéricas
    for nome_da_coluna in ["x", "y"]:
        tabela[nome_da_coluna] = pd.to_numeric(tabela[nome_da_coluna], errors="coerce")

    return tabela


def listar_colunas_de_texto(tabela):
    return [nome for nome in tabela.columns if pd.api.types.is_string_dtype(tabela[nome])]


def resumir_atributos(tabela):
    """Tipo, valores ausentes, valores 'nao_identificado' e papel de cada atributo."""
    linhas_do_resumo = []
    for nome_do_atributo in tabela.columns:
        coluna = tabela[nome_do_atributo]
        quantidade_ausentes = int(coluna.isna().sum())
        quantidade_nao_identificado = int((coluna == valor_nao_identificado).sum())
        linhas_do_resumo.append({
            "atributo": nome_do_atributo,
            "tipo": str(coluna.dtype),
            "valores únicos": coluna.nunique(),
            "ausentes": quantidade_ausentes,
            "% ausentes": round(100 * quantidade_ausentes / len(tabela), 2),
            "nao_identificado": quantidade_nao_identificado,
            "% nao_identificado": round(100 * quantidade_nao_identificado / len(tabela), 2),
            "papel no trabalho": papel_de_cada_atributo.get(nome_do_atributo, "não definido"),
        })
    return pd.DataFrame(linhas_do_resumo)


def contar_outliers_pelo_intervalo_interquartil(serie):
    primeiro_quartil = serie.quantile(0.25)
    terceiro_quartil = serie.quantile(0.75)
    amplitude_interquartil = terceiro_quartil - primeiro_quartil
    limite_inferior = primeiro_quartil - 1.5 * amplitude_interquartil
    limite_superior = terceiro_quartil + 1.5 * amplitude_interquartil
    return int(((serie < limite_inferior) | (serie > limite_superior)).sum())


def calcular_estatisticas_descritivas(tabela, colunas_numericas):
    estatisticas = tabela[colunas_numericas].describe().T
    estatisticas["outliers"] = [
        contar_outliers_pelo_intervalo_interquartil(tabela[nome].dropna())
        for nome in colunas_numericas
    ]
    estatisticas.columns = [
        "quantidade", "média", "desvio padrão", "mínimo",
        "1º quartil", "mediana", "3º quartil", "máximo", "outliers (IQR)",
    ]
    estatisticas.index.name = "atributo"
    return estatisticas.round(2).reset_index()


def encontrar_atributos_redundantes(tabela):
    """Atributos de texto constantes e pares em que um atributo determina o outro."""
    colunas_de_texto = [
        nome for nome in listar_colunas_de_texto(tabela) if nome not in ("id", "data_dt")
    ]
    atributos_constantes = [
        nome for nome in colunas_de_texto if tabela[nome].nunique(dropna=False) == 1
    ]

    pares_redundantes = []
    for atributo_determinante in colunas_de_texto:
        if atributo_determinante in atributos_constantes:
            continue
        for atributo_determinado in colunas_de_texto:
            if atributo_determinado == atributo_determinante:
                continue
            valores_por_grupo = tabela.groupby(atributo_determinante, dropna=False)[
                atributo_determinado
            ].nunique(dropna=False)
            if valores_por_grupo.max() == 1:
                pares_redundantes.append(f"{atributo_determinante} determina {atributo_determinado}")

    return atributos_constantes, pares_redundantes


def contar_registros_duplicados(tabela):
    """Conta registros repetidos (a primeira ocorrência não é contada como duplicata)."""
    colunas_sem_identificador = [nome for nome in tabela.columns if nome != "id"]
    colunas_que_identificam_a_ocorrencia = ["data_dt", "x", "y", "especie_nm"]

    quantidade_de_duplicados_identicos = int(
        tabela.duplicated(subset=colunas_sem_identificador).sum()
    )
    quantidade_de_duplicados_da_mesma_ocorrencia = int(
        tabela.duplicated(subset=colunas_que_identificam_a_ocorrencia).sum()
    )
    return quantidade_de_duplicados_identicos, quantidade_de_duplicados_da_mesma_ocorrencia


# ----------------------------------------------------------------------------
# Funções de página do relatório
# ----------------------------------------------------------------------------
def adicionar_pagina_de_texto(arquivo_pdf, titulo, paragrafos):
    altura_de_cada_linha = 0.0165
    figura = plt.figure(figsize=tamanho_da_pagina_em_retrato)
    figura.text(0.08, 0.95, titulo, fontsize=16, fontweight="bold", va="top")
    posicao_vertical = 0.90

    for paragrafo in paragrafos:
        recuo = "   " if paragrafo.startswith("- ") else ""
        texto_quebrado = textwrap.fill(paragrafo, width=95, subsequent_indent=recuo)
        quantidade_de_linhas = texto_quebrado.count("\n") + 1

        if posicao_vertical - quantidade_de_linhas * altura_de_cada_linha < 0.04:
            arquivo_pdf.savefig(figura)
            plt.close(figura)
            figura = plt.figure(figsize=tamanho_da_pagina_em_retrato)
            posicao_vertical = 0.95

        figura.text(0.08, posicao_vertical, texto_quebrado, fontsize=10, va="top")
        posicao_vertical -= quantidade_de_linhas * altura_de_cada_linha + 0.012

    arquivo_pdf.savefig(figura)
    plt.close(figura)


def adicionar_pagina_de_tabela(arquivo_pdf, titulo, tabela, tamanho_da_fonte=8):
    figura, eixo = plt.subplots(figsize=tamanho_da_pagina_em_paisagem)
    eixo.axis("off")
    eixo.set_title(titulo, fontsize=14, fontweight="bold")
    tabela_desenhada = eixo.table(
        cellText=tabela.astype(str).values,
        colLabels=list(tabela.columns),
        loc="center",
        cellLoc="center",
    )
    tabela_desenhada.auto_set_font_size(False)
    tabela_desenhada.set_fontsize(tamanho_da_fonte)
    tabela_desenhada.auto_set_column_width(list(range(len(tabela.columns))))
    tabela_desenhada.scale(1, 1.5)
    arquivo_pdf.savefig(figura)
    plt.close(figura)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def criar_figura_distribuicao_das_classes(contagem_por_classe, contagem_por_especie):
    figura, (eixo_classes, eixo_especies) = plt.subplots(
        1, 2, figsize=tamanho_da_pagina_em_paisagem
    )

    barras = eixo_classes.bar(contagem_por_classe.index.astype(str), contagem_por_classe.values)
    eixo_classes.bar_label(barras)
    eixo_classes.set_title("Atropelamentos por classe taxonômica")
    eixo_classes.set_ylabel("Quantidade de registros")

    dez_especies_mais_frequentes = contagem_por_especie.head(10).sort_values()
    eixo_especies.barh(dez_especies_mais_frequentes.index.astype(str), dez_especies_mais_frequentes.values)
    eixo_especies.set_title("10 espécies mais atropeladas")
    eixo_especies.set_xlabel("Quantidade de registros")

    figura.tight_layout()
    return figura


def criar_figura_distribuicao_espacial(tabela):
    tabela_com_coordenadas = tabela.dropna(subset=["x", "y"])
    figura, (eixo_pontos, eixo_densidade) = plt.subplots(1, 2, figsize=tamanho_da_pagina_em_paisagem)

    eixo_pontos.scatter(tabela_com_coordenadas["x"], tabela_com_coordenadas["y"], s=3, alpha=0.4)
    eixo_pontos.set_title("Localização dos atropelamentos")

    mapa_de_densidade = eixo_densidade.hexbin(
        tabela_com_coordenadas["x"], tabela_com_coordenadas["y"],
        gridsize=40, mincnt=1, bins="log", cmap="inferno",
    )
    figura.colorbar(mapa_de_densidade, ax=eixo_densidade, label="Registros por célula (escala log)")
    eixo_densidade.set_title("Densidade de atropelamentos")

    for eixo in (eixo_pontos, eixo_densidade):
        eixo.set_xlabel("x (UTM, metros)")
        eixo.set_ylabel("y (UTM, metros)")
        eixo.set_aspect("equal", adjustable="datalim")
        eixo.tick_params(axis="x", rotation=45)

    figura.tight_layout()
    return figura


def criar_figura_distribuicao_temporal(tabela):
    figura, eixos = plt.subplots(2, 2, figsize=tamanho_da_pagina_em_paisagem)
    eixo_ano, eixo_mes, eixo_dia_da_semana, eixo_serie_mensal = eixos.flatten()

    contagem_por_ano = tabela["ano"].value_counts().sort_index()
    eixo_ano.bar(contagem_por_ano.index.astype(int).astype(str), contagem_por_ano.values)
    eixo_ano.set_title("Registros por ano")

    contagem_por_mes = tabela["mes_numero"].value_counts().reindex(range(1, 13), fill_value=0)
    eixo_mes.bar(nomes_dos_meses, contagem_por_mes.values)
    eixo_mes.set_title("Registros por mês")
    eixo_mes.tick_params(axis="x", rotation=45)

    contagem_por_dia = tabela["dia_da_semana_numero"].value_counts().reindex(range(7), fill_value=0)
    eixo_dia_da_semana.bar(nomes_dos_dias_da_semana, contagem_por_dia.values)
    eixo_dia_da_semana.set_title("Registros por dia da semana")
    eixo_dia_da_semana.tick_params(axis="x", rotation=45)

    serie_mensal = tabela["data_convertida"].dt.to_period("M").value_counts().sort_index()
    eixo_serie_mensal.plot(serie_mensal.index.to_timestamp(), serie_mensal.values, marker="o", markersize=3)
    eixo_serie_mensal.set_title("Registros por mês ao longo do tempo")
    eixo_serie_mensal.tick_params(axis="x", rotation=45)

    for eixo in eixos.flatten():
        eixo.set_ylabel("Quantidade de registros")

    figura.tight_layout()
    return figura


def criar_figura_boxplots_das_coordenadas(tabela):
    figura, (eixo_x, eixo_y) = plt.subplots(1, 2, figsize=tamanho_da_pagina_em_paisagem)
    eixo_x.boxplot(tabela["x"].dropna())
    eixo_x.set_title("Coordenada x (UTM)")
    eixo_y.boxplot(tabela["y"].dropna())
    eixo_y.set_title("Coordenada y (UTM)")
    figura.suptitle("Boxplots para identificação de outliers", fontsize=14, fontweight="bold")
    figura.tight_layout()
    return figura


def criar_figura_matriz_de_correlacao(matriz_de_correlacao):
    figura, eixo = plt.subplots(figsize=tamanho_da_pagina_em_paisagem)
    imagem = eixo.imshow(matriz_de_correlacao, vmin=-1, vmax=1, cmap="coolwarm")
    quantidade_de_atributos = len(matriz_de_correlacao.columns)

    eixo.set_xticks(range(quantidade_de_atributos))
    eixo.set_yticks(range(quantidade_de_atributos))
    eixo.set_xticklabels(matriz_de_correlacao.columns, rotation=45, ha="right")
    eixo.set_yticklabels(matriz_de_correlacao.columns)

    for linha in range(quantidade_de_atributos):
        for coluna in range(quantidade_de_atributos):
            eixo.text(coluna, linha, f"{matriz_de_correlacao.iloc[linha, coluna]:.2f}",
                      ha="center", va="center", fontsize=9)

    figura.colorbar(imagem, ax=eixo, label="Correlação de Pearson")
    eixo.set_title("Correlação entre atributos numéricos", fontsize=14, fontweight="bold")
    figura.tight_layout()
    return figura


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    tabela_completa = carregar_dados()
    quantidade_de_registros_totais = len(tabela_completa)

    # Apenas fauna silvestre, conforme o enunciado
    tabela = tabela_completa[tabela_completa["silvestre_"] == "silvestre"].copy()
    quantidade_de_registros_silvestres = len(tabela)

    resumo_dos_atributos = resumir_atributos(tabela)
    quantidade_de_atributos = len(tabela.columns)

    # Registros duplicados (calculado antes de criar os atributos derivados)
    (quantidade_de_duplicados_identicos,
     quantidade_de_duplicados_da_mesma_ocorrencia) = contar_registros_duplicados(tabela)

    # Atributos temporais derivados de data_dt
    tabela["data_convertida"] = pd.to_datetime(tabela["data_dt"], dayfirst=True, errors="coerce")
    tabela["mes_numero"] = tabela["data_convertida"].dt.month
    tabela["dia_do_ano"] = tabela["data_convertida"].dt.dayofyear
    tabela["dia_da_semana_numero"] = tabela["data_convertida"].dt.dayofweek
    quantidade_de_datas_invalidas = int(tabela["data_convertida"].isna().sum())

    # Distribuição das classes
    contagem_por_classe = tabela["classe"].value_counts()
    percentual_por_classe = (100 * contagem_por_classe / len(tabela)).round(2)
    razao_de_desbalanceamento = contagem_por_classe.max() / contagem_por_classe.min()

    especies_identificadas = tabela[tabela["especie_nm"] != valor_nao_identificado]["especie_nm"]
    contagem_por_especie = especies_identificadas.value_counts()
    quantidade_de_especies_raras = int((contagem_por_especie < minimo_de_ocorrencias_por_especie).sum())
    registros_em_especies_raras = int(
        contagem_por_especie[contagem_por_especie < minimo_de_ocorrencias_por_especie].sum()
    )

    # Estatísticas descritivas e outliers
    colunas_numericas = ["ano", "x", "y"]
    estatisticas_descritivas = calcular_estatisticas_descritivas(tabela, colunas_numericas)

    # Correlação e redundância
    colunas_para_correlacao = ["ano", "x", "y", "mes_numero", "dia_do_ano", "dia_da_semana_numero"]
    matriz_de_correlacao = tabela[colunas_para_correlacao].corr()
    pares_com_correlacao_alta = []
    for posicao_linha in range(len(colunas_para_correlacao)):
        for posicao_coluna in range(posicao_linha + 1, len(colunas_para_correlacao)):
            valor = matriz_de_correlacao.iloc[posicao_linha, posicao_coluna]
            if abs(valor) >= limite_de_correlacao_alta:
                pares_com_correlacao_alta.append(
                    f"{colunas_para_correlacao[posicao_linha]} x {colunas_para_correlacao[posicao_coluna]} = {valor:.2f}"
                )
    atributos_constantes, pares_redundantes = encontrar_atributos_redundantes(tabela)

    # ------------------------------------------------------------------
    # Montagem do relatório em PDF
    # ------------------------------------------------------------------
    with PdfPages(caminho_do_relatorio_pdf) as arquivo_pdf:

        # Página 1: descrição do problema e variáveis alvo
        adicionar_pagina_de_texto(arquivo_pdf, "Etapa 1 - Análise Exploratória do Dataset", [
            "Universidade Federal de Santa Maria - Disciplina de Inteligência Artificial",
            f"Integrantes: {nomes_dos_integrantes_do_grupo}",
            "1. DESCRIÇÃO DO PROBLEMA",
            "O Projeto RodoFauna, do Instituto Brasília Ambiental (IBRAM/DF), estuda o impacto das rodovias "
            "sobre a fauna silvestre do Distrito Federal. Entre abril de 2010 e março de 2015 foram realizados "
            "484 percursos sistemáticos (cerca de 55.176 km), nos quais os animais encontrados atropelados foram "
            "identificados taxonomicamente e georreferenciados.",
            "O objetivo do trabalho é construir modelos de aprendizado supervisionado para prever a ocorrência, "
            "o tipo e a intensidade dos atropelamentos de fauna silvestre, de modo a apoiar a identificação de "
            "hotspots (locais críticos) de atropelamento.",
            "A base contém apenas registros de ocorrências (pontos onde houve atropelamento). Por isso, a classe "
            "negativa do problema binário não pode ser obtida apenas pela ausência de pontos e precisará ser "
            "construída de forma justificada.",
            "2. VARIÁVEL ALVO DE CADA EXPERIMENTO (proposta inicial)",
            "- Classificação binária: risco do segmento rodoviário (1 = com ocorrência ou alto risco; "
            "0 = sem ocorrência ou baixo risco). A unidade de análise sugerida é o segmento de 500 m, e o "
            "critério de alto/baixo risco será definido e justificado nas próximas etapas.",
            "- Classificação multiclasse: classe taxonômica do animal (coluna classe), por exemplo aves, "
            "mamíferos, répteis e anfíbios.",
            "- Regressão: número de atropelamentos por segmento rodoviário por mês.",
            "A coluna classe só será usada como alvo, nunca como atributo de entrada, pois só é conhecida "
            "depois do atropelamento.",
        ])

        # Página 2: visão geral do dataset
        periodo_inicial = tabela["data_convertida"].min()
        periodo_final = tabela["data_convertida"].max()
        adicionar_pagina_de_texto(arquivo_pdf, "Visão geral do dataset", [
            f"Quantidade total de registros no arquivo: {quantidade_de_registros_totais}",
            f"Quantidade de registros de fauna silvestre (utilizados): {quantidade_de_registros_silvestres}",
            f"Quantidade de atributos originais: {quantidade_de_atributos}",
            f"Período dos registros: de {periodo_inicial:%d/%m/%Y} até {periodo_final:%d/%m/%Y}",
            f"Datas que não puderam ser convertidas: {quantidade_de_datas_invalidas}",
            f"Registros duplicados idênticos (todas as colunas, exceto id): {quantidade_de_duplicados_identicos}",
            f"Registros duplicados da mesma ocorrência (mesma data, x, y e espécie): "
            f"{quantidade_de_duplicados_da_mesma_ocorrencia}",
            "Atributos derivados de data_dt para a análise: mês, dia do ano e dia da semana.",
            "A tabela seguinte apresenta o tipo, os valores ausentes, os valores nao_identificado e o papel "
            "de cada atributo no trabalho.",
        ])

        adicionar_pagina_de_tabela(arquivo_pdf, "Atributos: tipos, valores ausentes e nao_identificado",
                                   resumo_dos_atributos, tamanho_da_fonte=7)

        # Distribuição das classes
        figura_classes = criar_figura_distribuicao_das_classes(contagem_por_classe, contagem_por_especie)
        arquivo_pdf.savefig(figura_classes)
        plt.close(figura_classes)

        linhas_das_classes = [
            f"- {nome_da_classe}: {quantidade} registros ({percentual_por_classe[nome_da_classe]}%)"
            for nome_da_classe, quantidade in contagem_por_classe.items()
        ]
        adicionar_pagina_de_texto(arquivo_pdf, "Distribuição das classes", [
            "Distribuição da classe taxonômica (alvo da classificação multiclasse):",
            *linhas_das_classes,
            f"Razão entre a classe mais frequente e a menos frequente: {razao_de_desbalanceamento:.1f}",
            f"Registros com espécie não identificada: {len(tabela) - len(especies_identificadas)}",
            f"Espécies identificadas: {contagem_por_especie.shape[0]}. Com menos de "
            f"{minimo_de_ocorrencias_por_especie} ocorrências: {quantidade_de_especies_raras} espécies, "
            f"somando {registros_em_especies_raras} registros.",
        ])

        # Estatísticas descritivas
        adicionar_pagina_de_tabela(arquivo_pdf, "Estatísticas descritivas dos atributos numéricos",
                                   estatisticas_descritivas, tamanho_da_fonte=9)

        # Distribuições espacial e temporal
        for criar_figura in (criar_figura_distribuicao_espacial,
                             criar_figura_distribuicao_temporal,
                             criar_figura_boxplots_das_coordenadas):
            figura = criar_figura(tabela)
            arquivo_pdf.savefig(figura)
            plt.close(figura)

        # Correlação
        figura_correlacao = criar_figura_matriz_de_correlacao(matriz_de_correlacao)
        arquivo_pdf.savefig(figura_correlacao)
        plt.close(figura_correlacao)

        # Resumo dos problemas identificados
        atributos_com_ausentes = resumo_dos_atributos[resumo_dos_atributos["ausentes"] > 0]
        atributos_pouco_informativos = resumo_dos_atributos[
            resumo_dos_atributos["% nao_identificado"] >= percentual_para_considerar_atributo_pouco_informativo
        ]

        if atributos_com_ausentes.empty:
            texto_ausentes = "- Valores ausentes: nenhum atributo possui valores ausentes."
        else:
            lista = ", ".join(
                f"{linha['atributo']} ({linha['ausentes']})" for _, linha in atributos_com_ausentes.iterrows()
            )
            texto_ausentes = f"- Valores ausentes: {lista}."

        if atributos_pouco_informativos.empty:
            texto_nao_identificado = (
                f"- Valores nao_identificado: nenhum atributo com {percentual_para_considerar_atributo_pouco_informativo}% ou mais."
            )
        else:
            lista = ", ".join(
                f"{linha['atributo']} ({linha['% nao_identificado']}%)"
                for _, linha in atributos_pouco_informativos.iterrows()
            )
            texto_nao_identificado = (
                f"- Atributos com {percentual_para_considerar_atributo_pouco_informativo}% ou mais de valores "
                f"nao_identificado (pouco informativos): {lista}."
            )

        outliers_por_atributo = ", ".join(
            f"{linha['atributo']}: {linha['outliers (IQR)']}" for _, linha in estatisticas_descritivas.iterrows()
        )

        adicionar_pagina_de_texto(arquivo_pdf, "Problemas identificados", [
            f"- Classes desbalanceadas: a classe mais frequente é '{contagem_por_classe.index[0]}' "
            f"({percentual_por_classe.iloc[0]}%) e a menos frequente é '{contagem_por_classe.index[-1]}' "
            f"({percentual_por_classe.iloc[-1]}%). Com esse desbalanceamento, a Accuracy isolada pode ser enganosa.",
            texto_ausentes,
            texto_nao_identificado,
            f"- Registros duplicados: {quantidade_de_duplicados_identicos} idênticos (exceto id) e "
            f"{quantidade_de_duplicados_da_mesma_ocorrencia} com mesma data, coordenadas e espécie "
            f"({round(100 * quantidade_de_duplicados_da_mesma_ocorrencia / len(tabela), 2)}% dos registros "
            "silvestres). Duplicatas contam a mesma ocorrência mais de uma vez e podem inflar a contagem "
            "por segmento; a decisão de removê-las deve ser justificada na Etapa 2.",
            f"- Outliers pelo intervalo interquartil (IQR): {outliers_por_atributo}.",
            f"- Atributos constantes: {', '.join(atributos_constantes) if atributos_constantes else 'nenhum'}.",
            "- Atributos redundantes (um determina o outro): "
            + ("; ".join(pares_redundantes) if pares_redundantes else "nenhum par encontrado") + ".",
            "- Correlação elevada (|r| >= " + str(limite_de_correlacao_alta) + "): "
            + ("; ".join(pares_com_correlacao_alta) if pares_com_correlacao_alta else "nenhum par encontrado") + ".",
            f"- Classes com poucos exemplos: {quantidade_de_especies_raras} espécies com menos de "
            f"{minimo_de_ocorrencias_por_especie} ocorrências, o que inviabiliza a classificação por espécie "
            "sem agrupamento. A classificação por classe taxonômica é mais adequada.",
            "- Natureza espacial: registros muito próximos são correlacionados. Uma divisão aleatória entre "
            "treino e teste pode produzir resultados otimistas, o que será tratado na Etapa 2.",
        ])

    print(f"Relatório salvo em: {caminho_do_relatorio_pdf}")


if __name__ == "__main__":
    main()