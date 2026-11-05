#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 1 - Análise Exploratória do Dataset (RodoFauna / IBRAM-DF)
Trabalho de Aprendizado Supervisionado - Inteligência Artificial - UFSM

Uso (Windows / Linux / macOS):
    pip install -r requirements.txt
    python etapa1_eda.py --dados data/ProgramaRodofaunaAtropelada.xlsx --saida resultados/etapa1

Parâmetros opcionais:
    --celula 500      tamanho (m) da célula espacial usada como "segmento" (padrão 500 m)

O que o script gera em --saida:
    figuras/*.png          todos os gráficos (numerados na ordem do relatório)
    tabelas/*.csv          todas as tabelas (abrem no Excel/LibreOffice)
    base_silvestre_eda.csv base filtrada (só silvestres) + atributos temporais derivados
    resumo_etapa1.md       resumo com TODOS os números calculados, organizado nos
                           tópicos pedidos no enunciado - base para o relatório técnico

Observação: campanha, periodo, mes, hora, sexo, posicao etc. NÃO são usados como
preditores (regra do enunciado). Aqui eles só aparecem na análise exploratória,
para justificar a exclusão (ex.: % de nao_identificado) ou para medir o esforço
amostral (campanhas por mês).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # gera PNG sem precisar de janela
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
NI = "nao_identificado"

# Campos que o enunciado proíbe como preditores (após filtrar silvestres)
PROIBIDOS = ["objectid", "id", "silvestre_", "hora", "sexo", "posicao", "ordem",
             "familia", "nome_popul", "campanha", "periodo", "mes"]

# Nomes canônicos -> possíveis nomes no arquivo (shapefile corta em 10 caracteres)
CANONICAS = {
    "objectid": ["objectid", "fid", "oid"],
    "id": ["id"],
    "campanha": ["campanha"],
    "ano": ["ano"],
    "periodo": ["periodo"],
    "mes": ["mes"],
    "data_dt": ["data_dt", "data"],
    "dia_da_sem": ["dia_da_sem", "dia_da_se", "dia_semana", "dia_sem"],
    "silvestre_": ["silvestre_", "silvestre"],
    "classe": ["classe"],
    "ordem": ["ordem"],
    "familia": ["familia"],
    "especie_nm": ["especie_nm", "especie_n", "especie"],
    "nome_popul": ["nome_popul", "nome_pop", "nome_popular"],
    "uc": ["uc"],
    "x": ["x", "coord_x", "utm_x"],
    "y": ["y", "coord_y", "utm_y"],
    "posicao": ["posicao"],
    "hora": ["hora"],
    "sexo": ["sexo"],
}

MESES_PT = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5,
            "junho": 6, "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10,
            "novembro": 11, "dezembro": 12}
MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set",
               "out", "nov", "dez"]
DIAS = ["seg", "ter", "qua", "qui", "sex", "sab", "dom"]

# Nomes das classes (a base usa inglês) -> rótulo em português para gráficos
ROTULO_CLASSE = {"birds": "Aves", "mammals": "Mamíferos", "reptiles": "Répteis",
                 "amphibians": "Anfíbios"}

# Limites plausíveis do DF em UTM 23S (SIRGAS 2000), com folga de ~10 km.
# DF: lon -48,29 a -47,31 ; lat -15,50 a -16,05  ->  x ~148-253 km, y ~8.225-8.286 km
LIM_X = (135_000, 265_000)
LIM_Y = (8_210_000, 8_300_000)

# Paleta categórica (ordem fixa) e rampa sequencial azul
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#6250d6", "#e34948"]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]
RAMPA_AZUL = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
              "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_azul", RAMPA_AZUL)
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
CINZA = "#b5b3ad"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm_nome_coluna(c) -> str:
    return sem_acento(str(c)).strip().lower().replace(" ", "_")


def norm_texto(v):
    """Padroniza valores textuais: minúsculo, sem acento, '_' no lugar de espaço."""
    if pd.isna(v):
        return np.nan
    if not isinstance(v, str):
        return v
    s = sem_acento(v).strip().lower().replace(" ", "_")
    if s in ("", "nan", "none", "null", "-", "--"):
        return np.nan
    if s.startswith("nao_identific") or s in ("ni", "n/i", "indeterminado"):
        return NI
    return s


def tabela_md(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:,.2f}") -> str:
    """Converte DataFrame em tabela Markdown (sem depender do pacote tabulate)."""
    d = df.copy()
    if index:
        d = d.reset_index()
    def fmt(v):
        if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) and 1900 <= v <= 2100:
            return dec(v, 2) if not float(v).is_integer() else str(int(v))  # anos sem separador de milhar
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "-"
            if float(v).is_integer() and abs(v) < 1e15:
                return f"{int(v):,}".replace(",", ".")
            return floatfmt.format(v).replace(",", "X").replace(".", ",").replace("X", ".")
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}".replace(",", ".")
        return str(v)
    cab = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    linhas = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([cab, sep] + linhas)


def n_fmt(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def dec(v, k: int = 1) -> str:
    return f"{v:.{k}f}".replace(".", ",")


def pct(a, b) -> str:
    return "-" if b == 0 else f"{100 * a / b:.1f}%".replace(".", ",")


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér com correção de viés (Bergsma, 2013)."""
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else np.nan


def rotulo(c) -> str:
    return ROTULO_CLASSE.get(c, str(c))


class Relatorio:
    """Acumula o texto do resumo em Markdown."""

    def __init__(self):
        self.linhas: list[str] = []

    def h(self, nivel: int, texto: str):
        self.linhas += ["", "#" * nivel + " " + texto, ""]

    def p(self, texto: str = ""):
        if self.linhas and self.linhas[-1].startswith("- "):
            self.linhas.append("")
        self.linhas.append(texto)

    def item(self, texto: str):
        self.linhas.append(f"- {texto}")

    def tab(self, df: pd.DataFrame, **kw):
        self.linhas += ["", tabela_md(df, **kw), ""]

    def fig(self, nome: str, legenda: str):
        self.linhas += ["", f"![{legenda}](figuras/{nome})", f"*{legenda}*", ""]

    def salvar(self, caminho: Path):
        caminho.write_text("\n".join(self.linhas).strip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Leitura e padronização
# --------------------------------------------------------------------------- #
def carregar(caminho: Path) -> pd.DataFrame:
    if caminho.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(caminho, sheet_name=0)
    elif caminho.suffix.lower() == ".csv":
        df = pd.read_csv(caminho, sep=None, engine="python")
    else:
        sys.exit(f"Formato não suportado: {caminho.suffix}")
    df.columns = [norm_nome_coluna(c) for c in df.columns]

    # renomeia para os nomes canônicos (match exato e, depois, por prefixo)
    ren, usadas = {}, set()
    for canon, cands in CANONICAS.items():
        achou = next((c for c in df.columns if c in cands and c not in usadas), None)
        if achou is None:
            achou = next((c for c in df.columns if c not in usadas and
                          any(len(k) >= 4 and c.startswith(k) for k in cands)), None)
        if achou is not None:
            ren[achou] = canon
            usadas.add(achou)
    df = df.rename(columns=ren)

    faltando = [c for c in ["data_dt", "classe", "x", "y", "silvestre_"] if c not in df.columns]
    if faltando:
        sys.exit(f"Colunas essenciais não encontradas: {faltando}\nColunas lidas: {list(df.columns)}")

    for c in df.columns:
        if c != "data_dt" and df[c].dtype == object:
            df[c] = df[c].map(norm_texto)
    for c in ["x", "y", "ano", "campanha", "periodo", "id", "objectid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_datas(s: pd.Series) -> pd.Series:
    """Converte data_dt aceitando datetime do Excel, serial numérico, epoch (ms) ou texto dd/mm/aa."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    eh_data = s.map(lambda v: isinstance(v, (pd.Timestamp, dt.datetime, dt.date)))
    if eh_data.any():
        out[eh_data] = pd.to_datetime(s[eh_data].astype("object"), errors="coerce")
    num = pd.to_numeric(s.where(~eh_data), errors="coerce")
    if num.notna().any():
        grande = num > 1e11  # epoch em milissegundos (export do ArcGIS)
        out[grande] = pd.to_datetime(num[grande], unit="ms", errors="coerce")
        serial = num.between(20000, 60000)  # serial do Excel
        out[serial] = pd.to_datetime(num[serial], unit="D", origin="1899-12-30", errors="coerce")
    txt = s[out.isna() & s.notna() & ~eh_data].astype(str).str.strip()
    for fmt in ["%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%y %H:%M"]:
        if txt.empty:
            break
        conv = pd.to_datetime(txt, format=fmt, errors="coerce")
        out.loc[conv.dropna().index] = conv.dropna()
        txt = txt[conv.isna()]
    return out


def estacao(m: int) -> str:
    # estações astronômicas aproximadas (hemisfério sul), por mês
    return {12: "verao", 1: "verao", 2: "verao", 3: "outono", 4: "outono", 5: "outono",
            6: "inverno", 7: "inverno", 8: "inverno", 9: "primavera", 10: "primavera",
            11: "primavera"}[m]


def derivar_temporais(df: pd.DataFrame) -> pd.DataFrame:
    d = df["data"]
    df["mes_num"] = d.dt.month
    df["dia_do_ano"] = d.dt.dayofyear
    df["dia_semana_num"] = d.dt.dayofweek  # 0 = segunda
    df["fim_de_semana"] = (df["dia_semana_num"] >= 5).astype("Int64").where(d.notna())
    df["estacao"] = df["mes_num"].map(lambda m: estacao(int(m)) if pd.notna(m) else np.nan)
    # Clima do DF (Cerrado): chuvoso out-abr, seco mai-set
    df["periodo_climatico"] = df["mes_num"].map(
        lambda m: np.nan if pd.isna(m) else ("seco" if 5 <= m <= 9 else "chuvoso"))
    df["ano_mes"] = d.dt.to_period("M")
    return df


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def estilo():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.edgecolor": CINZA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#e7e6e2", "grid.linewidth": 0.6,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "legend.frameon": False,
        "font.size": 10,
    })


def salvar_fig(fig, pasta: Path, nome: str):
    fig.savefig(pasta / nome)
    plt.close(fig)


def barras(ax, rotulos, valores, cor=CORES[0], horizontal=False, fmt_pct_total=None):
    pos = np.arange(len(valores))
    if horizontal:
        ax.barh(pos, valores, color=cor, height=0.7, edgecolor="white", linewidth=1)
        ax.set_yticks(pos, rotulos)
        ax.invert_yaxis()
        for i, v in enumerate(valores):
            t = n_fmt(v) if fmt_pct_total is None else f"{n_fmt(v)} ({pct(v, fmt_pct_total)})"
            ax.text(v, i, " " + t, va="center", fontsize=8, color=TINTA_2)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(pos, valores, color=cor, width=0.7, edgecolor="white", linewidth=1)
        ax.set_xticks(pos, rotulos)
        for i, v in enumerate(valores):
            t = n_fmt(v) if fmt_pct_total is None else f"{n_fmt(v)}\n({pct(v, fmt_pct_total)})"
            ax.text(i, v, t, ha="center", va="bottom", fontsize=8, color=TINTA_2)
        ax.grid(axis="x", visible=False)


# --------------------------------------------------------------------------- #
# Análise principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Etapa 1 - EDA RodoFauna")
    ap.add_argument("--dados", default="data/ProgramaRodofaunaAtropelada.xlsx")
    ap.add_argument("--saida", default="resultados/etapa1")
    ap.add_argument("--celula", type=int, default=500, help="tamanho da célula espacial em metros")
    args = ap.parse_args()

    saida = Path(args.saida)
    FIG, TAB = saida / "figuras", saida / "tabelas"
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    estilo()
    R = Relatorio()
    problemas: list[str] = []
    CEL = args.celula

    print(f"Lendo {args.dados} ...")
    bruto = carregar(Path(args.dados))
    bruto["data"] = parse_datas(bruto["data_dt"])

    R.h(1, "Etapa 1 - Análise Exploratória do Dataset (RodoFauna - IBRAM/DF)")
    R.p(f"*Gerado automaticamente por `etapa1_eda.py` em {dt.datetime.now():%d/%m/%Y %H:%M}. "
        f"Arquivo: `{Path(args.dados).name}`.*")

    # ------------------------------------------------------------------ 1
    R.h(2, "1. Descrição do problema")
    R.p("Os dados vêm do Projeto RodoFauna (IBRAM/DF): percursos sistemáticos em rodovias do "
        "Distrito Federal (abr/2010 a mar/2015) registrando animais atropelados, com identificação "
        "taxonômica e coordenadas UTM. Cada linha é **uma ocorrência** (um animal encontrado "
        "atropelado). O objetivo do trabalho é prever a ocorrência, o tipo e a intensidade dos "
        "atropelamentos de fauna silvestre e identificar *hotspots*. Por isso a análise abaixo "
        "olha a base em dois níveis: o **registro individual** (para o problema multiclasse) e "
        f"**unidades espaciais/temporais agregadas** (células de {CEL} m, para o problema binário "
        "de hotspot e para a regressão de contagens).")

    # ------------------------------------------------------------------ 2
    R.h(2, "2. Base bruta: registros, atributos e tipos")
    n_bruto, a_bruto = bruto.shape
    a_bruto -= 1  # coluna 'data' criada pelo script
    R.item(f"Registros na base bruta: **{n_fmt(n_bruto)}**")
    R.item(f"Atributos na base bruta: **{a_bruto}**")
    tipos = pd.DataFrame({
        "tipo pandas": bruto.drop(columns="data").dtypes.astype(str),
        "valores distintos": bruto.drop(columns="data").nunique(),
        "exemplo": bruto.drop(columns="data").apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan).astype(str),
    })

    def natureza(c):
        if c in ("x", "y"):
            return "numérico contínuo (coordenada UTM)"
        if c in ("ano",):
            return "numérico discreto / temporal"
        if c == "data_dt":
            return "data"
        if c in ("objectid", "id"):
            return "identificador"
        if c in ("campanha", "periodo"):
            return "numérico discreto (controle amostral)"
        return "categórico nominal"
    tipos["natureza"] = [natureza(c) for c in tipos.index]
    tipos.index.name = "atributo"
    tipos.to_csv(TAB / "01_tipos_atributos.csv", encoding="utf-8-sig")
    R.tab(tipos)

    vc_silv = bruto["silvestre_"].value_counts(dropna=False)
    R.p("Distribuição de `silvestre_` na base bruta:")
    R.tab(pd.DataFrame({"registros": vc_silv, "%": 100 * vc_silv / n_bruto}).rename_axis("silvestre_"))

    # ------------------------------------------------------------------ 3
    R.h(2, "3. Filtragem: apenas fauna silvestre")
    df = bruto[bruto["silvestre_"] == "silvestre"].copy()
    n = len(df)
    R.p(f"Após `silvestre_ == 'silvestre'` restam **{n_fmt(n)} registros** "
        f"({pct(n, n_bruto)} da base bruta). Todo o restante da análise usa só esses registros.")

    # duplicatas
    cols_dup = [c for c in df.columns if c not in ("objectid", "id", "data")]
    dup_exata = int(df.duplicated(subset=cols_dup).sum())
    chave = [c for c in ["data", "x", "y", "especie_nm"] if c in df.columns]
    dup_chave = int(df.duplicated(subset=chave).sum())
    R.item(f"Linhas duplicadas (todas as colunas exceto identificadores): **{n_fmt(dup_exata)}**")
    R.item(f"Linhas repetidas em ({', '.join(chave)}) - possível mesmo animal registrado 2x: "
           f"**{n_fmt(dup_chave)}**")
    if dup_exata:
        problemas.append(f"{n_fmt(dup_exata)} registros duplicados exatos (remover no pré-processamento).")

    # ------------------------------------------------------------------ 4
    R.h(2, "4. Valores ausentes e `nao_identificado`")
    aus = pd.DataFrame({
        "ausentes (NaN)": df.drop(columns="data").isna().sum(),
        "nao_identificado": (df.drop(columns="data") == NI).sum(),
    })
    aus["% ausente"] = 100 * aus["ausentes (NaN)"] / n
    aus["% nao_ident."] = 100 * aus["nao_identificado"] / n
    aus["% total sem info"] = aus["% ausente"] + aus["% nao_ident."]
    aus["uso permitido como preditor?"] = ["não (regra do enunciado)" if c in PROIBIDOS else
                                          ("alvo" if c in ("classe", "especie_nm") else "sim")
                                          for c in aus.index]
    aus = aus.sort_values("% total sem info", ascending=False)
    aus.index.name = "atributo"
    aus.to_csv(TAB / "02_ausentes_nao_identificado.csv", encoding="utf-8-sig")
    R.tab(aus)
    n_data_invalida = int(df["data"].isna().sum())
    R.item(f"Datas que não puderam ser convertidas em `data_dt`: **{n_fmt(n_data_invalida)}**")

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(aus) + 1))
    a2 = aus.iloc[::-1]
    pos = np.arange(len(a2))
    ax.barh(pos, a2["% ausente"], color=CORES[0], height=0.7, label="ausente (NaN)",
            edgecolor="white", linewidth=1)
    ax.barh(pos, a2["% nao_ident."], left=a2["% ausente"], color=CORES[1], height=0.7,
            label="nao_identificado", edgecolor="white", linewidth=1)
    ax.set_yticks(pos, [f"{c}  ✕" if c in PROIBIDOS else c for c in a2.index])
    for i, v in enumerate(a2["% total sem info"]):
        if v > 0:
            ax.text(v, i, f" {v:.1f}%".replace(".", ","), va="center", fontsize=8, color=TINTA_2)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% dos registros silvestres")
    ax.set_title("Atributos sem informação (✕ = não pode ser preditor)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    salvar_fig(fig, FIG, "01_ausentes_nao_identificado.png")
    R.fig("01_ausentes_nao_identificado.png", "Figura 1 - Percentual de valores ausentes e nao_identificado por atributo.")

    altos = aus[aus["% total sem info"] >= 50]
    if len(altos):
        problemas.append("Atributos com ≥50% sem informação: " + ", ".join(
            f"`{c}` ({v:.0f}%)" for c, v in altos["% total sem info"].items()) +
            " - todos já excluídos como preditores pelo enunciado, o que se justifica também pela falta de informação.")
    for alvo in ("classe", "especie_nm"):
        if alvo in aus.index and aus.loc[alvo, "% total sem info"] > 0:
            problemas.append(f"Alvo `{alvo}` tem {aus.loc[alvo, 'nao_identificado'] + aus.loc[alvo, 'ausentes (NaN)']:.0f} "
                             f"registros sem identificação ({dec(aus.loc[alvo, '% total sem info'])}%) - "
                             "devem ser removidos do experimento que usa esse alvo.")
    if "uc" in aus.index and aus.loc["uc", "% total sem info"] > 0:
        problemas.append(f"`uc` sem informação em {dec(aus.loc['uc', '% total sem info'])}% dos registros - "
                         "tratar como categoria própria ('fora_de_uc'/'desconhecida') ou imputar pela localização.")

    # ------------------------------------------------------------------ 5
    df = derivar_temporais(df)
    R.h(2, "5. Atributos derivados (engenharia de atributos inicial)")
    R.p("A partir de `data_dt` foram derivados: `mes_num`, `dia_do_ano`, `dia_semana_num` "
        "(0 = segunda), `fim_de_semana` (sáb/dom = 1), `estacao` (verão = dez-fev, outono = mar-mai, "
        "inverno = jun-ago, primavera = set-nov) e `periodo_climatico` (chuvoso = out-abr, "
        "seco = mai-set, regime típico do Cerrado no DF). A partir de `x`,`y` foi criada a "
        f"célula espacial de {CEL} m (`cel_x`, `cel_y`), usada como aproximação de segmento rodoviário.")
    df["cel_x"] = (df["x"] // CEL).astype("Int64")
    df["cel_y"] = (df["y"] // CEL).astype("Int64")
    df["celula"] = df["cel_x"].astype(str) + "_" + df["cel_y"].astype(str)

    # ------------------------------------------------------------------ 6
    R.h(2, "6. Estatísticas descritivas das variáveis numéricas")
    nums = [c for c in ["x", "y", "ano", "mes_num", "dia_do_ano", "dia_semana_num", "fim_de_semana"]
            if c in df.columns]
    desc = df[nums].astype(float).describe().T
    desc["assimetria"] = df[nums].astype(float).skew()
    desc["curtose"] = df[nums].astype(float).kurt()
    desc.index.name = "variável"
    desc.to_csv(TAB / "03_estatisticas_descritivas.csv", encoding="utf-8-sig")
    R.tab(desc)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, c, cor in zip(axs, ["x", "y"], CORES[:2]):
        ax.hist(df[c].dropna(), bins=50, color=cor, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Distribuição de {c} (UTM, m)")
        ax.set_xlabel(c)
        ax.set_ylabel("registros")
        ax.ticklabel_format(style="plain", axis="x")
        ax.tick_params(axis="x", rotation=30)
    salvar_fig(fig, FIG, "02_hist_coordenadas.png")
    R.fig("02_hist_coordenadas.png", "Figura 2 - Histogramas das coordenadas UTM x e y.")

    # ------------------------------------------------------------------ 7
    R.h(2, "7. Outliers e consistência")
    q1x, q3x = df["x"].quantile([.25, .75]); iqx = q3x - q1x
    q1y, q3y = df["y"].quantile([.25, .75]); iqy = q3y - q1y
    out_iqr = ((df["x"] < q1x - 1.5 * iqx) | (df["x"] > q3x + 1.5 * iqx) |
               (df["y"] < q1y - 1.5 * iqy) | (df["y"] > q3y + 1.5 * iqy))
    fora_df = ~(df["x"].between(*LIM_X) & df["y"].between(*LIM_Y))
    coord_nula = df["x"].isna() | df["y"].isna() | (df["x"] == 0) | (df["y"] == 0)
    R.item(f"Coordenadas ausentes ou zero: **{n_fmt(coord_nula.sum())}**")
    R.item(f"Fora da envoltória plausível do DF em UTM 23S (x {LIM_X[0]:,}-{LIM_X[1]:,}; "
           f"y {LIM_Y[0]:,}-{LIM_Y[1]:,}): **{n_fmt((fora_df & ~coord_nula).sum())}**".replace(",", "."))
    R.item(f"Outliers pelo critério de Tukey (1,5·IQR) em x ou y: **{n_fmt(out_iqr.sum())}** "
           f"({pct(out_iqr.sum(), n)}). Em dados espaciais, ponto fora do IQR não é necessariamente "
           "erro: pode ser apenas uma rodovia mais afastada do centro da malha amostrada. "
           "Só os pontos fora do DF são candidatos a erro de digitação/projeção.")
    df[fora_df | coord_nula].to_csv(TAB / "04_coordenadas_suspeitas.csv", index=False, encoding="utf-8-sig")
    if (fora_df | coord_nula).sum():
        problemas.append(f"{n_fmt((fora_df | coord_nula).sum())} registros com coordenada ausente/zero ou fora do DF "
                         "(ver `tabelas/04_coordenadas_suspeitas.csv`) - remover ou corrigir.")

    # período coberto e consistência das datas
    dmin, dmax = df["data"].min(), df["data"].max()
    fora_periodo = int(((df["data"] < "2010-04-01") | (df["data"] > "2015-03-31")).sum())
    R.item(f"Período coberto por `data_dt`: **{dmin:%d/%m/%Y} a {dmax:%d/%m/%Y}**; "
           f"registros fora de abr/2010-mar/2015: **{n_fmt(fora_periodo)}**")

    consist = []
    if "ano" in df.columns:
        ok = (df["ano"] == df["data"].dt.year)[df["data"].notna() & df["ano"].notna()]
        consist.append(("ano", "data_dt.year", ok.mean() * 100, int((~ok).sum())))
    if "mes" in df.columns:
        mnum = df["mes"].map(lambda v: MESES_PT.get(v, np.nan) if isinstance(v, str) else v)
        mnum = pd.to_numeric(mnum, errors="coerce")
        ok = (mnum == df["mes_num"])[mnum.notna() & df["mes_num"].notna()]
        consist.append(("mes", "data_dt.month", ok.mean() * 100, int((~ok).sum())))
    if "dia_da_sem" in df.columns:
        dnum = df["dia_da_sem"].map(lambda v: DIAS.index(v[:3]) if isinstance(v, str) and v[:3] in DIAS else np.nan)
        ok = (dnum == df["dia_semana_num"])[dnum.notna() & df["dia_semana_num"].notna()]
        consist.append(("dia_da_sem", "data_dt.dayofweek", ok.mean() * 100, int((~ok).sum())))
    if consist:
        tc = pd.DataFrame(consist, columns=["atributo original", "derivado de data_dt", "% concordância", "divergências"]).set_index("atributo original")
        tc.to_csv(TAB / "05_consistencia_datas.csv", encoding="utf-8-sig")
        R.p("Consistência entre campos temporais originais e `data_dt` (se ~100%, o campo é **redundante**):")
        R.tab(tc)
        for a, _, p_ok, ndiv in consist:
            if ndiv:
                problemas.append(f"`{a}` diverge de `data_dt` em {n_fmt(ndiv)} registros - conferir qual campo está certo.")

    # ------------------------------------------------------------------ 8
    R.h(2, "8. Variáveis-alvo e distribuição das classes")

    # 8.1 multiclasse
    R.h(3, "8.1 Experimento multiclasse - alvo `classe` (grupo taxonômico)")
    vc = df["classe"].value_counts(dropna=False)
    ordem_classes = [c for c in vc.index if c not in (NI,) and pd.notna(c)]
    cor_classe = {c: CORES[i % len(CORES)] for i, c in enumerate(ordem_classes)}
    mk_classe = {c: MARCADORES[i % len(MARCADORES)] for i, c in enumerate(ordem_classes)}
    tclasse = pd.DataFrame({"registros": vc, "%": 100 * vc / n})
    tclasse.index = [f"{c} ({rotulo(c)})" if c in ROTULO_CLASSE else str(c) for c in tclasse.index]
    tclasse.index.name = "classe"
    tclasse.to_csv(TAB / "06_distribuicao_classe.csv", encoding="utf-8-sig")
    R.tab(tclasse)
    validas = vc[ordem_classes]
    razao = validas.max() / validas.min()
    R.item(f"Razão de desbalanceamento (maior/menor classe): **{dec(razao)} : 1** "
           f"({rotulo(validas.idxmax())} × {rotulo(validas.idxmin())})")
    R.item(f"Acurácia de um classificador trivial que sempre prevê '{rotulo(validas.idxmax())}': "
           f"**{pct(validas.max(), validas.sum())}** - é a linha de base que qualquer modelo precisa superar; "
           "por isso a métrica principal deve ser F1 macro, não acurácia.")
    if razao > 3:
        problemas.append(f"`classe` desbalanceada ({dec(razao)}:1). Usar estratificação no split, F1 macro, "
                         "`class_weight='balanced'` e/ou reamostragem.")
    pequenas = validas[validas < 100]
    if len(pequenas):
        problemas.append("Classes com menos de 100 exemplos: " + ", ".join(
            f"{rotulo(c)} ({n_fmt(v)})" for c, v in pequenas.items()) +
            " - poucas amostras no teste (15% ≈ " + ", ".join(n_fmt(v * .15) for v in pequenas.values) + ").")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(validas)), validas.values, color=[cor_classe[c] for c in validas.index],
           width=0.65, edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(validas)), [rotulo(c) for c in validas.index])
    for i, v in enumerate(validas.values):
        ax.text(i, v, f"{n_fmt(v)}\n({pct(v, validas.sum())})", ha="center", va="bottom", fontsize=9, color=TINTA_2)
    ax.set_ylim(0, validas.max() * 1.2)
    ax.set_ylabel("registros")
    ax.set_title("Distribuição da variável-alvo classe (fauna silvestre)")
    ax.grid(axis="x", visible=False)
    salvar_fig(fig, FIG, "03_distribuicao_classe.png")
    R.fig("03_distribuicao_classe.png", "Figura 3 - Distribuição das classes taxonômicas (alvo multiclasse).")

    # 8.2 espécie
    if "especie_nm" in df.columns:
        R.h(3, "8.2 Alvo alternativo `especie_nm`")
        esp = df.loc[df["especie_nm"].notna() & (df["especie_nm"] != NI)]
        ve = esp["especie_nm"].value_counts()
        cls_esp = esp.groupby("especie_nm")["classe"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
        te = pd.DataFrame({"registros": ve, "classe": cls_esp.reindex(ve.index)})
        te.index.name = "especie_nm"
        te.to_csv(TAB / "07_distribuicao_especies.csv", encoding="utf-8-sig")
        n_esp_ni = n - len(esp)
        R.item(f"Espécies distintas identificadas: **{len(ve)}**; registros sem espécie identificada: "
               f"**{n_fmt(n_esp_ni)}** ({pct(n_esp_ni, n)})")
        lim = pd.DataFrame({"espécies com ≥ k registros": [(ve >= k).sum() for k in (10, 30, 50, 100)],
                            "registros cobertos": [ve[ve >= k].sum() for k in (10, 30, 50, 100)]},
                           index=pd.Index([10, 30, 50, 100], name="k"))
        lim["% dos registros c/ espécie"] = 100 * lim["registros cobertos"] / max(len(esp), 1)
        R.tab(lim)
        R.item(f"Espécies com apenas 1 registro: **{(ve == 1).sum()}** ({pct((ve == 1).sum(), len(ve))} das espécies)")
        R.p("Top 15 espécies:")
        R.tab(te.head(15))
        # a espécie determina a classe? (vazamento se usada como entrada)
        cls_por_esp = esp[esp["classe"].isin(ordem_classes)].groupby("especie_nm")["classe"].nunique()
        n_multi = int((cls_por_esp > 1).sum())
        R.item(f"Espécies associadas a mais de uma `classe`: **{n_multi}**"
               + (" - cada espécie pertence a uma única classe, ou seja, `especie_nm` (assim como `ordem`/`familia`) "
                  "**determina** `classe`; usar uma como entrada para prever a outra seria vazamento de informação."
                  if n_multi == 0 else " - inconsistência taxonômica a conferir (`tabelas/07b_especies_multiclasse.csv`)."))
        if n_multi:
            esp[esp["especie_nm"].isin(cls_por_esp[cls_por_esp > 1].index)].groupby(["especie_nm", "classe"]).size() \
                .to_csv(TAB / "07b_especies_multiclasse.csv", encoding="utf-8-sig")
            problemas.append(f"{n_multi} espécies aparecem com mais de uma `classe` - erro de registro a corrigir.")
        if (ve == 1).sum() > len(ve) * 0.3:
            problemas.append(f"Cauda longa de espécies: {(ve == 1).sum()} das {len(ve)} espécies têm 1 único registro; "
                             f"só {(ve >= 30).sum()} têm ≥30. Classificar por espécie exige agrupar ou filtrar (ex.: top-k ou ≥30 registros).")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
        top = te.head(15)
        pos = np.arange(len(top))
        axs[0].barh(pos, top["registros"], color=[cor_classe.get(c, CINZA) for c in top["classe"]],
                    height=0.7, edgecolor="white", linewidth=1)
        axs[0].set_yticks(pos, [s.replace("_", " ") for s in top.index], fontsize=8)
        axs[0].invert_yaxis()
        for i, v in enumerate(top["registros"]):
            axs[0].text(v, i, " " + n_fmt(v), va="center", fontsize=8, color=TINTA_2)
        axs[0].set_title("15 espécies mais atropeladas")
        axs[0].set_xlabel("registros")
        axs[0].grid(axis="y", visible=False)
        from matplotlib.patches import Patch
        cls_top = [c for c in ordem_classes if c in set(top["classe"])]
        axs[0].legend(handles=[Patch(color=cor_classe[c], label=rotulo(c)) for c in cls_top],
                      loc="lower right", fontsize=8)
        axs[1].plot(np.arange(1, len(ve) + 1), ve.values, color=CORES[0], lw=2)
        axs[1].set_yscale("log")
        axs[1].axhline(30, color=CINZA, lw=1, ls="--")
        axs[1].text(len(ve) * 0.98, 32, "30 registros", ha="right", va="bottom", fontsize=8, color=TINTA_2)
        axs[1].set_title("Cauda longa: registros por espécie (ordenado)")
        axs[1].set_xlabel("posição da espécie no ranking")
        axs[1].set_ylabel("registros (escala log)")
        salvar_fig(fig, FIG, "04_especies.png")
        R.fig("04_especies.png", "Figura 4 - Espécies mais frequentes e distribuição de cauda longa.")

    # 8.3 binário (hotspot por célula)
    R.h(3, f"8.3 Experimento binário - hotspot por célula de {CEL} m")
    cel = df.dropna(subset=["x", "y"]).loc[~fora_df].groupby(["cel_x", "cel_y"]).size().rename("registros").reset_index()
    vcel = cel["registros"]
    qs = vcel.quantile([.5, .75, .9, .95])
    R.p(f"A base só contém **ocorrências**, então não existe classe negativa pronta. Como primeiro passo, os "
        f"registros foram agregados em células de {CEL} m × {CEL} m (aproximação de segmento rodoviário de "
        f"~{CEL} m). Há **{n_fmt(len(cel))} células com pelo menos 1 atropelamento**.")
    tcel = pd.DataFrame({"valor": [vcel.mean(), vcel.median(), vcel.std(), vcel.max(), qs[.75], qs[.9], qs[.95],
                                   (vcel == 1).mean() * 100]},
                        index=pd.Index(["média", "mediana", "desvio-padrão", "máximo", "P75", "P90", "P95",
                                        "% células com 1 só registro"], name=f"registros por célula ocupada ({CEL} m)"))
    R.tab(tcel)
    cand = []
    for nome_lim, lim_v in [("≥ 2 registros", 2), (f"≥ P75 ({qs[.75]:.0f})", qs[.75]), (f"≥ P90 ({qs[.9]:.0f})", qs[.9])]:
        pos_c = int((vcel >= lim_v).sum())
        cand.append((nome_lim, pos_c, 100 * pos_c / len(cel), int(vcel[vcel >= lim_v].sum()), 100 * vcel[vcel >= lim_v].sum() / vcel.sum()))
    tcand = pd.DataFrame(cand, columns=["critério de alto risco", "células positivas", "% das células ocupadas",
                                        "atropelamentos nessas células", "% dos atropelamentos"]).set_index("critério de alto risco")
    R.p("Candidatos de limiar para a classe 1 (alto risco), entre as células **ocupadas**:")
    R.tab(tcand)
    R.p("**Atenção para a Etapa 2:** a classe 0 não pode ser \"qualquer lugar sem ponto\". O caminho "
        "defensável é obter a geometria das rodovias percorridas (ex.: OpenStreetMap ou malha do DER-DF), "
        f"dividi-las em segmentos de {CEL} m e contar os atropelamentos por segmento; segmentos "
        "monitorados com 0 ou poucos registros viram a classe 0. Sem a malha, uma alternativa é "
        "classificar só entre células ocupadas (baixo × alto risco pelo limiar acima), deixando "
        "explícito que o modelo distingue intensidade, não presença/ausência.")
    sens = []
    for t in (250, 500, 1000, 2000):
        g = df.loc[~fora_df].dropna(subset=["x", "y"]).groupby([df["x"] // t, df["y"] // t]).size()
        sens.append((t, len(g), g.mean(), g.median(), (g == 1).mean() * 100, g.max()))
    tsens = pd.DataFrame(sens, columns=["célula (m)", "células ocupadas", "média reg./célula", "mediana",
                                        "% células c/ 1 registro", "máximo"]).set_index("célula (m)")
    tsens.to_csv(TAB / "08_sensibilidade_tamanho_celula.csv", encoding="utf-8-sig")
    R.p("Sensibilidade ao tamanho da célula (ajuda a justificar a unidade espacial):")
    R.tab(tsens)
    cel.sort_values("registros", ascending=False).to_csv(TAB / "09_contagem_por_celula.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    m = int(min(vcel.max(), max(qs[.95] * 2, 10)))
    cont = vcel.clip(upper=m).value_counts().sort_index()
    ax.bar(cont.index, cont.values, color=CORES[0], width=0.75, edgecolor="white", linewidth=1)
    ax.axvline(qs[.75] - 0.5, color=CORES[1], lw=2, ls="--")
    ax.text(qs[.75] + 0.6, cont.max() * 0.9, f"P75 = {qs[.75]:.0f}", color=TINTA_2, fontsize=9)
    ax.set_xlabel(f"atropelamentos na célula de {CEL} m (último valor = '{m} ou mais')")
    ax.set_ylabel("células")
    ax.set_title(f"Distribuição de atropelamentos por célula ocupada ({CEL} m)")
    ax.grid(axis="x", visible=False)
    salvar_fig(fig, FIG, "05_registros_por_celula.png")
    R.fig("05_registros_por_celula.png", f"Figura 5 - Quantas células de {CEL} m têm 1, 2, 3... atropelamentos.")

    # 8.4 regressão
    R.h(3, f"8.4 Experimento de regressão - atropelamentos por célula de {CEL} m por mês")
    dval = df.loc[~fora_df & df["data"].notna()].dropna(subset=["x", "y"])
    meses = pd.period_range(dval["ano_mes"].min(), dval["ano_mes"].max(), freq="M")
    cont_cm = dval.groupby(["celula", "ano_mes"]).size()
    painel = cont_cm.reindex(pd.MultiIndex.from_product([dval["celula"].unique(), meses],
                                                         names=["celula", "ano_mes"]), fill_value=0)
    R.p(f"Para a regressão os registros precisam ser agregados. Unidade proposta: **célula de {CEL} m × mês**, "
        f"completando com 0 os meses sem registro (assumindo que o trecho foi percorrido todo mês; ver "
        f"esforço amostral na seção 10). Painel: {n_fmt(dval['celula'].nunique())} células × {len(meses)} meses = "
        f"**{n_fmt(len(painel))} observações**.")
    treg = pd.DataFrame({"valor": [painel.mean(), painel.var(), painel.var() / painel.mean(), (painel == 0).mean() * 100,
                                   painel.max()]},
                        index=pd.Index(["média", "variância", "variância / média (dispersão)", "% de zeros", "máximo"],
                                       name="contagem célula×mês"))
    R.tab(treg)
    R.item("Variância/média > 1 indica **sobredispersão** e a grande proporção de zeros indica **inflação de zeros**: "
           "é esperado que MAE/RMSE fiquem baixos só por prever ~0, então convém comparar com um baseline "
           "(média histórica da célula) e considerar agregação mais grossa (célula × trimestre/estação, "
           "ou `uc` × mês) se o sinal for fraco.")
    if (painel == 0).mean() > 0.8:
        problemas.append(f"Alvo de regressão célula×mês tem {(painel == 0).mean() * 100:.0f}% de zeros - "
                         "considerar agregação temporal maior (trimestre/estação) ou espacial maior (1 km).")
    alt = []
    if "uc" in dval.columns:
        p_uc = dval.groupby(["uc", "ano_mes"]).size().reindex(
            pd.MultiIndex.from_product([dval["uc"].dropna().unique(), meses]), fill_value=0)
        alt.append(("uc × mês", len(p_uc), p_uc.mean(), p_uc.var() / p_uc.mean(), (p_uc == 0).mean() * 100))
    dval_tri = dval.assign(tri=dval["data"].dt.to_period("Q"))
    p_tri = dval_tri.groupby(["celula", "tri"]).size().reindex(
        pd.MultiIndex.from_product([dval["celula"].unique(), pd.period_range(dval_tri["tri"].min(), dval_tri["tri"].max(), freq="Q")]),
        fill_value=0)
    alt.append((f"célula {CEL} m × trimestre", len(p_tri), p_tri.mean(), p_tri.var() / p_tri.mean(), (p_tri == 0).mean() * 100))
    alt.insert(0, (f"célula {CEL} m × mês", len(painel), painel.mean(), painel.var() / painel.mean(), (painel == 0).mean() * 100))
    talt = pd.DataFrame(alt, columns=["unidade", "observações", "média", "var/média", "% zeros"]).set_index("unidade")
    talt.to_csv(TAB / "10_unidades_regressao.csv", encoding="utf-8-sig")
    R.p("Comparação de unidades de agregação candidatas:")
    R.tab(talt)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    vc_p = painel.value_counts().sort_index()
    ax.bar(vc_p.index, vc_p.values, color=CORES[0], width=0.75, edgecolor="white", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("atropelamentos na célula no mês")
    ax.set_ylabel("observações (escala log)")
    ax.set_title(f"Alvo de regressão: contagem por célula ({CEL} m) × mês")
    ax.grid(axis="x", visible=False)
    salvar_fig(fig, FIG, "06_alvo_regressao.png")
    R.fig("06_alvo_regressao.png", "Figura 6 - Distribuição do alvo de regressão (observe a massa de zeros).")

    # ------------------------------------------------------------------ 9
    R.h(2, "9. Distribuição espacial dos atropelamentos")
    esp_ok = df.loc[~fora_df & ~coord_nula]
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in ordem_classes[::-1]:  # classe menos frequente por cima
        s = esp_ok[esp_ok["classe"] == c]
        ax.scatter(s["x"], s["y"], s=10, color=cor_classe[c], marker=mk_classe[c], alpha=0.6,
                   linewidths=0.3, edgecolors="white", label=f"{rotulo(c)} ({n_fmt(len(s))})")
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain")
    ax.set_xlabel("x (UTM 23S, m)")
    ax.set_ylabel("y (UTM 23S, m)")
    ax.set_title("Ocorrências de atropelamento de fauna silvestre no DF")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="best", markerscale=1.8)
    ax.tick_params(axis="x", rotation=30)
    salvar_fig(fig, FIG, "07_mapa_ocorrencias.png")
    R.fig("07_mapa_ocorrencias.png", "Figura 7 - Mapa (UTM) das ocorrências por classe. Os pontos desenham a malha rodoviária percorrida.")

    k = len(ordem_classes)
    ncol = 2 if k <= 4 else 3
    nlin = int(np.ceil(k / ncol))
    razao_xy = (esp_ok["y"].max() - esp_ok["y"].min()) / max(esp_ok["x"].max() - esp_ok["x"].min(), 1)
    fig, axs = plt.subplots(nlin, ncol, figsize=(5 * ncol, (5 * razao_xy + 0.5) * nlin + 0.4),
                            sharex=True, sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, c in zip(axs, ordem_classes):
        ax.scatter(esp_ok["x"], esp_ok["y"], s=3, color="#dedcd6", linewidths=0)
        s = esp_ok[esp_ok["classe"] == c]
        ax.scatter(s["x"], s["y"], s=7, color=cor_classe[c], linewidths=0, alpha=0.8)
        ax.set_title(f"{rotulo(c)} (n = {n_fmt(len(s))})")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[k:]:
        ax.axis("off")
    fig.suptitle("Distribuição espacial por classe (cinza = todas as ocorrências)", fontweight="bold")
    salvar_fig(fig, FIG, "08_mapa_por_classe.png")
    R.fig("08_mapa_por_classe.png", "Figura 8 - Mapas por classe taxonômica.")

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(esp_ok["x"], esp_ok["y"], gridsize=60, cmap=CMAP_SEQ, mincnt=1, bins="log", linewidths=0.2)
    top_cel = cel.sort_values("registros", ascending=False).head(10)
    for i, r in enumerate(top_cel.itertuples(), 1):
        cx, cy = (r.cel_x + 0.5) * CEL, (r.cel_y + 0.5) * CEL
        ax.annotate(str(i), (cx, cy), xytext=(6, 6), textcoords="offset points", fontsize=8, color=TINTA,
                    fontweight="bold")
        ax.plot(cx, cy, marker="o", ms=9, mfc="none", mec=CORES[1], mew=1.6)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain")
    ax.tick_params(axis="x", rotation=30)
    cb = fig.colorbar(hb, ax=ax, shrink=0.7)
    cb.set_label("atropelamentos no hexágono (escala log)")
    ax.set_title(f"Densidade de atropelamentos e\nas 10 células de {CEL} m mais críticas")
    salvar_fig(fig, FIG, "09_densidade_hotspots.png")
    R.fig("09_densidade_hotspots.png", "Figura 9 - Densidade espacial (hexbin) e as 10 células com mais registros (círculos numerados).")
    R.p(f"10 células de {CEL} m com mais atropelamentos (centro da célula em UTM):")
    tt = top_cel.assign(x_centro=(top_cel["cel_x"] + 0.5) * CEL, y_centro=(top_cel["cel_y"] + 0.5) * CEL)
    if "uc" in df.columns:
        uc_cel = df.groupby(["cel_x", "cel_y"])["uc"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        tt["uc predominante"] = [uc_cel.get((a, b), np.nan) for a, b in zip(tt["cel_x"], tt["cel_y"])]
    tt = tt.drop(columns=["cel_x", "cel_y"]).reset_index(drop=True)
    tt.index = tt.index + 1
    tt.index.name = "rank"
    R.tab(tt)
    R.item(f"As 10 células mais críticas concentram **{pct(top_cel['registros'].sum(), vcel.sum())}** dos "
           f"atropelamentos com coordenada válida, ocupando {pct(10, len(cel))} das células ocupadas.")

    # UC
    if "uc" in df.columns:
        R.h(3, "9.1 Unidade de conservação (`uc`)")
        vu = df["uc"].fillna("(ausente)").value_counts()
        tu = pd.DataFrame({"registros": vu, "%": 100 * vu / n})
        tu.index.name = "uc"
        tu.to_csv(TAB / "11_distribuicao_uc.csv", encoding="utf-8-sig")
        R.tab(tu)
        ct = pd.crosstab(df["uc"].fillna("(ausente)"), df["classe"])[ordem_classes]
        ctn = ct.div(ct.sum(axis=1), axis=0) * 100
        ctn.to_csv(TAB / "12_uc_x_classe_pct.csv", encoding="utf-8-sig")
        fig, axs = plt.subplots(1, 2, figsize=(12, 0.45 * len(vu) + 1.8), gridspec_kw={"width_ratios": [1, 1.2]})
        barras(axs[0], list(vu.index), vu.values, horizontal=True, fmt_pct_total=n)
        axs[0].set_title("Registros por UC")
        axs[0].set_xlim(0, vu.max() * 1.35)
        sns.heatmap(ctn.loc[vu.index], annot=True, fmt=".0f", cmap=CMAP_SEQ, cbar_kws={"label": "% da linha"},
                    ax=axs[1], linewidths=1, linecolor="white",
                    xticklabels=[rotulo(c) for c in ordem_classes], yticklabels=False)
        axs[1].set_title("Composição por classe em cada UC (% da linha)")
        axs[1].set_xlabel(""); axs[1].set_ylabel("")
        salvar_fig(fig, FIG, "10_uc.png")
        R.fig("10_uc.png", "Figura 10 - Registros por unidade de conservação e composição taxonômica.")
        v_uc = cramers_v(df["uc"].fillna("(ausente)"), df["classe"])
        chi = stats.chi2_contingency(ct)
        R.item(f"Associação `uc` × `classe`: χ² = {dec(chi[0])}, p = {chi[1]:.2e}, V de Cramér = **{dec(v_uc, 3)}**")
        raras_uc = vu[vu < 30]
        if len(raras_uc):
            problemas.append(f"`uc` tem {len(raras_uc)} categoria(s) com <30 registros ({', '.join(raras_uc.index)}) - "
                             "agrupar em 'outras' antes do One-Hot Encoding.")

    # autocorrelação espacial -> implicação para o split
    R.h(3, "9.2 Proximidade entre ocorrências (implicação para a divisão treino/teste)")
    xy = esp_ok[["x", "y"]].to_numpy(dtype=float)
    arv = cKDTree(xy)
    dist, idx = arv.query(xy, k=2)
    dnn = dist[:, 1]
    R.item(f"Distância até a ocorrência mais próxima: mediana **{np.median(dnn):.0f} m**, "
           f"P25 = {np.percentile(dnn, 25):.0f} m, P75 = {np.percentile(dnn, 75):.0f} m")
    R.item(f"Ocorrências com outra a menos de 100 m: **{pct((dnn < 100).sum(), len(dnn))}**; "
           f"a menos de 500 m: **{pct((dnn < 500).sum(), len(dnn))}**; "
           f"coordenada idêntica a outra: **{pct((dnn == 0).sum(), len(dnn))}**")
    viz = esp_ok[esp_ok["classe"].isin(ordem_classes)]
    _, idx_v = cKDTree(viz[["x", "y"]].to_numpy(dtype=float)).query(viz[["x", "y"]].to_numpy(dtype=float), k=2)
    cls_arr = viz["classe"].to_numpy()
    mesma = cls_arr == cls_arr[idx_v[:, 1]]
    pk = viz["classe"].value_counts(normalize=True)
    esperado = float((pk ** 2).sum())
    R.item(f"Proporção de ocorrências cujo vizinho mais próximo é da **mesma classe**: **{dec(mesma.mean() * 100)}%** "
           f"(esperado ao acaso, dadas as proporções: {dec(esperado * 100)}%).")
    R.p("Com split aleatório, quase todo ponto de teste terá um vizinho a poucas dezenas de metros no "
        "treino. Se o vizinho tende a ter a mesma classe (valor acima do esperado ao acaso), um modelo "
        "baseado em x,y pode \"decorar\" o local e a métrica de teste fica otimista. Isso motiva o "
        "experimento com **split espacial** (ex.: blocos de vários km ou por UC/rodovia) e/ou **temporal** "
        "(ex.: treinar 2010-2013 e testar 2014-2015) na Etapa 2.")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(np.clip(dnn, 0, 2000), bins=50, color=CORES[0], edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(dnn), color=CORES[1], lw=2, ls="--")
    ax.text(np.median(dnn) + 30, ax.get_ylim()[1] * 0.9, f"mediana = {np.median(dnn):.0f} m", fontsize=9, color=TINTA_2)
    ax.set_xlabel("distância até a ocorrência mais próxima (m, truncado em 2 km)")
    ax.set_ylabel("ocorrências")
    ax.set_title("Ocorrências muito próximas entre si")
    ax.grid(axis="x", visible=False)
    salvar_fig(fig, FIG, "11_distancia_vizinho.png")
    R.fig("11_distancia_vizinho.png", "Figura 11 - Distância ao vizinho mais próximo: base do argumento contra o split aleatório.")
    if mesma.mean() > esperado + 0.05:
        problemas.append(f"Autocorrelação espacial do alvo: {mesma.mean() * 100:.0f}% dos vizinhos mais próximos "
                         f"têm a mesma classe (acaso: {esperado * 100:.0f}%). Split aleatório tende a superestimar o desempenho.")

    # ------------------------------------------------------------------ 10
    R.h(2, "10. Distribuição temporal dos atropelamentos")
    dt_ok = df[df["data"].notna()]
    # esforço amostral (usa a base bruta: campanhas mesmo sem registro silvestre)
    if "campanha" in bruto.columns:
        bruto["ano_mes"] = bruto["data"].dt.to_period("M")
        esforco = bruto.dropna(subset=["ano_mes"]).groupby("ano_mes")["campanha"].nunique()
    else:
        esforco = bruto.dropna(subset=["data"]).groupby(bruto["data"].dt.to_period("M"))["data"].apply(lambda s: s.dt.date.nunique())
    serie = dt_ok.groupby("ano_mes").size().reindex(pd.period_range(dt_ok["ano_mes"].min(), dt_ok["ano_mes"].max(), freq="M"), fill_value=0)
    esforco = esforco.reindex(serie.index, fill_value=0)
    taxa = (serie / esforco.replace(0, np.nan))

    por_ano = dt_ok.groupby(dt_ok["data"].dt.year).agg(registros=("data", "size"), meses_com_registro=("ano_mes", "nunique"))
    por_ano["registros por mês"] = por_ano["registros"] / por_ano["meses_com_registro"]
    if "campanha" in bruto.columns:
        camp_ano = bruto.dropna(subset=["data"]).groupby(bruto["data"].dt.year)["campanha"].nunique()
        por_ano["campanhas"] = camp_ano.reindex(por_ano.index)
        por_ano["registros por campanha"] = por_ano["registros"] / por_ano["campanhas"]
    por_ano.index = por_ano.index.astype(str)
    por_ano.index.name = "ano"
    por_ano.to_csv(TAB / "13_registros_por_ano.csv", encoding="utf-8-sig")
    R.tab(por_ano)
    R.p("2010 (a partir de abril) e 2015 (até março) são anos **incompletos**; comparar anos pela contagem "
        "bruta é enganoso - usar registros por mês ou por campanha.")

    fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    xs = serie.index.to_timestamp()
    axs[0].plot(xs, serie.values, color=CORES[0], lw=2, marker="o", ms=3)
    for a in range(xs.min().year, xs.max().year + 1):  # destaca estação seca
        ini, fim = pd.Timestamp(a, 5, 1), pd.Timestamp(a, 9, 30)
        if ini <= xs.max() and fim >= xs.min():
            for ax_ in axs:
                ax_.axvspan(max(ini, xs.min()), min(fim, xs.max()), color="#f3efe4", zorder=0)
    axs[1].set_xlim(xs.min() - pd.Timedelta(days=20), xs.max() + pd.Timedelta(days=20))
    axs[0].set_ylabel("registros no mês")
    axs[0].set_title("Atropelamentos por mês (faixas bege = estação seca, mai-set)")
    axs[1].plot(xs, taxa.values, color=CORES[2], lw=2, marker="o", ms=3)
    axs[1].set_ylabel("registros por campanha")
    axs[1].set_title("Normalizado pelo esforço amostral (campanhas no mês)")
    salvar_fig(fig, FIG, "12_serie_mensal.png")
    R.fig("12_serie_mensal.png", "Figura 12 - Série mensal bruta e normalizada pelo esforço amostral.")
    R.item(f"Meses com 0 campanhas no período: **{int((esforco == 0).sum())}**; campanhas por mês: "
           f"mínimo {esforco.min():.0f}, mediana {esforco.median():.0f}, máximo {esforco.max():.0f}. "
           "Se o esforço varia, a contagem mensal mistura risco com intensidade de amostragem.")
    if esforco.max() > 0 and esforco[esforco > 0].std() / esforco[esforco > 0].mean() > 0.3:
        problemas.append("Esforço amostral (campanhas/mês) varia bastante - na regressão, normalizar pela quantidade de "
                         "campanhas ou incluí-la como variável de exposição (sem usar `campanha` como identificador).")

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.6, 1, 0.7]})
    pm = dt_ok["mes_num"].value_counts().reindex(range(1, 13), fill_value=0)
    cores_m = [CORES[3] if 5 <= m <= 9 else CORES[0] for m in pm.index]
    axs[0].bar(range(12), pm.values, color=cores_m, width=0.7, edgecolor="white", linewidth=1)
    axs[0].set_xticks(range(12), MESES_ABREV)
    axs[0].set_title("Por mês do ano")
    axs[0].set_ylabel("registros")
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color=CORES[0], label="chuvoso (out-abr)"), Patch(color=CORES[3], label="seco (mai-set)")],
                  fontsize=8, loc="upper right")
    pdw = dt_ok["dia_semana_num"].value_counts().reindex(range(7), fill_value=0)
    axs[1].bar(range(7), pdw.values, color=[CORES[0]] * 5 + [CORES[2]] * 2, width=0.7, edgecolor="white", linewidth=1)
    axs[1].set_xticks(range(7), DIAS)
    axs[1].set_title("Por dia da semana")
    pc = dt_ok["periodo_climatico"].value_counts().reindex(["chuvoso", "seco"])
    axs[2].bar([0, 1], pc.values, color=[CORES[0], CORES[3]], width=0.6, edgecolor="white", linewidth=1)
    axs[2].set_xticks([0, 1], ["chuvoso\n(7 meses)", "seco\n(5 meses)"])
    for i, v in enumerate(pc.values):
        axs[2].text(i, v, f"{n_fmt(v)}\n{v / (7 if i == 0 else 5):.0f}/mês", ha="center", va="bottom", fontsize=8, color=TINTA_2)
    axs[2].set_ylim(0, pc.max() * 1.25)
    axs[2].set_title("Por período climático")
    for ax in axs:
        ax.grid(axis="x", visible=False)
    salvar_fig(fig, FIG, "13_temporal_mes_semana.png")
    R.fig("13_temporal_mes_semana.png", "Figura 13 - Atropelamentos por mês do ano, dia da semana e período climático.")

    # dia da semana: o esforço pode ser concentrado em dias úteis
    if "campanha" in bruto.columns:
        dias_camp = bruto.dropna(subset=["data"]).groupby(bruto["data"].dt.dayofweek)["campanha"].nunique().reindex(range(7), fill_value=0)
        tdw = pd.DataFrame({"registros": pdw.values, "campanhas": dias_camp.values}, index=pd.Index(DIAS, name="dia"))
        tdw["registros por campanha"] = tdw["registros"] / tdw["campanhas"].replace(0, np.nan)
        tdw.to_csv(TAB / "14_dia_semana.csv", encoding="utf-8-sig")
        R.p("Registros e campanhas por dia da semana (se não há campanhas no fim de semana, `fim_de_semana` "
            "reflete a **agenda da equipe**, não o risco):")
        R.tab(tdw)
        if (dias_camp.iloc[5:] == 0).all():
            problemas.append("Não há campanhas aos sábados/domingos: `dia_da_sem`/`fim_de_semana` mede a rotina de "
                             "amostragem, não o tráfego - pouco útil como preditor.")

    ct_m = pd.crosstab(dt_ok["classe"], dt_ok["mes_num"]).reindex(index=ordem_classes, columns=range(1, 13), fill_value=0)
    ct_mn = ct_m.div(ct_m.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(ordem_classes) + 1.4))
    sns.heatmap(ct_mn, annot=True, fmt=".0f", cmap=CMAP_SEQ, ax=ax, linewidths=1, linecolor="white",
                cbar_kws={"label": "% dos registros da classe"},
                xticklabels=MESES_ABREV, yticklabels=[rotulo(c) for c in ordem_classes])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("Sazonalidade por classe (% de cada classe em cada mês)")
    salvar_fig(fig, FIG, "14_sazonalidade_classe.png")
    R.fig("14_sazonalidade_classe.png", "Figura 14 - Sazonalidade por classe (cada linha soma 100%).")
    ct_m.to_csv(TAB / "15_classe_x_mes.csv", encoding="utf-8-sig")

    ct_pc = pd.crosstab(dt_ok["classe"], dt_ok["periodo_climatico"])
    chi_pc = stats.chi2_contingency(ct_pc)
    R.item(f"`classe` × `periodo_climatico`: χ² = {dec(chi_pc[0])}, p = {chi_pc[1]:.2e}, "
           f"V de Cramér = **{dec(cramers_v(dt_ok['classe'], dt_ok['periodo_climatico']), 3)}**")
    ct_ano = pd.crosstab(dt_ok["classe"], dt_ok["data"].dt.year)
    chi_a = stats.chi2_contingency(ct_ano)
    R.item(f"`classe` × `ano`: χ² = {dec(chi_a[0])}, p = {chi_a[1]:.2e}, "
           f"V de Cramér = **{dec(cramers_v(dt_ok['classe'], dt_ok['data'].dt.year), 3)}**")

    # ------------------------------------------------------------------ 11
    R.h(2, "11. Correlação e redundância entre atributos")
    cand_num = [c for c in ["x", "y", "ano", "mes_num", "dia_do_ano", "dia_semana_num", "fim_de_semana"] if c in df.columns]
    dnum = df[cand_num].astype(float)
    sp = dnum.corr(method="spearman")
    sp.to_csv(TAB / "16_correlacao_spearman.csv", encoding="utf-8-sig")
    cats = [c for c in ["classe", "uc", "estacao", "periodo_climatico", "dia_da_sem"] if c in df.columns]
    dcat = df[cats].copy()
    dcat["ano"] = df["ano"] if "ano" in df.columns else df["data"].dt.year
    dcat["mes_num"] = df["mes_num"]
    cats2 = list(dcat.columns)
    V = pd.DataFrame(np.eye(len(cats2)), index=cats2, columns=cats2)
    for i, a in enumerate(cats2):
        for b in cats2[i + 1:]:
            ok = dcat[a].notna() & dcat[b].notna()
            V.loc[a, b] = V.loc[b, a] = cramers_v(dcat.loc[ok, a].astype(str), dcat.loc[ok, b].astype(str))
    V.to_csv(TAB / "17_cramers_v.csv", encoding="utf-8-sig")

    fig, axs = plt.subplots(1, 2, figsize=(16, 6.2))
    fig.subplots_adjust(wspace=0.55)
    mask = np.triu(np.ones_like(sp, dtype=bool), 1)
    sns.heatmap(sp, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=axs[0],
                linewidths=1, linecolor="white", cbar_kws={"label": "ρ de Spearman"})
    axs[0].set_title("Variáveis numéricas (Spearman)")
    mask = np.triu(np.ones_like(V, dtype=bool), 1)
    sns.heatmap(V.astype(float), mask=mask, annot=True, fmt=".2f", cmap=CMAP_SEQ, vmin=0, vmax=1, ax=axs[1],
                linewidths=1, linecolor="white", cbar_kws={"label": "V de Cramér"})
    axs[1].set_title("Variáveis categóricas (V de Cramér)")
    for ax in axs:
        ax.tick_params(axis="x", rotation=40)
    salvar_fig(fig, FIG, "15_correlacoes.png")
    R.fig("15_correlacoes.png", "Figura 15 - Correlação entre numéricas (Spearman) e associação entre categóricas (V de Cramér).")

    pares = []
    for M, tipo in ((sp, "Spearman"), (V.astype(float), "Cramér")):
        cols = list(M.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                v = M.loc[a, b]
                if pd.notna(v) and abs(v) >= 0.7:
                    pares.append((f"{a} × {b}", tipo, v))
    if pares:
        tp = pd.DataFrame(pares, columns=["par", "medida", "valor"]).set_index("par")
        R.p("Pares com associação forte (|valor| ≥ 0,7):")
        R.tab(tp)
        problemas.append("Atributos redundantes/fortemente associados: " + "; ".join(f"{p} ({dec(v, 2)})" for p, _, v in pares) +
                         ". Manter apenas um de cada par (ou usar representação cíclica seno/cosseno para mês/dia do ano).")
    R.p("Redundâncias **por construção** (independentes do coeficiente):")
    R.item("`ano`, `mes`, `dia_da_sem` são deriváveis de `data_dt` (ver tabela de consistência na seção 7).")
    R.item("`estacao` e `periodo_climatico` são funções de `mes_num`; `dia_do_ano` ≈ `mes_num` refinado. "
           "Para a MLP, recomenda-se codificar mês/dia do ano de forma **cíclica** (seno e cosseno), evitando a "
           "falsa distância entre dezembro (12) e janeiro (1).")
    R.item("`ordem` → `familia` → `especie_nm` → `classe` formam uma hierarquia: qualquer um deles revela a classe (vazamento).")
    if "uc" in df.columns:
        pur = df.dropna(subset=["uc"]).groupby("celula")["uc"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        R.item(f"`uc` é praticamente função da posição: em média **{dec(pur.mean() * 100)}%** dos registros de uma célula "
               f"de {CEL} m têm a mesma `uc` ({pct((pur == 1).sum(), len(pur))} das células são 100% de uma única UC). "
               "Com `x`,`y` no modelo, `uc` acrescenta pouca informação nova - mas é mais interpretável.")

    # ------------------------------------------------------------------ 12
    R.h(2, "12. Resumo dos problemas encontrados e encaminhamento para a Etapa 2")
    if problemas:
        for p_ in problemas:
            R.item(p_)
    else:
        R.item("Nenhum problema relevante detectado automaticamente.")

    R.h(2, "13. Variáveis-alvo propostas por experimento")
    alvos = pd.DataFrame([
        ("Classificação multiclasse", "classe (aves, mamíferos, répteis, anfíbios)", "registro individual",
         "x, y (ou célula), uc, ano, mês (cíclico), dia do ano, período climático, estação"),
        ("Classificação binária", f"hotspot = 1 se nº de atropelamentos no segmento/célula de {CEL} m ≥ limiar", f"célula/segmento de {CEL} m",
         "posição do centro, uc, densidade na vizinhança (excluindo a própria célula), dados externos"),
        ("Regressão", "nº de atropelamentos", f"célula de {CEL} m × mês (ou × trimestre)",
         "posição, uc, mês (cíclico), período climático, esforço (campanhas), histórico defasado"),
    ], columns=["experimento", "variável-alvo", "unidade de análise", "atributos de entrada candidatos"]).set_index("experimento")
    alvos.to_csv(TAB / "18_alvos_por_experimento.csv", encoding="utf-8-sig")
    R.tab(alvos)

    # salva base intermediária e resumo
    df.drop(columns=["ano_mes"]).to_csv(saida / "base_silvestre_eda.csv", index=False, encoding="utf-8-sig")
    R.salvar(saida / "resumo_etapa1.md")
    print(f"Pronto. {len(list(FIG.glob('*.png')))} figuras, {len(list(TAB.glob('*.csv')))} tabelas.")
    print(f"Resumo: {saida / 'resumo_etapa1.md'}")


if __name__ == "__main__":
    main()
