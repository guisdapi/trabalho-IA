"""
Etapa 1 - Analise Exploratoria - RodoFauna (IBRAM/DF)
Uso: python etapa1_eda_rodofauna.py > resumo.txt   (os numeros vao para resumo.txt)
Os graficos sao salvos na pasta figuras/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ARQUIVO = "rodofauna_bruto.csv"
URL = ("https://onda.ibram.df.gov.br/server/rest/services/Monitoramento_Ambiental/"
       "Programa_Rodofauna_Atropelada/FeatureServer/2/query")
MIN_EXEMPLOS = 30
os.makedirs("figuras", exist_ok=True)
sns.set_style("whitegrid")


def fig(nome):
    plt.tight_layout(); plt.savefig(f"figuras/{nome}", dpi=150); plt.close()


# ---------- Carga e filtro ----------
if os.path.exists(ARQUIVO):
    df = pd.read_csv(ARQUIVO)
else:
    import requests
    p = {"where": "1=1", "outFields": "*", "f": "json", "returnGeometry": "false",
         "resultOffset": 0, "resultRecordCount": 1000}
    rows = []
    while True:
        feats = requests.get(URL, params=p, timeout=60).json().get("features", [])
        if not feats:
            break
        rows += [f["attributes"] for f in feats]
        p["resultOffset"] += len(feats)
    df = pd.DataFrame(rows)
    df.to_csv(ARQUIVO, index=False)

df.columns = df.columns.str.strip().str.lower()
for c in df.select_dtypes("object"):
    df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})

d = df[df["silvestre_"] == "silvestre"].copy()
if pd.api.types.is_numeric_dtype(d["data_dt"]):
    d["data_dt"] = pd.to_datetime(d["data_dt"], unit="ms", errors="coerce")
else:
    d["data_dt"] = pd.to_datetime(d["data_dt"], dayfirst=True, errors="coerce")
d["ano_dt"] = d["data_dt"].dt.year
d["mes_num"] = d["data_dt"].dt.month
d["dia_sem"] = d["data_dt"].dt.dayofweek

# ---------- 1. Estrutura ----------
print("== REGISTROS E ATRIBUTOS ==")
print("Registros totais:", len(df), "| silvestres:", len(d))
print("Atributos:", d.shape[1])
print("\n== TIPOS ==")
print(d.dtypes.to_string())

# ---------- 2. Ausentes ----------
print("\n== VALORES AUSENTES ==")
aus = pd.DataFrame({"nan": d.isna().sum()})
aus["nao_identificado"] = [(d[c] == "nao_identificado").sum() if d[c].dtype == "object" else 0
                           for c in d.columns]
aus["% vazio total"] = ((aus["nan"] + aus["nao_identificado"]) / len(d) * 100).round(2)
print(aus.sort_values("% vazio total", ascending=False).to_string())

# ---------- 3. Alvo: distribuicao das classes ----------
print("\n== DISTRIBUICAO DE 'classe' ==")
cont = d["classe"].value_counts()
print(pd.DataFrame({"n": cont, "%": (cont / cont.sum() * 100).round(2)}).to_string())
print(f"Razao de desbalanceamento (maior/menor): {cont.max() / cont.min():.2f}")
plt.figure(figsize=(7, 4))
sns.countplot(data=d, y="classe", order=cont.index, color="steelblue")
plt.title("Registros por classe taxonomica")
fig("classes.png")

esp = d["especie_nm"].value_counts()
print("\n== ESPECIES ==")
print("Numero de especies:", len(esp))
print(f"Especies com < {MIN_EXEMPLOS} registros:", int((esp < MIN_EXEMPLOS).sum()))
print(esp.head(10).to_string())
plt.figure(figsize=(8, 6))
esp.head(20).sort_values().plot.barh()
plt.title("20 especies mais frequentes")
fig("top20_especies.png")

# ---------- 4. Estatisticas descritivas ----------
print("\n== ESTATISTICAS DESCRITIVAS ==")
num = [c for c in ["ano", "x", "y"] if c in d.columns]
print(d[num].describe().T.round(2).to_string())

# ---------- 5. Espacial ----------
plt.figure(figsize=(8, 8))
plt.scatter(d["x"], d["y"], s=3, alpha=0.3)
plt.axis("equal"); plt.title("Distribuicao espacial (UTM)")
fig("mapa_pontos.png")

plt.figure(figsize=(9, 8))
plt.hexbin(d["x"], d["y"], gridsize=60, cmap="YlOrRd", mincnt=1)
plt.colorbar(label="Atropelamentos"); plt.axis("equal")
plt.title("Densidade de atropelamentos")
fig("mapa_densidade.png")

print("\n== REGISTROS POR UC ==")
print(d["uc"].value_counts(dropna=False).to_string())

# ---------- 6. Temporal ----------
print("\n== TEMPORAL ==")
print("Periodo:", d["data_dt"].min().date(), "a", d["data_dt"].max().date())
print("Por ano:\n", d["ano_dt"].value_counts().sort_index().to_string())
print("Por mes:\n", d["mes_num"].value_counts().sort_index().to_string())
fig_, ax = plt.subplots(1, 3, figsize=(15, 4))
d["ano_dt"].value_counts().sort_index().plot.bar(ax=ax[0], title="Por ano")
d["mes_num"].value_counts().sort_index().plot.bar(ax=ax[1], title="Por mes")
d["dia_sem"].value_counts().sort_index().plot.bar(ax=ax[2], title="Por dia da semana (0=seg)")
fig("temporal.png")

# ---------- 7. Problemas ----------
print("\n== OUTLIERS (IQR) ==")
for c in ["x", "y"]:
    q1, q3 = d[c].quantile([.25, .75]); i = q3 - q1
    n = ((d[c] < q1 - 1.5 * i) | (d[c] > q3 + 1.5 * i)).sum()
    print(f"{c}: {n} outliers")
fig_, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.boxplot(y=d["x"], ax=ax[0]); sns.boxplot(y=d["y"], ax=ax[1])
fig("boxplots.png")

print("\n== REDUNDANCIA ==")
print("Atributos constantes:", [c for c in d.columns if d[c].nunique() <= 1])
print(d[[c for c in ["classe", "ordem", "familia", "especie_nm", "nome_popul"]
        if c in d.columns]].nunique().to_string())

print("\n== CORRELACAO (Pearson) ==")
corr = d[["ano_dt", "mes_num", "dia_sem", "x", "y"]].corr()
print(corr.round(3).to_string())
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
fig("correlacao.png")