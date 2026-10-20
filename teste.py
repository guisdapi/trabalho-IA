import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_excel("data/rodofauna.xlsx")

print(df.shape)
print(df.head())
print(df.dtypes)