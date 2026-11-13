# Trabalho de IA – Predição de atropelamentos de fauna silvestre (RodoFauna/DF)

Trabalho de Aprendizado Supervisionado da disciplina de Inteligência Artificial (UFSM).
Pipeline completo de Machine Learning e Redes Neurais (MLP) sobre os registros do
Programa RodoFauna Atropelada (IBRAM/DF), com três problemas:

| Problema | Unidade (linha) | Alvo | Divisões testadas |
|---|---|---|---|
| Classificação binária (hotspot) | segmento de 500 m | 1 = hotspot (contagem ≥ percentil 75), 0 = baixo risco | aleatória, espacial |
| Classificação multiclasse | atropelamento | classe taxonômica (amphibians, birds, mammals, reptiles) | aleatória, espacial, temporal |
| Regressão | segmento × período (abril a março) | número de atropelamentos | aleatória, espacial, temporal |

Em todos os problemas, a MLP é comparada com o Random Forest (e, na regressão, também com a
Regressão Linear), usando exatamente as mesmas divisões de treino, validação e teste.

## Instalação

Requer Python 3 em uma versão compatível com o TensorFlow.

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Ordem de execução

Os scripts dependem uns dos outros (cada etapa lê o que a anterior salvou), então devem ser
rodados nesta ordem, a partir da pasta do projeto:

```
# Etapa 1 - análise exploratória
python etapa1.py

# Etapa 2 - pré-processamento (o comum primeiro)
python etapa2_comum.py
python etapa2_binario.py
python etapa2_multiclasse.py
python etapa2_regressao.py

# Etapa 3 - classificação com MLP x Random Forest
python etapa3_binario.py aleatoria
python etapa3_binario.py espacial
python etapa3_multiclasse.py aleatoria
python etapa3_multiclasse.py espacial
python etapa3_multiclasse.py temporal
python comparar_divisoes.py hotspot
python comparar_divisoes.py multiclasse

# Etapa 4 - regressão com MLP x Regressão Linear x Random Forest
python etapa4.py regressao_aleatoria
python etapa4.py regressao_espacial
python etapa4.py regressao_temporal
python comparar_divisoes_regressao.py

# Etapa 5 - otimização de hiperparâmetros da MLP com Optuna (multiclasse)
python etapa5_optuna.py aleatoria
```

Para conferir se a etapa 5 roda antes da busca completa (50 trials, cerca de 30 a 50 minutos),
dá para passar a quantidade de trials: `python etapa5_optuna.py aleatoria 3`.

## Estrutura

```
data/
  rodofauna.xlsx            dados originais (Programa RodoFauna Atropelada - IBRAM/DF)
  rodofauna_etapa1.csv      fauna silvestre sem duplicados, com colunas criadas na etapa 1
  etapa2/                   tabelas comuns, bases legíveis (base_*.csv) e conjuntos prontos (*.joblib)
etapa1.py                   análise exploratória
etapa2_comum.py             período de monitoramento, segmentos de 500 m, alvo do hotspot e divisão espacial
etapa2_binario.py           pré-processamento do hotspot
etapa2_multiclasse.py       pré-processamento do multiclasse
etapa2_regressao.py         pré-processamento da regressão
etapa3_binario.py           MLP x Random Forest no hotspot
etapa3_multiclasse.py       MLP x Random Forest no multiclasse
etapa4.py                   MLP x Regressão Linear x Random Forest na regressão
etapa5_optuna.py            busca de hiperparâmetros da MLP do multiclasse com Optuna
comparar_divisoes.py        compara as divisões da etapa 3 (hotspot ou multiclasse)
comparar_divisoes_regressao.py   compara as divisões da etapa 4
graficosE1/ ... graficosE5/ gráficos de cada etapa
resultadosE3/ ... resultadosE5/   tabelas de métricas (.csv) usadas no relatório
```

## Observações

- **Reprodutibilidade:** todos os sorteios usam a semente 42, e o TensorFlow roda com a
  otimização oneDNN desligada, para que cada execução gere os mesmos resultados.
- **Teste guardado:** o pré-processamento é ajustado só com o treino; arquitetura, parâmetros
  do Random Forest e corte de decisão são escolhidos só com a validação. O teste é usado
  uma única vez, no final.
- **Tempo de execução:** a etapa 3 do hotspot com divisão aleatória é a mais demorada
  (as redes podem treinar até 1000 épocas antes da parada antecipada).
