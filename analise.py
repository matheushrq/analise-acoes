import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

# -- Montando carteiras de ações -- #

tickers = ["EGIE3.SA", "ITSA4.SA", "BBAS3.SA", "PSSA3.SA", "^BVSP"]
dados = yf.download(tickers, period="1y")["Close"]

# Rentabilidade histórica (Ações x IBOV)
dados_normalizados = dados / dados.iloc[0]
dados_normalizados.plot(title ="Desempenho Relativo x Ibovespa", figsize=(12, 6))

# Multiplicação de capital
retornos = dados.pct_change().dropna()

retorno_acumulado = (1 + retornos).cumprod()
retorno_acumulado.plot(title="Retorno Acumulado", figsize=(12,6))

# Volatilidade histórica
volatilidade_hist = retornos.rolling(window=21).std() * np.sqrt(252)
volatilidade_hist.plot(title="Volatilidade Histórica", figsize=(12,6))

# Volatilidade anualizada
volatilidade_anual = retornos.std() * np.sqrt(252)
volatilidade_anual.plot(title="Volatilidade Anualizada", kind='bar', figsize=(12,6))

# volatilidade x retorno
plt.figure(figsize=(12,6))
sns.scatterplot(x=volatilidade_anual, y=retornos.mean() * 252)
plt.title("Volatilidade x Retorno Anualizado")

# Correlação entre ativos
correlacao = retornos.corr()
sns.heatmap(correlacao, annot=True, cmap="coolwarm")
plt.title("Correlação entre Ativos")
plt.show()