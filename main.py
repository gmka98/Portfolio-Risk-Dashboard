import yfinance as yf
import numpy as np
import pandas as pd

# Configuration of assets and weights for the portfolio
assets = ['AAPL', 'JPM', 'XOM', 'GLD', 'TLT']
weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
risk_free_rate = 0.04 # Risk-free rate for Sharpe ratio calculation 

prices = yf.download(assets, start='2019-01-01', end='2024-12-31')['Close']
returns = prices.pct_change().dropna()

# Portfolio daily returns
port_returns = returns @ weights

day_var = np.percentile(port_returns, 5) # 5th percentile for historical value at risk (VaR)

# Key metrics

annual_ret = port_returns.mean() * 252
annual_vol = port_returns.std() * np.sqrt(252)


sharpe = (annual_ret - risk_free_rate) / annual_vol

print(f"Annual Return: {annual_ret:.1%}")
print(f"Annual Volatility: {annual_vol:.1%}")
print(f"Sharpe Ratio: {sharpe:.2f}")