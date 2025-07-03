# train.py without pmdarima
import yfinance as yf
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

# Download data
df = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31')
df = df[['Close']].dropna()

# Fit ARIMA model manually (you must choose p, d, q)
model = ARIMA(df['Close'], order=(5, 1, 0))  # <-- manually chosen
model_fit = model.fit()

# Save model and data
with open("model.pkl", "wb") as f:
    pickle.dump((model_fit, df), f)

df.to_csv("btc_price.csv")
print("✅ Model trained and saved.")


