# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

st.set_page_config(page_title="🪙 Bitcoin Forecast", layout="centered")
st.title("📈 Bitcoin Price Forecast (ARIMA Model)")

# Load model and historical data
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model_fit, df = pickle.load(f)
    return model_fit, df

model_fit, df = load_model()

# Forecast duration
n_days = st.slider("📅 Select number of days to forecast:", 7, 90, 30)

# Forecasting
forecast = model_fit.get_forecast(steps=n_days)
forecast_values = forecast.predicted_mean
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days)

# Build DataFrame
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast_values
}).set_index("Date")

# Plot historical and forecast
fig, ax = plt.subplots(figsize=(10, 5))
df['Close'].plot(ax=ax, label="📘 Historical Price")
forecast_df['Forecast'].plot(ax=ax, label="🔴 Forecasted Price")
ax.set_title("Bitcoin Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
plt.legend()
st.pyplot(fig)

# Show table
st.subheader("🔮 Forecasted Prices")
st.dataframe(forecast_df.reset_index())

# Download option
csv = forecast_df.reset_index().to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast CSV", data=csv, file_name="btc_forecast.csv", mime="text/csv")

