
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import plotly.graph_objs as go

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📈 AI Stock Forecast Dashboard")

# Top 50 stocks
stock_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "DIS", "HD", "BAC", "KO", "PEP", "INTC", "VZ",
    "PFE", "CSCO", "XOM", "ABT", "T", "ADBE", "NFLX", "CRM", "PYPL", "ORCL",
    "NKE", "MCD", "COST", "LLY", "QCOM", "DHR", "WFC", "CVX", "UPS", "MDT",
    "NEE", "TMO", "UNP", "PM", "IBM", "AVGO", "TXN", "LIN", "SBUX", "GE"
]

selected_stock = st.sidebar.selectbox("Select Stock", stock_symbols)
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Load stock data
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

df = load_data(selected_stock, start_date, end_date)

if df.empty:
    st.warning("No data found.")
    st.stop()

# Technical indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# Plotly chart
st.subheader(f"{selected_stock} Price Chart + Indicators")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20'))
st.plotly_chart(fig, use_container_width=True)

# LSTM Model
st.subheader("🔮 Next 7 Days Forecast (Deep Learning)")
data = df[["Close"]]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled_data)-7):
    X.append(scaled_data[i-seq_len:i, 0])
    y.append(scaled_data[i:i+7, 0])  # next 7 days

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(7)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

last_60 = scaled_data[-60:]
last_60 = np.reshape(last_60, (1, 60, 1))
predicted_scaled = model.predict(last_60)[0]
predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
future_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predicted})
st.dataframe(future_df.set_index("Date"))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=future_dates, y=predicted, mode='lines+markers', name='Forecast'))
fig2.update_layout(title="Forecasted Close Prices", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig2, use_container_width=True)

# Optional: Placeholder for News (dummy text for now)
st.subheader("📰 Latest Market News")
st.info("Integrate live news APIs like NewsAPI or Finnhub here for stock-specific headlines.")
