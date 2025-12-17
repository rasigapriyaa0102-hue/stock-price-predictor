import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import joblib, os

# Create models folder
os.makedirs("models", exist_ok=True)

# Fetch data (use META or any stock)
# Fetch data with fallback tickers
tickers = ["META", "AAPL", "MSFT", "GOOG", "INFY.NS"]
df = pd.DataFrame()

for t in tickers:
    try:
        df = yf.download(t, start="2020-01-01", end="2025-01-01")
        if not df.empty:
            print(f"✅ Using data from {t}")
            break
    except Exception as e:
        print(f"⚠️ Failed for {t}: {e}")

if df.empty:
    raise Exception("❌ Could not download data from any ticker. Try again later.")


# Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df)

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data
split = int(len(X)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# ---------------- LSTM ----------------
lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)),
    LSTM(50),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
lstm.save("models/lstm.h5")

# ---------------- GRU ----------------
gru = Sequential([
    GRU(50, return_sequences=True, input_shape=(X_train.shape[1],1)),
    GRU(50),
    Dense(1)
])
gru.compile(optimizer='adam', loss='mse')
gru.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
gru.save("models/gru.h5")

# Flatten for ML models
X_flat = X.reshape((X.shape[0], X.shape[1]))
y_flat = y

# ---------------- Random Forest ----------------
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_flat, y_flat)
joblib.dump(rf, "models/rf.pkl")

# ---------------- XGBoost ----------------
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_flat, y_flat)
joblib.dump(xgb_model, "models/xgb.pkl")

# ---------------- SVR ----------------
svr = SVR()
svr.fit(X_flat, y_flat)
joblib.dump(svr, "models/svr.pkl")

# ---------------- Linear Regression ----------------
lr = LinearRegression()
lr.fit(X_flat, y_flat)
joblib.dump(lr, "models/lr.pkl")

print("✅ All models trained and saved in /models/")
