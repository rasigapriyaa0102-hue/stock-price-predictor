# app.py
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
# If using Keras models
from tensorflow.keras.models import load_model

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Make datetime available to Jinja templates (for footer year, etc.)
from datetime import datetime
app.jinja_env.globals['datetime'] = datetime


# Config
TRAIN_LOOKBACK_DAYS = 180   # how many days to use for retraining / evaluation
EVAL_WINDOW = 30            # how many recent days to compute MAPE on

# ---------- Helpers ----------




import requests
import pandas as pd

TWELVE_API = "4672863676bb446fba4012f405423fac"

def fetch_stock_data(symbol, period_days=30):
    """Fetch daily historical data using TwelveData (Free API)."""
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol.upper(),
            "interval": "1day",
            "outputsize": period_days + 10,   # give buffer
            "apikey": TWELVE_API
        }

        r = requests.get(url, params=params).json()

        # Check for errors
        if "values" not in r:
            print("❌ No data received:", r)
            return None

        df = pd.DataFrame(r["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)

        df = df.sort_values("datetime")

        df.rename(columns={"datetime": "Date", "close": "Close"}, inplace=True)
        df.set_index("Date", inplace=True)

        # Limit to required days
        df = df.last(f"{period_days}D")

        return df

    except Exception as e:
        print("❌ TwelveData fetch error:", e)
        return None




def compute_mape(actual, predicted):
    # avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.inf
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def simple_feature_engineering(df):
    # add moving averages and volume-like features if available (we only have 'Close' here)
    df2 = df.copy()
    df2['MA7'] = df2['Close'].rolling(7).mean().fillna(method='bfill')
    df2['MA21'] = df2['Close'].rolling(21).mean().fillna(method='bfill')
    # percent change
    df2['PctChange1'] = df2['Close'].pct_change().fillna(0)
    return df2

def scale_data(train_values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_values.reshape(-1,1))
    return scaler, scaled

def prepare_sequences(values, seq_len=30):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i, 0])
        y.append(values[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def load_all_models():
    """Load all model files found in models/ directory.
       Expect file naming: lstm.h5, gru.h5, rf.pkl, xgb.pkl, lr.pkl etc."""
    models = {}
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    for f in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, f)
        name = os.path.splitext(f)[0].lower()
        if f.endswith('.h5'):
            try:
               models[name] = load_model(path, compile=False)

            except Exception as e:
                print("Failed to load keras model", path, e)
        elif f.endswith('.pkl'):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                print("Failed to load sklearn model", path, e)
    return models

# Load available models once (you can add reload logic)
MODELS = load_all_models()

# ---------- Prediction workflow ----------
def evaluate_models_and_predict(symbol):
    """
    Fetch data, compute features, evaluate existing models on recent window,
    select best model using MAPE and produce final prediction and confidence.
    """
    df = fetch_stock_data(symbol, period_days=TRAIN_LOOKBACK_DAYS + 30)
    if df is None or df.empty:
        return None

    df_fe = simple_feature_engineering(df)
    close_values = df_fe['Close'].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_values)

    seq_len = 30
    X, y = prepare_sequences(scaled, seq_len=seq_len)

    # If not enough data
    if len(y) < 10:
        # fallback: naive predictor (last price)
        last_price = float(df_fe['Close'].iloc[-1])
        return {
            "symbol": symbol.upper(),
            "predicted_price": last_price,
            "confidence": 50,
            "suggestion": "Avoid",
            "history": df_fe['Close'].tail(60).reset_index().rename(columns={'index':'Date'}),
            "model_used": "naive"
        }

    results = []
    # Evaluate each loaded model
    for name, model in MODELS.items():
        try:
            if name.endswith('lstm') or name.endswith('gru') or hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
                # Keras model expects 3D
                pred_scaled = model.predict(X, verbose=0).reshape(-1,1)
                # invert scale
                pred = scaler.inverse_transform(pred_scaled).flatten()
            else:
                # sklearn models expect 2D flat sequence features (we will use last value or use rolling features)
                # Prepare features for sklearn: use last value of sequence as simple feature or average
                X_flat = X.reshape(X.shape[0], X.shape[1])
                pred_scaled = model.predict(X_flat).reshape(-1,1)
                pred = scaler.inverse_transform(pred_scaled).flatten()
            # Evaluate on last EVAL_WINDOW samples (align lengths)
            eval_n = min(EVAL_WINDOW, len(y))
            actual_recent = scaler.inverse_transform(y.reshape(-1,1)).flatten()[-eval_n:]
            pred_recent = pred[-eval_n:]
            mape = compute_mape(actual_recent, pred_recent)
            rmse = np.sqrt(np.mean((actual_recent - pred_recent)**2))
            results.append({
                "name": name,
                "mape": float(mape),
                "rmse": float(rmse),
                "predicted_full": pred  # series for plotting
            })
        except Exception as e:
            print("Model eval error", name, e)

    # If no models loaded or all failed fallback to naive last value
    if not results:
        last_price = float(df_fe['Close'].iloc[-1])
        return {
            "symbol": symbol.upper(),
            "predicted_price": last_price,
            "confidence": 45,
            "suggestion": "Avoid",
            "history": df_fe['Close'].tail(60).reset_index().rename(columns={'index':'Date'}),
            "model_used": "naive"
        }

    # select best by lowest mape
    best = sorted(results, key=lambda x: x['mape'])[0]

    # Compute final predicted price: use last predicted value from best model
    final_pred_series = best['predicted_full']
    predicted_price = float(final_pred_series[-1])

    # Confidence metric: map MAPE to confidence roughly: lower MAPE => higher confidence
    # simple heuristic: confidence = max(10, min(99, 100 - best_mape*2))
    best_mape = best['mape']
    confidence = max(10, min(99, 100 - best_mape * 2))

    # Suggestion: compare predicted price vs last actual
    last_actual = float(df_fe['Close'].iloc[-1])
    pct_change = ((predicted_price - last_actual) / last_actual) * 100
    if pct_change > 1.0 and confidence > 50:
        suggestion = "Buy"
    elif pct_change < -1.0 and confidence > 50:
        suggestion = "Sell"
    else:
        suggestion = "Avoid"

    # Prepare history for plotting (actual and predicted aligned)
    # We will return last 60 days of actual and corresponding predicted shape (if available)
    history_actual = df_fe['Close'].tail(60).reset_index().rename(columns={'index':'Date'})

    # To create predicted series for last 60 days, attempt to align
    # We'll inverse-transform the predicted full series and match last 60 points if lengths fit
    try:
        predicted_inv = final_pred_series  # already inverse transformed earlier
        pred_len = len(predicted_inv)
        # if pred_len >= 60, take last 60
        if pred_len >= 60:
            predicted_for_plot = predicted_inv[-60:]
        else:
            # pad front with NaN
            predicted_for_plot = np.concatenate([np.full(60 - pred_len, np.nan), predicted_inv[-pred_len:]])
        predicted_for_plot = predicted_for_plot.tolist()
    except Exception as e:
        predicted_for_plot = [None]*60

    response = {
        "symbol": symbol.upper(),
        "predicted_price": round(predicted_price, 2),
        "confidence": round(float(confidence), 1),
        "suggestion": suggestion,
        "model_used": best['name'],
        "model_mape": round(best_mape, 3),
        "history_dates": history_actual['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "history_actual": history_actual['Close'].tolist(),
        "history_predicted": predicted_for_plot
    }
    return response

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip()
        if not symbol:
            return render_template('prediction.html', error="Please enter a stock symbol.")
        return redirect(url_for('result', symbol=symbol.upper()))
    return render_template('prediction.html')

@app.route('/result/<symbol>')
def result(symbol):
    try:
        res = evaluate_models_and_predict(symbol)
    except Exception as e:
        print("Prediction error:", e)
        res = None

    if res is None:
        return render_template('result.html', error_message=f"Could not fetch data or run models for {symbol}.")

    # ✅ Pass chart data to template
    chart_data = {
        "dates": res["history_dates"],
        "actual": res["history_actual"],
        "predicted": res["history_predicted"]
    }

    return render_template(
        'result.html',
        stock_name=res["symbol"],
        best_model=res["model_used"],
        suggestion=res["suggestion"],
        chart_data=chart_data
    )




@app.route('/api/prediction/<symbol>')
def api_prediction(symbol):
    res = evaluate_models_and_predict(symbol)
    if res is None:
        return jsonify({"error":"no data"}), 400
    return jsonify(res)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        # you can store the message or send email
        return render_template('contact.html', sent=True)
    return render_template('contact.html')

@app.route('/models')
def models_page():
    # technical page for reviewers
    # compute quick summary of loaded models
    model_summaries = []
    for name in MODELS.keys():
        model_summaries.append({"name": name})
    return render_template('models.html', models=model_summaries)

# ---------- LIVE REAL-TIME PREDICTION ----------
# ---------- LIVE REAL-TIME PREDICTION (TwelveData Version) ----------
@app.route('/predict_live/<symbol>')
def predict_live(symbol):
    """Fetch 1-minute live data using TwelveData and predict next price."""
    try:
        def fetch_intraday(sym):
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": sym.upper(),
                "interval": "1min",
                "outputsize": 50,        # last 50 minutes of data
                "apikey": TWELVE_API
            }
            return requests.get(url, params=params).json()

        # Fetch from TwelveData
        data = fetch_intraday(symbol)

        if "values" not in data:
            return jsonify({"error": f"No live data found for {symbol}", "details": data}), 400

        ts = data["values"]

        # Convert to DataFrame
        df = pd.DataFrame([
            {"Date": d["datetime"], "Close": float(d["close"])}
            for d in ts
        ])

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df.set_index("Date", inplace=True)

        # Prepare values
        close_values = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_values)

        seq_len = 30
        if len(scaled) < seq_len:
            return jsonify({"error": "Not enough live data for prediction"}), 400

        X = scaled[-seq_len:].reshape((1, seq_len, 1))

        # Select LSTM model
        model = None
        for name in MODELS:
            if "lstm" in name:
                model = MODELS[name]
                break

        if model is None:
            return jsonify({"error": "No LSTM model loaded"}), 500

        pred_scaled = model.predict(X)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        last_price = float(close_values[-1])
        change_pct = ((pred_price - last_price) / last_price) * 100

        # Suggestion logic
        if change_pct > 1:
            suggestion = "Buy"
        elif change_pct < -1:
            suggestion = "Sell"
        else:
            suggestion = "Avoid"

        return jsonify({
            "symbol": symbol.upper(),
            "current_price": round(last_price, 2),
            "predicted_price": round(float(pred_price), 2),
            "change_pct": round(float(change_pct), 2),
            "suggestion": suggestion
        })

    except Exception as e:
        print("Realtime error:", e)
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
