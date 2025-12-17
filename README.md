# 📈 Stock Price Prediction Web Application

A full-stack **Stock Price Prediction Web App** built using **Flask**, **Machine Learning**, and **Deep Learning models (LSTM, GRU, XGBoost, SVR)**.  
The application fetches **real-time stock market data** using Yahoo Finance and provides **Buy / Sell / Avoid** suggestions based on model predictions.

---

## 🚀 Features

- 📊 Real-time stock price data using **yfinance**
- 🤖 Multiple prediction models:
  - LSTM
  - GRU
  - XGBoost
  - Support Vector Regression (SVR)
- 🏆 Automatic **best model selection** using **lowest MAPE**
- 💡 Trading suggestions:
  - **Buy**
  - **Sell**
  - **Avoid**
- 📉 Interactive chart (Actual vs Predicted prices)
- 🌐 REST API endpoint for predictions
- 🎨 Clean and responsive UI

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask
- TensorFlow / Keras
- Scikit-learn
- yfinance
- Pandas, NumPy

### Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

---

## 📂 Project Structure


stock-price-predictor/
│
├── app.py
├── train_models.py
├── requirements.txt
│
├── models/
│ ├── lstm.h5
│ ├── gru.h5
│ ├── xgb.pkl
│ └── svr.pkl
│
├── templates/
│ ├── base.html
│ ├── index.html
│ ├── prediction.html
│ ├── result.html
│ ├── models.html
│ ├── about.html
│ └── contact.html
│
├── static/
│ ├── css/
│ │ └── style.css
│ └── js/
│ └── result_chart.js
│
└── README.md

yaml

## 📸 Output / Results

### Prediction Result Page
Below screenshot shows the final stock price prediction result with model comparison and suggestion (BUY / SELL / AVOID).

![Prediction Output](screenshots/output.png)


## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor

2️⃣ Install dependencies
bash
pip install -r requirements.txt

3️⃣ Run the application
bash
python app.py

4️⃣ Open in browser
cpp http://127.0.0.1:5000

🔮 How Prediction Works

User enters a stock symbol (AAPL, TSLA, RELIANCE, TCS, etc.)

Real-time historical data is fetched from Yahoo Finance

Data preprocessing & scaling is applied

Each model predicts future prices

MAPE (Mean Absolute Percentage Error) is calculated

Best model is selected automatically

Final prediction & suggestion is displayed

📊 Suggestion Logic

Condition	Suggestion
Price ↑ and Confidence > 50%	Buy
Price ↓ and Confidence > 50%	Sell
Small change / Low confidence	Avoid

🔗 API Endpoint
Get prediction as JSON
bash

GET /api/prediction/<STOCK_SYMBOL>
Example:

ruby

http://127.0.0.1:5000/api/prediction/AAPL
⚠️ Limitations
Yahoo Finance rate limits may occur

Predictions are not financial advice

Accuracy depends on market volatility

📌 Future Enhancements
Live auto-refresh every 60 seconds

Candlestick charts

News sentiment analysis

User authentication & portfolio tracking

Deployment on cloud (AWS / Render)

📜 Disclaimer
This project is built for academic and learning purposes only.
Do not use it for real financial trading decisions.

👩‍💻 Author
Rasigapriya A
BE – Computer Science Engineering
Stock Price Prediction Mini Project

⭐ If you like this project, give it a star on GitHub!