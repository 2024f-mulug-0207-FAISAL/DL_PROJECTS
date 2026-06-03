# 📈 Stock Price Forecasting using LSTM Networks

A sequential deep learning model designed to predict future stock/crypto prices based on historical time-series financial data.

## 🧠 Model Architecture
* **Type:** RNN / LSTM (Long Short-Term Memory).
* **Why LSTM?** Unlike standard networks, LSTM has a "hidden state memory" to retain past temporal patterns.
* **Layers:** LSTM Layer -> Dropout -> Dense Linear Output.
* **Loss Function:** Mean Squared Error (MSE Loss) for regression.

## 📊 Dataset Structure
* **Format:** CSV (`stock_prices.csv`)
* **Features:** Historical daily stock metrics (`Close` price sequence is used as input).