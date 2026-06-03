# 🏦 Bank Customer Churn Prediction using ANN

This project predicts whether a bank customer will leave the bank (churn) or stay, based on their credit score, balance, tenure, and activity status.

## 🧠 Model Architecture
* Built using **PyTorch**.
* **Type:** Artificial Neural Network (Multi-Layer Perceptron).
* **Layers:** Input Layer -> Hidden Layer 1 (ReLU) -> Hidden Layer 2 (ReLU) -> Output Layer (Sigmoid).
* **Optimizer:** Adam
* **Loss Function:** Binary Cross Entropy Loss (BCELoss).

## 📊 Dataset Description
* **Format:** CSV (`customer_churn.csv`)
* **Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
* **Target Variable:** `Exited` (0 for Stayed, 1 for Churned).