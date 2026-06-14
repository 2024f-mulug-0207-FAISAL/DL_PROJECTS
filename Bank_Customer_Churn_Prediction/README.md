# Bank Customer Churn Prediction System (ANN Topology)

## 📌 Project Overview
Customer attrition (churn) directly limits financial growth and inflates customer acquisition costs in retail banking. This project implements an end-to-end multi-layer deep neural network to dynamically analyze tabular metrics and flag customers exhibiting a high flight risk. The foundational engineering objective is to mathematically solve non-linear multi-feature behavioral boundaries that traditional parametric classification algorithms fail to capture smoothly.

## 🧠 Model Architecture & Mathematical Intuition
Built entirely in **PyTorch**, the system initializes a feedforward Artificial Neural Network (Multi-Layer Perceptron) sub-classed from `nn.Module`:

* **Input Topology:** Receives 10 engineered feature nodes matching normalized data structures.
* **Hidden Layers (Dense):** * `Layer 1 (fc1)`: Linear transformation from 10 inputs to 16 hidden nodes via $y = w \cdot x + b$, stabilized by a Non-Linear Rectified Linear Unit (`nn.ReLU`) activation function to cancel vanishing gradient propagation.
  * `Layer 2 (fc2)`: Further structural compaction reducing representation from 16 neurons to 8 hidden units, also mapped using `nn.ReLU`.
* **Output Node (fc3):** Compresses the 8 hidden neurons down to a singular structural outcome scaled through an explicit Sigmoid activation function (`nn.Sigmoid`), forcing the final scalar vector directly between $0$ and $1$ to reflect a clear, actionable churn probability score.

## 📊 Pipeline & Dataset Schema
* **Format:** Locally managed structured CSV format (`customer_churn.csv`).
* **Volume:** 10,000 highly realistic chronological client samples.
* **Feature Processing Strategy:** Categorical elements (`Geography`, `Gender`) undergo systematic Label Encoding. High-variance scalar ranges (e.g., `Balance`, `EstimatedSalary`) are processed via Scikit-Learn's `StandardScaler`, enforcing standard normalization ($\mu=0, \sigma=1$) across the operational network tensor tensors to protect global learning rates.
* **Target Vector:** `Exited` (Binary state indicator where `0` = Stable Customer, `1` = Churned Customer).