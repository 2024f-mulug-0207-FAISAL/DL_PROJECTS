# 🧠 Deep Learning Master Portfolio (Pure PyTorch Pipeline)

Welcome to my Deep Learning repository. This portfolio showcases a structured, production-grade implementation of deep learning algorithms built entirely using **PyTorch**, spanning across Tabular Analysis, Computer Vision (CV), Time-Series Forecasting, and Natural Language Processing (NLP).

Each project is self-contained within its own directory featuring dedicated datasets (structured in CSV formats) and independent professional technical documentation.

---

## 🛠️ Core Technology Stack & Architecture

* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Computer Vision Tools:** Torchvision (`transforms`, `datasets`), PIL (Pillow)
* **Data Preprocessing & Analytics:** Scikit-Learn (`sklearn`), Pandas, NumPy
* **Visualization Engineering:** Matplotlib, Seaborn

---

## 📂 Repository Structure

```text
DL_PROJECTS/
│
├── .gitignore                         # Prevents tracking of model weights (.pt) and heavy CSVs
├── README.md                          # Main master portfolio guide
│
├── Project1_Bank_Churn_ANN/
│   ├── data/customer_churn.csv        # 10,000 structured bank records
│   ├── notebooks/bank_churn_ann.ipynb # Pure PyTorch ANN training pipeline
│   └── README.md                      # Detailed project documentation
│
├── Project2_Plant_Disease_CNN/
│   ├── data/plant_dataset.csv         # Image metadata and category mappings
│   ├── notebooks/plant_disease.ipynb  # Custom Multi-layer Convolutional Network
│   └── README.md                      # Feature map & vision documentation
│
├── Project3_Sign_Language_CNN/
│   ├── data/sign_digits.csv           # 4,096 flattened pixel arrays per row
│   ├── notebooks/sign_digits.ipynb    # Multi-class spatial pixel transformation CNN
│   └── README.md                      # Gesture classification guide
│
├── Project4_Stock_LSTM/
│   ├── data/stock_prices.csv          # 5-year chronological closing prices
│   ├── notebooks/stock_lstm.ipynb     # Time-Series LSTM Recurrent Architecture
│   └── README.md                      # Forecasting and evaluation graphs
│
└── Project5_Sentiment_RNN/
    ├── data/customer_reviews.csv      # Natural language feedback text & polarity labels
    ├── notebooks/sentiment_rnn.ipynb  # Word-Embedding GRU Recurrent Network
    └── README.md                      # Tokenization and NLP sequence guide