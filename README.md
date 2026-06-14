# 🧠 Deep Learning Production Portfolio (Pure PyTorch Pipeline)

Welcome to my core Deep Learning repository. This portfolio demonstrates professional, production-grade deep learning engineering implementations built entirely from the ground up using **PyTorch**. The repository covers multi-domain machine learning workflows including Tabular Classification, Advanced Spatial Computer Vision, Pixel-Array Transformation, Chronological Time-Series Forecasting, and Natural Language Processing (NLP) sequence modeling.

Every project inside this ecosystem is strictly isolated within its own dedicated micro-directory containing standalone data structures managed in scalable CSV formats, decoupled algorithmic notebooks, and independent architectural documentation.

---

## 🛠️ Unified Technology Stack & Engineering Ecosystem

* **Core AI Framework:** PyTorch Core Framework (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)
* **Computer Vision Suite:** Torchvision Suite (`transforms`, `datasets`), Pillow (`PIL`)
* **Analytics Engineering:** Scikit-Learn Ecosystem (`sklearn`), Pandas Core DataFrame, NumPy Core Arrays
* **Visualization Engineering:** Matplotlib Pipelines, Seaborn Visualization Systems

---

## 📂 Structural Directory Topology

```text
DL_PROJECTS/
│
├── .gitignore                         # Strategic mask blocking datasets and runtime models
├── README.md                          # Master global repository guide & portfolio mapping
│
├── Project1_Bank_Churn_ANN/
│   ├── data/customer_churn.csv        # Tabular client behavior array (10,000 samples)
│   ├── notebooks/bank_churn_ann.ipynb # Feedforward Multi-Layer Perceptron pipeline
│   └── README.md                      # Tabular classification documentation
│
├── Project2_Plant_Disease_CNN/
│   ├── data/plant_dataset.csv         # Local path indices and label references
│   ├── notebooks/plant_disease.ipynb  # Multi-stage image feature extraction CNN
│   └── README.md                      # Image classification documentation
│
├── Project3_Sign_Language_CNN/
│   ├── data/sign_digits.csv           # 4,096 flat row-wise pixel structural components
│   ├── notebooks/sign_digits.ipynb    # Multi-class spatial pixel tensor reshaping model
│   └── README.md                      # Alphanumeric gesture documentation
│
├── Project4_Stock_LSTM/
│   ├── data/stock_prices.csv          # 5-year daily chronological asset closing quotes
│   ├── notebooks/stock_lstm.ipynb     # Chronological sequential memory LSTM model
│   └── README.md                      # Regression forecasting documentation
│
└── Project5_Sentiment_RNN/
    ├── data/customer_reviews.csv      # Customer review text feedback pairs
    ├── notebooks/sentiment_rnn.ipynb  # Dense Word-Embedding GRU Recurrent Classifier
    └── README.md                      # Natural Language Processing documentation