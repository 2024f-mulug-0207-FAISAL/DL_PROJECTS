# 📝 Product Review Sentiment Analysis using RNN

An NLP Deep Learning project that processes text sequences (customer reviews) and classifies them into Positive or Negative sentiments.

## 🧠 Model Architecture
* **Type:** Recurrent Neural Network (RNN / GRU) with Text Embedding.
* **Layers:** Embedding Layer (converts words to dense vectors) -> RNN/GRU Layer -> Fully Connected Layer -> Sigmoid.
* **Loss Function:** Binary Cross Entropy Loss (BCELoss).

## 📊 Dataset Structure
* **Format:** CSV (`customer_reviews.csv`)
* **Columns:** `review_text` (Raw textual feedback) and `sentiment` (0 for Negative, 1 for Positive).