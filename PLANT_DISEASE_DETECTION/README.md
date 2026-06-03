# 🌿 Plant Disease Detection using CNN

This project classifies plant leaves into "Healthy" or "Diseased" categories using Computer Vision and Deep Learning.

## 🧠 Model Architecture
* Built using **PyTorch**.
* **Type:** Convolutional Neural Network (CNN).
* **Layers:** Multiple `Conv2d` layers for feature extraction, `MaxPool2d` for downsampling, and `Linear` (Fully Connected) layers for final classification.
* **Optimizer:** Adam / SGD
* **Loss Function:** Cross Entropy Loss.

## 📊 Dataset Structure (CSV Format)
Kyunki images ka size bada hota hai, isliye portfolio ko professional rakhne ke liye humne images ke paths aur unke labels ko CSV file mein save kiya hai:
* **Format:** CSV (`plant_dataset.csv`)
* **Columns:** `image_path` (Path to the local image), `label` (0 for Healthy, 1 for Diseased).