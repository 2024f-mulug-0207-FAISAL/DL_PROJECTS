# 🤟 Sign Language Digits Recognition (Pixel CSV)

This project classifies hand gesture images of digits (0-9) using a Deep Convolutional Neural Network built from scratch in PyTorch.

## 🧠 Model Architecture
* **Type:** Deep CNN (Reshaping 4096 pixels back to 1x64x64 tensors).
* **Layers:** Conv2d -> BatchNorm -> ReLU -> MaxPool2d -> Dropout -> Linear Output (10 classes).
* **Loss Function:** CrossEntropyLoss (Multi-class classification).

## 📊 Dataset Structure
* **Format:** CSV (`sign_digits.csv`)
* **Columns:** `pixel_0` to `pixel_4095` (Flattened 64x64 image) and `label` (Target digit: 0 to 9).
