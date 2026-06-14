# Automated Plant Disease Detection System (Custom 2D-CNN)

## 📌 Project Overview
Manual crop inspection scales poorly and delays essential diagnostic triage in modern automated agronomy. This production-grade Computer Vision pipeline introduces a specialized, multi-stage 2D Convolutional Neural Network (CNN) engineered to parse raw graphical visual matrices and categorize plant leaf structural changes into "Healthy" or "Diseased" classification fields without relying on bloated, computationally expensive pre-trained multi-gigabyte models.

## 🧠 Model Architecture & Spatial Matrix Intuition
Traditional flat layers destroy the structural relationships of spatial dimensions (height/width coordinates) in images. This system implements a localized PyTorch feature extraction sequence:

* **Convolution Block 1 (`nn.Conv2d`):** Tracks a 3-channel (RGB) input image matrix across 16 independent convolutional filters using a $3\times3$ spatial kernel window with matching zero-padding (`padding=1`). This retains boundary definitions while generating distinct feature activation maps. Mapped using `nn.ReLU`.
* **Pooling Block 1 (`nn.MaxPool2d`):** Downsamples spatial resolution across a $2\times2$ matrix window with a fixed stride of 2. This shrinks pixel bounds ($64\times64 \rightarrow 32\times32$), forcing position-invariant recognition while reducing system floating-point operational costs.
* **Convolution Block 2:** Converts the 16 features into 32 deep structural representations via an identical $3\times3$ filter matrix, isolating deeper patterns (e.g., leaf rot textures vs. healthy vein structures).
* **Pooling Block 2:** Executes an identical $2\times2$ max-pooling operation, compressing spatial resolution down to $16\times16$ dimensions.
* **Fully Connected Classifier Layer:** Flattens the deep channels into a linear array ($32 \times 16 \times 16 = 8,192$ structural features) and runs them through a hidden layer of 64 nodes before projecting down to a terminal single-node classification scalar with a Sigmoid mask for definitive target parsing.