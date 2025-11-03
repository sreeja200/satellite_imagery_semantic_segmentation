# 🌍 Satellite Imagery Semantic Segmentation  

TensorFlow implementation of **U-Net with Spatial Pyramid Pooling (SPP)** for semantic segmentation of satellite imagery.  
The model segments land cover types such as water, vegetation, urban areas, and barren land using Sentinel-2 data.  

---

## 🚀 Project Overview  
This project performs **semantic segmentation** on satellite images to classify each pixel into meaningful land cover categories.  
A U-Net architecture enhanced with **Spatial Pyramid Pooling (SPP)** is implemented to improve multi-scale feature learning and segmentation accuracy.

---

## 🧠 Model Architecture  
- **Base Model:** U-Net  
- **Enhancement:** Spatial Pyramid Pooling (SPP)  
- **Framework:** TensorFlow / Keras  
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, IoU  

---

## 🧠 Training Progress

| Training & Validation | Accuracy / Loss Curves |
|:----------------------:|:----------------------:|
| ![Training Accuracy & Loss](results/training_accuracy.png) | ![Validation Accuracy](results/validation_accuracy.png) |
| **Best Epoch Performance** | ![Best Epoch](results/best_epoch.png) |

---

## 🖼️ Visual Results  

| Original Image | Ground Truth | Predicted Mask |
|----------------|---------------|----------------|
| ![Original 1](https://github.com/sreeja200/satellite_imagery_semantic_segmentation/blob/main/results/visuals/original_1.jpg) | ![Ground Truth 1](https://github.com/sreeja200/satellite_imagery_semantic_segmentation/blob/main/results/visuals/ground_truth_1.jpg) | ![Predicted 1](https://github.com/sreeja200/satellite_imagery_semantic_segmentation/blob/main/results/visuals/predicted_1.jpg) |
| ![Original 2](https://github.com/sreeja200/satellite_imagery_semantic_segmentation/blob/main/results/visuals/original_2.jpg) | ![Ground Truth 2](https://github.com/sreeja200/satellite_imagery_semantic_segmentation/blob/main/results/visuals/ground_truth_2.jpg) | ![Predicted 2](https://github.com/sreeja200/satellite_imagery_semantic_segmentation/blob/main/results/visuals/predicted_2.jpg) |

---

## ⚙️ How to Run  

```bash
# Clone this repository
git clone https://github.com/sreeja200/satellite_imagery_semantic_segmentation.git
cd satellite_imagery_semantic_segmentation

# Install dependencies
pip install -r requirements.txt

# Run training
python semantic.py


