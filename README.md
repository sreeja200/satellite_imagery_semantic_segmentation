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

<table>
  <tr>
    <td align="center"><b>Training & Validation Loss</b><br>
      <img src="https://github.com/user-attachments/assets/0d816aa5-8970-4a08-a1d8-fb2471d9e3be" width="350">
    </td>
    <td align="center"><b>Training & Validation Accuracy</b><br>
      <img src="https://github.com/user-attachments/assets/a8f3ce01-f07f-4d51-bc3b-88e4f51c0cd0" width="350">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>Best Epoch Performance</b><br>
      <img src="https://github.com/user-attachments/assets/37eeac57-c63e-4f45-b3f3-10b6a35a79ef" width="600">
    </td>
  </tr>
</table>

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


