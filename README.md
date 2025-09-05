🩺 Breast Cancer Classification using Deep Learning (BreakHis Dataset)

📌 Overview

This project focuses on automated breast cancer classification using the BreakHis histopathological image dataset. The goal is to distinguish between benign and malignant tumors by leveraging state-of-the-art deep learning models. The dataset poses unique challenges due to its multi-magnification images (40X, 100X, 200X, 400X) and class imbalance, making it ideal for testing robustness and generalization of models.

🎯 Objectives

* Classify breast cancer histopathology images into benign or malignant.
* Compare performance of multiple deep learning architectures (ResNet50, VGG16, MobileNetV2, Vision Transformer (ViT), and a Custom CNN).
* Analyze the effect of different magnification levels on model performance.
* Apply Grad-CAM and SHAP for model interpretability and explainability.

📂 Dataset

* Dataset: BreakHis (Breast Cancer Histopathological Image Dataset)
* Total Images: 7,909
* Classes:

  * Benign (2,480 images)    → Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma
  * Malignant (5,429 images) → Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma
  * Magnifications: 40X, 100X, 200X, 400X

📌 [Dataset Link (Kaggle)](https://www.kaggle.com/datasets/ambarish/breakhis)

## 🛠️ Methodology

1.  Preprocessing: Image resizing (224×224), normalization, and stratified train-validation-test split (80/10/10).
2.  Models Implemented:

   * ResNet50 (Transfer Learning)
   * VGG16 (Baseline CNN)
   * MobileNetV2 (Lightweight CNN)
   * Custom CNN (Built from scratch)
   * Vision Transformer (ViT)
3. Training:

   * Optimizer: Adam (lr=0.0001)
   * Batch Size: 32 (ViT → 16 due to GPU memory)
   * Loss: Binary Cross-Entropy
   * Early stopping & learning rate scheduling applied
     
4. Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

📊 Results

| Model                    | Accuracy  | Precision | Recall | F1-Score |
| ------------------------ | --------- | --------- | ------ | -------- |
| MobileNetV2              | 91.2%     | 90.8%     | 91.5%  | 91.1%    |
| Vision Transformer (ViT) | 93.4%     | 93.0%     | 93.7%  | 93.3%    |
| ResNet50                 | 92.8%     | 92.5%     | 93.1%  | 92.8%    |
| VGG16                    | 90.5%     | 90.2%     | 90.7%  | 90.4%    |
| Custom CNN               | 89.7%     | 89.4%     | 89.9%  | 89.6%    |

✅ ViT performed best (93.4% accuracy)**, while MobileNetV2 provided a lightweight and efficient option.

🔍 Model Interpretability

* Grad-CAM: Visualized model attention, showing CNNs focus on dense cellular regions, while ViT spreads attention across multiple patches.
* SHAP Analysis:Explained ViT predictions, highlighting global contextual cues in tissue patterns.

 🚀 Key Takeaways

* ViT outperformed CNNs, proving the effectiveness of self-attention in histopathology.
* MobileNetV2 balanced accuracy and efficiency, suitable for real-world deployment.
* Custom CNN performed competitively, showing the value of dataset-specific architectures.
* Combining CNNs & Transformers may yield more robust medical imaging solutions.

 📦 Tech Stack

* Language:Python 3
* Frameworks:PyTorch, TorchVision
* Libraries:NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SHAP
* Hardware:GPU-enabled training

 📈 Future Work

* Explore hybrid CNN-Transformer architectures.
* Apply advanced data augmentation for imbalance handling.
* Extend classification to multi-class subtypes instead of binary classification.

🙌 Acknowledgements

* Dataset:BreakHis – P\&D Laboratory, Brazil
* Pretrained Models: PyTorch ImageNet weights

