# 🫁 Pediatric Pneumonia Detection with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning system for detecting pneumonia from pediatric chest X-rays, achieving **92.31% accuracy** with Vision Transformer architecture.

![Project Banner](docs/banner.png)

## 🎯 Overview

This project implements a comprehensive medical AI pipeline for pneumonia classification using state-of-the-art deep learning models. The system includes data preprocessing, multiple model architectures, explainability features (Grad-CAM), and a production-ready Streamlit web application.

### ✨ Key Features

- **🤖 Multiple Model Architectures**: Baseline CNN, DenseNet121, EfficientNet-B0, Vision Transformer (ViT-B/16)
- **📊 Comprehensive Evaluation**: ROC curves, confusion matrices, sensitivity/specificity analysis
- **🔍 Model Explainability**: Grad-CAM heatmap visualizations for clinical validation
- **🚀 Production Deployment**: ONNX export, Streamlit web interface
- **⚖️ Handles Class Imbalance**: Weighted loss functions and proper evaluation metrics
- **📈 Excellent Performance**: 92.31% test accuracy, 98.21% sensitivity, 0.978 AUC

## 📊 Model Performance

| Model | Test Accuracy | Sensitivity | Specificity | AUC | Parameters |
|-------|---------------|-------------|-------------|-----|------------|
| Baseline CNN | 84.94% | 94.87% | 68.38% | 0.9355 | 26.1M |
| DenseNet121 | 84.13% | 99.74% | 58.12% | 0.9649 | 7.0M |
| EfficientNet-B0 | 90.87% | 97.95% | 79.06% | 0.9580 | 4.0M |
| **ViT-B/16** | **92.31%** | **98.21%** | **82.48%** | **0.9780** | 85.8M |

## 🗂️ Project Structure
```
pneumonia-xray-classification/
├── notebooks/              # Jupyter notebooks (development pipeline)
├── src/                    # Source code modules
├── models/                 # Trained model checkpoints
├── deployment/             # Production deployment files
│   ├── streamlit_app.py   # Streamlit web application
│   ├── *.onnx             # ONNX exported models
│   └── requirements.txt   # Deployment dependencies
├── results/                # Evaluation results and visualizations
├── data/                   # Dataset (not included in repo)
├── docs/                   # Documentation
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for training)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pneumonia-xray-classification.git
cd pneumonia-xray-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**

Download the [Pediatric Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in:
```
data/chest_xray/
├── train/
├── val/
└── test/
```

## 💻 Usage

### Training Models

Run the Jupyter notebooks in sequence:
```bash
jupyter notebook notebooks/01_environment_setup.ipynb
```

Or train directly using Python:
```bash
python src/train.py --model efficientnet --epochs 10
```

### Running the Streamlit App
```bash
cd deployment
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Making Predictions
```python
from src.models import load_model
from src.utils import preprocess_image

# Load model
model = load_model('models/efficientnet_best.pth')

# Predict
image = preprocess_image('path/to/xray.jpg')
prediction, confidence = model.predict(image)
print(f"Prediction: {prediction} ({confidence:.2%})")
```

## 📈 Reproducing Results

1. **Run all notebooks sequentially** (01-10)
2. **Models are saved** in `models/` directory
3. **Results visualizations** in `results/` directory
4. **Evaluation metrics** in notebook 08

Expected training time:
- Baseline CNN: ~30 min (CPU)
- EfficientNet: ~45 min (CPU)
- ViT-B/16: ~60 min (CPU)
- All models: ~10-15 min (GPU)

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from `deployment/streamlit_app.py`
4. Ensure `requirements.txt` is in deployment folder

### Docker (Optional)
```bash
docker build -t pneumonia-classifier .
docker run -p 8501:8501 pneumonia-classifier
```

### ONNX Runtime

Use exported ONNX models for production inference:
```python
import onnxruntime as ort

session = ort.InferenceSession('deployment/pneumonia_classifier_efficientnet.onnx')
outputs = session.run(None, {'input': preprocessed_image})
```

## 🔬 Model Explainability

The system includes Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of X-rays the model focuses on:

![Grad-CAM Example](results/explainability/gradcam_example.png)

This helps clinicians:
- Validate model decisions
- Identify potential biases
- Build trust in AI predictions

## ⚠️ Medical Disclaimer

**This is a research project and educational tool.**

- ❌ Not FDA approved for clinical use
- ❌ Not a substitute for professional medical diagnosis
- ✅ Intended for research and educational purposes only
- ✅ Always consult qualified healthcare professionals

## 📊 Dataset

**Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Training**: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation**: 47 images
- **Test**: 624 images (234 Normal, 390 Pneumonia)
- **Population**: Pediatric patients (1-5 years old)

**Citation**:
```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), 
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", 
Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **Data Processing**: NumPy, OpenCV, Albumentations
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: Pytorch-Grad-CAM
- **Web App**: Streamlit
- **Deployment**: ONNX Runtime
- **Evaluation**: scikit-learn

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset provided by Kermany et al.
- Pretrained models from PyTorch/Torchvision
- Streamlit for the amazing web framework
- Medical imaging community for guidance

## 📚 References

1. Kermany et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"
2. Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
3. Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

---

**⭐ If you find this project helpful, please consider giving it a star!**



## 📥 Downloading Pretrained Models

Model files are hosted in [GitHub Releases](https://github.com/yourusername/pneumonia-xray-classification/releases) due to their size.

### Quick Download (Recommended)

**For Streamlit deployment:**
```bash
python download_models.py
# Select option 1 (EfficientNet only)
```

**For all models:**
```bash
python download_models.py
# Select option 4 (Everything)
```

### Manual Download

Download from [Releases page](https://github.com/yourusername/pneumonia-xray-classification/releases/tag/v1.0):

| File | Size | Purpose | Download |
|------|------|---------|----------|
| `efficientnet_best.pth` | 46 MB | **Recommended** for Streamlit | [Download](https://github.com/yourusername/pneumonia-xray-classification/releases/download/v1.0/efficientnet_best.pth) |
| `pneumonia_classifier_efficientnet.onnx` | 15 MB | ONNX (production) | [Download](https://github.com/yourusername/pneumonia-xray-classification/releases/download/v1.0/pneumonia_classifier_efficientnet.onnx) |
| `vit_best.pth` | 978 MB | Best accuracy (92.31%) | [Download](https://github.com/yourusername/pneumonia-xray-classification/releases/download/v1.0/vit_best.pth) |
| `densenet121_best.pth` | 80 MB | High sensitivity (99.74%) | [Download](https://github.com/yourusername/pneumonia-xray-classification/releases/download/v1.0/densenet121_best.pth) |
| `baseline_cnn_best.pth` | 299 MB | Baseline model | [Download](https://github.com/yourusername/pneumonia-xray-classification/releases/download/v1.0/baseline_cnn_best.pth) |

**Place downloaded files:**
- `.pth` files → `models/` directory
- `.onnx` files → `deployment/` directory

### Dataset

Download the [Chest X-Ray Images dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

Extract to: `data/chest_xray/`
