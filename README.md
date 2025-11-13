# 🏥 Skin Cancer Detection using Deep Learning

A deep learning-based skin cancer classification system that uses Convolutional Neural Networks (CNN) to classify skin lesions into 7 different types using the HAM10000 dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results](#results)
- [Class Labels](#class-labels)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a CNN-based classifier for detecting and classifying skin cancer lesions. The model is trained on the HAM10000 dataset, which contains over 10,000 dermatoscopic images of skin lesions. The system can classify skin lesions into 7 different categories, aiding in early detection and diagnosis of skin cancer.

## ✨ Features

- **Improved Deep Learning Model**: Enhanced CNN architecture with batch normalization and deeper layers
- **7-Class Classification**: Classifies skin lesions into 7 different types
- **Data Augmentation**: Rotation, shifting, zooming, and flipping for better generalization
- **Class Weight Balancing**: Handles imbalanced dataset automatically
- **Advanced Training**: Early stopping, learning rate scheduling, and model checkpointing
- **Comprehensive Evaluation**: Detailed metrics including confusion matrix, precision, recall, and F1-score per class
- **Data Preprocessing**: Automatic image resizing and normalization
- **Prediction Pipeline**: Easy-to-use prediction function for new images
- **Training History Visualization**: Plot accuracy and loss curves
- **Stratified Splitting**: Ensures balanced train/test splits

## 📊 Dataset

This project uses the **HAM10000** (Human Against Machine with 10000 training images) dataset, which contains:

- **10,015 dermatoscopic images** of skin lesions
- **7 different classes** of skin cancer types
- Images stored in two parts: `HAM10000_images_part_1` and `HAM10000_images_part_2`
- Metadata CSV file with labels and additional information

### Dataset Structure
```
data/
└── HAM10000/
    ├── HAM10000_images_part_1/
    ├── HAM10000_images_part_2/
    ├── HAM10000_metadata.csv
    ├── hmnist_28_28_L.csv
    ├── hmnist_28_28_RGB.csv
    ├── hmnist_8_8_L.csv
    └── hmnist_8_8_RGB.csv
```

## 🏗️ Model Architecture

The model uses an improved sequential CNN architecture with batch normalization and deeper layers:

### Convolutional Blocks:
1. **Conv2D** (32 filters, 3x3) + **BatchNormalization** + ReLU + MaxPooling2D + Dropout(0.25)
2. **Conv2D** (64 filters, 3x3) + **BatchNormalization** + ReLU + MaxPooling2D + Dropout(0.25)
3. **Conv2D** (128 filters, 3x3) + **BatchNormalization** + ReLU + MaxPooling2D + Dropout(0.25)
4. **Conv2D** (256 filters, 3x3) + **BatchNormalization** + ReLU + MaxPooling2D + Dropout(0.25)

### Fully Connected Layers:
5. **Flatten**
6. **Dense** (256 units) + **BatchNormalization** + ReLU + Dropout(0.5)
7. **Dense** (128 units) + ReLU + Dropout(0.5)
8. **Dense** (7 units) + Softmax activation

**Key Improvements:**
- ✅ Batch Normalization for stable training
- ✅ Deeper architecture (4 conv blocks + 2 dense layers)
- ✅ Progressive dropout regularization
- ✅ Data augmentation during training
- ✅ Class weights for imbalanced data

**Input Shape**: (64, 64, 3) RGB images  
**Output**: 7-class probability distribution  
**Optimizer**: Adam  
**Loss Function**: Categorical Crossentropy  
**Metrics**: Accuracy

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nermine-1/skin-cancer-detection.git
   cd skin-cancer-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - The HAM10000 dataset should be placed in the `data/HAM10000/` directory
   - Ensure you have both image folders: `HAM10000_images_part_1` and `HAM10000_images_part_2`
   - Place the metadata CSV file in the same directory

## 💻 Usage

### Training the Model

To train the model from scratch:

```bash
cd src
python train.py
```

The training process will:
- Load and preprocess the data
- Calculate class weights to handle imbalanced dataset
- Build the improved CNN model with batch normalization
- Apply data augmentation (rotation, shifting, zooming, flipping)
- Train with early stopping (patience=10) and learning rate scheduling
- Save the best model to `models/skin_cancer_model.h5` based on validation accuracy
- Save training history to `models/training_history.pkl`

### Making Predictions

To predict the class of a skin lesion image:

```python
from predict import predict_image

# Predict a single image
prediction = predict_image("path/to/your/image.jpg")
print(f"Predicted class: {prediction}")
```

Or run the prediction script directly:

```bash
cd src
python predict.py
```

Make sure to update the image path in `predict.py` before running.

### Evaluating the Model

To evaluate the trained model on the test set:

```bash
cd src
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on the test set
- Display comprehensive metrics: accuracy, loss, precision, recall, F1-score
- Generate confusion matrix visualization
- Show per-class performance metrics
- Save classification report and sample predictions in `evaluation_results/`

### Visualizing Training History

To plot training history (accuracy and loss curves):

```bash
cd src
python utils.py
```

Or in Python:

```python
from utils import plot_history

# Plot from saved history file
plot_history('../models/training_history.pkl')
```

This generates training curves showing accuracy and loss over epochs, saved to `evaluation_results/training_history.png`.

## 📁 Project Structure

```
skin-cancer-detection/
│
├── data/
│   └── HAM10000/
│       ├── HAM10000_images_part_1/    # Image dataset part 1
│       ├── HAM10000_images_part_2/    # Image dataset part 2
│       ├── HAM10000_metadata.csv      # Labels and metadata
│       └── hmnist_*.csv               # Preprocessed datasets
│
├── src/
│   ├── load_data.py      # Data loading and preprocessing
│   ├── model.py          # CNN model architecture
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction script
│   ├── evaluate.py       # Model evaluation script
│   └── utils.py          # Utility functions
│
├── models/
│   └── skin_cancer_model.h5    # Trained model (generated after training)
│
├── evaluation_results/          # Evaluation visualizations
│   └── image_*.png
│
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 📦 Requirements

The project requires the following Python packages:

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning utilities
- `tensorflow` - Deep learning framework
- `seaborn` - Statistical data visualization
- `Pillow` - Image processing

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## 📈 Results

After training, the model evaluation results are saved in the `evaluation_results/` directory. The evaluation script generates:

- **Overall Metrics**: Test accuracy and loss
- **Confusion Matrix**: Visual heatmap showing classification performance
- **Per-Class Metrics**: Precision, Recall, and F1-score for each class
- **Classification Report**: Detailed text report saved to `classification_report.txt`
- **Sample Predictions**: Visualization images with predictions vs. ground truth labels
- **Training Curves**: Accuracy and loss plots over training epochs

**Expected Performance**: With the improved architecture and training techniques, the model should achieve:
- Overall accuracy: **>80%** (target: 85%+)
- Balanced performance across all 7 classes
- Good generalization with data augmentation

**Note**: Actual performance metrics depend on the training process and may vary. Run `python src/evaluate.py` to see your model's specific metrics.

## 🏷️ Class Labels

The model classifies skin lesions into 7 categories:

| Code | Class Name | Description |
|------|------------|-------------|
| `akiec` | Actinic keratoses | Pre-cancerous lesions |
| `bcc` | Basal cell carcinoma | Common skin cancer |
| `bkl` | Benign keratosis | Benign lesions |
| `df` | Dermatofibroma | Benign skin lesion |
| `mel` | Melanoma | Most dangerous skin cancer |
| `nv` | Melanocytic nevi | Benign moles |
| `vasc` | Vascular lesions | Blood vessel lesions |

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This project is for educational and research purposes only. **It should not be used as a substitute for professional medical diagnosis, treatment, or advice.** Always consult with qualified healthcare professionals for medical concerns.

## 👤 Author

**Nermine**

- GitHub: [@Nermine-1](https://github.com/Nermine-1)

## 🙏 Acknowledgments

- HAM10000 dataset creators and contributors
- TensorFlow and Keras communities
- Open source contributors

---

⭐ If you find this project helpful, please consider giving it a star!
