# Malware Classification using Deep Learning and SVM

A comprehensive machine learning project for classifying malware samples into 12 different families (A-L) using both deep learning (CNN, DenseNet) and traditional machine learning (SVM) approaches.

## 📋 Project Overview

This project implements multiple approaches for malware classification:

1. **Deep Learning Models**:
   - Custom Convolutional Neural Network (CNN)
   - DenseNet-based architecture
   - Automated hyperparameter tuning using grid search

2. **Classical Machine Learning**:
   - Support Vector Machine (SVM) with CNN feature extraction
   - Multiple kernel configurations (RBF, Linear, Polynomial)
   - Grid search for optimal hyperparameters

## 🗂️ Project Structure

```
ML_finals/
├── src/                          # Deep learning implementation
│   ├── main.py                   # Main training script with grid search
│   ├── config.yaml               # Hyperparameter configuration
│   ├── models/                   # Model architectures
│   │   ├── cnn.py               # Custom CNN model
│   │   └── densenet.py          # DenseNet model
│   └── utils/                    # Utility functions
│       ├── createDataset.py     # Dataset loading utilities
│       ├── cachedDataset.py     # Cached dataset for faster loading
│       ├── train.py             # Training loop
│       ├── test.py              # Testing and max voting
│       ├── gridSearch.py        # Grid search implementation
│       └── plotResults.py       # Results visualization
├── SVM/                          # SVM implementation
│   ├── svm_train.py             # SVM training script
│   ├── svm_train_gridsearch.py  # SVM with grid search
│   ├── cnn_feature_extraction.py # Extract CNN features
│   ├── extract_challenge_features.py # Extract challenge features
│   ├── predict_challenge.py     # Challenge dataset predictions
│   └── challenge_dataset.py     # Challenge dataset loader
├── CS271_final_data/             # Dataset directory
│   ├── A-L/                     # Training data (12 malware families)
│   └── X/                       # Challenge dataset (unlabeled)
├── weights/                      # Saved model weights
├── plots/                        # Training plots and visualizations
├── results/                      # Experiment results
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.9.1+
- CUDA-compatible GPU (optional, for faster training)
- Apple Silicon MPS support (for Mac users)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ML_finals
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Download

Download the dataset from: [CS271_final_data.zip](https://www.cs.sjsu.edu/~stamp/data/CS271_final_data.zip)

Extract the zip file to the project root directory:
```bash
unzip CS271_final_data.zip
```

### Structure

The dataset contains malware samples organized into:
- **Training data**: 12 malware families (A-L), each in separate folders
- **Challenge data**: 1200 unlabeled samples in folder X for final classification

Each malware sample is represented as grayscale images for neural network input.

## 🔧 Usage

### Deep Learning Training

#### Train with Grid Search

```bash
cd src

# Train CNN model with grid search
python main.py --model cnn --malwareType all

# Train DenseNet model
python main.py --model densenet --malwareType f

# Train for a specific malware type
python main.py --model cnn --malwareType a
```

**Available options:**
- `--model`: `cnn` or `densenet`
- `--malwareType`: `a`, `b`, `c`, `d`, `e`, `f`, or `all` (for all 12 families)

#### Configuration

Edit `src/config.yaml` to customize hyperparameter search space:

```yaml
params:
  learning_rates:
    - 0.01
    - 0.001
  
  optimizers:
    - adam
  
  batch_sizes:
    - 32
    - 64
  
  dropout_rates:
    - 0.0
    - 0.5
  
  weight_decays:
    - 0.0
    - 0.005
```

### SVM Training

#### Step 1: Extract CNN Features

```bash
cd SVM

# Extract features using pre-trained CNN
python cnn_feature_extraction.py
```

This creates:
- `svm_features_X.npy`: Feature vectors
- `svm_features_y.npy`: Labels

#### Step 2: Train SVM

```bash
# Train SVM with automatic hyperparameter search
python svm_train.py
```

This will:
- Test multiple SVM configurations (RBF, Linear, Polynomial kernels)
- Find the best hyperparameters
- Save the trained model to `svm_model.pkl`
- Save the feature scaler to `scaler.pkl`
- Generate confusion matrix and performance plots

#### Step 3: Predict Challenge Dataset

```bash
# Extract features from challenge dataset
python extract_challenge_features.py

# Generate predictions
python predict_challenge.py
```

Predictions are saved to `challenge_predictions.txt` in the required format.

## 📈 Features

### Deep Learning
- **Automated Grid Search**: Systematically searches hyperparameter space
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Max Voting**: Ensemble predictions for improved accuracy
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Device Agnostic**: Supports CUDA, MPS (Apple Silicon), and CPU

### SVM
- **CNN Feature Extraction**: Uses pre-trained CNN as feature extractor
- **Multiple Kernel Support**: RBF, Linear, and Polynomial kernels
- **Standardization**: Feature scaling for optimal SVM performance
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and visualizations

## 📊 Model Performance

The project evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Loss curves**: Training and validation loss over epochs
- **Confusion Matrix**: Detailed per-class performance
- **Classification Report**: Precision, recall, and F1-score per class

Results are automatically saved to:
- `plots/`: Training curves and visualizations
- `results/`: Model performance metrics
- `SVM/results/`: SVM-specific results and plots

## 🔍 Key Components

### Grid Search
Automatically finds optimal hyperparameters by:
1. Creating all combinations of specified parameters
2. Training a model for each combination
3. Evaluating on validation data
4. Returning best parameters and results

### Max Voting (Ensemble)
For multi-image samples:
1. Each image gets a prediction
2. Final prediction is the most common class
3. Improves robustness and accuracy

### Early Stopping
Monitors validation accuracy and stops training when:
- No improvement for `patience` epochs (default: 5)
- Prevents overfitting
- Saves training time

## 🎯 Challenge Dataset Prediction

The project includes a complete pipeline for the challenge dataset:

```bash
cd SVM
python extract_challenge_features.py  # Extract features
python predict_challenge.py           # Generate predictions
```

Output format (in `challenge_predictions.txt`):
```
0, A
1, B
2, C
...
1199, L
```

## 📦 Saved Artifacts

- **Model weights**: `weights/` directory
- **Trained SVM**: `SVM/svm_model.pkl`
- **Feature scaler**: `SVM/scaler.pkl`
- **Training plots**: `plots/` directory
- **Predictions**: `SVM/challenge_predictions.txt`

## 🛠️ Development

### Adding New Models

1. Create new model in `src/models/your_model.py`
2. Update `src/main.py` to include your model
3. Add model-specific configuration to `config.yaml`

### Customizing Hyperparameters

Edit `src/config.yaml` to modify search space for:
- Learning rates
- Batch sizes
- Dropout rates
- Weight decay values
- Optimizers

## 📝 Notes

- The project uses MPS (Metal Performance Shaders) on Apple Silicon for GPU acceleration
- CUDA is supported for NVIDIA GPUs
- CPU fallback is available for systems without GPU support
- Grid search can be computationally intensive; adjust parameter combinations accordingly
- SVM training requires pre-extracted CNN features

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is for educational purposes as part of CS271 Final Project.

## 🙋 Support

For questions or issues, please refer to the project documentation or contact the development team.

---

**Happy Malware Classification! 🔒🤖**
