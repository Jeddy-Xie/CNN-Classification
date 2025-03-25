# CNN-Classification
Convolutional neural networks for classifying variants of MNIST dataset

## Project Overview
This project implements and evaluates Convolutional Neural Networks (CNNs) for classifying different variants of the MNIST dataset. The implementation includes both handcrafted feature extraction and CNN-based approaches, with a focus on comparing their performance across various dataset modifications.

## Dataset Variants
The project evaluates the models on three MNIST dataset variants:
1. Normal MNIST (baseline)
2. Scaled MNIST (images scaled to 50% of original size)
3. Jittered MNIST (images with random pixel jittering)

## Model Architecture
The CNN model implements the following architecture:
- Input Layer: 28x28 grayscale images
- First Convolutional Block:
  - Conv2D layer (32 filters, 3x3 kernel)
  - BatchNorm2D
  - ReLU activation
  - Conv2D layer (32 filters, 3x3 kernel)
  - BatchNorm2D
  - ReLU activation
  - MaxPooling2D (2x2)
  - Dropout (0.25)
- Second Convolutional Block:
  - Conv2D layer (64 filters, 3x3 kernel)
  - BatchNorm2D
  - ReLU activation
  - Conv2D layer (64 filters, 3x3 kernel)
  - BatchNorm2D
  - ReLU activation
  - MaxPooling2D (2x2)
  - Dropout (0.25)
- Classifier:
  - Flattening Layer
  - Dense Layer (512 units with BatchNorm and ReLU)
  - Dropout (0.25)
  - Dense Layer (1024 units with BatchNorm and ReLU)
  - Dropout (0.5)
  - Output Layer (10 units)

## Training Results
The model was trained using Bayesian optimization to find optimal hyperparameters for each dataset variant.

### Normal MNIST
- Test Accuracy: 96.20%
- Best Hyperparameters:
  - Batch Size: 34
  - Learning Rate: 0.0092
  - Number of Epochs: 92
  - Best Validation Accuracy: 96.77%

### Scaled MNIST
- Test Accuracy: 95.20%
- Best Hyperparameters:
  - Batch Size: 32
  - Learning Rate: 0.0028
  - Number of Epochs: 60
  - Best Validation Accuracy: 94.17%

### Jittered MNIST
- Test Accuracy: 79.80%
- Best Hyperparameters:
  - Batch Size: 116
  - Learning Rate: 0.0042
  - Number of Epochs: 62
  - Best Validation Accuracy: 84.27%

## Key Findings
1. The CNN model achieves high accuracy (>95%) on both normal and scaled MNIST datasets
2. Performance significantly degrades on jittered MNIST (79.80%), indicating sensitivity to pixel-level noise
3. Bayesian optimization found different optimal hyperparameters for each dataset variant
4. The model architecture with batch normalization and dropout shows good regularization properties

## Project Structure
```
.
├── code/
│   ├── CNNModel.py         # CNN model implementation
│   ├── TestCNN.ipynb       # Main notebook for CNN evaluation
│   ├── TestCNN.html        # HTML export of the notebook
│   ├── digitFeatures.py    # Handcrafted feature extraction
│   ├── linearModel.py      # Linear model implementation
│   └── testHandcrafted.py  # Handcrafted feature testing
├── data/                   # Dataset storage
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Setup and Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook code/TestCNN.ipynb
   ```

## Dependencies
- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Jupyter
- scikit-learn
