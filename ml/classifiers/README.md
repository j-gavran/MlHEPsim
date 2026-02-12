# Simple DNN Classifiers

This directory contains PyTorch Lightning-based neural network classifiers for machine learning tasks in high-energy physics.

## Overview

The classifiers implement deep neural network (DNN) models with flexible architectures for both binary and multi-label classification tasks. All models are built on PyTorch Lightning for easy training, validation, and testing with automatic logging and checkpointing.

## Available Models

### Binary Classifier (`binary_model.py`)
A specialized classifier for binary classification tasks (e.g., signal vs. background discrimination).

**Features:**
- Inherits from `MultilabelClassifier` with binary-specific logic
- Automatic configuration validation for loss functions and activations
- Built-in accuracy metrics using `torchmetrics.BinaryAccuracy`
- Supports multiple loss functions: `BCELoss`, `BCEWithLogitsLoss`, `MSELoss`

**Supported Configurations:**
- **BCEWithLogitsLoss** (recommended): Use without output activation (raw logits)
- **BCELoss**: Requires Sigmoid output activation
- **MSELoss**: Can be used with Sigmoid activation (not recommended)

**Key Methods:**
- `training_step()`: Computes and logs training loss
- `validation_step()`: Computes loss and accuracy on validation set
- `test_step()`: Evaluates final model performance

### Multilabel Classifier (`multilabel_model.py`)
A flexible classifier for multi-class classification problems.

**Features:**
- Supports both standard MLP and Residual MLP architectures
- Configurable hidden layers, activation functions, batch normalization, and dropout
- Uses CrossEntropyLoss by default for multi-class problems
- Automatic hyperparameter logging via PyTorch Lightning

**Architecture Options:**
1. **BasicMLP**: Standard feed-forward network with configurable:
   - Hidden layer dimensions
   - Activation functions (ReLU, LeakyReLU, Tanh, etc.)
   - Batch normalization
   - Dropout regularization
   - Output activation (optional)

2. **BasicResMLP**: Residual MLP with skip connections:
   - All features of BasicMLP
   - Additional `repeats` parameter for residual block depth
   - Better gradient flow for deeper networks

## Architecture Configuration

Both classifiers are configured via dictionaries passed during initialization:

### Model Configuration (`model_conf`)
```python
model_conf = {
    "model": "MLP" or "ResMLP",           # Architecture type
    "input_dim": int,                     # Input feature dimension
    "hidden_dim": [int, int, ...],        # Hidden layer sizes
    "output_dim": int,                    # Number of output classes
    "activation_function": str,           # e.g., "ReLU", "LeakyReLU"
    "act_out": str or None,               # Output activation (e.g., "Sigmoid")
    "batchnorm": bool,                    # Enable batch normalization
    "act_first": bool,                    # Activation before or after BatchNorm
    "dropout": float,                     # Dropout probability (0.0 to 1.0)
    "repeats": int,                       # (ResMLP only) Residual block repeats
}
```

### Training Configuration (`training_conf`)
```python
training_conf = {
    "loss": str,                          # Loss function name (e.g., "BCEWithLogitsLoss")
    "optimizer": str,                     # Optimizer name (e.g., "Adam")
    "learning_rate": float,               # Learning rate
    # Additional optimizer-specific parameters
}
```

## Usage Example

```python
from ml.classifiers.models import BinaryClassifier

# Define model architecture
model_conf = {
    "model": "MLP",
    "input_dim": 6,
    "hidden_dim": [128, 64, 32],
    "output_dim": 1,
    "activation_function": "ReLU",
    "act_out": None,  # Use raw logits
    "batchnorm": True,
    "dropout": 0.1,
}

# Define training parameters
training_conf = {
    "loss": "BCEWithLogitsLoss",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
}

# Initialize classifier
classifier = BinaryClassifier(model_conf, training_conf)

# Train using PyTorch Lightning Trainer
from pytorch_lightning import Trainer
trainer = Trainer(max_epochs=100)
trainer.fit(classifier, train_dataloader, val_dataloader)
```

## Integration with MLHEPsim Framework

These classifiers are designed to work seamlessly with the broader MLHEPsim framework:
- Compatible with data processors in `ml/common/data_utils/`
- Inherits logging capabilities from `ml/common/nn/modules.py`
- Can use custom trackers (MLflow, TensorBoard, etc.)
- Automatic hyperparameter saving and checkpointing

## Best Practices

1. **Loss Function Selection:**
   - Use `BCEWithLogitsLoss` for binary classification (numerically stable)
   - Use `CrossEntropyLoss` for multi-class classification
   - Avoid `MSELoss` for classification tasks

2. **Activation Functions:**
   - Never combine `Sigmoid` with `BCEWithLogitsLoss` (logits already expected)
   - Always use `Sigmoid` with `BCELoss`
   - For multi-class, no output activation needed (CrossEntropyLoss expects logits)

3. **Regularization:**
   - Start with dropout ~0.1-0.2 for hidden layers
   - Use batch normalization for deeper networks
   - Consider weight decay in optimizer for additional regularization

4. **Architecture Design:**
   - Use `ResMLP` for networks with >4 hidden layers
   - Gradually decrease hidden layer sizes (e.g., [256, 128, 64, 32])
   - Ensure input/output dimensions match your data/task

## Dependencies

- PyTorch
- PyTorch Lightning
- torchmetrics
- ml.common.nn (MLHEPsim common neural network modules)