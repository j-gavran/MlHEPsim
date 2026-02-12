# Drell-Yan Flow Model Training

This directory contains scripts for training normalizing flow models on Drell-Yan physics data using PyTorch Lightning and Hydra for configuration management.

## Overview

Two main training scripts are available:
- **`main_flows.py`** - Full model with 6 features (2 PT, 2 Eta, 2 Phi)
- **`main_flows_reduced.py`** - Reduced model with 6 features (1 PT, 2 Eta, 2 Phi, 1 Mass)

Both scripts train generative flow models on Drell-Yan dimuon data and support multiple flow architectures with comprehensive tracking, logging, and model checkpointing.

## Supported Flow Models

The script supports the following normalizing flow architectures:

- **NICE** - Non-linear Independent Components Estimation ([arXiv:1410.8516](https://arxiv.org/abs/1410.8516))
- **RealNVP** - Real-valued Non-Volume Preserving transformations ([arXiv:1605.08803](https://arxiv.org/abs/1605.08803))
- **Glow** - Generative Flow with Invertible 1x1 Convolutions ([arXiv:1807.03039](https://arxiv.org/abs/1807.03039))
- **MAF** - Masked Autoregressive Flow ([arXiv:1705.07057](https://arxiv.org/abs/1705.07057))
- **MAFMADEMOG** - Combination of MAF and MADEMOG
- **MADEMOG** - Masked Autoregressive Density Estimation with Mixture of Gaussians ([arXiv:1306.0186](https://arxiv.org/abs/1306.0186))
- **PolynomialSplineFlow** - Polynomial spline-based flows ([arXiv:1808.03856](https://arxiv.org/abs/1808.03856))
- **RqSplineFlow** - Rational Quadratic Spline flows ([arXiv:1906.04032](https://arxiv.org/abs/1906.04032))

## Usage

### Full Model (6 Features)

Train with all kinematic features including both muon transverse momenta:

```bash
python ml/custom/DrellYan/main_flows.py
```

Uses `variables.json` with features:
- `Muons_PT_Lead`, `Muons_PT_Sub`
- `Muons_Eta_Lead`, `Muons_Eta_Sub`
- `Muons_Phi_Lead`, `Muons_Phi_Sub`

### Reduced Model (6 Features)

Train with invariant mass instead of subleading PT (ablation study):

```bash
python ml/custom/DrellYan/main_flows_reduced.py
```

Uses `variables_reduced.json` with features:
- `Muons_Pos_PT` (only leading PT)
- `Muons_Pos_Eta`, `Muons_Neg_Eta`
- `Muons_Pos_Phi`, `Muons_Neg_Phi`
- `Muons_Minv_MuMu` (invariant mass)

The reduced model tests whether the flow can learn to predict the missing PT from the mass and angular separations.

### Custom Configuration

Both scripts use Hydra for configuration management. Configuration files should be located in `ml/custom/DrellYan/config/flows/`. You can override parameters from the command line:

```bash
python ml/custom/DrellYan/main_flows.py \
    model_config.model_name=mademog \
    training_config.epochs=100 \
    experiment_config.seed=42
```

## Configuration Structure

The script expects a configuration file (`main_config.yaml`) with four main sections:

### 1. Experiment Configuration (`experiment_config`)
- `run_name`: Name for the training run (defaults to current timestamp)
- `seed`: Random seed for reproducibility
- `epochs`: Maximum number of training epochs
- `accelerator`: Training device (e.g., "gpu", "cpu")
- `devices`: Number of devices to use
- `check_eval_n_epoch`: Frequency of validation checks
- `check_metrics_n_epoch`: Frequency of metric computation
- `log_every_n_steps`: Logging frequency
- `num_sanity_val_steps`: Number of validation steps before training
- `precision`: Training precision (e.g., "16-mixed", "32")
- `save_dir`: Directory for saving models and logs
- `model_postfix`: Automatically set based on preprocessing configuration

### 2. Data Configuration (`data_config`)
- `train_split`: Training data split ratio
- `val_split`: Validation data split ratio
- `feature_selection`: Features to include in training
  - `keep_names`: List of feature names to keep
- `preprocessing`: Data preprocessing settings
  - `cont_rescale_type`: Continuous feature scaling method (e.g., "gauss_rank", "standard")
  - `disc_rescale_type`: Discrete feature scaling method (optional)
- `dataloader_config`: PyTorch DataLoader settings
  - `batch_size`
  - `num_workers`
  - etc.

### 3. Model Configuration (`model_config`)
- `model_name`: Name of the flow architecture to use
- Model-specific hyperparameters (varies by architecture)

### 4. Training Configuration (`training_config`)
- `epochs`: Number of training epochs
- `early_stop_patience`: Patience for early stopping (optional)
- Learning rate and optimizer settings
- Other training hyperparameters

## Data Processing Pipeline

The script uses a chained data processing approach:

1. **DrellYanNpyProcessor**: Loads and processes raw Drell-Yan data from `.npy` files
2. **DrellYanFeatureSelector**: Selects relevant features based on configuration
3. **Preprocessor**: Applies feature scaling and normalization
4. **ProcessorChainer**: Chains these processors together for a complete pipeline

## Training Features

### Callbacks
- **LearningRateMonitor**: Tracks learning rate changes
- **EarlyStopping**: Stops training when validation loss stops improving
- **ModelCheckpoint**: Saves best model based on validation loss
- **FinalEpochLogger**: Logs final training statistics

### Logging
- **MLFlow Integration**: All experiments are logged to MLFlow
- **FlowTracker**: Custom tracker for flow-specific metrics and visualizations
  - Density plots
  - Generated vs. reference sample comparisons
  - Training/validation loss history plots

### Optimizations
- High-precision matrix multiplication (`torch.set_float32_matmul_precision("high")`)
- Gradient clipping (value: 1.0)
- Deterministic seeding for reproducibility

## Output

The script produces:

1. **Trained Model**: Registered and saved to the model registry
2. **MLFlow Logs**: Complete training logs in the MLFlow tracking server
3. **Visualizations**: Stored in `ml/custom/DrellYan/metrics/`
   - Density plots
   - Generated sample comparisons
   - Loss history plots
4. **Checkpoints**: Best model checkpoint based on validation loss

## Model Naming Convention

Models are automatically named using the pattern:
```
{model_name}_flow_model_{cont_rescale_type}_{disc_rescale_type}
```

For example: `mademog_flow_model_gauss_rank`

## Requirements

- PyTorch
- PyTorch Lightning
- Hydra
- MLFlow
- NumPy
- Custom modules from `ml.common` and `ml.flows`

## Notes

- Progress bars are disabled by default to avoid large log files when running on batch systems
- The script uses deterministic seeding for reproducibility
- Training is optimized for GPU execution but can run on CPU
- All metrics and plots are automatically logged to MLFlow

## Example Workflow

1. Prepare your Drell-Yan data in `.npy` format in `ml/data/drellyan/`
2. Configure your experiment in `config/flows/main_config.yaml`
3. Run the training script
4. Monitor training progress in MLFlow UI
5. Retrieve the best model from the model registry
6. Use the trained model for sample generation or density estimation

## Troubleshooting

- If loss plots aren't being generated, check that `on_train_epoch_end()` is being called in the FlowTracker
- Ensure your configuration file paths are correct
- Verify that your data files exist in the expected location
- Check MLFlow logs for detailed error messages
