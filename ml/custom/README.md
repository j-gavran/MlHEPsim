# General config settings 

## Data config

### Input processing 

| Parameter         | Value          | Description    |
|-------------------|----------------| -------------- |
| data_dir          | str         | Path to data directory |
| base_file_name | str | Base file name for data files |
| keep_ratio        | float (0.0 < keep_ratio <= 1.0) | Keep this fraction of all data |

### Feature selector

| Parameter         | Value          | Description    |
|-------------------|----------------| -------------- |
| n_data | int | Number of data points to use |
| drop_types        | ["uni", "disc", "cont", "label"] | What types of variables to not include in feature selector |
| drop_names | ["name1", "name2", ...] | What variables to not include in feature selector |
| keep_names | ["name1", "name2", ...] | What variables to include in feature selector |

### Preprocessing

| Parameter         | Value          | Description    |
|-------------------|----------------| -------------- |
| cont_rescale_type | normal, robust, sigmoid, tanh, maxabs, logit, logit_normal, gauss_rank | Continious feature scaling |
| disc_rescale_type | onehot, dequant, dequant_normal, dequant_logit_normal | Discrete feature scaling | 
| no_process | ["name1", "name2", ...] | What variables to not process |

### Dataloader config
See: [torch.utils.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

## Experiment config

| Parameter             | Value           | Description                                          |
|-----------------------|-----------------|------------------------------------------------------|
| experiment_name        | str            | Name of the experiment                               |
| run_name               | str             | Name of the run                                      |
| save_dir               | mlruns/         | Directory to save the experiment results             |
| seed                   | int             | Random seed for reproducibility                       |
| accelerator            | gpu             | Accelerator to use for training (e.g. cpu, gpu)       |
| device                 | cuda            | Device to use for training (e.g. cpu, cuda)           |
| devices                | 1               | Number of devices to use for training                 |
| max_epochs             | int             | Maximum number of epochs to train for                 |
| log_every_n_steps      | int             | Log training progress every n steps                   |
| check_eval_n_epoch     | 1               | Evaluate the model every n epochs                     |
| num_sanity_val_steps   | 0               | Number of validation steps to run at the beginning    |

See: [pytorch-lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api).

### Logging

| Parameter             | Value           | Description                                          |
|-----------------------|-----------------|------------------------------------------------------|
| include_caller        | bool            | If include function call signiture                     |
| stream               | bool             | If print to output                                    |
| min_level               | debig, info, warning, critical        | Minimal logging level           |


## Training config

| Parameter             | Value           | Description                                          |
|-----------------------|-----------------|------------------------------------------------------|
| learning_rate         | float          | Learning rate for the optimizer                      |
| epochs                | int            | Number of epochs to train for                         |
| optimizer             | Adam or similar            | Optimizer to use for training                         |
| early_stop_patience   | int              | Number of epochs to wait before early stopping        |
| weight_decay          | float           | Weight decay for the optimizer                        |
| scheduler_name        | null or CosineAnnealingWarmRestarts | Name of the learning rate scheduler to use            |


### Learning rate scheduler config

| Parameter             | Value           | Description                                          |
|-----------------------|-----------------|------------------------------------------------------|
| interval              | "step" or "epoch"            | Interval at which to reduce the learning rate         |
| reduce_lr_on_epoch    | float            | Factor by which to reduce the learning rate           |
| T_0                   | int           | Number of iterations before the first restart         |
| T_mult                | int               | Multiplicative factor by which to increase T_0        |
| eta_min               | float               | Minimum learning rate for the scheduler               |
