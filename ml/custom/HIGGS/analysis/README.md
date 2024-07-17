# Analysis code

## Setup 

- CPU monitorig: `htop`  
- GPU monitoring: `nvtop` or `nvidia-smi -l 1`  
- Training monitoring: `mlflow ui --port <number>`. Registered model names can be found in the Models tab. Comparison of two (or more) models can be done by clicking on the model name and selecting the Compare button.

## Model training

### Flows

Flow models can be trained from `main_flows.py`. Configuration is in `config/flows/` where different model can be chosen from the `main_config.yaml` file in the `model_config` field. Each model has its own configuration file in the `model_config/` directory.

### Binary classification

Sig/bkg classiffer can be trained from the `main_dnn.py`. Configuration is set in the `config/dnn/` directory.

### C2ST

C2ST models can be trained from the `main_c2st.py`. Configuration is set in the `config/dnn/` directory.

## Analysis

Model names are set in the `utils.py` in the `MODEL_MAP` dictionary. All generated events are cached in the `data/` directory as `.npy` files. Plots are saved in the `analysis/plots/` folder.

Data is additionaly split into two independent partitions: `hold_partition_1` and `hold_partition_2`. The (second) holdout partitions are used to evaluate the performance of the generative models.

## Part 1: model performance

### Generative model performance

`gen_model_perf.py` can be used to evaluate the performance of the generative models. 

### Distances

`distances.py` can be used to calculate the distances between the generated and real data (KL, hellinger, chi2 and Wasserstein).

### C2ST performance

- `c2st_train_perf.py` plots accuracy and loss during the training of the C2ST models.  
- `c2st.py` ROC, confusion matrix and TP/FP/FN/TN distributions.  
- `c2st_tests.py` two sample plots on tail cuts, KS and chi2 tests and c2st outputs.  

### Classification

`class_model_perf.py` performs binary sig/bkg classification on trained classifier. Plots ROC curve and classifier sigmoid outputs.

### Cut tests*

`cut_tests.py` cuts on generated data on two variables to check correlation between them.

### Anomaly detection*

`flow_anomlay.py` try to do anomaly detection of signal with flow density estimation (trained on background).

## Part 2: statistical analysis

### Classifier cuts

`fit_cuts.py` perform cut on classifier score. Plot after cuts distributions, bkg/gen ratio and ROC curve.

### Histograms

`fit_hists.py` makes MC/data histograms.

### Pyhf setup

`fit_pyhf.py` main fit setup. Pyhf model specification and defines `Template` and `FitSetup` classes.

### MLE fit

`fit_mle.py` signal strength $\mu$ and spurious signal fit. Prefit and postfit histograms.

### Upper limits

`fit_ul.py` upper limits plots.

### CLs

`fit_cls.py` CLs plots.

### Pulls*

`fit_pulls.py` pull plots.