# MlHEPsim

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" /></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
</p>

Using Machine Learning to Simulate Distributions of Observables at the Large Hadron Collider

# Table of contents: 

- [MlHEPsim](#mlhepsim)
- [Table of contents:](#table-of-contents)
- [Setup](#setup)
  - [Setup script](#setup-script)
  - [Logging](#logging)
- [Configuration](#configuration)
- [Running](#running)
- [Documentation](#documentation)
  - [Datasets](#datasets)
  - [Feature rescaling](#feature-rescaling)
  - [Neural networks](#neural-networks)
  - [VAEs](#vaes)
  - [Normalizing flows](#normalizing-flows)
  - [Statistics](#statistics)
    - [Tests](#tests)
    - [Other](#other)
  - [Plotting](#plotting)
  - [Using pipeline](#using-pipeline)
    - [Stage 1 blocks: Model building, training and saving](#stage-1-blocks-model-building-training-and-saving)
    - [Stage 2 blocks: Model loading, generation and verification](#stage-2-blocks-model-loading-generation-and-verification)
    - [Stage 3 blocks: Testing](#stage-3-blocks-testing)
    - [Other blocks](#other-blocks)
    - [Example: building a pipeline](#example-building-a-pipeline)
  - [Analysis setup](#analysis-setup)
    - [Workflow](#workflow)
    - [List of analysis blocks](#list-of-analysis-blocks)
  - [Miscellaneous](#miscellaneous)
# Setup

Clone repository:
```bash
git clone https://github.com/j-gavran/MlHEPsim.git
```

Make virtual environment:
```bash
pip install pip --upgrade
pip install virtualenv
python3 -m venv ml_hep_sim_env
```
Activate env:
```bash
source ml_hep_sim_env/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup script
```bash
source setup.sh
```
Need to change VENV_PATH to your location.

## Logging
MLflow runs are saved in ```mlruns/``` and can be accesed with:
```bash
mlflow ui
```

# Configuration
All model parameters can be set inside yaml files in [ml_hep_sim/conf/](ml_hep_sim/conf/) using Hydra library.

# Running
1. Directly from model files 
2. Using a pipeline

# Documentation

## Datasets
- ml_hep_sim/data_utils/toy_datasets.py, collection of 2D point datasets for testing and debugging:
    ```
    TOY_DATASETS = ["swissroll", "circles", "rings", "moons", "4gaussians", "8gaussians", "pinwheel", "2spirals", "checkerboard", "line", "cos", "fmf_normal", "fmf_uniform", "einstein"]
    ```
- ml_hep_sim/data_utils/mnist, see [preprocess_mnist.py](ml_hep_sim/data_utils/mnist/preprocess_mnist.py) for more info.
- ml_hep_sim/data_utils/higgs, download and preprocess Higgs dataset:
    ```bash
    python3 ml_hep_sim/data_utils/higgs/process_higgs_dataset.py
    ```

## Feature rescaling

- ml_hep_sim/data_utils/dataset_utils.py, rescale features using one of the following methods:
    ```
    - normal: zero mean and unit variance
    - robust: removes the median and scales the data according to the quantile range
    - sigmoid: [0, 1] range
    - tanh: [-1, 1] range
    - logit: [0, 1] -> [-inf, inf] ranges
    - logit_normal: [0, 1] -> [-inf, inf] -> normal ranges
    - Gauss scaler: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927
    ```

## Neural networks
- [Autoencoder](ml_hep_sim/nets/autoencoder.py)
- [Multi layer perceptron](ml_hep_sim/nets/mlp.py)
- [Residual networks](ml_hep_sim/nets/resnet.py)
- [U-net](ml_hep_sim/nets/u_net.py)
- [Classifiers](ml_hep_sim/nets/classifiers.py)
  - Binary classifier
  - Multi label class classifier

## VAEs
- [Vanilla VAE](ml_hep_sim/vaes/vae.py)
- [beta-VAE](ml_hep_sim/vaes/beta_vae.py)
- [sigma-VAE](ml_hep_sim/vaes/sigma_vae.py) 
- [2-stage VAE](ml_hep_sim/vaes/two_stage_vae.py)
- [B-VAE](ml_hep_sim/vaes/B_vae.py)

## Normalizing flows 
- [NICE](ml_hep_sim/normalizing_flows/nice.py)
- [RealNVP](ml_hep_sim/normalizing_flows/real_nvp.py)
- [Glow](ml_hep_sim/normalizing_flows/glow.py)
- [MADEMOG](ml_hep_sim/normalizing_flows/made_mog.py) 
- [MAF](ml_hep_sim/normalizing_flows/maf.py) 
- [Polynomial](ml_hep_sim/normalizing_flows/polynomial_splines.py)
- [Rational quadratic](ml_hep_sim/normalizing_flows/rq_splines.py)

## Statistics
### Tests
1. [N-dim tests](ml_hep_sim/stats/n_dim_tests/classifier_test.py)
   - Classification (train a classifier)
2. [1-dim tests](ml_hep_sim/stats/one_dim_tests/py_two_sample.py)
   - Two sample $\chi^2$ test
   - Kolmogorov-Smirnov test

### Other 
- pyhf [upper limits](ml_hep_sim/stats/ul.py) and [specs](ml_hep_sim/stats/pyhf_json_specs.py) 
- [Wasserstein distance](ml_hep_sim/stats/wasserstein.py)
- [Maximum mean discrepancy](ml_hep_sim/stats/mmd.py)
- [Statistics plots](ml_hep_sim/stats/stat_plots.py)
    - Two sample plot
    - N sample plot

## Plotting
- [HEP plot](ml_hep_sim/plotting/hep_plots.py) (standard stacked plot)
- Style (colors, font sizes, etc.)
- Matplotlib setup (latex)

## Using pipeline
Roughly, the ML [pipeline](ml_hep_sim/pipeline/pipes.py) consists of 3 stages built out of [blocks](ml_hep_sim/pipeline/blocks.py) holding intermediate results. The blocks are connected to each other in a direct acyclic graph fashion that can be thought of as a compositum of functions. This can be visualized as a tree with a ```draw_pipeline_tree()``` method. The pipeline is run by calling ```fit()``` method after composing all blocks with the ```compose()``` method. The blocks are:

### Stage 1 blocks: Model building, training and saving
- ConfigBuilderBlock 
- ModelBuilderBlock
- DatasetBuilderBlock
- ModelTrainerBlock

### Stage 2 blocks: Model loading, generation and verification
- ModelLoaderBlock
- DataGeneratorBlock
- GeneratedDataVerifierBlock
- ClassifierRunnerBlock

### Stage 3 blocks: Testing
- ReferenceDataLoaderBlock
- DistanceMetricRunnerBlock
- PCARunnerBlock
- StatTestRunnerBlock
- CouplingModelTestingBlock
- MADEMOGModelTestingBlock
- ScalingTestBlock

### Other blocks
Post stage 3, analysis specific:
- VariableExtractBlock
- RatioHighestValuesCutBlock
- CutBlock
- CutByIndexBlock
- RedoRescaleDataBlock
- GCBlock

### Example: building a pipeline
Training a HIGGS classifier with default parameters.

```python
from ml_hep_sim.pipeline.blocks import (
    ConfigBuilderBlock,
    DatasetBuilderBlock,
    ModelBuilderBlock,
    ModelTrainerBlock,
)
from ml_hep_sim.pipeline.distributed_pipes import Pipeline


class_train_pipeline = Pipeline(pipeline_name="classifier_train_pipeline", pipeline_path="ml_pipeline/")

x1 = ConfigBuilderBlock(config_path="../conf", config_name="classifier_config", model_name="BinaryClassifier")()
x2 = ModelBuilderBlock(model_type="other")(x1)
x3 = DatasetBuilderBlock()(x1)
x4 = ModelTrainerBlock()(x2, x3)

class_train_pipeline.compose(x1, x2, x3, x4)
class_train_pipeline.fit().save()
```

## Analysis setup
Most of the script files are also available as jupyter notebooks. The [notebooks](ml_hep_sim/analysis/notebook_pdfs/) are used for analysis and plotting. The ```.py``` files are used for running the analysis in a pipeline. Some quick generator tests are given in ```generators.py```.

### Workflow
Scripts should generally be run in the following order:
1. [Generator pipeline](ml_hep_sim/analysis/generator_pipeline.py)
2. [Cut pipeline](ml_hep_sim/analysis/cut_pipeline.py)
3. [Histogram pipeline](ml_hep_sim/analysis/hists_pipeline.py)
4. [Upper limit pipeline](ml_hep_sim/analysis/ul_pipeline.py)
    - Pull plots ([example pipeline tree](ml_hep_sim/analysis/results/pulls/pull_pipe.png))
5. [CLs pipeline](ml_hep_sim/analysis/cls_pipeline.py)
6. [Spurious signal pipeline](ml_hep_sim/analysis/spur_pipeline.py)

### List of analysis blocks 
- utils.py
  - SigBkgBlock
- hists_pipeline.py
  - MakeHistsFromSamples
  - MakeHistsFromSamplesLumi
- ul_pipeline.py
  - UpperLimitScannerBlock
  - PullBlock
- cls_pipeline.py
  - CLsBlock
  - CLsBlockResultsParser
- spur_pipeline.py
  - SpurBlock
  - SpurBlockResultsParser

## Miscellaneous
- [Dask testing](ml_hep_sim/pipeline/dask_testing/)
- [Prebuilt pipelines](ml_hep_sim/pipeline/prebuilt/)
- [Model testing notebooks](ml_hep_sim/model_testing_notebooks/)
- [Analysis results plots](ml_hep_sim/analysis/results/)
