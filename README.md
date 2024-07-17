# MlHEPsim

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" /></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
</p>

Using Machine Learning to Simulate Distributions of Observables at the Large Hadron Collider

Arxiv link: https://arxiv.org/abs/2310.08994

## Setup

More detailed instructions can be found [here](ml/custom/HIGGS/analysis/README.md).

### Virtual environment configuration

If you don't want to use Poetry, you can create a standard python virtual environment. In case you don't already have it, install `virtualenv`:

```bash
pip install pip --upgrade
pip install virtualenv
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment with:

```bash
source venv/bin/activate
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### Veryfing PyTorch installation

```bash
$ ipython
Python 3.11.3 (main, May 29 2023, 05:18:21) [GCC 13.1.1 20230520]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.15.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import torch

In [2]: torch.__version__
Out[2]: '2.0.1+cu118'

In [3]: torch.cuda.is_available()
Out[3]: True

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.current_device()
Out[5]: 0

In [6]: torch.cuda.device(0)
Out[6]: <torch.cuda.device at 0x7f3348ea1ed0>

In [7]: torch.cuda.get_device_name(0)
Out[7]: 'NVIDIA GeForce RTX 4090'
```

## Logging 

### Code logging

A basic logging configuration is set up [here](ml/common/utils/loggers.py). The logger is configured to write to a file in the `logs/` directory. The log file is name is a timestamp. The log level is set to `INFO` by default. Use it like this:
  
```python
from ml.common.loggers import setup_logger

setup_logger(min_level="info")
```
and log events in python code using the logging library:
```python
import logging

logging.debug("This is a debug message!")
logging.info("This is an info message!")
logging.warning("This is a warning message!")
logging.error("This is an error message!")
logging.critical("This is a critical message!")
```

### ML logging

ML logging is done with [MLflow](https://mlflow.org/). It is integrated into the codebase with ```lightning``` and can  be used to log parameters, metrics, and artifacts. The MLflow server is running on ```http://localhost:5000``` by default. To start the server, run:

```bash
mlflow ui
```
where port forwarding is done for you by VSCode.

To delete old runs, use the following command:

```bash
mlflow gc
```

## Formatting and linting

Formatting is done with [black](https://github.com/psf/black) and linting with [flake8](https://github.com/PyCQA/flake8). When in VSCode type `Ctrl+Shift+X` to open extensions. Type `@recommended` and install all workspace recommended extensions. To check the settings go to `.vscode/settings.json`.  
