# python
import logging
import os
import pathlib
import random

# other stuff
import matplotlib.pyplot as plt
import mlflow
import netron
import numpy as np

# pytorch lightning
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

# datasets
from ml_hep_sim.data_utils.higgs.higgs_dataset import HiggsDataModule
from ml_hep_sim.data_utils.mnist.mnist_dataset import MnistDataModule
from ml_hep_sim.data_utils.toy_datasets import TOY_DATASETS, ToyDataModule


def set_random(seed=0, deterministic=False):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def load_from_checkpoint(model, logger=None, path=None):
    """Loads model using Tensorboard.

    References
    ----------
    [1] - https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html#pytorch_lightning.loggers.tensorboard.TensorBoardLogger

    """
    assert not (logger is None and model.logger is None and path is None)
    if logger is None:
        logger = model.logger

    if path is None:
        save_dir, name, version = logger._save_dir, logger._name, logger._version
        path = f"{save_dir}{name}/version_{version}/checkpoints/"

    try:
        files = [str(p) for p in pathlib.Path(path).iterdir() if p.is_file()]
    except FileNotFoundError:
        return None

    logging.warning(f"found checkpoints: {files}")

    models = []
    for f in files:
        models.append(model.load_from_checkpoint(f))

    return models


def netron_model(model, forward_sample, onnx_path=None):
    """Viewer for neural networks.

    Parameters
    ----------
    model : nn.Module
        Torch model.
    forward_sample : torch.Tensor
        Dummy batch.
    onnx_path : str, optional
        Path to an already existing onnx file, by default None.

    References
    ----------
    [1] - https://github.com/lutzroeder/netron

    """
    if onnx_path is None:
        onnx_path = "model.onnx"

    torch.onnx.export(model, forward_sample, onnx_path)
    netron.start(onnx_path)


def lr_finder(trainer, model, data_module, plot=False):
    """Find best learning rate automatically.

    Note
    ----
    Suggested lr is half way between last loss (at smallest scanned lr) and lowest loss.

    References
    ----------
    [1] - https://pytorch-lightning.readthedocs.io/en/stable/advanced/lr_finder.html

    """

    lr_finder = trainer.tuner.lr_find(
        model,
        data_module,
        num_training=1000,
        early_stop_threshold=None,
        mode="exponential",
        min_lr=1e-6,
        max_lr=1,
    )

    lr, loss = np.array(lr_finder.results["lr"]), np.array(lr_finder.results["loss"])
    not_nan_idx = np.argwhere(~np.isnan(loss))
    lr, loss = lr[not_nan_idx], loss[not_nan_idx]

    suggest_loss = (np.min(loss) + loss[0]) / 2
    idx = np.argmin(np.abs(loss - suggest_loss))
    suggest_lr = lr[idx][0]

    pl_suggest_lr = lr_finder.suggestion()

    if plot:
        fig, ax = plt.subplots()
        ax.plot(lr.flatten(), loss.flatten())
        ax.axvline(suggest_lr, c="r", ls="--", label=f"mine: {suggest_lr:.5e}")
        ax.axvline(pl_suggest_lr, c="b", ls="--", label=f"pl: {pl_suggest_lr:.5e}")
        ax.set_xscale("log")
        plt.legend()
        plt.show()

    logging.warning(f"pl lr: {pl_suggest_lr} (using this), my lr: {suggest_lr}")

    model.hparams.learning_rate = pl_suggest_lr
    model.learning_rate = pl_suggest_lr


def mlf_trainer(
    pl_model,
    data_module,
    *,
    experiment_name,
    run_name,
    trainer_dict,
    model_name="",
    torch_model_name=None,
    early_stop_dict=None,
    find_lr=False,
    **logger_kwargs,
):
    """Setup for:
        - tracking with https://mlflow.org/
        - lr finding
        - early stopping
        - saving model

    Parameters
    ----------
    pl_model : pl.LightningModule
        Lightning module.
    data_module : pl.LightningDataModule
        Lightning data module.
    experiment_name : str, optional
        See [3], by default None.
    run_name : str, optional
        See [3], by default None.
    trainer_dict : dict, optional
        Dict for pl.Trainer kwargs, by default None.
    model_name : str, optional
        Trained torch model name, by default "".
    torch_model_name : str
        Torch submodel to save. If None saves pl model.
    early_stop_dict: dict
        {"monitor": "val_loss", "mode": "min/max", "patience": int}.
    find_lr: bool
        If True finds lr using pl. Raises Exception and plots loss vs lr curve when done.

    References
    ----------
    [1] - mlflow docs https://www.mlflow.org/docs/latest/python_api/mlflow.html#module-mlflow
    [2] - mlflow Pytorch docs https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html#module-mlflow.pytorch
    [3] - MLFlowLogger from Pytorch Lightning docs https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.loggers.MLFlowLogger.html#mlflowlogger

    Note
    ----
    Live logs: write `mlflow ui` in terminal and view at http://localhost:5000.

    """
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name, **logger_kwargs)

    callbacks = [LearningRateMonitor(logging_interval="step")]

    if early_stop_dict is not None:
        callbacks.append(EarlyStopping(**early_stop_dict))

    if type(trainer_dict) is not dict:
        trainer_dict = dict(trainer_dict)

    if trainer_dict is None:
        trainer = pl.Trainer(logger=mlf_logger, callbacks=callbacks)
    else:
        gpus = trainer_dict.pop("gpus")

        if gpus > 0:
            accelerator, devices = "gpu", gpus
        else:
            accelerator, devices = "cpu", "auto"

        trainer = pl.Trainer(
            logger=mlf_logger, accelerator=accelerator, devices=devices, callbacks=callbacks, **trainer_dict
        )

    if find_lr:
        lr_finder(trainer, pl_model, data_module, plot=True)
        raise Exception("lr found, exiting...")

    trainer.fit(pl_model, data_module)

    attr = None
    if torch_model_name is None:
        attr = pl_model
    elif type(torch_model_name) is list:
        for a in torch_model_name:
            attr = getattr(pl_model, a)
    else:
        attr = getattr(pl_model, torch_model_name)

    path = os.getcwd() + "/mlruns/" + mlf_logger.experiment_id + "/" + mlf_logger.run_id
    artifact_path = path + "/artifacts/"
    checkpoint_path = path + "/checkpoints/"

    try:
        mlflow.pytorch.save_model(attr, artifact_path + model_name)
        logging.warning("saved {} to {}".format(model_name, artifact_path))
    except TypeError:
        logging.warning("cannot pickle 'weakref' object, saving state dict instead")
        torch.save(attr.state_dict(), checkpoint_path + "model.pth")
        logging.warning("saved {} to {}".format("model.pth", checkpoint_path))

    return path, artifact_path, checkpoint_path


def mlf_loader(path, from_checkpoint=False, device="cpu"):
    """https://pytorch.org/tutorials/beginner/saving_loading_models.html"""
    if from_checkpoint:
        logging.warning("loaded from state dict: use model.load_state_dict(<return>)")
        logging.warning("if model not initialized forward pass a dummy tensor")
        return torch.load(os.getcwd() + path + "/checkpoints/model.pth")
    else:
        # this is a wrapper for https://pytorch.org/docs/stable/generated/torch.load.html
        return mlflow.pytorch.load_model(path, map_location=torch.device(device))


def train_wrapper(
    config,
    *,
    input_dim,
    pl_model,
    trainer_dict,
    data_name="",
    data_dir="",
    mlf_trainer_dict=None,
    data_param_dict=None,
    trained_model=None,
    data_module=None,
    **model_kwargs,
):
    if data_name.lower() == "mnist":
        data_module = MnistDataModule(
            [
                data_dir + "data/mnist/train.npy",
                data_dir + "data/mnist/test.npy",
            ],
            **data_param_dict,
        )
    elif data_name.lower() == "higgs":
        data_module = HiggsDataModule(
            [
                data_dir + "data/higgs/HIGGS_18_feature_train.npy",
                data_dir + "data/higgs/HIGGS_18_feature_val.npy",
                data_dir + "data/higgs/HIGGS_18_feature_test.npy",
            ],
            **data_param_dict,
        )
    elif data_name.lower() == "higgs_bkg":
        data_module = HiggsDataModule(
            [
                data_dir + "data/higgs/HIGGS_18_feature_bkg_train.npy",
                data_dir + "data/higgs/HIGGS_18_feature_bkg_val.npy",
                data_dir + "data/higgs/HIGGS_18_feature_bkg_test.npy",
            ],
            **data_param_dict,
        )
    elif data_name.lower() == "higgs_sig":
        data_module = HiggsDataModule(
            [
                data_dir + "data/higgs/HIGGS_18_feature_sig_train.npy",
                data_dir + "data/higgs/HIGGS_18_feature_sig_val.npy",
                data_dir + "data/higgs/HIGGS_18_feature_sig_test.npy",
            ],
            **data_param_dict,
        )
    elif data_name in TOY_DATASETS:
        data_module = ToyDataModule(data_name, **data_param_dict)
    else:
        assert data_module
        logging.warning("using custom data module...")

    if trained_model is None:
        model = pl_model(config, input_dim, data_name=data_name, **model_kwargs)
        if hasattr(data_module, "scalers"):
            model.scalers = data_module.scalers
    else:
        model = trained_model

    mlf_trainer(model, data_module, trainer_dict=trainer_dict, **mlf_trainer_dict)

    return model
