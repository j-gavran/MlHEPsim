import logging
import time

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.utils.loggers import log_num_trainable_params, setup_logger, timeit
from ml.common.utils.register_model import register_from_checkpoint
from ml.custom.HIGGS.higgs_dataset import HiggsDataModule
from ml.custom.HIGGS.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)
from ml.flows.models import (
    MADEMOG,
    MAF,
    MAFMADEMOG,
    NICE,
    FlowModel,
    Glow,
    MOGFlowModel,
    PolynomialSplineFlow,
    RealNVP,
    RqSplineFlow,
)
from ml.flows.trackers import FlowTracker as Tracker


class DropLabelProcessor:
    def __init__(self, drop_labels):
        self.drop_labels = drop_labels

    def __call__(self, data, selection, scalers):
        data = self.drop(data, selection)
        return data, selection, scalers

    def drop(self, data, selection):
        logging.info(f"Dropping labels {self.drop_labels}!")

        labels_idx = selection[selection["type"] == "label"].index

        for label_idx in labels_idx:
            for drop_label in self.drop_labels:
                mask_label = data[:, label_idx] == drop_label
                data = data[~mask_label]
                logging.info(f"Dropped label {drop_label}! New data shape: {data.shape}.")

        return data


@timeit(unit="min")
@hydra.main(config_path="config/flows/", config_name="main_config", version_base=None)
def main(config):
    setup_logger()

    # get configuration
    experiment_conf = config.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "flows"

    data_conf = config.data_config
    model_conf = config.model_config
    training_conf = config.training_config

    # change hold modes
    data_conf["input_processing"]["hold_mode"] = True
    data_conf["input_processing"]["use_hold"] = False

    # match model postfix to rescale type
    experiment_conf["model_postfix"] = data_conf["preprocessing"]["cont_rescale_type"]

    if data_conf["preprocessing"]["disc_rescale_type"] is not None:
        experiment_conf["model_postfix"] += f"_{data_conf['preprocessing']['disc_rescale_type']}"

    # train on background
    on_train = data_conf["feature_selection"]["on_train"]

    # hack for mixed scaling (on both signal and background data and then drop 1 labels after)
    if on_train == "mixed":
        logging.warning("Will scale on mixed data and drop signal labels after!")
        data_conf["feature_selection"]["on_train"] = None
        experiment_conf["model_postfix"] += "_mixed"
        drop_labels = [1]

    # matmul precision and seed
    torch.set_float32_matmul_precision("high")
    L.seed_everything(experiment_conf["seed"], workers=True)

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    if on_train == "mixed":
        drop_proc = DropLabelProcessor(drop_labels)
        chainer = ProcessorChainer(npy_proc, f_sel, pre, drop_proc)
    else:
        chainer = ProcessorChainer(npy_proc, f_sel, pre)

    # create a data module
    data_module = HiggsDataModule(
        chainer,
        train_split=data_conf["train_split"],
        val_split=data_conf["val_split"],
        **data_conf["dataloader_config"],
    )

    # model configuration
    logging.info(f"Setting up {model_conf['model_name']} model.")

    # https://arxiv.org/abs/1410.8516
    if model_conf["model_name"].lower() == "nice":
        model = NICE(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1605.08803
    elif model_conf["model_name"].lower() == "realnvp":
        model = RealNVP(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1807.03039
    elif model_conf["model_name"].lower() == "glow":
        model = Glow(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1705.07057
    elif model_conf["model_name"].lower() == "maf":
        model = MAF(model_conf, data_conf, experiment_conf)

    elif model_conf["model_name"].lower() == "mafmademog":
        model = MAFMADEMOG(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1306.0186
    elif model_conf["model_name"].lower() == "mademog":
        model = MADEMOG(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1808.03856
    elif model_conf["model_name"].lower() == "polysplines":
        model = PolynomialSplineFlow(model_conf, data_conf, experiment_conf)

    # https://arxiv.org/abs/1906.04032
    elif model_conf["model_name"].lower() == "rqsplines":
        model = RqSplineFlow(model_conf, data_conf, experiment_conf)

    else:
        raise NameError

    tracker = Tracker(experiment_conf, tracker_path="ml/custom/HIGGS/metrics")

    logging.info("Done model setup.")

    log_num_trainable_params(model, unit="k")

    if model_conf["model_name"].lower() not in ["mademog", "mafmademog"]:
        flow = FlowModel(model_conf, training_conf, data_conf, model, tracker=tracker)
    else:
        flow = MOGFlowModel(model_conf, training_conf, data_conf, model, tracker=tracker)

    # define callbacks
    callbacks = [
        TQDMProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                experiment_conf["epochs"]
                if training_conf["early_stop_patience"] is None
                else training_conf["early_stop_patience"]
            ),
        ),
        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
    ]

    # initialize mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f'{model_conf["model_name"]}_{experiment_conf["run_name"]}',
        save_dir=experiment_conf["save_dir"],
        log_model=True,
    )

    # define trainer
    trainer = L.Trainer(
        max_epochs=training_conf["epochs"],
        accelerator=experiment_conf["accelerator"],
        devices=experiment_conf["devices"],
        check_val_every_n_epoch=experiment_conf["check_eval_n_epoch"],
        log_every_n_steps=experiment_conf["log_every_n_steps"],
        num_sanity_val_steps=experiment_conf["num_sanity_val_steps"],
        precision=experiment_conf["precision"],
        logger=mlf_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    if experiment_conf["model_postfix"] is not None:
        model_name = f"{model_conf['model_name']}_flow_model_{experiment_conf['model_postfix']}"
    else:
        model_name = f"{model_conf['model_name']}_flow_model"

    # run training
    trainer.fit(flow, data_module)

    # save model
    register_from_checkpoint(trainer, flow, model_name=model_name)


if __name__ == "__main__":
    main()
