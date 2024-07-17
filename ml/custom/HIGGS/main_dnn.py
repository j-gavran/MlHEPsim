import logging
import time

import hydra
import lightning as L

# lightning imports
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

# import common classes
from ml.classifiers.models.binary_model import BinaryClassifier
from ml.common.data_utils.processors import (
    Preprocessor,
    ProcessorChainer,
    SingleLabelPreprocessor,
)
from ml.common.utils.loggers import setup_logger
from ml.common.utils.register_model import register_from_checkpoint

# custom imports
from ml.custom.HIGGS.higgs_dataset import HiggsDataModule
from ml.custom.HIGGS.process_higgs_dataset import (
    CatGenerated,
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)


@hydra.main(config_path="config/dnn/", config_name="main_config", version_base=None)
def main(config):
    # get configuration
    experiment_conf = config.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    gen_model_name = "MADEMOG_flow_model_gauss_rank_best"
    experiment_name = "sigbkgClassifier"

    data_conf = config.data_config
    model_conf = config.model_config
    training_conf = config.training_config

    # change hold modes
    data_conf["input_processing"]["hold_mode"] = True
    data_conf["input_processing"]["use_hold"] = False

    # train on background and signal
    data_conf["feature_selection"]["on_train"] = None

    # match model postfix to rescale type
    experiment_conf["model_postfix"] = f"sigbkg_{data_conf['preprocessing']['cont_rescale_type']}"

    # do scaling on just bkg
    try:
        scaler_label = data_conf["preprocessing"]["scaler_label"]
    except KeyError:
        scaler_label = None

    # matmul precision and seed
    L.seed_everything(experiment_conf["seed"], workers=True)

    # internal logging
    setup_logger()

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    if scaler_label is not None:
        logging.info(f"[red]Scaling on label: {scaler_label}[/red]")
        scale_str = "bkg" if scaler_label == 0 else "sig"
        experiment_conf["model_postfix"] += f"_scale_on_{scale_str}"
        pre = SingleLabelPreprocessor(**data_conf["preprocessing"])
    else:
        pre = Preprocessor(**data_conf["preprocessing"])

    cat_gen = CatGenerated(gen_model_name, cat_label=0)

    chainer = ProcessorChainer(npy_proc, f_sel, pre, cat_gen)

    # create a data module
    dm = HiggsDataModule(chainer, **data_conf["dataloader_config"])

    # make model
    classifier = BinaryClassifier(model_conf, training_conf, tracker=None)

    # define callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                training_conf["epochs"]
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
        logger=mlf_logger,
        callbacks=callbacks,
    )

    if experiment_conf["model_postfix"] is not None:
        model_name = f"{model_conf['model_name']}_{experiment_conf['model_postfix']}"
    else:
        model_name = f"{model_conf['model_name']}_model"

    model_name += "_best7"
    # run training
    trainer.fit(classifier, dm)

    # save model
    register_from_checkpoint(trainer, classifier, model_name=model_name)


if __name__ == "__main__":
    main()
