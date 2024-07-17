import time

# pytorch, pytorch lightning and hydra imports
import hydra
import lightning as L

# lightning imports
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

# import common classes
from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.utils.loggers import log_model_summary, setup_logger
from ml.common.utils.register_model import (
    fetch_registered_module,
    register_from_checkpoint,
)

# custom imports
from ml.custom.HIGGS.higgs_dataset import HiggsDataModule
from ml.custom.HIGGS.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

# vae imports
from ml.vae.models.base_vae import VAEModule
from ml.vae.models.beta_vae import BetaVAE
from ml.vae.models.sigma_vae import SigmaVAE
from ml.vae.models.vae import VAE
from ml.vae.trackers import SigBkgVAETracker


@hydra.main(config_path="config/vae/", config_name="main_config", version_base=None)
def main(config):
    # get configuration
    experiment_conf = config.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    if experiment_conf["experiment_name"] is None:
        t = time.localtime()
        experiment_name = f"vae_HIGGS_{t.tm_mday:02d}{t.tm_mon:02d}{(t.tm_year%100):02d}"

    data_conf = config.data_config
    model_conf = config.model_config
    training_conf = config.training_config

    # matmul precision and seed
    L.seed_everything(experiment_conf["seed"], workers=True)

    # internal logging
    setup_logger()

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    # create a data module
    data_module = HiggsDataModule(chainer, **data_conf["dataloader_config"])

    if model_conf["model_name"].lower() == "vae":
        model = VAE(model_conf, data_conf, experiment_conf)

    elif model_conf["model_name"].lower() == "beta_vae":
        model = BetaVAE(model_conf, data_conf, experiment_conf)

    elif model_conf["model_name"].lower() == "sigma_vae":
        model = SigmaVAE(model_conf, data_conf, experiment_conf)

    else:
        raise NameError

    log_model_summary(model, data_conf)

    tracker = SigBkgVAETracker(experiment_conf, tracker_path="ml/custom/HIGGS/metrics", proc=pre)

    vae = VAEModule(model_conf, training_conf, model, data_conf, tracker=tracker)

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
        run_name=f'{model_conf["model_name"]}_{experiment_conf["run_name"]}_{experiment_conf["stage"]}',
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

    model_name = f"{model_conf['model_name']}_model"

    # run training or return for testing
    if experiment_conf["stage"] == "train":
        trainer.fit(vae, data_module)
        trainer.test(vae, data_module)
        register_from_checkpoint(trainer, vae, model_name=model_name)

    elif experiment_conf["stage"] == "test":
        module = fetch_registered_module(
            model_name,
            model_version=experiment_conf["model_version"],
            device=experiment_conf["device"],
        )
        vae.model = module.model

        return trainer, vae, data_module, tracker

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
