import os
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
from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.stats.c2st import GeneratedProcessor, TwoSampleBuilder
from ml.common.utils.loggers import setup_logger
from ml.common.utils.register_model import register_from_checkpoint
from ml.custom.HIGGS.analysis.utils import get_model, sample_from_models

# custom imports
from ml.custom.HIGGS.higgs_dataset import HiggsDataModule
from ml.custom.HIGGS.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)


def get_generated_data(select_model, N, chunks=20, sample_i=0, ver=-1):
    model_dct = {select_model: get_model(select_model, ver=ver).eval()}

    if ver == -1:
        ver = ""

    model_dct[f"{select_model}{ver}"] = model_dct.pop(select_model)
    select_model = f"{select_model}{ver}"

    _, npy_file_names = sample_from_models(model_dct, N, ver=ver, chunks=chunks, resample=1, return_npy_files=True)

    file_path = npy_file_names[select_model][sample_i]
    head, tail = os.path.split(file_path)

    tail = tail.split(".")[0]

    return head, tail


@hydra.main(config_path="config/dnn/", config_name="main_config", version_base=None)
def main(config):
    closure_test = False

    N = 2 * 10**6

    # if closure test, double the number of samples in MC data and use that as baseline
    if closure_test:
        N = 2 * N

    select_models = [
        "RealNVP_flow_model_gauss_rank",
        "Glow_flow_model_gauss_rank",
        "rqsplines_flow_model_gauss_rank",
        "MAF_flow_model_gauss_rank",
        "MAFMADEMOG_flow_model_gauss_rank",
        "MADEMOG_flow_model_gauss_rank",
        "MADEMOG_flow_model_gauss_rank_mini",
    ]

    ver = 7
    cont_rescale_type = "gauss_rank"
    select_model = select_models[-2]

    # get configuration
    experiment_conf = config.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "c2st_all"

    data_conf = config.data_config
    model_conf = config.model_config
    training_conf = config.training_config

    # change cont_rescale_type
    data_conf["preprocessing"]["cont_rescale_type"] = cont_rescale_type

    # override n_data
    data_conf["feature_selection"]["n_data"] = N

    # change hold modes
    data_conf["input_processing"]["hold_mode"] = True
    data_conf["input_processing"]["use_hold"] = True

    # hijack background data for training in c2st
    data_conf["feature_selection"]["on_train"] = "bkg"

    # match model postfix to rescale type
    experiment_conf["model_postfix"] = f"{data_conf['preprocessing']['cont_rescale_type']}"

    # set early stop patience and epochs
    # training_conf["early_stop_patience"] = 4
    # training_conf["epochs"] = 7

    # small batch size
    data_conf["dataloader_config"]["batch_size"] = 128

    # matmul precision and seed
    L.seed_everything(experiment_conf["seed"], workers=True)

    # internal logging
    setup_logger()

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    # build two sample object from MC and generated data
    if not closure_test:
        file_dir, file_name = get_generated_data(select_model=select_model, N=N, chunks=20, ver=ver)
        gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)
    else:
        gen_proc = None

    two_samples_proc = TwoSampleBuilder(
        processor_X=chainer,
        processor_Y=gen_proc,
        add_label_X=False,
        add_label_Y=True,
        hold_out_ratio=0.5,
        shuffle_random_state=0,
    )

    # create MC+gen two sample data module
    dm = HiggsDataModule(two_samples_proc, **data_conf["dataloader_config"])

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

    if closure_test:
        model_name = f"{model_conf['model_name']}_mc_c2st_model"
    else:
        model_name = f"{model_conf['model_name']}_{select_model}_c2st_gen_model"

    model_name += "_all"

    # run training
    trainer.fit(classifier, dm)

    # save model
    register_from_checkpoint(trainer, classifier, model_name=model_name)


if __name__ == "__main__":
    main()
