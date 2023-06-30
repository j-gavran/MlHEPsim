import copy

from ml_hep_sim.analysis.utils import get_colnames_dict
from ml_hep_sim.pipeline.blocks import (
    ClassifierRunnerBlock,
    DatasetBuilderBlock,
    ModelLoaderBlock,
    ReferenceDataLoaderBlock,
    VariableExtractBlock,
)
from ml_hep_sim.pipeline.pipeline_loggers import setup_logger
from ml_hep_sim.pipeline.pipes import Pipeline
from ml_hep_sim.pipeline.prebuilt.classifier_pipeline import ClassifierPipeline
from ml_hep_sim.pipeline.prebuilt.flow_pipeline import FlowPipeline


def get_generator_pipeline(use_classifier=True, var="m bb", N_gen=10**6, logger=None):
    """See notebook for details. This is used for importing and to seperate classification."""

    if logger is None:
        logger = setup_logger(dummy_logger=True)

    sig_override = {
        "datasets": {
            "data_name": "higgs_sig",
            "data_params": {
                "subset_n": [10**6, 10**5, 10**5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": "Higgs_MADEMOG", "experiment_name": "analysis"},
        "trainer_config": {"gpus": 1, "max_epochs": 1001},
        "model_config": {
            "num_flows": 10,
            "num_hidden_layers_mog_net": 32,
            "lr_scheduler_dct": {
                "scheduler": "ReduceLROnPlateau",
                "interval": "epoch",
                "params": {"mode": "min", "factor": 0.5, "patience": 8},
            },
        },
    }

    FP_sig = FlowPipeline(
        run_name="Higgs_MADEMOG_sig",
        model_name="MADEMOG",
        override=sig_override,
        pipeline_path="ml_pipeline/analysis/Higgs_MADEMOG/",
        logger=logger,
    )

    FP_sig.build_train_pipeline()
    FP_sig.fit()

    bkg_override = copy.deepcopy(sig_override)
    bkg_override["datasets"]["data_name"] = "higgs_bkg"

    FP_bkg = FlowPipeline(
        run_name="Higgs_MADEMOG_bkg",
        model_name="MADEMOG",
        override=bkg_override,
        pipeline_path="ml_pipeline/analysis/Higgs_MADEMOG/",
        logger=logger,
    )

    FP_bkg.build_train_pipeline()
    FP_bkg.fit()

    FP_sig.build_inference_pipeline(N_gen, rescale_data=False if use_classifier else True, device="cuda")
    FP_bkg.build_inference_pipeline(N_gen, rescale_data=False if use_classifier else True, device="cuda")

    sig_infer_pipeline = FP_sig.pipeline["inference_pipeline"]
    bkg_infer_pipeline = FP_bkg.pipeline["inference_pipeline"]

    mc_sig_config = copy.deepcopy(FP_sig.pipeline["train_pipeline"].pipes[0])  # same config as before
    mc_sig_config.config["datasets"]["data_params"]["subset_n"] = [0, 0, N_gen]
    mc_sig_config.config["datasets"]["data_params"]["rescale"] = "none"

    b_mc_sig_dataset = DatasetBuilderBlock()(mc_sig_config)
    b_mc_sig_data = ReferenceDataLoaderBlock(
        rescale_reference="logit_normal" if use_classifier else None, device="cpu"
    )(b_mc_sig_dataset)

    mc_bkg_config = copy.deepcopy(FP_bkg.pipeline["train_pipeline"].pipes[0])
    mc_bkg_config.config["datasets"]["data_params"]["subset_n"] = [0, 0, N_gen]
    mc_bkg_config.config["datasets"]["data_params"]["rescale"] = "none"

    b_mc_bkg_dataset = DatasetBuilderBlock()(mc_bkg_config)
    b_mc_bkg_data = ReferenceDataLoaderBlock(
        rescale_reference="logit_normal" if use_classifier else None, device="cpu"
    )(b_mc_bkg_dataset)

    b_flow_sig_generated = sig_infer_pipeline.pipes[-1]
    b_flow_bkg_generated = bkg_infer_pipeline.pipes[-1]

    if use_classifier:
        override = {
            "datasets": {
                "data_name": "higgs",
                "data_params": {
                    "subset_n": [10**6, 10**5, 10**5],
                    "rescale": "logit_normal",
                    "to_gpu": True,
                },
            },
            "logger_config": {"run_name": "Higgs_classifier", "experiment_name": "analysis"},
            "trainer_config": {"gpus": 1, "max_epochs": 101},
            "model_config": {
                "resnet": False,
                "learning_rate": 1e-3,
                "bayes_net": False,
                "hidden_layers": [128, 128, 128, 1],
                "lr_scheduler_dct": {
                    "scheduler": "ReduceLROnPlateau",
                    "interval": "epoch",
                    "params": {"mode": "min", "factor": 0.5, "patience": 4},
                },
            },
        }

        CP = ClassifierPipeline(
            "Higgs_classifier",
            override=override,
            pipeline_path="ml_pipeline/analysis/classifiers/",
            logger=logger,
        )

        CP.build_train_pipeline()
        CP.fit(force=False)

        class_train_pipeline = CP.pipeline["train_pipeline"]

        config = class_train_pipeline.pipes[0]  # classifier config block
        model = class_train_pipeline.pipes[3]  # classifier model trainer block

        b_classifier_model = ModelLoaderBlock(device="cuda")(config, model)

        b_sig_gen_class = ClassifierRunnerBlock(save_data=False, device="cuda")(
            b_flow_sig_generated, b_classifier_model
        )  # sig gen
        b_bkg_gen_class = ClassifierRunnerBlock(save_data=False, device="cuda")(
            b_flow_bkg_generated, b_classifier_model
        )  # bkg gen

        b_sig_mc_class = ClassifierRunnerBlock(save_data=False, device="cuda")(
            b_mc_sig_data, b_classifier_model
        )  # MC sig
        b_bkg_mc_class = ClassifierRunnerBlock(save_data=False, device="cuda")(
            b_mc_bkg_data, b_classifier_model
        )  # MC bkg

        pipe = Pipeline(logger=logger)
        pipe.compose(
            b_mc_sig_dataset,
            b_mc_sig_data,
            b_mc_bkg_dataset,
            b_mc_bkg_data,
            b_classifier_model,
            sig_infer_pipeline,
            bkg_infer_pipeline,
            b_sig_gen_class,
            b_bkg_gen_class,
            b_sig_mc_class,
            b_bkg_mc_class,
        )

    # using just one variable and not a classifier is largely untested
    else:
        dct = get_colnames_dict()
        idx = dct[var]

        b_sig_gen_var = VariableExtractBlock(idx, save_data=False, device="cuda")(b_flow_sig_generated)  # sig gen var
        b_bkg_gen_var = VariableExtractBlock(idx, save_data=False, device="cuda")(b_flow_bkg_generated)  # bkg gen var

        b_sig_mc_var = VariableExtractBlock(idx, save_data=False, device="cuda")(b_mc_sig_data)  # MC sig var
        b_bkg_mc_var = VariableExtractBlock(idx, save_data=False, device="cuda")(b_mc_bkg_data)  # MC bkg var

        pipe = Pipeline(logger=logger)
        pipe.compose(
            b_mc_sig_dataset,
            b_mc_sig_data,
            b_mc_bkg_dataset,
            b_mc_bkg_data,
            b_classifier_model,
            sig_infer_pipeline,
            bkg_infer_pipeline,
            b_sig_gen_var,
            b_bkg_gen_var,
            b_sig_mc_var,
            b_bkg_mc_var,
        )

    return pipe


if __name__ == "__main__":
    pipe = get_generator_pipeline()
    pipe.fit()
