"""Deprecated! See prebuilt pipelines in ml_hep_sim/pipeline. Only here for backwards compatibility."""
from ml_hep_sim.plotting.style import style_setup

import numpy as np
import copy
from tqdm import tqdm

style_setup(seaborn_pallete=True)


from ml_hep_sim.pipeline.blocks import (
    ConfigBuilderBlock,
    ModelBuilderBlock,
    DatasetBuilderBlock,
    ModelTrainerBlock,
    ReferenceDataLoaderBlock,
    ClassifierRunnerBlock,
    ModelLoaderBlock,
)
from ml_hep_sim.pipeline.pipes import Pipeline


def classifier_pipeline(run_name, override=None, train=True, run=True, test_dataset="higgs_bkg"):
    """Classifier pipeline example.

    Parameters
    ----------
    run_name : str
        Name of this run.
    override : _type_, optional
        Hydra config parameters, by default None.
    train : bool, optional
        If True train the model else skip training and load trained model, by default True.
    run : bool, optional
        Run classification if True else load classified, by default True.
    test_dataset : str, optional
        Name of the dataset for classification testing, by default "higgs_bkg".

    Example
    -------
    ```
    pipeline = classifier_pipeline(
        run_name="Higgs_resnet_classifier",
        override={
            "datasets": {
                "data_name": "higgs",
                "data_params": {
                    "subset_n": [10 ** 6, 10 ** 5, 10 ** 5],
                },
            },
            "logger_config": {"run_name": "Higgs_resnet_classifier"},
            "trainer_config": {"gpus": 1, "max_epochs": 101},
            "model_config": {"resnet": True},
        },
        train=True,
        run=True,
    )
    ```

    Returns
    -------
    tuple
        Pipelines.
    """
    class_train_pipeline = Pipeline(pipeline_name=f"{run_name}_train_pipeline", pipeline_path="ml_pipeline/")

    x1 = ConfigBuilderBlock(
        override_config=override,
        config_path="../conf",
        config_name="classifier_config",
        model_name="BinaryClassifier",
    )()
    x2 = ModelBuilderBlock(model_type="other")(x1)
    x3 = DatasetBuilderBlock()(x1)
    x4 = ModelTrainerBlock()(x2, x3)

    class_train_pipeline.compose(x1, x2, x3, x4)

    if train:
        class_train_pipeline.fit().save()
    else:
        class_train_pipeline.load()

    class_run_pipeline = Pipeline(pipeline_name=f"{run_name}_run_pipeline", pipeline_path="ml_pipeline/")

    x1.override_config["datasets"]["data_name"] = test_dataset
    x1.fit_block = True
    x5 = DatasetBuilderBlock()(x1)
    x6 = ReferenceDataLoaderBlock()(x5)
    x7 = ModelLoaderBlock()(x1, x4)
    x8 = ClassifierRunnerBlock(save_data=True)(x6, x7)

    class_run_pipeline.compose(x1, x5, x6, x7, x8)

    if run:
        class_run_pipeline.fit().save()
    else:
        class_run_pipeline.load()

    return class_train_pipeline, class_run_pipeline


from ml_hep_sim.pipeline.blocks import (
    ConfigBuilderBlock,
    ModelBuilderBlock,
    DatasetBuilderBlock,
    ModelTrainerBlock,
    ModelLoaderBlock,
    DataGeneratorBlock,
    GeneratedDataVerifierBlock,
    DistanceMetricRunnerBlock,
    ReferenceDataLoaderBlock,
    ClassifierRunnerBlock,
    PCARunnerBlock,
    StatTestRunnerBlock,
)
from ml_hep_sim.pipeline.pipes import Pipeline


CONFIGS = {
    "Glow": "glow_config",
    "MADEMOG": "made_mog_config",
    "MAFMADE": "maf_config",
    "MAFMADEMOG": "maf_config",
    "NICE": "nice_config",
    "PolynomialSpline": "polynomial_splines_config",
    "RealNVP": "realnvp_config",
    "RqSpline": "rq_splines_config",
}


def flow_pipeline(
    run_name,
    flow_model_name,
    train=True,
    gen=True,
    test=True,
    override=None,
    N=10 ** 5,
    skip_gen_test=False,
    device="cpu",
    class_run_name="Higgs_classifier",
    pipeline_path="ml_pipeline/",
):
    """Flow pipeline example.

    Parameters
    ----------
    run_name : str
        Name of this run.
    flow_model_name : str
        Name of the model, see CONFIGS.
    train : bool, optional
        If True train the model else skip training and load trained model, by default True.
    gen : bool, optional
        If True perform data generation on model, by default True.
    test : bool, optional
        If True do testing with generated data, by default True.
    override : dict, optional
        Hydra config parameters, by default None.
    N : int, optional
        Number of generated data points, by default 10**5.
    skip_gen_test : bool, optional
        If True only train and skip generation and testing steps, by default False.
    device : str, optional
        Tensor device, by default "cpu".
    class_run_name : str, optional
        Classifier pipeline string, by default "Higgs_classifier".
    pipeline_path : str, optional
        Path where to save this pipeline, by default "ml_pipeline/".

    Example
    -------
    ```
    pipeline = flow_pipeline(
        run_name="Higgs_glow",
        flow_model_name="Glow",
        override={
            "datasets": {
                "data_name": "higgs_bkg",
                "data_params": {
                    "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                },
            },
            "logger_config": {"run_name": "Higgs_glow", "experiment_name": "TEST"},
            "trainer_config": {"gpus": 1, "max_epochs": 1},
        },
        class_run_name="Higgs_resnet_classifier_train_pipeline",
        train=True,
        gen=True,
        test=True,
    )
    ```

    Returns
    -------
    tuple
        Pipelines.
    """
    flow_train_pipe = Pipeline(pipeline_name=f"{run_name}_train_pipe", pipeline_path=pipeline_path)

    x1 = ConfigBuilderBlock(
        override_config=override,
        config_path="../conf",
        config_name=CONFIGS[flow_model_name],
        model_name=flow_model_name,
    )()  # build configuration from yaml and overriden config dict
    x2 = ModelBuilderBlock(model_type="flow")(x1)  # build model
    x3 = DatasetBuilderBlock()(x1)  # build dataset
    x4 = ModelTrainerBlock()(x2, x3)  # train model (save trained model path from mlruns folder)

    flow_train_pipe.compose(x1, x2, x3, x4)

    if train:
        flow_train_pipe.fit().save()
    else:
        flow_train_pipe.load()

    if not skip_gen_test:
        flow_generation_pipe = Pipeline(pipeline_name=f"{run_name}_generation_pipe", pipeline_path=pipeline_path)

        x5 = ModelLoaderBlock(device=device)(x1, x4)  # load trained model
        x6 = DataGeneratorBlock(N, model_type="flow", chunks=10, device=device)(x5)  # generate data from model
        x7 = GeneratedDataVerifierBlock(save_data=True, device=device)(
            x5, x6
        )  # rescale and remove invalid values in data

        flow_generation_pipe.compose(x5, x6, x7)

        if gen:
            flow_generation_pipe.fit().save()
        else:
            flow_generation_pipe.load()

        flow_test_pipe = Pipeline(pipeline_name=f"{run_name}_testing_pipe", pipeline_path=pipeline_path)

        x8 = DatasetBuilderBlock()(x1)  # build dataset for validation (reference) data
        x9 = ReferenceDataLoaderBlock()(x8)  # get validation data from dataset
        x10 = DistanceMetricRunnerBlock()(x7, x9)  # calculate f-divergences from reference and generated

        class_train_pipeline = Pipeline(pipeline_name=class_run_name, pipeline_path="ml_pipeline/")
        class_train_pipeline.load()

        x11 = ModelLoaderBlock(device=device)(
            class_train_pipeline.pipes[0],
            class_train_pipeline.pipes[-1],
        )  # load classifier
        x12 = ClassifierRunnerBlock(save_data=True)(x9, x11)  # run classifier on reference
        x13 = ClassifierRunnerBlock(save_data=True)(x7, x11)  # run classifier on generated

        x14 = PCARunnerBlock(save_data=True)(x9)  # run PCA on reference
        x15 = PCARunnerBlock(save_data=True)(x7)  # run PCA on generated

        x16 = StatTestRunnerBlock(save_data=True)(x14, x15)  # run stat tests on PCA reduction
        x17 = StatTestRunnerBlock(save_data=True)(x12, x13)  # run stat tests on classifier reduction

        flow_test_pipe.compose(x8, x9, x10, x11, x12, x13, x14, x15, x16, x17)

        if test:
            flow_test_pipe.fit().save()
        else:
            flow_test_pipe.load()

        return flow_train_pipe, flow_generation_pipe, flow_test_pipe
    else:
        return flow_train_pipe


def run_glow_pipeline(train, gen, test, sig=False, num_flows=None, num_train=None, skip_gen_test=True):
    if num_train is None:
        num_trains = [int(2.5 * 10 ** 5)]
    else:
        num_trains = num_train

    if num_flows is None:
        num_flows = np.arange(4, 32, 2)

    run_name = "Higgs_Glow"

    base_config = {
        "datasets": {
            "data_name": "higgs_sig" if sig else "higgs_bkg",
            "data_params": {
                "subset_n": [num_trains[0], 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
            },
        },
        "logger_config": {"run_name": run_name, "experiment_name": "Glow"},
        "trainer_config": {"gpus": 1, "max_epochs": 81},
        "model_config": {"num_flows": 10},
    }

    configs = []

    for n in num_flows:
        base_config["model_config"]["num_flows"] = n
        base_config["logger_config"]["run_name"] = run_name + (
            f"_flow_blocks_{n}" if sig is False else f"_flow_blocks_{n}_sig"
        )
        if num_train is not None:
            for t in num_trains:
                base_config["datasets"]["data_params"]["subset_n"] = [t, 10 ** 5, 10 ** 5]
                base_config["logger_config"]["run_name"] = run_name + (
                    f"_flow_blocks_{n}" + f"_{t}" if sig is False else f"_flow_blocks_{n}_sig" + f"_{t}"
                )
                configs.append(copy.deepcopy(base_config))
        else:
            configs.append(copy.deepcopy(base_config))

    pipelines = []

    for config in tqdm(configs):
        pipeline = flow_pipeline(
            run_name=config["logger_config"]["run_name"],
            flow_model_name="Glow",
            pipeline_path=f"ml_pipeline/{run_name}/",
            override=config,
            class_run_name="Higgs_linear_classifier_train_pipeline",
            train=train,
            gen=gen,
            test=test,
            skip_gen_test=skip_gen_test,
        )
        pipelines.append(pipeline)

    return pipelines


def run_maf_pipeline(
    train, gen, test, sig=False, use_mog=True, use_maf=True, num_mogs=None, name_str="", num_train=None
):
    if use_mog and use_maf:
        run_name = "Higgs_MAFMADEMOG"
    elif use_mog and not use_maf:
        run_name = "Higgs_MADEMOG"
    elif not use_mog and use_maf:
        run_name = "Higgs_MAFMADE"
    else:
        raise NameError

    if num_train is None:
        num_trains = [int(2.5 * 10 ** 5)]
    else:
        num_trains = num_train

    if num_mogs is None:
        num_mogs = np.concatenate([[1], np.arange(2, 26, 2)])

    base_config = {
        "datasets": {
            "data_name": "higgs_sig" if sig else "higgs_bkg",
            "data_params": {
                "subset_n": [int(2.5 * 10 ** 5), 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
            },
        },
        "logger_config": {"run_name": run_name, "experiment_name": "MAF"},
        "trainer_config": {"gpus": 1, "max_epochs": 101},
        "model_config": {"num_flows": 10, "use_mog": use_mog},
    }

    if use_mog:
        configs = []

        for n in num_mogs:
            base_config["model_config"]["n_mixtures"] = n
            base_config["logger_config"]["run_name"] = run_name + (
                f"_mogs_{n}" + name_str if sig is False else f"_mogs_{n}_sig" + name_str
            )
            if num_train is not None:
                for t in num_trains:
                    base_config["datasets"]["data_params"]["subset_n"] = [t, 10 ** 5, 10 ** 5]
                    base_config["logger_config"]["run_name"] = run_name + (
                        f"_mogs_{n}" + f"_{t}" + name_str if sig is False else f"_mogs_{n}_sig" + f"_{t}" + name_str
                    )
                    configs.append(copy.deepcopy(base_config))
            else:
                configs.append(copy.deepcopy(base_config))

        pipelines = []

        for config in tqdm(configs):
            pipeline = flow_pipeline(
                run_name=config["logger_config"]["run_name"],
                flow_model_name="MAFMADEMOG" if use_maf else "MADEMOG",
                pipeline_path=f"ml_pipeline/{run_name}/",
                override=config,
                class_run_name="Higgs_linear_classifier_train_pipeline",
                train=train,
                gen=gen,
                test=test,
                skip_gen_test=True,
            )
            pipelines.append(pipeline.load())

        return pipelines

    else:
        base_config["logger_config"]["run_name"] = run_name + (
            f"_mogs_0" + name_str if sig is False else f"_mogs_0_sig" + name_str
        )

        if num_train:
            base_config["datasets"]["data_params"]["subset_n"] = [t, 10 ** 5, 10 ** 5]
            base_config["logger_config"]["run_name"] += f"_{t}"

        pipeline = flow_pipeline(
            run_name=base_config["logger_config"]["run_name"],
            flow_model_name="MAFMADE",
            pipeline_path=f"ml_pipeline/{run_name}/",
            override=base_config,
            class_run_name="Higgs_linear_classifier_train_pipeline",
            train=train,
            gen=gen,
            test=test,
            skip_gen_test=True,
        )

        return pipeline


def run_spline_pipeline(train, gen, test, sig=False, num_splines=None, name_str="", num_train=None):
    run_name = "Higgs_spline"

    if num_train is None:
        num_trains = [int(2.5 * 10 ** 5)]
    else:
        num_trains = num_train

    base_config = {
        "datasets": {
            "data_name": "higgs_sig" if sig else "higgs_bkg",
            "data_params": {
                "subset_n": [num_trains, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
            },
        },
        "logger_config": {"run_name": run_name, "experiment_name": "Splines"},
        "trainer_config": {"gpus": 1, "max_epochs": 101},
        "model_config": {"num_flows": 10, "ar": True, "resnet": True},
    }

    if num_splines is None:
        num_splines = np.arange(4, 32, 2)

    configs = []

    for n in num_splines:
        base_config["model_config"]["bins"] = n
        base_config["logger_config"]["run_name"] = run_name + (
            f"_splines_{n}" + name_str if sig is False else f"_splines_{n}_sig" + name_str
        )

        if num_train is not None:
            for t in num_trains:
                base_config["datasets"]["data_params"]["subset_n"] = [t, 10 ** 5, 10 ** 5]
                base_config["logger_config"]["run_name"] = run_name + (
                    f"_splines_{n}" + f"_{t}" + name_str if sig is False else f"_splines_{n}_sig" + f"_{t}" + name_str
                )
                configs.append(copy.deepcopy(base_config))
        else:
            configs.append(copy.deepcopy(base_config))

    pipelines = []

    for config in tqdm(configs):
        pipeline = flow_pipeline(
            run_name=config["logger_config"]["run_name"],
            flow_model_name="RqSpline",
            pipeline_path=f"ml_pipeline/{run_name}/",
            override=config,
            class_run_name="Higgs_linear_classifier_train_pipeline",
            train=train,
            gen=gen,
            test=test,
            skip_gen_test=True,
        )
        pipelines.append(pipeline)

    return pipelines


if __name__ == "__main__":
    # run_glow_pipeline(
    #     True,
    #     False,
    #     False,
    #     sig=False,
    #     num_flows=[10],
    #     num_train=[599484],
    # )
    #
    #     run_maf_pipeline(
    #         True,
    #         False,
    #         False,
    #         sig=False,
    #         use_mog=True,
    #         use_maf=True,
    #         num_mogs=np.concatenate([[1], np.arange(2, 26, 2)]),
    #     )
    #
    # run_maf_pipeline(
    #     True,
    #     False,
    #     False,
    #     sig=False,
    #     use_mog=True,
    #     use_maf=False,
    #     num_mogs=np.concatenate([[1], np.arange(2, 22, 2)]),
    #     name_str="new_implementation",
    # )
    # run_maf_pipeline(
    #     True,
    #     False,
    #     False,
    #     sig=False,
    #     use_mog=True,
    #     use_maf=True,
    #     num_mogs=[10],
    #     num_train=np.logspace(4, 6, 10).astype(int),
    # )

    run_spline_pipeline(
        True,
        False,
        False,
        sig=True,
        num_splines=[32],
        name_str="",
        num_train=[10 ** 6],
    )
