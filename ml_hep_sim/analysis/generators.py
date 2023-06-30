""" Functions to train and generate samples from a flow model examples."""

from ml_hep_sim.pipeline.prebuilt.flow_pipeline import FlowPipeline


def glow_generator(override, N_gen=10 ** 5, use_glow=True, debug=False, force=False):
    """
    override_example = {
        "datasets": {
            "data_name": "higgs_sig" if sig else "higgs_bkg",
            "data_params": {
                "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": run_name, "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 5},
        "model_config": {"num_flows": 10},
    }
    """
    run_name = "Higgs_Glow" if use_glow else "Higgs_RealNVP"

    if debug:
        run_name += "_debug"

    FP = FlowPipeline(
        run_name,
        model_name="Glow" if use_glow else "RealNVP",
        override=override,
        pipeline_path=f"ml_pipeline/{run_name}/",
    )

    FP.build_train_pipeline()
    FP.fit(force=force)

    FP.build_inference_pipeline(N_gen, device="cuda")

    res = FP.infer(return_results=True)

    flow_train_pipeline, flow_infer_pipeline = FP.pipeline["train_pipeline"], FP.pipeline["inference_pipeline"]

    return res, [flow_train_pipeline, flow_infer_pipeline]


def made_generator(override, model_name, N_gen=10 ** 5, debug=False, force=False):
    """
    override_example = {
        "datasets": {
            "data_name": "higgs_sig" if sig else "higgs_bkg",
            "data_params": {
                "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": run_name, "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 3},
        "model_config": {"num_flows": 10, "use_mog": True if model_name in ["MADEMOG", "MAFMADEMOG"] else False},
    }
    """
    if model_name not in ["MADEMOG", "MAFMADE", "MAFMADEMOG"]:
        raise NameError

    run_name = f"Higgs_{model_name}"

    if debug:
        run_name += "_debug"

    FP = FlowPipeline(
        run_name,
        model_name=model_name,
        override=override,
        pipeline_path=f"ml_pipeline/{run_name}/",
    )

    FP.build_train_pipeline()
    FP.fit(force=force)

    FP.build_inference_pipeline(N_gen, device="cuda")

    res = FP.infer(return_results=True)

    flow_train_pipeline, flow_infer_pipeline = FP.pipeline["train_pipeline"], FP.pipeline["inference_pipeline"]

    return res, [flow_train_pipeline, flow_infer_pipeline]


def spline_generator(override, model_name, N_gen=10 ** 5, debug=False, force=False):
    """
    override_example = {
        "datasets": {
            "data_name": "higgs_bkg",
            "data_params": {
                "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
            },
        },
        "logger_config": {"run_name": "Higgs_PolynomialSpline", "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 2},
        "model_config": {"num_flows": 10, "ar": True, "resnet": True, "bins": 8},
    }
    """
    run_name = f"Higgs_{model_name}"

    if debug:
        run_name += "_debug"

    FP = FlowPipeline(
        run_name,
        model_name=model_name,
        override=override,
        pipeline_path=f"ml_pipeline/{run_name}/",
    )

    FP.build_train_pipeline()
    FP.fit(force=force)

    FP.build_inference_pipeline(N_gen, device="cuda")

    res = FP.infer(return_results=True)

    flow_train_pipeline, flow_infer_pipeline = FP.pipeline["train_pipeline"], FP.pipeline["inference_pipeline"]

    return res, [flow_train_pipeline, flow_infer_pipeline]


if __name__ == "__main__":
    # quick tests

    override = {
        "datasets": {
            "data_name": "higgs_bkg",
            "data_params": {
                "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": "Higgs_glow", "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 2},
        "model_config": {"num_flows": 10},
    }

    glow_generator(override, debug=True)  # Glow

    override["logger_config"]["run_name"] = "Higgs_RealNVP"
    glow_generator(override, use_glow=False, debug=True)  # RealNVP

    # ------------------------------------------------------------------------------------------------------------------

    override = {
        "datasets": {
            "data_name": "higgs_bkg",
            "data_params": {
                "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": "Higgs_MADEMOG", "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 2},
        "model_config": {"num_flows": 10, "use_mog": True},
    }

    made_generator(override, model_name="MADEMOG", debug=True)  # MADEMOG

    override["logger_config"]["run_name"] = "Higgs_MAFMADEMOG"
    made_generator(override, model_name="MAFMADEMOG", debug=True)  # MAFMADEMOG

    override["logger_config"]["run_name"] = "Higgs_MAFMADE"
    del override["model_config"]["use_mog"]
    made_generator(override, model_name="MAFMADE", debug=True)  # MAFMADE

    # ------------------------------------------------------------------------------------------------------------------

    override = {
        "datasets": {
            "data_name": "higgs_bkg",
            "data_params": {
                "subset_n": [10 ** 5, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
            },
        },
        "logger_config": {"run_name": "Higgs_PolynomialSpline", "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 2},
        "model_config": {"num_flows": 10, "ar": True, "resnet": True, "bins": 8},
    }
    spline_generator(override, "PolynomialSpline", debug=True)  # PolynomialSpline

    override["logger_config"]["run_name"] = "Higgs_RqSpline"
    spline_generator(override, "RqSpline", debug=True)  # RqSpline
