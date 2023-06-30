from ml_hep_sim.pipeline.prebuilt.classifier_pipeline import ClassifierPipeline


def nn_classifier(override, test_dataset="higgs", debug=False):
    """
    override_example = {
        "datasets": {
            "data_name": "higgs",
            "data_params": {
                "subset_n": [10 ** 6, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": "Higgs_resnet_classifier", "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 3},
        "model_config": {
            "resnet": True,
            "hidden_layers": [256, 128, 64, 1],
        },
    }

    """
    if debug:
        run_name += "_debug"

    CP = ClassifierPipeline(run_name, override=override, pipeline_path="ml_pipeline/test/")

    CP.build_train_pipeline()
    CP.fit(force=True)

    CP.build_inference_pipeline(test_dataset=test_dataset)

    res = CP.infer(return_results=True)

    class_train_pipeline, class_infer_pipeline = CP.pipeline["train_pipeline"], CP.pipeline["inference_pipeline"]

    return res, [class_train_pipeline, class_infer_pipeline]


if __name__ == "__main__":
    override = {
        "datasets": {
            "data_name": "higgs",
            "data_params": {
                "subset_n": [10 ** 6, 10 ** 5, 10 ** 5],
                "rescale": "logit_normal",
                "to_gpu": True,
            },
        },
        "logger_config": {"run_name": "Higgs_resnet_classifier", "experiment_name": "TEST"},
        "trainer_config": {"gpus": 1, "max_epochs": 3},
        "model_config": {
            "resnet": True,
            "hidden_layers": [256, 128, 64, 1],
        },
    }

    run_name = "Higgs_resnet_classifier"

    nn_classifier(override, run_name, debug=True)
