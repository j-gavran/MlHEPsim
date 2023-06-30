import copy

from ml_hep_sim.pipeline.blocks import (
    ClassifierRunnerBlock,
    ConfigBuilderBlock,
    DatasetBuilderBlock,
    ModelBuilderBlock,
    ModelLoaderBlock,
    ModelTrainerBlock,
    ReferenceDataLoaderBlock,
)
from ml_hep_sim.pipeline.pipes import Pipeline
from ml_hep_sim.pipeline.prebuilt.base_pipeline import BasePipeline


class ClassifierPipeline(BasePipeline):
    def __init__(self, run_name, override=None, pipeline_path="ml_pipeline/", logger=None):
        super().__init__(run_name, override, pipeline_path)
        self.pipeline = {"train_pipeline": None, "inference_pipeline": None}
        self.logger = logger

    def build_train_pipeline(self):
        class_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_train_pipeline",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        x1 = ConfigBuilderBlock(
            override_config=self.override,
            config_path="../conf",
            config_name="classifier_config",
            model_name="BinaryClassifier",
        )()
        x2 = ModelBuilderBlock(model_type="other")(x1)
        x3 = DatasetBuilderBlock()(x1)
        x4 = ModelTrainerBlock()(x2, x3)

        class_train_pipeline.compose(x1, x2, x3, x4)

        self.pipeline["train_pipeline"] = class_train_pipeline
        self.logger.info("Built classification training pipeline...")

        return self

    def build_inference_pipeline(self, test_dataset):
        """Classifier inference.

        Parameters
        ----------
        test_dataset : str
            One from ["higgs_bkg", "higgs_sig", "higgs].

        """
        if not self.fitted:
            self.logger.error("Fit first...")
            raise ValueError

        class_infer_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_infer_pipeline",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        class_train_pipes = self.pipeline["train_pipeline"].pipes
        x1, x4 = copy.deepcopy(class_train_pipes[0]), class_train_pipes[3]

        x1.override_config["datasets"]["data_name"] = test_dataset
        x1.fit_block = True
        x5 = DatasetBuilderBlock()(x1)
        x6 = ReferenceDataLoaderBlock(rescale_reference=x1.config["datasets"]["data_params"]["rescale"])(x5)
        x7 = ModelLoaderBlock()(x1, x4)
        x8 = ClassifierRunnerBlock(save_data=True)(x6, x7)

        class_infer_pipeline.compose(x1, x5, x6, x7, x8)

        self.pipeline["inference_pipeline"] = class_infer_pipeline
        self.logger.info("Built classification inference pipeline...")

        return self

    def fit(self, force=False):
        class_train_pipeline = self.pipeline["train_pipeline"]

        try:
            if not force:
                self.logger.info("Loadindg fitted classifier...")
                class_train_pipeline.load()
        except FileNotFoundError:
            self.logger.info("Did not find fitted classifier. Fitting new one...")
            class_train_pipeline.fit().save()

        if force:
            self.logger.warning("Force refitting classifier...")
            class_train_pipeline.fit().save()

        self.fitted = True
        return class_train_pipeline

    def infer(self, return_results=False):
        if not self.fitted:
            self.fit()

        class_infer_pipeline = self.pipeline["inference_pipeline"]

        self.logger.info("Using loaded classifer for inference...")
        class_infer_pipeline.fit().save()

        self.infered = True
        if return_results:
            return self.pipeline["inference_pipeline"].pipes[-1].results
        else:
            return class_infer_pipeline
