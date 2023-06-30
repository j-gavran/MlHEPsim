import copy
import logging

from ml_hep_sim.pipeline.blocks import (
    ConfigBuilderBlock,
    DataGeneratorBlock,
    DatasetBuilderBlock,
    GeneratedDataVerifierBlock,
    ModelBuilderBlock,
    ModelLoaderBlock,
    ModelTrainerBlock,
)
from ml_hep_sim.pipeline.pipes import Pipeline
from ml_hep_sim.pipeline.prebuilt.base_pipeline import BasePipeline

CONFIGS = {
    "NICE": "nice_config",
    "RealNVP": "realnvp_config",
    "Glow": "glow_config",
    "MADEMOG": "made_mog_config",
    "MAFMADE": "maf_config",
    "MAFMADEMOG": "maf_config",
    "PolynomialSpline": "polynomial_splines_config",
    "RqSpline": "rq_splines_config",
}


class FlowPipeline(BasePipeline):
    def __init__(self, run_name, model_name, override=None, pipeline_path="ml_pipeline/", logger=None):
        super().__init__(run_name, override, pipeline_path)
        self.model_name = model_name
        self.pipeline = {"train_pipeline": None, "inference_pipeline": None}
        self.logger = logger

    def build_train_pipeline(self):
        flow_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_train_pipe", pipeline_path=self.pipeline_path, logger=self.logger
        )

        # build configuration from yaml and overriden config dict
        x1 = ConfigBuilderBlock(
            override_config=self.override,
            config_path="../conf",
            config_name=CONFIGS[self.model_name],
            model_name=self.model_name,
        )()
        # build model
        x2 = ModelBuilderBlock(model_type="flow")(x1)
        # build dataset
        x3 = DatasetBuilderBlock()(x1)
        # train model (save trained model path from mlruns folder)
        x4 = ModelTrainerBlock()(x2, x3)

        flow_train_pipeline.compose(x1, x2, x3, x4)

        self.pipeline["train_pipeline"] = flow_train_pipeline
        self.logger.info("Built flow training pipeline...")

        return self

    def build_inference_pipeline(self, N_gen, rescale_data=True, device="cpu"):
        if not self.fitted:
            self.logger.error("Fit first...")
            raise ValueError

        flow_generation_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_generation_pipe", pipeline_path=self.pipeline_path, logger=self.logger
        )

        flow_train_pipes = self.pipeline["train_pipeline"].pipes
        x1, x4 = copy.deepcopy(flow_train_pipes[0]), flow_train_pipes[3]

        # load trained model
        x5 = ModelLoaderBlock(device=device)(x1, x4)
        # generate data from model
        x6 = DataGeneratorBlock(N_gen, model_type="flow", chunks=10, device=device)(x5)
        # rescale and remove invalid values in data
        x7 = GeneratedDataVerifierBlock(save_data=True, rescale_data=rescale_data, device=device)(x5, x6)

        flow_generation_pipeline.compose(x5, x6, x7)

        self.pipeline["inference_pipeline"] = flow_generation_pipeline
        self.logger.info("Built flow inference pipeline...")

        return self

    def fit(self, force=False):
        flow_train_pipeline = self.pipeline["train_pipeline"]

        try:
            if not force:
                self.logger.info("Loadindg fitted flow...")
                flow_train_pipeline.load()
        except FileNotFoundError:
            self.logger.info("Did not find fitted flow. Fitting new one...")
            flow_train_pipeline.fit().save()

        if force:
            self.logger.warning("Force refitting flow...")
            flow_train_pipeline.fit().save()

        self.fitted = True
        return flow_train_pipeline

    def infer(self, return_results=True):
        if not self.fitted:
            self.fit()

        flow_infer_pipeline = self.pipeline["inference_pipeline"]

        self.logger.info("Using loaded flow for inference...")
        flow_infer_pipeline.fit().save()

        self.infered = True
        if return_results:
            return self.pipeline["inference_pipeline"].pipes[-1].generated_data
        else:
            return flow_infer_pipeline
