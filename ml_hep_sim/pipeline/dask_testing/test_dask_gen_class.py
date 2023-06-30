import copy
import unittest

import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from ml_hep_sim.pipeline.distributed_blocks import (
    AggregateResultsBlock,
    ClassifierRunnerBlock,
    ConfigBuilderBlock,
    DataGeneratorBlock,
    DatasetBuilderBlock,
    GeneratedDataVerifierBlock,
    ModelBuilderBlock,
    ModelLoaderBlock,
    ModelTrainerBlock,
    ReferenceDataLoaderBlock,
)
from ml_hep_sim.pipeline.pipeline_loggers import setup_logger
from ml_hep_sim.pipeline.distributed_pipes import Pipeline


class TestDaskFlow(unittest.TestCase):
    run_name = "Higgs_MADEMOG_sig_test"
    model_name = "MADEMOG"
    pipeline_path = "ml_pipeline/test/"

    logger = setup_logger()

    def get_sig_train_pipeline(self):
        override = {
            "datasets": {
                "data_name": "higgs_sig",
                "data_params": {
                    "subset_n": [10**4, 10**4, 10**4],
                    "rescale": "logit_normal",
                    "to_gpu": True,
                },
            },
            "logger_config": {"run_name": "Higgs_MADEMOG_test", "experiment_name": "TEST"},
            "trainer_config": {"gpus": 1, "max_epochs": 1},
            "model_config": {"num_flows": 4},
        }

        flow_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_train_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        return flow_train_pipeline, override

    def get_sig_infer_pipeline(self):
        flow_generation_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_generation_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )
        return flow_generation_pipeline

    def test_train_sig_pipeline(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        flow_train_pipeline, override = self.get_sig_train_pipeline()

        # build configuration from yaml and overriden config dict
        x1 = ConfigBuilderBlock(
            override_config=override,
            config_path="../conf",
            config_name="made_mog_config",
            model_name=self.model_name,
        )
        # build model
        x2 = ModelBuilderBlock(model_type="flow").take(x1)
        # build dataset
        x3 = DatasetBuilderBlock().take(x1)
        # train model (save trained model path from mlruns folder)
        x4 = ModelTrainerBlock().take(x2, x3)

        flow_train_pipeline.compose(x1, x2, x3, x4)

        flow_train_pipeline.fit()
        flow_train_pipeline.save()

        client.close()

    def test_load_train_sig_pipeline(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        flow_train_pipeline, _ = self.get_sig_train_pipeline()

        flow_train_pipeline.load()

        client.close()

    def test_infer_sig_pipeline(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        flow_train_pipeline, _ = self.get_sig_train_pipeline()
        flow_train_pipeline.load()

        flow_generation_pipeline = self.get_sig_infer_pipeline()

        N_gen = 10**6

        x1, x4 = copy.deepcopy(flow_train_pipeline.pipes[0]), flow_train_pipeline.pipes[3]

        # load trained model
        x5 = ModelLoaderBlock(device="cuda").take(x1, x4)
        # generate data from model
        x6 = DataGeneratorBlock(N_gen, model_type="flow", chunks=10, device="cuda").take(x5)
        # rescale and remove invalid values in data
        x7 = GeneratedDataVerifierBlock(save_data=True, rescale_data=True, device="cuda").take(x5, x6)

        flow_generation_pipeline.compose(x1, x4, x5, x6, x7)

        flow_generation_pipeline.fit()

        self.assertEqual(flow_generation_pipeline.computed.generated_data.shape[0], N_gen)

        client.close()


class TestDaskClassifier(unittest.TestCase):
    run_name = "Higgs_class_test"
    pipeline_path = "ml_pipeline/test/"

    logger = setup_logger()

    def get_class_train_pipeline(self):
        override = {
            "datasets": {
                "data_name": "higgs",
                "data_params": {
                    "subset_n": [10**6, 10**5, 10**5],
                    "rescale": "logit_normal",
                    "to_gpu": True,
                },
            },
            "logger_config": {"run_name": self.run_name, "experiment_name": "TEST"},
            "trainer_config": {"gpus": 1, "max_epochs": 1},
            "model_config": {
                "resnet": False,
                "bayes_net": False,
                "hidden_layers": [128, 128, 1],
            },
        }

        class_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_train_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        return class_train_pipeline, override

    def get_class_infer_pipeline(self):
        class_infer_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_infer_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )
        return class_infer_pipeline

    def test_train_classifier(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        class_train_pipeline, override = self.get_class_train_pipeline()

        # build configuration from yaml and overriden config dict
        x1 = ConfigBuilderBlock(
            override_config=override,
            config_path="../conf",
            config_name="classifier_config",
            model_name="BinaryClassifier",
        )
        # build model
        x2 = ModelBuilderBlock(model_type="other").take(x1)
        # build dataset
        x3 = DatasetBuilderBlock().take(x1)
        # train model
        x4 = ModelTrainerBlock().take(x2, x3)

        class_train_pipeline.compose(x1, x2, x3, x4)

        class_train_pipeline.fit()
        class_train_pipeline.save()

        client.close()

    def test_load_classifier(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        class_train_pipeline, _ = self.get_class_train_pipeline()

        class_train_pipeline.load()

        client.close()

    def test_infer_classifier(self):
        cluster = LocalCUDACluster(threads_per_worker=2, dashboard_address=":8787")
        client = Client(cluster)

        class_train_pipeline, _ = self.get_class_train_pipeline()
        class_train_pipeline.load()

        class_infer_pipeline = self.get_class_infer_pipeline()

        x1, x4 = copy.deepcopy(class_train_pipeline.pipes[0]), copy.deepcopy(class_train_pipeline.pipes[3])

        x5 = DatasetBuilderBlock().take(x1)
        x6 = ReferenceDataLoaderBlock(rescale_reference=x1.config["datasets"]["data_params"]["rescale"]).take(x5)
        x7 = ModelLoaderBlock().take(x1, x4)
        x8 = ClassifierRunnerBlock(save_data=True).take(x6, x7)

        class_infer_pipeline.compose(x1, x4, x5, x6, x7, x8)
        class_infer_pipeline.fit()

        self.assertEqual(len(class_infer_pipeline.computed.results.shape), 1)

        client.close()


class TestGeneratorClassifierPipeline(unittest.TestCase):
    run_name = "Higgs_MADEMOG_test"
    model_name = "MADEMOG"
    pipeline_path = "ml_pipeline/test/"
    config_name = "made_mog_config"

    N_gen = 10**6
    use_classifier = True

    logger = setup_logger(dummy_logger=True)

    def get_sig_train_pipeline(self):
        sig_override = {
            "datasets": {
                "data_name": "higgs_sig",
                "data_params": {
                    "subset_n": [10**6, 10**6, 10**5],
                    "rescale": "logit_normal",
                    "to_gpu": True,
                },
            },
            "logger_config": {"run_name": f"Higgs_{self.model_name}_test", "experiment_name": "TEST"},
            "trainer_config": {"gpus": 1, "max_epochs": 101},
            "model_config": {
                "num_hidden_layers_mog_net": 100,
                "hidden_layer_dim": 1024,
                "lr_scheduler_dct": {
                    "scheduler": "ReduceLROnPlateau",
                    "interval": "epoch",
                    "params": {"mode": "min", "factor": 0.5, "patience": 10},
                },
            },
        }

        flow_sig_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_sig_train_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        return flow_sig_train_pipeline, sig_override

    def get_bkg_train_pipeline(self):
        bkg_override = {
            "datasets": {
                "data_name": "higgs_bkg",
                "data_params": {
                    "subset_n": [10**6, 10**6, 10**5],
                    "rescale": "logit_normal",
                    "to_gpu": True,
                },
            },
            "logger_config": {"run_name": f"Higgs_{self.model_name}_test", "experiment_name": "TEST"},
            "trainer_config": {"gpus": 1, "max_epochs": 101},
            "model_config": {
                "hidden_layer_dim": 1024,
                "num_hidden_layers_mog_net": 100,
                "lr_scheduler_dct": {
                    "scheduler": "ReduceLROnPlateau",
                    "interval": "epoch",
                    "params": {"mode": "min", "factor": 0.5, "patience": 10},
                },
            },
        }

        flow_bkg_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_bkg_train_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        return flow_bkg_train_pipeline, bkg_override

    def get_class_train_pipeline(self):
        class_override = {
            "datasets": {
                "data_name": "higgs",
                "data_params": {
                    "subset_n": [10**6, 10**5, 10**5],
                    "rescale": "logit_normal",
                    "to_gpu": True,
                },
            },
            "logger_config": {"run_name": "Higgs_class", "experiment_name": "TEST"},
            "trainer_config": {"gpus": 1, "max_epochs": 21},
            "model_config": {
                "resnet": True,
                "bayes_net": False,
                "hidden_layers": [512, 512, 512, 512, 512, 1],
            },
        }

        class_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_class_train_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        return class_train_pipeline, class_override

    def get_sig_infer_pipeline(self):
        flow_sig_generation_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_sig_generation_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )
        return flow_sig_generation_pipeline

    def get_bkg_infer_pipeline(self):
        flow_bkg_generation_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_bkg_generation_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )
        return flow_bkg_generation_pipeline

    def get_class_infer_pipeline(self):
        class_infer_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_class_infer_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )
        return class_infer_pipeline

    def test_gen_class_pipeline(self):
        cluster = LocalCUDACluster(threads_per_worker=4, dashboard_address=":8787")
        client = Client(cluster)

        # sig
        flow_sig_train_pipeline, sig_override = self.get_sig_train_pipeline()

        x11 = ConfigBuilderBlock(
            override_config=sig_override,
            config_path="../conf",
            config_name=self.config_name,
            model_name=self.model_name,
        ).set_name("ml sig config")
        x21 = ModelBuilderBlock(model_type="flow").take(x11).set_name("build sig model")
        x31 = DatasetBuilderBlock().take(x11).set_name("sig dataset")
        x41 = ModelTrainerBlock().take(x21, x31).set_name("sig trainer")

        flow_sig_train_pipeline.compose(x11, x21, x31, x41)

        # bkg
        flow_bkg_train_pipeline, bkg_override = self.get_bkg_train_pipeline()

        x12 = ConfigBuilderBlock(
            override_config=bkg_override,
            config_path="../conf",
            config_name=self.config_name,
            model_name=self.model_name,
        ).set_name("ml bkg config")
        x22 = ModelBuilderBlock(model_type="flow").take(x12).set_name("build bkg model")
        x32 = DatasetBuilderBlock().take(x12).set_name("bkg dataset")
        x42 = ModelTrainerBlock().take(x22, x32).set_name("bkg trainer")

        flow_bkg_train_pipeline.compose(x12, x22, x32, x42)

        # sig + bkg aggregate block
        b_agg = AggregateResultsBlock().take(x41, x42).set_name("sig bkg agg")

        # combined pipeline (train sig and bkg in parallel)
        flow_sig_bkg_train_pipeline = Pipeline(
            pipeline_name=f"{self.run_name}_sig_bkg_train_pipe",
            pipeline_path=self.pipeline_path,
            logger=self.logger,
        )

        try:
            flow_sig_bkg_train_pipeline.load()
        except Exception as e:
            flow_sig_bkg_train_pipeline.compose(flow_sig_train_pipeline, flow_bkg_train_pipeline, b_agg)
            flow_sig_bkg_train_pipeline.fit(visualize=True)
            flow_sig_bkg_train_pipeline.save()

        # infer (generate) sig
        flow_sig_generation_pipeline = self.get_sig_infer_pipeline()

        x13, x43 = copy.deepcopy(flow_sig_bkg_train_pipeline.pipes[0]), flow_sig_bkg_train_pipeline.pipes[3]
        x53 = ModelLoaderBlock(device="cuda").take(x13, x43).set_name("sig model loader")
        x63 = (
            DataGeneratorBlock(self.N_gen, model_type="flow", chunks=10, device="cuda")
            .take(x53)
            .set_name("sig data gen")
        )
        x73 = (
            GeneratedDataVerifierBlock(save_data=True, rescale_data=True, device="cuda")
            .take(x53, x63)
            .set_name("sig data verifier")
        )

        flow_sig_generation_pipeline.compose(x13, x43, x53, x63, x73)

        # infer (generate) bkg
        flow_bkg_generation_pipeline = self.get_bkg_infer_pipeline()

        x14, x44 = copy.deepcopy(flow_sig_bkg_train_pipeline.pipes[4]), flow_sig_bkg_train_pipeline.pipes[7]
        x54 = ModelLoaderBlock(device="cuda").take(x14, x44).set_name("bkg model loader")
        x64 = (
            DataGeneratorBlock(self.N_gen, model_type="flow", chunks=10, device="cuda")
            .take(x54)
            .set_name("bkg data gen")
        )
        x74 = (
            GeneratedDataVerifierBlock(save_data=True, rescale_data=True, device="cuda")
            .take(x54, x64)
            .set_name("bkg data verifier")
        )

        flow_bkg_generation_pipeline.compose(x14, x44, x54, x64, x74)

        # reference sig mc data
        mc_sig_config = copy.deepcopy(flow_sig_bkg_train_pipeline.pipes[0])  # same config as before
        mc_sig_config.config["datasets"]["data_params"]["subset_n"] = [0, 0, self.N_gen]
        mc_sig_config.config["datasets"]["data_params"]["rescale"] = "none"

        b_mc_sig_dataset = DatasetBuilderBlock().take(mc_sig_config).set_name("mc sig dataset")
        b_mc_sig_data = (
            ReferenceDataLoaderBlock(rescale_reference="logit_normal" if self.use_classifier else None, device="cpu")
            .take(b_mc_sig_dataset)
            .set_name("mc sig data")
        )

        # reference bkg mc data
        mc_bkg_config = copy.deepcopy(flow_sig_bkg_train_pipeline.pipes[4])
        mc_bkg_config.config["datasets"]["data_params"]["subset_n"] = [0, 0, self.N_gen]
        mc_bkg_config.config["datasets"]["data_params"]["rescale"] = "none"

        b_mc_bkg_dataset = DatasetBuilderBlock().take(mc_bkg_config).set_name("mc bkg dataset")
        b_mc_bkg_data = (
            ReferenceDataLoaderBlock(rescale_reference="logit_normal" if self.use_classifier else None, device="cpu")
            .take(b_mc_bkg_dataset)
            .set_name("mc bkg data")
        )

        # classifier
        class_train_pipeline, class_override = self.get_class_train_pipeline()

        x15 = ConfigBuilderBlock(
            override_config=class_override,
            config_path="../conf",
            config_name="classifier_config",
            model_name="BinaryClassifier",
        ).set_name("class config")
        x25 = ModelBuilderBlock(model_type="other").take(x15).set_name("build class model")
        x35 = DatasetBuilderBlock().take(x15).set_name("class dataset")
        x45 = ModelTrainerBlock().take(x25, x35).set_name("class trainer")

        try:
            class_train_pipeline.load()
        except Exception as e:
            class_train_pipeline.compose(x15, x25, x35, x45)
            class_train_pipeline.fit(visualize=True)
            class_train_pipeline.save()

        # classifier infer
        b_flow_sig_generated = flow_sig_generation_pipeline.pipes[-1]
        b_flow_bkg_generated = flow_bkg_generation_pipeline.pipes[-1]

        class_config = copy.deepcopy(class_train_pipeline.pipes[0])  # classifier config block
        class_model = class_train_pipeline.pipes[3]  # classifier model trainer block

        b_classifier_model = (
            ModelLoaderBlock(device="cuda").take(class_config, class_model).set_name("class model loader")
        )

        b_sig_gen_class = (
            ClassifierRunnerBlock(save_data=True, device="cuda")
            .take(b_flow_sig_generated, b_classifier_model)
            .set_name("sig gen class")
        )  # sig gen
        b_bkg_gen_class = (
            ClassifierRunnerBlock(save_data=True, device="cuda")
            .take(b_flow_bkg_generated, b_classifier_model)
            .set_name("bkg gen class")
        )  # bkg gen

        b_sig_mc_class = (
            ClassifierRunnerBlock(save_data=True, device="cuda")
            .take(b_mc_sig_data, b_classifier_model)
            .set_name("sig mc class")
        )  # MC sig
        b_bkg_mc_class = (
            ClassifierRunnerBlock(save_data=True, device="cuda")
            .take(b_mc_bkg_data, b_classifier_model)
            .set_name("bkg mc class")
        )  # MC bkg

        b_agg_class = (
            AggregateResultsBlock()
            .take(b_sig_gen_class, b_bkg_gen_class, b_sig_mc_class, b_bkg_mc_class)
            .set_name("class agg")
        )

        # fit
        pipe = Pipeline(pipeline_name="gen_class_pipeline", pipeline_path="ml_pipeline/test/", logger=self.logger)
        pipe.compose(
            mc_sig_config,
            b_mc_sig_dataset,
            mc_bkg_config,
            b_mc_sig_data,
            b_mc_bkg_dataset,
            b_mc_bkg_data,
            flow_sig_generation_pipeline,
            flow_bkg_generation_pipeline,
            class_config,
            class_model,
            b_classifier_model,
            b_sig_gen_class,
            b_bkg_gen_class,
            b_sig_mc_class,
            b_bkg_mc_class,
            b_agg_class,
        )
        pipe.fit(visualize=True)

        class_results = pipe.computed.results
        sig_gen_res, bkg_gen_res, sig_mc_res, bkg_mc_res = class_results

        r = np.unique(sig_gen_res > 0.5, return_counts=True)[1]
        sig_gen_res_acc = r[1] / np.sum(r)

        r = np.unique(bkg_gen_res < 0.5, return_counts=True)[1]
        bkg_gen_res_acc = r[1] / np.sum(r)

        r = np.unique(sig_mc_res > 0.5, return_counts=True)[1]
        sig_mc_res_acc = r[1] / np.sum(r)

        r = np.unique(bkg_mc_res < 0.5, return_counts=True)[1]
        bkg_mc_res_acc = r[1] / np.sum(r)

        self.assertGreater(sig_gen_res_acc, 0.4)
        self.assertGreater(bkg_gen_res_acc, 0.4)
        self.assertGreater(sig_mc_res_acc, 0.4)
        self.assertGreater(bkg_mc_res_acc, 0.4)

        client.close()


if __name__ == "__main__":
    unittest.main()
