import gc
import os
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from torch.distributions import Normal
from tqdm import tqdm

from ml_hep_sim.data_utils.dataset_utils import rescale_data
from ml_hep_sim.ml_utils import mlf_loader, mlf_trainer, netron_model
from ml_hep_sim.pipeline.available_models import flow_models, other_models, vae_models
from ml_hep_sim.pipeline.avaliable_datasets import DataModuleBuilder
from ml_hep_sim.pipeline.pipes import Block
from ml_hep_sim.stats.f_divergences import fDivergence
from ml_hep_sim.stats.one_dim_tests.py_two_sample import (
    chi2_twosample_test,
    ks_twosample_test,
)

# ----------------------------------------------------------------------------------------------------------------------
#                                                    STAGE 1
# ----------------------------------------------------------------------------------------------------------------------


class ConfigBuilderBlock(Block):
    def __init__(self, override_config=None, config_path=None, config_name=None, model_name=None):
        """Hydra configuration manager.

        Parameters
        ----------
        override_config : dict, optional
            This dictionary is used to override deafult values in conf/ yaml files, by default None.
        config_path : str, optional
            Path to hydra conf folder, by default None.
        config_name : str, optional
            Name of model's yaml config file, by default None.
        model_name : str, optional
            Name of the model from avaliable models, by default None.

        Note
        ----
        Can be run only with override_config if config_path and config_name are both None.

        """
        super().__init__()
        self.override_config = override_config
        self.config_path = config_path
        self.config_name = config_name
        self.model_name = model_name

        self.config = None

    def _hydra_initialization(self):
        """https://hydra.cc/docs/advanced/compose_api/"""
        with initialize(version_base=None, config_path=self.config_path, job_name=None):
            config = compose(config_name=self.config_name)
        return OmegaConf.to_container(config)

    def _deep_update(self, source, overrides):
        """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
        for key, value in overrides.items():
            if isinstance(value, Mapping) and value:
                returned = self._deep_update(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = overrides[key]
        return source

    def run(self):
        """Config parser."""
        if self.config_path and self.config_name:
            self.config = self._hydra_initialization()

            if self.override_config:
                self.config = self._deep_update(self.config, self.override_config)
        else:
            self.config = self.override_config

        if self.model_name == "none":
            self.logger.warning("Model name is set to 'none'!")
        elif self.model_name is None:
            self.model_name = self.config["logger_config"]["model_name"]
        else:
            self.config = self._deep_update(self.config, {"logger_config": {"model_name": self.model_name}})

        return self.config


class ModelBuilderBlock(Block):
    def __init__(self, model_type, model_name=None, config=None):
        """Build model from lightning model.

        Parameters
        ----------
        model_type : str
            Type of model. Can be flow or vae.
        model_name : ConfigBuilderBlock, optional
            Block obj, by default None.
        config : ConfigBuilderBlock, optional
            Block obj, by default None.
        """
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.config = config

        self.model_class = None
        self.model = None

    @staticmethod
    def _check_key(model_name, model_type):
        if model_type.lower() == "flow":
            if model_name not in flow_models:
                raise NameError
        elif model_type.lower() == "vae":
            if model_name not in vae_models:
                raise NameError
        elif model_type.lower() == "other":
            if model_name not in other_models:
                raise NameError
        else:
            raise NameError

    def _get_model(self):
        if self.model_type == "vae":
            return vae_models[self.model_name]
        elif self.model_type == "flow":
            return flow_models[self.model_name]
        else:
            return other_models[self.model_name]

    def run(self, **model_kwargs):
        """Build the model."""
        self._check_key(self.model_name, self.model_type)
        self.model_class = self._get_model()

        self.model = self.model_class(
            self.config["model_config"],
            input_dim=self.config["datasets"]["input_dim"],
            device="cuda" if self.config["datasets"]["data_params"]["to_gpu"] else "cpu",
            data_name=self.config["datasets"]["data_name"],
            **model_kwargs,
        )
        return self.model

    def to_netron(self):
        netron_model(
            self.model,
            torch.randn(1, self.config["datasets"]["input_dim"]),
        )

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["model"] = None
        return attributes


class DatasetBuilderBlock(Block):
    def __init__(self, config=None, dataset_name=None, data_param_dict=None, data_paths=None, data_dir=""):
        """Build dataset and return lightning data module.

        Parameters
        ----------
        config : ConfigBuilderBlock, optional
            Block obj, by default None.
        dataset_name : str, optional
            See DataModuleBuilder, by default None.
        data_param_dict : dict, optional
            See DataModuleBuilder, by default None.
        data_paths : list, optional
            See DataModuleBuilder, by default None.
        data_dir : str, optional
            Prefix for data directory location (defined in DataModuleBuilder), by default "".
        """
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.data_param_dict = data_param_dict
        self.data_paths = data_paths
        self.data_dir = data_dir

        self.data_module = None

    def run(self):
        """Get data module."""
        if self.config is not None:
            self.dataset_name = self.config["datasets"]["data_name"]
            self.data_param_dict = self.config["datasets"]["data_params"]

        data_module_builder = DataModuleBuilder(
            self.dataset_name,
            self.data_dir,
            self.data_param_dict,
            data_paths=self.data_paths,
        )
        self.data_module = data_module_builder.get_data_module()
        return self.data_module

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["data_module"] = None
        return attributes


class ModelTrainerBlock(Block):
    def __init__(self, config=None, model=None, data_module=None, trainer_dict=None, logger_dict=None):
        """Wrapper for mlf_trainer in ml_hep_sim/ml_utils.

        Parameters
        ----------
        config : ConfigBuilderBlock, optional
            Block obj, by default None.
        model : ModelBuilderBlock, optional
            Block obj, by default None.
        data_module : DatasetBuilderBlock, optional
            Block obj, by default None.
        trainer_dict : dict
            See mlf_trainer function, by default None.
        logger_dict: dict
            See mlf_trainer function, by default None.
        """
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.config = config

        self.trainer_dict, self.logger_dict = trainer_dict, logger_dict

        self.model_path = None

    def run(self):
        """Run model training."""
        if self.config:
            self.trainer_dict = self.config["trainer_config"]
            self.logger_dict = self.config["logger_config"]

        if hasattr(self.data_module, "scalers"):
            self.model.scalers = self.data_module.scalers

        self.model_path = mlf_trainer(
            self.model,
            self.data_module,
            trainer_dict=self.trainer_dict,
            **self.logger_dict,
        )

        return self.model_path

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["model"] = None
        attributes["data_module"] = None
        return attributes


# ----------------------------------------------------------------------------------------------------------------------
#                                                    STAGE 2
# ----------------------------------------------------------------------------------------------------------------------


class ModelLoaderBlock(Block):
    def __init__(self, config=None, model_path=None, model_name=None, device="cpu", read_metrics=True):
        """Wrapper for mlf_loader in ml_hep_sim/ml_utils.

        Parameters
        ----------
        config : ConfigBuilderBlock, optional
            Block obj, by default None.
        model_path : ModelTrainerBlock, optional
            Block obj, by default None.
        model_name : ModelBuilderBlock
            Block obj, by default None.
        device : str, optional
            Device to put trained model on, bt default cpu.
        read_metrics : bool, optional
            If True read mlflows metrics (validation loss etc.) from mlruns folder, by default True.
        """
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.read_metrics = read_metrics

        self.trained_model = None
        self.metrics = None

    def run(self):
        """Load the model."""
        if self.config:
            self.model_name = self.config["logger_config"]["model_name"]

        with torch.no_grad():
            self.trained_model = mlf_loader(self.model_path[1] + self.model_name, device=self.device).eval()

        if self.read_metrics:
            metrics_path = self.model_path[0] + "/metrics/"
            metrics_names = os.listdir(metrics_path)
            metrics = []
            for metric_name in metrics_names:
                r = pd.read_csv(metrics_path + metric_name, delimiter=r"\s+", names=["timestamp", metric_name, "step"])
                metrics.append(r)

            self.metrics = metrics

        return self.trained_model

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["trained_model"] = None
        return attributes


class DataGeneratorBlock(Block):
    def __init__(self, N, trained_model=None, model_type=None, device="cpu", chunks=1, save_data=False):
        """Data generation block for generative models.

        Parameters
        ----------
        N : int
            Number of generated examples.
        trained_model : ModelLoaderBlock, optional
            Block obj, by default None.
        model_type : ModelBuilderBlock, optional
            Block obj, by default None.
        device : str, optional
            Device to put trained model on, by default cpu.
        chunks : int, optional
            Number of generated partitioned data (reduces gpu vram usage), by default 1.
        save_data : bool, optional
            If True save generated data to pickle, by default False.
        """
        super().__init__()
        self.trained_model, self.model_type = trained_model, model_type
        self.N, self.chunks = N, chunks
        self.device = device
        self.save_data = save_data

        self.generated_data = None

    def run(self):
        """Data generation."""
        data = []
        n = self.N // self.chunks

        self.logger.info(
            f"Generating {self.N} examples in {self.chunks} chunks of {n} examples each using {self.trained_model.__class__.__name__}."
        )

        with torch.no_grad():
            if self.model_type == "flow":
                # replace devices in base distribution
                try:
                    self.trained_model.flow.base_distribution = Normal(
                        torch.zeros(len(self.trained_model.flow.base_distribution.loc)).to(self.device),
                        torch.ones(len(self.trained_model.flow.base_distribution.scale)).to(self.device),
                    )
                except AttributeError:
                    self.trained_model.base_distribution = Normal(
                        torch.zeros(len(self.trained_model.base_distribution.loc)).to(self.device),
                        torch.ones(len(self.trained_model.base_distribution.scale)).to(self.device),
                    )
                model = self.trained_model.flow.to(self.device).eval()
                for chunk in tqdm(range(self.chunks)):
                    data.append(model.sample(n))

            elif self.model_type == "vae":
                model = self.trained_model.to(self.device).eval()
                for chunk in tqdm(range(self.chunks)):
                    data.append(model.sample(n, device=self.device))

            else:
                raise NotImplemented

        self.generated_data = torch.vstack(data).cpu().numpy()

        torch.cuda.empty_cache()

        return self.generated_data

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["trained_model"] = None
        if not self.save_data:
            attributes["generated_data"] = None
        return attributes


class GeneratedDataVerifierBlock(Block):
    def __init__(self, trained_model=None, generated_data=None, device="cpu", rescale_data=True, save_data=False):
        """Rescale generated data and drop invalid values.

        Parameters
        ----------
        trained_model : ModelLoaderBlock, optional
            Block obj, by default None.
        generated_data : DataGeneratorBlock, optional
            Block obj, by default None.
        device : str, optional
            Device to put trained model on, by default cpu.
        rescale_data : bool, optional
            Rescale data back to original scaling (see ml_hep_sim/data_utils/dataset_utils), by default True.
        save_data : bool, optional
            If True save generated data to pickle, by default False.
        """
        super().__init__()
        self.trained_model = trained_model
        self.generated_data = generated_data
        self.rescale_data = rescale_data
        self.device = device
        self.save_data = save_data

        self.generated_scalers = None
        self.vdata = None

    def _check_invalid(self, ar, msg="running data check"):
        check = {
            "nan": np.isnan(ar),
            "pos-inf": np.isposinf(ar),
            "neg-inf": np.isneginf(ar),
            "pos-inf or neg-inf": np.isinf(ar),
            "pos-inf or neg-inf or nan": ~np.isfinite(ar),
        }

        self.logger.info(msg)
        for err, c in check.items():
            if c.any() == True:
                self.logger.info(
                    err + " ERROR -> found " + str(np.count_nonzero(c)) + " invalid values that will be removed"
                )
            else:
                self.logger.info(err + " OK")

    def run(self):
        """Data rescaling (transformation) and dropping invalid values."""
        self.generated_scalers = self.trained_model.scalers  # test on idx -1, gen on idx 0

        self._check_invalid(self.generated_data, msg="Generated data check...")

        self.vdata = self.generated_data[~np.isnan(self.generated_data).any(axis=1)]
        self.vdata = self.vdata[~np.isinf(self.vdata).any(axis=1)]

        if self.rescale_data:
            self.vdata = self.generated_scalers[0].inverse_transform(self.vdata)

            self._check_invalid(self.vdata, msg="Scaled data check...")

            self.vdata = self.vdata[~np.isnan(self.vdata).any(axis=1)]
            self.vdata = self.vdata[~np.isinf(self.vdata).any(axis=1)]

        self.generated_data = torch.from_numpy(self.vdata).to(self.device)

        return self.generated_data

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["trained_model"] = None
        attributes["vdata"] = None
        if not self.save_data:
            attributes["generated_data"] = None
        return attributes


class ClassifierRunnerBlock(Block):
    def __init__(
        self,
        reference_data=None,
        generated_data=None,
        trained_model=None,
        save_data=False,
        results=None,
        device="cpu",
        chunks=10,
    ):
        """Run classifier model on data.

        Parameters
        ----------
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        trained_model : ModelLoaderBlock, optional
            Block obj, by default None.
        save_data : bool, optional
            If True save classification results, by default False.
        device : str, optional
            Device to put trained model on, by default cpu.
        """
        super().__init__()
        self.reference_data, self.generated_data = reference_data, generated_data

        self.trained_model = trained_model
        self.device = device
        self.save_data = save_data
        self.chunks = chunks

        self.results = results

    def _get_classification(self, data, remove_nans=True):
        results = []
        s = len(data) // self.chunks

        with torch.no_grad():
            model = self.trained_model.network.to(self.device).eval()

            for i in tqdm(range(self.chunks)):
                chunk = data[i * s : (i + 1) * s, :]
                result = model(chunk.to(self.device))  # .squeeze().cpu().numpy()
                results.append(result)

            chunk = data[self.chunks * s :, :]
            if len(chunk) != 0:
                result = model(chunk.to(self.device))
                results.append(result)

        results = torch.vstack(results).cpu().numpy()

        if remove_nans:
            results = results[~np.isnan(results)]

        return results

    def run(self):
        """Get classification results."""
        if self.reference_data is not None:
            data = self.reference_data
        elif self.generated_data is not None:
            data = self.generated_data
        else:
            raise NotImplemented

        self.results = self._get_classification(data)

        torch.cuda.empty_cache()

        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["trained_model"] = None
        attributes["reference_data"] = None
        attributes["generated_data"] = None
        if not self.save_data:
            attributes["results"] = None
        return attributes


class VariableExtractBlock(Block):
    def __init__(
        self, idx, reference_data=None, generated_data=None, save_data=False, override_data=True, device="cpu"
    ):
        super().__init__()

        self.idx = idx
        self.reference_data, self.generated_data = reference_data, generated_data

        self.device = device
        self.save_data = save_data
        self.override_data = override_data

        self.results = None

    def _extract(self):
        if self.generated_data is not None:
            r = self._tensor_check(self.generated_data[:, self.idx])
            if self.override_data:
                self.generated_data = r
            return r
        elif self.reference_data is not None:
            r = self._tensor_check(self.reference_data[:, self.idx])
            if self.override_data:
                self.reference_data = r
            return r
        else:
            raise ValueError

    def run(self):
        self.results = self._extract()
        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["reference_data"] = None
        attributes["generated_data"] = None
        if not self.save_data:
            attributes["results"] = None
        return attributes


# ----------------------------------------------------------------------------------------------------------------------
#                                                    STAGE 3
# ----------------------------------------------------------------------------------------------------------------------


class ReferenceDataLoaderBlock(Block):
    def __init__(
        self,
        data_module=None,
        data_str=None,
        N=-1,
        drop_first_column=True,
        rescale_reference=None,
        device="cpu",
    ):
        """Get (test) data from DatasetBuilderBlock.

        Parameters
        ----------
        data_module : DatasetBuilderBlock, optional
            Block obj, by default None.
        data_str : str, optional
            For manual loading, by default None.
        N : int, optional
            Number of data points to use (all by default), by default -1.
        drop_first_column : bool, optional
            Drop labels, by default True.
        rescale_reference : str, optional
            Rescale function (see ml_hep_sim/data_utils/dataset_utils), by default None.
        device : str, optional
            Device to put data on, by default cpu.
        """
        super().__init__()
        self.data_module = data_module

        self.data_str = data_str
        self.N = N
        self.drop_first_column = drop_first_column
        self.rescale_reference = rescale_reference

        self.device = device

        self.reference_scalers = None
        self.reference_data = None
        self.reference_labels = None

    def run(self):
        """Get validation data (i.e. reference data) from DatasetBuilderBlock in stage 1."""
        if self.data_module:
            self.data_module.rescale = self.rescale_reference
            self.data_module.to_gpu = False if self.device == "cpu" else True
            self.data_module.prepare_data()
            self.data_module.setup()
            self.reference_data = self.data_module.test.X[: self.N, :]
            if not self.drop_first_column:
                self.reference_labels = self.data_module.test.y[: self.N]

            if self.rescale_reference is not None:
                self.reference_scalers = self.data_module.scalers

        # get validation data by hand
        elif self.data_str:
            if self.drop_first_column:
                self.reference_data = np.load(self.data_str)[:, 1:][: self.N, :].astype(np.float32)
            else:
                self.reference_data = np.load(self.data_str)[: self.N, :].astype(np.float32)

            if self.rescale_reference:
                self.reference_data, _ = rescale_data(self.reference_data, rescale_type=self.rescale_reference)

            self.reference_data = torch.from_numpy(self.reference_data).to(self.device)
        else:
            raise ValueError

        return self.reference_data

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["data_module"] = None
        attributes["reference_data"] = None
        attributes["reference_labels"] = None
        return attributes


class DistanceMetricRunnerBlock(Block):
    def __init__(self, reference_data=None, generated_data=None):
        """Wrapper for ml_hep_sim/stats/f_divergences.

        Parameters
        ----------
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        """
        super().__init__()
        self.reference_data = reference_data
        self.generated_data = generated_data

        self.results = dict()

    def run(self, **kwargs):
        """Calculate KL, Hellinger, chi2 and alpha divergences."""
        l_gen, l_ref = self.generated_data.shape[0], self.reference_data.shape[0]
        if l_gen > l_ref:
            self.generated_data = self.generated_data[:l_ref, :]
        elif l_gen < l_ref:
            self.reference_data = self.reference_data[:l_gen, :]

        f_div = fDivergence(self.reference_data, self.generated_data, **kwargs)

        self.results["kl"] = f_div.kl_divergence()
        self.results["hellinger"] = f_div.hellinger_distance()
        self.results["chi2"] = f_div.chi2_distance()

        alpha = np.linspace(0, 1, 11)[1:-1]
        self.results["alpha"] = [f_div.alpha_divergence(a) for a in alpha]
        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["generated_data"] = None
        attributes["reference_data"] = None
        return attributes


class PCARunnerBlock(Block):
    def __init__(self, reference_data=None, generated_data=None, n_components=1, save_data=False):
        """Perform PCA on reference or generated data.

        Parameters
        ----------
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        n_components : int, optional
            Number of PCA components, by default 1.
        save_data : bool, optional
            If True save PCA results, by default False.
        """
        super().__init__()
        self.reference_data, self.generated_data = reference_data, generated_data
        self.n_components = n_components
        self.save_data = save_data

        self.results = None

    def run(self, remove_nans=True):
        if self.reference_data is not None:
            data = self.reference_data
        elif self.generated_data is not None:
            data = self.generated_data
        else:
            raise NotImplemented

        data = self._tensor_check(data)

        if remove_nans:
            data = data[~np.isnan(data).any(axis=1)]

        self.results = PCA(n_components=self.n_components).fit_transform(data).flatten()

        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["generated_data"] = None
        attributes["reference_data"] = None
        if not self.save_data:
            attributes["results"] = None
        return attributes


class StatTestRunnerBlock(Block):
    def __init__(
        self,
        reference_data=None,
        generated_data=None,
        results=None,
        use_results=True,
        save_data=False,
        add_dim=True,
    ):
        """Run chi2 and KS tests.

        Parameters
        ----------
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        results : Block, optional
            Block obj with results (a list of len 2 of tensors or 1D arrays) attribute, by default None.
        use_results : bool, optional
            If use results or use generated and reference, by default True.
        save_data : bool, optional
            If True save stat test results, by default False.
        add_dim : bool, optional
            Add dimension to data. If True assume 1D data else assume 2D tensor
            (see ml_hep_sim/stats/one_dim_tests/py_two_sample), by default True.
        """
        super().__init__()
        self.reference_data = reference_data
        self.generated_data = generated_data
        self.use_results = use_results
        self.save_data = save_data
        self.add_dim = add_dim

        self.results = results

    def run(self, n_bins="auto", use_r=False, bin_range=None):
        """Run tests on data."""
        if self.use_results and self.results is not None:
            reference_results = self.results[0]
            generated_results = self.results[1]
        else:
            reference_results = self.reference_data
            generated_results = self.generated_data

        if self.add_dim:
            reference_results = reference_results[:, None]
            generated_results = generated_results[:, None]

        if use_r:
            raise DeprecationWarning("R is no longer supported.")
        else:
            chi2_score, _ = chi2_twosample_test(
                reference_results, generated_results, n_bins=n_bins, return_histograms=True, bin_range=bin_range
            )
            ks_score = ks_twosample_test(reference_results, generated_results)

        self.results = [chi2_score, ks_score]

        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["generated_data"] = None
        attributes["reference_data"] = None
        if not self.save_data:
            attributes["results"] = None
        return attributes


class CouplingModelTestingBlock(Block):
    def __init__(
        self,
        N,
        data_module=None,
        trained_model=None,
        config=None,
        device="cpu",
        loss_cutoff=None,
        mean=True,
        save_data=True,
    ):
        """Test trained coupling models. Calculates loss on testing/validation data.

        Parameters
        ----------
        N : int
            Number of batches to use in mean and std calculation.
        data_module : DatasetBuilderBlock, optional
            Block obj, by default None.
        trained_model : ModelLoaderBlock, optional
            Block obj, by default None.
        config : ConfigBuilderBlock, optional
            Block obj, by default None.
        device : str, optional
            Tensor device, by default cpu.
        loss_cutoff : float, optional
            If loss above cutoff discard as outlier, by default None.
        mean : bool, optional
            Calculate mean of loss if True, by default True.
        save_data : bool, optional
            If True save results, by default True.
        """
        super().__init__()
        self.N = N
        self.data_module = data_module
        self.trained_model = trained_model
        self.config = config
        self.device = device
        self.loss_cutoff = loss_cutoff
        self.mean = mean
        self.save_data = save_data

        self.results = None

    def _get_test_dl(self):
        self.data_module.rescale = self.config["datasets"]["data_params"]["rescale"]
        self.data_module.to_gpu = False if self.device == "cpu" else True
        self.data_module.prepare_data()
        self.data_module.setup()
        dl = iter(self.data_module.test_dataloader())
        return dl

    def run(self):
        dl = self._get_test_dl()

        losses = []
        for _ in range(self.N):
            try:
                data, _ = next(dl)
            except StopIteration:
                self.logger.critical("DataLoader iterated through the whole dataset! Decrease N.")
                raise StopIteration

            with torch.no_grad():
                z, log_jac = self.trained_model.flow.eval()(data)

                jac_loss = sum(log_jac)
                nll = self.trained_model.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
                if self.mean:
                    loss = -torch.mean(jac_loss + nll)
                else:
                    loss = -(jac_loss + nll)

                if self.loss_cutoff is not None and self.mean is True:
                    if loss > self.loss_cutoff:
                        continue

                losses.append(loss.cpu())

        losses = torch.hstack(losses)
        self.results = [losses.mean(), losses.std(), losses]
        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["trained_model"] = None
        attributes["data_module"] = None
        if not self.save_data:
            attributes["results"] = None
        return attributes


class MADEMOGModelTestingBlock(CouplingModelTestingBlock):
    def __init__(
        self,
        N,
        data_module=None,
        trained_model=None,
        config=None,
        device="cpu",
        loss_cutoff=None,
        mean=True,
        save_data=True,
    ):
        super().__init__(N, data_module, trained_model, config, device, loss_cutoff, mean, save_data)
        self.pop_idx = -1

    def run(self):
        flow = self.trained_model.flow.eval()
        dl = self._get_test_dl()

        losses = []
        for _ in range(self.N):
            try:
                data, _ = next(dl)
            except StopIteration:
                self.logger.critical("DataLoader iterated through the whole dataset! Decrease N.")
                raise StopIteration

            with torch.no_grad():
                flow.bijectors[self.pop_idx].naninf_mask = True
                _, log_jac = flow(data)
                naninf_mask = flow.bijectors[self.pop_idx].naninf_mask
                flow.bijectors[self.pop_idx].naninf_mask = False

                log_jac.pop(self.pop_idx)
                mog_nll = flow.bijectors[self.pop_idx].log_prob

                if len(log_jac) != 0:
                    sum_of_log_det_jacobian = sum(log_jac)[naninf_mask]

                    if self.mean:
                        loss = -torch.mean(sum_of_log_det_jacobian + mog_nll)
                    else:
                        loss = -(sum_of_log_det_jacobian + mog_nll)
                else:
                    if self.mean:
                        loss = -torch.mean(mog_nll)
                    else:
                        loss = -mog_nll

                if self.loss_cutoff is not None and self.mean is True:
                    if loss > self.loss_cutoff:
                        continue

                losses.append(loss.cpu())

        losses = torch.hstack(losses)
        self.results = [losses.mean(), losses.std(), losses]
        return self.results


class ScalingTestBlock(Block):
    def __init__(
        self,
        N_min,
        N_max,
        steps,
        reference_data=None,
        generated_data=None,
        results=None,
        save_data=False,
    ):
        """Check chi2 and KS scaling (test value vs N).

        Parameters
        ----------
        N_min : int
            Start.
        N_max : int
            Stop.
        steps : int
            Step.
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        results : Block, optional
            Block obj with results (a list of len 2 of tensors or 1D arrays) attribute, by default None.
        save_data : bool, optional
            If True save scaling results, by default False.
        """
        super().__init__()
        self.N_range = np.linspace(N_min, N_max, steps).astype(np.int64)
        self.save_data = save_data
        self.reference_data, self.generated_data = reference_data, generated_data

        self.use_dim_reduction = True
        self.results = results

    def run(self):
        if self.reference_data is not None and self.generated_data is not None:
            self.results = [self.reference_data, self.generated_data]
            self.use_dim_reduction = False

        results = []

        for N in self.N_range:
            data_1, data_2 = self.results[0][:N], self.results[1][:N]
            res = StatTestRunnerBlock(
                reference_data=data_1,
                generated_data=data_2,
                use_results=False,
                add_dim=True if self.use_dim_reduction else False,
            ).run()
            results.append(res)

        self.results = results
        return self.results

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["reference_data"] = None
        attributes["generated_data"] = None
        if not self.save_data:
            attributes["results"] = None
        return attributes


class RatioHighestValuesCutBlock(Block):
    def __init__(self, cut_ratio, N_take, generated_data=None, reference_data=None):
        """Sort column-wise (event-wise per each feature) and cut some ratio of the highest kinematic valued events.

        Parameters
        ----------
        cut_ratio : float
            Fraction of events to keep.
        N_take : int
            Number of events to consider.
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        """
        super().__init__()
        self.cut_ratio = cut_ratio
        self.N_take = N_take
        self.generated_data = generated_data
        self.reference_data = reference_data

    def run(self):
        if self.generated_data is not None:
            self.generated_data = np.sort(self._tensor_check(self.generated_data), axis=0)
            cut_idx = int(self.cut_ratio * self.N_take)
            self.generated_data = shuffle(self.generated_data[:cut_idx, :])

        if self.reference_data is not None:
            self.reference_data = np.sort(self._tensor_check(self.reference_data), axis=0)
            cut_idx = int(self.cut_ratio * self.N_take)
            self.reference_data = shuffle(self.reference_data[:cut_idx, :])

        return self.generated_data, self.reference_data

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["reference_data"] = None
        attributes["generated_data"] = None
        return attributes


class CutBlock(Block):
    def __init__(self, cut_value, results=None, save_data=False):
        """Do a cut on classifier or on a kinematic variable.

        Parameters
        ----------
        cut_value : float
            Loose/medium/tight working points depending on classifier or observable.
        results : ClassifierRunnerBlock or variableExtractBlock, optional
            Block obj, by default None.
        """
        super().__init__()
        self.cut_value = cut_value
        self.results = results
        self.save_data = save_data
        self.cut_idx = None

    def _cut(self):
        results = self._tensor_check(self.results[0])
        cut_idx = np.where(results >= self.cut_value)[0]
        cut_results = results[cut_idx]
        return cut_results, cut_idx

    def run(self):
        self.results, self.cut_idx = self._cut()
        return self.results, self.cut_idx

    def __getstate__(self):
        attributes = self.__dict__.copy()
        if not self.save_data:
            attributes["results"] = None
        return attributes


class CutByIndexBlock(Block):
    def __init__(self, cut_idx=None, generated_data=None, reference_data=None, save_data=False):
        """Cut feature matrix of events by some index.

        Parameters
        ----------
        cut_idx : CutBlock, optional
            Block obj, by default None.
        generated_data : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        reference_data : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        save_data : bool, optional
            Save cut events, by default False.
        """
        super().__init__()
        self.cut_idx = cut_idx
        self.generated_data = generated_data
        self.reference_data = reference_data
        self.save_data = save_data

    def _cut(self):
        if self.generated_data is not None:
            self.generated_data = self._tensor_check(self.generated_data[self.cut_idx])
        elif self.reference_data is not None:
            self.reference_data = self._tensor_check(self.reference_data[self.cut_idx])
        else:
            raise ValueError

        return self.generated_data, self.reference_data

    def run(self):
        return self._cut()

    def __getstate__(self):
        attributes = self.__dict__.copy()
        if not self.save_data:
            attributes["generated_data"] = None
            attributes["reference_data"] = None
        return attributes


class RedoRescaleDataBlock(Block):
    def __init__(
        self, generated_scalers=None, generated_data=None, reference_scalers=None, reference_data=None, scaler_idx=0
    ):
        """Reverts generated data or reference data rescaling as saved by appropriate blocks.

        Parameters
        ----------
        generated_scalers : GeneratedDataVerifierBlock, optional
            Block obj, by default None.
        generated_data : CutByIndexBlock, optional
            Block obj, by default None.
        reference_scalers : ReferenceDataLoaderBlock, optional
            Block obj, by default None.
        reference_data : CutByIndexBlock, optional
            Block obj, by default None.
        scaler_idx : int, optional
            Index of train/val/test in scalers list, by default 0.
        """
        super().__init__()
        self.generated_scalers = generated_scalers
        self.generated_data = generated_data

        self.reference_scalers = reference_scalers
        self.reference_data = reference_data

        self.scaler_idx = scaler_idx

    def _rescale(self):
        if self.generated_data is not None:
            self.generated_data = self.generated_scalers[self.scaler_idx].inverse_transform(self.generated_data)
        elif self.reference_data is not None:
            self.reference_data = self.reference_scalers[self.scaler_idx].inverse_transform(self.reference_data)

        return self.generated_data, self.reference_data

    def run(self):
        return self._rescale()

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["generated_data"] = None
        attributes["reference_data"] = None
        return attributes


class GCBlock(Block):
    def __init__(self, exclude=None):
        """Garbage collector block.

        Frees memory by setting attributes of prior blocks to None.

        Parameters
        ----------
        exclude : list, optional
            Do not garbage collect attributes of prior blocks in this list, by default None
        """
        super().__init__()
        if exclude is None:
            self.exclude = [None]
        else:
            self.exclude = exclude

    def run(self):
        for block in self.priors:
            _p_keys = []
            for k, v in block.__dict__.items():
                if k in self.exclude:
                    continue
                else:
                    if v is not None:
                        block.__dict__[k] = None
                        _p_keys.append(k)

            self.logger.debug(f"garbage collected attrs {_p_keys} of block {block}")
            gc.collect()
            torch.cuda.empty_cache()

        return self


if __name__ == "__main__":
    from ml_hep_sim.pipeline.pipes import Pipeline

    override_config = {
        "datasets": {
            "input_dim": 18,
            "data_name": "higgs",
            "data_params": {
                "batch_size": 1024,
                "rescale": "none",
                "to_gpu": True,
                "subset_n": [250000, 100000, 100000],
                "shuffle_data": True,
            },
        }
    }

    x1 = ConfigBuilderBlock(override_config=override_config, model_name="none")()
    x2 = DatasetBuilderBlock()(x1)

    data_pipe = Pipeline()
    data_pipe.compose(x1, x2)
    data_pipe.fit()
