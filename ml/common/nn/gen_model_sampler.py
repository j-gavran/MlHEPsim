import glob
import logging
import os

import lightning as L
import numpy as np
import torch

from ml.common.utils.picklers import mkdir
from ml.common.utils.register_model import fetch_registered_module


class GenModelSampler:
    def __init__(self, model_names, save_dir=".", file_name="", versions=None, disable_cache=False):
        """Generative model sampler helper class. This is used to sample from generative models and cache the outputs.

        Note
        ----
        1. Model needs to implement a `sample` method that returns a numpy array of samples.
        2. Will always convert the samples to numpy arrays and save them as .npy files (even on disable_cache).
        3. If the cache is enabled, it will check if the cache matches the requested number of samples. If not, it will resample.
        4. If the cache is disabled, it will skip saving.
        5. To resample, remove the cache directory.

        Parameters
        ----------
        model_names : str or list
            Strings of model names (loaded from mlflow) to sample from.
        save_dir : str
            Directory to save the samples.
        file_name : str
            File name to save the samples.
        versions : list, int or None, optional
            List of versions to sample from. If None, it will use the latest version for all models, by default None.
        disable_cache : bool, optional
            If True, it will not use the cache, by default False.
        """
        L.seed_everything(workers=True)

        if type(model_names) is not list:
            model_names = [model_names]

        if versions is None:
            versions = -1

        if type(versions) is not None and type(versions) is not list:
            versions = [versions]

        if versions is None:
            logging.info("No versions provided, using latest version for all models")
            versions = [-1] * len(model_names)
        else:
            assert len(versions) == len(model_names), "Model names and versions should have the same length"

        self.model_names = model_names
        self.save_dir, self.file_name = save_dir, file_name
        self.versions = versions
        self.disable_cache = disable_cache

        self._check_cache_dir()

    def _check_cache_dir(self):
        """Check if the save directory exists."""
        if not os.path.exists(self.save_dir):
            logging.warning(f"Save directory {self.save_dir} does not exist. Creating it.")
            mkdir(self.save_dir)

        n_files = 0
        for file in glob.iglob(f"{self.save_dir}/{self.file_name}*.npy"):
            for model_name in self.model_names:
                if model_name in file:
                    file = file.split("/")[-1]
                    logging.debug(f"[yellow]Cache check:[/yellow] found cached file {file} that matches the file name.")
                    n_files += 1

        if n_files == 0:
            logging.warning(f"[yellow]Cache check:[/yellow] No existing cache found for {self.file_name}.")

    def _get_model(self, model_name, ver=-1):
        module = fetch_registered_module(model_name, ver, device="cuda")
        return module.model.eval()

    def _sample(self, N, resample, chunks=None, **kwargs):
        """Helper function to sample from the generative models."""
        samples, npy_files = dict(), dict()

        for model_name in self.model_names:
            ver = self.versions[self.model_names.index(model_name)]

            if ver == -1:
                ver_str = ""
            else:
                ver_str = f"_v{ver}"

            model = self._get_model(model_name, ver=ver)

            samples[model_name], npy_files[model_name] = [], []

            for i in range(resample):

                if resample == 1 or i == 0:
                    npy_file = f"{self.save_dir}/{self.file_name}_{model_name}{ver_str}.npy"
                else:
                    npy_file = f"{self.save_dir}/{self.file_name}_{model_name}{ver_str}_{i}.npy"

                npy_files[model_name].append(npy_file)

                M = None
                if os.path.exists(npy_file):
                    cached = np.load(npy_file)
                    M = len(cached)

                    if M >= N:
                        logging.info(f"[red]Using cached sample {i} for {model_name}.[/red]")
                        samples[model_name].append(cached[:N, :])
                        continue
                    else:
                        logging.info(f"Cache {npy_file} did not match N. Resampling.")

                if M is not None:
                    S = N - M
                    logging.info(f"Sampling extra {S} events to match {N} requested.")
                else:
                    S = N
                    logging.info(f"Sampling {N} events for {model_name}.")

                if chunks:
                    sampled = model.sample(S, chunks=chunks, **kwargs)
                else:
                    sampled = model.sample(S, **kwargs)

                if type(sampled) is torch.Tensor:
                    sampled = sampled.cpu().numpy()

                if M is not None:
                    samples[model_name].append(np.concatenate([cached, sampled], axis=0))
                else:
                    samples[model_name].append(sampled)

                if not self.disable_cache:
                    logging.info(f"Caching sample {i} for {model_name}.")
                    np.save(npy_file, samples[model_name][i])

        return samples, npy_files

    def sample(self, N, resample=1, chunks=None, return_npy_files=False, **kwargs):
        """Sample from the generative models.

        Parameters
        ----------
        N : inr
            Number of samples to generate.
        resample : int, optional
            Number of times to resample, by default 1.
        chunks : int, optional
            Number of chunks to sample at a time, by default None.
        return_npy_files : bool, optional
            If True, it will return the npy files, by default False.

        Returns
        -------
        dict or lists where key is model name and value or a list is numpy array with samples
            Dictionary of samples.
        """
        if type(N) is not int:
            N = int(N)

        with torch.no_grad():
            samples, npy_files = self._sample(N, resample, chunks=chunks, **kwargs)

        if return_npy_files:
            return samples, npy_files
        else:
            return samples
