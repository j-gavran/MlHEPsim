import logging
import multiprocessing
from typing import Type

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer


class UnsupervisedDataset(Dataset):
    def __init__(self, data: np.ndarray, selection: pd.DataFrame):
        super().__init__()
        self.X = torch.from_numpy(data)
        self.selection = selection

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.X[idx]


class SupervisedDataset(Dataset):
    def __init__(self, data: np.ndarray, selection: pd.DataFrame):
        super().__init__()
        self.selection = selection

        features = self.selection[self.selection["type"] != "label"]
        labels = self.selection[self.selection["type"] == "label"]

        self.X = torch.from_numpy(data[:, features.index])
        self.y = torch.from_numpy(data[:, labels.index])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        processor: ProcessorChainer | Preprocessor,
        dataset_class: Type[object],
        train_split: float = 0.7,
        val_split: float = 0.5,
        **dataloader_kwargs,
    ):
        """Base class for data modules.

        Parameters
        ----------
        processor : object
            Custom class that processes the data and returns data, selection and scalers (e.g. Preprocessor).
        dataset : object
            Custom troch.Dataset class that takes in data and selection.
        train_split : float, optional
            Train split, by default 0.7.
        val_split : float, optional
            Validation split, by default 0.5. Train is split 0.7 and 0.3, test and val are split 0.5 and 0.5 from this 0.3.
        dataloader_kwargs : dict, optional
            Kwargs for torch.utils.data.DataLoader, by default {}.
        """
        super().__init__()
        self.processor = processor
        self.dataset = dataset_class
        self.train_split, self.val_split = train_split, val_split
        self.dataloader_kwargs = dataloader_kwargs

        if self.dataloader_kwargs.get("num_workers", None) == -1:
            self.dataloader_kwargs["num_workers"] = multiprocessing.cpu_count()

        self.train_idx, self.val_idx, self.test_idx = None, None, None
        self.selection, self.scalers = None, None

    def _get_splits(self, n: int | None = None) -> tuple[list, list, list]:
        idx = torch.arange(n)

        if self.train_split == 1.0:
            logging.info("Using all data for training! Using 0.1 and 0.1 for val and test for debug!")
            self.train_idx, self.val_idx, self.test_idx = idx, idx[: int(len(idx) * 0.1)], idx[: int(len(idx) * 0.1)]
            return self.train_idx, self.val_idx, self.test_idx

        if self.train_idx is None or self.val_idx is None or self.test_idx is None:
            remaining, train_idx = train_test_split(idx, test_size=self.train_split)
            test_idx, val_idx = train_test_split(idx[remaining], test_size=self.val_split)
            logging.info(
                f"Created train, val and test splits with sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}"
            )
            self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

        return self.train_idx, self.val_idx, self.test_idx

    def prepare_data(self):
        # this is done in the processor class
        pass

    def setup(self, stage: str | None = None) -> None:
        data, self.selection, self.scalers = self.processor()
        data = np.float32(data)

        self._get_splits(len(data))

        if stage == "fit" or stage is None:
            self.train = self.dataset(data[self.train_idx], self.selection)
            self.val = self.dataset(data[self.val_idx], self.selection)
        if stage == "test":
            self.test = self.dataset(data[self.test_idx], self.selection)

    def teardown(self, stage=None):
        if stage == "fit":
            self.train = None
            self.val = None
        if stage == "test":
            self.test = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, shuffle=False, **self.dataloader_kwargs)
