import copy
import logging
import os

import numpy as np
from sklearn.utils import shuffle


class GeneratedProcessor:
    def __init__(self, data_dir, base_file_name, shuffle=True):
        """Fake processor for generated data from ML models. Generated data is saved in .npy file."""
        self.data_dir = data_dir
        self.base_file_name = base_file_name
        self.npy_file = os.path.join(self.data_dir, self.base_file_name + ".npy")

        self.shuffle = shuffle

    def get_dataset(self):
        dataset = np.load(self.npy_file).astype(np.float32)
        dataset = shuffle(dataset)
        return dataset

    def __call__(self, *args, **kwargs):
        dataset = self.get_dataset()
        return dataset, None, None


class TwoSampleBuilder:
    def __init__(
        self,
        processor_X,
        processor_Y=None,
        hold_out_ratio=0.2,
        add_label_X=False,
        add_label_Y=False,
        shuffle_random_state=0,
    ):
        """Builds two sample data from two processors.

        Sample X has labels 0, sample Y has labels 1.

        References
        ----------
        [1] - Revisiting Classifier Two-Sample Tests: https://arxiv.org/abs/1610.06545

        Parameters
        ----------
        processor_X : object
            Processor for sample X.
        processor_Y : object, optional
            Processor for sample Y, by default None. If None then sample Y is assumed to be the same as sample X.
        hold_out_ratio : float, optional
            Ratio of hold out data, by default 0.2.
        add_label_X : bool, optional
            Force insert label column in sample X, by default False.
        add_label_Y : bool, optional
            Force insert label column in sample Y, by default False.
        shuffle_random_state : int, optional
            Random state for shuffling, by default 0.
        """
        self.processor_X, self.processor_Y = processor_X, processor_Y
        self.hold_out_ratio = hold_out_ratio

        self.add_label_X, self.add_label_Y = add_label_X, add_label_Y

        self.shuffle_random_state = shuffle_random_state

        self.train_idx, self.hold_idx = None, None
        self.selection = None
        self.scalers = None
        self.hold_XY = None

    def _check_valid(self, data_X, data_Y, selection_X, selection_Y):
        data_X = data_X[~np.isnan(data_X).any(axis=1)]
        data_X = data_X[~np.isinf(data_X).any(axis=1)]

        data_Y = data_Y[~np.isnan(data_Y).any(axis=1)]
        data_Y = data_Y[~np.isinf(data_Y).any(axis=1)]

        if data_X.shape[0] != data_Y.shape[0]:
            min_shape = min(data_X.shape[0], data_Y.shape[0])
            logging.info(f"NaNs found, cutting to shape {min_shape} from {data_X.shape} and {data_Y.shape}!")
            data_X, data_Y = data_X[:min_shape], data_Y[:min_shape]

        assert data_X.shape[0] == data_Y.shape[0], "Data samples dim are not equal!"

        if not self.add_label_X and not self.add_label_Y:
            assert data_X.shape[1] == data_Y.shape[1], "Data features dim are not equal!"

        if self.add_label_X:
            assert data_X.shape[1] + 1 == data_Y.shape[1], "Data features dim are not equal!"

        if self.add_label_Y:
            assert data_Y.shape[1] + 1 == data_X.shape[1], "Data features dim are not equal!"

        assert selection_X is not None or selection_Y is not None, "Both selections are None!"

        if selection_X is not None and selection_Y is not None:
            assert selection_X.equals(selection_Y), "Selections are not equal!"

        logging.info("Validated X and Y samples! Will train on X, setting Y as holdout to be used for inference later.")

        return data_X, data_Y, selection_X, selection_Y

    def prepare(self):
        logging.info("Preparing two sample data...")
        data_X, selection_X, scalers_X = self.processor_X()

        if self.processor_Y is not None:
            data_Y, selection_Y, scalers_Y = self.processor_Y()
        else:
            logging.info("Using same processor for both samples (assuming closure test), split at half!")
            selection_Y, scalers_Y, self.add_label_Y = None, None, False
            data_X, data_Y = copy.deepcopy(data_X[: len(data_X) // 2, :]), copy.deepcopy(data_X[len(data_X) // 2 :, :])

        if data_X.shape[0] != data_Y.shape[0]:
            min_shape = min(data_X.shape[0], data_Y.shape[0])
            logging.info(f"Cutting data_X({len(data_X)}) and data_Y({len(data_Y)}) to len {min_shape}!")
            data_X, data_Y = data_X[:min_shape], data_Y[:min_shape]

        data_X, data_Y, selection_X, selection_Y = self._check_valid(data_X, data_Y, selection_X, selection_Y)

        if selection_X is not None:
            selection = selection_X
        elif selection_Y is not None:
            selection = selection_Y

        label_idx = selection[selection["feature"] == "label"].index

        if self.add_label_X:
            data_X = np.insert(data_X, label_idx, np.zeros((len(data_X), 1)), axis=1)
        else:
            data_X[:, label_idx] = np.zeros((len(data_X), 1))

        if self.add_label_Y:
            data_Y = np.insert(data_Y, label_idx, np.ones((len(data_Y), 1)), axis=1)
        else:
            data_Y[:, label_idx] = np.ones((len(data_Y), 1))

        idx = np.arange(len(data_X))
        idx = shuffle(idx, random_state=self.shuffle_random_state)
        n, m = int(len(idx) * self.hold_out_ratio), int(len(idx) * (1 - self.hold_out_ratio))

        self.hold_idx, self.train_idx = idx[:n], idx[m:]

        data_X, data_hold_X, data_Y, data_hold_Y = (
            data_X[self.train_idx],
            data_X[self.hold_idx],
            data_Y[self.train_idx],
            data_Y[self.hold_idx],
        )

        logging.info(f"Have data X: {data_X.shape} and data Y: {data_Y.shape}!")
        logging.info(f"Have holdout data X: {data_hold_X.shape} and holdout data Y: {data_hold_Y.shape}!")

        data_XY = np.concatenate([data_X, data_Y], axis=0)

        hold_XY = {"X": data_hold_X, "Y": data_hold_Y}

        return data_XY, hold_XY, selection, {"X": scalers_X, "Y": scalers_Y}

    def __call__(self, *args, **kwargs):
        """Return combined data, selection and scalers and save hold out data."""
        data_XY, hold_XY, selection, scalers = self.prepare()

        self.selection = selection
        self.scalers = scalers
        self.hold_XY = hold_XY

        return data_XY, selection, scalers
