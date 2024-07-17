import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from ml.common.data_utils.processors import FeatureSelector, NpyProcessor
from ml.common.data_utils.utils import url_download
from ml.common.nn.gen_model_sampler import GenModelSampler


class HIGGSNpyProcessor(NpyProcessor):
    def __init__(
        self,
        data_dir,
        base_file_name="HIGGS",
        keep_ratio=1.0,
        shuffle=True,
        hold_mode=False,
        use_hold=False,
        hold_ratio=0.2,
    ):
        """HIGGS dataset to .npy starting processor.

        Note
        ----
        Supports holdout mode, where data is split into two partitions. Partition 1 is used for training and partition 2
        is used for independent holdout evaluation.

        Parameters
        ----------
        keep_ratio : float, optional
            Keep only a fraction of the data, by default 1.0.
        shuffle : bool, optional
            Shuffle data loaded from starting file, by default True.
        hold_mode : bool, optional
            Holdout mode to use partition_1 of partition_2 of holdout data, by default False.
        use_hold : bool, optional
            If True use partition_1 else use partition_2, by default False.
        hold_ratio : float, optional
            Ratio of holdout data in partition_2, by default 0.2.

        Other parameters
        ----------------
        colnames_dct : dict
            Dictionary of column names and types.
        file_name : str
            File name of downloaded file.

        """
        super().__init__(data_dir, base_file_name)
        self.file_name = None
        self.keep_ratio = keep_ratio
        self.shuffle = shuffle
        self.hold_mode, self.use_hold, self.hold_ratio = hold_mode, use_hold, 1 - hold_ratio

        if self.hold_mode:
            self.hold_npy_partition_1 = self.npy_file.replace(".npy", "_hold_partition_1.npy")
            self.hold_npy_partition_2 = self.npy_file.replace(".npy", "_hold_partition_2.npy")

    def _select_npy_file(self):
        if self.hold_mode and not self.use_hold:
            logging.info(f"Using holdout partition 1 from {self.hold_npy_partition_1}!")
            return self.hold_npy_partition_1, self.features
        elif self.hold_mode and self.use_hold:
            logging.info(f"Using holdout partition 2 from {self.hold_npy_partition_2}!")
            return self.hold_npy_partition_2, self.features
        else:
            logging.info(f"Using {self.npy_file}!")
            return self.npy_file, self.features

    def __call__(self, *args, **kwargs):
        dataset = self.get_dataset()

        if dataset is None:
            logging.info(f"{self.npy_file} already exists!")
            return self._select_npy_file()

        dataset = self.process_dataset(dataset)

        if self.hold_mode:
            hold_idx = int(len(dataset) * self.hold_ratio)

            hold_dataset = dataset[:hold_idx]
            dataset = dataset[hold_idx:]

            self.make_npy_file(dataset, self.hold_npy_partition_1)
            self.make_npy_file(hold_dataset, self.hold_npy_partition_2)
        else:
            dataset = self.process_dataset(dataset)
            self.make_npy_file(dataset, self.npy_file)

        return self._select_npy_file()

    def get_dataset(self):
        """Creates higgs dataset dataframe. Runs download_higgs_dataset and _read_gzip_df.

        Parameters
        ----------
        data_dir : str, optional
            Path to higgs data, by default "data/".

        Returns
        -------
        pd.DataFrame
            29 dim dataframe of all downloaded data.

        """
        if self.hold_mode:
            if Path(self.hold_npy_partition_1).is_file() and Path(self.hold_npy_partition_2).is_file():
                return None
        else:
            if Path(self.npy_file).is_file():
                return None

        self.download()
        dataset = self._read_gzip_df()

        return dataset

    def download(self):
        """Downloads Higgs dataset from https://archive.ics.uci.edu/ml/datasets/HIGGS. If already downloaded returns file path.

        Returns
        -------
        file_name : str
            File path of downloaded file.

        References
        ----------
        [1] - Searching for exotic particles in high-energy physics with deep learning: https://www.nature.com/articles/ncomms5308

        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        self.file_name = url_download(url, self.data_dir)
        return self.file_name

    def _read_gzip_df(self):
        """Read .gz file using pandas (used in get_higgs_dataset function). Can be slow."""
        logging.info("Started reading csv zip!")
        dataset = pd.read_csv(self.file_name, engine="c", compression="gzip", dtype=np.float32)

        logging.info("Done reading csv zip to pandas DataFrame!")

        return dataset

    def process_dataset(self, dataset):
        logging.info("Processing dataset!")

        if self.shuffle:
            logging.info("Shuffling data!")
            dataset = dataset.sample(frac=1).reset_index(drop=True)

        if self.keep_ratio < 1.0:
            logging.info(f"Keeping only {self.keep_ratio:.2f} of data!")
            dataset = dataset[: int(len(dataset) * self.keep_ratio)]

        return dataset

    def make_npy_file(self, dataset, npy_file=None):
        """Make .npy files from higgs.csv file. Used for faster loading times."""
        if npy_file is None:
            npy_file = self.npy_file

        X = dataset.to_numpy()
        np.save(npy_file, X)
        logging.info(f"saved {npy_file} of shape {X.shape}!")

        return self


class HIGGSFeatureSelector(FeatureSelector):
    def __init__(self, file_path, n_data=None, on_train=None, **kwargs):
        super().__init__(file_path, **kwargs)
        assert on_train in ["bkg", "sig", None], "on_train must be one of ['bkg', 'sig'] or None for both!"

        self.n_data = n_data

        if on_train == "bkg":
            logging.info("Will drop signal data! Training on background data only!")
            self.drop_labels = [1]
        if on_train == "sig":
            logging.info("Will drop background data! Training on signal data only!")
            self.drop_labels = [0]

    def load_data(self):
        logging.info(f"Loading data from {self.file_path}!")
        data = np.load(self.file_path)

        if self.n_data is not None:
            data = data[: self.n_data]
            logging.info(f"Using {self.n_data} data points!")

        return data

    def select_features(self, data):
        return super().select_features(data)


class CatGenerated:
    def __init__(self, model_name, cat_label, save_dir="ml/data/higgs", file_name="HIGGS_generated"):
        self.model_name = model_name
        self.cat_label = cat_label
        self.save_dir, self.file_name = save_dir, file_name

    def __call__(self, data, selection, scalers, *args, **kwargs):
        return self.cat_gen(data, selection), selection, scalers

    def cat_gen(self, data, selection):
        label_idx = selection[selection["type"] == "label"].index[0]

        label_mask = data[:, label_idx] == self.cat_label
        data_label, data_other = data[label_mask], data[~label_mask]

        N = len(data_other)

        # BPK
        N_mc_org = len(data_label)  # why other?
        N_mc = 3 * N_mc_org // 4  # // 2 originally
        N_gen = N_mc_org - N_mc  # equal originally

        sampler = GenModelSampler(self.model_name, save_dir=self.save_dir, file_name=self.file_name)
        data_gen = sampler.sample(N_gen, resample=1)[self.model_name][0]

        data_gen_labels = np.ones(len(data_gen)) * self.cat_label
        data_gen = np.insert(data_gen, label_idx, data_gen_labels, axis=1)

        logging.info(f"Concatenating {len(data_gen)} generated data with {N} other data points!")

        # replace half of MC with some label (0 for bkg) with generated data
        # both need to have same scaling!
        cat = np.concatenate([data_gen, data_label[:N_mc], data_other], axis=0)

        return shuffle(cat)


if __name__ == "__main__":
    from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
    from ml.common.utils.loggers import setup_logger

    setup_logger()

    npy_proc = HIGGSNpyProcessor(
        "ml/data/higgs/",
        base_file_name="HIGGS_data",
        keep_ratio=0.02,
        shuffle=True,
    )

    f_sel = HIGGSFeatureSelector(
        npy_proc.npy_file,
        drop_types=["uni"],
    )

    pre = Preprocessor(
        cont_rescale_type=None,
        disc_rescale_type="dequant",
        no_process=["label"],
    )

    chainer = ProcessorChainer(npy_proc, f_sel, pre)
    data, selection, scalers = chainer()

    print(selection)
    print("selected data shape:", data.shape)

    # save for pytorch tutorial
    np.save("ml/data/higgs/HIGGS_data_pytorch_tutorial.npy", data)

    from ml.custom.HIGGS.higgs_dataset import HiggsDataModule

    higgs_dm = HiggsDataModule(chainer)
    higgs_dm.setup()

    batch = next(iter(higgs_dm.train_dataloader()))
    print("debug batch:", batch)
