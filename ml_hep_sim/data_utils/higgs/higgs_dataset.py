import numpy as np
import pytorch_lightning as pl
from ml_hep_sim.data_utils.dataset_utils import Dataset, rescale_data
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import logging


class HiggsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=128,
        rescale=None,
        subset_n=None,
        to_gpu=False,
        shuffle_data=False,
        train_val_split=None,
        **dl_kwargs,
    ):
        """Higgs dataset DataModule (serves as a base for all other data modules).

        Note
        ----
        See data_utils.dummy_dataset for an additional example.

        Parameters
        ----------
        data_dir : str or list of str
            List of train/val/test data locations or string to file location.
        batch_size : int, optional
            Batch size, by default 128.
        rescale : str, optional
           See data_utils.dataset_utils, by default None.
        subset_n : int, optional
            Subset of data to consider, by default None.
        to_gpu : bool, optional
            Put data directly to gpu memory, by default False.
        shuffle_data : bool, optional
            Shuffle arrays of data, by default False.
        train_val_split : float, optional
            See load_data function, by default None.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.rescale = rescale
        if self.rescale is not None and self.rescale.lower() == "none":
            self.rescale = None
        self._subset_n = subset_n
        self.dl_kwargs = dl_kwargs  # additional dataloader kwargs
        self.to_gpu = to_gpu
        self.shuffle = shuffle_data
        self.scalers = []  # save list for sklearn scalers
        self.load_data(train_val_split)

    def load_data(self, train_val_split):
        """Load npy files into lists of train/val/test arrays.

        Parameters
        ----------
        train_val_split : float in (0, 1), optional
            If not None splits data into partitions defined by train_val_split %.
            Can only be used if only one data_dir file given.
        """
        if type(self.data_dir) is str:
            self.data_dir = [self.data_dir]

        if train_val_split:
            assert len(self.data_dir) == 1
            assert 0.0 < train_val_split < 1.0
            data = np.load(self.data_dir[0]).astype(np.float32)
            idx = int(len(data) * train_val_split)
            self.data = [data[:idx, :], data[idx:, :]]
        else:
            self.data = [np.load(d).astype(np.float32) for d in self.data_dir]

        self.subset_n = self._subset_n if self._subset_n is not None else [None] * len(self.data)

    def prepare_data(self):
        if self.shuffle:
            for i in range(len(self.data)):
                self.data[i] = shuffle(self.data[i])

        if self.rescale is not None:
            # zeroth column are labels that we dont want to normalize
            for i in range(len(self.data)):
                self.data[i][:, 1:], scaler = rescale_data(self.data[i][:, 1:], rescale_type=self.rescale)
                self.scalers.append(scaler)  # <- this is where scalers get saved!

        for i, s in enumerate(self.subset_n):
            if s is not None:
                if s > len(self.data[i]):
                    logging.warning(
                        f"subset N < len(data) at index i; using len(data) as subset. Have diff: {len(self.data[i]) - s}"
                    )
                    s = len(self.data[i])
                self.data[i] = self.data[i][:s]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = Dataset(self.data[0][:, 1:], self.data[0][:, 0][:, None], to_gpu=self.to_gpu)
            self.val = Dataset(self.data[1][:, 1:], self.data[1][:, 0][:, None], to_gpu=self.to_gpu)
            if len(self.data) == 3:
                self.test = Dataset(self.data[2][:, 1:], self.data[2][:, 0][:, None], to_gpu=self.to_gpu)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=True, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)

    def get_data_shape(self):
        data = [np.load(d) for d in self.data_dir]
        return [i.shape for i in data]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _data = [
        "data/higgs/HIGGS_18_feature_train.npy",
        "data/higgs/HIGGS_18_feature_val.npy",
        "data/higgs/HIGGS_18_feature_test.npy",
    ]

    higgs_data = HiggsDataModule(
        _data,
        batch_size=1024,
        rescale=None,
        num_workers=12,
        subset_n=[10 ** 5, 10 ** 5, 10 ** 5],
        shuffle_data=True,
    )

    higgs_data.prepare_data()
    higgs_data.setup()
    train_data = higgs_data.train.X.numpy()

    fig, axs = plt.subplots(6, 3)
    axs = axs.flatten()

    for i in range(len(axs)):
        axs[i].hist(train_data[:, i], bins=40, histtype="step")

    plt.show()

    print(np.unique(higgs_data.train.y, return_counts=True))
