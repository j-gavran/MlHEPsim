import numpy as np
import pytorch_lightning as pl
from ml_hep_sim.data_utils.dataset_utils import Dataset
from torch.utils.data import DataLoader


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, data_size=None, batch_size=None, to_gpu=False, **dl_kwargs):
        """Dummy pl data module for testing.

        Parameters
        ----------
        data_size : tuple
            Size of data matrix X. First feature column is reserved for label.
        batch_size : int
            Batch size.
        to_gpu : bool, optional
            If True put data on gpu, by default False.
        dl_kwargs: \*\*kwargs
            Additional parameters for torch DataLoader class. See https://pytorch.org/docs/stable/data.html .

        References
        ----------
        - https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html

        """
        super().__init__()
        self.data_size = data_size
        self.batch_size = batch_size
        self.to_gpu = to_gpu
        self.dl_kwargs = dl_kwargs
        self.data, self.labels = None, None

    def prepare_data(self):
        """Creates a list [train, val, test] of numpy float32 normaly distributed X matrices."""
        self.data = [np.random.normal(loc=0, scale=1, size=self.data_size).astype(np.float32) for _ in range(3)]
        self.labels = [np.zeros(self.data_size) for _ in range(3)]

    def setup(self, stage=None):
        """Creates train, val and test datasets."""
        if stage == "fit" or stage is None:
            self.train = Dataset(self.data[0], self.labels[0], to_gpu=self.to_gpu)
            self.val = Dataset(self.data[1], self.labels[1], to_gpu=self.to_gpu)
            self.test = Dataset(self.data[2], self.labels[2], to_gpu=self.to_gpu)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=True, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)


if __name__ == "__main__":
    ddm = DummyDataModule((1000, 10), 100)
    ddm.prepare_data()
    ddm.setup()

    t_dl = ddm.train_dataloader()

    print(ddm.train.X.shape)
