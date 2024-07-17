import numpy as np

from ml.common.data_utils.data_modules import DataModule, SupervisedDataset


class HiggsDataset(SupervisedDataset):
    def __init__(self, data, selection):
        super().__init__(data, selection)


class HiggsDataModule(DataModule):
    def __init__(self, processor, dataset=HiggsDataset, **kwargs):
        super().__init__(processor, dataset, **kwargs)

    def setup(self, stage=None):
        data, self.selection, self.scalers = self.processor()
        data = np.float32(data)

        self._get_splits(len(data))

        if stage == "fit" or stage is None:
            self.train = self.dataset(data[self.train_idx], self.selection)
            self.val = self.dataset(data[self.val_idx], self.selection)

        if stage == "test":
            self.test = self.dataset(data[self.test_idx], self.selection)
