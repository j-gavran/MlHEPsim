import numpy as np
import pytorch_lightning as pl
from ml_hep_sim.data_utils.dataset_utils import Dataset
from sklearn.utils import shuffle
from torch.utils.data import DataLoader


class MnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=128,
        rescale="unit",
        subset_n=None,
        to_gpu=False,
        shuffle_data=False,
        labels=None,
        **dl_kwargs
    ):
        """Mnist pl data module.

        Parameters
        ----------
        data_dir : list
            List of train.npy and test.npy file locations.
        batch_size : int
            Batch size.
        subset_n : list of ints, optional
            List of 2 ints for train and test subsets (number of rows in X to keep), by default None (all rows).
        to_gpu : bool, optional
            If True put data on gpu, by default False.
        rescale: str, optional
            unit : [0, 1] (divide by 255), normal - Gaussian: ((X / 255) - 0.1307) / 0.3081, dequantize or None.
        labels : list of str, optional
            List of file names with labels, by default None.
        shuffle_data : bool, optional
            Shuffles input data arrays, by default False.

        Note
        ----
        - Labels are all 0.
        - 28 x 28 images are reshaped to dim 784 vectors and put into X matrix.

        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.subset_n = subset_n if subset_n is not None else [None, None, None]
        self.dl_kwargs = dl_kwargs
        self.to_gpu = to_gpu
        self.rescale = rescale
        self.labels = labels
        self.shuffle = shuffle_data

    def prepare_data(self):
        """Prepares MNIST data:

        - Load .npy files with numpy.
        - Shuffle.
        - Convert to float32.
        - Subset data.
        - Preprocess (rescale).
        - Add labels.

        """
        self.data = [np.load(d).astype(np.float32) for d in self.data_dir]

        # reshape
        for i in range(len(self.data)):
            self.data[i] = self.data[i].reshape(self.data[i].shape[0], self.data[i].shape[1] * self.data[i].shape[2])

        # add labels
        if self.labels:
            self.label_data = [np.load(l).astype(np.int64) for l in self.labels]
        else:
            self.label_data = [np.zeros(len(self.data[i]), dtype=np.float32) for i in range(2)]

        # shuffle
        if self.shuffle:
            for i in range(len(self.data)):
                self.data[i], self.label_data[i] = shuffle(self.data[i], self.label_data[i])

        # subset of data
        for i, s in enumerate(self.subset_n):
            if s is not None:
                assert s <= len(self.data[i])
                self.data[i] = self.data[i][:s]
                self.label_data[i] = self.label_data[i][:s]

        # preprocess
        for i in range(len(self.data)):

            if self.rescale == "unit":
                self.data[i] = self.data[i] / 255
            elif self.rescale == "normal":
                self.data[i] = self.data[i] / 255
                self.data[i] = (self.data[i] - 0.1307) / 0.3081  # zero mean and unit variance for MNIST
            elif self.rescale == "dequantize":
                self.data[i] = (self.data[i] + np.random.uniform(size=self.data[i].shape).astype(np.float32)) / 255
                self.data[i] = self.data[i].clip(0, 1)
            elif self.rescale is None:
                pass
            else:
                raise ValueError

    def setup(self, stage=None):
        """Creates train, and test/val datasets."""
        if stage == "fit" or stage is None:
            self.train = Dataset(self.data[0], self.label_data[0], to_gpu=self.to_gpu)
            self.val = Dataset(self.data[1], self.label_data[1], to_gpu=self.to_gpu)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=True, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)

    def get_data_shape(self):
        data = [np.load(d) for d in self.data_dir]
        return [i.shape for i in data]


if __name__ == "__main__":
    mnist_data = MnistDataModule(
        [
            "data/mnist/train.npy",
            "data/mnist/test.npy",
        ],
        batch_size=1024,
        num_workers=12,
        rescale="dequantize",
        labels=None,
        shuffle_data=True,
    )

    idx = -1

    print(mnist_data.get_data_shape())

    mnist_data.prepare_data()
    mnist_data.setup()
    s = mnist_data.train.X[idx]

    print(s.dtype)

    print(mnist_data.train.y[idx])

    import matplotlib.pyplot as plt

    plt.imshow(s.reshape(28, 28))
    plt.colorbar()
    plt.show()
