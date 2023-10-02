import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from torch.utils.data import Dataset, Subset

from ml_hep_sim.data_utils.gauss_rank_scaler import GaussRankScaler


def plot_data_hist(X, axs, n_bins, log_scale=False, col_names=None, **hist_kwargs):
    if len(axs.shape) > 1:
        axs = axs.flatten()

    if torch.is_tensor(X):
        X = X.numpy()

    for i in range(X.shape[1]):
        bin_edges = np.histogram_bin_edges(X[:, i], bins=n_bins)
        axs[i].hist(X[:, i], bins=bin_edges, histtype="step", **hist_kwargs)

        if log_scale:
            axs[i].set_yscale("log")

        if col_names:
            axs[i].set_xlabel(col_names[i])

    return axs


def train_val_split(dataset, split, **kwargs):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=split, **kwargs)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_val_test_split(dataset, split_train, split_test=0.5, **kwargs):
    train_sub, test_val_sub = train_val_split(dataset, split_train, **kwargs)

    val_sub, test_sub = train_val_split(test_val_sub, split_test, **kwargs)

    return train_sub, val_sub, test_sub


class LogitTransform(BaseEstimator, TransformerMixin):
    def __init__(self, lam=0.0, epsilon=1e-5, validate=True):
        """https://en.wikipedia.org/wiki/Logit#Definition"""
        super().__init__()
        self.lam = lam
        self.epsilon = epsilon
        self.validate = validate

    def validate_data(self, X):
        if self.validate:
            test = (X + self.epsilon >= 0) & (X - self.epsilon <= 1)
            assert test.all(), "values must be in range [0, 1]"

    def logistic_transform(self, x):
        # ignore RuntimeWarning: overflow encountered in exp but watch out for nans in the final result!
        return 1 / (1 + np.exp(-x))

    def logit_transform(self, p):
        p_ = self.lam + (1 - 2 * self.lam) * p
        p_ = np.clip(p_, self.epsilon, 1 - self.epsilon)
        return np.log(p_ / (1 - p_))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.validate_data(X)
        return self.logit_transform(X)

    def inverse_transform(self, X, y=None):
        return self.logistic_transform(X)


def rescale_data(x_data, rescale_type):
    """Feature normalization.

    Parameters
    ----------
    x_data: np.ndarray
        Design matrix.
    rescale_type: str
        - normal: zero mean and unit variance
        - robust: removes the median and scales the data according to the quantile range
        - sigmoid: [0, 1] range
        - tanh: [-1, 1] range
        - logit: [0, 1] -> [-inf, inf] ranges
        - logit_normal: [0, 1] -> [-inf, inf] -> normal ranges
        - Gauss scaler: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927

    Note
    ----
    axis 0 -> column normalization (features)
    axis 1 -> row normalization (samples)

    Example
    -------
    Consider a data matrix M and |M - max(M)| normalization:

             height  weight  age (axis 1) samples
    person1  180     65      40
    person2  150     45      40
    person3  190     80      20
    (axis 0)
    features

    - option 1 (default)
        max(M, axis=0): [190,  80,  40]

        |M - max_1|:
                 height  weight  age
        person1  10      15      0
        person2  40      35      0
        person3  0       0       20

    - option 2
        norm = abs(max(M, axis=1)): [180, 150, 190]

        |M - max_2|:
                 height  weight  age
        person1  0      115      140
        person2  0      105      110
        person3  0      110      170

    - option 3
        norm = abs(max(M, axis=None)): 190

        |M - max_3|:
                 height  weight  age
        person1  10      125     150
        person2  40      145     150
        person3  0       110     170

    References
    ----------
    [1] - https://scikit-learn.org/stable/modules/preprocessing.html
    [2] - https://stackoverflow.com/questions/51032601/why-scale-across-rows-not-columns-for-standardizing-preprocessing-of-data-befo/51032946
    [3] - https://towardsdatascience.com/creating-custom-transformers-for-sklearn-pipelines-d3d51852ecc1

    """
    scaler = None

    if rescale_type == "normal":
        scaler = StandardScaler().fit(x_data)
        x_scaled = scaler.transform(x_data)

    elif rescale_type == "robust":
        scaler = RobustScaler().fit(x_data)
        x_scaled = scaler.transform(x_data)

    elif rescale_type == "sigmoid":
        scaler = MinMaxScaler().fit(x_data)
        x_scaled = scaler.transform(x_data)

    elif rescale_type == "tanh":
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x_data)
        x_scaled = scaler.transform(x_data)

    elif rescale_type == "logit":
        transforms = Pipeline(
            steps=[
                ("sigmoid transform", MinMaxScaler()),
                ("logit transform", LogitTransform()),
            ]
        )
        scaler = transforms.fit(x_data)
        x_scaled = scaler.transform(x_data)

    elif rescale_type == "logit_normal":
        transforms = Pipeline(
            steps=[
                ("sigmoid transform", MinMaxScaler()),
                ("logit transform", LogitTransform()),
                ("normal transform", StandardScaler()),
            ]
        )
        scaler = transforms.fit(x_data)
        x_scaled = scaler.transform(x_data)

    elif rescale_type == "gauss_scaler":
        scaler = GaussRankScaler()
        x_scaled = scaler.fit_transform(x_data)

    elif rescale_type == "maxabs":
        scaler = MaxAbsScaler()
        x_scaled = scaler.fit_transform(x_data)

    elif rescale_type in ["none", None]:
        return x_data, None

    else:
        raise ValueError

    return x_scaled, scaler


class Dataset(Dataset):
    def __init__(self, X, y, to_gpu=False):
        """https://pytorch.org/docs/stable/data.html#module-torch.utils.data"""
        if to_gpu:
            self.X, self.y = torch.from_numpy(X).cuda(), torch.from_numpy(y).cuda()
        else:
            self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    from ml_hep_sim.data_utils.higgs.higgs_dataset import HiggsDataModule

    _data = [
        "data/higgs/HIGGS_18_feature_train.npy",
        "data/higgs/HIGGS_18_feature_val.npy",
        "data/higgs/HIGGS_18_feature_test.npy",
    ]

    higgs_data = HiggsDataModule(
        _data,
        batch_size=1,
        rescale="logit",
        num_workers=12,
        subset_n=[10**5, 0, 0],
        shuffle_data=False,
    )

    higgs_data.prepare_data()
    higgs_data.setup()
