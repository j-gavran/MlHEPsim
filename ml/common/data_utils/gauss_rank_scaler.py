import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class GaussRankScaler:
    def __init__(
        self,
        epsilon=1e-12,
        noise_level=1e-6,
        interp_kind="linear",
        do_copy=True,
        fill_value="extrapolate",
        flatten=False,
        interp_limit=10**6,
    ):
        """Gauss Rank Scaler.

        TODO: use multiprocessing for the scaling per feature.

        Parameters
        ----------
        epsilon : float, optional
            Bound for the rank, by default 1e-9.
        noise_level : float, optional
            Noise level for the scaling to make unique values, by default 1e-6.
        interp_kind : str, optional
            Interpolation kind, by default "linear".
        do_copy : bool, optional
            Copy the data in interp1d, by default True.
        fill_value : str, optional
            Fill value for the interpolation, by default "extrapolate".
        flatten : bool, optional
            Flatten the data, by default False.
        interp_limit : int, optional
            Limit of points to use (and save) for the interpolation, by default 10**5.

        References
        ----------
        [1] - https://github.com/aldente0630/gauss-rank-scaler

        """
        self.bound = 1.0 - epsilon
        self.noise_level = noise_level
        self.interp_kind = interp_kind
        self.do_copy = do_copy
        self.fill_value = fill_value
        self.flatten = flatten
        self.interp_limit = interp_limit

    def __call__(self, X, reverse=False, interp_funcs_lst=None, append_interp_funcs=True):
        assert not reverse or interp_funcs_lst is not None

        if self.flatten:
            logging.warning("Flattening the data! Not recommended.")
            original_shape = X.shape
            X = X.flatten()[:, None]

        if not reverse and append_interp_funcs:
            interp_funcs_lst = []

        X_transf, dim = np.zeros_like(X), X.shape[1]

        for i in tqdm(range(dim), desc="Scaling with GaussRankScaler", leave=False):
            x = X[:, i]

            if not reverse and self.noise_level is not None:
                x = x + np.random.normal(0, self.noise_level, len(x))

            if not reverse:
                scaled_rank = self.get_rank(x)
                interp_transf_x, interp_func = self.interpolate_erfinv(x, scaled_rank)

                X_transf[:, i] = interp_transf_x

                if append_interp_funcs:
                    interp_funcs_lst.append(interp_func)
            else:
                interp_func = interp_funcs_lst[i]
                X_transf[:, i] = self.inverse_interpolate_erf(x, interp_func)

        if self.flatten:
            X_transf = X_transf.reshape(original_shape)

        if not reverse:
            return X_transf, interp_funcs_lst
        else:
            return X_transf

    @staticmethod
    def _nan_warn(x, msg=""):
        if np.any(np.isnan(x)):
            n_nans = np.sum(np.isnan(x))
            logging.warning(f"{n_nans} NaNs in the interpolated transformed data! {msg}")
            return True
        else:
            return False

    def get_rank(self, x):
        rank = np.argsort(np.argsort(x))
        factor = np.max(rank) / 2.0 * self.bound

        scaled_rank = np.clip(rank / factor - self.bound, -self.bound, self.bound)

        return scaled_rank

    def interpolate_erfinv(self, x, scaled_rank):
        if len(x) < self.interp_limit:
            interp_limit = -1
        else:
            interp_limit = self.interp_limit

        interp_func = interp1d(
            x[:interp_limit],
            scaled_rank[:interp_limit],
            kind=self.interp_kind,
            copy=self.do_copy,
            fill_value=self.fill_value,
        )

        interp_transf_x = interp_func(x)
        interp_transf_x = np.clip(interp_transf_x, -self.bound, self.bound)
        interp_transf_x = erfinv(interp_transf_x)

        self._nan_warn(interp_transf_x, msg="From erfinv.")

        return interp_transf_x, interp_func

    def inverse_interpolate_erf(self, x, interp_func):
        inv_interp_func = interp1d(
            interp_func.y,
            interp_func.x,
            kind=self.interp_kind,
            copy=self.do_copy,
            fill_value=self.fill_value,
        )

        inv_interp_transf_x = erf(x)
        inv_interp_transf_x = inv_interp_func(inv_interp_transf_x)

        self._nan_warn(inv_interp_transf_x, msg="From erf.")

        return inv_interp_transf_x


class GaussRankTransform(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.training = True
        self.interp_funcs_lst = None
        self.gauss_rank_scaler = GaussRankScaler(*args, **kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        transf_X, interp_funcs_lst = self.gauss_rank_scaler(X, reverse=False, append_interp_funcs=self.training)

        if self.training:
            self.interp_funcs_lst = interp_funcs_lst
            self.training = False

        return transf_X

    def inverse_transform(self, X, y=None):
        transf_X = self.gauss_rank_scaler(X, reverse=True, interp_funcs_lst=self.interp_funcs_lst)
        return transf_X


if __name__ == "__main__":
    # test

    import matplotlib.pyplot as plt

    from ml.custom.HIGGS.analysis.utils import run_chainer

    data, selection, scalers = run_chainer(cont_rescale_type="gauss_rank", on_train="bkg", n_data=10**5)

    idx = selection[selection["type"] != "label"].index

    inv_data = scalers["cont"][0][1].inverse_transform(data[:, idx])

    fig, axs = plt.subplots(6, 3, figsize=(16, 28))
    axs = axs.flatten()

    labels = selection[selection["type"] != "label"]["feature"].values

    for i, (ax, label) in enumerate(zip(axs, labels)):
        ax.hist(data[:, i], bins=50, histtype="step", lw=2, label="scaled", range=(-4, 4), density=True)
        ax.hist(inv_data[:, i], bins=50, histtype="step", lw=2, label="original", range=(-4, 4), density=True)
        ax.set_xlabel(label)

    axs[0].legend()

    fig.tight_layout()
    fig.savefig("test.pdf")
    plt.close(fig)
