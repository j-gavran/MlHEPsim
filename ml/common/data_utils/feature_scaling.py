import logging

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ml.common.data_utils.dequantization import DequantizationTransform
from ml.common.data_utils.gauss_rank_scaler import GaussRankTransform
from ml.common.data_utils.logit_transform import LogitTransform


def rescale_continuous_data(x, rescale_type, **kwargs):
    """Feature normalization of continious features.

    Parameters
    ----------
    x: np.ndarray
        Design matrix.
    rescale_type: str
        - normal: zero mean and unit variance
        - robust: removes the median and scales the data according to the quantile range
        - sigmoid: [0, 1] range
        - tanh: [-1, 1] range
        - maxabs: see [4]
        - logit: [0, 1] -> [-inf, inf] ranges
        - logit_normal: [0, 1] -> [-inf, inf] -> normal ranges
        - gauss_rank: https://github.com/aldente0630/gauss-rank-scaler
        - quantile: same as gauss_rank but in sklearn

    Note
    ----
    axis 0 -> column normalization (features)
    axis 1 -> row normalization (samples)

    References
    ----------
    [1] - https://scikit-learn.org/stable/modules/preprocessing.html
    [2] - https://stackoverflow.com/questions/51032601/why-scale-across-rows-not-columns-for-standardizing-preprocessing-of-data-befo/51032946
    [3] - https://towardsdatascience.com/creating-custom-transformers-for-sklearn-pipelines-d3d51852ecc1
    [4] - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html

    """
    if torch.is_tensor(x):
        x = x.numpy()

    if rescale_type == "normal":
        scaler = StandardScaler(**kwargs).fit(x)
        x_scaled = scaler.transform(x)
        scaler = [("standard scaler", scaler)]

    elif rescale_type == "robust":
        scaler = RobustScaler(**kwargs).fit(x)
        x_scaled = scaler.transform(x)
        scaler = [("robust scaler", scaler)]

    elif rescale_type == "sigmoid":
        scaler = MinMaxScaler(**kwargs).fit(x)
        x_scaled = scaler.transform(x)
        scaler = [("sigmoid scaler", scaler)]

    elif rescale_type == "tanh":
        scaler = MinMaxScaler(feature_range=(-1, 1), **kwargs).fit(x)
        x_scaled = scaler.transform(x)
        scaler = [("tanh scaler", scaler)]

    elif rescale_type == "maxabs":
        scaler = MaxAbsScaler(**kwargs).fit(x)
        x_scaled = scaler.transform(x)
        scaler = [("maxabs scaler", scaler)]

    elif rescale_type == "logit":
        transforms = Pipeline(
            steps=[
                ("sigmoid transform", MinMaxScaler()),
                ("logit transform", LogitTransform()),
            ]
        )
        x_scaled = transforms.fit_transform(x)
        scaler = [s for s in transforms.steps]

    elif rescale_type == "logit_normal":
        transforms = Pipeline(
            steps=[
                ("sigmoid transform", MinMaxScaler()),
                ("logit transform", LogitTransform()),
                ("normal transform", StandardScaler()),
            ]
        )
        x_scaled = transforms.fit_transform(x)
        scaler = [s for s in transforms.steps]

    elif rescale_type == "gauss_rank":
        scaler = GaussRankTransform(**kwargs)
        x_scaled = scaler.fit_transform(x)
        scaler = [("gauss rank transform", scaler)]

    elif rescale_type == "quantile":
        scaler = QuantileTransformer(output_distribution="normal", **kwargs).fit(x)
        x_scaled = scaler.transform(x)
        scaler = [("quantile transform", scaler)]

    elif rescale_type in ["none", None]:
        scaler, x_scaled = None, x

    else:
        raise ValueError

    return x_scaled, scaler


def rescale_discrete_data(x, rescale_type):
    """Feature normalization of discrete features.

    Parameters
    ----------
    x: np.ndarray
        Design matrix.
    rescale_type: str
        - onehot
        - dequant
        - dequant_normal
        - dequant_logit_normal

    References
    ----------
    [1] - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    """
    if torch.is_tensor(x):
        x = x.numpy()

    if rescale_type == "onehot":
        scaler = OneHotEncoder(sparse_output=False).fit(x)
        x_scaled = scaler.transform(x).astype(np.float32)
        scaler = [("onehot", scaler)]

    elif rescale_type == "dequant":
        scaler = DequantizationTransform()
        x_scaled = scaler.fit_transform(x)
        scaler = [("dequant transform", scaler)]

    elif rescale_type == "dequant_normal":
        transforms = Pipeline(
            steps=[
                ("dequant transform", DequantizationTransform()),
                ("normal transform", StandardScaler()),
            ],
        )
        x_scaled = transforms.fit_transform(x)
        scaler = [s for s in transforms.steps]

    elif rescale_type == "dequant_logit_normal":
        transforms = Pipeline(
            steps=[
                ("dequant transform", DequantizationTransform()),
                ("sigmoid transform", MinMaxScaler()),
                ("logit transform", LogitTransform()),
                ("normal transform", StandardScaler()),
            ],
        )
        x_scaled = transforms.fit_transform(x)
        scaler = [s for s in transforms.steps]

    elif rescale_type in ["none", None]:
        scaler, x_scaled = None, x

    else:
        raise ValueError

    return x_scaled, scaler


class RescalingHandler:
    def __init__(self, selection, scalers=None):
        """Helper class to handle the rescaling of continuous and discrete features.

        Parameters
        ----------
        scalers : dict of list of tuples or None
            Dictionary containing the scalers for continuous and discrete features. Passed from Preprocessor class.
        selection : pd.DataFrame
            DataFrame containing the selection of continuous and discrete features. Passed from Preprocessor class.

        Example
        -------
        scalers = {
                    "cont": [("standard scaler", StandardScaler()), ("robust scaler", RobustScaler())],
                    "disc": [("onehot", OneHotEncoder(sparse_output=False))],
                }

        Methods
        -------
        transform: rescale the input sample
        inverse_transform: rescale the input sample back to the original scale
        rescale_from_dct: rescale the input samples using the scalers where both are in dictionary format

        Note
        ----
        For rescale_forward and rescale_back scalers are needed in initialization.

        Other parameters
        ----------------
        samples : list or np.ndarray

        """
        self.selection = selection
        self.scalers = scalers

    def _handle_rescale_type(self, sample):
        cont_idx = self.selection[self.selection["type"] == "cont"].index
        disc_idx = self.selection[self.selection["type"] == "disc"].index
        other_idx = self.selection[(self.selection["type"] != "cont") & (self.selection["type"] != "disc")].index

        scale_dct = {"cont": None, "disc": None, "other": None}

        if len(cont_idx) != 0:
            cont_sample = sample[:, cont_idx]
            cont_scaler_lst = self.scalers["cont"]
            scale_dct["cont"] = (cont_sample, cont_scaler_lst, cont_idx)

        if len(disc_idx) != 0:
            disc_sample = sample[:, disc_idx]
            disc_scaler_lst = self.scalers["disc"]
            scale_dct["disc"] = (disc_sample, disc_scaler_lst, disc_idx)

        if len(other_idx) != 0 and len(self.selection) == sample.shape[1]:
            logging.warning(f"Other features detected. No scaling will be performed on feature index {other_idx}!")
            other_sample = sample[:, other_idx]
            scale_dct["other"] = [other_sample, ("none", None), other_idx]

        return scale_dct

    def _scaling_handler(self, sample, forward=True):
        if type(sample) is list:
            logging.warning("List of samples detected. Using the first sample!")
            sample = sample[0]

        if not isinstance(sample, np.ndarray):
            sample = sample.cpu().numpy()

        scale_dct = self._handle_rescale_type(sample)

        cont_scale, disc_scale, other_scale = False, False, False

        if scale_dct["cont"] is not None:
            cont_scale = True
            cont_sample, cont_scaler_lst, cont_idx = scale_dct.pop("cont")

            if forward:
                for scaler in cont_scaler_lst:
                    cont_sample = scaler[1].fit_transform(cont_sample)
            else:
                for scaler in cont_scaler_lst[::-1]:
                    cont_sample = scaler[1].inverse_transform(cont_sample)

        if scale_dct["disc"] is not None:
            disc_scale = True
            disc_sample, disc_scaler_lst, disc_idx = scale_dct.pop("disc")

            if forward:
                for scaler in disc_scaler_lst:
                    disc_sample = scaler[1].fit_transform(disc_sample)
            else:
                for scaler in disc_scaler_lst[::-1]:
                    disc_sample = scaler[1].inverse_transform(disc_sample)

        if scale_dct["other"] is not None:
            other_scale = True
            other_sample, _, other_idx = scale_dct.pop("other")

        sample = np.zeros_like(sample)

        if cont_scale and disc_scale:
            sample[:, cont_idx], sample[:, disc_idx] = cont_sample, disc_sample
        elif cont_scale:
            sample[:, cont_idx] = cont_sample
        elif disc_scale:
            sample[:, disc_idx] = disc_sample
        else:
            logging.error("No scaling was performed!")

        if other_scale:
            sample[:, other_idx] = other_sample

        logging.info(f"[green]Rescaling with forward mode set to {forward} was successful![/green]")

        return sample

    def transform(self, sample):
        return self._scaling_handler(sample, forward=True)

    def inverse_transform(self, sample):
        return self._scaling_handler(sample, forward=False)

    def rescale_from_dct(
        self,
        samples_dct: dict[str : list[np.ndarray] | np.ndarray],
        scalers_dct: dict[str : list[dict[str:tuple]] | dict[str:tuple]],
        forward: bool = True,
    ) -> dict[str : list[np.ndarray] | np.ndarray]:
        """Rescale the input samples using the scalers where both are in dictionary format.

        Example
        ------
        samples_dct = {model_name_1: [sample_1, sample_2, ...], model_name_2: [sample_1, sample_2, ...]}
        scalers_dct = {model_name_1: [scaler_1, scaler_2, ...], model_name_2: [scaler_1, scaler_2, ...]}

        Parameters
        ----------
        samples_dct : dict where key is str and value is list or np.ndarray
            Dictionary containing the samples for each model.
        scalers_dct : dict where key is str and value is dict (assume same scaling for all samples in the list)
            Dictionary containing the scalers for each model.
        forward : bool, optional
            Forward or backward rescaling, by default True.
        """
        for key, samples in samples_dct.items():
            if type(samples) is not list:
                samples = [samples]

            scaled_sample = []

            for sample in samples:
                self.scalers = scalers_dct[key]

                if forward:
                    sample = self.transform(sample)
                else:
                    sample = self.inverse_transform(sample)

                scaled_sample.append(sample)

            samples_dct[key] = scaled_sample

        return samples_dct
