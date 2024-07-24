import logging
import os
from functools import lru_cache

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC
from tqdm import tqdm

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.utils.plot_utils import filter_array
from ml.common.utils.register_model import fetch_registered_module
from ml.custom.HIGGS.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

MODEL_MAP = {
    "RealNVP_flow_model_gauss_rank": "RealNVP",
    "Glow_flow_model_gauss_rank": "Glow",
    "rqsplines_flow_model_gauss_rank": "RQSplines",
    "MAF_flow_model_gauss_rank": "MAF",
    "MAFMADEMOG_flow_model_gauss_rank": "MAFMADEMOG",
    "MADEMOG_flow_model_gauss_rank": "MADEMOG",
    "RealNVP_flow_model_gauss_rank_interp": "RealNVP",
    "Glow_flow_model_gauss_rank_interp": "Glow",
    "rqsplines_flow_model_gauss_rank_interp": "RQSplines",
    "MAF_flow_model_gauss_rank_interp": "MAF",
    "MAFMADEMOG_flow_model_gauss_rank_interp": "MAFMADEMOG",
    "MADEMOG_flow_model_gauss_rank_interp": "MADEMOG",
    "MADEMOG_flow_model_gauss_rank3": "MADEMOG 128",
    "MADEMOG_flow_model_gauss_rank4": "MADEMOG 256",
    "MADEMOG_flow_model_gauss_rank5": "MADEMOG 512",
    "MADEMOG_flow_model_gauss_rank6": "MADEMOG 1024",
    "MADEMOG_flow_model_gauss_rank7": "MADEMOG 2048",
    "MADEMOG_flow_model_logit_normal": "MADEMOG logit normal",
    "MADEMOG_flow_model_gauss_rank_mini": "MADEMOG",
    "MADEMOG_flow_model_gauss_rank_best": "MADEMOG",
    "ref": "MC",
}

LABELS_MAP = {
    "lepton pT": r"lepton $p_\mathrm{T}$",
    "lepton eta": r"lepton $\eta$",
    "missing energy": r"missing energy",
    "jet1 pT": r"jet1 $p_\mathrm{T}$",
    "jet1 eta": r"jet1 $\eta$",
    "jet2 pT": r"jet2 $p_\mathrm{T}$",
    "jet2 eta": r"jet2 $\eta$",
    "jet3 pT": r"jet3 $p_\mathrm{T}$",
    "jet3 eta": r"jet3 $\eta$",
    "jet4 pT": r"jet4 $p_\mathrm{T}$",
    "jet4 eta": r"jet4 $\eta$",
    "m jj": r"$m_{jj}$",
    "m jjj": r"$m_{jjj}$",
    "m lv": r"$m_{\ell\nu}$",
    "m jlv": r"$m_{j\ell\nu}$",
    "m bb": r"$m_{bb}$",
    "m wbb": r"$m_{Wbb}$",
    "m wwbb": r"$m_{WWbb}$",
}


BINNING = {
    "lepton pT": (0, 5),
    "lepton eta": (-4, 4),
    "missing energy": (0, 5),
    "jet1 pT": (0, 5),
    "jet1 eta": (-4, 4),
    "jet2 pT": (0, 5),
    "jet2 eta": (-4, 4),
    "jet3 pT": (0, 5),
    "jet3 eta": (-4, 4),
    "jet4 pT": (0, 4),
    "jet4 eta": (-4, 4),
    "m jj": (0, 3),
    "m jjj": (0, 3),
    "m lv": (0.9, 1.2),
    "m jlv": (0, 3),
    "m bb": (0, 5),
    "m wbb": (0, 4),
    "m wwbb": (0, 3),
}


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iqr_outliers(data, bound=1.5, q1_set=25, q3_set=75, mode="inside"):
    q1 = np.percentile(data, q1_set)
    q3 = np.percentile(data, q3_set)
    iqr = q3 - q1

    lower_bound = q1 - bound * iqr
    upper_bound = q3 + bound * iqr

    inside_idx = (data >= lower_bound) & (data <= upper_bound)  # inside of whiskers
    outside_idx = (data < lower_bound) | (data > upper_bound)  # outliers

    outliers = np.zeros_like(data)

    if mode == "inside":
        outliers[inside_idx] = data[inside_idx]
        outliers[outside_idx] = np.nan
    elif mode == "outside":
        outliers[inside_idx] = np.nan
        outliers[outside_idx] = data[outside_idx]
    else:
        raise ValueError(f"Unknown mode {mode}")

    return outliers


def outlier_selection(data, mode):
    assert mode in ["inside", "outside", "all"]
    assert len(data.shape) <= 2

    logging.info(f"Selecting outliers with mode {mode}")

    if mode == "all":
        return data

    if len(data.shape) == 1:
        return iqr_outliers(data, mode=mode)

    for i in range(data.shape[1]):
        data[:, i] = iqr_outliers(data[:, i], mode=mode)

    return data


def get_model(model_name, ver=-1):
    module = fetch_registered_module(model_name, ver, device="cuda")
    return module.model


@lru_cache(maxsize=None)
def get_scaler(model_name, ver=-1):
    logging.info(f"Fetching scalers from {model_name} for version {ver}")
    module = fetch_registered_module(model_name, ver, device="cpu")
    return module.scalers


def run_chainer(
    model_type="flows",
    return_data=True,
    n_data=None,
    call=True,
    cont_rescale_type=None,
    disc_rescale_type=None,
    drop_types=None,
    use_hold=False,
    on_train="bkg",
):
    with open(f"ml/custom/HIGGS/config/{model_type}/data_config.yaml", "r") as file:
        data_conf = yaml.safe_load(file)["data_config"]

    data_conf["preprocessing"]["cont_rescale_type"] = cont_rescale_type

    data_conf["preprocessing"]["disc_rescale_type"] = disc_rescale_type

    if drop_types is None:
        data_conf["feature_selection"]["drop_types"] = ["uni", "disc"]
    else:
        data_conf["feature_selection"]["drop_types"] = drop_types

    data_conf["feature_selection"]["on_train"] = on_train

    data_conf["input_processing"]["hold_mode"] = True
    data_conf["input_processing"]["use_hold"] = use_hold

    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    if n_data is not None:
        f_sel.n_data = n_data

    data_conf["preprocessing"].pop("scaler_label", None)
    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    if call:
        data, selection, scalers = chainer()
    else:
        return chainer

    if return_data:
        return data, selection, scalers
    else:
        return selection, scalers


def get_versions(versions, model_dct):
    if versions is None:
        return [""] * len(model_dct)

    if type(versions) is not list:
        versions = [versions]

    assert len(versions) == len(model_dct), "Number of versions does not match number of models."

    _versions = []
    for ver in versions:
        _versions.append(f"_{ver}")

    return _versions


def sample_from_models(model_dct, N, ver=None, chunks=20, resample=None, return_npy_files=False):
    L.seed_everything(workers=True)

    if type(N) is not int:
        N = int(N)

    versions = get_versions(ver, model_dct)

    if resample is None:
        resample = 1

    samples, npy_files = dict(), dict()

    with torch.no_grad():
        for ver, (model_name, model) in tqdm(
            zip(versions, model_dct.items()), leave=False, desc="Running model sampling"
        ):
            samples[model_name], npy_files[model_name] = [], []

            for i in tqdm(range(resample), leave=False, desc=f"Sampling model {model_name}"):

                if resample == 1 or i == 0:
                    npy_file = f"ml/data/higgs/HIGGS_generated_{model_name}{ver}.npy"
                else:
                    npy_file = f"ml/data/higgs/HIGGS_generated_{model_name}{ver}_{i}.npy"

                npy_files[model_name].append(npy_file)

                M = None
                if os.path.exists(npy_file):
                    cached = np.load(npy_file)
                    M = len(cached)

                    if M >= N:
                        logging.warning(f"Using cached sample {i} for {model_name}.")
                        samples[model_name].append(cached[:N, :])
                        continue
                    else:
                        logging.info(f"Cache {npy_file} did not match N. Resampling.")

                if M is not None:
                    S = N - M
                    logging.info(f"Sampling extra {S} events to match {N} requested.")
                else:
                    S = N

                if "vae" in model_name.lower():
                    sampled = model.sample(S)
                else:
                    sampled = model.sample(S, chunks=chunks)

                if M is not None:
                    samples[model_name].append(np.concatenate([cached, sampled], axis=0))
                else:
                    samples[model_name].append(sampled)

                logging.info(f"Caching sample {i} for {model_name}.")
                np.save(npy_file, samples[model_name][i])

    if return_npy_files:
        return samples, npy_files
    else:
        return samples


def handle_rescale_type(selection, sample, scalers):
    cont_idx = selection[selection["type"] == "cont"].index
    disc_idx = selection[selection["type"] == "disc"].index

    dct = {"cont": None, "disc": None}

    if len(cont_idx) != 0:
        cont_sample = sample[:, cont_idx]
        cont_scaler_lst = scalers["cont"]
        dct["cont"] = (cont_sample, cont_scaler_lst, cont_idx)

    if len(disc_idx) != 0:
        disc_sample = sample[:, disc_idx]
        disc_scaler_lst = scalers["disc"]
        dct["disc"] = (disc_sample, disc_scaler_lst, disc_idx)

    return dct


def scaling_handler(sample, scalers, selection, forward=True):
    if type(sample) is list:
        sample = sample[0]

    if not isinstance(sample, np.ndarray):
        sample = sample.cpu().numpy()

    scale_dct = handle_rescale_type(selection, sample, scalers)

    if scale_dct["cont"] is not None:
        cont_sample, cont_scaler_lst, cont_idx = scale_dct["cont"]

        if forward:
            for scaler in cont_scaler_lst:
                cont_sample = scaler[1].fit_transform(cont_sample)
        else:
            for scaler in cont_scaler_lst[::-1]:
                cont_sample = scaler[1].inverse_transform(cont_sample)

    if scale_dct["disc"] is not None:
        disc_sample, disc_scaler_lst, disc_idx = scale_dct["disc"]

        if forward:
            for scaler in disc_scaler_lst:
                disc_sample = scaler[1].fit_transform(disc_sample)
        else:
            for scaler in disc_scaler_lst[::-1]:
                disc_sample = scaler[1].inverse_transform(disc_sample)

    if scale_dct["cont"] is not None and scale_dct["disc"] is not None:
        sample = np.zeros_like(sample)
        sample[:, cont_idx] = cont_sample
        sample[:, disc_idx] = disc_sample
    elif scale_dct["cont"] is not None:
        sample = cont_sample
    elif scale_dct["disc"] is not None:
        sample = disc_sample
    else:
        raise ValueError("No scaling was performed.")

    return sample


def single_scale_forward(sample, scalers, selection):
    return scaling_handler(sample, scalers, selection, forward=True)


def single_rescale_back(sample, scalers, selection):
    return scaling_handler(sample, scalers, selection, forward=False)


def rescale_back(samples, scalers_dct, selection):
    for model_name, sample in tqdm(samples.items(), leave=False, desc="Rescaling"):
        scaled_sample = []

        for resampled_sample in tqdm(sample, leave=False, desc=f"Rescaling {model_name} samples"):
            scalers = scalers_dct[model_name]
            resampled_sample = single_rescale_back(resampled_sample, scalers, selection)

            scaled_sample.append(resampled_sample)

        samples[model_name] = scaled_sample

    return samples


def histogram_samples(samples_dct, selection, n_bins=20, use_bin_range=False, binning=None):
    if binning is None:
        binning = BINNING

    feature_names = selection[selection["type"] != "label"]["feature"].values

    mc = samples_dct.pop("ref")[0]
    feature_hists, feature_std = dict(), dict()

    feature_dim = mc.shape[1]
    bin_edges = []

    feature_hists["ref"], feature_std["ref"] = dict(), dict()
    for f in range(feature_dim):
        data = filter_array(mc[:, f])

        if binning == "none":
            bin_range = None
        else:
            bin_range = binning[feature_names[f]] if use_bin_range else None

        mc_bin_edges = np.histogram_bin_edges(data, bins=n_bins, range=bin_range)
        bin_edges.append(mc_bin_edges)

        key = feature_names[f]

        feature_hists["ref"][key], _ = np.histogram(mc[:, f], bins=mc_bin_edges)
        feature_std["ref"][key] = np.sqrt(feature_hists["ref"][key])

    for model_name, samples in tqdm(samples_dct.items(), leave=False, desc="Histogramming"):
        feature_hists[model_name], feature_std[model_name] = dict(), dict()

        for sample in samples:
            for f in range(feature_dim):
                key = feature_names[f]

                if key not in feature_hists[model_name]:
                    feature_hists[model_name][key] = []

                # data = filter_array(sample[:, f])
                data = sample[:, f]

                h, _ = np.histogram(data, bins=bin_edges[f])
                feature_hists[model_name][key].append(h)

        for f in range(feature_dim):
            key = feature_names[f]
            feature_std[model_name][key] = np.std(feature_hists[model_name][key], axis=0)
            feature_hists[model_name][key] = np.mean(feature_hists[model_name][key], axis=0)

    bin_edges_dct = dict()
    for f in range(feature_dim):
        key = feature_names[f]
        bin_edges_dct[key] = bin_edges[f]

    return feature_hists, feature_std, bin_edges_dct


def equalize_counts_to_ref(samples_dct, select_outliers=None, ref_str="ref"):
    logging.info("Equalizing counts to reference. Selecting outliers. Removing NaNs.")

    ref = samples_dct[ref_str][0]

    if select_outliers:
        ref = outlier_selection(ref, select_outliers)
        ref = ref[~np.isnan(ref).any(axis=1)]

    N = len(ref)

    count_miss = 0
    for model_name, samples in samples_dct.items():
        if model_name == ref_str:
            continue

        for i, resampled_sample in enumerate(samples):

            if select_outliers is not None:
                resampled_sample = outlier_selection(resampled_sample, select_outliers)

            resampled_sample = resampled_sample[~np.isnan(resampled_sample).any(axis=1)]

            M = len(resampled_sample)

            if N != M:
                count_miss += 1
                logging.info(
                    f"Sample {i} {resampled_sample.shape} from {model_name} did not match ref {ref.shape}. Equalizing!"
                )
                min_shape = min(M, N)
                samples_dct[model_name][i] = resampled_sample[:min_shape, :]
                ref = ref[:min_shape, :]  # this is off for more than 1 model, will also cut to the smallest sample

    samples_dct[ref_str][0] = ref

    if count_miss == 0:
        logging.info("All samples matched reference data.")
    else:
        logging.info(f"Equalized {count_miss} samples to reference data.")

    return samples_dct


def get_binary_classification_scores(x, labels, ax=None, fig=None, device="cpu"):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels).to(torch.int)

    x, labels = x.to(device), labels.to(device)

    if ax is None and fig is None:
        fig, ax = plt.subplots()

    metric = BinaryROC().to(device)
    metric.update(x, labels)
    metric.plot(score=True, ax=ax)

    binary_acc = BinaryAccuracy().to(device)
    binary_acc_value = binary_acc(x, labels).cpu().numpy()

    auroc = BinaryAUROC(thresholds=None).to(device)
    auroc_value = auroc(x, labels).cpu().numpy()

    return ax, fig, {"acc": binary_acc_value, "auroc": auroc_value}


def get_sig_bkg_ref(N):
    ref_data, selection, _ = run_chainer(
        n_data=-1,
        return_data=True,
        on_train=None,
        cont_rescale_type="none",
        model_type="flows",
        use_hold=True,
    )

    label_idx = selection[selection["feature"] == "label"].index

    bkg_mask = (ref_data[:, label_idx] == 0).flatten()
    sig_mask = (ref_data[:, label_idx] == 1).flatten()
    bkg_ref_data = ref_data[bkg_mask]
    sig_ref_data = ref_data[sig_mask]

    label_mask = np.ones(bkg_ref_data.shape[1], dtype=bool)
    label_mask[label_idx] = False
    bkg_ref_data = bkg_ref_data[:, label_mask]
    sig_ref_data = sig_ref_data[:, label_mask]

    bkg_ref_data = bkg_ref_data[:N]
    sig_ref_data = sig_ref_data[:N]

    return bkg_ref_data, sig_ref_data, selection
