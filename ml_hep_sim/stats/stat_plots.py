import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_hep_sim.plotting.matplotlib_setup import set_size
from ml_hep_sim.plotting.style import style_setup as _style_steup
from ml_hep_sim.stats.stat_utils import ecdf


def style_setup(color):
    _style_steup(color)


def two_sample_plot(
    A,
    B,
    axs,
    n_bins="auto",
    label=None,
    labels=None,
    log_scale=False,
    bin_range=None,
    xlim=None,
    ylim=None,
    titles=None,
    **kwargs,
):
    assert A.shape[1] == B.shape[1]

    n_features = A.shape[1]

    if bin_range is not None:
        if not any(isinstance(el, list) for el in bin_range):
            bin_range = [bin_range] * n_features

    if torch.is_tensor(A) and torch.is_tensor(B):
        A, B = A.numpy(), B.numpy()

    combined_sample = np.concatenate([A, B])

    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(
            combined_sample[:, feature], bins=n_bins, range=bin_range[feature] if bin_range else None
        )

        axs[feature].hist(A[:, feature], bins=bin_edges, histtype="step", **kwargs)
        axs[feature].hist(B[:, feature], bins=bin_edges, histtype="step", **kwargs)

        if feature == 0 and label is not None:
            axs[feature].legend(label)

        if labels is not None:
            axs[feature].set_xlabel(labels[feature], size=15)

        if log_scale:
            axs[feature].set_yscale("log")

        if xlim:
            axs[feature].set_xlim(xlim[feature])

        if ylim:
            if ylim[feature] is not None:
                axs[feature].set_ylim(ylim[feature])

        if titles is not None:
            axs[feature].set_title(titles[feature], size=15, loc="right")

    return axs


def N_sample_plot(
    samples,
    axs,
    n_bins="auto",
    label=None,
    labels=None,
    log_scale=False,
    bin_range=None,
    xlim=None,
    titles=None,
    last_c="C7",
    **kwargs,
):
    # TODO: make better
    n_features = samples[0].shape[1]

    if bin_range is not None:
        if not any(isinstance(el, list) for el in bin_range):
            bin_range = [bin_range] * n_features

    combined_sample = np.concatenate(samples)

    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(
            combined_sample[:, feature], bins=n_bins, range=bin_range[feature] if bin_range else None
        )

        for i, sample in enumerate(samples):
            if i == len(samples) - 1:
                axs[feature].hist(sample[:, feature], bins=bin_edges, histtype="step", color=last_c, lw=2, alpha=0.7)
            else:
                axs[feature].hist(sample[:, feature], bins=bin_edges, histtype="step", **kwargs)

        if feature == 0 and label is not None:
            axs[feature].legend(label)

        if labels is not None:
            axs[feature].set_xlabel(labels[feature], size=15)

        if log_scale:
            axs[feature].set_yscale("log")

        if xlim:
            axs[feature].set_xlim(xlim[feature])

        if titles is not None:
            axs[feature].set_title(titles[feature], size=15, loc="right")

    return axs


def two_sample_ecdf_plot(A, B, axs, label=None, labels=None, **kwargs):
    n_features = A.shape[1]

    for feature in range(n_features):
        ecdf_A, ecdf_B = ecdf(A[:, feature]), ecdf(B[:, feature])

        axs[feature].plot(ecdf_A[0], ecdf_A[1], **kwargs)
        axs[feature].plot(ecdf_B[0], ecdf_B[1], ls="--", **kwargs)

        if feature == 0:
            axs[feature].legend(["sample A", "sample B"] if label is None else label)

        if labels:
            axs[feature].set_title(labels[feature])

    return axs


def classifier_comparison_plot(
    A, B, n_bins="auto", text_dict=None, lr_margin=None, save_loc=None, title="izhod klasifikatorja", log_scale=False
):
    fig, axs = plt.subplots(1, 1, figsize=set_size(subplots=(1, 1), fraction=1.25))

    combined_sample = np.concatenate([A, B])

    if lr_margin:
        combined_sample = combined_sample[(lr_margin[0] < combined_sample) & (combined_sample < lr_margin[1])]

    bin_edges = np.histogram_bin_edges(combined_sample, bins=n_bins)

    axs.hist(A, bins=bin_edges, histtype="step", lw=2)
    axs.hist(B, bins=bin_edges, histtype="step", lw=2)
    axs.set_xlabel(title)
    axs.set_ylabel("dogodki")
    axs.legend(["referenca", "ML"], loc="upper left")

    if log_scale:
        axs.set_yscale("log")

    if text_dict:
        s = ""
        for k, v in text_dict.items():
            s += f"{k}{v:.3e}$\quad$"

        axs.set_title(s)

    plt.tight_layout()

    if save_loc:
        plt.savefig(save_loc + ".pdf")


def single_plot_scan_stat_results(v, N_range, save_loc=None, xlim=None, v_err=None):
    chi2_res, ks_res, chi2_res_crit, ks_res_crit = v

    if v_err is not None:
        chi2_err, ks_err = v_err

    fig, axs1 = plt.subplots(1, 1, figsize=set_size(subplots=(1, 1), fraction=1.25))
    axs1.set_xlabel(r"$N$", loc="center")

    axs1.plot(N_range, chi2_res, c="C0")
    axs1.scatter(N_range, chi2_res, c="C0", s=30)

    axs1.scatter(N_range, chi2_res_crit, color="C0", marker="x", zorder=10)

    axs1.set_ylabel(r"$\chi^2$", color="C0")
    # axs1.tick_params(axis="y", color="C0")

    axs2 = axs1.twinx()
    axs2.plot(N_range, ks_res, c="C1")
    axs2.scatter(N_range, ks_res, c="C1", s=30)

    axs2.scatter(N_range, ks_res_crit, color="C1", marker="x", zorder=10)

    axs2.set_ylabel(r"ks", color="C1")
    # axs2.tick_params(axis="y", color="C1")

    if v_err:
        axs1.errorbar(N_range, chi2_res, yerr=chi2_err, c="C0", capsize=4)
        axs2.errorbar(N_range, ks_res, yerr=ks_err, c="C1", capsize=4)

    axs2.set_yscale("log")

    if xlim is not None:
        axs1.set_xlim(xlim)

    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc + ".pdf")


def local_density_comparison_plot(x, y, text_dict=None, lr_margin=None, save=None):
    fig, axs = plt.subplots(1, 1, figsize=set_size(subplots=(1, 1), fraction=1.25))

    axs.plot(range(len(x)), x, c="C0")
    axs.plot(range(len(y)), y, c="C1")

    axs.set_xlabel(r"$r$")
    axs.set_ylabel(r"$K$")
    axs.legend(["reference", "generated"])

    axs.set_xlim(lr_margin)

    if text_dict:
        s = ""
        for k, v in text_dict.items():
            s += f"{k}{v:.3e}$\quad$"

        axs.set_title(s)

    plt.tight_layout()
    if save:
        plt.savefig(save)

    plt.close()
