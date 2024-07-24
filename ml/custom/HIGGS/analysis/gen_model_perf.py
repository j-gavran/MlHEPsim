import copy
import logging

import corner
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from uncertainties import unumpy as unp

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import errorbar_plot, set_size, style_setup
from ml.custom.HIGGS.analysis.utils import (
    BINNING,
    LABELS_MAP,
    MODEL_MAP,
    equalize_counts_to_ref,
    get_model,
    get_scaler,
    histogram_samples,
    rescale_back,
    run_chainer,
    sample_from_models,
)

PERF_BINNING = {
    "lepton pT": (0, 5),
    "lepton eta": (-4, 4),
    "missing energy": (0, 5),
    "jet1 pT": (0, 4),
    "jet1 eta": (-4, 4),
    "jet2 pT": (0, 4.5),
    "jet2 eta": (-4, 4),
    "jet3 pT": (0, 4),
    "jet3 eta": (-4, 4),
    "jet4 pT": (0, 4),
    "jet4 eta": (-4, 4),
    "m jj": (0, 5),
    "m jjj": (0, 4),
    "m lv": (0.9, 2.5),
    "m jlv": (0, 4),
    "m bb": (0, 5),
    "m wbb": (0, 4),
    "m wwbb": (0, 3.5),
}


def step_hist_plot(
    ax,
    hist,
    bin_edges,
    label=None,
    xlabel=None,
    ylabel=None,
    legend_loc=None,
    histtype="step",
    **kwargs,
):
    hep.histplot(
        hist,
        bin_edges,
        ax=ax,
        histtype=histtype,
        label=label if label is not None else None,
        **kwargs,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel, loc="center")

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if label is not None:
        if legend_loc:
            ax.legend(loc=legend_loc)
        else:
            ax.legend()

    return ax


def N_sample_plot(
    feature_hists,
    bin_edges,
    axs,
    label=None,
    labels=None,
    **kwargs,
):
    mc = feature_hists.pop("ref")

    for f, (feature_name, hist) in enumerate(mc.items()):
        step_hist_plot(
            axs[f],
            hist,
            bin_edges[feature_name],
            label="MC" if f == 0 else None,
            color="C7",
            histtype="fill",
            alpha=0.5,
            **kwargs,
        )

    for i, (_, hists_dct) in enumerate(feature_hists.items()):
        for f, (feature_name, hist) in enumerate(hists_dct.items()):
            step_hist_plot(
                axs[f],
                hist,
                bin_edges[feature_name],
                label=labels[f] if f == 0 else None,
                color=f"C{i}",
                histtype="step",
                **kwargs,
            )

            if f == 0:
                axs[f].set_ylabel("$N$")

            if f == 0 and label is not None:
                axs[f].legend(["MC"] + label, fontsize=14)

            if labels is not None:
                axs[f].set_xlabel(labels[f], size=20)

    return axs


def binned_perf_plot(
    axs,
    feature_hists,
    feature_std,
    bin_edges,
    latex_map=None,
    model_map=None,
    ylim=None,
    max_y=None,
    max_yerr=None,
    use_ecdf=False,
    all_ratio=False,
):
    assert (use_ecdf and not all_ratio) or (all_ratio and not use_ecdf) or (not use_ecdf and not all_ratio)

    mc_hist_dct = feature_hists.pop("ref")
    mc_std_dct = feature_std.pop("ref")

    for c, (model_name, hists_dct) in enumerate(feature_hists.items()):
        for f, (feature_name, feature_hist) in enumerate(hists_dct.items()):
            feature_mc = mc_hist_dct[feature_name].astype(np.float32)
            std_mc = mc_std_dct[feature_name].astype(np.float32)

            std = feature_std[model_name][feature_name]

            feature_mc[feature_mc == 0.0] = np.nan

            mc_unp = unp.uarray(feature_mc, std_mc)
            feature_unp = unp.uarray(feature_hist, std)

            ratio = feature_unp / mc_unp
            y = unp.nominal_values(ratio)
            yerr = unp.std_devs(ratio)

            if max_yerr is not None:
                yerr_mask = yerr > max_yerr
                yerr[yerr_mask], y[yerr_mask] = np.nan, np.nan

            if max_y is not None:
                y_mask = y > max_y
                yerr[y_mask], y[y_mask] = np.nan, np.nan

            feature_bin_edges = bin_edges[feature_name]

            dx = 0.5 * np.diff(feature_bin_edges)
            x = feature_bin_edges[:-1] + dx

            axs[f].set_xlim((feature_bin_edges[0], feature_bin_edges[-1]))

            errorbar_plot(
                ax=axs[f],
                x=x,
                y=y,
                xerr=0,
                yerr=yerr,
                color=f"C{c}",
                label=model_map[model_name] if f == 0 else None,
            )

            if f == 0:
                axs[f].legend(loc="upper left", fontsize=14)

            if c == 0:
                axs[f].axhline(1, color="k", ls="--", lw=1)

                if ylim is not None:
                    axs[f].set_ylim(ylim)

                axin = axs[f].inset_axes(bounds=[0, -0.42, 1, 0.3])

                no_nan_feature_mc = np.nan_to_num(feature_mc)
                ecdf = np.cumsum(no_nan_feature_mc) / np.sum(no_nan_feature_mc)

                if "eta" not in latex_map[feature_name]:
                    x_vline_1 = x[ecdf > 0.9973][0]
                    axs[f].axvline(x_vline_1, color="r", ls="--", lw=1.5, zorder=100)
                    axs[f].text(
                        x_vline_1 + 0.1,
                        ylim[1] + 0.01,
                        r"$3\sigma$ tail events",
                        rotation=0,
                        fontsize=11,
                        c="r",
                        zorder=100,
                    )

                if use_ecdf:
                    axin.plot(
                        x,
                        ecdf,
                        color="k",
                    )
                    axin.set_ylim(-0.05, 1.05)

                elif all_ratio:
                    ratio = feature_mc / np.sum(no_nan_feature_mc)

                    axin.scatter(x, ratio, color="k", s=16)
                    axin.plot(x, ratio, color="k", lw=1.2)

                    if "eta" not in latex_map[feature_name]:
                        axin.axvline(x_vline_1, color="r", ls="--", lw=1.5, zorder=100)

                    axin.set_yscale("log")

                else:
                    step_hist_plot(
                        ax=axin,
                        hist=feature_mc,
                        bin_edges=feature_bin_edges,
                        color="k",
                        histtype="fill",
                        alpha=0.25,
                    )
                    step_hist_plot(
                        ax=axin,
                        hist=feature_mc,
                        bin_edges=feature_bin_edges,
                        color="k",
                        histtype="step",
                        lw=1,
                    )
                    axin.set_yscale("log")

                axin.set_xlim(axs[f].get_xlim())

                axin.xaxis.set_tick_params(pad=10)

                axs[f].tick_params(labelbottom=False)

                axin.set_xlabel(latex_map[feature_name])
                axs[f].set_xlabel("")

                if use_ecdf:
                    axin.set_ylabel("ECDF")
                elif all_ratio:
                    axin.set_ylabel("bin / all")
                else:
                    axin.set_ylabel("MC")

                axs[f].set_ylabel("generated / MC")

    return axs


def plot_N_samples(feature_hists, bin_edges, selection, postfix="", log_scale=True, save_dir=""):
    fig, axs = plt.subplots(6, 3, figsize=(13, 21))
    axs = axs.flatten()

    labels = selection[selection["type"] != "label"]["feature"].values

    N_sample_plot(
        feature_hists,
        bin_edges,
        axs,
        label=[MODEL_MAP[i] for i in list(samples.keys())],
        labels=[LABELS_MAP[i] for i in labels],
        lw=1,
    )

    if log_scale:
        for ax in axs:
            ax.set_yscale("log")

    if postfix is None:
        postfix = ""

    if len(postfix) > 1:
        postfix = "_" + postfix

    fig.tight_layout()

    logging.info("Saving N_gen plot!")

    if log_scale:
        fig.savefig(f"{save_dir}/N_gen{postfix}_log.pdf")
    else:
        fig.savefig(f"{save_dir}/N_gen{postfix}.pdf")

    plt.close(fig)


def plot_binned_ratios(
    feature_hists,
    feature_std,
    bin_edges,
    postfix="",
    ylim=None,
    ecdf=False,
    all_ratio=False,
    save_dir="",
    **kwargs,
):
    fig, axs = plt.subplots(6, 3, figsize=(16, 32))
    axs = axs.flatten()

    binned_perf_plot(
        axs,
        feature_hists,
        feature_std,
        bin_edges,
        ylim=ylim,
        latex_map=LABELS_MAP,
        model_map=MODEL_MAP,
        use_ecdf=ecdf,
        all_ratio=all_ratio,
        **kwargs,
    )

    logging.info("Saving binned performance plot!")

    fig.tight_layout()

    if postfix is None:
        fig.savefig(f"{save_dir}/binned_perf_plot.pdf")
    else:
        fig.savefig(f"{save_dir}/binned_perf_plot_{postfix}.pdf")

    plt.close(fig)


def plot_correlations(
    samples,
    model_name,
    selection,
    postfix=None,
    max_plot_points=10000,
    sample_i=0,
    use_bin_range=False,
    save_dir="",
):
    labels = selection[selection["type"] != "label"]["feature"].values

    labels = [LABELS_MAP[i] for i in labels]

    mc = samples["ref"][sample_i][:max_plot_points, :]
    gen = samples[model_name][sample_i][:max_plot_points, :]

    logging.info(f"Making corner plot for {model_name} and {max_plot_points} events.")

    fig = corner.corner(
        mc,
        labels=labels,
        hist_kwargs={"histtype": "step"},
        range=list(BINNING.values()) if use_bin_range else None,
        plot_contours=False,
        plot_density=False,
        plot_datapoints=True,
        fill_contours=False,
        color="C1",
        marker="x",
        contour_kwargs={"linestyles": "-", "linewidths": 1},
    )

    fig = corner.corner(
        gen,
        labels=labels,
        fig=fig,
        hist_kwargs={"histtype": "step"},
        range=list(BINNING.values()) if use_bin_range else None,
        plot_contours=False,
        plot_density=False,
        plot_datapoints=True,
        fill_contours=False,
        color="C0",
        contour_kwargs={"linestyles": "--", "linewidths": 1},
    )

    plt.legend(["MC", "gen"])

    if postfix is None:
        postfix = ""

    logging.info("Saving corner plot!")

    fig.tight_layout()
    fig.savefig(f"{save_dir}/corner_plot_{postfix}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    save_dir = "ml/custom/HIGGS/analysis/plots/gen_model_perf"
    mkdir(save_dir)

    select_models = [
        # "RealNVP_flow_model_gauss_rank",
        # "Glow_flow_model_gauss_rank",
        # "rqsplines_flow_model_gauss_rank",
        # "MAF_flow_model_gauss_rank",
        # "MAFMADEMOG_flow_model_gauss_rank",
        "MADEMOG_flow_model_gauss_rank_best",
        # "MADEMOG_flow_model_gauss_rank_mini",  # 10 layers with 2x512 nodes
    ]

    # highr interp_limit and smaller epsilon -> better histograms in the tails
    # select_models = [
    #     "RealNVP_flow_model_gauss_rank_interp",
    #     "Glow_flow_model_gauss_rank_interp",
    #     "rqsplines_flow_model_gauss_rank_interp",
    #     "MAF_flow_model_gauss_rank_interp",
    #     "MAFMADEMOG_flow_model_gauss_rank_interp",
    #     "MADEMOG_flow_model_gauss_rank_interp",
    # ]

    # sample settings
    N = 2 * 10**6
    n_bins = 50
    chunks = 20
    resample = 1

    # plot settings
    select_outliers = None  # "inside"
    log_scale = True  # false for correct shape plot !
    use_bin_range = True
    plot_corr = False

    # prepare reference data
    ref_data, selection, _ = run_chainer(
        n_data=-1,
        return_data=True,
        cont_rescale_type="none",
        disc_rescale_type="none",
        drop_types=["uni", "disc"],
        use_hold=True,
    )

    label_idx = selection[selection["feature"] == "label"].index

    # analysis on background data
    bkg_mask = (ref_data[:, label_idx] == 0).flatten()
    bkg_ref_data = ref_data[bkg_mask]

    label_mask = np.ones(bkg_ref_data.shape[1], dtype=bool)
    label_mask[label_idx] = False
    bkg_ref_data = bkg_ref_data[:, label_mask]

    bkg_ref_data = bkg_ref_data[:N]

    # filter models
    model_dct = {select_model: get_model(select_model, ver=-1).eval() for select_model in select_models}

    # fetch scalers
    scalers_dct = {select_model: get_scaler(select_model, ver=-1) for select_model in select_models}

    # sample from models
    samples = sample_from_models(model_dct, N, chunks=chunks, resample=resample)

    # rescale back
    samples = rescale_back(samples, scalers_dct, selection)

    # add reference data to samples
    samples["ref"] = [bkg_ref_data]

    # make sure all samples have the same number of events
    equalize_counts_to_ref(samples, select_outliers=select_outliers)

    # histogram samples
    feature_hists, feature_std, bin_edges = histogram_samples(
        copy.deepcopy(samples),
        selection,
        n_bins=n_bins,
        use_bin_range=use_bin_range,
        binning=PERF_BINNING if not log_scale else "none",
    )

    # plot correlations with corner plot
    if plot_corr:
        for select_model in select_models:
            plot_correlations(
                samples,
                select_model,
                selection,
                postfix=select_model,
                use_bin_range=use_bin_range,
                save_dir=save_dir,
            )

    # plot distributions of features
    plot_N_samples(
        copy.deepcopy(feature_hists),
        bin_edges,
        selection,
        postfix=select_outliers,
        log_scale=log_scale,
        save_dir=save_dir,
    )

    # plot ratios of binned distributions
    plot_binned_ratios(
        feature_hists,
        feature_std,
        bin_edges,
        postfix=select_outliers,
        ecdf=False,
        all_ratio=True,
        max_y=3.0,
        max_yerr=0.1,
        ylim=(0.8, 1.2),
        save_dir=save_dir,
    )
