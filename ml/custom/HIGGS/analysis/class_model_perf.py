import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.common.stats.two_sample_tests import two_sample_plot
from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.utils import (
    BINNING,
    LABELS_MAP,
    MODEL_MAP,
    get_binary_classification_scores,
    get_model,
    get_scaler,
    np_sigmoid,
    run_chainer,
    sample_from_models,
    single_scale_forward,
)


def plot_ref_data(bkg_ref_data, sig_ref_data, log_scale=False, save_str=""):
    fig, axs = plt.subplots(6, 3, figsize=(13, 19))
    axs = axs.flatten()

    two_sample_plot(
        bkg_ref_data,
        sig_ref_data,
        axs=axs,
        n_bins=40,
        label=["bkg", "sig"],
        bin_range=[list(b) for b in BINNING.values()],
        labels=[label for label in LABELS_MAP.values()],
        log_scale=log_scale,
    )

    logging.info("Saving reference data plot.")

    fig.tight_layout()

    if log_scale:
        fig.savefig(f"{save_str}/sigbkg_reference_log.pdf")
    else:
        fig.savefig(f"{save_str}/sigbkg_reference.pdf")

    plt.close(fig)


def plot_classifier_output(
    out_bkg,
    out_sig,
    out_gen_dct,
    use_sigmoid=False,
    bin_range=(0.0, 1.0),
    save_model_name=None,
    log_scale=False,
    plot_gen=False,
    save_str="",
):
    if save_model_name is None:
        save_model_name = ""
    else:
        save_model_name = f"_{save_model_name}"

    if use_sigmoid:
        save_model_name += "_sigmoid"

    if log_scale:
        save_model_name += "_log"

    if use_sigmoid:
        out_bin_range = (0.02, 1.02)
        out_bkg = np_sigmoid(out_bkg)
        out_sig = np_sigmoid(out_sig)

        for model_name, out_gen in out_gen_dct.items():
            out_gen_dct[model_name] = np_sigmoid(out_gen)
    else:
        out_bin_range = bin_range

    plt.hist(out_bkg, bins=100, histtype="step", label="bkg MC", lw=2, range=out_bin_range, density=True)
    plt.hist(out_sig, bins=100, histtype="step", label="sig MC", lw=2, range=out_bin_range, density=True)

    plt.xlim(out_bin_range)

    if plot_gen:
        for model_name, out_gen in out_gen_dct.items():
            plt.hist(
                out_gen,
                bins=200,
                histtype="step",
                label="bkg " + MODEL_MAP[model_name],
                lw=1.5,
                range=out_bin_range,
            )

    plt.legend(loc="upper left", fontsize=12)
    plt.xlabel("classifier sigmoid output")
    plt.ylabel("density [a.u.]")

    if log_scale:
        plt.yscale("log")

    logging.info("Saving classifier output plot.")

    plt.tight_layout()
    plt.savefig(f"{save_str}/sigbkg_classifier_output{save_model_name}.pdf")
    plt.close()

    # ratio
    ratios = dict()
    n = len(out_bkg) // 2
    for model_name, out_gen in out_gen_dct.items():
        ratios[model_name] = out_bkg[:n] / out_gen[:n]

    baseline_ratio = out_bkg[:n] / out_bkg[n:]

    for model_name, ratio in ratios.items():
        plt.hist(ratio, bins=200, histtype="step", lw=1.5, label=MODEL_MAP[model_name] + " MC / model", range=bin_range)

    plt.hist(baseline_ratio, bins=200, histtype="step", lw=1.5, label="MC / MC", range=bin_range)

    plt.legend(fontsize=12)
    plt.xlabel("classifier output")
    plt.ylabel("ratio")

    if log_scale:
        plt.yscale("log")

    logging.info("Saving classifier ratio plot.")

    plt.tight_layout()
    plt.savefig(f"{save_str}/sigbkg_classifier_ratio{save_model_name}.pdf")
    plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    save_str = "ml/custom/HIGGS/analysis/plots/sig_bkg_class"
    mkdir(save_str)

    cont_rescale_type = "gauss_rank"

    N = 10**6
    chunks = 20
    select_models = [f"MADEMOG_flow_model_{cont_rescale_type}"]  # needs to have same scaling as classifier
    classifier_model = f"BinaryClassifier_sigbkg_{cont_rescale_type}"

    log_scale = False
    use_sigmoid = False

    # prepare reference data
    ref_data, selection, _ = run_chainer(
        n_data=-1,
        return_data=True,
        on_train=None,
        cont_rescale_type="none",
        model_type="dnn",
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

    # plot ref data
    plot_ref_data(bkg_ref_data, sig_ref_data, log_scale=log_scale, save_str=save_str)

    # get classifier model
    classifier = get_model(classifier_model, ver=-1).eval()

    # get classifier scaler
    classifier_scaler = get_scaler(classifier_model)

    # rescale ref data
    bkg_ref_data = single_scale_forward(bkg_ref_data, classifier_scaler, selection)
    sig_ref_data = single_scale_forward(sig_ref_data, classifier_scaler, selection)

    # predict
    with torch.no_grad():
        logging.info("Starting classification for references.")
        out_bkg = classifier(torch.from_numpy(bkg_ref_data).cuda()).cpu().numpy().flatten()
        out_sig = classifier(torch.from_numpy(sig_ref_data).cuda()).cpu().numpy().flatten()

    torch.cuda.empty_cache()

    # filter models
    model_dct = {m: get_model(m).eval() for m in select_models}

    # sample from models
    samples = sample_from_models(model_dct, N, chunks=chunks, resample=1)

    # run predictions on generated
    out_gen_dct = dict()
    with torch.no_grad():
        for model_name, sample in samples.items():

            if isinstance(sample, list):
                sample = sample[0]  # forget about averaging (not important here)
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample).cuda()

            out_gen_dct[model_name] = classifier(sample).cpu().numpy().flatten()

    # plot classifier output distribution
    plot_classifier_output(
        out_bkg,
        out_sig,
        out_gen_dct,
        use_sigmoid=use_sigmoid,
        save_model_name=select_models[0],
        log_scale=log_scale,
        bin_range=(-1, 1) if use_sigmoid else (-0.05, 1.05),
        save_str=save_str,
    )

    # roc and acc
    # baseline
    logging.info("Starting ROC and accuracy calculation.")

    x = np.concatenate([out_bkg, out_sig])
    y = np.concatenate([np.zeros_like(out_bkg), np.ones_like(out_sig)]).astype(int)

    if use_sigmoid:
        x = np_sigmoid(x)

    ax, fig, scores_base = get_binary_classification_scores(x, y, device="cpu")
    auc_base, acc_base = scores_base["auroc"], scores_base["acc"]

    # gen
    aucs, accs = [], []
    for model_name, out_gen in out_gen_dct.items():
        x = np.concatenate([out_gen, out_sig])
        y = np.concatenate([np.zeros_like(out_gen), np.ones_like(out_sig)]).astype(int)

        if use_sigmoid:
            x = np_sigmoid(x)

        ax, fig, scores = get_binary_classification_scores(x, y, ax=ax, fig=fig, device="cpu")

        auc, acc = scores["auroc"], scores["acc"]
        aucs.append(auc)
        accs.append(acc)

    ax.set_title("")
    ax.plot([0, 1], [0, 1], color="black", lw=1.5, ls="--")

    labels = [f"Baseline AUC: {auc_base:.4f}, ACC: {acc_base:.4f}"]
    labels += [f"{MODEL_MAP[m]} AUC: {auc:.4f}, ACC: {acc:.4f}" for m, auc, acc in zip(select_models, aucs, accs)]

    ax.legend(labels)
    ax.grid(True, linestyle="-", lw=0.5, color="black", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(f"{save_str}/sigbkg_classifier_roc_{cont_rescale_type}.pdf")
    plt.close(fig)
