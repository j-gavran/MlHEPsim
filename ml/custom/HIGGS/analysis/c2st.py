import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC

from ml.common.stats.binary_confmat import binary_confusion_matrix
from ml.common.stats.c2st import GeneratedProcessor, TwoSampleBuilder
from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.utils import (
    BINNING,
    LABELS_MAP,
    MODEL_MAP,
    get_model,
    run_chainer,
)
from ml.custom.HIGGS.main_c2st import get_generated_data


def c2st_perf_plot(
    labels,
    c2st_data_XY,
    mask_1,
    mask_2,
    perf_idx,
    n_bins=100,
    perf_label="",
    axs=None,
    fig=None,
    do_split=False,
    bin_edges=None,
):
    if axs is None and fig is None:
        fig, axs = plt.subplots(6, 3, figsize=(13, 19))
        axs = axs.flatten()

    if bin_edges is None:
        bin_edges = [None for _ in range(len(labels))]

    if do_split:
        data_XY_1 = c2st_data_XY[perf_idx & mask_1]
        data_XY_2 = c2st_data_XY[perf_idx & mask_2]

        for i, label in enumerate(labels):
            if bin_edges[i] is None:
                bin_edges[i] = np.histogram_bin_edges(np.concatenate((data_XY_1[:, i], data_XY_2[:, i])), bins=n_bins)

            axs[i].hist(data_XY_1[:, i], bins=bin_edges[i], histtype="step", label=f"{perf_label} MC", lw=2)
            axs[i].hist(data_XY_2[:, i], bins=bin_edges[i], histtype="step", label=f"{perf_label} GEN", lw=2)

            if i == 0:
                axs[i].legend()

            axs[i].set_xlabel(LABELS_MAP[label])
            axs[i].set_ylabel("$N$")

    else:
        c2st_data_XY = c2st_data_XY[perf_idx]

        for i, label in enumerate(labels):
            if bin_edges[i] is None:
                bin_edges[i] = np.histogram_bin_edges(c2st_data_XY[:, i], bins=n_bins, range=BINNING[label])

            axs[i].hist(c2st_data_XY[:, i], bins=bin_edges[i], histtype="step", label=f"{perf_label}", lw=2)

            if i == 0:
                axs[i].legend()

            axs[i].set_xlabel(LABELS_MAP[label])
            axs[i].set_ylabel("$N$")

    return bin_edges


def plot_mis_id(
    X,
    Y,
    selection,
    scalers,
    tp_idx,
    fp_idx,
    tn_idx,
    fn_idx,
    n_bins=70,
    is_closure=False,
    do_split=False,
    save_dir="",
):
    event_XY = np.concatenate((X.cpu().numpy(), Y.cpu().numpy()))

    X_scaler = scalers["X"]["cont"]

    if type(X_scaler) is not list:
        X_scaler = [X_scaler]

    for scaler in X_scaler[::-1]:
        try:
            event_XY = scaler[1].inverse_transform(event_XY)
        except TypeError:
            event_XY = scaler.inverse_transform(event_XY)

    split_at = len(event_XY) // 2

    labels = selection[selection["type"] != "label"]["feature"].values

    mask_1 = np.concatenate((np.ones(split_at), np.zeros(split_at))).astype(bool)
    mask_2 = np.concatenate((np.zeros(split_at), np.ones(split_at))).astype(bool)

    fig, axs = plt.subplots(6, 3, figsize=(13, 19))
    axs = axs.flatten()

    b = []
    for i in range(len(labels)):
        be = np.histogram_bin_edges(event_XY[:, i], bins=n_bins, range=BINNING[labels[i]])
        b.append(be)

    perf_labels = ["TP", "FP", "TN", "FN"]
    indices = [tp_idx, fp_idx, tn_idx, fn_idx]

    for perf_label, idx in zip(perf_labels, indices):
        c2st_perf_plot(
            labels,
            event_XY,
            mask_1,
            mask_2,
            perf_idx=idx,
            perf_label=perf_label,
            axs=axs,
            fig=fig,
            bin_edges=b,
            do_split=do_split,
        )

    fig.tight_layout()

    if is_closure:
        fig.savefig(f"{save_dir}/c2st_perf_closure.pdf")
    else:
        fig.savefig(f"{save_dir}/c2st_perf.pdf")

    plt.close(fig)


def setup_c2st_procedure(
    N,
    select_model,
    cont_rescale_type,
    closure_classifier=False,
    sample_i=0,
    sort_var=None,
    get_baseline=False,
):
    """Setup c2st procedure for given model and data.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    select_model : str
        Model name to use for c2st.
    cont_rescale_type : str
        Type of rescaling to use for continuous variables.
    closure_classifier : bool
        Whether to use separate classifier for closure test. Needs to be trained separately, by default False.
    sample_i : int
        Resampled data file index, if multiple files are generated, by default 0.
    sort_var : str
        Variable to sort data by, by default None.
    get_baseline : bool
        Whether to get and return baseline data for closure test, by default False.

    Other parameters
    ----------------
    X ... holdout MC 1
    Y ... gen
    X_base ... holdout MC 1
    Y_base ... holdout MC 2

    c2st_X ... holdout MC 1 after classifier
    c2st_Y ... gen after classifier
    c2st_X_base ... holdout MC 1 after classifier
    c2st_Y_base ... holdout MC 2 after classifier

    selection ... feature selection
    scalers ... scalers used for data preprocessing

    Note
    ----
    1. Start with all data D
    2. Split D into holdout sets D_1, D_2
    3. Use D_1 for flow training
    4. Make two sets from D_2, S_1 and S_2 (split in half, 50% each)
        S_1 has X_1 (label 0) and Y_1 (label 1) sets
        S_2 has X_2 (label 0) and Y_2 (label 1) sets
    5. Train c2st on S_1 and use S_2 as holdout for testing

    D -> D_1, D_2
              D_2 -> S_1 -> X_1, Y_1 (train c2st)
                  -> S_2 -> X_2, Y_2 (infer c2st)

    Returns
    -------
    tuple
        X, Y, c2st_X, c2st_Y, label_X, label_Y, selection, scalers, X_base, Y_base, c2st_X_base, c2st_Y_base
    tuple
        X, Y, c2st_X, c2st_Y, label_X, label_Y, selection, scalers if get_baseline is False

    """
    chainer = run_chainer(
        model_type="flows",
        return_data=False,
        n_data=N,
        call=False,
        cont_rescale_type=cont_rescale_type,
    )

    file_dir, file_name = get_generated_data(select_model=select_model, N=N, chunks=20, sample_i=sample_i)
    gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)

    two_samples_proc = TwoSampleBuilder(
        processor_X=chainer,
        processor_Y=gen_proc,
        add_label_X=False,
        add_label_Y=True,
        hold_out_ratio=0.5,
    )

    closure_chainer = copy.deepcopy(chainer)
    closure_chainer.N = 2 * N

    two_samples_proc_closure = TwoSampleBuilder(
        processor_X=closure_chainer,
        processor_Y=None,
        add_label_X=False,
        add_label_Y=True,
        hold_out_ratio=0.5,
    )

    c2st_classifier = get_model(f"BinaryClassifier_{select_model}_c2st_gen_model_all").eval()

    if closure_classifier:
        c2st_classifier_closure = get_model("BinaryClassifier_mc_c2st_model").eval()

    # get hold out data
    two_samples_proc()
    selection, scalers = two_samples_proc.selection, two_samples_proc.scalers
    hold_XY = two_samples_proc.hold_XY

    X, Y = hold_XY["X"], hold_XY["Y"]
    X, Y = torch.from_numpy(X).cuda(), torch.from_numpy(Y).cuda()

    # split labels and data
    X, label_X, Y, label_Y = X[:, :-1], X[:, -1], Y[:, :-1], Y[:, -1]

    if sort_var is not None:
        sort_var_idx = selection[selection["feature"] == sort_var].index[0]

        sort_idx_X = X[:, sort_var_idx].argsort(dim=0, descending=True)
        sort_idx_Y = Y[:, sort_var_idx].argsort(dim=0, descending=True)

        X, label_X = X[sort_idx_X], label_X[sort_idx_X]
        Y, label_Y = Y[sort_idx_Y], label_Y[sort_idx_Y]

    # run predictions
    with torch.no_grad():
        c2st_X = c2st_classifier(X)
        c2st_Y = c2st_classifier(Y)

    if get_baseline:
        # get closure test data
        two_samples_proc_closure()
        hold_XY_closure = two_samples_proc_closure.hold_XY

        X_base, Y_base = hold_XY_closure["X"], hold_XY_closure["Y"]
        X_base, Y_base = torch.from_numpy(X_base).cuda(), torch.from_numpy(Y_base).cuda()

        # split labels and data
        X_base, Y_base = X_base[:, :-1], Y_base[:, :-1]

        # run predictions
        with torch.no_grad():
            # use separate classifier for closure test
            if closure_classifier:
                c2st_X_base = c2st_classifier_closure(X_base)
                c2st_Y_base = c2st_classifier_closure(Y_base)
            # use the same classifier for closure test
            else:
                c2st_X_base = c2st_classifier(X_base)
                c2st_Y_base = c2st_classifier(Y_base)

        return X, Y, c2st_X, c2st_Y, label_X, label_Y, selection, scalers, X_base, Y_base, c2st_X_base, c2st_Y_base

    else:
        return X, Y, c2st_X, c2st_Y, label_X, label_Y, selection, scalers


def get_confmat_scores(c2st_X, c2st_Y, label_X, label_Y, threshold=0.0):
    np_c2st_X, np_c2st_Y = c2st_X.cpu().numpy().flatten(), c2st_Y.cpu().numpy().flatten()
    np_label_X, np_label_Y = label_X.cpu().numpy().flatten(), label_Y.cpu().numpy().flatten()

    # compute confusion matrix and binary accuracy
    # we used BCEWithLogitsLoss or BCELoss, so we need to apply sigmoid to the output
    c2st_data_XY = np.concatenate((np_c2st_X, np_c2st_Y))
    label_XY = np.concatenate((np_label_X, np_label_Y))

    binary_acc = BinaryAccuracy().to("cpu")
    binary_acc_value = binary_acc(torch.from_numpy(c2st_data_XY), torch.from_numpy(label_XY)).numpy()

    # threshold should be 0.0 if logits are used as output
    confmat, confmat_idx = binary_confusion_matrix(label_XY, c2st_data_XY, threshold=threshold)

    tp_idx, fp_idx, tn_idx, fn_idx = confmat_idx["tp"], confmat_idx["fp"], confmat_idx["tn"], confmat_idx["fn"]

    print(f"confmat:\n{confmat}")
    print(f"binary_acc: {binary_acc_value}")

    return tp_idx, fp_idx, tn_idx, fn_idx, (confmat, binary_acc_value)


def plot_ROC(c2st_X, c2st_Y, label_X, label_Y, closure_test=False, ax=None, fig=None, save=False, save_dir=""):
    c2st_XY = torch.cat((c2st_X, c2st_Y)).squeeze()
    label_XY = torch.cat((label_X, label_Y)).to(torch.int)

    if ax is None and fig is None:
        fig, ax = plt.subplots()

    metric = BinaryROC().to("cuda")
    metric.update(c2st_XY, label_XY)
    metric.plot(score=True, ax=ax)

    binary_acc = BinaryAccuracy().to("cuda")
    binary_acc_value = binary_acc(c2st_XY, label_XY).cpu().numpy()

    auroc = BinaryAUROC(thresholds=None).to("cuda")
    auroc_value = auroc(c2st_XY, label_XY).cpu().numpy()

    print(f"AUROC: {auroc_value}")

    if save:
        fig.tight_layout()

        if not closure_test:
            fig.savefig(f"{save_dir}/c2st_roc.pdf")
        else:
            fig.savefig(f"{save_dir}/c2st_roc_closure.pdf")

        plt.close(fig)

    return auroc_value, binary_acc_value


def plot_confmat(confmat, save_dir="", postfix=""):
    sns.heatmap(confmat / np.sum(confmat), annot=True, fmt=".4f", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0.5, 1.5], ["MC", "gen"])
    plt.yticks([0.5, 1.5], ["MC", "gen"])
    plt.title("c2st confusion matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confmat_{select_model}{postfix}.pdf")
    plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    save_dir = "ml/custom/HIGGS/analysis/plots/c2st_rocs"
    mkdir(save_dir)

    N = 2 * 10**6

    select_model = "RealNVP_flow_model_gauss_rank"
    cont_rescale_type = "gauss_rank"

    # get c2st ingredients
    X, Y, c2st_X, c2st_Y, label_X, label_Y, selection, scalers, X_base, Y_base, c2st_X_base, c2st_Y_base = (
        setup_c2st_procedure(
            N,
            select_model,
            cont_rescale_type=cont_rescale_type,
            get_baseline=True,
        )
    )

    # get confusion matrix with indices and binary accuracy
    tp_idx, fp_idx, tn_idx, fn_idx, conf_scores = get_confmat_scores(c2st_X, c2st_Y, label_X, label_Y, threshold=0.5)

    label_X_base, label_Y_base = torch.zeros_like(c2st_X_base), torch.ones_like(c2st_Y_base)
    _, _, _, _, conf_scores_base = get_confmat_scores(
        c2st_X_base, c2st_Y_base, label_X_base, label_Y_base, threshold=0.5
    )

    # plot confusion matrix
    confmat, confmat_base = conf_scores[0], conf_scores_base[0]
    plot_confmat(confmat, save_dir=save_dir)
    plot_confmat(confmat_base, save_dir=save_dir, postfix="_base")

    # plot event mis-identification distributions
    plot_mis_id(X, Y, selection, scalers, tp_idx, fp_idx, tn_idx, fn_idx, is_closure=False, save_dir=save_dir)

    # ROC curves
    roc_models = [
        "RealNVP_flow_model_gauss_rank",
        "Glow_flow_model_gauss_rank",
        "rqsplines_flow_model_gauss_rank",
        "MAF_flow_model_gauss_rank",
        "MAFMADEMOG_flow_model_gauss_rank",
        "MADEMOG_flow_model_gauss_rank_mini",
    ]
    cont_rescale_types = ["gauss_rank"] * len(roc_models)

    fig, ax = plt.subplots()
    aucs, accs = [], []
    for _select_model, rescale_type in zip(roc_models, cont_rescale_types):
        logging.info(f"[red]Running ROC for {_select_model}...[/red]")
        X, Y, c2st_X, c2st_Y, label_X, label_Y, selection, scalers = setup_c2st_procedure(
            N,
            _select_model,
            cont_rescale_type=rescale_type,
            get_baseline=False,
        )
        auc, acc = plot_ROC(c2st_X, c2st_Y, label_X, label_Y, ax=ax, fig=fig, save=False)
        aucs.append(auc)
        accs.append(acc)

    ax.set_title("")
    ax.plot([0, 1], [0, 1], color="black", lw=1.5, ls="--")

    legend_labels = [f"{MODEL_MAP[m]} AUC: {auc:.4f}, ACC: {acc:.4f}" for m, auc, acc in zip(roc_models, aucs, accs)]
    legend_labels += ["Baseline AUC: 0.5"]

    ax.legend(legend_labels, fontsize=11, loc="upper left")
    ax.grid(True, linestyle="-", lw=0.5, color="black", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/c2st_roc.pdf")
    plt.close(fig)
