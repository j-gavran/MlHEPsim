import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.utils import (
    equalize_counts_to_ref,
    get_binary_classification_scores,
    get_model,
    get_scaler,
    get_sig_bkg_ref,
    sample_from_models,
    single_scale_forward,
)


def get_density(model, sample, device="cuda", chunks=10):
    if type(sample) is list:
        sample = np.vstack(sample)

    if type(sample) is np.ndarray:
        sample = torch.from_numpy(sample.astype(np.float32))

    sample = sample.to(device)

    # note that this is for backwards compatibility and has been updated in the model
    density_data = []
    N = sample.shape[0]
    chunks_lst = chunks * [N // chunks]

    if N % chunks != 0:
        chunks_lst += [N % chunks]

    for i, chunk in tqdm(enumerate(chunks_lst), desc=f"Estimating density for {N} examples", total=chunks, leave=False):
        chunk_sample = sample[i * chunk : (i + 1) * chunk]
        log_density = model.estimate_density(chunk_sample, mean=False, exp=False)
        density_data.append(log_density.flatten())

    density_data = np.concatenate(density_data)

    return density_data


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    save_dir = "ml/custom/HIGGS/analysis/plots/flow_anomaly"
    mkdir(save_dir)

    cont_rescale_type = "gauss_rank"

    N = 10**6
    chunks = 20
    select_models = [f"MADEMOG_flow_model_{cont_rescale_type}"]

    bkg_ref, sig_ref, selection = get_sig_bkg_ref(N)

    # filter models
    model_dct = {m: get_model(m).eval() for m in select_models}

    # sample from models
    samples = sample_from_models(model_dct, N, chunks=chunks, resample=1)

    # fetch scalers
    scalers_dct = {select_model: get_scaler(select_model, ver=1) for select_model in select_models}

    # scale ref
    bkg_ref = single_scale_forward(bkg_ref, scalers_dct[select_models[0]], selection)
    sig_ref = single_scale_forward(sig_ref, scalers_dct[select_models[0]], selection)

    samples["bkg"], samples["sig"] = [bkg_ref], [sig_ref]

    # make sure all samples have the same number of events
    equalize_counts_to_ref(samples, ref_str="bkg")

    bkg_density = get_density(model_dct[select_models[0]], samples["bkg"])
    sig_density = get_density(model_dct[select_models[0]], samples["sig"])

    # auc
    fig, ax = plt.subplots(1, 1)

    x = np.concatenate([bkg_density, sig_density])
    labels = np.concatenate([np.zeros_like(bkg_density), np.ones_like(sig_density)])

    ax, fig, auc_dct = get_binary_classification_scores(x, labels, ax=ax, fig=fig)

    ax.plot([0, 1], [0, 1], "k--")

    auc = auc_dct["auroc"]

    print(auc)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/flow_anomaly_rocauc.pdf")
    plt.close(fig)

    # plot
    out_bkg_h, bins, _ = plt.hist(
        bkg_density, bins=150, histtype="step", label="bkg MC", density=True, lw=2, range=(-10, 60)
    )
    out_sig_h, _, _ = plt.hist(
        sig_density, bins=bins, histtype="step", label="sig MC", density=True, lw=2, range=(-10, 60)
    )

    plt.xlabel("flow log density")
    plt.ylabel("density [a.u.]")

    plt.xlim((-10, 60))

    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/flow_anomaly.pdf")
    plt.close()
