import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit

from ml.common.stats.two_sample_tests import (
    chi2_twosample_test,
    ks_twosample_test,
    two_sample_plot,
)
from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.c2st import setup_c2st_procedure
from ml.custom.HIGGS.analysis.utils import BINNING, LABELS_MAP, rescale_back


class C2stOutTests:
    def __init__(self, select_model, cont_rescale_type, N, save_dir="ml/custom/HIGGS/analysis/plots/c2st_tests"):
        self.select_model = select_model
        self.cont_rescale_type = cont_rescale_type
        self.N = N
        self.save_dir = save_dir

        mkdir(self.save_dir)

        self.c2st_dct = self.setup()

        self.threshold_left, self.threshold_right, self.center = None, None, None

    def setup(self):
        X, Y, c2st_X, c2st_Y, _, _, selection, scalers = setup_c2st_procedure(
            self.N,
            self.select_model,
            closure_classifier=False,
            cont_rescale_type=self.cont_rescale_type,
            get_baseline=False,
        )

        samples = {"X": [X], "Y": [Y]}
        scalers_dct = {"X": scalers["X"], "Y": scalers["X"]}
        samples = rescale_back(samples, scalers_dct, selection)

        X, Y = samples["X"][0], samples["Y"][0]

        logging.info("[red]C2ST test setup finished.[/red]")

        c2st_X = c2st_X.cpu().numpy()
        c2st_Y = c2st_Y.cpu().numpy()

        return {
            "X": X,
            "Y": Y,
            "c2st_X": c2st_X,
            "c2st_Y": c2st_Y,
            "scalers": scalers,
        }

    def set_thresholds(self, threshold_left, threshold_right, center=0.5):
        self.threshold_left = threshold_left
        self.threshold_right = threshold_right
        self.center = center

    def test_features(self):
        X, Y = self.c2st_dct["X"], self.c2st_dct["Y"]

        chi2_test = chi2_twosample_test(X, Y)
        ks_test = ks_twosample_test(X, Y)

        chi2_test["feature"] = list(LABELS_MAP.keys())
        ks_test["feature"] = list(LABELS_MAP.keys())

        return chi2_test, ks_test

    def test_c2st_outputs(self):
        c2st_X, c2st_Y = self.c2st_dct["c2st_X"], self.c2st_dct["c2st_Y"]

        chi2_test = chi2_twosample_test(c2st_X, c2st_Y)
        ks_test = ks_twosample_test(c2st_X, c2st_Y)

        chi2_test["feature"] = ["c2st_out"]
        ks_test["feature"] = ["c2st_out"]

        return chi2_test, ks_test

    def plot_c2st_inputs(self, n_bins=50, **kwargs):
        X, Y = self.c2st_dct["X"], self.c2st_dct["Y"]

        fig, axs = plt.subplots(6, 3, figsize=(15, 20))
        axs = axs.flatten()

        two_sample_plot(X, Y, axs=axs, n_bins=n_bins, lw=2, **kwargs)

        logging.info("[green]Saving test two sample c2st inputs plot.[/green]")

        fig.tight_layout()
        fig.savefig(f"{self.save_dir}/two_sample_input_c2st_plot.pdf")
        plt.close()

    def _plot_cuts(self, center=None):
        if center is None:
            center = self.center

        plt.axvline(self.threshold_left, color="black", linestyle="--", lw=1)
        plt.axvline(self.threshold_right, color="black", linestyle="--", lw=1, label="cut")
        plt.axvline(center, color="black", linestyle="-", lw=2, alpha=0.2)

        plt.axvspan(-10, self.threshold_left, alpha=0.1, color="k")
        plt.axvspan(self.threshold_right, 10, alpha=0.1, color="k")

    def plot_c2st_outputs(self, n_bins=100, bin_range=(0.0, 1.0)):
        c2st_X, c2st_Y = self.c2st_dct["c2st_X"], self.c2st_dct["c2st_Y"]

        plt.figure(figsize=(8, 6))

        _, b, _ = plt.hist(c2st_X, bins=n_bins, label="MC c2st", histtype="step", lw=2, density=True, range=bin_range)
        plt.hist(c2st_Y, bins=b, label="ML c2st", histtype="step", lw=2, density=True)

        plt.xlim(bin_range)

        # plt.xlabel("C sigmoid output")
        plt.xlabel("r(x)")
        plt.ylabel("density [a.u.]")

        plt.axvline(self.center, color="black", linestyle="-", lw=2, alpha=0.2)
        # self._plot_cuts()

        plt.legend()

        logging.info("[green]Saving two sample c2st outputs plot.[/green]")

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/two_sample_output_c2st_plot.pdf")
        plt.close()

    def plot_c2st_ratio(self, weights=True, bin_range=(0.0, 1.0), log_scale=False):
        c2st_X, c2st_Y = self.c2st_dct["c2st_X"], self.c2st_dct["c2st_Y"]

        if weights:
            ratio = np.exp(logit(c2st_Y))  # ratio = c2st_Y / (1 - c2st_Y)
            ratio_base = np.exp(logit(c2st_X))  # ratio_base = c2st_X / (1 - c2st_X)
        else:
            ratio = c2st_Y / c2st_X
            ratio_base = None

        ratio = ratio.flatten()

        plt.figure(figsize=(8, 6))

        if weights:
            plt.hist(ratio_base, bins=100, histtype="step", range=bin_range, density=True, lw=2, label="MC ratio")

        plt.hist(ratio, bins=100, histtype="step", range=bin_range, density=True, lw=2, label="ML  ratio")

        plt.xlim(bin_range)

        self._plot_cuts(center=1.0)

        if weights:
            plt.xlabel("r(x)")
        else:
            plt.xlabel("c2st ML / c2st MC")

        plt.ylabel("density [a.u.]")

        if weights:
            plt.legend(["MC c2st", "ML c2st", "tail cut"], loc="upper right")

        if log_scale:
            plt.yscale("log")

        logging.info("[green]Saving ratio plot.[/green]")

        plt.tight_layout()

        if weights:
            plt.savefig(f"{self.save_dir}/c2st_mc_gen_ratio_weighs.pdf")
        else:
            plt.savefig(f"{self.save_dir}/c2st_mc_gen_ratio.pdf")

        plt.close()

        return ratio

    def plot_c2st_tails(self, n_bins=60, weights=True, log_scale=False, cut_on_X=False):
        X, Y = self.c2st_dct["X"], self.c2st_dct["Y"]
        c2st_X, c2st_Y = self.c2st_dct["c2st_X"], self.c2st_dct["c2st_Y"]

        if weights:
            if cut_on_X:
                cut_var = np.exp(logit(c2st_X))
            else:
                cut_var = np.exp(logit(c2st_Y))  # cut_var = c2st_Y / (1 - c2st_Y)
        else:
            cut_var = c2st_Y

        cut_var = cut_var.flatten()

        tail_mask_left = cut_var < self.threshold_left
        tail_mask_right = cut_var > self.threshold_right

        Y_masked_left = Y[: len(tail_mask_left)][tail_mask_left, :]
        Y_maksed_right = Y[: len(tail_mask_right)][tail_mask_right, :]

        logging.info(f"Number of samples in left/right Y tail: {Y_masked_left.shape[0]}/{Y_maksed_right.shape[0]}")

        fig, axs = plt.subplots(6, 3, figsize=(16, 24))
        axs = axs.flatten()

        binning = list(BINNING.values())

        if log_scale:
            binning = [None] * len(binning)

        labels_map = list(LABELS_MAP.values())

        Y_masked_left_right = [Y_masked_left, Y_maksed_right]
        labels = [f"ML tail cut $<$ {self.threshold_left}", f"ML tail cut $>$ {self.threshold_right}"]

        for i, ax in enumerate(axs):
            _, bins, _ = ax.hist(
                X[:, i], bins=n_bins, histtype="step", density=True, range=binning[i], lw=1, color="k", label="MC"
            )
            ax.hist(X[:, i], bins=n_bins, histtype="stepfilled", density=True, range=binning[i], color="k", alpha=0.2)

            for Y_masked, label in zip(Y_masked_left_right, labels):
                ax.hist(Y_masked[:, i], bins=bins, histtype="step", density=True, lw=1.5, zorder=100, label=label)

            if i == 0:
                ax.legend()

            if not log_scale:
                ax.set_xlim(binning[i])

            ax.set_xlabel(labels_map[i])
            ax.set_ylabel("density [a.u.]")

        if log_scale:
            for ax in axs:
                ax.set_yscale("log")

        logging.info("[green]Saving test two sample plot for tails.[/green]")

        fig.tight_layout()

        if log_scale:
            save_str = f"{self.save_dir}/two_sample_plot_tails_log.pdf"
        else:
            save_str = f"{self.save_dir}/two_sample_plot_tails.pdf"

        fig.savefig(save_str)
        plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    c2st_out_tester = C2stOutTests(
        select_model="MADEMOG_flow_model_gauss_rank_best",
        cont_rescale_type="gauss_rank",
        N=2 * 10**6,
    )

    c2st_out_tester.set_thresholds(
        threshold_left=0.94,
        threshold_right=1.08,
        # threshold_left=0.8,
        # threshold_right=1.14,
    )

    # stat. tests
    chi2_test_features, ks_test_features = c2st_out_tester.test_features()

    print(chi2_test_features)
    print(ks_test_features)

    chi2_test_c2st, ks_test_c2st = c2st_out_tester.test_c2st_outputs()

    print(chi2_test_c2st)
    print(ks_test_c2st)

    # plots
    c2st_out_tester.plot_c2st_inputs()

    c2st_out_tester.plot_c2st_outputs(bin_range=(0.4, 0.6))

    # c2st_out_tester.plot_c2st_ratio(weights=True, log_scale=False, bin_range=(0.7, 1.3))
    c2st_out_tester.plot_c2st_ratio(weights=True, log_scale=False, bin_range=(0.8, 1.2))

    c2st_out_tester.plot_c2st_tails(weights=True, log_scale=False)
