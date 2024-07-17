import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyhf

from ml.common.utils.loggers import setup_logger
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.fit_mle import FitMLE


class Pull(FitMLE):
    def __init__(self, model_name, classifier_model_name, save_dir="ml/custom/HIGGS/analysis/plots/pull", **kwargs):
        super().__init__(model_name, classifier_model_name, save_dir, **kwargs)
        self.pull_results = None

    def make_pull(self, lumi_idx=0):
        if lumi_idx == -1:
            lumi_idx = len(self.templates) - 1

        lumi = self.templates[lumi_idx].lumi["data"]
        bestfits = self.mle_fit().iloc[lumi_idx]

        bestfit = np.hstack((bestfits["mu"], bestfits["gamma"]))
        errors = np.hstack((bestfits["mu_err"], bestfits["gamma_err"]))

        model = self.templates[lumi_idx].model

        pulls = pyhf.tensorlib.concatenate(
            [
                (bestfit[model.config.par_slice(k)] - model.config.param_set(k).suggested_init)
                / model.config.param_set(k).width()
                for k in model.config.par_order
                if model.config.param_set(k).constrained
            ]
        )

        pullerr = pyhf.tensorlib.concatenate(
            [
                errors[model.config.par_slice(k)] / model.config.param_set(k).width()
                for k in model.config.par_order
                if model.config.param_set(k).constrained
            ]
        )

        labels = np.asarray(
            [
                f"{k}[{i}]" if model.config.param_set(k).n_parameters > 1 else k
                for k in model.config.par_order
                if model.config.param_set(k).constrained
                for i in range(model.config.param_set(k).n_parameters)
            ]
        )

        labels = np.array([r"$\gamma_{" + "{}".format(i) + "}$" for i in range(len(labels))])

        self.pull_results = [pulls, pullerr, errors, labels, lumi]

        return self.pull_results

    def pull_plot(self, vline=None, title=None, text=True, mark_sigmas=False):
        pulls, pullerr, errors, labels, lumi = self.pull_results

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 5)

        # set up axes labeling, ranges, etc...
        ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(labels.size).tolist()))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(pulls) - 0.5)
        # ax.set_title("Pull Plot", fontsize=18)
        ax.set_ylabel(r"$(\theta - \hat{\theta})\,/ \Delta \theta$", fontsize=18)

        # draw the +/- 2.0 horizontal lines
        ax.hlines([-2, 2], -0.5, len(pulls) - 0.5, colors="black", linestyles="dotted")
        # draw the +/- 1.0 horizontal lines
        ax.hlines([-1, 1], -0.5, len(pulls) - 0.5, colors="black", linestyles="dashdot")
        # draw the +/- 2.0 sigma band
        ax.fill_between([-0.5, len(pulls) - 0.5], [-2, -2], [2, 2], facecolor="yellow")
        # drawe the +/- 1.0 sigma band
        ax.fill_between([-0.5, len(pulls) - 0.5], [-1, -1], [1, 1], facecolor="green")
        # draw a horizontal line at pull=0.0
        ax.hlines([0], -0.5, len(pulls) - 0.5, colors="black", linestyles="dashed")
        # finally draw the pulls
        ax.scatter(range(len(pulls)), pulls, color="black")
        # and their uncertainties
        ax.errorbar(
            range(len(pulls)),
            pulls,
            color="black",
            xerr=0,
            yerr=pullerr,
            marker=".",
            fmt="none",
        )

        if vline:
            ax.axvline(vline, c="r", ls="--")

        if title:
            ax.set_title(title, fontsize=18)

        if text and vline:
            ax.text(0.5, 1.5, r"Relative systematic$\longrightarrow$", color="red", fontsize=18)
            ax.text(vline + 1, 1.5, r"MC statistical uncertainty$\longrightarrow$", color="red", fontsize=18)

        # error > 1
        error_gt1 = np.argmax(errors > 1) - 0.5
        ax.axvline(x=error_gt1, color="red", linestyle="--")

        if mark_sigmas:
            ax.text(error_gt1 + 0.1, 1.5, r"$\sigma \geq 1 \longrightarrow$", color="red", fontsize=18)

        logging.info("Saving pull plot!")

        save_str = f"pulls_{self.N_gen}gen"
        save_str += "_mc_only" if self.mc_only else ""
        save_str += "_bkg_only" if self.bkg_only else ""
        save_str += f" {self.cut_variable}" if self.cut_variable else ""
        save_str += f"_{lumi:.1f}fb"
        save_str.replace(" ", "_")

        fig.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_str}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    mc_only, bkg_only = True, False
    cut_variable = "m bb"

    if cut_variable == "m bb":
        bin_range = (0.0, 3.0)
    else:
        bin_range = (0.55, 1.0)

    sys_err = 0.1
    sig_frac = 0.01

    N_mc_bkg_lst = np.logspace(4, 6, 30).astype(int)
    N_data_sig_lst = N_mc_bkg_lst * sig_frac
    N_data_sig_lst = N_data_sig_lst.astype(int)

    pull = Pull(
        model_name="MADEMOG_flow_model_gauss_rank",
        classifier_model_name="BinaryClassifier_sigbkg_gauss_rank",
        sys_err=sys_err,
    )

    pull.setup_templates(
        N_gen=10**6,
        N_mc_bkg_lst=N_mc_bkg_lst,
        N_data_sig_lst=N_data_sig_lst,
        scale_mc_sig=True,
        mc_only=mc_only,
        bkg_only=bkg_only,
        cut_variable=cut_variable,
        bin_range=bin_range,
        n_bins=25,
    )

    pull.make_pull(lumi_idx=0)
    pull.pull_plot(vline=23.5)
