import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhf
from tqdm import tqdm
from uncertainties import ufloat
from uncertainties import unumpy as unp

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import (
    HEPPlot,
    errorbar_plot,
    set_size,
    step_hist_plot,
    style_setup,
)
from ml.custom.HIGGS.analysis.fit_pyhf_N import FitSetup
from ml.custom.HIGGS.analysis.utils import LABELS_MAP


class FitMLE(FitSetup):
    def __init__(self, model_name, classifier_model_name, save_dir="ml/custom/HIGGS/analysis/plots/spur", **kwargs):
        super().__init__(model_name, classifier_model_name, **kwargs)
        self.save_dir = mkdir(save_dir)
        self.bestfits = dict()

    def mle_fit(self):
        pyhf.set_backend("numpy", "minuit")

        bounds = self._set_par_bounds()
        inits = self._set_init()

        bestfits = []

        for template in tqdm(self.templates, desc="Fitting templates", leave=False):
            model = template.model

            observations = list(template.data) + model.config.auxdata

            result, twice_nll = pyhf.infer.mle.fit(
                observations,
                model,
                return_uncertainties=True,
                return_fitted_val=True,
                par_bounds=bounds,
                init_pars=inits,
            )

            bestfit, errors = result.T

            bestfits.append(
                {
                    "lumi": template.lumi["data"],
                    "mu": bestfit[0],
                    "gamma": bestfit[1:],
                    "mu_err": errors[0],
                    "gamma_err": errors[1:],
                    "twice_nll": float(twice_nll),
                }
            )

        self.bestfits = pd.DataFrame(bestfits)

        return self.bestfits

    def plot_poi(self, label="", clip_yerr=0.2):
        fig, ax = plt.subplots(1, 1)

        ax.plot(self.bestfits["lumi"], self.bestfits["mu"])

        yerr = self.bestfits["mu_err"].to_numpy()

        try:
            yerr[yerr > clip_yerr] = np.max(yerr[yerr < clip_yerr])
        except ValueError:
            pass

        errorbar_plot(
            ax,
            x=self.bestfits["lumi"],
            y=self.bestfits["mu"],
            xerr=None,
            yerr=self.bestfits["mu_err"],
            label="Fit bkg only" if self.bkg_only else "Fit",
            color="C0",
            markersize=6,
        )

        if self.bkg_only:
            ax.axhline(0, color="r", linestyle="--", label="Best match")
        else:
            ax.axhline(1, color="r", linestyle="--", label="Best match")

        ax.set_xlabel(r"L [fb$^{-1}$]")
        ax.set_ylabel(r"$\mu$" + f" {label}")
        ax.legend()

        ax.set_xscale("log")
        ax.set_xlim((self.bestfits["lumi"].min(), self.bestfits["lumi"].max() + 50))

        logging.info("Saving mu_lumi plot!")

        if self.mc_only:
            label += "_mc_only"

        if self.bkg_only:
            label += "_bkg_only"

        if self.cut_variable:
            label += f"_cut_{self.cut_variable}"

        fig.tight_layout()
        label = label.replace(" ", "_")
        plt.savefig(f"{self.save_dir}/mu_lumi_{label}.pdf")
        plt.close(fig)

    def plot_spur(self, label="", rel=False):
        spurs, spurs_std = [], []
        for i in range(len(self.templates)):
            mu = unp.uarray(self.bestfits["mu"].iloc[i], self.bestfits["mu_err"].iloc[i])
            S = unp.uarray(self.templates[i].sig, self.templates[i].sig_stat_err)

            if self.bkg_only:
                spur_bin = mu * S
            else:
                spur_bin = mu * S - S

            if rel:
                spur_bin = spur_bin / ufloat(np.sum(self.templates[i].sig), 0) * 100

            spur = spur_bin.mean()
            spur, spur_std = unp.nominal_values(spur), unp.std_devs(spur)

            spurs.append(float(spur))
            spurs_std.append(float(spur_std))

        fig, ax = plt.subplots(1, 1)

        ax.plot(self.bestfits["lumi"], spurs)

        errorbar_plot(
            ax,
            x=self.bestfits["lumi"],
            y=spurs,
            xerr=None,
            yerr=spurs_std,
            color="C0",
            markersize=6,
        )

        ax.set_xlabel(r"L [fb$^{-1}$]")

        if rel:
            ax.set_ylabel(r"$\frac{1}{S}(\mu S - S)$" + f"  {label} [%]", fontsize=15)
            ax.axhline(0, color="r", linestyle="--")
        else:
            ax.set_ylabel(r"$\mu S - S$" + f"  {label} [events]", fontsize=15)

        ax.set_xscale("log")
        ax.set_xlim((self.bestfits["lumi"].min(), self.bestfits["lumi"].max() + 50))

        logging.info("Saving spur plot!")

        if self.mc_only:
            label += "_mc_only"

        if self.bkg_only:
            label += "_bkg_only"

        if self.cut_variable:
            label += f"_cut_{self.cut_variable}"

        if rel:
            label += "_rel"

        fig.tight_layout()
        label = label.replace(" ", "_")
        plt.savefig(f"{self.save_dir}/spur_{label}.pdf")
        plt.close(fig)

    def plot_fit(self, lumi_idx=0, ylim=(0.5, 1.5), top_ylim=None, postfit=True, label=""):
        if lumi_idx == -1:
            lumi_idx = len(self.bestfits) - 1

        lumi = self.bestfits["lumi"].iloc[lumi_idx]

        mu, mu_err = self.bestfits["mu"][lumi_idx], self.bestfits["mu_err"][lumi_idx]
        gamma, gamma_err = self.bestfits["gamma"][lumi_idx], self.bestfits["gamma_err"][lumi_idx]

        mu = ufloat(mu, mu_err)
        gamma = unp.uarray(gamma, gamma_err)

        gamma_stat, gamma_sys = gamma[: len(gamma) // 2], gamma[len(gamma) // 2 :]

        template = self.templates[lumi_idx]
        bin_edges = template.bin_edges

        sig, bkg, data = template.sig, template.bkg, template.data
        sig_stat_err, bkg_stat_err = template.sig_stat_err, template.bkg_stat_err

        sig = unp.uarray(sig, sig_stat_err)
        bkg = unp.uarray(bkg, bkg_stat_err)

        if postfit:
            sig = mu * sig
            bkg = gamma_stat * gamma_sys * bkg

        hep_plot = HEPPlot(
            data=data,
            mc=[
                unp.nominal_values(bkg),
                unp.nominal_values(sig),
            ],
            mc_err=[
                unp.std_devs(bkg),
                unp.std_devs(sig),
            ],
            bin_edges=bin_edges,
        )

        hep_plot.setup_figure(figsize=(8, 8))

        hep_plot.ax.set_ylabel("$N$")
        hep_plot.ax.set_xlim([bin_edges.min(), bin_edges.max()])

        if self.cut_variable:
            hep_plot.ax.set_xlabel(LABELS_MAP[self.cut_variable])
        else:
            hep_plot.ax.set_xlabel("classifier output")

        hep_plot.plot_data(label=f"data MC", color="black")
        hep_plot.plot_mc(labels=["bkg ML", "sig MC"], colors=["C0", "C1"])
        hep_plot.plot_ratio(lower_ylabel="MC / ML", ylim=ylim)

        hep_plot.ax.set_ylim(bottom=0.0, top=top_ylim)

        step_hist_plot(
            hep_plot.ax,
            unp.nominal_values(sig),
            bin_edges,
            color="r",
            label="signal model",
            lw=1.5,
            ls="-",
        )

        save_str = "ml_bkg_postfit_hist" if postfit else "ml_bkg_prefit_hist"
        save_str += "_mc_only" if self.mc_only else ""
        save_str += "_bkg_only" if self.bkg_only else ""
        save_str += f" {self.cut_variable}" if self.cut_variable else ""
        save_str += label if label != "" else ""
        save_str += f"_{lumi:.1f}fb"
        save_str.replace(" ", "_")
        save_str += "_Bmod"
        if postfit:
            logging.info(f"Saving post-fit plot for lumi {lumi:.3f} fb^-1")
        else:
            logging.info(f"Saving pre-fit plot for lumi {lumi:.3f} fb^-1")

        hep_plot.save(self.save_dir, save_str)


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    matplotlib.rcParams.update({"errorbar.capsize": 3})

    # mc_only, bkg_only = True, True
    mc_only, bkg_only = False, False
    cut_variable = False

    if cut_variable == "m bb":
        bin_range = (0.0, 3.0)
    else:
        bin_range = (0.55, 1.0)

    sys_err = 0.1
    sig_frac = 0.05

    N_mc_bkg_lst = np.logspace(4, 6, 30).astype(int)
    N_data_sig_lst = N_mc_bkg_lst * sig_frac
    N_data_sig_lst = N_data_sig_lst.astype(int)

    mle_fit = FitMLE(
        model_name="MADEMOG_flow_model_gauss_rank_best",
        classifier_model_name="BinaryClassifier_sigbkg_gauss_rank_best7",
        sys_err=sys_err,
    )

    mle_fit.setup_templates(
        N_gen=10**6,
        # N_gen=100000,
        N_mc_bkg_lst=N_mc_bkg_lst,
        N_data_sig_lst=N_data_sig_lst,
        scale_mc_sig=True,
        mc_only=mc_only,
        bkg_only=bkg_only,
        cut_variable=cut_variable,
        bin_range=bin_range,
        n_bins=25,
    )

    bestfits = mle_fit.mle_fit()

    print(bestfits)

    mle_fit.plot_poi(label=f"at {sig_frac*100}% sig fraction")

    mle_fit.plot_spur(label=f"at {sig_frac*100}% sig fraction", rel=True)
    mle_fit.plot_spur(label=f"at {sig_frac*100}% sig fraction", rel=False)

    mle_fit.plot_fit(lumi_idx=9, postfit=True, label=f"_{sig_frac}")
    mle_fit.plot_fit(lumi_idx=9, postfit=False, label=f"_{sig_frac}")
