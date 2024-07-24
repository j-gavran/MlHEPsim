import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhf
import scipy
from tqdm import tqdm

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.fit_pyhf import FitSetup


class P0s(FitSetup):
    def __init__(
        self,
        model_name,
        classifier_model_name,
        sys_err=0.1,
        par_bounds=None,
        inits=1,
        save_dir="ml/custom/HIGGS/analysis/plots/p0",
    ):
        super().__init__(model_name, classifier_model_name, sys_err, par_bounds, inits)
        self.save_dir = mkdir(save_dir)
        self.fit_results = dict()

    def setup_templates(
        self,
        N_gen,
        N_mc_bkg_lst,
        N_data_sig_lst,
        n_bins=25,
        bounds_low=0,
        bounds_up=5,
        rtol=0.1,
        cache_dir="ml/data/higgs/p0",
        bkg_only=False,
        cut_variable=False,
        scale_mc_sig=True,
        **kwargs,
    ):
        logging.info("[red]Setting up ML templates for p0 fit![/red]")
        ml_templates = super().setup_templates(
            N_gen,
            N_mc_bkg_lst,
            N_data_sig_lst,
            n_bins=n_bins,
            bounds_low=bounds_low,
            bounds_up=bounds_up,
            rtol=rtol,
            cache_dir=cache_dir,
            mc_only=False,
            bkg_only=bkg_only,
            cut_variable=cut_variable,
            scale_mc_sig=scale_mc_sig,
            **kwargs,
        )

        self.templates = []

        logging.info("[red]Setting up MC templates for p0 fit![/red]")
        mc_templates = super().setup_templates(
            N_gen,
            N_mc_bkg_lst,
            N_data_sig_lst,
            n_bins=n_bins,
            bounds_low=bounds_low,
            bounds_up=bounds_up,
            rtol=rtol,
            cache_dir=cache_dir,
            mc_only=True,
            bkg_only=bkg_only,
            cut_variable=cut_variable,
            scale_mc_sig=scale_mc_sig,
            **kwargs,
        )

        self.templates = {"ml": ml_templates, "mc": mc_templates}

    def fit_p0_templates(self, templates, test_stat="q0", poi_test=0.0):
        # BPK what we are doing here is actually calculating the discovery p-value ; which is what we
        # actually want here...
        # p0 = pyhf.infer.hypotest(0.0, data, model, test_stat="q0")
        # p0_Z = scipy.stats.norm.isf(p0)
        # print(f"p(s)=0 (Z): {p0:0.2f} ({p0_Z:0.2f})")
        pyhf.set_backend("numpy", "minuit")

        bounds = self._set_par_bounds()
        inits = self._set_init()

        results = []
        for template in tqdm(templates, desc="Fitting templates", leave=False):
            lumi = template.lumi["data"]
            model = template.model

            observations = list(template.data) + model.config.auxdata
            p0 = pyhf.infer.hypotest(poi_test, observations, model, test_stat=test_stat)

            results.append(
                {
                    "lumi": lumi,
                    "p0": float(p0),
                }
            )

        return pd.DataFrame(results)

    def fit_p0(self, test_stat="q0", poi_test=0.0):
        ml_results = self.fit_p0_templates(self.templates["ml"], test_stat=test_stat, poi_test=poi_test)
        mc_results = self.fit_p0_templates(self.templates["mc"], test_stat=test_stat, poi_test=poi_test)

        self.fit_results = {
            "ml": ml_results,
            "mc": mc_results,
        }

        return self.fit_results

    def plot_p0(self, label="", log_scale=True):
        ml, mc = self.fit_results["ml"], self.fit_results["mc"]

        # plt.plot(ml["lumi"], ml["p0"], c="C0", lw=2)
        # plt.plot(ml["lumi"], mc["p0"], c="C1", lw=2)
        plt.plot(ml["lumi"], scipy.stats.norm.isf(ml["p0"]), c="C0", lw=2, zorder=20)
        plt.plot(ml["lumi"], scipy.stats.norm.isf(mc["p0"]), c="C1", lw=2, zorder=20)

        # plt.plot(mc["lumi"], mc["p0"], ls="--", c="C0", lw=2)

        plt.legend(
            [
                "p-value ML",
                "p-value MC",
            ],
            ncol=2,
            fontsize=14,
            loc="lower left" if log_scale else "upper right",
        )

        plt.xlabel(r"L [fb$^{-1}$]", fontsize=22)
        plt.ylabel(r"$p$-value [$\sigma$]", fontsize=22)

        if log_scale:
            plt.yscale("log")

        # plot.xscale("log")
        plt.axhline(0, color="k")
        plt.axhline(1, color="0.5", linestyle="--", lw=1)

        if log_scale:
            plt.ylim(bottom=0.005, top=2.5)
        else:
            plt.ylim(bottom=-0.25, top=2.5)
        plt.axhline(2, color="0.5", linestyle="--", lw=1)

        plt.xlim((ml["lumi"].min(), ml["lumi"].max()))

        plt.tight_layout()

        logging.info("Saving p0 plot!")

        if log_scale:
            label += "_log"

        if self.cut_variable:
            label += f"_{self.cut_variable}"
        # BPK
        label += "_B"

        if self.bkg_only:
            plt.savefig(f"{self.save_dir}/p0_q0_mu0_bkg_only{label}.pdf")
        else:
            plt.savefig(f"{self.save_dir}/p0_q0_mu0{label}.pdf")

        plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    bkg_only = True
    cut_variable = False  # "m bb"  # False

    if cut_variable == "m bb":
        bin_range = (0.0, 3.0)
    else:
        bin_range = (0.55, 1.0)

    sys_err = 0.1
    sig_frac = 0.05

    N_mc_bkg_lst = np.logspace(4, 6, 30).astype(int)
    N_data_sig_lst = N_mc_bkg_lst * sig_frac
    N_data_sig_lst = N_data_sig_lst.astype(int)

    p_0 = P0s(
        model_name="MADEMOG_flow_model_gauss_rank_best",
        classifier_model_name="BinaryClassifier_sigbkg_gauss_rank_best7",
        sys_err=sys_err,
    )

    p_0.setup_templates(
        N_gen=10**6,
        N_mc_bkg_lst=N_mc_bkg_lst,
        N_data_sig_lst=N_data_sig_lst,
        scale_mc_sig=True,
        bkg_only=bkg_only,
        cut_variable=cut_variable,
        bin_range=bin_range,
        n_bins=25,
    )

    res = p_0.fit_p0()
    print(res)

    p_0.plot_p0(
        label=f"_{sig_frac}",
        log_scale=False,
    )
