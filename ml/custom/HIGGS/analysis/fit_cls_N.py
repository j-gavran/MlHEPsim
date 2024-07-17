import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhf
from tqdm import tqdm

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.fit_pyhf_N import FitSetup


class CLs_N(FitSetup):
    def __init__(
        self,
        model_name,
        classifier_model_name,
        sys_err=0.1,
        par_bounds=None,
        inits=1,
        save_dir="ml/custom/HIGGS/analysis/plots/cls",
    ):
        super().__init__(model_name, classifier_model_name, sys_err, par_bounds, inits)
        self.save_dir = mkdir(save_dir)
        self.fit_results = dict()

    def setup_templates_N(
        self,
        N_gen_lst,
        N_mc_bkg,
        N_data_sig,
        n_bins=25,
        bounds_low=0,
        bounds_up=5,
        rtol=0.1,
        cache_dir="ml/data/higgs/cls",
        bkg_only=False,
        cut_variable=False,
        scale_mc_sig=True,
        **kwargs,
    ):
        logging.info("[red]Setting up ML templates for CLs fit![/red]")
        ml_templates = super().setup_templates_N(
            N_gen_lst,
            N_mc_bkg,
            N_data_sig,
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

        logging.info("[red]Setting up MC templates for CLs fit![/red]")
        mc_templates = super().setup_templates_N(
            N_gen_lst,
            N_mc_bkg,
            N_data_sig,
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

    def fit_cls_templates(self, templates, test_stat="q0", poi_test=0.0):
        pyhf.set_backend("numpy", "minuit")

        bounds = self._set_par_bounds()
        inits = self._set_init()

        results = []
        for template in tqdm(templates, desc="Fitting templates", leave=False):
            lumi = template.lumi["data"]
            model = template.model

            observations = list(template.data) + model.config.auxdata

            calc = pyhf.infer.calculators.AsymptoticCalculator(
                observations,
                model,
                test_stat=test_stat,
                par_bounds=bounds,
                init_pars=inits,
            )

            teststat = calc.teststatistic(poi_test=poi_test)

            # probability distributions of the test statistic
            sb_dist, b_dist = calc.distributions(poi_test=poi_test)

            # calculate the p-values for the observed test statistic under
            # the signal + background and background-only model hypotheses.
            p_sb, p_b, p_s = calc.pvalues(teststat, sb_dist, b_dist)

            # calculate the CLs values corresponding to the
            # median significance of variations of the signal strength from the
            # background only hypothesis :math:`\left(\mu=0\right) at :math:`(-2,-1,0,1,2)\sigma`.
            p_exp_sb, p_exp_b, p_exp_s = calc.expected_pvalues(sb_dist, b_dist)

            results.append(
                {
                    "lumi": lumi,
                    "p_sb": float(p_sb),
                    "p_b": float(p_b),
                    "p_s": float(p_s),
                    "p_exp_sb": float(p_exp_sb[2]),
                    "p_exp_b": float(p_exp_b[2]),
                    "p_exp_s": float(p_exp_s[2]),
                    "teststat": float(teststat),
                }
            )

        return pd.DataFrame(results)

    def fit_cls(self, test_stat="q0", poi_test=0.0):
        ml_results = self.fit_cls_templates(self.templates["ml"], test_stat=test_stat, poi_test=poi_test)
        mc_results = self.fit_cls_templates(self.templates["mc"], test_stat=test_stat, poi_test=poi_test)

        self.fit_results = {
            "ml": ml_results,
            "mc": mc_results,
        }

        return self.fit_results

    def plot_cls(self, label="", log_scale=True):
        ml, mc = self.fit_results["ml"], self.fit_results["mc"]

        plt.plot(ml["lumi"], ml["p_sb"], c="C0", lw=2)
        plt.plot(ml["lumi"], ml["p_b"], c="C1", lw=2)
        plt.plot(ml["lumi"], ml["p_s"], c="C2", lw=2)

        plt.plot(mc["lumi"], mc["p_sb"], ls="--", c="C0", lw=2)
        plt.plot(mc["lumi"], mc["p_b"], ls="--", c="C1", lw=2)
        plt.plot(mc["lumi"], mc["p_s"], ls="--", c="C2", lw=2)

        plt.legend(
            ["CLsb ML", "CLb ML", "CLs ML", "CLsb MC", "CLb MC", "CLs MC"],
            ncol=2,
            fontsize=12,
            loc="lower left" if log_scale else "upper right",
        )
        plt.xlabel(r"$N_{ML}$ generated", fontsize=22)
        plt.ylabel(r"$p$-value", fontsize=22)

        if log_scale:
            plt.yscale("log")

        plt.xscale("log")

        plt.xlim((ml["lumi"].min(), ml["lumi"].max()))

        plt.tight_layout()

        logging.info("Saving CLs plot!")

        if log_scale:
            label += "_log"

        if self.cut_variable:
            label += f"_{self.cut_variable}"
        # BPK
        label += "_NB"

        if self.bkg_only:
            plt.savefig(f"{self.save_dir}/CLs_q0_mu0_bkg_only{label}.pdf")
        else:
            plt.savefig(f"{self.save_dir}/CLs_q0_mu0{label}.pdf")

        plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    bkg_only = False
    cut_variable = False

    if cut_variable == "m bb":
        bin_range = (0.0, 3.0)
    else:
        bin_range = (0.55, 1.0)

    sys_err = 0.1
    sig_frac = 0.05

    N_gen_lst = np.logspace(4, 6, 30).astype(int)  # think about optimizing!!
    # N_gen_lst = np.linspace(10**4,10**6, 30).astype(int) # think about optimizing!!
    N_mc_bkg = 10**4
    N_data_sig = N_mc_bkg * sig_frac
    N_data_sig = int(N_data_sig)

    cl_s = CLs_N(
        model_name="MADEMOG_flow_model_gauss_rank_best",
        classifier_model_name="BinaryClassifier_sigbkg_gauss_rank_best7",
        sys_err=sys_err,
    )

    cl_s.setup_templates_N(
        N_gen_lst=N_gen_lst,
        N_mc_bkg=N_mc_bkg,
        N_data_sig=N_data_sig,
        scale_mc_sig=True,
        bkg_only=bkg_only,
        cut_variable=cut_variable,
        bin_range=bin_range,
        n_bins=25,
    )

    res = cl_s.fit_cls()
    print(res)

    cl_s.plot_cls(
        label=f"_{sig_frac}",
        log_scale=True,
    )
