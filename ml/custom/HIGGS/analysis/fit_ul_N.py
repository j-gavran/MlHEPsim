import logging
import pickle
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhf

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.fit_pyhf_N import FitSetup


def calculate_upper_limit(templates, par_bounds, init_pars):
    results = []

    for template in templates:
        model = template.model

        data = list(template.data) + model.config.auxdata

        obs_limit, exp_limits = pyhf.infer.intervals.upper_limits.toms748_scan(
            data,
            model,
            template.bounds_low,
            template.bound_up,
            level=0.05,
            rtol=template.rtol,  # https://scikit-hep.org/pyhf/_modules/pyhf/infer/utils.html#create_calculator
            par_bounds=par_bounds,
            init_pars=init_pars,
        )
        results.append([obs_limit, exp_limits, template.lumi])

        logging.info(f"Fit done for {template}!")

    return results


class UpperLimitScan_N(FitSetup):
    def __init__(self, model_name, classifier_model_name, sys_err, workers=1, **kwargs):
        super().__init__(model_name, classifier_model_name, sys_err, **kwargs)
        self.sys_err = sys_err
        self.workers = workers
        self.results = []

    def scan_upper_limits(self, parse_results=True):
        """For kwargs see https://scikit-hep.org/pyhf/_generated/pyhf.infer.hypotest.html#pyhf.infer.hypotest."""
        pyhf.set_backend("numpy", "minuit")

        templates_splits = np.array_split(self.templates, self.workers)

        bounds = self._set_par_bounds()
        inits = self._set_init()

        logging.info(f"[bold][red]Starting scan with {self.workers} workers![/red][/bold]")
        logging.info(f"Bounds: {self.par_bounds}")
        logging.info(f"Init pars: {self.inits}")

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for template_split in templates_splits:
                futures.append(
                    executor.submit(calculate_upper_limit, template_split, par_bounds=bounds, init_pars=inits)
                )

        for future in futures:
            self.results.append(future.result())

        if parse_results:
            self.results = self._parse_results(self.results)

        return self.results

    def _parse_results(self, results, save_name="ml/data/higgs/upper_limit_N_df"):
        dct = {
            "lumi": [],
            "cls_obs": [],
            "cls_exp": [],
            "minus_sigma_2": [],
            "minus_sigma_1": [],
            "plus_sigma_1": [],
            "plus_sigma_2": [],
        }

        for result in results:
            for scan_result in result:
                cls_obs = scan_result[0]
                minus_sigma_2, minus_sigma_1, cls_exp, plus_sigma_1, plus_sigma_2 = scan_result[1]
                lumi = scan_result[2]

                dct["cls_obs"].append(float(cls_obs))
                dct["cls_exp"].append(float(cls_exp))
                dct["minus_sigma_2"].append(float(minus_sigma_2))
                dct["minus_sigma_1"].append(float(minus_sigma_1))
                dct["plus_sigma_2"].append(float(plus_sigma_2))
                dct["plus_sigma_1"].append(float(plus_sigma_1))
                dct["lumi"].append(lumi["data"])

        df = pd.DataFrame(dct)

        if self.cut_variable:
            save_name += f"_{self.cut_variable}"

        if self.mc_only:
            save_name += "_mc_only"

        sig_frac = self.N_data_sig / self.N_mc_bkg
        sig_frac = np.round(100 * sig_frac, 3)

        save_name += f"_{sig_frac:.2f}_sys_err_{100*self.sys_err:.2f}"

        pickle.dump(
            {
                "df": df,
                "sys_err": self.sys_err,
                "N_gen": self.N_gen,
                "N_mc_bkg": self.N_mc_bkg,
                "N_data_sig": self.N_data_sig,
            },
            open(f"{save_name}.p", "wb"),
        )

        return df


def plot_ul_N(results_p, results_p_mc_only, plot_lumi=True, save_dir="ml/custom/HIGGS/analysis/plots/upper_limits"):
    mkdir(save_dir)

    results_dct = pickle.load(open(results_p, "rb"))
    results_dct_mc_only = pickle.load(open(results_p_mc_only, "rb"))

    res_gen = results_dct["df"]
    res_mc = results_dct_mc_only["df"]

    logging.info(f"ML results:\n{res_gen}")
    logging.info(f"MC only results:\n{res_mc}")

    N_gen = results_dct["N_gen"]  # probably empty?
    N_mc_bkg = results_dct["N_mc_bkg"]
    N_data_sig = results_dct["N_data_sig"]

    sig_frac = N_data_sig / N_mc_bkg
    sig_frac = np.round(100 * sig_frac, 3)

    sys_err = 100 * results_dct["sys_err"]

    if plot_lumi:
        x = res_gen["lumi"]
        xlim = (x.iloc[0], x.iloc[-1])
    else:
        x = N_gen
        xlim = (0, x[-1])

    y = res_mc.cls_obs - res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="yellow", label=r"$\pm 2\sigma$")
    plt.fill_between(x, y + res_mc.minus_sigma_1, y + res_mc.plus_sigma_1, color="green", label=r"$\pm 1\sigma$")

    plt.plot(x, y + res_mc.minus_sigma_1, c="k", ls="dotted")

    plt.plot(x, y + res_mc.plus_sigma_1, c="k", ls="dotted")

    plt.plot(x, y + res_mc.minus_sigma_2, c="k", ls="dotted")
    plt.plot(x, y + res_mc.plus_sigma_2, c="k", ls="dotted")

    plt.plot(x, res_mc.cls_obs, zorder=20, color="k", label=r"$\mu$ MC obs")

    plt.plot(x, res_gen.cls_obs, zorder=20, color="r", label=r"$\mu$ ML obs")

    if plot_lumi:
        plt.xlabel(r"$N_{\rm ML}$ generated", loc="center")
    else:
        plt.xlabel(r"$N_{ML}$ generated empty", loc="center")

    plt.ylabel(r"$\mu_{\rm UL}$")

    plt.xlim(xlim)
    # plt.ylim(top=3.5)
    plt.xscale("log")

    plt.title(f"Signal fraction: {sig_frac:.2f} %, sys. error: {sys_err:.2f} %", fontsize=15)

    logging.info("Saving upper limit plot!")

    save_str = f"{save_dir}/"

    save_str += results_p.split("/")[-1]
    save_str = save_str.replace(".p", "")
    save_str = save_str.replace("_df", "")

    save_str += "_NB"  # log/lin

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_str}.pdf")
    plt.close()


def plot_ul_N_money(
    res5_p,
    res5_p_mc_only,
    res10_p,
    res10_p_mc_only,
    res20_p,
    res20_p_mc_only,
    save_dir="ml/custom/HIGGS/analysis/plots/upper_limits",
):
    mkdir(save_dir)

    results_dct20 = pickle.load(open(res20_p, "rb"))
    results_dct_mc_only20 = pickle.load(open(res20_p_mc_only, "rb"))

    res_gen = results_dct20["df"]
    res_mc = results_dct_mc_only20["df"]

    x = res_gen["lumi"]
    xlim = (x.iloc[0], x.iloc[-1])

    plt.plot(x, np.zeros(len(x)), c="k", ls="--", zorder=2)

    y = -res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="C2", alpha=0.3)
    plt.plot(
        x,
        res_gen.cls_obs - res_mc.cls_obs,
        zorder=20,
        color="C2",
        label=r"$\Delta \mu_{\rm UL} \pm 2\sigma$ (sys=20%)",
    )

    results_dct10 = pickle.load(open(res10_p, "rb"))
    results_dct_mc_only10 = pickle.load(open(res10_p_mc_only, "rb"))

    res_gen = results_dct10["df"]
    res_mc = results_dct_mc_only10["df"]

    y = -res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="C1", alpha=0.3)

    plt.plot(
        x,
        res_gen.cls_obs - res_mc.cls_obs,
        zorder=20,
        color="C1",
        label=r"$\Delta \mu_{\rm UL} \pm 2\sigma$ (sys=10%)",
    )

    results_dct5 = pickle.load(open(res5_p, "rb"))
    results_dct_mc_only5 = pickle.load(open(res5_p_mc_only, "rb"))

    res_gen = results_dct5["df"]
    res_mc = results_dct_mc_only5["df"]

    y = -res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="C0", alpha=0.3)
    plt.plot(
        x,
        res_gen.cls_obs - res_mc.cls_obs,
        zorder=20,
        color="C0",
        label=r"$\Delta \mu_{\rm UL} \pm 2\sigma$ (sys=5%)",
    )

    plt.xlabel(r"$N_{\rm ML}$ generated", loc="center")

    plt.ylabel(r"$\Delta$($\mu_{\rm UL}$)")

    plt.xlim(xlim)
    # plt.ylim(top=3.5)
    plt.xscale("log")

    sig_frac = 5.00
    plt.title(f"Signal fraction: {sig_frac:.2f} %", fontsize=15)

    logging.info("Saving upper money limit plot!")

    save_str = f"{save_dir}/"

    save_str += res5_p.split("/")[-1]
    save_str = save_str.replace(".p", "")
    save_str = save_str.replace("_df", "")

    save_str += "_NB_mon"  # log/lin

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_str}.pdf")
    plt.close()


def plot_ul_N_money2(
    res5_p,
    res5_p_mc_only,
    res10_p,
    res10_p_mc_only,
    res20_p,
    res20_p_mc_only,
    save_dir="ml/custom/HIGGS/analysis/plots/upper_limits",
):
    mkdir(save_dir)

    results_dct20 = pickle.load(open(res20_p, "rb"))
    results_dct_mc_only20 = pickle.load(open(res20_p_mc_only, "rb"))

    res_gen = results_dct20["df"]
    res_mc = results_dct_mc_only20["df"]

    x = res_gen["lumi"]
    xlim = (x.iloc[0], x.iloc[-1])

    y = res_gen.cls_obs - res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="C2", alpha=0.3)
    plt.plot(
        x,
        res_gen.cls_obs,
        zorder=20,
        color="C2",
        label=r"$\mu_{\rm UL}(\rm{ML}) \pm 2\sigma$ (sys=20%)",
    )
    plt.plot(
        x,
        res_mc.cls_obs,
        zorder=10,
        ls="--",
        color="C2",
        label=r"$\mu_{\rm UL}({\rm MC}) \pm 2\sigma$ (sys=20%)",
    )

    results_dct10 = pickle.load(open(res10_p, "rb"))
    results_dct_mc_only10 = pickle.load(open(res10_p_mc_only, "rb"))

    res_gen = results_dct10["df"]
    res_mc = results_dct_mc_only10["df"]

    y = res_gen.cls_obs - res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="C1", alpha=0.3)

    plt.plot(
        x,
        res_gen.cls_obs,
        zorder=20,
        color="C1",
        label=r"$\Delta \mu_{\rm UL}({\rm ML}) \pm 2\sigma$ (sys=10%)",
    )
    plt.plot(
        x,
        res_mc.cls_obs,
        zorder=10,
        ls="--",
        color="C1",
        label=r"$\mu_{\rm UL}({\rm MC}) \pm 2\sigma$ (sys=10%)",
    )

    results_dct5 = pickle.load(open(res5_p, "rb"))
    results_dct_mc_only5 = pickle.load(open(res5_p_mc_only, "rb"))

    res_gen = results_dct5["df"]
    res_mc = results_dct_mc_only5["df"]

    y = res_gen.cls_obs - res_mc.cls_exp

    plt.fill_between(x, y + res_mc.minus_sigma_2, y + res_mc.plus_sigma_2, color="C0", alpha=0.3)
    plt.plot(
        x,
        res_gen.cls_obs,
        zorder=20,
        color="C0",
        label=r"$\mu_{\rm UL}({\rm ML})  \pm 2\sigma$ (sys=5%)",
    )
    plt.plot(
        x,
        res_mc.cls_obs,
        zorder=10,
        ls="--",
        color="C0",
        label=r"$\mu_{\rm UL}({\rm MC}) \pm 2\sigma$ (sys=5%)",
    )

    plt.xlabel(r"$N_{\rm ML}$ generated", loc="center")

    plt.ylabel(r"$\Delta$($\mu_{\rm UL}$)")

    plt.xlim(xlim)
    # plt.ylim(top=3.5)
    plt.xscale("log")

    sig_frac = 5.00
    plt.title(f"Signal fraction: {sig_frac:.2f} %", fontsize=15)

    logging.info("Saving upper money2 limit plot!")

    save_str = f"{save_dir}/"

    save_str += res5_p.split("/")[-1]
    save_str = save_str.replace(".p", "")
    save_str = save_str.replace("_df", "")

    save_str += "_NB_mon2"  # log/lin

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_str}.pdf")
    plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    run = False  # False for plots only

    cut_variable = False  # "m bb"

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

    if run:
        scan = UpperLimitScan_N(
            model_name="MADEMOG_flow_model_gauss_rank_best",
            classifier_model_name="BinaryClassifier_sigbkg_gauss_rank_best7",
            sys_err=sys_err,
            par_bounds=None,
            workers=23,
        )
        scan.setup_templates_N(
            N_gen_lst=N_gen_lst,
            N_mc_bkg=N_mc_bkg,
            N_data_sig=N_data_sig,
            scale_mc_sig=True,
            mc_only=True,
            cut_variable=cut_variable,
            bin_range=bin_range,
            n_bins=25,
            bounds_low=0.0,
            bounds_up=10.0,
        )

        print(scan.scan_upper_limits())

        scan2 = UpperLimitScan_N(
            model_name="MADEMOG_flow_model_gauss_rank_best",
            classifier_model_name="BinaryClassifier_sigbkg_gauss_rank_best7",
            sys_err=sys_err,
            par_bounds=None,
            workers=23,
        )
        scan2.setup_templates_N(
            N_gen_lst=N_gen_lst,
            N_mc_bkg=N_mc_bkg,
            N_data_sig=N_data_sig,
            scale_mc_sig=True,
            mc_only=False,
            cut_variable=cut_variable,
            bin_range=bin_range,
            n_bins=25,
            bounds_low=0.0,
            bounds_up=10.0,
        )

        print(scan2.scan_upper_limits())

    plot_ul_N(
        # results_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_20.00.p",
        # results_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_20.00.p",
        results_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_10.00.p",
        results_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_10.00.p",
        # results_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_5.00.p",
        # results_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_5.00.p",
        plot_lumi=True,
    )

    # plot_ul_N_money(
    #     res5_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_5.00.p",
    #     res5_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_5.00.p",
    #     res10_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_10.00.p",
    #     res10_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_10.00.p",
    #     res20_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_20.00.p",
    #     res20_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_20.00.p",
    # )

    # plot_ul_N_money2(
    #     res5_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_5.00.p",
    #     res5_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_5.00.p",
    #     res10_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_10.00.p",
    #     res10_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_10.00.p",
    #     res20_p="ml/data/higgs/upper_limit_N_df_m bb_5.00_sys_err_20.00.p",
    #     res20_p_mc_only="ml/data/higgs/upper_limit_N_df_m bb_mc_only_5.00_sys_err_20.00.p",
    # )

    plot_ul_N_money(
        res5_p="ml/data/higgs/upper_limit_N_df_5.00_sys_err_5.00.p",
        res5_p_mc_only="ml/data/higgs/upper_limit_N_df_mc_only_5.00_sys_err_5.00.p",
        res10_p="ml/data/higgs/upper_limit_N_df_5.00_sys_err_10.00.p",
        res10_p_mc_only="ml/data/higgs/upper_limit_N_df_mc_only_5.00_sys_err_10.00.p",
        res20_p="ml/data/higgs/upper_limit_N_df_5.00_sys_err_20.00.p",
        res20_p_mc_only="ml/data/higgs/upper_limit_N_df_mc_only_5.00_sys_err_20.00.p",
    )

    plot_ul_N_money2(
        res5_p="ml/data/higgs/upper_limit_N_df_5.00_sys_err_5.00.p",
        res5_p_mc_only="ml/data/higgs/upper_limit_N_df_mc_only_5.00_sys_err_5.00.p",
        res10_p="ml/data/higgs/upper_limit_N_df_5.00_sys_err_10.00.p",
        res10_p_mc_only="ml/data/higgs/upper_limit_N_df_mc_only_5.00_sys_err_10.00.p",
        res20_p="ml/data/higgs/upper_limit_N_df_5.00_sys_err_20.00.p",
        res20_p_mc_only="ml/data/higgs/upper_limit_N_df_mc_only_5.00_sys_err_20.00.p",
    )
