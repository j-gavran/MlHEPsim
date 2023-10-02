import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyhf

from ml_hep_sim.analysis.hists_pipeline import (
    MakeHistsFromSamplesLumi,
    get_hists_pipeline,
)
from ml_hep_sim.pipeline.blocks import Block
from ml_hep_sim.pipeline.pipes import Pipeline
from ml_hep_sim.stats.pyhf_json_specs import prep_data
from ml_hep_sim.stats.ul import (
    ProcessPoolExecutor,
    UpperLimitCalculator,
    calculate_upper_limit,
)


class UpperLimitScannerBlock(Block):
    def __init__(
        self,
        bkg_err,
        mc_test=False,
        lumi=None,
        lumi_histograms=None,
        lumi_errors=None,
        par_bounds=None,
        workers=1,
    ):
        """Block to scan upper limits.

        Uses ml_hep_sim.stats.ul.UpperLimitCalculator with histograms from ml_hep_sim.analysis.hists_pipeline.MakeHistsFromSamples.

        Parameters
        ----------
        bkg_err : float
            Background uncertainty.
        mc_test : bool, optional
            If True use MC histograms, by default False.
        lumi : MakeHistsFromSamplesLumi, optional
            Block obj, by default None.
        lumi_histograms : MakeHistsFromSamplesLumi, optional
            Block obj, by default None.
        workers : int, optional
            Number of workers to use, by default 1.
        """
        super().__init__()
        self.bkg_err = bkg_err
        self.lumi = lumi
        self.lumi_histograms = lumi_histograms
        self.lumi_errors = lumi_errors
        self.workers = workers
        self.mc_test = mc_test
        self.par_bounds = par_bounds

        self.ul_calcs = []
        self.results = []

    def make_ul_calculators(self, bounds_low=0.1, bounds_up=10.0, rtol=0.1, scan_pts=None):
        """Make UpperLimitCalculator objects for each luminosity point.

        Parameters
        ----------
        bounds_low : float, optional
            Lower bound of scan, by default 0.1.
        bounds_up : float, optional
            Upper bound of scan, by default 5.0.
        rtol : float, optional
            Relative tolerance of scan, by default 0.01.
        scan_pts : int, optional
            See UpperLimitCalculator class, by default None.
        """
        lumis = np.linspace(*self.lumi)
        for lumi, hist, err in zip(lumis, self.lumi_histograms, self.lumi_errors):
            if self.mc_test:
                sig, bkg, data = hist["sig_mc"], hist["bkg_mc"], hist["data_mc"]
            else:
                sig, bkg, data = hist["sig_mc"], hist["bkg_gen"], hist["data_mc"]

            calc = UpperLimitCalculator(
                sig,
                bkg,
                data,
                self.bkg_err,
                err["nu_b_ml"],
                lumi,
                bounds_low=bounds_low,
                bounds_up=bounds_up,
                rtol=rtol,
                scan_pts=scan_pts,
            )

            self.ul_calcs.append(calc)

        return self.ul_calcs

    def scan_upper_limits(self, parse_results=True, **kwargs):
        """For kwargs see https://scikit-hep.org/pyhf/_generated/pyhf.infer.hypotest.html#pyhf.infer.hypotest."""
        pyhf.set_backend("numpy", "minuit")

        ul_calcs_splits = np.array_split(self.ul_calcs, self.workers)

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for ul_calc_split in ul_calcs_splits:
                futures.append(executor.submit(calculate_upper_limit, ul_calc_split, **kwargs))

        for future in futures:
            self.results.append(future.result())

        if parse_results:
            self.results = self._parse_results(self.results)

        return self.results

    @staticmethod
    def _parse_results(results):
        dct = {
            "N": [],
            "cls_obs": [],
            "cls_exp": [],
            "minus_sigma_2": [],
            "minus_sigma_1": [],
            "plus_sigma_1": [],
            "plus_sigma_2": [],
        }

        for r in results:
            for w_r in r:
                cls_obs = w_r[1]
                minus_sigma_2, minus_sigma_1, cls_exp, plus_sigma_1, plus_sigma_2 = w_r[2]
                dct["N"].append(w_r[0])
                dct["cls_obs"].append(float(cls_obs))
                dct["cls_exp"].append(float(cls_exp))
                dct["minus_sigma_2"].append(float(minus_sigma_2))
                dct["minus_sigma_1"].append(float(minus_sigma_1))
                dct["plus_sigma_2"].append(float(plus_sigma_2))
                dct["plus_sigma_1"].append(float(plus_sigma_1))

        df = pd.DataFrame(dct)
        df.sort_values(by=["N"], inplace=True)
        df.reset_index(inplace=True, drop=True)

        return df

    def run(self):
        self.make_ul_calculators()
        return self.scan_upper_limits(par_bounds=self.par_bounds)


def get_ul_pipeline(
    lumi_start=10,
    lumi_end=300,
    lumi_step=24,
    xsec=10,
    bkg_err=0.1,
    bins=22,
    use_classifier=False,
    bin_range=(0, 4),
    N_gen=10**6,
):
    hists_pipeline = get_hists_pipeline(use_classifier=use_classifier)
    hists_pipeline.pipes = hists_pipeline.pipes[:-1]  # replace with lumi block

    lumi = [lumi_start, lumi_end, lumi_step]

    sig_fracs = np.linspace(0.01, 0.1, 6)  # do different signal fraction (injections), alpha

    # set mu and gamma (mc correction) bounds
    poi_bound = (0, 100.0)
    gamma_bound = (1e-10, 100.0)

    bounds = []
    for b in range(bins + 1):
        if b == 0:
            bounds.append(poi_bound)
        else:
            bounds.append(gamma_bound)

    b_sig_bkg = hists_pipeline.pipes[-1]  # aggregate block
    hists_ul_blocks = []

    # MC
    for sf in sig_fracs:
        hists_block = MakeHistsFromSamplesLumi(
            bin_range=bin_range,
            N_gen=N_gen,
            bins=bins,
            alpha=sf,
            lumi=lumi,
            xsec=xsec,
        )(b_sig_bkg)
        ul_block = UpperLimitScannerBlock(bkg_err=bkg_err, workers=24, par_bounds=bounds, mc_test=False)(hists_block)

        hists_ul_blocks.append(hists_block)
        hists_ul_blocks.append(ul_block)

    # ML
    for sf in sig_fracs:
        hists_block = MakeHistsFromSamplesLumi(
            bin_range=bin_range,
            N_gen=N_gen,
            bins=bins,
            alpha=sf,
            lumi=lumi,
            xsec=xsec,
        )(b_sig_bkg)
        ul_block = UpperLimitScannerBlock(bkg_err=bkg_err, workers=24, par_bounds=bounds, mc_test=True)(hists_block)

        hists_ul_blocks.append(hists_block)
        hists_ul_blocks.append(ul_block)

    # results locations in pipes list for both MC and ML
    idxs_gen = [-1, -3, -5, -7, -9, -11]
    idxs_mc = [-13, -15, -17, -19, -21, -23]

    pipe = Pipeline()
    pipe.compose(hists_pipeline, hists_ul_blocks)

    return pipe, idxs_gen, idxs_mc


class PullBlock(Block):
    def __init__(self, bkg_err, cut_histograms=None, histograms=None, errors=None, mc_test=False, gamma_labels=True):
        """Block to calculate pulls.

        https://pyhf.github.io/pyhf-tutorial/PullPlot.html

        Parameters
        ----------
        bkg_err : float
            Background uncertainty (sys error).
        cut_histograms : MakeHistsFromSamplesBlock
            Block obj, by default None.
        histograms : MakeHistsFromSamplesBlock
            Block obj, by default None.
        errors : MakeHistsFromSamplesBlock
            Block obj, by default None.
        mc_test : bool, optional
            If True use MC data (for comparison), by default False.
        gamma_labels : bool, optional
            If True use gamma labels, by default True.
        """
        super().__init__()
        self.bkg_err = bkg_err

        self.cut_histograms = cut_histograms
        self.histograms = histograms
        self.errors = errors

        self.mc_test = mc_test
        self.gamma_labels = gamma_labels

        self.bestfit = None
        self.results = None  # pulls, pullerr, errors (including mu error at index 0), labels

    def mle_fit(self):
        pyhf.set_backend("numpy", "minuit")

        if self.mc_test:
            sig, bkg, data = self.histograms["sig_mc"], self.histograms["bkg_mc"], self.histograms["data_mc"]
        else:
            sig, bkg, data = self.histograms["sig_gen"], self.histograms["bkg_gen"], self.histograms["data_mc"]

        eps = 1e-12
        spec = prep_data(sig + eps, bkg + eps, self.bkg_err, mc_err=self.errors["nu_b_ml"])

        # par_bounds = [0, 10]

        model = pyhf.Model(spec)
        observations = list(data) + model.config.auxdata
        result, twice_nll = pyhf.infer.mle.fit(
            observations,
            model,
            return_uncertainties=True,
            return_fitted_val=True,
            # init_pars=[1.0 for i in range(len(bkg) + 1)],
            # par_bounds=[par_bounds] * (len(bkg) + 1),
        )

        bestfit, errors = result.T
        self.bestfit = [bestfit, twice_nll]

        return model, bestfit, errors

    def run(self):
        model, bestfit, errors = self.mle_fit()

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

        if self.gamma_labels:
            labels = np.array(["$\gamma_{" + "{}".format(i) + "}$" for i in range(len(labels))])

        self.results = [pulls, pullerr, errors, labels]

        return self.results


def pull_plot(pulls, pullerr, errors, labels, save=None, l=None, title=None, text=False):
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

    if l:
        ax.axvline(l, c="r", ls="--")

    if title:
        ax.set_title(title, fontsize=18)

    if text:
        ax.text(0.5, 1.5, r"Relative systematic$\longrightarrow$", color="red", fontsize=18)
        ax.text(31, 1.5, r"MC statistical uncertainty$\longrightarrow$", color="red", fontsize=18)

    # error > 1
    error_gt1 = np.argmax(errors > 1) - 0.5
    ax.axvline(x=error_gt1, color="red", linestyle="--")
    # ax.text(error_gt1 + 0.1, 1.5, r"$\sigma \geq 1 \longrightarrow$", color="red", fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
