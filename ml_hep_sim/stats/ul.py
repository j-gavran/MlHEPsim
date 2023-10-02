import copy
import logging
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhf.contrib.viz import brazil
from sklearn.utils import shuffle

from ml_hep_sim.data_utils.higgs.process_higgs_dataset import COLNAMES
from ml_hep_sim.pipeline.blocks import *
from ml_hep_sim.pipeline.pipes import *
from ml_hep_sim.stats.pyhf_json_specs import prep_data


def get_colnames_dict():
    """Make variable name to index mapping dict. For HIGGS."""
    colnames_mapping_dct = {}

    for i, name in enumerate(COLNAMES[1:]):
        colnames_mapping_dct[name] = i

    logging.warning("available variables: {}".format(colnames_mapping_dct))
    return colnames_mapping_dct


def generate_N_data_from_pipes(
    N,
    pipe_sig_str,
    pipe_bkg_str,
    run_name,
    device="cuda",
    class_run_name="Higgs_resnet_classifier_train_pipeline",
    var=None,
    rescale="logit_normal",
):
    """Depreciated. Use pipeline instead."""

    pipeline_path = f"ml_pipeline/{run_name}/"
    pipe_sig = Pipeline(pipeline_name=pipe_sig_str, pipeline_path=pipeline_path).load().pipes
    pipe_bkg = Pipeline(pipeline_name=pipe_bkg_str, pipeline_path=pipeline_path).load().pipes

    x1 = ModelLoaderBlock(device=device)(pipe_sig[0], pipe_sig[-1], pipe_sig[1])
    x2 = ModelLoaderBlock(device=device)(pipe_bkg[0], pipe_bkg[-1], pipe_bkg[1])

    # sig
    x3 = DataGeneratorBlock(N, model_type="flow", chunks=10, device=device)(x1)
    x4 = GeneratedDataVerifierBlock(
        save_data=False, device=device, rescale_data=False if rescale is not None else True
    )(x1, x3)

    # bkg
    x5 = DataGeneratorBlock(N, model_type="flow", chunks=10, device=device)(x2)
    x6 = GeneratedDataVerifierBlock(
        save_data=False, device=device, rescale_data=False if rescale is not None else True
    )(x2, x5)

    config = copy.deepcopy(pipe_sig[0].config)
    config["datasets"]["data_name"] = "higgs_bkg"
    config["datasets"]["data_params"]["subset_n"] = [0, 0, N]

    x71 = DatasetBuilderBlock(config=config)()
    x81 = ReferenceDataLoaderBlock(rescale_reference=rescale)(x71)

    config = copy.deepcopy(pipe_sig[0].config)
    config["datasets"]["data_name"] = "higgs_sig"
    config["datasets"]["data_params"]["subset_n"] = [0, 0, N]

    x72 = DatasetBuilderBlock(config=config)()
    x82 = ReferenceDataLoaderBlock(rescale_reference=rescale, device=device)(x72)

    class_train_pipeline = Pipeline(pipeline_name=class_run_name, pipeline_path="ml_pipeline/")
    class_train_pipeline.load()

    x9 = ModelLoaderBlock(device=device)(class_train_pipeline.pipes[0], class_train_pipeline.pipes[-1])

    if var is None:
        x10 = ClassifierRunnerBlock(save_data=False, device=device)(x4, x9)  # sig gen
        x11 = ClassifierRunnerBlock(save_data=False, device=device)(x6, x9)  # bkg gen

        x12 = ClassifierRunnerBlock(save_data=False, device=device)(x81, x9)  # MC bkg
        x13 = ClassifierRunnerBlock(save_data=False, device=device)(x82, x9)  # MC sig
    else:
        dct = get_colnames_dict()
        idx = dct[var]

        x10 = VariableExtractBlock(idx, save_data=False, device=device)(x4)
        x11 = VariableExtractBlock(idx, save_data=False, device=device)(x6)

        x12 = VariableExtractBlock(idx, save_data=False, device=device)(x81)
        x13 = VariableExtractBlock(idx, save_data=False, device=device)(x82)

    pipe = Pipeline()
    pipe.compose(x1, x2, x3, x4, x5, x6, x71, x81, x72, x82, x9, x10, x11, x12, x13)
    pipe.fit()

    sig_gen = pipe.pipes[-4].results
    bkg_gen = pipe.pipes[-3].results
    sig_mc = pipe.pipes[-1].results[: len(sig_gen)]
    bkg_mc = pipe.pipes[-2].results[: len(sig_gen)]

    return [sig_gen, bkg_gen, sig_mc, bkg_mc]


class UpperLimitCalculator:
    def __init__(
        self,
        sig,
        bkg,
        data,
        bkg_err,
        mc_err,
        N,
        bounds_low=0.1,
        bounds_up=5.0,
        rtol=0.01,
        eps=1e-3,
        scan_pts=None,
    ):
        """Class for upper limit params.

        Parameters
        ----------
        sig : np.array
            Histogram for signal.
        bkg : np.array
            Histogram for background.
        data : np.array
            Histogram for data.
        bkg_err : float
            Background error.
        N : int
            Number of points.
        bounds_low : float, optional
            Lower poi scan limit, by default 0.1.
        bounds_up : float, optional
            Upper poi scan limit, by default 5.0.
        rtol : float, optional
            Minimizer tolarance, by default 0.01.
        eps : float, optional
            Small value to add to histograms to avoid numerical issues, by default 1e-7.
        scan_pts : int, optional
            Number of poi scan points, if not None use pyhf.infer.intervals.upper_limits.upper_limit, by default None.

        Note
        ----
        If scan_pts is None calculate upper limit on poi [1]. If int given calculate grid of pois for brazil plot [2].

        References
        ----------
        [1] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.intervals.upper_limits.toms748_scan.html
        [2] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.intervals.upper_limits.upper_limit.html#pyhf.infer.intervals.upper_limits.upper_limit
        [3] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.hypotest.html#pyhf.infer.hypotest
        [4] - https://scikit-hep.org/pyhf/api.html#confidence-intervals

        """
        self.sig, self.bkg, self.data = sig + eps, bkg + eps, data + eps
        self.bkg_err = bkg_err
        self.mc_err = mc_err
        self.bounds_low, self.bound_up = bounds_low, bounds_up
        self.rtol = rtol
        self.eps = eps
        self.N = N

        if type(scan_pts) is int:
            self.scan_pts = np.linspace(bounds_low, bounds_up, scan_pts)
        else:
            self.scan_pts = scan_pts

    def get_upper_limit(self, **kwargs):
        return calculate_upper_limit([self], **kwargs)


def calculate_upper_limit(calcs, **kwargs):
    results = []

    for c in calcs:
        logging.warning(f"calculator {c} with N={c.N} in loop")
        spec = prep_data(c.sig, c.bkg, c.bkg_err, c.mc_err)
        model = pyhf.Model(spec)

        data = list(c.data) + model.config.auxdata

        obs_limit, exp_limits = pyhf.infer.intervals.upper_limits.toms748_scan(
            data,
            model,
            c.bounds_low,
            c.bound_up,
            level=0.05,
            rtol=c.rtol,  # https://scikit-hep.org/pyhf/_modules/pyhf/infer/utils.html#create_calculator
            # **kwargs,
        )
        results.append([c.N, obs_limit, exp_limits])

        logging.warning(f"calculator {c} with N={c.N} done")

    return results


def make_hists_from_samples(
    sig_gen,
    bkg_gen,
    sig_mc,
    bkg_mc,
    bins,
    bin_range,
    N_sig,
    N_bkg,
    use_gen_bkg=True,
    use_mc_all=False,
    return_all=False,
):
    """Depreciated. Use pipeline instead."""

    sig_gen = sig_gen[:N_sig]
    sig_mc = sig_mc[:N_sig]
    bkg_gen = bkg_gen[:N_bkg]
    bkg_mc = bkg_mc[:N_bkg]

    sig_bkg_gen = shuffle(np.concatenate([sig_gen, bkg_gen]))
    sig_bkg_mc = shuffle(np.concatenate([sig_mc, bkg_mc]))

    bin_edges = np.histogram_bin_edges(sig_bkg_mc, bins=bins, range=bin_range)

    sig_gen_hist = np.histogram(sig_gen, bins=bin_edges)[0]
    bkg_gen_hist = np.histogram(bkg_gen, bins=bin_edges)[0]
    sig_mc_hist = np.histogram(sig_mc, bins=bin_edges)[0]
    bkg_mc_hist = np.histogram(bkg_mc, bins=bin_edges)[0]

    sig_bkg_gen_hist = np.histogram(sig_bkg_gen, bins=bin_edges)[0]
    sig_bkg_mc_hist = np.histogram(sig_bkg_mc, bins=bin_edges)[0]

    if use_gen_bkg:
        return sig_gen_hist, bkg_gen_hist, sig_bkg_mc_hist  # sig, bkg, data
    if use_mc_all:
        return sig_mc_hist, bkg_mc_hist, sig_bkg_mc_hist
    if return_all:
        return sig_gen_hist, bkg_gen_hist, sig_mc_hist, bkg_mc_hist, sig_bkg_gen_hist, sig_bkg_mc_hist


class ExpectedEventCalculator:
    def __init__(
        self,
        sig_gen_full,
        bkg_gen_full,
        sig_mc_full,
        bkg_mc_full,
        *,
        sig_frac,
        N=None,
        lumi=None,
        xsec=None,
        bin_range=None,
        bins=20,
        use_gen_bkg=True,
    ):
        """Depreciated. Use pipeline instead."""

        self.sig_gen_full = sig_gen_full
        self.bkg_gen_full = bkg_gen_full
        self.sig_mc_full = sig_mc_full
        self.bkg_mc_full = bkg_mc_full

        self.sig_frac = sig_frac

        self.bin_range, self.bins, self.use_gen_bkg = bin_range, bins, use_gen_bkg

        self.lumi = lumi  # luminosity [1/fb]
        self.xsec = xsec  # cross section [fb]
        self.N = N

    def build(self):
        if self.N is None:
            self.N = min([len(self.sig_gen_full), len(self.bkg_gen_full), len(self.sig_mc_full), len(self.bkg_mc_full)])

        self.N_sig = int(self.expected_N * self.sig_frac)
        self.N_bkg = self.expected_N - self.N_sig

        self.sig, self.bkg, self.data = self.make_hist(
            self.sig_gen_full,
            self.bkg_gen_full,
            self.sig_mc_full,
            self.bkg_mc_full,
            self.bins,
            self.bin_range,
            self.use_gen_bkg,
        )

        return self.sig, self.bkg, self.data

    def make_hist(self, sig_gen, bkg_gen, sig_mc, bkg_mc, bins, bin_range, use_gen_bkg):
        return make_hists_from_samples(
            sig_gen, bkg_gen, sig_mc, bkg_mc, bins, bin_range, self.N_sig, self.N_bkg, use_gen_bkg=use_gen_bkg
        )

    @property
    def expected_N(self):
        if self.lumi is None:
            return self.N
        else:
            return int(self.lumi * self.xsec)

    @property
    def eff(self):
        return (self.sig.sum() + self.bkg.sum()) / (2 * self.N)

    @property
    def sf(self):
        return (self.data.sum()) / (self.sig.sum() + self.bkg.sum())

    def eff_sigma(self):
        eps = self.eff
        return np.sqrt((eps * (1 - eps)) / self.N)

    def plot(self):
        x = range(len(self.sig))
        plt.scatter(x, self.sig * self.sf)
        plt.scatter(x, self.bkg * self.sf)
        plt.scatter(x, self.data)
        plt.scatter(x, self.sig * self.sf + self.bkg * self.sf)
        plt.legend(["sig", "bkg", "data", "sig + bkg"])


def make_brazil_plot(poi_pts, results):
    """Pyhf brazil plot."""
    fig, ax = plt.subplots()
    brazil.plot_results(poi_pts, results, ax=ax)
    return ax


class UpperLimitScanner:
    def __init__(self, scan_vars, use_lumi=False, N_max=10**6):
        """Depreciated. Use pipeline instead."""

        self.scan_vars = scan_vars
        self.use_lumi = use_lumi
        self.N_max = N_max

    def generate_data(self, pipe_sig, pipe_bkg, run_name, var=None, rescale=None, **kwargs):
        sig_gen_full, bkg_gen_full, sig_mc_full, bkg_mc_full = generate_N_data_from_pipes(
            self.N_max,
            pipe_sig,
            pipe_bkg,
            run_name,
            var=var,
            rescale=rescale,
            **kwargs,
        )
        self.sig_gen_full, self.bkg_gen_full, self.sig_mc_full, self.bkg_mc_full = (
            sig_gen_full,
            bkg_gen_full,
            sig_mc_full,
            bkg_mc_full,
        )
        return sig_gen_full, bkg_gen_full, sig_mc_full, bkg_mc_full

    def configure_expected_calculator(
        self,
        lumi=None,
        xsec=None,
        sig_frac=0.1,
        bin_range=(0.15, 3),
        bins=20,
        use_gen_bkg=True,
    ):
        exp_calculator = ExpectedEventCalculator(
            self.sig_gen_full,
            self.bkg_gen_full,
            self.sig_mc_full,
            self.bkg_mc_full,
            lumi=lumi,
            xsec=xsec,
            sig_frac=sig_frac,
            bin_range=bin_range,
            bins=bins,
            use_gen_bkg=use_gen_bkg,
        )
        self.exp_calculator = exp_calculator
        return exp_calculator

    def make_ul_calculators(self, bkg_err, bounds_low=0.1, bounds_up=5.0, rtol=0.01, scan_pts=None):
        self.ul_calcs = []
        self.exp_calculators = []

        for scan_var in self.scan_vars:
            exp_calculator = self.exp_calculator
            if self.use_lumi:
                exp_calculator.lumi = scan_var
            else:
                exp_calculator.N = scan_var

            sig, bkg, data = exp_calculator.build()

            calc = UpperLimitCalculator(
                sig,
                bkg,
                data,
                bkg_err,
                scan_var,
                bounds_low=bounds_low,
                bounds_up=bounds_up,
                rtol=rtol,
                scan_pts=scan_pts,
            )

            self.ul_calcs.append(calc)
            self.exp_calculators.append(copy.deepcopy(exp_calculator))

        return self.ul_calcs

    def scan_upper_limits(self, workers=1, parse_results=True, **kwargs):
        """For kwargs see https://scikit-hep.org/pyhf/_generated/pyhf.infer.hypotest.html#pyhf.infer.hypotest."""
        ul_calcs_splits = np.array_split(self.ul_calcs, workers)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for ul_calc_split in ul_calcs_splits:
                futures.append(executor.submit(calculate_upper_limit, ul_calc_split, **kwargs))

        results = []
        for future in futures:
            results.append(future.result())

        if parse_results:
            return self._parse_results(results)
        else:
            return results

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


if __name__ == "__name__":
    run_name = "Higgs_Glow"
    pipe_sig_str = f"{run_name}_flow_blocks_10_sig_train_pipe"
    pipe_bkg_str = f"{run_name}_flow_blocks_10_train_pipe"

    scanner = UpperLimitScanner(scan_pts=np.linspace(10, 300, 22).astype(int), use_lumi=True, N_max=10**6)
    scanner.generate_data(pipe_sig_str, pipe_bkg_str, run_name, var="lepton pT", rescale="logit_normal")
    scanner.configure_expected_calculator(
        lumi=30.8,
        xsec=10**3,
        sig_frac=0.1,
        bins=20,
        bin_range=(0.15, 3),
        use_gen_bkg=True,
    )
    scanner.make_ul_calculators(bkg_err=0.1, scan_pts=None)

    res = scanner.scan_upper_limits(workers=22)
