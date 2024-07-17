import logging
import os
import pickle

import numpy as np
import pyhf

from ml.custom.HIGGS.analysis.fit_hists import HistMaker


def get_pyhf_model(bkg, bkg_err, sig, bkg_sys_perc):
    assert (
        isinstance(bkg, np.ndarray) and isinstance(bkg_err, np.ndarray) and isinstance(sig, np.ndarray)
    ), "All inputs should be np arrays!"

    spec = {
        "channels": [
            {
                "name": "fit_region",
                "samples": [
                    {
                        "name": "background",
                        "data": list(bkg),
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": list(bkg * bkg_sys_perc),
                            },
                            {
                                "name": "mc_staterror",
                                "type": "staterror",
                                "data": list(bkg_err),
                            },
                        ],
                    },
                    {
                        "name": "signal",
                        "data": list(sig),
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            },
                        ],
                    },
                ],
            },
        ]
    }

    return pyhf.Model(spec)


class Template:
    def __init__(
        self,
        sig,
        bkg,
        data,
        bkg_stat_err,
        sig_stat_err,
        sys_err,
        bin_edges,
        lumi=None,
        bounds_low=0.1,
        bounds_up=5.0,
        rtol=0.01,
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
        bkg_stat_err : float
            Background error.
        sig_stat_err : float
            Signal error.
        sys_err : float
            Systematic error of background.
        lumi : float, optional
            Luminosity, by default None.
        bin_edges : np.array
            Bin edges for the histograms.
        bounds_low : float, optional
            Lower poi scan limit, by default 0.1.
        bounds_up : float, optional
            Upper poi scan limit, by default 5.0.
        rtol : float, optional
            Minimizer tolarance, by default 0.01.

        References
        ----------
        [1] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.intervals.upper_limits.toms748_scan.html
        [2] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.intervals.upper_limits.upper_limit.html#pyhf.infer.intervals.upper_limits.upper_limit
        [3] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.hypotest.html#pyhf.infer.hypotest
        [4] - https://scikit-hep.org/pyhf/api.html#confidence-intervals

        """
        self.sig, self.bkg, self.data = sig, bkg, data
        self.bkg_stat_err, self.sig_stat_err = bkg_stat_err, sig_stat_err
        self.sys_err = sys_err
        self.bin_edges = bin_edges
        self.lumi = lumi

        self.bounds_low, self.bound_up = bounds_low, bounds_up
        self.rtol = rtol

        self.model = get_pyhf_model(bkg=bkg, bkg_err=bkg_stat_err, sig=sig, bkg_sys_perc=sys_err)

        logging.info(f"Template created for {self}!")

    def __str__(self):
        return f"Template(bkg_sum={int(np.sum(self.bkg))}, sig_sum={int(np.sum(self.sig))}, data_sum={int(np.sum(self.data))}, sys_err={self.sys_err}, lumi={self.lumi['data']:.2f})"


class FitSetup:
    def __init__(self, model_name, classifier_model_name, sys_err=0.1, par_bounds=None, inits=1.0):
        super().__init__()
        self.model_name = model_name
        self.classifier_model_name = classifier_model_name
        self.sys_err = sys_err

        if par_bounds is None:
            par_bounds = {"mu": (0, 10.0), "gamma": (1e-6, 10.0)}

        self.par_bounds, self.inits = par_bounds, inits

        self.n_bins, self.mc_only, self.bkg_only, self.cut_variable = None, False, False, None

        self.N_gen = None
        self.N_mc_bkg_lst, self.N_data_sig_lst = None, None
        self.N_mc_bkg, self.N_data_sig = None, None
        self.N_gen_lst = None

        self.templates = []

    def _setup_hists(
        self,
        N_gen,
        N_mc_bkg,
        N_data_sig,
        cut_threshold=0.55,
        cut_variable=False,
        use_weights=False,
        mc_only=False,
        bkg_only=False,
        n_bins=25,
        bin_range=(0.55, 1),
        bkg_xsec=1 * 1000,  # fb
        expected_bkg=None,  # if None use after cut as expected
        expected_sig=None,
        scale_mc_sig=True,
    ):
        hist_maker = HistMaker(
            # N=2 * N_gen,
            # N=20 * int(N_gen*0.1), #BPK hack for 20 cunks in cuts parallelisation?
            N=2 * 10**6,  # BPK - source of all evil if dependent on N_gen? keep constant? (this is the default...)
            model_name=self.model_name,
            classifier_model_name=self.classifier_model_name,
        )
        hist_maker.setup(
            N_gen_bkg=N_gen,
            N_mc_bkg=N_mc_bkg,
            N_mc_sig=10**6,  # keep as a fixed ~infinite number? not to interfere with N_gen variation...
            # N_mc_sig=N_gen,
            N_data_sig=N_data_sig,
            cut_threshold=cut_threshold,
            cut_variable=cut_variable,
            use_weights=use_weights,
            mc_only=mc_only,
            bkg_only=bkg_only,
            n_bins=n_bins,
            bin_range=bin_range,
            bkg_xsec=bkg_xsec,
        )
        hist_maker.make_fit_input(
            expected_bkg=expected_bkg,
            expected_sig=expected_sig,
            scale_mc_sig=scale_mc_sig,
        )

        return hist_maker.weighted_histograms, hist_maker.errors, hist_maker.lumi, hist_maker.bin_edges

    def setup_templates(
        self,
        N_gen,
        N_mc_bkg_lst,
        N_data_sig_lst,
        n_bins=25,
        bounds_low=0.0,
        bounds_up=5.0,
        rtol=0.1,
        cache_dir="ml/data/higgs",
        mc_only=False,
        bkg_only=False,
        cut_variable=False,
        scale_mc_sig=True,
        **kwargs,
    ):

        self.n_bins = n_bins
        self.mc_only = mc_only
        self.bkg_only = bkg_only
        self.cut_variable = cut_variable

        self.N_gen = N_gen
        self.N_mc_bkg_lst, self.N_data_sig_lst = N_mc_bkg_lst, N_data_sig_lst

        for i, (N_mc_bkg, N_data_sig) in enumerate(zip(N_mc_bkg_lst, N_data_sig_lst)):

            file_name = f"{cache_dir}/{self.model_name}_b{n_bins}_{int(N_mc_bkg)}_{int(N_data_sig)}_{self.N_gen}_hists"

            if self.mc_only:
                logging.info("[blue]Using MC only![/blue]")
                file_name += "_mc_only"

            if self.bkg_only:
                logging.info("[blue]Using bkg only![/blue]")
                file_name += "_bkg_only"

            if self.cut_variable:
                file_name += f"_{self.cut_variable}"

            if not scale_mc_sig:
                logging.info("[blue]Not scaling MC signal![/blue]")
                file_name += "_no_sig_scaling"

            file_name += ".p"

            if os.path.exists(file_name):
                logging.info(f"[blue]{i}: Loading hists for {N_mc_bkg} bkg and {N_data_sig} sig events![/blue]")
                hists, errors, lumi, bin_edges = pickle.load(open(file_name, "rb"))
            else:
                logging.info(f"[blue]{i}: Setting up hists for N_mc_bkg={N_mc_bkg} and N_data_sig={N_data_sig}![/blue]")
                hists, errors, lumi, bin_edges = self._setup_hists(
                    N_gen,
                    N_mc_bkg,
                    N_data_sig,
                    mc_only=mc_only,
                    bkg_only=bkg_only,
                    cut_variable=cut_variable,
                    scale_mc_sig=scale_mc_sig,
                    **kwargs,
                )
                pickle.dump((hists, errors, lumi, bin_edges), open(file_name, "wb"))

            logging.info(f"[green]Signal fraction: {N_data_sig / N_mc_bkg:.2f}[/green]")

            template = Template(
                sig=hists["sig_mc"],
                bkg=hists["bkg_gen"],
                data=hists["data"],
                bkg_stat_err=errors["bkg_gen"],
                sig_stat_err=errors["sig_mc"],
                bin_edges=bin_edges,
                lumi=lumi,
                sys_err=self.sys_err,
                bounds_low=bounds_low,
                bounds_up=bounds_up,
                rtol=rtol,
            )

            self.templates.append(template)

        return self.templates

    # BPK add templates for different N_ML (N_gen)
    def setup_templates_N(
        self,
        N_gen_lst,
        N_mc_bkg,
        N_data_sig,
        n_bins=25,
        bounds_low=0.0,
        bounds_up=5.0,
        rtol=0.1,
        cache_dir="ml/data/higgs",
        mc_only=False,
        bkg_only=False,
        cut_variable=False,
        scale_mc_sig=True,
        **kwargs,
    ):

        self.n_bins = n_bins
        self.mc_only = mc_only
        self.bkg_only = bkg_only
        self.cut_variable = cut_variable

        self.N_gen_lst = N_gen_lst
        self.N_mc_bkg, self.N_data_sig = N_mc_bkg, N_data_sig

        for i, N_gen in enumerate(N_gen_lst):

            logging.info(f"[green]Requested ML events: {int(N_gen)}[/green]")

            file_name = f"{cache_dir}/{self.model_name}_b{n_bins}_{int(N_mc_bkg)}_{int(N_data_sig)}_{N_gen}_N_hists"

            if self.mc_only:
                logging.info("[blue]Using MC only![/blue]")
                file_name += "_mc_only"

            if self.bkg_only:
                logging.info("[blue]Using bkg only![/blue]")
                file_name += "_bkg_only"

            if self.cut_variable:
                file_name += f"_{self.cut_variable}"

            if not scale_mc_sig:
                logging.info("[blue]Not scaling MC signal![/blue]")
                file_name += "_no_sig_scaling"

            file_name += ".p"

            logging.info(f"[green]Requested file: {file_name}[/green]")
            # BPK: override and reuse lumi=N_gen
            if os.path.exists(file_name):
                logging.info(
                    f"[blue]{i}: Loading hists for {N_gen} ML generated, {N_mc_bkg} bkg and {N_data_sig} sig events![/blue]"
                )
                hists, errors, lumi, bin_edges = pickle.load(open(file_name, "rb"))
            else:
                logging.info(
                    f"[blue]{i}: Setting up hists for for N_gen={N_gen} ML generated, N_mc_bkg={N_mc_bkg} and N_data_sig={N_data_sig}![/blue]"
                )
                hists, errors, lumi, bin_edges = self._setup_hists(
                    N_gen,
                    N_mc_bkg,
                    N_data_sig,
                    mc_only=mc_only,
                    bkg_only=bkg_only,
                    cut_variable=cut_variable,
                    scale_mc_sig=scale_mc_sig,
                    **kwargs,
                )
                lumi = {
                    "data": N_gen,  # BPK: re-purpose lumi ...
                }
                pickle.dump((hists, errors, lumi, bin_edges), open(file_name, "wb"))

            logging.info(f"[green]Signal fraction: {N_data_sig / N_mc_bkg:.2f}[/green]")

            template = Template(
                sig=hists["sig_mc"],
                bkg=hists["bkg_gen"],
                data=hists["data"],
                bkg_stat_err=errors["bkg_gen"],
                sig_stat_err=errors["sig_mc"],
                bin_edges=bin_edges,
                lumi=lumi,
                sys_err=self.sys_err,
                bounds_low=bounds_low,
                bounds_up=bounds_up,
                rtol=rtol,
            )

            self.templates.append(template)

        return self.templates

    def _set_par_bounds(self):
        bounds = []

        for b in range(2 * self.n_bins + 1):
            if b == 0:
                bounds.append(self.par_bounds["mu"])
            else:
                bounds.append(self.par_bounds["gamma"])

        return bounds

    def _set_init(self):
        inits = []

        for b in range(2 * self.n_bins + 1):
            if b == 0:
                inits.append(self.inits)
            else:
                inits.append(self.inits)

        return inits
