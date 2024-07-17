import logging

import numpy as np

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import HEPPlot, set_size, step_hist_plot, style_setup
from ml.custom.HIGGS.analysis.fit_cuts import ClassifierCut
from ml.custom.HIGGS.analysis.utils import LABELS_MAP


class HistMaker:
    def __init__(self, model_name, classifier_model_name, N=2 * 10**6, save_dir="ml/custom/HIGGS/analysis/plots/hists"):
        """Make histograms for fitting.

        Parameters
        ----------
        model_name : str
            Name of the generative model.
        classifier_model_name : str
            Name of the classifier model.
        N : int, optional
            Number of ML generated events, by default 2 * 10**6.  - should have it fixed at this large value...
        save_dir : str, optional
            Save dir for plots, by default "ml/custom/HIGGS/analysis/plots/hists".

        Note
        ----
        1 Picobarn [pb] = 1000 Femtobarn [fb]

        References
        [1] - https://ipnp.cz/scheirich/?page_id=292

        """
        self.model_name = model_name
        self.classifier_model_name = classifier_model_name
        self.N = N
        self.save_dir = mkdir(save_dir)

        self.histograms, self.bin_edges = dict(), None
        self.weighted_histograms = dict()
        self.errors = dict()
        self.lumi = None

        self.bkg_xsec, self.sig_xsec = None, None

    def setup(
        self,
        N_gen_bkg,
        N_mc_bkg,
        N_mc_sig,
        cut_threshold,
        n_bins,
        bin_range,
        use_weights=False,
        N_data_sig=None,
        cut_variable=False,
        mc_only=False,
        bkg_only=False,
        bkg_xsec=None,
    ):
        # if not (N_gen_bkg >= N_mc_bkg and N_gen_bkg >= N_mc_sig):
        if not (N_gen_bkg >= N_mc_bkg):
            logging.warning("[yellow]N_gen_bkg >= N_mc_bkg and N_gen_bkg >= N_mc_sig![/yellow]")

        self.N_gen_bkg = N_gen_bkg
        self.N_mc_bkg = N_mc_bkg
        self.N_mc_sig = N_mc_sig

        self.mc_only = mc_only
        self.bkg_only = bkg_only

        if N_data_sig is None:
            N_data_sig = N_mc_sig

        self.N_data_sig = N_data_sig

        self.cut_threshold = cut_threshold
        self.cut_variable = cut_variable

        events = self.run_classifier_setup(use_weights)
        # BPK Fix Asimov data to always be asymptotic - also for MC only ...

        # get correct number of events in signal MC and sig data
        eff = len(events["sig"]) / (self.N_mc_sig + self.N_data_sig)
        N_mc_sig_data_cut = int(
            self.N_data_sig * eff
        )  # misleading name - better N_data_sig_cut ? signal is always MC only!
        N_mc_sig_cut = len(events["sig"]) - N_mc_sig_data_cut

        events["sig"], events["sig_data"] = (
            events["sig"][:N_mc_sig_cut],  # use in fit, I guess
            events["sig"][N_mc_sig_cut : N_mc_sig_cut + N_mc_sig_data_cut],  # asimov sig N and events
        )

        # get correct number of events in bkg MC and sig data
        lbkg = len(events["bkg"])
        eff = lbkg / (
            self.N_mc_sig + self.N_mc_bkg
        )  # N_mc_sig = large = 10^6 for MC stats in Asimov data -> inf (closure)
        N_mc_bkg_data_cut = int(self.N_mc_bkg * eff)
        N_mc_bkg_cut = lbkg - N_mc_bkg_data_cut

        events["bkg"], events["bkg_data"] = (
            events["bkg"][:N_mc_bkg_cut],  # use in fit, I guess
            events["bkg"][N_mc_bkg_cut : N_mc_bkg_cut + N_mc_bkg_data_cut],  # asimov sig N and events
        )
        logging.info(f"Splits ML bkg-to-data eff={eff}, all={lbkg} bkg={N_mc_bkg_cut} data_bkg={N_mc_bkg_data_cut} ")

        if self.mc_only:
            # replace ML bkg with MC bkg
            logging.info("[yellow]Replacing ML bkg with MC bkg![/yellow]")
            N_gen_bkg_data_cut = int(self.N_gen_bkg * eff)
            events[self.model_name] = events["bkg"][:N_gen_bkg_data_cut]  # replace ML with MC

            logging.info(f"Replacing ML with  MC len={len(events[self.model_name] )} ")

        events = {k: v.flatten() if isinstance(v, np.ndarray) else v.numpy().flatten() for k, v in events.items()}

        if bkg_xsec is None:
            logging.info("[yellow]Setting bkg xsec to 1 fb![/yellow]")
            self.bkg_xsec = 1.0
        else:
            self.bkg_xsec = bkg_xsec

        self.histograms, self.bin_edges = self.get_histograms(events, n_bins, bin_range)

        return events, self.histograms

    def run_classifier_setup(self, use_weights):
        clip_dct_N = {
            self.model_name: self.N_gen_bkg,
            "bkg": self.N_mc_sig
            + self.N_mc_bkg,  # BPK define large MC stats by adding N_mc_sig -  even for Asimov signal (and rescale!)
            "sig": self.N_mc_sig + self.N_data_sig,
        }

        classifier_cut = ClassifierCut(
            model_name=self.model_name,
            N=self.N,
            classifier_name=self.classifier_model_name,
            use_c2st_weights=use_weights,
            clip_dct_N=clip_dct_N,
        )
        return self._get_cut_samples(classifier_cut)

    def _get_cut_samples(self, cut_obj):
        if self.cut_variable:
            cut_variables = True
        else:
            cut_variables = False
        cut_obj.cut(cut_threshold=self.cut_threshold, cut_variables=cut_variables)

        samples_dct = cut_obj.samples_dct_cut

        if cut_variables:
            selection = cut_obj.selection
            variable_idx = selection[selection["feature"] == self.cut_variable].index[0]
            samples_dct = {k: v[:, variable_idx] for k, v in samples_dct.items()}

        return samples_dct

    def get_histograms(self, events, n_bins, bin_range, eps=1e-6):
        for k, v in events.items():
            if len(v) == 0:
                raise ValueError(f"Empty array for {k}!")

        # make bins edges from MC sig and MC bkg
        binning_array = np.concatenate([events["bkg"], events["sig"]])  # Asimov data
        bin_edges = np.histogram_bin_edges(binning_array, bins=n_bins, range=bin_range)

        gen_bkg_hist, _ = np.histogram(events[self.model_name], bins=bin_edges)
        mc_bkg_hist, _ = np.histogram(events["bkg"], bins=bin_edges)
        mc_sig_hist, _ = np.histogram(events["sig"], bins=bin_edges)
        mc_sig_data, _ = np.histogram(events["sig_data"], bins=bin_edges)
        mc_bkg_data, _ = np.histogram(events["bkg_data"], bins=bin_edges)
        # TEST: what happens without eps? (with eps=0.)
        histograms = {
            "bkg_gen": gen_bkg_hist + eps,
            "bkg_mc": mc_bkg_hist + eps,
            "sig_mc": mc_sig_hist + eps,
            "sig_mc_data": mc_sig_data + eps,
            "bkg_mc_data": mc_bkg_data + eps,
            "data": mc_bkg_hist + mc_sig_data + eps,
        }

        return histograms, bin_edges

    def make_fit_input(self, expected_bkg=None, expected_sig=None, scale_mc_sig=True):
        gen_bkg_hist, mc_bkg_hist, mc_sig_hist, mc_sig_data_hist = (
            self.histograms["bkg_gen"],
            self.histograms["bkg_mc"],
            self.histograms["sig_mc"],
            self.histograms["sig_mc_data"],
        )

        # per-bin eff
        eff_gen_bkg_hist = gen_bkg_hist / self.N_gen_bkg
        eff_mc_bkg_hist = mc_bkg_hist / self.N_mc_sig  # sig is the MC size variable!
        eff_mc_sig_hist = mc_sig_hist / self.N_mc_sig
        eff_mc_sig_data_hist = mc_sig_data_hist / self.N_data_sig
        # BPK enforce real closure on MC signal and MC background (if MC_only)
        eff_mc_sig_data_hist = eff_mc_sig_hist
        # if self.mc_only:
        #     eff_gen_bkg_hist = eff_mc_bkg_hist

        logging.info(
            (
                f"Efficiency of bkg ML: {eff_gen_bkg_hist.sum():.4f}, "
                f"bkg MC: {eff_mc_bkg_hist.sum():.4f}, "
                f"sig MC: {eff_mc_sig_hist.sum():.4f}, "
                f"sig MC data: {eff_mc_sig_data_hist.sum():.4f}"
            )
        )

        if expected_bkg is None:
            expected_bkg = self.histograms["bkg_mc_data"].sum()
            logging.info(f"[yellow]Using bkg MC for expected bkg! Setting {int(expected_bkg)}![/yellow]")

        if expected_sig is None:
            expected_sig = self.histograms["sig_mc_data"].sum()
            logging.info(f"[yellow]Using sig MC for expected sig! Setting {int(expected_sig)}![/yellow]")

        if expected_bkg > self.N:
            logging.warning("[red]Expected bkg > N generated! Fits might not behave well in this limit![/red]")

        # weights (this is sigma x L in [1])
        w_gen_bkg = expected_bkg / eff_gen_bkg_hist.sum()  # not used ?
        w_mc_bkg = expected_bkg / eff_mc_bkg_hist.sum()

        if scale_mc_sig:
            w_mc_sig = expected_sig / eff_mc_sig_hist.sum()
        else:
            w_mc_sig = self.N_mc_sig

        if self.bkg_only:
            expected_sig = 0.0
            logging.info("[red]Bkg only, overwriting expected sig in data to 0![/red]")

        w_mc_data_sig = expected_sig / eff_mc_sig_data_hist.sum()

        # scale by weights
        gen_bkg_hist = w_mc_bkg * eff_gen_bkg_hist  # need to take mc (data!), background ML model for fit
        mc_bkg_hist = w_mc_bkg * eff_mc_bkg_hist  # background MC model for data
        mc_sig_hist = w_mc_sig * eff_mc_sig_hist  # signal MC model for fit
        mc_sig_data_hist = w_mc_data_sig * eff_mc_sig_data_hist  # signal MC model for data

        # make fake data from MC
        data_hist = mc_bkg_hist + mc_sig_data_hist

        # BPK check - realclosure on MC - now done by default for bkg for MC_only
        # gen_bkg_hist = mc_bkg_hist
        # data_hist = mc_bkg_hist+mc_sig_hist -> now done by default for signal...

        # errors
        gen_bkg_err = w_mc_bkg * np.sqrt(eff_gen_bkg_hist * (1 - eff_gen_bkg_hist) / self.N_gen_bkg)
        mc_bkg_err = w_mc_bkg * np.sqrt(eff_mc_bkg_hist * (1 - eff_mc_bkg_hist) / self.N_mc_sig)
        mc_sig_err = w_mc_sig * np.sqrt(eff_mc_sig_hist * (1 - eff_mc_sig_hist) / self.N_mc_sig)
        data_err = np.sqrt(data_hist)

        self.weighted_histograms = {
            "bkg_gen": gen_bkg_hist,
            "bkg_mc": mc_bkg_hist,
            "sig_mc": mc_sig_hist,
            "data": data_hist,
            "sig_mc_data": mc_sig_data_hist,
        }

        self.errors = {
            "bkg_gen": gen_bkg_err,
            "bkg_mc": mc_bkg_err,
            "sig_mc": mc_sig_err,
            "data": data_err,
        }

        # set lumi
        bkg_lumi = w_mc_bkg / self.bkg_xsec

        self.sig_xsec = w_mc_data_sig / bkg_lumi
        sig_lumi = w_mc_data_sig / self.sig_xsec

        logging.info(f"[bold][green]Calculated sig xsec: {self.sig_xsec:.3f} fb![/green][/bold]")

        self.lumi = {
            "bkg": bkg_lumi,
            "sig": sig_lumi,  # is the same as bkg_lumi
            "data": bkg_lumi,
        }

        logging.info(f"[bold][green]Calculated lumi for data: {self.lumi['data']:.3f} fb^-1![/green][/bold]")

        return self.weighted_histograms, self.errors, self.lumi

    def plot_histograms(self, ylim=(0.5, 1.5), top_ylim=None, plot_separate_sig=False, weighted=True):
        if weighted:
            logging.info("[green]Plotting weighted histograms[/green]")
            histograms = self.weighted_histograms
            errors = self.errors
        else:
            logging.info("[green]Plotting unweighted histograms[/green]")
            histograms = self.histograms
            errors = {k: np.sqrt(v) for k, v in histograms.items()}

        logging.info(
            (
                f"Events in bkg ML: {int(histograms['bkg_gen'].sum())}, "
                f"bkg MC: {int(histograms['bkg_mc'].sum())}, "
                f"sig MC: {int(histograms['sig_mc'].sum())}, "
                f"data: {int(histograms['data'].sum())}, "
                f"data sig: {int(histograms['sig_mc_data'].sum())}"
            )
        )

        if not plot_separate_sig:
            hep_plot = HEPPlot(
                data=histograms["data"],
                mc=[
                    histograms["bkg_gen"],
                    histograms["sig_mc"],
                ],
                mc_err=[
                    errors["bkg_gen"],
                    errors["sig_mc"],
                ],
                bin_edges=self.bin_edges,
            )
        else:
            hep_plot = HEPPlot(
                data=histograms["data"],
                mc=[histograms["bkg_gen"]],
                mc_err=[errors["bkg_gen"]],
                bin_edges=self.bin_edges,
            )

        hep_plot.setup_figure(figsize=(8, 8))

        hep_plot.ax.set_ylabel("$N$")
        hep_plot.ax.set_xlim([self.bin_edges.min(), self.bin_edges.max()])

        if self.cut_variable:
            hep_plot.ax.set_xlabel(LABELS_MAP[self.cut_variable])
        else:
            hep_plot.ax.set_xlabel("classifier output")

        hep_plot.plot_data(label=f"data MC, $L=${self.lumi['data']:.0f} " + r"fb$^{-1}$", color="black")

        if not plot_separate_sig:
            hep_plot.plot_mc(labels=["bkg ML", "sig MC"], colors=["C0", "C1"])
            hep_plot.plot_ratio(lower_ylabel="MC / ML", ylim=ylim)
        else:
            hep_plot.plot_mc(labels=["expected bkg ML"], colors=["C0"])
            hep_plot.plot_ratio(lower_ylabel="MC / ML", ylim=ylim)

        if plot_separate_sig:
            step_hist_plot(
                hep_plot.ax,
                histograms["sig_mc"],
                self.bin_edges,
                color="C1",
                label="expected sig MC",
                lw=1.5,
            )

        hep_plot.ax.set_ylim(bottom=0.0, top=top_ylim)

        save_str = "ml_bkg_hist_weighted" if weighted else "ml_bkg_hist_unweighted"
        save_str += "_mc_only" if self.mc_only else ""
        save_str += "_bkg_only" if self.bkg_only else ""
        save_str += f" {self.cut_variable}" if self.cut_variable else ""
        save_str += f"_{self.lumi['data']:.1f}fb"
        save_str += "_add" if not plot_separate_sig else ""
        save_str.replace(" ", "_")

        save_str += "_NB_MCS"  # BPK extra label

        hep_plot.save(self.save_dir, save_str)


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    hist_maker = HistMaker(
        model_name="MADEMOG_flow_model_gauss_rank_best",
        classifier_model_name="BinaryClassifier_sigbkg_gauss_rank_best7",
    )

    # set to None to take expected of MC bkg and MC sig after cut
    expected_bkg = None  # 37000
    expected_sig = None  # 100

    N_gen = 10**4
    N_mc_bkg = 10**5
    sig_frac = 0.05

    hist_maker.setup(
        N_gen_bkg=N_gen,
        N_mc_bkg=N_mc_bkg,  # expected_bkg,
        N_mc_sig=10**6,  # MC statistics
        N_data_sig=int(N_mc_bkg * sig_frac),  # expected_sig,
        cut_threshold=0.55,
        cut_variable=False,
        use_weights=False,
        mc_only=True,  # org. False
        bkg_only=False,  # org. False
        n_bins=25,
        bin_range=(0.55, 1),
        bkg_xsec=1 * 1000,  # fb
    )
    hist_maker.make_fit_input(
        expected_bkg=expected_bkg,  # if None use after cut as expected
        expected_sig=expected_sig,
        scale_mc_sig=True,
    )

    hist_maker.plot_histograms(
        ylim=(0.5, 1.5),
        plot_separate_sig=False,
        weighted=True,
    )
    hist_maker.plot_histograms(
        ylim=(0.5, 1.5),
        plot_separate_sig=True,
        weighted=True,
    )
