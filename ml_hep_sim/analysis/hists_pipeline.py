import logging

import numpy as np

from ml_hep_sim.analysis.cut_pipeline import get_cut_pipeline
from ml_hep_sim.analysis.utils import SigBkgBlock, get_colnames_dict
from ml_hep_sim.pipeline.blocks import Block, VariableExtractBlock
from ml_hep_sim.pipeline.pipes import Pipeline


class MakeHistsFromSamples(Block):
    def __init__(
        self,
        bin_range,
        N_gen,
        bins=22,
        sig_generated_data=None,
        bkg_generated_data=None,
        sig_reference_data=None,
        bkg_reference_data=None,
        N_bkg=None,
        N_sig=None,
        alpha=None,
        bonly=False,
        scale_by_alpha=True,
    ):
        """Make histograms from generated (ML) and reference (MC) samples.

        Note
        ----
        If alpha not given and both N_sig and N_bkg are not None -> alpha = N_sig / N_bkg.

        Parameters
        ----------
        bin_range : tuple
            Lower and upper range for bins.
        N_gen : int
            Number of generated events.
        bins : int, optional
            Number of bins, by default 22.
        sig_generated_data : SigBkgBlock
            Block obj, by default None.
        bkg_generated_data : SigBkgBlock
            Block obj, by default None.
        sig_reference_data : SigBkgBlock
            Block obj, by default None.
        bkg_reference_data : SigBkgBlock
            Block obj, by default None.
        N_sig : int, optional
            Number of signal events, by default None.
        N_bkg : int, optional
            Number of background events, by default None.
        alpha : float, optional
            Signal fraction, by default None.
        bonly : bool, optional
            Only use background in data, by default False.
        scale_by_alpha : bool, optional
            If True scale nu_S / alpha. If True we get for MC mu=alpha, if False we get for MC mu=1, by default True.
        """
        super().__init__()
        # number of generated MC and ML events pre-cut (assume equal)
        self.N_gen = N_gen

        # post-cut arrays
        self.sig_generated_data = sig_generated_data
        self.bkg_generated_data = bkg_generated_data
        self.sig_reference_data = sig_reference_data
        self.bkg_reference_data = bkg_reference_data

        # binning
        self.bin_range, self.bins = bin_range, bins

        # signal fraction
        if N_sig is not None and N_bkg is not None:
            self.alpha = N_sig / N_bkg
        else:
            self.alpha = alpha

        # expected background MC counts
        if N_sig is not None and N_bkg is not None:
            self.nu_b_mc = N_bkg
            self.nu_s_mc = self.alpha * self.nu_b_mc
        else:
            self.nu_b_mc, self.nu_s_mc = None, None

        # if background only
        self.bonly = bonly

        # additionally scale by alpha for mu fit
        self.scale_by_alpha = scale_by_alpha

        # after cut histograms
        self.cut_histograms = dict()

        # post-cut rescaled histograms and their errors
        self.histograms, self.errors = dict(), dict()

    def _get_histograms(self):
        binning_array = np.concatenate([self.sig_reference_data, self.bkg_reference_data])
        bin_edges = np.histogram_bin_edges(binning_array, bins=self.bins, range=self.bin_range)

        sig_ml_hist = np.histogram(self.sig_generated_data, bins=bin_edges)[0]
        bkg_ml_hist = np.histogram(self.bkg_generated_data, bins=bin_edges)[0]
        sig_mc_hist = np.histogram(self.sig_reference_data, bins=bin_edges)[0]
        bkg_mc_hist = np.histogram(self.bkg_reference_data, bins=bin_edges)[0]

        if True in [
            (sig_ml_hist == 0).any(),
            (bkg_ml_hist == 0).any(),
            (sig_mc_hist == 0).any(),
            (bkg_mc_hist == 0).any(),
        ]:
            logging.warning("One or more bin is empty! Change bin range or number of bins...")

        self.cut_histograms = {
            "sig_gen": sig_ml_hist,
            "bkg_gen": bkg_ml_hist,
            "sig_mc": sig_mc_hist,
            "bkg_mc": bkg_mc_hist,
        }

        return self.cut_histograms

    def _make_hists_from_gen_mc_samples(self):
        sig_ml_hist, bkg_ml_hist, sig_mc_hist, bkg_mc_hist = self._get_histograms().values()

        eff_b_mc = bkg_mc_hist / self.N_gen  # eff per bin for bkg MC
        eff_s_mc = sig_mc_hist / self.N_gen  # eff per bin for sig MC

        eff_b_ml = bkg_ml_hist / self.N_gen  # eff per bin for bkg ML
        eff_s_ml = sig_ml_hist / self.N_gen  # eff per bin for sig ML

        B_b_mc = self.nu_b_mc / eff_b_mc.sum()  # Lumi x xsec for bkg MC
        B_s_mc = self.nu_s_mc / eff_s_mc.sum()  # Lumi x xsec for sig MC

        # expected bkg ML counts and expected sig ML counts
        nu_b_ml_hist = B_b_mc * eff_b_ml
        nu_s_ml_hist = B_s_mc * eff_s_ml / self.alpha if self.scale_by_alpha else B_s_mc * eff_s_ml

        # expected bkg MC counts and expected sig MC counts
        nu_b_mc_hist = B_b_mc * eff_b_mc
        nu_s_mc_hist = B_s_mc * eff_s_mc / self.alpha if self.scale_by_alpha else B_s_mc * eff_s_mc

        # Background only Asimov data
        if self.bonly:
            data_mc = nu_b_mc_hist
            data_ml = nu_b_ml_hist
            data_mlmc = nu_b_ml_hist
        # ASimov sig and bkg data
        else:
            data_mc = nu_b_mc_hist + self.alpha * nu_s_mc_hist if self.scale_by_alpha else nu_b_mc_hist + nu_s_mc_hist
            data_ml = nu_b_ml_hist + self.alpha * nu_s_ml_hist if self.scale_by_alpha else nu_b_ml_hist + nu_s_ml_hist
            data_mlmc = nu_b_ml_hist + self.alpha * nu_s_mc_hist if self.scale_by_alpha else nu_b_ml_hist + nu_s_mc_hist

        self.histograms = {
            "sig_gen": nu_s_ml_hist,
            "bkg_gen": nu_b_ml_hist,
            "sig_mc": nu_s_mc_hist,
            "bkg_mc": nu_b_mc_hist,
            "data_gen": data_ml,
            "data_mc": data_mc,
            "data_mlmc": data_mlmc,
            "total_S": sig_mc_hist.sum(),
            "total_B": bkg_mc_hist.sum(),
            "eff": [eff_s_mc.sum(), eff_b_mc.sum(), eff_s_ml.sum(), eff_b_ml.sum()],
        }

        sigma_eff_b_mc = np.sqrt(eff_b_mc * (1 - eff_b_mc) / self.N_gen)  # eff error per bin for bkg MC
        sigma_eff_s_mc = np.sqrt(eff_s_mc * (1 - eff_s_mc) / self.N_gen)  # eff error per bin for sig MC

        sigma_eff_b_ml = np.sqrt(eff_b_ml * (1 - eff_b_ml) / self.N_gen)  # eff error per bin for bkg ML
        sigma_eff_s_ml = np.sqrt(eff_s_ml * (1 - eff_s_ml) / self.N_gen)  # eff error per bin for sig ML

        sigma_nu_b_mc = B_b_mc * sigma_eff_b_mc  # count error per bin for bkg MC
        sigma_nu_s_mc = B_s_mc * sigma_eff_s_mc  # count error per bin for sig MC

        sigma_nu_b_ml = B_b_mc * sigma_eff_b_ml  # count error per bin for bkg ML
        sigma_nu_s_ml = B_s_mc * sigma_eff_s_ml  # count error per bin for sig ML

        sigma_data_mc = np.sqrt(data_mc)  # count error per bin for Asimov MC data
        sigma_data_ml = np.sqrt(data_ml)  # count error per bin for Asimov ML data
        sigma_data_mlmc = np.sqrt(data_mlmc)  # count error per bin for Asimov ML-MC data

        self.errors = {
            "eff_b_mc": sigma_eff_b_mc,
            "eff_s_mc": sigma_eff_s_mc,
            "eff_b_ml": sigma_eff_b_ml,
            "eff_s_ml": sigma_eff_s_ml,
            "nu_b_mc": sigma_nu_b_mc,
            "nu_s_mc": sigma_nu_s_mc,
            "nu_b_ml": sigma_nu_b_ml,
            "nu_s_ml": sigma_nu_s_ml,
            "data_mc": sigma_data_mc,
            "data_ml": sigma_data_ml,
            "data_mlmc": sigma_data_mlmc,
        }

        self.logger.debug(f"Making hists for alpha {self.alpha} and {self.nu_b_mc} nu_B_mc")

        return self.histograms, self.errors

    def run(self):
        return self._make_hists_from_gen_mc_samples()

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["sig_generated_data"] = None
        attributes["sig_reference_data"] = None
        attributes["bkg_generated_data"] = None
        attributes["bkg_reference_data"] = None


class MakeHistsFromSamplesLumi(MakeHistsFromSamples):
    def __init__(
        self,
        bin_range,
        N_gen,
        bins=22,
        sig_generated_data=None,
        bkg_generated_data=None,
        sig_reference_data=None,
        bkg_reference_data=None,
        alpha=0.1,
        lumi=None,
        xsec=None,
        eff=1.0,
        **kwargs,
    ):
        """Same as MakeHistsFromSamples but with a list of luminosity points as a parameter.

        Note
        ----
        This is a scan over different number of bkg and sig events with the same alpha=sig/bkg ratio. Number of signal
        events is defined as N = L x xsec x eff, where L is the luminosity, xsec is the cross section and eff is set to
        1.0 by default.

        Parameters
        ----------
        alpha : float, optional
            Signal fraction in sample, by default 0.1.
        lumi : list, optional
            [start_lumi, end_lumi, step_lumi], by default None.
        eff : float, optional
            Efficiency (not used), by default 1.0.
        """
        super().__init__(
            bin_range,
            N_gen,
            bins,
            sig_generated_data,
            bkg_generated_data,
            sig_reference_data,
            bkg_reference_data,
            alpha=alpha,
            **kwargs,
        )
        self.lumi, self.xsec, self.eff = lumi, xsec, eff
        self.lumi_histograms = []
        self.lumi_errors = []
        self.expected_N = []

    def run(self):
        lumis = np.linspace(self.lumi[0], self.lumi[1], self.lumi[2])
        self.expected_N = (lumis * self.xsec * self.eff).astype(int)

        for i, lumi in enumerate(lumis):
            self.nu_b_mc = self.expected_N[i]
            self.nu_s_mc = self.alpha * self.nu_b_mc

            self.logger.debug(
                f"considering: S_mc={self.nu_s_mc}, B_mc={self.nu_b_mc} and alpha={self.alpha} with {lumi} lumi"
            )

            histograms, errors = self._make_hists_from_gen_mc_samples()
            self.lumi_histograms.append(histograms)
            self.lumi_errors.append(errors)

        return self.lumi_histograms, self.lumi_errors


def get_hists_pipeline(
    var="m bb",
    bin_range=(0, 4),
    bins=40,
    N_sig=1000,
    N_bkg=10000,
    alpha=None,
    logger=None,
    use_classifier=False,  # use NN classifier instead of variable
    N_gen=10**6,
    bonly=False,
    scale_by_alpha=True,
):
    cut_pipeline = get_cut_pipeline(cut_value=0.5, N_gen=N_gen, logger=logger)

    if use_classifier:
        b_sig_gen_var, b_bkg_gen_var, b_sig_mc_var, b_bkg_mc_var = (
            cut_pipeline.pipes[15],
            cut_pipeline.pipes[16],
            cut_pipeline.pipes[17],
            cut_pipeline.pipes[18],
        )

        b_sig_bkg_gen_mc = SigBkgBlock(b_sig_gen_var, b_bkg_gen_var, b_sig_mc_var, b_bkg_mc_var, use_results=True)(
            b_sig_gen_var, b_bkg_gen_var, b_sig_mc_var, b_bkg_mc_var
        )
    else:
        dct = get_colnames_dict()
        idx = dct[var]

        b_sig_gen_data_cut, b_bkg_gen_data_cut, b_sig_mc_data_cut, b_bkg_mc_data_cut = (
            cut_pipeline.pipes[-4],
            cut_pipeline.pipes[-3],
            cut_pipeline.pipes[-2],
            cut_pipeline.pipes[-1],
        )

        b_sig_gen_var = VariableExtractBlock(idx, save_data=False)(b_sig_gen_data_cut)
        b_bkg_gen_var = VariableExtractBlock(idx, save_data=False)(b_bkg_gen_data_cut)
        b_sig_mc_var = VariableExtractBlock(idx, save_data=False)(b_sig_mc_data_cut)
        b_bkg_mc_var = VariableExtractBlock(idx, save_data=False)(b_bkg_mc_data_cut)

        b_sig_bkg_gen_mc = SigBkgBlock(b_sig_gen_var, b_bkg_gen_var, b_sig_mc_var, b_bkg_mc_var, use_results=False)(
            b_sig_gen_var, b_bkg_gen_var, b_sig_mc_var, b_bkg_mc_var
        )

    b_hists = MakeHistsFromSamples(
        bin_range=bin_range,
        bins=bins,
        N_sig=N_sig,
        N_bkg=N_bkg,
        alpha=alpha,
        N_gen=N_gen,
        bonly=bonly,
        scale_by_alpha=scale_by_alpha,
    )(b_sig_bkg_gen_mc)

    pipe = Pipeline(logger=logger)
    pipe.compose(
        cut_pipeline,
        b_sig_gen_var,
        b_bkg_gen_var,
        b_sig_mc_var,
        b_bkg_mc_var,
        b_sig_bkg_gen_mc,
        b_hists,
    )

    return pipe


if __name__ == "__main__":
    pipe = get_hists_pipeline(bin_range=(0.5, 1.2), use_classifier=True)
    pipe.fit()
