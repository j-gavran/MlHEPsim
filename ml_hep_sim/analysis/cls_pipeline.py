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


class CLsBlock(Block):
    def __init__(
        self,
        bkg_err,
        mc_test=False,
        lumi=None,
        lumi_histograms=None,
        alpha=None,
        par_bounds=None,
        use_toys=False,
        ntoys=500,
        test_stat="q0",
        poi_test=0.0,
    ):
        super().__init__()
        self.bkg_err = bkg_err
        self.lumi = lumi
        self.lumi_histograms = lumi_histograms
        self.alpha = alpha
        self.mc_test = mc_test
        self.par_bounds = par_bounds
        self.use_toys = use_toys
        self.ntoys = ntoys
        self.test_stat = test_stat
        self.poi_test = poi_test

        self.results = []

    def run(self):
        lumis = np.linspace(*self.lumi)

        # only supports 1 lumi point at the moment!
        for _, hist in zip(lumis, self.lumi_histograms):

            if self.mc_test:
                sig, bkg, data = hist["sig_mc"], hist["bkg_mc"], hist["data_mc"]
            else:
                sig, bkg, data = hist["sig_mc"], hist["bkg_gen"], hist["data_mc"]

            spec = prep_data(sig, bkg, self.bkg_err)
            model = pyhf.Model(spec)

            data = list(data) + model.config.auxdata

            # the hypothesis test computes test statistics
            if self.use_toys:
                calc = pyhf.infer.calculators.ToyCalculator(
                    data, model, test_stat=self.test_stat, ntoys=self.ntoys, par_bounds=self.par_bounds
                )
            else:
                calc = pyhf.infer.calculators.AsymptoticCalculator(
                    data, model, test_stat=self.test_stat, par_bounds=self.par_bounds
                )

            teststat = calc.teststatistic(poi_test=self.poi_test)

            # Probability distributions of the test statistic
            sb_dist, b_dist = calc.distributions(poi_test=self.poi_test)

            # Calculate the p-values for the observed test statistic under the signal + background and background-only model hypotheses.
            p_sb, p_b, p_s = calc.pvalues(teststat, sb_dist, b_dist)

            # print(f'CL_sb = {p_sb}')
            # print(f'CL_b = {p_b}')
            # print(f'CL_s = CL_sb / CL_b = {p_s}')

            # Calculate the CLs values corresponding to the
            # median significance of variations of the signal strength from the
            # background only hypothesis :math:`\left(\mu=0\right) at :math:`(-2,-1,0,1,2)\sigma`.
            p_exp_sb, p_exp_b, p_exp_s = calc.expected_pvalues(sb_dist, b_dist)

            # print(f'exp. CL_sb = {p_exp_sb}')
            # print(f'exp. CL_b = {p_exp_b}')
            # print(f'exp. CL_s = CL_sb / CL_b = {p_exp_s}')

            self.results.append(
                {
                    "sig_frac": self.alpha,
                    "bkg_err": self.bkg_err,
                    "p_sb": p_sb,
                    "p_b": p_b,
                    "p_s": p_s,
                    "p_exp_sb": p_exp_sb[2],
                    "p_exp_b": p_exp_b[2],
                    "p_exp_s": p_exp_s[2],
                    "teststat": teststat,
                }
            )

            # self.results.append(
            #     {"CLs": [p_sb, p_b, p_s], "CLs_exp": [p_exp_sb[2], p_exp_b[2], p_exp_s[2]], "teststat": teststat}
            # )

        return self


class CLsBlockResultsParser(Block):
    def __init__(self, results=None):
        super().__init__()
        self.results = results
        self.parsed_results = None

    def run(self):
        split = len(self.results) // 2

        # have 1 lumi point only!
        mc_res = [i[0] for i in self.results[:split]]
        ml_res = [i[0] for i in self.results[split:]]

        self.parsed_results = {"mc_res": pd.DataFrame(mc_res), "ml_res": pd.DataFrame(ml_res)}

        return self.parsed_results


def get_cls_pipeline(
    bins=22,
    lumi=100,
    xsec=10,
    pts=10,
    bin_range=(0, 4),
    use_classifier=False,
    N_gen=10 ** 6,
    logger=None,
    scale_by_alpha=True,
):
    hists_pipeline = get_hists_pipeline(
        use_classifier=use_classifier,
        N_gen=N_gen,
        logger=logger,
        scale_by_alpha=scale_by_alpha,
    )
    hists_pipeline.pipes = hists_pipeline.pipes[:-1]  # remove MakeHistsFromSamples (will replace with lumi block)

    sig_fracs = np.linspace(0.01, 0.1, pts)  # do different signal fraction (injections)
    bkg_errs = np.linspace(0.01, 0.1, pts)  # do different sys errors

    # set mu and gamma (mc correction) bounds
    poi_bound = (0, 1000.0)
    gamma_bound = (1e-10, 1000.0)

    bounds = []
    for b in range(bins + 1):
        if b == 0:
            bounds.append(poi_bound)
        else:
            bounds.append(gamma_bound)

    b_sig_bkg = hists_pipeline.pipes[-1]  # aggregate block
    hists_cls_blocks = []

    lumi = [lumi, lumi, 1]  # scan for 1 lumi point only

    # MC
    for sf in sig_fracs:
        for err in bkg_errs:
            hists_block = MakeHistsFromSamplesLumi(
                bin_range=bin_range,
                N_gen=N_gen,
                bins=bins,
                alpha=sf,
                lumi=lumi,
                xsec=xsec,
                scale_by_alpha=scale_by_alpha,
            )(b_sig_bkg)

            cls_block = CLsBlock(bkg_err=err, par_bounds=bounds, mc_test=True)(hists_block)

            hists_cls_blocks.append(hists_block)
            hists_cls_blocks.append(cls_block)

    # ML
    for sf in sig_fracs:
        for err in bkg_errs:
            hists_block = MakeHistsFromSamplesLumi(
                bin_range=bin_range,
                N_gen=N_gen,
                bins=bins,
                alpha=sf,
                lumi=lumi,
                xsec=xsec,
                scale_by_alpha=scale_by_alpha,
            )(b_sig_bkg)

            cls_block = CLsBlock(bkg_err=err, par_bounds=bounds, mc_test=False)(hists_block)

            hists_cls_blocks.append(hists_block)
            hists_cls_blocks.append(cls_block)

    b_parse = CLsBlockResultsParser()(*hists_cls_blocks)

    pipe = Pipeline(logger=logger)
    pipe.compose(hists_pipeline, hists_cls_blocks, b_parse)

    return pipe


if __name__ == "__main__":
    p = get_cls_pipeline()
    p.fit()
