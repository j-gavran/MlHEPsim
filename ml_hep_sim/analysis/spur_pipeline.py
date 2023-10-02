import numpy as np
import pandas as pd
import pyhf

from ml_hep_sim.analysis.hists_pipeline import MakeHistsFromSamples, get_hists_pipeline
from ml_hep_sim.pipeline.blocks import GCBlock
from ml_hep_sim.pipeline.pipeline_loggers import setup_logger
from ml_hep_sim.pipeline.pipes import Block, Pipeline
from ml_hep_sim.stats.pyhf_json_specs import prep_data


class SpurBlock(Block):
    def __init__(
        self,
        histograms=None,
        errors=None,
        mc_test=False,
        bkg_err=None,
        alpha=None,
        par_bounds=None,
        use_minuit=True,
        bonly=None,
    ):
        super().__init__()
        self.histograms = histograms
        self.errors = errors
        self.mc_test = mc_test
        self.bkg_err = bkg_err
        self.alpha = alpha
        self.par_bounds = par_bounds
        self.use_minuit = use_minuit
        self.bonly = bonly

        self.results = []

    def mle_fit(self):
        pyhf.set_backend("numpy", "minuit")

        if self.mc_test:
            sig, bkg, data = self.histograms["sig_mc"], self.histograms["bkg_mc"], self.histograms["data_mc"]
        else:
            sig, bkg, data = self.histograms["sig_gen"], self.histograms["bkg_gen"], self.histograms["data_mc"]

        eps = 1e-12
        spec = prep_data(sig + eps, bkg + eps, self.bkg_err, mc_err=self.errors["nu_b_ml"])

        model = pyhf.Model(spec)
        observations = list(data) + model.config.auxdata
        result, twice_nll = pyhf.infer.mle.fit(
            observations,
            model,
            return_uncertainties=True,
            return_fitted_val=True,
            par_bounds=self.par_bounds,
        )

        bestfit, errors = result.T
        self.bestfit = [bestfit, twice_nll]

        return model, bestfit, errors

    def run(self):
        """
        References:
        -----------
        [1] - https://scikit-hep.org/pyhf/_generated/pyhf.infer.mle.fit.html
        [2] - https://scikit-hep.org/pyhf/_generated/pyhf.optimize.opt_minuit.minuit_optimizer.html#pyhf.optimize.opt_minuit.minuit_optimizer

        """
        _, bestfit, errors = self.mle_fit()

        self.results.append(
            {
                "alpha": self.alpha,
                "mu": bestfit[0],
                "gamma": bestfit[1:],
                "mu_err": errors[0],
                "gamma_err": errors[1:],
                "twice_nll": bestfit[1],
            }
        )

        return self


class SpurBlockResultsParser(Block):
    def __init__(self, results=None):
        super().__init__()
        self.results = results
        self.parsed_results = None

    def run(self):
        df_lst = []
        for res in self.results:
            df = pd.DataFrame(res)
            df_lst.append(df)

        self.parsed_results = pd.concat(df_lst, ignore_index=True)
        return self


def get_spur_pipeline(
    nu_bs,
    alphas,
    bins=22,
    bin_range=(0, 4),
    bkg_err=0.1,
    use_classifier=False,
    bonly=False,
    N_gen=10**6,
    mc_test=False,
    scale_by_alpha=True,
    **kwargs,
):
    logger = setup_logger("spur_pipeline")
    logger = None

    hists_pipeline = get_hists_pipeline(
        use_classifier=use_classifier,
        N_gen=N_gen,
        logger=logger,
        scale_by_alpha=scale_by_alpha,
    )
    hists_pipeline.pipes = hists_pipeline.pipes[:-1]
    b_sig_bkg = hists_pipeline.pipes[-1]

    spur_pipelines, spure_parse_blocks = [], []
    for nu_b in nu_bs:
        for alpha in alphas:
            b_hists = MakeHistsFromSamples(
                bin_range=bin_range,
                bins=bins,
                N_sig=nu_b * alpha,
                N_bkg=nu_b,
                N_gen=N_gen,
                bonly=bonly,
                scale_by_alpha=scale_by_alpha,
            )(b_sig_bkg)

            b_spur = SpurBlock(bkg_err=bkg_err, mc_test=mc_test, **kwargs)(b_hists)
            spure_parse_blocks.append(b_spur)

            b_gc1 = GCBlock(exclude=["alpha", "histograms", "logger", "bonly", "errors"])(b_hists)
            b_gc2 = GCBlock(exclude=["results", "logger"])(b_spur)

            spur_pipelines += [*hists_pipeline.pipes, b_hists, b_gc1, b_spur, b_gc2]

    b_parse = SpurBlockResultsParser()(*spure_parse_blocks)
    spur_pipelines.append(b_parse)

    pipe = Pipeline(logger=logger)
    pipe.compose(spur_pipelines)

    return pipe


if __name__ == "__main__":
    nu_bs = np.linspace(10**3, 10**5, 2)
    alphas = np.linspace(0.01, 0.1, 2)

    pipe = get_spur_pipeline(nu_bs, alphas, N_gen=2 * 10**4)
    pipe.fit()

    res = pipe.pipes[-1]
    print(res.parsed_results)
