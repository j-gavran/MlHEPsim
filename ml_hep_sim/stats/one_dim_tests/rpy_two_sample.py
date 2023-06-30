from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from ml_hep_sim.stats.one_dim_tests.py_two_sample import make_histograms
from rpy2.robjects import numpy2ri


class RStatTests:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self._assert()
        self._r_setup()

    @staticmethod
    def _r_setup():
        # activate automatic numpy <-> R conversion
        numpy2ri.activate()

        # path setup and R script import
        scripts_dir = str(Path(__file__).parent.resolve())
        r_source = robjects.r["source"]
        r_source(scripts_dir + "/two_sample_tests.R")

    def _assert(self):
        sx, sy = self.x.shape, self.y.shape

        # checks for: same dimension, dim <= 2, same number of features if dim == 2
        assert len(sx) == len(sy)
        if len(sx) == 2:
            assert sx[1] == sy[1]
        elif len(sx) > 2:
            raise NotImplemented

    def ks_test(self):
        r_ks_test = robjects.globalenv["ks_test"]

        result = {"ks": [], "p": []}

        if len(self.x.shape) == 1:
            r = r_ks_test(self.x, self.y)
            result["ks"], result["p"] = r[0][0], r[1][0]
        else:
            for i in range(self.x.shape[1]):
                r = r_ks_test(self.x[:, i], self.y[:, i])
                result["ks"].append(r[0][0])
                result["p"].append(r[1][0])

        return pd.DataFrame(result, columns=["ks", "p"])

    def ad_test(self):
        r_ad_test = robjects.globalenv["ad_test"]

        result = {"ad": [], "p": []}

        # version 1, T.AD and asympt. P-value
        if len(self.x.shape) == 1:
            r = r_ad_test(self.x, self.y)
            result["ad"], result["p"] = r[6][0, 1], r[6][0, 2]
        else:
            for i in range(self.x.shape[1]):
                r = r_ad_test(self.x[:, i], self.y[:, i])
                result["ad"].append(r[6][0, 1])
                result["p"].append(r[6][0, 2])

        return pd.DataFrame(result, columns=["ad", "p"])

    def chi2_test(self, n_bins="auto", return_hists=False):
        r_chi2_test = robjects.globalenv["chi2_test"]
        x_hist, y_hist = make_histograms(self.x, self.y, n_bins=n_bins)

        result = {"chi2": [], "p": []}

        if len(self.x.shape) == 1:
            r = r_chi2_test(x_hist, y_hist)
            result["chi2"], result["p"] = r[0][0], r[2][0]
        else:
            for i in range(self.x.shape[1]):
                r = r_chi2_test(x_hist[i], y_hist[i])
                result["chi2"].append(r[0][0])
                result["p"].append(r[2][0])

        if return_hists:
            return pd.DataFrame(result, columns=["chi2", "p"]), (x_hist, y_hist)
        else:
            return pd.DataFrame(result, columns=["chi2", "p"])


if __name__ == "__main__":
    x = np.random.uniform(size=(10000, 2))
    eps = np.random.normal(loc=0, scale=1, size=(10000, 2)) * 0.009
    y = np.random.uniform(size=(10000, 2)) + eps

    tests = RStatTests(x, y)

    print(tests.ks_test())
    print(tests.ad_test())

    from ml_hep_sim.stats.one_dim_tests.py_two_sample import (
        chi2_twosample_test,
        ks_twosample_test,
    )

    print(ks_twosample_test(x, y))

    res, histx, histy = chi2_twosample_test(x, y, n_bins=30, return_histograms=True)
    print(res)

    a = np.array(histx[0])
    b = np.array(histy[0])

    print(tests.chi2_test(n_bins=30))

    import matplotlib.pyplot as plt

    plt.hist(x[:, 0], bins=30, histtype="step", density=False)
    plt.hist(y[:, 0], bins=30, histtype="step", density=False)
    plt.hist(eps[:, 0], bins=30, histtype="step", density=False)
    plt.show()
