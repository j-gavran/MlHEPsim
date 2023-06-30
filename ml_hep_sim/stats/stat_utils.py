import numpy as np
import pandas as pd
import torch


def ecdf(S):
    if torch.is_tensor(S):
        S = S.numpy()

    if len(S.shape) == 1:
        S = [S]
    else:
        S = S.T

    xs, cumsums = [], []
    for s in S:
        x, counts = np.unique(s, return_counts=True)
        cumsum = np.cumsum(counts)
        cumsum = cumsum / cumsum[-1]
        xs.append(x)
        cumsums.append(cumsum)

    if type(S) == list:
        return xs[0], cumsums[0]
    else:
        return xs, cumsums


def parse_test_results(results):
    parsed = np.zeros((len(results), 3))
    statistic = list(set(results[0].keys()) ^ {"crit", "p"})[0]
    for i, res_dct in enumerate(results):
        parsed[i, 0] = res_dct[statistic]
        parsed[i, 1] = res_dct["crit"]
        parsed[i, 2] = res_dct["p"]

    df = pd.DataFrame(parsed, columns=[statistic, "crit", "p"])
    return df
