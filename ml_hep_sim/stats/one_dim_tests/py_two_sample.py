import numpy as np
import torch
from ml_hep_sim.stats.stat_utils import parse_test_results
from scipy.stats import chi2, ks_2samp, kstwo


def chisquare_2samp(h_A, h_B):
    c_1, c_2 = np.sqrt(np.sum(h_B) / np.sum(h_A)), np.sqrt(np.sum(h_A) / np.sum(h_B))
    chi2 = np.sum((c_1 * h_A - c_2 * h_B) ** 2 / (h_A + h_B))
    return chi2


def make_histograms(A, B, n_bins, bin_range=None):
    n_features = A.shape[1]

    combined_sample = np.concatenate([A, B])

    histograms_A, histograms_B = [], []
    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(combined_sample[:, feature], bins=n_bins, range=bin_range)

        h_A, _ = np.histogram(A[:, feature], bins=bin_edges)
        h_B, _ = np.histogram(B[:, feature], bins=bin_edges)

        idx_A, idx_B = np.where(h_A == 0)[0], np.where(h_B == 0)[0]
        zero_bins = list(set(idx_A) & set(idx_B))
        h_A, h_B = np.delete(h_A, zero_bins), np.delete(h_B, zero_bins)

        histograms_A.append(h_A)
        histograms_B.append(h_B)

    return histograms_A, histograms_B


def chi2_twosample_test(A, B, n_bins="auto", alpha=0.05, return_histograms=False, bin_range=None):
    assert A.shape[1] == B.shape[1]

    if torch.is_tensor(A) and torch.is_tensor(B):
        A, B = A.cpu().numpy(), B.cpu().numpy()

    histograms_A, histograms_B = make_histograms(A, B, n_bins, bin_range=bin_range)

    test_results = []
    for h_A, h_B in zip(histograms_A, histograms_B):
        test_result = chisquare_2samp(h_A, h_B)
        critical_value = chi2.ppf(1 - alpha, len(h_A) - 1)
        p_value = 1 - chi2.cdf(test_result, len(h_A) - 1)
        test_results.append({"chi2": test_result, "crit": critical_value, "p": p_value})

    if return_histograms:
        return parse_test_results(test_results), (histograms_A, histograms_B)
    else:
        return parse_test_results(test_results)


def ks_twosample_test(A, B, alpha=0.05):
    assert A.shape[1] == B.shape[1]

    if torch.is_tensor(A) and torch.is_tensor(B):
        A, B = A.cpu().numpy(), B.cpu().numpy()

    N, M = len(A), len(B)

    test_results = []
    for a, b in zip(A.T, B.T):
        test_result = ks_2samp(a, b)
        critical_value = kstwo.ppf(1 - alpha, N * M // (N + M))
        test_results.append({"ks": test_result[0], "crit": critical_value, "p": test_result[1]})

    return parse_test_results(test_results)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from ml_hep_sim.stats.stat_plots import samples_ecdf_plot, samples_histogram_plot

    print(kstwo.ppf(1 - 0.001, (25 * 25) // (25 + 25)))
    print(350 / (25 * 25))

#     torch.manual_seed(0)
#     test_tensor1 = torch.normal(0, 1, (500, 3))
#     test_tensor2 = torch.normal(0, 1, (600, 3))
#
#     result = chi2_twosample_test(test_tensor1, test_tensor2)
#     print(f"chi2 results\n: {result}")
#
#     result = ks_twosample_test(test_tensor1, test_tensor2)
#     print(f"\nks results\n: {result}")
#
#     fig, axs = plt.subplots(1, 3, figsize=(10, 3))
#     samples_histogram_plot(test_tensor1, test_tensor2, axs)
#
#     fig, axs = plt.subplots(1, 3, figsize=(10, 3))
#     samples_ecdf_plot(test_tensor1, test_tensor2, axs)
