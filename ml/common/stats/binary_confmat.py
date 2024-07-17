import numpy as np
from numba import njit


@njit
def _calc_binary_confusion_matrix(y_actual, y_hat):
    """jit-compiled function to compute binary confusion matrix.

    References
    ----------
    [1] - https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal

    """
    tp, fp, tn, fn = 0, 0, 0, 0

    tp_idx, fp_idx, tn_idx, fn_idx = (
        np.zeros(len(y_actual)),
        np.zeros(len(y_actual)),
        np.zeros(len(y_actual)),
        np.zeros(len(y_actual)),
    )

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            tp += 1
            tp_idx[i] = 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            fp += 1
            fp_idx[i] = 1
        if y_actual[i] == y_hat[i] == 0:
            tn += 1
            tn_idx[i] = 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            fn += 1
            fn_idx[i] = 1

    return tp, fp, fn, tn, tp_idx, fp_idx, fn_idx, tn_idx


def binary_confusion_matrix(y_actual, y_hat, threshold=0.5):
    """Compute binary confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix.

    Parameters
    ----------
    y_actual : np.ndarray
        1D array of actual binary labels (0 or 1).
    y_hat : np.ndarray
        1D array of predicted labels (e.g. output of a classifier).
    threshold : float, optional
        Threshold for binary classification, by default 0.5.

    Returns
    -------
    np.ndarray, dict
        2D array of shape (2, 2) representing the confusion matrix, and a dictionary with the indices of
        the true positive, false positive, true negative, and false negative samples.
    """
    y_hat_thres = (y_hat >= threshold).astype(int)

    res = _calc_binary_confusion_matrix(y_actual, y_hat_thres)

    confmat = np.zeros((2, 2))
    confmat[0, 0] = res[0]
    confmat[0, 1] = res[1]
    confmat[1, 0] = res[2]
    confmat[1, 1] = res[3]

    idx = {"tp": res[4], "fp": res[5], "tn": res[6], "fn": res[7]}
    idx = {k: idx[k].astype(bool) for k in idx.keys()}

    return confmat, idx
