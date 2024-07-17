import numpy as np
import torch
import torch.distributions as distributions


class LogisitcDistribution:
    """https://pytorch.org/docs/stable/distributions.html#transformeddistribution"""

    def __init__(self, input_dim, device):
        if device == "cuda":
            a, b = torch.zeros(input_dim).cuda(), torch.ones(input_dim).cuda()
        elif device == "cpu":
            a, b = torch.zeros(input_dim), torch.ones(input_dim)
        else:
            raise ValueError("Device must be either 'cuda' or 'cpu'!")

        base_distribution = distributions.Uniform(a, b)
        transforms = [distributions.SigmoidTransform().inv, distributions.AffineTransform(loc=a, scale=b)]
        self.logistic = distributions.TransformedDistribution(base_distribution, transforms)

    def log_prob(self, value):
        return self.logistic.log_prob(value)

    def sample(self, *args):
        return self.logistic.sample(*args)

    def __getstate__(self):
        # hack for pickling weakref objects
        state = self.__dict__.copy()
        state["logistic"] = None
        return state


def remove_outliers(arr, k):
    """https://stackoverflow.com/questions/25447453/removing-outliers-in-each-column-and-corresponding-row"""
    mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)
    return arr[np.all(np.abs((arr - mu) / sigma) < k, axis=1)]
