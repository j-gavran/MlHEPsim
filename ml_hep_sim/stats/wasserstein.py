from scipy.stats import wasserstein_distance
import torch


def _tensor_check(t):
    if torch.is_tensor(t):
        t = t.cpu().numpy()
    return t


def get_wasserstein_distance(a, b):
    a, b = _tensor_check(a), _tensor_check(b)
    return wasserstein_distance(a, b)


if __name__ == "__main__":
    a = torch.randn(10000000)
    b = torch.randn(10000000)

    print(get_wasserstein_distance(a, b))
