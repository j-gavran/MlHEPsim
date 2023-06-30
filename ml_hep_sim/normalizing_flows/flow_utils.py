import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions

from ml_hep_sim.data_utils.toy_datasets import TOY_DATASETS
from ml_hep_sim.stats.one_dim_tests.py_two_sample import chi2_twosample_test
from ml_hep_sim.stats.stat_plots import two_sample_plot


class LogisitcDistribution:
    """https://pytorch.org/docs/stable/distributions.html#transformeddistribution"""

    def __init__(self, input_dim, to_gpu=True):
        if to_gpu:
            a, b = torch.zeros(input_dim).cuda(), torch.ones(input_dim).cuda()
        else:
            a, b = torch.zeros(input_dim), torch.ones(input_dim)

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


def get_2d_density(model, mesh):
    if model.training:
        raise ValueError

    z, jac = model.flow(torch.from_numpy(mesh).to(model.device))
    sum_jac = sum(jac)
    base = model.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
    log_density = (base + sum_jac).cpu().numpy()

    return np.exp(log_density.flatten())


def make_2d_mesh(xmin, xmax, mesh_points):
    points = 2 * [np.linspace(xmin, xmax, mesh_points, dtype=np.float32)]
    mesh = np.array(np.meshgrid(*points)).T.reshape(-1, 2)
    return mesh


def _plot_2d_density(xmin, xmax, prob, mesh_points, axs=None, fig=None, colorbar=True):
    if axs is None and fig is None:
        fig, axs = plt.subplots(1, 1)

    im = axs.imshow(prob.reshape(mesh_points, mesh_points).T, origin="lower")

    axs.set_xticks([0, int(mesh_points * 0.25), int(mesh_points * 0.5), int(mesh_points * 0.75), mesh_points])
    axs.set_xticklabels([xmin, int(xmin / 2), 0, int(xmax / 2), xmax])
    axs.set_yticks([0, int(mesh_points * 0.25), int(mesh_points * 0.5), int(mesh_points * 0.75), mesh_points])
    axs.set_yticklabels([xmin, int(xmin / 2), 0, int(xmax / 2), xmax])

    if colorbar:
        fig.colorbar(im, orientation="vertical")

    return axs


def plot_2d_density(model, xmin=-4, xmax=4, mesh_points=200, axs=None, fig=None, colorbar=True):
    mesh = make_2d_mesh(xmin, xmax, mesh_points)
    prob = get_2d_density(model, mesh)

    _plot_2d_density(xmin, xmax, prob, mesh_points, axs, fig, colorbar)

    return prob, mesh


def remove_outliers(arr, k):
    """https://stackoverflow.com/questions/25447453/removing-outliers-in-each-column-and-corresponding-row"""
    mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)
    return arr[np.all(np.abs((arr - mu) / sigma) < k, axis=1)]


def plot_flow_result_on_event(data_name, base_distribution, model, logger, dataset, idx=0, use_hexbin=False):
    """https://forums.pytorchlightning.ai/t/understanding-logging-and-validation-step-validation-epoch-end/291

    TODO: rewrite this into class.

    """

    if 0 < idx < 100:
        idx = f"0{idx}"

    def _flow_on_train_end():
        if data_name.lower() == "mnist":
            n = 10
            fig, axs = plt.subplots(n, n, figsize=(15, 15))
            axs = axs.flatten()

            x = model.flow.sample(n * n).cpu().numpy()

            for i in range(n * n):
                axs[i].imshow(x[i, :].reshape(28, 28).clip(0, 1), cmap="gray")

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"MNIST_generated_{idx}.jpg")

        elif data_name.lower() in ["higgs", "higgs_bkg", "higgs_sig"]:
            subset = 10**5
            X = copy.deepcopy(dataset.X[:subset].cpu().numpy())
            sample = model.flow.sample(subset).cpu().numpy()

            scalers = model.scalers
            X = scalers[2].inverse_transform(X)

            sample = sample[~np.isnan(sample).any(axis=1)]
            sample = sample[~np.isinf(sample).any(axis=1)]

            sample = scalers[0].inverse_transform(sample)

            sample = sample[~np.isnan(sample).any(axis=1)]
            sample = sample[~np.isinf(sample).any(axis=1)]

            sample = remove_outliers(sample, 5)

            try:
                chi2 = chi2_twosample_test(X, sample)
                labels = [f"chi2={i:.3f}" for i in chi2["chi2"].to_numpy()]
            except Exception as e:
                logging.warning(f"Exception {e} has occurred in chi2 test while plotting...")
                labels = np.zeros(X.shape[1])

            fig, axs = plt.subplots(6, 3, figsize=(10, 12))
            axs = axs.flatten()

            two_sample_plot(X, sample, axs, n_bins=50, log_scale=False, density=True, lw=2, labels=labels)

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"Higgs_generated_{idx}.jpg")

            fig, axs = plt.subplots(6, 3, figsize=(10, 12))
            axs = axs.flatten()

            two_sample_plot(X, sample, axs, n_bins=50, log_scale=True, density=False, lw=2, labels=labels)

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"Higgs_generated_log_{idx}.jpg")

        elif data_name in TOY_DATASETS:
            subset = 3 * 10**4
            generated = model.flow.sample(subset).cpu().numpy()

            X = dataset.X[:subset]
            base_dist, _ = model.flow.forward(X)
            base_dist = base_dist.cpu().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            if use_hexbin:
                axs[0].hexbin(generated[:, 0], generated[:, 1], cmap="jet", extent=[-4, 4, -4, 4], gridsize=150)
                axs[1].hexbin(base_dist[:, 0], base_dist[:, 1], cmap="jet", extent=[-4, 4, -4, 4], gridsize=150)
            else:
                axs[0].scatter(generated[:, 0], generated[:, 1], s=0.25)
                axs[1].scatter(base_dist[:, 0], base_dist[:, 1], s=0.25)

            axs[0].set_title("generating direction")
            axs[1].set_title("normalizing direction")

            logger.experiment.log_figure(logger.run_id, fig, f"2d_test_{idx}.jpg")

            fig, axs = plt.subplots()
            plot_2d_density(model, axs=axs, fig=fig, mesh_points=200)
            axs.set_title("density estimation")
            logger.experiment.log_figure(logger.run_id, fig, f"2d_test_density_{idx}.jpg")

        else:
            logging.warning("data_name not implemented...")

    try:
        with torch.no_grad():
            model.eval()
            _flow_on_train_end()
            model.train()

            plt.tight_layout()
            plt.close()
    except Exception as e:
        logging.warning(f"Mlflow plotting quit with exception {e}")

    torch.cuda.empty_cache()
