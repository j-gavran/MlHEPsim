import logging

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from ml.common.nn.modules import Tracker
from ml.common.stats.two_sample_tests import two_sample_plot
from ml.common.utils.plot_utils import (
    filter_array,
    handle_plot_exception,
    iqr_remove_outliers,
    make_subplots_grid,
)


class SigBkgVAETracker(Tracker):
    def __init__(self, experiment_conf, tracker_path, proc=None, n_samples=10**5, n_bins=50, bin_range=None):
        super().__init__(experiment_conf, tracker_path)
        self.proc = proc
        self.n_samples = n_samples

        self.n_bins = n_bins
        self.bin_range = bin_range

        self.reference = None
        self.latent_samples = None
        self.generated = None

        self.sig_reference, self.bkg_reference = None, None
        self.sig_latent_samples, self.bkg_latent_samples = None, None
        self.elbo_dct = None

        plt.rcParams["font.size"] = 15

    def make_plotting_dirs(self):
        return {
            "latent_samples": f"{self.base_dir}/latent_samples/",
            "generated": f"{self.base_dir}/generated/",
            "elbo": f"{self.base_dir}/elbo/",
        }

    def get_val_predictions(self):
        dl = self.module._trainer.datamodule.val_dataloader()

        self.reference, self.latent_samples = [], []
        for b in tqdm(dl, desc="Looping over val dataloader", leave=False):
            x, _ = b
            self.reference.append(x)
            z = self.module.model.encoder.sample(x.to(self.module.device))
            self.latent_samples.append(z.cpu().numpy())

        self.reference = np.concatenate(self.reference)
        self.latent_samples = np.concatenate(self.latent_samples)
        self.generated = self.module.model.sample(self.n_samples).cpu().numpy()

        return True

    def get_test_predictions(self):
        dl = self.module._trainer.datamodule.test_dataloader()

        self.module.model.eval()

        if type(dl) is not list:
            logging.warning("Dataloader needs to be a list of two Dataloader objects, returning None!")
            self.sig_latent_samples, self.bkg_latent_samples = None, None
            self.elbo_dct = None
            return None

        test_dl_sig, test_dl_bkg = dl

        self.sig_reference, self.bkg_reference = [], []
        self.sig_latent_samples, self.bkg_latent_samples = [], []

        self.elbo_dct = {
            "sig_ELBO": [],
            "bkg_ELBO": [],
            "sig_RE": [],
            "bkg_RE": [],
            "sig_KL": [],
            "bkg_KL": [],
        }

        for b in tqdm(test_dl_sig, desc="Looping over sig test dataloader", leave=False):
            x, _ = b
            self.sig_reference.append(x)

            z = self.module.model.encoder.sample(x.to(self.module.device))
            self.sig_latent_samples.append(z.cpu().numpy())

            elbo, RE, KL = self.module.model.forward(x.to(self.module.device), reduction=None)
            self.elbo_dct["sig_ELBO"].append(elbo.cpu().numpy())
            self.elbo_dct["sig_RE"].append(RE.cpu().numpy())
            self.elbo_dct["sig_KL"].append(KL.cpu().numpy())

        for b in tqdm(test_dl_bkg, desc="Looping over bkg test dataloader", leave=False):
            x, _ = b
            self.bkg_reference.append(x)

            z = self.module.model.encoder.sample(x.to(self.module.device))
            self.bkg_latent_samples.append(z.cpu().numpy())

            elbo, RE, KL = self.module.model.forward(x.to(self.module.device), reduction=None)
            self.elbo_dct["bkg_ELBO"].append(elbo.cpu().numpy())
            self.elbo_dct["bkg_RE"].append(RE.cpu().numpy())
            self.elbo_dct["bkg_KL"].append(KL.cpu().numpy())

        self.sig_reference = np.concatenate(self.sig_reference)
        self.bkg_reference = np.concatenate(self.bkg_reference)

        self.sig_latent_samples = np.concatenate(self.sig_latent_samples)
        self.bkg_latent_samples = np.concatenate(self.bkg_latent_samples)

        for k, v in self.elbo_dct.items():
            self.elbo_dct[k] = filter_array(np.concatenate(v))

        return True

    def get_predictions(self, stage):
        if stage == "val" or stage is None:
            with torch.no_grad():
                self.get_val_predictions()
        elif stage == "test":
            with torch.no_grad():
                self.get_test_predictions()
        else:
            raise ValueError(f"Stage must be one of ['val', 'test', None], got {stage} instead!")

        torch.cuda.empty_cache()

    def compute(self, stage=None):
        return super().compute(stage)

    def plot(self):
        if self.current_epoch % self.experiment_conf["check_metrics_n_epoch"] != 0 and self.stage != "test":
            return False

        if self.stage == "val" or self.stage is None:
            self.corner_plot([self.latent_samples], title="Latent samples")
            self.gen_vs_ref_plot()

        if self.stage == "test":
            if self.sig_latent_samples is not None and self.bkg_latent_samples is not None:
                logging.info("Plotting for test data.")

                self.plot_elbo()

                self.pca_plot(
                    [self.sig_reference, self.bkg_reference],
                    labels=["sig", "bkg"],
                    title="True latent PCA samples",
                    n_components=1,
                    save_str="_true",
                )

                self.pca_plot(
                    [self.sig_latent_samples, self.bkg_latent_samples],
                    labels=["sig", "bkg"],
                    title="Latent PCA samples",
                    n_components=1,
                    save_str="_latent",
                )

                self.corner_plot(
                    [self.sig_latent_samples, self.bkg_latent_samples],
                    labels=["sig", "bkg"],
                    title="Latent samples",
                )

        self.module.logger.experiment.log_artifact(local_path=self.base_dir, run_id=self.module.logger.run_id)

        return True

    @staticmethod
    def _get_roc_auc(data: list):
        min_length = min(len(data[0]), len(data[1]))

        loss_values = np.concatenate([data[0][:min_length], data[1][:min_length]])
        labels = np.concatenate([np.ones(min_length), np.zeros(min_length)])
        fpr, tpr, _ = roc_curve(labels, loss_values)
        roc_auc = auc(fpr, tpr)

        if roc_auc < 0.5:
            roc_auc = 1 - roc_auc

        return roc_auc

    def _get_bin_edges(self, data):
        cat = np.concatenate([iqr_remove_outliers(data[0]), iqr_remove_outliers(data[1])])
        bin_edges = np.histogram_bin_edges(cat, bins=self.n_bins)
        return bin_edges

    def _plot_roc_hist(self, ax, data, xlabel="", legend=None):
        min_length = min(len(data[0]), len(data[1]))

        data_1, data_2 = data[0][:min_length], data[1][:min_length]

        bin_edges = self._get_bin_edges([data_1, data_2])

        ax.hist(data_1, density=True, bins=bin_edges, histtype="step", lw=2)
        ax.hist(data_2, density=True, bins=bin_edges, histtype="step", lw=2)

        roc_auc = self._get_roc_auc([data_1, data_2])
        ax.set_title(f"ROC AUC = {roc_auc:.3f}")
        ax.set_xlabel(xlabel)

        if legend:
            ax.legend(legend)

        return ax

    @handle_plot_exception
    def plot_elbo(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        self._plot_roc_hist(
            axs[0],
            [self.elbo_dct["sig_ELBO"], self.elbo_dct["bkg_ELBO"]],
            xlabel="ELBO",
            legend=["sig", "bkg"],
        )
        self._plot_roc_hist(
            axs[1],
            [self.elbo_dct["sig_RE"], self.elbo_dct["bkg_RE"]],
            xlabel="RE",
            legend=["sig", "bkg"],
        )
        self._plot_roc_hist(
            axs[2],
            [self.elbo_dct["sig_KL"], self.elbo_dct["bkg_KL"]],
            xlabel="KL",
            legend=["sig", "bkg"],
        )

        plt.tight_layout()
        fig.savefig(f"{self.plotting_dirs['elbo']}elbo_epoch{self.current_epoch:02d}_{self.stage}.png")
        plt.close()

    @handle_plot_exception
    def corner_plot(
        self,
        data: list[np.ndarray],
        labels: list[str] | None = None,
        title: str = "Corner plot",
        max_plot_points: int | None = 20000,
        **kwargs,
    ):
        """Plots a cornerplot for data.

        Parameters
        ----------
        - `data`: list of numpy arrays, each array is a 2D array of shape (n_samples, n_features),
        - `labels`: graph labels, one for each data category,
        - `title`: string, optional, default: 'Corner plot'. The title of the plot,
        - `max_plot_points`: integer, optional, default: 2000. maximum number of points on a scatter plot.

        """

        if data[0].shape[1] == 1:
            fig, ax = plt.subplots()

            for i in range(len(data)):
                ax.hist(
                    data[i][:max_plot_points, :] if max_plot_points is not None else data[i],
                    bins=self.n_bins,
                    density=True,
                    histtype="step",
                    lw=2,
                )

            ax.set_xlabel("z")

            if labels is not None:
                ax.legend(labels)

            if len(data) == 2:
                roc_auc = self._get_roc_auc(data, max_points=max_plot_points)
                plt.title(f"ROC AUC = {roc_auc:.3f}")

            plt.tight_layout()
            fig.savefig(f"{self.plotting_dirs['latent_samples']}latent_epoch{self.current_epoch:02d}_{self.stage}.png")
            plt.close()

            return None

        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig = None
        for i in range(len(data)):
            fig = corner.corner(
                data[i][:max_plot_points, :] if max_plot_points is not None else data[i],
                fig=fig,
                color=cycle[i],
                plot_contours=False,
                no_fill_contours=True,
                plot_density=False,
                **kwargs,
            )

        if labels:
            plt.figlegend(labels)

        plt.suptitle(title)

        plt.tight_layout()
        fig.savefig(f"{self.plotting_dirs['latent_samples']}latent_epoch{self.current_epoch:02d}_{self.stage}.png")

        plt.close()

    @handle_plot_exception
    def gen_vs_ref_plot(self):
        x, y = make_subplots_grid(self.generated.shape[1])
        fig, axs = plt.subplots(x, y, figsize=(4 * y, 3 * x))
        axs = axs.flatten()

        sel = self.proc.selection
        labels = sel[sel["type"] != "label"]["feature"].to_list()

        axs = two_sample_plot(
            self.reference,
            self.generated,
            axs,
            n_bins=self.n_bins,
            log_scale=False,
            label=["True", "Generated"],
            density=True,
            lw=2,
            bin_range=self.bin_range,
            labels=labels,
        )

        plt.tight_layout()
        fig.savefig(f"{self.plotting_dirs['generated']}generated_epoch{self.current_epoch:02d}_{self.stage}.png")
        plt.close()

    @handle_plot_exception
    def pca_plot(
        self,
        data: list[np.ndarray],
        labels: list[str] | None = None,
        title: str = "PCA plot",
        max_plot_points: int = 20000,
        n_components: int = 2,
        use_pca: bool = True,
        save_str: str = "",
        **kwargs,
    ):
        """Plots a principle component analysis for data.

        Parameters
        ----------
        - `data`: list of numpy arrays.
        - `labels`: list of trings,
            graph labels, one for each data category.
        - `title`: string, optional, default: 'Principle component analysis'.
            The title of the plot.
        - `max_plot_points`: integer, optional, default: 2000.
            Maximum number of points on a scatter plot.
        - `n_components`: integer, optional, default: 2.
            The number of components to keep.
        - `pca`: bool, optional, default: True.
            If True, it uses PCA, otherwise TSNE.
        - `kwargs`: optional.
            Additional arguments for scatter or histogram plot.
        - `save_str`: string, optional, default: "".
            Additional string to add to the saved file name.

        """
        if n_components > 2:
            raise ValueError("Only 1 or 2 components are supported!")

        transformed = []

        for d in data:
            d = d[:max_plot_points, :] if max_plot_points is not None else d

            if use_pca:
                pca = PCA(n_components=n_components)
                pca.fit(d)
                transformed.append(pca.transform(d))
            else:
                tsne = TSNE(n_components=n_components)
                transformed.append(tsne.fit_transform(d))

        bins = self.n_bins

        for t in transformed:
            if n_components == 1:
                _, bins, _ = plt.hist(
                    iqr_remove_outliers(t[:max_plot_points, :]),
                    bins=bins,
                    histtype="step",
                    lw=2,
                    density=True,
                    **kwargs,
                )
            else:
                plt.scatter(t[:, 0], t[:, 1], alpha=0.2, s=2, **kwargs)

        if labels:
            plt.legend(labels)

        if title:
            plt.title(title)

        if n_components == 1:
            roc_auc = self._get_roc_auc(transformed)
            plt.title(f"ROC AUC = {roc_auc:.3f}")

        plt.tight_layout()
        plt.savefig(
            f"{self.plotting_dirs['latent_samples']}pca_epoch{self.current_epoch:02d}{save_str}_{self.stage}.png"
        )
        plt.close()
