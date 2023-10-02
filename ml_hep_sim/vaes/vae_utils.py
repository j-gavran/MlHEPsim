import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ml_hep_sim.data_utils.dataset_utils import rescale_data
from ml_hep_sim.data_utils.toy_datasets import TOY_DATASETS
from ml_hep_sim.stats.stat_plots import two_sample_plot


def plot_vae_result_on_event(data_name, model, logger, dataset, idx=0, use_hexbin=False):
    if 0 < idx < 100:
        idx = f"0{idx}"

    def _vae_on_train_end():
        if data_name.lower() == "mnist":
            n = 10
            fig, axs = plt.subplots(n, n, figsize=(15, 15))
            axs = axs.flatten()

            x = model.sample(n * n, device=model.device).cpu().numpy()

            for i in range(n * n):
                axs[i].imshow(x[i, :].reshape(28, 28).clip(0, 1), cmap="gray")

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"MNIST_generated_{idx}.jpg")

        elif data_name.lower() in ["higgs", "higgs_bkg", "higgs_sig"]:
            subset = 10**5
            X = dataset.X[:subset].cpu().numpy()
            sample = model.sample(subset, device=model.device).cpu().numpy()

            scalers = model.scalers
            if len(scalers) != 0:
                X = scalers[2].inverse_transform(X)
                sample = scalers[0].inverse_transform(sample)

            fig, axs = plt.subplots(6, 3, figsize=(10, 10))
            axs = axs.flatten()

            sample = sample[~np.isnan(sample).any(axis=1)]

            two_sample_plot(X, sample, axs, n_bins=50, log_scale=False, density=True, lw=2)

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"Higgs_generated_{idx}.jpg")

            fig, axs = plt.subplots(6, 3, figsize=(10, 10))
            axs = axs.flatten()

            two_sample_plot(X, sample, axs, n_bins=50, log_scale=True, density=False, lw=2)

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"Higgs_generated_log_{idx}.jpg")

        #             subset = 10**4
        #             X_sig = np.load("data/higgs/HIGGS_18_feature_sig_train.npy")[:subset]
        #             X_bkg = np.load("data/higgs/HIGGS_18_feature_bkg_train.npy")[:subset]
        #             X_sig, _ = rescale_data(X_sig[:, 1:], rescale_type="maxabs")
        #             X_bkg, _ = rescale_data(X_bkg[:, 1:], rescale_type="maxabs")
        #
        #             X_sig = torch.from_numpy(X_sig.astype(np.float32)).cuda()
        #             X_bkg = torch.from_numpy(X_bkg.astype(np.float32)).cuda()
        #
        #             with torch.no_grad():
        #                 model.eval()
        #
        #                 mu_sig, logvar_sig = model.vae.encode(X_sig)
        #                 z_sig = model.vae.reparameterize(mu_sig, logvar_sig).cpu().numpy()
        #
        #                 mu_bkg, logvar_bkg = model.vae.encode(X_bkg)
        #                 z_bkg = model.vae.reparameterize(mu_bkg, logvar_bkg).cpu().numpy()
        #
        #                 loss_sigs_rec = []
        #                 loss_sigs_kld = []
        #                 for i in tqdm(range(len(X_sig))):
        #                     recon_batch, mu, logvar = model(X_sig[i, :])
        #                     loss_sig = model.vae.loss_function(recon_batch, X_sig, mu, logvar)
        #                     loss_sigs_rec.append(loss_sig[0].cpu().numpy())
        #                     loss_sigs_kld.append(loss_sig[1].cpu().numpy())
        #
        #                 loss_bkgs_rec = []
        #                 loss_bkgs_kld = []
        #                 for i in tqdm(range(len(X_bkg))):
        #                     recon_batch, mu, logvar = model(X_bkg[i, :])
        #                     loss_bkg = model.vae.loss_function(recon_batch, X_bkg, mu, logvar)
        #                     loss_bkgs_rec.append(loss_bkg[0].cpu().numpy())
        #                     loss_bkgs_kld.append(loss_bkg[1].cpu().numpy())
        #
        #             fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #             _, bins, _ = axs[0].hist(loss_bkgs_kld, bins=50, histtype="step", lw=2)
        #             axs[0].hist(loss_sigs_kld, bins=50, histtype="step", lw=2)
        #
        #             _, bins, _ = axs[1].hist(loss_bkgs_rec, bins=50, histtype="step", lw=2)
        #             axs[1].hist(loss_sigs_rec, bins=50, histtype="step", lw=2)
        #
        #             axs[0].set_title("KL loss")
        #             axs[1].set_title("reconstruction loss")
        #
        #             plt.tight_layout()
        #             logger.experiment.log_figure(logger.run_id, fig, f"losses_{idx}.jpg")
        #
        #             fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #             axs = axs.flatten()
        #             axs[0].scatter(z_bkg[:, 0], z_bkg[:, 1], s=1, alpha=0.2)
        #             axs[0].scatter(z_sig[:, 0], z_sig[:, 1], s=1, alpha=0.2)
        #             axs[0].set_title("z")
        #
        #             axs[1].scatter(mu_bkg[:, 0].cpu().numpy(), mu_bkg[:, 1].cpu().numpy(), s=1, alpha=0.2)
        #             axs[1].scatter(mu_sig[:, 0].cpu().numpy(), mu_sig[:, 1].cpu().numpy(), s=1, alpha=0.2)
        #             axs[1].set_title("mu")
        #
        #             axs[2].scatter(logvar_bkg[:, 0].cpu().numpy(), logvar_bkg[:, 1].cpu().numpy(), s=1, alpha=0.2)
        #             axs[2].scatter(logvar_sig[:, 0].cpu().numpy(), logvar_sig[:, 1].cpu().numpy(), s=1, alpha=0.2)
        #             axs[2].set_title("logvar")
        #
        #             plt.tight_layout()
        #             logger.experiment.log_figure(logger.run_id, fig, f"proj_z_2d_{idx}.jpg")

        elif data_name in TOY_DATASETS:
            subset = 3 * 10**4
            generated = model.sample(subset, device=model.device).cpu().numpy()

            fig, axs = plt.subplots(1, 1, figsize=(10, 8))
            if use_hexbin:
                axs.hexbin(generated[:, 0], generated[:, 1], cmap="jet", extent=[-4, 4, -4, 4], gridsize=150)
            else:
                axs.scatter(generated[:, 0], generated[:, 1], s=0.25)

            axs.set_title("generated")

            logger.experiment.log_figure(logger.run_id, fig, f"2d_test_{idx}.jpg")
        else:
            logging.warning("data_name not implemented...")

    try:
        with torch.no_grad():
            model.eval()
            _vae_on_train_end()
            model.train()

            plt.tight_layout()
            plt.close()
    except Exception as e:
        logging.warning(f"plotting quit with exception {e}")
