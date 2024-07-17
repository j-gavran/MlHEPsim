import logging

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.utils import (
    LABELS_MAP,
    MODEL_MAP,
    equalize_counts_to_ref,
    get_model,
    get_scaler,
    rescale_back,
    run_chainer,
    sample_from_models,
)


class CutTest:
    def __init__(self, select_model, cont_rescale_type, N, save_dir="ml/custom/HIGGS/analysis/plots/cut_tests"):
        self.select_model = select_model
        self.cont_rescale_type = cont_rescale_type
        self.N = N
        self.save_dir = save_dir

        mkdir(self.save_dir)

        self.ref, self.gen, self.selection = self.setup()

    def setup(self):
        bkg_ref_data, selection, _ = run_chainer(
            n_data=-1,
            return_data=True,
            cont_rescale_type="none",
            disc_rescale_type="none",
            drop_types=["uni", "disc"],
            use_hold=True,
            on_train="bkg",
        )

        label_idx = selection[selection["feature"] == "label"].index

        # analysis on background data, so remove label
        label_mask = np.ones(bkg_ref_data.shape[1], dtype=bool)
        label_mask[label_idx] = False
        bkg_ref_data = bkg_ref_data[:, label_mask]

        bkg_ref_data = bkg_ref_data[: self.N]

        # filter models
        model_dct = {self.select_model: get_model(self.select_model, ver=1).eval()}

        # fetch scalers
        scalers_dct = {self.select_model: get_scaler(self.select_model, ver=1)}

        # sample from models
        samples = sample_from_models(model_dct, self.N, chunks=20, resample=1)

        # rescale back
        samples = rescale_back(samples, scalers_dct, selection)

        # add reference data to samples
        samples["ref"] = [bkg_ref_data]

        # make sure all samples have the same number of events
        samples = equalize_counts_to_ref(samples)

        return samples["ref"][0], samples[self.select_model][0], selection

    def make_cut(self, cut_value, feature_name):
        feature_idx = self.selection[self.selection["feature"] == feature_name].index

        ref_cut, gen_cut = self.ref[:, feature_idx].flatten(), self.gen[:, feature_idx].flatten()
        ref_cut_mask, gen_cut_mask = ref_cut > cut_value, gen_cut > cut_value

        return ref_cut, gen_cut, ref_cut_mask, gen_cut_mask

    def make_N_cuts(self, n_cuts, cut_start, cut_end, feature_name, apply_mask=True):
        cuts = np.linspace(cut_start, cut_end, n_cuts)

        cuts_dct = dict()

        for cut in cuts:
            ref_cut, gen_cut, ref_cut_mask, gen_cut_mask = self.make_cut(cut, feature_name)

            if apply_mask:
                cuts_dct[cut] = {"ref": ref_cut[ref_cut_mask], "gen": gen_cut[gen_cut_mask]}
            else:
                cuts_dct[cut] = {"ref": ref_cut, "gen": gen_cut, "ref_mask": ref_cut_mask, "gen_mask": gen_cut_mask}

        return cuts_dct

    def plot_N_cuts(
        self,
        feature_name,
        axs,
        n_cuts=None,
        cut_start=None,
        cut_end=None,
        fig=None,
        bin_end_range=10,
        save=True,
        log_scale=False,
        cuts_dct=None,
        postfix="",
    ):
        if cuts_dct is None:
            cuts_dct = self.make_N_cuts(n_cuts, cut_start, cut_end, feature_name)
            make_cut_dct = True
        else:
            make_cut_dct = False

        feature_idx = self.selection[self.selection["feature"] == feature_name].index

        ref = self.ref[:, feature_idx].flatten()

        bins = []

        for i in range(n_cuts):
            _, bin_cut, _ = axs[i].hist(
                ref,
                bins=100,
                histtype="stepfilled",
                label="ref w/o cut",
                range=(cut_start, bin_end_range),
                color="k",
                alpha=0.25,
            )
            bins.append(bin_cut)

        for i, (cut, cut_dct) in enumerate(cuts_dct.items()):
            if make_cut_dct:
                h = cut_dct["gen"]
            else:
                h = self.gen[:, feature_idx].flatten()
                h = h[cut_dct["gen_mask"]]

            axs[i].hist(
                h,
                bins=bins[i],
                histtype="step",
                lw=0.8,
                label=MODEL_MAP[self.select_model],
                range=(cut, bin_end_range),
                color="C0",
            )
            axs[i].set_xlabel(f"{LABELS_MAP[feature_name]} > {cut:.2f}")
            axs[i].set_ylabel("$N$")

        axs[0].legend()

        if log_scale:
            for ax in axs:
                ax.set_yscale("log")

        if save:
            logging.info(
                f"[green]Saving cut plot for {feature_name} with {n_cuts} cuts{postfix.replace('_', ' ')}[/green]"
            )

            save_str = f"{self.save_dir}/cut_plot_{feature_name}_{n_cuts}_{self.select_model}{postfix}.pdf"
            save_str = save_str.replace(" ", "_")

            fig.tight_layout()
            fig.savefig(save_str)
            plt.close(fig)

        return axs

    def plot_all_feature_cuts(self):
        features = list(LABELS_MAP.keys())

        for feature_name in features:

            fig, axs = plt.subplots(5, 5, figsize=(24, 20))
            axs = axs.flatten()

            if "eta" in feature_name:
                cut_tuple, bin_end_range = (25, -4.0, 4.0), 5.0
            else:
                cut_tuple, bin_end_range = (25, 0.0, 3.0), 4.0

            self.plot_N_cuts(
                feature_name,
                n_cuts=cut_tuple[0],
                cut_start=cut_tuple[1],
                cut_end=cut_tuple[2],
                axs=axs,
                fig=fig,
                bin_end_range=bin_end_range,
            )

    def plot_N_minus_one_cuts(self, cut_feature):
        features = list(LABELS_MAP.keys())

        if "eta" in cut_feature:
            cut_tuple, bin_end_range = (25, -4.0, 4.0), 5.0
        else:
            cut_tuple, bin_end_range = (25, 0.0, 3.0), 4.0

        cuts_dct = self.make_N_cuts(*cut_tuple, cut_feature, apply_mask=False)

        for feature_name in features:
            fig, axs = plt.subplots(5, 5, figsize=(24, 20))
            axs = axs.flatten()

            self.plot_N_cuts(
                feature_name,
                axs=axs,
                fig=fig,
                n_cuts=cut_tuple[0],
                cut_start=cut_tuple[1],
                cut_end=cut_tuple[2],
                cuts_dct=cuts_dct,
                bin_end_range=bin_end_range,
                postfix=f"_minus_{cut_feature}",
            )

    def plot_N_2d_cuts(
        self,
        cut_tuple_x,
        cut_tuple_y,
        feature_name_x,
        feature_name_y,
        axs_dct,
        fig_dct=None,
        bin_end_range_tuple=(10, 10),
        log_scale=False,
        save=True,
    ):
        cuts_dct_x = self.make_N_cuts(*cut_tuple_x, feature_name_x, apply_mask=False)
        cuts_dct_y = self.make_N_cuts(*cut_tuple_y, feature_name_y, apply_mask=False)

        n_cuts = f"{cut_tuple_x[0]}_{cut_tuple_y[0]}"

        # plot 2d cuts histograms

        hists2d = {"gen": [], "ref": [], "cuts": []}
        for j, label in enumerate(["gen", "ref"]):

            axs = axs_dct[label]
            if save:
                fig = fig_dct[label]

            c = 0
            for cut_x, cut_dct_x in tqdm(cuts_dct_x.items(), leave=False):
                x = cut_dct_x[label]
                x_mask = cut_dct_x[f"{label}_mask"]

                for cut_y, cut_dct_y in tqdm(cuts_dct_y.items(), leave=False):
                    y = cut_dct_y[label]
                    y_mask = cut_dct_y[f"{label}_mask"]

                    xy_mask = x_mask & y_mask
                    x_cut_arr, y_cut_arr = x[xy_mask], y[xy_mask]

                    h, _, _, _ = axs[c].hist2d(
                        x_cut_arr,
                        y_cut_arr,
                        bins=80,
                        range=[(cut_x, bin_end_range_tuple[0]), (cut_y, bin_end_range_tuple[1])],
                        cmap="viridis",
                    )
                    hists2d[label].append(h)

                    if j == 0:
                        hists2d["cuts"].append((cut_x, cut_y))

                    axs[c].set_xlabel(f"{label} {LABELS_MAP[feature_name_x]} > {cut_x:.2f}")
                    axs[c].set_ylabel(f"{label} {LABELS_MAP[feature_name_y]} > {cut_y:.2f}")

                    if log_scale:
                        axs[c].set_xscale("log")
                        axs[c].set_yscale("log")

                    axs[c].set_xlim([cut_x, bin_end_range_tuple[0]])
                    axs[c].set_ylim([cut_y, bin_end_range_tuple[1]])

                    c += 1

            if save:
                logging.info(
                    f"[green]Saving 2d cut plot {feature_name_x}/{feature_name_y} for {label} with {n_cuts} cuts[/green]"
                )

                save_str = f"{self.save_dir}/cut_plot_2d_{feature_name_x}_{feature_name_y}_{n_cuts}_{label}_{self.select_model}.png"
                save_str = save_str.replace(" ", "_")

                fig.tight_layout()
                fig.savefig(
                    save_str,
                    dpi=300,
                )
                plt.close(fig)

        # plot difference

        axs, fig = axs_dct["diff"], fig_dct["diff"]
        for i, (gen_hists2d, ref_hists2d, cuts) in enumerate(zip(hists2d["gen"], hists2d["ref"], hists2d["cuts"])):
            ratio = np.abs(gen_hists2d - ref_hists2d)

            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.08)

            im = axs[i].imshow(
                ratio,
                cmap="viridis",
                origin="lower",
            )

            cbar = plt.colorbar(im, cax=cax, orientation="vertical")
            cbar.ax.set_title("|gen - ref|", fontsize=16)

            axs[i].set_xlabel(f"{LABELS_MAP[feature_name_x]} > {cuts[0]:.2f} bins")
            axs[i].set_ylabel(f"{LABELS_MAP[feature_name_y]} > {cuts[1]:.2f} bins")

        if save:
            logging.info("[green]Saving 2d cut plot difference.[/green]")

            save_str = (
                f"{self.save_dir}/cut_plot_2d_diff_{feature_name_x}_{feature_name_y}_{n_cuts}_{self.select_model}.png"
            )
            save_str = save_str.replace(" ", "_")

            fig.tight_layout()
            fig.savefig(
                save_str,
                dpi=300,
            )
            plt.close(fig)

        return axs


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    # make cut plot
    cut_test = CutTest(
        select_model="MADEMOG_flow_model_gauss_rank",
        cont_rescale_type="gauss_rank",
        N=2 * 10**6,
    )

    # plot 1d cuts
    # cut_test.plot_all_feature_cuts()
    # cut_test.plot_N_minus_one_cuts("m bb")

    # plot 2d cuts
    fig_ref, axs_ref = plt.subplots(4, 4, figsize=(22, 20))
    axs_ref = axs_ref.flatten()

    fig_gen, axs_gen = plt.subplots(4, 4, figsize=(22, 20))
    axs_gen = axs_gen.flatten()

    fig_d, axs_d = plt.subplots(4, 4, figsize=(22, 20))
    axs_d = axs_d.flatten()

    cut_test.plot_N_2d_cuts(
        cut_tuple_x=(4, 0.26, 1.5),
        cut_tuple_y=(4, 0.0, 1.5),
        feature_name_x="lepton pT",
        feature_name_y="missing energy",
        axs_dct={"ref": axs_ref, "gen": axs_gen, "diff": axs_d},
        fig_dct={"ref": fig_ref, "gen": fig_gen, "diff": fig_d},
        bin_end_range_tuple=(2, 2),
    )
