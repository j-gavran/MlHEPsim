import copy
import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logit
from sklearn import metrics
from torchmetrics.classification import BinaryAccuracy

from ml.common.data_utils.feature_scaling import RescalingHandler
from ml.common.nn.gen_model_sampler import GenModelSampler
from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.flow_anomaly import get_density
from ml.custom.HIGGS.analysis.utils import (
    LABELS_MAP,
    get_model,
    get_scaler,
    get_sig_bkg_ref,
    np_sigmoid,
)

CUT_BINNING = {
    "lepton pT": (0, 4),
    "lepton eta": (-4, 4),
    "missing energy": (0, 4),
    "jet1 pT": (0, 4),
    "jet1 eta": (-4, 4),
    "jet2 pT": (0, 4),
    "jet2 eta": (-4, 4),
    "jet3 pT": (0, 4),
    "jet3 eta": (-4, 4),
    "jet4 pT": (0, 4),
    "jet4 eta": (-4, 4),
    "m jj": (0, 3),
    "m jjj": (0, 3),
    "m lv": (0.9, 1.2),
    "m jlv": (0, 3),
    "m bb": (0, 3),
    "m wbb": (0, 3),
    "m wwbb": (0, 3),
}


class Cut(ABC):
    def __init__(self, model_name, N, use_c2st_weights=False, save_dir="ml/custom/HIGGS/analysis/plots/fit_cut"):
        self.model_name = model_name
        self.N = N
        self.use_c2st_weights = use_c2st_weights
        self.save_dir = save_dir

        mkdir(self.save_dir)

        self.selection = None

        self.samples_dct, self.samples_dct_cut = None, None

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def cut(self, cut_threshold=0.5):
        pass

    def get_gen_events_weights(self, gen_events):
        c2st_classifier = get_model(f"BinaryClassifier_{self.model_name}_c2st_gen_model").eval()

        if isinstance(gen_events, np.ndarray):
            gen_events = torch.from_numpy(gen_events).cuda()

        with torch.no_grad():
            weights = c2st_classifier(gen_events).cpu().numpy()
            weights = np.exp(logit(weights))

        return weights

    def _cut(self, cut_threshold, use_as_mask_dct, cut_on_dct):
        cut_masks = dict()

        cut_masks["bkg"] = use_as_mask_dct["bkg"] > cut_threshold
        cut_masks["sig"] = use_as_mask_dct["sig"] > cut_threshold
        cut_masks[self.model_name] = use_as_mask_dct[self.model_name] > cut_threshold

        cut_masks = {k: v.numpy() if not isinstance(v, np.ndarray) else v for k, v in cut_masks.items()}

        dct_cut = dict()

        dct_cut["bkg"] = cut_on_dct["bkg"][cut_masks["bkg"]]
        dct_cut["sig"] = cut_on_dct["sig"][cut_masks["sig"]]
        dct_cut[self.model_name] = cut_on_dct[self.model_name][cut_masks[self.model_name]]

        return dct_cut, cut_masks

    def plot_after_cut(self, n_bins=100, legend=None, postfix="", cs=None):
        fig, axs = plt.subplots(6, 3, figsize=(16, 24))
        axs = axs.flatten()

        bin_edges = [None for _ in range(self.samples_dct_cut["bkg"].shape[1])]

        bin_ranges = list(CUT_BINNING.values())

        if cs is None:
            cs = [f"C{i}" for i in range(len(self.samples_dct_cut))]

        for j, (name, sample) in enumerate(self.samples_dct_cut.items()):
            logging.info(f"Plotting {name} after cut")

            for i in range(sample.shape[1]):
                bins = bin_edges[i] if bin_edges[i] is not None else n_bins

                _, bin_edge, _ = axs[i].hist(
                    sample[:, i],
                    bins=bins,
                    range=bin_ranges[i],
                    histtype="step",
                    lw=2,
                    color=cs[j],
                )

                if bins is None:
                    bin_edges[i] = bin_edge

        labels = list(LABELS_MAP.values())

        for i, ax in enumerate(axs):
            ax.set_xlim(bin_ranges[i])
            ax.set_xlabel(labels[i])
            ax.set_ylabel("$N$")

        axs[0].legend(legend, loc="upper right")

        fig.tight_layout()
        plt.savefig(f"{self.save_dir}/after_cut{postfix}.pdf")
        plt.close(fig)

    def plot_roc_acc(self, samples, cut_threshold=0.5, M=-1, postfix="", use_sigmoid=False):
        sig = samples["sig"][:M]
        bkg = samples["bkg"][:M]
        bkg_gen = samples[self.model_name][:M]

        if type(sig) is not torch.Tensor:
            sig = torch.from_numpy(sig)

        if type(bkg) is not torch.Tensor:
            bkg = torch.from_numpy(bkg)

        if type(bkg_gen) is not torch.Tensor:
            bkg_gen = torch.from_numpy(bkg_gen)

        scores = torch.cat([bkg, sig], dim=0)
        y = torch.cat([torch.zeros_like(bkg), torch.ones_like(sig)], dim=0)
        fpr, tpr, _ = metrics.roc_curve(y.numpy(), scores.numpy())

        scores_gen = torch.cat([bkg_gen, sig], dim=0)
        y_gen = torch.cat([torch.zeros_like(bkg_gen), torch.ones_like(sig)], dim=0)
        fpr_gen, tpr_gen, _ = metrics.roc_curve(y_gen.numpy(), scores_gen.numpy())

        if use_sigmoid:
            scores, scores_gen = torch.sigmoid(scores), torch.sigmoid(scores_gen)
            cut_threshold = torch.sigmoid(torch.tensor(cut_threshold)).item()

        acc_metric = BinaryAccuracy(threshold=cut_threshold).cpu()

        acc = acc_metric(scores, y)
        acc_gen = acc_metric(scores_gen, y_gen)

        auc, auc_gen = metrics.auc(fpr, tpr), metrics.auc(fpr_gen, tpr_gen)

        plt.plot(fpr, tpr, lw=2, label=f"MC bkg vs sig: AUC = {auc:.4f}, ACC ={acc:.4f}")
        plt.plot(fpr_gen, tpr_gen, lw=2, label=f"ML  vs MC sig: AUC = {auc_gen:.4f}, ACC ={acc_gen:.4f}")

        plt.plot([0, 1], [0, 1], "k--", lw=2)

        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

        plt.legend(loc="lower right")

        plt.grid(alpha=0.5)

        logging.info(f"AUC - bkg vs sig: {auc:.4f}, generated bkg vs sig: {auc_gen:.4f}")
        logging.info(f"ACC - bkg vs sig: {acc:.4f}, generated bkg vs sig: {acc_gen:.4f}")
        logging.info("Plotting ROC curve")

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/roc_curve{postfix}.pdf")
        plt.close()


class ClassifierCut(Cut):
    def __init__(self, model_name, N, classifier_name, clip_dct_N=None, **kwargs):
        super().__init__(model_name, N, **kwargs)
        self.classifier_name = classifier_name

        if clip_dct_N is not None:
            assert set(clip_dct_N.keys()).issubset(["bkg", "sig", model_name]), "Invalid keys in clip_dct_N"

        self.clip_dct_N = clip_dct_N

        self.classifier, self.samples_dct, self.scalers_dct, self.scaler = self.setup()

        self.classifier_samples_dct = None

    def setup(self):
        # get reference data - start with N(MC-sig)=N(MC-bkg)=N = N_ML (what if we run out??)
        bkg_ref, sig_ref, self.selection = get_sig_bkg_ref(self.N)

        # sample gen. model
        sampler = GenModelSampler(self.model_name, save_dir="ml/data/higgs", file_name="HIGGS_generated")
        samples_dct = sampler.sample(self.N)

        # reweight
        if self.use_c2st_weights:
            logging.critical("[red]Using C2ST weights![/red]")
            c2st_input = samples_dct[self.model_name][0]
            weights = self.get_gen_events_weights(c2st_input)
            samples_dct[self.model_name] = [c2st_input * weights]

        # get scalers
        scalers_dct = dict()

        scalers_dct[self.model_name] = get_scaler(self.model_name)  # pure bkg scaler

        scaler = RescalingHandler(self.selection)
        samples_dct = scaler.rescale_from_dct(samples_dct, scalers_dct, forward=False)

        # add ref data
        samples_dct["bkg"] = bkg_ref
        samples_dct["sig"] = sig_ref

        # clip to expected number of events
        if self.clip_dct_N is not None:
            logging.info(f"[yellow]Clipping to {self.clip_dct_N} events![/yellow]")
            samples_dct["bkg"] = samples_dct["bkg"][: self.clip_dct_N["bkg"]]
            samples_dct["sig"] = samples_dct["sig"][: self.clip_dct_N["sig"]]
            samples_dct[self.model_name] = [samples_dct[self.model_name][0][: self.clip_dct_N[self.model_name]]]

        # get classifier scaler
        classifier_scaler = get_scaler(self.classifier_name)

        # get back to Gauss rank of classifier
        if (
            scalers_dct[self.model_name]["cont"][0][0] == "gauss rank transform"
            and classifier_scaler["cont"][0][0] == "gauss rank transform"
        ):
            combined_scaler = copy.deepcopy(scalers_dct[self.model_name]["cont"][0][1])
            # fit again on the combined sample!
            combined_scaler.training = True
            logging.info(
                (
                    "[yellow]Fitting combined Gauss rank scaler on combined sig-bkg-gen sample![/yellow] "
                    "[yellow]Make sure to use a classifier trained on sig-bkg-gen sample![/yellow]"
                )
            )

            combined_sample = np.concatenate(
                [samples_dct["bkg"], samples_dct["sig"], samples_dct[self.model_name][0]],
                axis=0,
            )

            combined_scaler.interp_limit = len(combined_sample)
            combined_sample = combined_scaler.fit_transform(combined_sample)

            l_bkg, l_sig = len(samples_dct["bkg"]), len(samples_dct["sig"])

            samples_dct["bkg"] = combined_sample[:l_bkg]
            samples_dct["sig"] = combined_sample[l_bkg : l_bkg + l_sig]
            samples_dct[self.model_name] = combined_sample[l_bkg + l_sig :]

            combined_scaler = {"disc": None, "cont": [("gauss rank transform", combined_scaler)]}

            scalers_dct["bkg"] = combined_scaler
            scalers_dct["sig"] = combined_scaler
            scalers_dct[self.model_name] = combined_scaler

        # all other scalers
        else:
            scalers_dct["bkg"] = classifier_scaler
            scalers_dct["sig"] = scalers_dct["bkg"]
            scalers_dct[self.model_name] = scalers_dct["bkg"]  # switch scalers to classifier one

            samples_dct = scaler.rescale_from_dct(samples_dct, scalers_dct, forward=True)

            # get rid of resamples
            samples_dct = {k: v[0] for k, v in samples_dct.items()}

        # get classifier
        classifier = get_model(self.classifier_name, ver=-1).eval()

        return classifier, samples_dct, scalers_dct, scaler

    def run_classifier(self, chunks=20):
        classifier_samples_dct = {"bkg": [], "sig": [], self.model_name: []}

        chunks_lst = chunks * [self.N // chunks]

        if self.N % chunks != 0:
            chunks_lst += [self.N % chunks]

        for i, chunk in enumerate(chunks_lst):
            s = i * chunk  # start
            e = (i + 1) * chunk  # end
            # if i > chunks: # BPK FIX
            #     s=i*chunks_lst[chunks-1]
            #     e = s+  chunk  # end
            logging.info(f"Chunk {i + 1}/{chunks}, start={s} and end={e}")

            # k="bkg"
            # logging.info(f"Classifier array  key={k} and len={len(self.samples_dct[k])}")
            with torch.no_grad():
                classifier_samples_dct["bkg"].append(
                    self.classifier(torch.from_numpy(self.samples_dct["bkg"][s:e]).cuda()).squeeze().cpu()
                )
                classifier_samples_dct["sig"].append(
                    self.classifier(torch.from_numpy(self.samples_dct["sig"][s:e]).cuda()).squeeze().cpu()
                )
                classifier_samples_dct[self.model_name].append(
                    self.classifier(torch.from_numpy(self.samples_dct[self.model_name][s:e]).cuda()).squeeze().cpu()
                )

        for k, v in classifier_samples_dct.items():
            # logging.info(f"Classifier key={k} , type={type(v)} and shape={bool(v[10].size())}")
            empt = torch.Tensor()  # BPK fix for empty tensors ...
        classifier_samples_dct = {
            k: torch.cat([x if bool(x.size()) else x + empt for x in v], dim=0)
            for k, v in classifier_samples_dct.items()
        }
        # classifier_samples_dct = {k: torch.cat(v, dim=0) for k, v in classifier_samples_dct.items()}
        lbkg = len(self.samples_dct["bkg"])
        lsig = len(self.samples_dct["sig"])
        lmod = len(self.samples_dct[self.model_name])
        logging.info(f"Inp. classifier sample len bkg={lbkg} sig={lsig} and mod={lmod}")
        lbkg = len(classifier_samples_dct["bkg"])
        lsig = len(classifier_samples_dct["sig"])
        lmod = len(classifier_samples_dct[self.model_name])
        logging.info(f"Classifier sample len bkg={lbkg} sig={lsig} and mod={lmod}")

        return classifier_samples_dct

    def plot_classifier_cut(
        self, cut_threshold=0.5, n_bins=50, bin_range=(-0.05, 1.05), density=True, M=-1, log_scale=False
    ):
        if self.classifier_samples_dct is None:
            self.classifier_samples_dct = self.run_classifier()

        # plot classifier output
        fig, ax = plt.subplots(1, 1)

        mc_bkg, bins, _ = ax.hist(
            self.classifier_samples_dct["bkg"][:M].numpy(),
            bins=n_bins,
            histtype="step",
            lw=1,
            label="MC bkg",
            density=density,
            range=bin_range,
        )
        ax.hist(
            self.classifier_samples_dct["sig"][:M].numpy(),
            bins=bins,
            histtype="step",
            lw=1,
            label="MC sig",
            density=density,
        )
        ml_bkg, _, _ = ax.hist(
            self.classifier_samples_dct[self.model_name][:M].numpy(),
            bins=bins,
            histtype="step",
            lw=1,
            label="ML generated",
            density=density,
        )

        if cut_threshold is not None:
            ax.axvline(cut_threshold, color="r", linestyle="--", lw=2, label="cut")

        ax.set_xlabel("classifier score")
        ax.set_ylabel("density [a.u.]")

        ax.set_xlim(bin_range)

        if log_scale:
            ax.set_yscale("log")

        ax.legend(loc="upper left", ncol=2)

        logging.info("Plotting classifier cut")

        fig.tight_layout()
        plt.savefig(f"{self.save_dir}/classifier_sigmoid_sig_bkg.pdf")
        plt.close(fig)

        # plot ratio
        fig, ax = plt.subplots(1, 1)
        plt.plot(bins[:-1], mc_bkg / ml_bkg, lw=2, label="MC / ML generated")

        ax.axhline(1, color="r", linestyle="--", lw=2, label="ratio = 1")

        ax.set_xlabel("classifier score")
        ax.set_ylabel("MC / ML ")

        ax.set_xlim(bin_range)
        ax.set_ylim(bottom=0.5, top=1.5)
        ax.legend()

        fig.tight_layout()
        plt.savefig(f"{self.save_dir}/classifier_sigmoid_sig_bkg_ratio.pdf")
        plt.close(fig)

    def cut(self, cut_threshold=0.5, cut_variables=True):
        if self.classifier_samples_dct is None:
            self.classifier_samples_dct = self.run_classifier()

        if cut_variables:
            # cut on features
            self.samples_dct_cut, cut_masks = self._cut(
                cut_threshold,
                use_as_mask_dct=self.classifier_samples_dct,
                cut_on_dct=self.samples_dct,
            )
        else:
            # cut on classifier output score
            self.samples_dct_cut, cut_masks = self._cut(
                cut_threshold,
                use_as_mask_dct=self.classifier_samples_dct,
                cut_on_dct=self.classifier_samples_dct,
            )
            self.samples_dct_cut = {k: v[:, None] for k, v in self.samples_dct_cut.items()}

        if cut_variables:
            # get back to original scaling
            self.samples_dct_cut = self.scaler.rescale_from_dct(self.samples_dct_cut, self.scalers_dct, forward=False)
            # get rid of resamples
            self.samples_dct_cut = {k: v[0] for k, v in self.samples_dct_cut.items()}

        e_model, e_bkg, e_sig = (
            len(self.samples_dct_cut[self.model_name]),
            len(self.samples_dct_cut["bkg"]),
            len(self.samples_dct_cut["sig"]),
        )
        i_model, i_bkg, i_sig = (
            len(self.samples_dct[self.model_name]),
            len(self.samples_dct["bkg"]),
            len(self.samples_dct["sig"]),
        )
        logging.info(f"Events entrering cut for classifier - bkg gen: {i_model}, bkg: {i_bkg},sig: {i_sig}")
        logging.info(f"Events surviving cut for classifier - bkg gen: {e_model}, bkg: {e_bkg},sig: {e_sig}")
        logging.info(
            f"After cut ratios - bkg gen: {e_model / i_model:.3f} bkg: {e_bkg / i_bkg:.3f}, sig: {e_sig / i_sig:.3f}"
        )

        return self.samples_dct_cut

    def plot_after_cut(self, n_bins=70):
        colors = ["C0", "C1", "C2"]
        return super().plot_after_cut(
            n_bins, legend=["MC bkg", "MC sig", "ML generated"], postfix="_classifier", cs=colors
        )

    def plot_roc_acc(self, cut_threshold=0.5, M=-1):
        return super().plot_roc_acc(self.classifier_samples_dct, cut_threshold, M, postfix="_classifier")


class FlowCut(Cut):
    def __init__(self, model_name, N, **kwargs):
        super().__init__(model_name, N, **kwargs)

        self.model, self.samples_dct, self.scaler = self.setup()

        self.density_dct = None

    def setup(self):
        # get reference data
        bkg_ref, sig_ref, self.selection = get_sig_bkg_ref(self.N)

        # sample gen. model
        sampler = GenModelSampler(self.model_name, save_dir="ml/data/higgs", file_name="HIGGS_generated")
        samples_dct = sampler.sample(self.N)

        # reweight
        if self.use_c2st_weights:
            logging.warning("[red]Using C2ST weights![/red]")
            c2st_input = samples_dct[self.model_name][0]

            weights = self.get_gen_events_weights(c2st_input)
            samples_dct[self.model_name] = [weights * c2st_input]

        samples_dct[self.model_name] = samples_dct[self.model_name][0]

        # get model for density estimation
        model = get_model(self.model_name, ver=-1).eval()
        scaler = get_scaler(self.model_name, ver=-1)

        scaler = RescalingHandler(self.selection, scaler)

        bkg_ref = scaler.transform(bkg_ref)
        sig_ref = scaler.transform(sig_ref)

        samples_dct["bkg"] = bkg_ref
        samples_dct["sig"] = sig_ref

        return model, samples_dct, scaler

    def estimate_density(self):
        density_dct = dict()
        density_dct[self.model_name] = get_density(self.model, self.samples_dct[self.model_name], chunks=20)
        density_dct["bkg"] = get_density(self.model, self.samples_dct["bkg"], chunks=20)
        density_dct["sig"] = get_density(self.model, self.samples_dct["sig"], chunks=20)

        return density_dct

    def plot_density_cut(
        self,
        n_bins=200,
        bin_range=(-10, 60),
        cut_thereshold=None,
        use_sigmoid=False,
        log_scale=False,
    ):
        fig, ax = plt.subplots(1, 1)

        if self.density_dct is None:
            self.density_dct = self.estimate_density()

        ax.hist(
            np_sigmoid(self.density_dct["bkg"]) if use_sigmoid else self.density_dct["bkg"],
            bins=n_bins,
            histtype="step",
            lw=2,
            label="MC bkg",
            density=True,
            range=bin_range,
        )
        ax.hist(
            np_sigmoid(self.density_dct["sig"]) if use_sigmoid else self.density_dct["sig"],
            bins=n_bins,
            histtype="step",
            lw=2,
            label="MC sig",
            density=True,
            range=bin_range,
        )
        ax.hist(
            np_sigmoid(self.density_dct[self.model_name]) if use_sigmoid else self.density_dct[self.model_name],
            bins=n_bins,
            histtype="step",
            lw=2,
            label="ML generated",
            density=True,
            range=bin_range,
        )

        ax.set_xlabel("ML log density", loc="left" if use_sigmoid else "right")
        ax.set_ylabel("density [a.u.]")

        if cut_thereshold is not None:
            ax.axvline(cut_thereshold, color="r", linestyle="--", lw=2, label="cut")

        ax.legend()
        ax.set_xlim(bin_range)

        if log_scale:
            ax.set_yscale("log")

        fig.tight_layout()

        logging.info("Plotting density cut")

        if use_sigmoid:
            plt.savefig(f"{self.save_dir}/flow_density_sig_bkg_sigmoid.pdf")
        else:
            plt.savefig(f"{self.save_dir}/flow_density_sig_bkg.pdf")

        plt.close(fig)

    def cut(self, cut_threshold=12.0, cut_variables=False):
        if self.density_dct is None:
            self.density_dct = self.estimate_density()

        if cut_variables:
            self.samples_dct_cut, _ = self._cut(
                cut_threshold,
                use_as_mask_dct=self.density_dct,
                cut_on_dct=self.samples_dct,
            )
        else:
            self.samples_dct_cut, _ = self._cut(
                cut_threshold,
                use_as_mask_dct=self.density_dct,
                cut_on_dct=self.density_dct,
            )
            self.samples_dct_cut = {k: v[:, None] for k, v in self.samples_dct_cut.items()}

        if cut_variables:
            self.samples_dct_cut[self.model_name] = self.scaler.inverse_transform(self.samples_dct_cut[self.model_name])
            self.samples_dct_cut["bkg"] = self.scaler.inverse_transform(self.samples_dct_cut["bkg"])
            self.samples_dct_cut["sig"] = self.scaler.inverse_transform(self.samples_dct_cut["sig"])

        e_model, e_bkg, e_sig = (
            len(self.samples_dct_cut[self.model_name]),
            len(self.samples_dct_cut["bkg"]),
            len(self.samples_dct_cut["sig"]),
        )
        logging.info(f"Events surviving cut for flow - bkg gen: {e_model}, bkg: {e_bkg},sig: {e_sig}")
        logging.info(
            f"After cut ratios - bkg gen: {e_model / self.N:.3f} bkg: {e_bkg / self.N:.3f}, sig: {e_sig / self.N:.3f}"
        )

        return self.samples_dct_cut

    def plot_after_cut(self, n_bins=70):
        colors = ["C2", "C0", "C1"]
        return super().plot_after_cut(n_bins, legend=["ML generated", "MC bkg", "MC sig"], postfix="_flow", cs=colors)

    def plot_roc_acc(self, cut_threshold=0.5, M=-1):
        return super().plot_roc_acc(self.density_dct, cut_threshold, M, postfix="_flow", use_sigmoid=True)


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    N = 10**6
    model_name = "MADEMOG_flow_model_gauss_rank_best"
    classifier_model = "BinaryClassifier_sigbkg_gauss_rank_best7"

    cut_threshold = 0.55
    classifier_cut = ClassifierCut(model_name, N, classifier_model, use_c2st_weights=False)
    classifier_cut.plot_classifier_cut(n_bins=100, cut_threshold=cut_threshold, M=-1, log_scale=False, density=False)
    classifier_cut.cut(cut_threshold=cut_threshold)
    classifier_cut.plot_after_cut()
    classifier_cut.plot_roc_acc(cut_threshold=cut_threshold, M=-1)

    # TODO: fix this
    # flow_cut = FlowCut(model_name, N, use_c2st_weights=True)
    # flow_cut.plot_density_cut(cut_thereshold=13.2)
    # flow_cut.plot_density_cut(cut_thereshold=None, bin_range=(1 - 1e-4, 1 + 1e-5), use_sigmoid=True, log_scale=True)
    # flow_cut.cut(cut_threshold=13.2)
    # flow_cut.plot_after_cut()
    # flow_cut.plot_roc_acc(cut_threshold=13.2)
