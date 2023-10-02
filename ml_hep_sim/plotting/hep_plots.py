import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from uncertainties import unumpy

ALPHA = 0.5
HATCH = "//////"


class StackPlot:
    def __init__(self, x, hists_lst, data_hist=None, subplots_kwargs=None):
        self.x = x
        self.hists_lst = hists_lst
        self.data_hist = data_hist
        self.fig, self.ax = plt.subplots(1, 1, **subplots_kwargs if subplots_kwargs else {})
        self.ax_lower = None

    def plot_stack(self, labels):
        self.ax.stackplot(self.x, *self.hists_lst, step="post", labels=labels)
        return self.ax

    def plot_data(self, label, err=None, capsize=1.5, fmt=".", lw=1, **kwargs):
        self.ax.errorbar(
            self.x[:-1] + 0.5,
            self.data_hist[:-1],
            xerr=0.5,
            yerr=err[:-1],
            ls="none",
            c="k",
            lw=lw,
            capsize=capsize,
            fmt=fmt,
            label=label,
            **kwargs,
        )

        return self.ax

    def plot_mc_errors(self, errors):
        stacked_h = np.zeros(len(self.x))
        for h in self.hists_lst:
            stacked_h += h

        for i, (hi, b_left, b_right) in enumerate(zip(stacked_h, self.x, self.x[1:])):
            err = errors[i]
            d = b_right - b_left
            self.ax.add_patch(
                Rectangle((b_left, hi - err), d, err, zorder=10, color="k", fill=False, alpha=ALPHA, lw=0, hatch=HATCH)
            )
            self.ax.add_patch(
                Rectangle((b_left, hi), d, err, zorder=10, color="k", fill=False, alpha=ALPHA, lw=0, hatch=HATCH)
            )

        return self.ax

    def plot_lower_panel(
        self,
        counts_num,
        counts_den,
        counts_num_err,
        counts_den_err,
        ylabel="",
        label_x_start=0,
        label_x_end=1,
        label_x_pts=7,
        use_errorbar=False,
        ylim=None,
    ):
        self.ax_lower = self.ax.inset_axes(bounds=[0, -0.25, 1, 0.2])

        self.ax_lower.axhline(1, lw=1, c="k", ls="--")
        self.ax_lower.set_ylabel(ylabel, fontsize=15, labelpad=10)

        self.ax_lower.set_xticks(np.linspace(label_x_start, self.x[-1], label_x_pts))
        self.ax_lower.set_xticklabels(
            ["{:.1f}".format(i) for i in np.linspace(label_x_start, label_x_end, label_x_pts)]
        )

        z_idx = np.where(counts_den != 0.0)[0]

        unc_num_arr = unumpy.uarray(counts_num, counts_num_err)[z_idx]
        unc_den_arr = unumpy.uarray(counts_den, counts_den_err)[z_idx]

        ratio = counts_num[z_idx] / counts_den[z_idx]

        ratio_unc = unc_num_arr / unc_den_arr

        ratio_err = np.array([i.std_dev for i in ratio_unc])

        if use_errorbar:
            self.ax_lower.errorbar(
                self.x, ratio, xerr=0.5, yerr=ratio_err, ls="none", c="k", lw=1, capsize=1.5, fmt="."
            )
        else:
            self.ax_lower.scatter(self.x[z_idx][:-1] + 0.5, ratio[:-1], c="k", s=20)

            for j, i in zip(self.x[z_idx][:-1], range(len(self.x[z_idx][:-1]))):
                self.ax_lower.plot([j, j + 1], [ratio[i], ratio[i]], c="k", zorder=10, lw=1)

            for i, (error, b_left, b_right) in enumerate(zip(ratio_err, self.x[z_idx], self.x[z_idx][1:])):
                d = b_right - b_left
                self.ax_lower.add_patch(
                    Rectangle(
                        (b_left, 1 - error),
                        d,
                        error,
                        zorder=10,
                        color="k",
                        fill=False,
                        alpha=ALPHA,
                        lw=0,
                        hatch=HATCH,
                    )
                )
                self.ax_lower.add_patch(
                    Rectangle(
                        (b_left, 1),
                        d,
                        error,
                        zorder=10,
                        color="k",
                        fill=False,
                        alpha=ALPHA,
                        lw=0,
                        hatch=HATCH,
                    )
                )

        self.ax.set_xticks([])
        self.ax.set_xticklabels([])

        if ylim:
            self.ax_lower.set_ylim(ylim)

        return self.ax
