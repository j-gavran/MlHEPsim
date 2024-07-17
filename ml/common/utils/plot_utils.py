import logging

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


def step_hist_plot(ax, hist, bin_edges, label=None, **kwargs):
    """Step histogram plot."""
    hep.histplot(
        hist,
        bin_edges,
        ax=ax,
        histtype="step",
        label=label,
        **kwargs,
    )
    return ax


def errorbar_plot(ax, x, y, xerr, yerr, markersize=5, linestyle="None", color="black", marker="o", label=None):
    """Plot points with error bars."""
    ax.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        markersize=markersize,
        linestyle=linestyle,
        color=color,
        marker=marker,
        label=label,
    )
    return ax


def stacked_hist_plot(ax, hists, bin_edges, labels=None, colors=None, **kwargs):
    """Stacked histogram plot.

    https://mplhep.readthedocs.io/en/latest/api.html

    Parameters
    ----------
    ax : axes object
        Matplotlib axes object.
    hists : list of np.arrays
        List of histogram counts.
    bin_edges : np.array
        Array of bin edges.
    labels : list of str
        List of names for the histograms.
    colors : list of str, optional
        List of colors for the histograms, by default None.

    Returns
    -------
    ax : axes object
        Matplotlib axes object.
    """
    hep.histplot(
        hists,
        bins=bin_edges,
        label=labels,
        stack=True,
        histtype="fill",
        alpha=[1.0] * len(hists),
        edgecolor=["k"] * len(hists),
        linewidth=0.15,
        color=colors,
        ax=ax,
        **kwargs,
    )
    return ax


def add_hatched_error(ax, bin_edges, ys, yerrs):
    """Helper function to add hatched error to the plot."""
    dxs = 0.5 * np.diff(bin_edges)  # bin widths
    xs = bin_edges[:-1] + dxs  # bin centers

    for x, dx, y, y_err in zip(xs, dxs, ys, yerrs):
        ax.add_patch(
            Rectangle(
                (x - dx, y - y_err / 2),
                2 * dx,
                y_err,
                hatch="/////",
                fill=False,
                color="k",
                linewidth=0.0,
                zorder=100,
            )
        )
    return ax


def add_data_mc_ratio(
    ax,
    bin_edges,
    data_hist,
    data_yerr,
    mc_hists,
    mc_yerrs,
    ylim=(0.5, 1.5),
    lower_ylabel="Data / MC",
):
    """Add data/MC ratio plot to the axes as lower panel."""
    axin = ax.inset_axes(bounds=[0, -0.24, 1, 0.2])

    mc_hist = np.sum(mc_hists, axis=0)
    ratio_data_mc = data_hist / mc_hist

    mc_yerr = np.sqrt(np.sum(mc_yerrs**2, axis=0))

    ratio_yerr = np.sqrt((data_yerr / data_hist) ** 2 + (mc_yerr / mc_hist) ** 2) * ratio_data_mc

    dx = 0.5 * np.diff(bin_edges)
    axin = errorbar_plot(
        axin,
        x=bin_edges[:-1] + dx,
        y=ratio_data_mc,
        xerr=dx,
        yerr=ratio_yerr,
    )

    axin.set_xlim(ax.get_xlim())
    axin.axhline(1, lw=1, c="k", ls="--", alpha=0.8)

    ax.tick_params(labelbottom=False)
    axin.set_xlabel(ax.get_xlabel())
    ax.set_xlabel("")

    axin.xaxis.set_tick_params(pad=10)

    axin.set_ylabel(lower_ylabel)

    if ylim:
        axin.set_ylim(ylim)

    data_mc_yerr_ratio = mc_yerr / mc_hist

    add_hatched_error(
        axin,
        bin_edges,
        ys=np.ones(len(data_mc_yerr_ratio)),
        yerrs=data_mc_yerr_ratio,
    )

    return ax, axin


class HEPPlot:
    def __init__(self, data, mc, mc_err, bin_edges):
        self._check_valid(data, mc, mc_err, bin_edges)

        self.data = data
        # convert lists to np.array
        self.mc, self.mc_err = np.array(mc), np.array(mc_err)

        self.bin_edges = bin_edges
        self.dx = 0.5 * np.diff(bin_edges)

        self.data_err = np.sqrt(data)

        self.fig, self.ax, self.axin = None, None, None

    @staticmethod
    def _check_valid(data, mc, mc_err, bin_edges):
        assert isinstance(data, np.ndarray), "Data must be a numpy array."
        assert isinstance(mc, list), "MC must be a list of numpy arrays."
        assert isinstance(mc_err, list), "MC errors must be a list of numpy arrays."
        assert isinstance(bin_edges, np.ndarray), "Bin edges must be a numpy array."

    def setup_figure(self, **kwargs):
        self.fig, self.ax = plt.subplots(**kwargs)
        return self

    def plot_mc(self, add_error=True, **kwargs):
        stacked_hist_plot(self.ax, self.mc, self.bin_edges, **kwargs)

        if add_error:
            add_hatched_error(
                self.ax,
                self.bin_edges,
                ys=np.sum(self.mc, axis=0),
                yerrs=np.sqrt(np.sum(self.mc_err**2, axis=0)),
            )

        return self

    def plot_data(self, **kwargs):
        errorbar_plot(
            self.ax,
            x=self.bin_edges[:-1] + self.dx,
            y=self.data,
            xerr=self.dx,
            yerr=self.data_err,
            **kwargs,
        )
        return self

    def plot_ratio(self, **kwargs):
        self.ax, self.axin = add_data_mc_ratio(
            self.ax,
            bin_edges=self.bin_edges,
            data_hist=self.data,
            data_yerr=self.data_err,
            mc_hists=self.mc,
            mc_yerrs=self.mc_err,
            **kwargs,
        )
        return self

    def save(self, save_dir, file_name, extension=".pdf"):
        logging.info(f"Saving plot {file_name} to {save_dir}.")

        self.ax.legend()
        self.fig.tight_layout()
        plt.savefig(f"{save_dir}/{file_name}{extension}")
        plt.close(self.fig)

        return self


def handle_plot_exception(func):
    """Handle exceptions in plotting functions."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Plotting of {func.__name__} failed with exception: {e}")
            return None

    return wrapper


def add_atlas_label(loc=0, llabel="Internal", rlabel=""):
    """Setup the ATLAS style with ATLAS mark on the plot."""
    hep.atlas.label(loc=loc, llabel=llabel, rlabel=rlabel)


def set_size(small_size=24, medium_size=24, bigger_size=24):
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title


def style_setup(seaborn_pallete="deep", use_mplhep=True):
    if use_mplhep:
        plt.style.use([hep.style.ATLAS])

    sns.set_palette(seaborn_pallete)


def make_subplots_grid(n_plots, ratio=1 / 1.618):
    """Make a grid of subplots with a given ratio."""
    n_cols = int(np.sqrt(n_plots / ratio))
    n_rows = int(n_plots / n_cols)
    if n_cols * n_rows < n_plots:
        n_rows += 1
    return n_rows, n_cols


def iqr_remove_outliers(data, q1_set=25, q3_set=75):
    """https://en.wikipedia.org/wiki/Interquartile_range"""
    q1 = np.percentile(data, q1_set)
    q3 = np.percentile(data, q3_set)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]


def filter_array(arr):
    invalid_mask = np.logical_or(np.isnan(arr), np.isinf(arr))
    return arr[~invalid_mask]


def set_df_print(max_rows=None, max_cols=None, width=None, max_colwidth=None):
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_cols)
    pd.set_option("display.width", width)
    pd.set_option("max_colwidth", max_colwidth)


if __name__ == "__main__":
    # test example

    np.random.seed(0)

    A = np.random.normal(size=3000, loc=0.0, scale=1.0)
    B = np.random.normal(size=1000, loc=0.0, scale=1.0)
    C = np.random.normal(size=1000, loc=0.0, scale=1.0)
    D = np.random.normal(size=1000, loc=0.0, scale=1.0)

    A_hist, bin_edges = np.histogram(A, bins=30, range=(-3, 3))
    B_hist, _ = np.histogram(B, bins=bin_edges, range=(-3, 3))
    C_hist, _ = np.histogram(C, bins=bin_edges, range=(-3, 3))
    D_hist, _ = np.histogram(D, bins=bin_edges, range=(-3, 3))

    hep_plot = HEPPlot(
        data=A_hist,
        mc=[B_hist, C_hist, D_hist],
        mc_err=[np.sqrt(B_hist), np.sqrt(C_hist), np.sqrt(D_hist)],  # just for testing
        bin_edges=bin_edges,
    )
    hep_plot.setup_figure(figsize=(8, 8))

    add_atlas_label()

    hep_plot.ax.set_ylabel("$N$")
    hep_plot.ax.set_xlim((-3, 3))

    hep_plot.plot_data(label="Data", color="black")
    hep_plot.plot_mc(labels=["MC B", "MC C", "MC D"], colors=["C0", "C1", "C2"])
    hep_plot.plot_ratio()

    hep_plot.save(".", "test")
