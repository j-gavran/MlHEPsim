import matplotlib.pyplot as plt
import mplhep
import seaborn as sns


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
        plt.style.use([mplhep.style.ATLAS])

    sns.set_palette(seaborn_pallete)
