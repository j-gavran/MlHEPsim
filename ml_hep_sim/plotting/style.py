import matplotlib
import matplotlib.pyplot as plt
import mplhep
from ml_hep_sim.plotting.matplotlib_setup import configure_latex
from palettable.colorbrewer.qualitative import Dark2_8, Set1_9
import seaborn as sns

palettes = {"Set1": Set1_9.mpl_colors, "Dark2": Dark2_8.mpl_colors}


def set_size(s=24):
    SMALL_SIZE = s
    MEDIUM_SIZE = s
    BIGGER_SIZE = s

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def style_setup(color="Dark2", seaborn_pallete=False):
    plt.style.use([mplhep.style.ATLAS, mplhep.style.firamath])
    if seaborn_pallete:
        sns.set_palette("deep")
    else:
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=palettes[color])
    configure_latex()

    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
