import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    """Distribution graphs (histogram/bar graph) of column data

    Note
    ----
    Not in use (here for completeness).

    """
    nunique = df.nunique()
    df = df[
        [col for col in df if nunique[col] > 1 and nunique[col] < 50]
    ]  # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor="w", edgecolor="k")
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel("counts")
        plt.xticks(rotation=90)
        plt.title(f"{columnNames[i]} (column {i})")
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth, labels=None):
    """Correlation matrix plot.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    graphWidth : float
        Plot width.

    """
    df = df.dropna("columns")  # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f"No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2")
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor="w", edgecolor="k")
    corrMat = plt.matshow(corr, fignum=1)

    if labels is None:
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
    else:
        plt.xticks(range(len(corr.columns)), labels, rotation=90)
        plt.yticks(range(len(corr.columns)), labels)

    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.show()


def plotScatterMatrix(df, plotSize, textSize):
    """Scatter and density plots

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    plotSize : float
        Plot size: plotSize x plotSize.
    textSize : float
        Text size.
    """
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna("columns")
    # df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    columnNames = list(df)
    # if len(columnNames) > 10:  # reduce the number of columns for matrix inversion of kernel density plots
    #     columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.65, figsize=[plotSize, plotSize], diagonal="kde", s=2)
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate(
            "Corr = %.3f" % corrs[i, j],
            (0.5, 0.8),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=textSize,
        )
    # plt.suptitle("Scatter and Density Plot")
    plt.show()
