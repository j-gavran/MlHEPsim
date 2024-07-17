import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.HIGGS.analysis.utils import MODEL_MAP

if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    save_dir = "ml/custom/HIGGS/analysis/plots/c2st_train"
    mkdir(save_dir)

    # needs to be updated by hand
    # reads csv files from mlruns
    model_run_id = {
        "RealNVP_flow_model_gauss_rank": "mlruns/837377447645369355/a42fb896bea2409dab12b4c7b25cf377/",
        "Glow_flow_model_gauss_rank": "mlruns/837377447645369355/2840789b251f48c4bd42f4f1ef44bf6e/",
        "rqsplines_flow_model_gauss_rank": "mlruns/837377447645369355/2e940409ec974ed0b6c59c0f0ae72b38/",
        "MAF_flow_model_gauss_rank": "mlruns/837377447645369355/dc6a1b4182514237886d495afae2dbe7/",
        "MAFMADEMOG_flow_model_gauss_rank": "mlruns/837377447645369355/5a164db633a349919bd77675b77806aa/",
        "MADEMOG_flow_model_gauss_rank_mini": "mlruns/837377447645369355/86e2a0f6f23747d39cb21fed3aad076e/",
        "ref": "mlruns/837377447645369355/51930b9d72cd45c39ad2faa51f473b9a/",
    }

    model_run_id = {
        "MADEMOG_flow_model_gauss_rank3": "mlruns/837377447645369355/7b3b29a56139462aaeb72cd64867a1ad/",
        "MADEMOG_flow_model_gauss_rank4": "mlruns/837377447645369355/048dee57262c4a53a52688d3fefdc166/",
        "MADEMOG_flow_model_gauss_rank5": "mlruns/837377447645369355/ecb9f8408a1147b096329c67d6c5949e/",
        "MADEMOG_flow_model_gauss_rank6": "mlruns/837377447645369355/f14ff2f2e14f4b8b99db0e86d159456f/",
        "MADEMOG_flow_model_gauss_rank7": "mlruns/837377447645369355/997860b6c69e4ee991c9b881e763196a/",
    }

    metrics = ["val_accuracy", "val_loss"]

    for metric in metrics:
        for name, run_id in model_run_id.items():
            path = f"{run_id}/metrics/"
            val_path = f"{path}{metric}"
            df = pd.read_csv(val_path, delimiter=" ", names=["timestamp", "val", "step"])

            val = df["val"].to_numpy()

            x = np.arange(len(val))

            plt.plot(x, val, label=MODEL_MAP[name], lw=2)
            plt.scatter(x, val, s=17)

        plt.legend(fontsize=12)
        plt.xlabel("Epoch")

        if metric == "val_accuracy":
            plt.ylabel("C2ST accuracy")
        else:
            plt.ylabel("C2ST loss")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/c2st_{metric}.pdf")
        logging.info(f"[green]Saved plot to {save_dir}/c2st_{metric}.pdf[/green]")
        plt.close()
