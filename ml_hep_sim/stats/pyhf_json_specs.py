import pyhf
import numpy as np


def prep_data(sig, bkg, bkg_unc):
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "signal",
                        "data": list(sig),
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            },
                        ],
                    },
                    {
                        "name": "background",
                        "data": list(bkg),
                        "modifiers": [
                            {
                                "name": "mc_staterror",
                                "type": "staterror",
                                "data": list((bkg * bkg_unc).astype(int)),
                            },
                        ],
                    },
                ],
            },
        ]
    }

    return spec


if __name__ == "__main__":
    spec = prep_data(np.array([5, 7]), np.array([10, 12]), 0.1)

    pyhf.Model(spec)
