import numpy as np
import pyhf


def prep_data(sig, bkg, bkg_unc=None, mc_err=None):
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
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": list(bkg * bkg_unc),
                            },
                            {
                                "name": "mc_staterror",
                                "type": "staterror",
                                "data": list(mc_err),
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
