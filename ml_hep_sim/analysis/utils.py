import logging

from ml_hep_sim.data_utils.higgs.process_higgs_dataset import COLNAMES
from ml_hep_sim.pipeline.pipes import Block


def get_colnames_dict(logger=None):
    """Make variable name to index mapping dict (for Higgs dataset)."""
    colnames_mapping_dct = {}

    for i, name in enumerate(COLNAMES[1:]):
        colnames_mapping_dct[name] = i

    msg = "available variables: {}".format(colnames_mapping_dct)
    if logger is None:
        logging.warning(msg)
    else:
        logger.debug(msg)

    return colnames_mapping_dct


class SigBkgBlock(Block):
    def __init__(self, sig_gen_block, bkg_gen_block, sig_mc_block, bkg_mc_block, use_results=False, **kwargs):
        """Aggregate signal (gen and MC) and background (gen and MC) blocks.

        Parameters
        ----------
        sig_gen_block : Block
            VariableExtractBlock or similar.
        bkg_gen_block : Block
            VariableExtractBlock or similar.
        sig_mc_block : Block
            VariableExtractBlock or similar.
        bkg_mc_block : Block
            VariableExtractBlock or similar.
        **kwargs : dict
            Additional arguments that are the same as input variables (used so the tree plot looks correct).
        """
        super().__init__()
        self.sig_gen_block = sig_gen_block
        self.bkg_gen_block = bkg_gen_block
        self.sig_mc_block = sig_mc_block
        self.bkg_mc_block = bkg_mc_block

        self.use_results = use_results

        self.sig_generated_data = None
        self.bkg_generated_data = None
        self.sig_reference_data = None
        self.bkg_reference_data = None

    def run(self):
        if self.use_results:
            self.sig_generated_data = self.sig_gen_block.results
            self.bkg_generated_data = self.bkg_gen_block.results
            self.sig_reference_data = self.sig_mc_block.results
            self.bkg_reference_data = self.bkg_mc_block.results
        else:
            self.sig_generated_data = self.sig_gen_block.generated_data
            self.bkg_generated_data = self.bkg_gen_block.generated_data
            self.sig_reference_data = self.sig_mc_block.reference_data
            self.bkg_reference_data = self.bkg_mc_block.reference_data

        return self

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["sig_gen_block"] = None
        attributes["bkg_gen_block"] = None
        attributes["sig_mc_block"] = None
        attributes["sig_mc_block"] = None

        attributes["sig_generated_data"] = None
        attributes["sig_reference_data"] = None
        attributes["bkg_generated_data"] = None
        attributes["bkg_reference_data"] = None
