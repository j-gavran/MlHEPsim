from ml_hep_sim.analysis.generator_pipeline import get_generator_pipeline
from ml_hep_sim.pipeline.blocks import CutBlock, CutByIndexBlock, RedoRescaleDataBlock
from ml_hep_sim.pipeline.pipes import Pipeline


def get_cut_pipeline(cut_value=0.5, use_classifier=True, N_gen=10**6, logger=None):
    # get generator pipeline with classification
    class_pipeline = get_generator_pipeline(use_classifier=use_classifier, N_gen=N_gen, logger=logger)

    # use cut on classifier
    b_sig_gen_class = class_pipeline.pipes[-4]
    b_bkg_gen_class = class_pipeline.pipes[-3]

    b_sig_mc_class = class_pipeline.pipes[-2]
    b_bkg_mc_class = class_pipeline.pipes[-1]

    b_sig_gen_class_cut = CutBlock(cut_value)(b_sig_gen_class)
    b_bkg_gen_class_cut = CutBlock(cut_value)(b_bkg_gen_class)

    b_sig_mc_class_cut = CutBlock(cut_value)(b_sig_mc_class)
    b_bkg_mc_class_cut = CutBlock(cut_value)(b_bkg_mc_class)

    # cut all events
    b_sig_gen_data = class_pipeline.pipes[-8]
    b_bkg_gen_data = class_pipeline.pipes[-5]

    b_sig_mc_data = class_pipeline.pipes[1]
    b_bkg_mc_data = class_pipeline.pipes[3]

    b_sig_gen_data_cut = CutByIndexBlock()(b_sig_gen_class_cut, b_sig_gen_data)
    b_bkg_gen_data_cut = CutByIndexBlock()(b_bkg_gen_class_cut, b_bkg_gen_data)

    b_sig_mc_data_cut = CutByIndexBlock()(b_sig_mc_class_cut, b_sig_mc_data)
    b_bkg_mc_data_cut = CutByIndexBlock()(b_bkg_mc_class_cut, b_bkg_mc_data)

    # rescale back to original
    b_sig_gen_data_cut_rescale = RedoRescaleDataBlock(scaler_idx=0)(class_pipeline.pipes[7], b_sig_gen_data_cut)
    b_bkg_gen_data_cut_rescale = RedoRescaleDataBlock(scaler_idx=0)(class_pipeline.pipes[10], b_bkg_gen_data_cut)

    b_sig_mc_data_cut_rescale = RedoRescaleDataBlock(scaler_idx=-1)(class_pipeline.pipes[1], b_sig_mc_data_cut)
    b_bkg_mc_data_cut_rescale = RedoRescaleDataBlock(scaler_idx=-1)(class_pipeline.pipes[3], b_bkg_mc_data_cut)

    # do fit
    pipe = Pipeline(logger=logger)
    pipe.compose(
        class_pipeline,
        b_sig_gen_class_cut,
        b_bkg_gen_class_cut,
        b_sig_mc_class_cut,
        b_bkg_mc_class_cut,
        b_sig_gen_data_cut,
        b_bkg_gen_data_cut,
        b_sig_mc_data_cut,
        b_bkg_mc_data_cut,
        b_sig_gen_data_cut_rescale,
        b_bkg_gen_data_cut_rescale,
        b_sig_mc_data_cut_rescale,
        b_bkg_mc_data_cut_rescale,
    )

    return pipe


if __name__ == "__main__":
    pipe = get_cut_pipeline()
    pipe.fit()
