from ml_hep_sim.normalizing_flows.glow import GlowFlowModel
from ml_hep_sim.normalizing_flows.made_mog import MADEMOGFlowModel
from ml_hep_sim.normalizing_flows.maf import MAFMADEFlowModel, MAFMADEMOGFlowModel
from ml_hep_sim.normalizing_flows.nice import NICEFlowModel
from ml_hep_sim.normalizing_flows.polynomial_splines import PolynomialSplineFlowModel
from ml_hep_sim.normalizing_flows.real_nvp import RealNVPFlowModel
from ml_hep_sim.normalizing_flows.rq_splines import RqSplineFlowModel
from ml_hep_sim.nets.classifiers import BinaryLabelClassifier, MultiLabelClassifier

from ml_hep_sim.vaes.beta_vae import BetaVAEModel
from ml_hep_sim.vaes.sigma_vae import SigmaVAEModel
from ml_hep_sim.vaes.two_stage_vae import StageTwoBetaVAEModel
from ml_hep_sim.vaes.vae import VAEModel


flow_models = {
    "Glow": GlowFlowModel,
    "MADEMOG": MADEMOGFlowModel,
    "MAFMADE": MAFMADEFlowModel,
    "MAFMADEMOG": MAFMADEMOGFlowModel,
    "NICE": NICEFlowModel,
    "PolynomialSpline": PolynomialSplineFlowModel,
    "RealNVP": RealNVPFlowModel,
    "RqSpline": RqSplineFlowModel,
}

vae_models = {
    "BetaVAE": BetaVAEModel,
    "SigmaVAE": SigmaVAEModel,
    "TwoStage": StageTwoBetaVAEModel,
    "VAE": VAEModel,
}

other_models = {
    "BinaryClassifier": BinaryLabelClassifier,
    "MultiClassifier": MultiLabelClassifier,
}
