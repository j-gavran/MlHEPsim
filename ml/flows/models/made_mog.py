import numpy as np
import torch
import torch.distributions as D

from ml.flows.models.base_flows import (
    AutoregressiveNormalizingFlow,
    BaseFlowModel,
    FlowModel,
)
from ml.flows.models.made import GaussianMADE, GaussianResMADE, MaskedLinear
from ml.flows.models.mogs import MixtureNet


class MADEMOGModel(GaussianMADE):
    def __init__(self, *args, n_mixtures=1, **kwargs):
        """
        References
        ----------
        [1] - https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

        """
        super().__init__(*args, **kwargs)
        self.normalizing_direction = False

        self.register_buffer("base_distribution_mu", torch.zeros(self.input_size))
        self.register_buffer("base_distribution_var", torch.ones(self.input_size))

        self.n_mixtures = n_mixtures
        self.log_prob = None

        self.net[-1] = MaskedLinear(
            self.hidden_size, 3 * self.input_size * n_mixtures, self.masks[-1].repeat(3 * n_mixtures, 1)
        )
        self.net = MixtureNet(self.input_size, n_mixtures, self.net)

    @property
    def base_distribution(self):
        return D.Normal(self.base_distribution_mu, self.base_distribution_var, validate_args=False)

    def forward(self, x, log_prob=True):
        mu, log_std, log_weights = self.net(x)

        N, C, L = mu.shape
        x = x.repeat(1, C).view(N, C, L)

        u = (x - mu) * torch.exp(-log_std)
        log_abs_det_jacobian = -log_std

        if log_prob:
            self.log_prob = torch.sum(
                torch.logsumexp(log_weights + self.base_distribution.log_prob(u) + log_abs_det_jacobian, dim=1),
                dim=-1,
                keepdim=True,
            )  # N x C x L -> N x L -> N x 1

        return u.view(u.shape[0], C * L), log_abs_det_jacobian

    def inverse(self, u):
        N, L = u.shape
        x = torch.zeros(N, L, device=u.device)

        for i in range(self.input_size):
            mu, log_std, log_weights = self.net(x)  # N x C x L

            mu_x = mu[:, :, i].unsqueeze(-1)  # N x C x 1
            std_x = torch.exp(log_std[:, :, i].unsqueeze(-1))  # N x C x 1
            log_weights_x = log_weights[:, :, i].unsqueeze(-1)  # N x C x 1

            x[:, i] = self.net.sample(log_weights_x, std_x, mu_x)

        log_abs_det_jacobian = log_std
        return x, log_abs_det_jacobian


class ResMADEMOGModel(MADEMOGModel, GaussianResMADE):
    def __init__(self, *args, n_mixtures=1, **kwargs):
        super().__init__(*args, n_mixtures=n_mixtures, **kwargs)


class MADEMOG(BaseFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = self.model_conf["activation"]
        self.num_hidden_layers_mog_net = self.model_conf["num_hidden_layers_mog_net"]
        self.hidden_layer_mog_dim = self.model_conf["hidden_layer_mog_dim"]
        self.n_mixtures = self.model_conf["n_mixtures"]
        self.mog_residuals = self.model_conf["mog_residuals"]
        self.res_layers_in_mog_block = self.model_conf["res_layers_in_mog_block"]
        self.res_batchnorm = self.model_conf.get("res_batchnorm", False)
        self.act_first = self.model_conf.get("act_first", False)

        self.pop_idx = -1

        if self.mog_residuals:
            self.blocks = [
                ResMADEMOGModel(
                    self.input_dim,
                    k=self.hidden_layer_mog_dim,
                    n_blocks=self.num_hidden_layers_mog_net,
                    l=self.res_layers_in_mog_block,
                    activation=self.activation,
                    n_mixtures=self.n_mixtures,
                    res_batchnorm=self.res_batchnorm,
                    act_first=self.act_first,
                )
            ]
        else:
            self.blocks = [
                MADEMOGModel(
                    self.input_dim,
                    self.hidden_layer_mog_dim,
                    self.num_hidden_layers_mog_net,
                    activation=self.activation,
                    n_mixtures=self.n_mixtures,
                )
            ]

        self.model = AutoregressiveNormalizingFlow(self.input_dim, self.blocks, density=None, device=self.device)

    def estimate_density(self, data_points, exp=True, mean=True):
        if self.model.training:
            raise ValueError("Model must be in eval mode!")

        if not torch.is_tensor(data_points):
            data_points = torch.from_numpy(data_points.astype(np.float32)).to(self.device)

        with torch.no_grad():
            _, log_jac = self.forward(data_points)
            log_jac.pop(self.pop_idx)
            mog_nll = self.model.bijectors[self.pop_idx].log_prob
            sum_of_log_det_jacobian = sum(log_jac)

        if mean:
            log_density = -torch.mean(mog_nll + sum_of_log_det_jacobian)
        else:
            log_density = -(mog_nll + sum_of_log_det_jacobian)

        if exp:
            return torch.exp(log_density).cpu().numpy()
        else:
            return log_density.cpu().numpy()


class MOGFlowModel(FlowModel):
    def __init__(self, model_conf, training_conf, data_conf, model=None, loss_func=None, tracker=None):
        super().__init__(model_conf, training_conf, data_conf, model, loss_func, tracker)
        self.data_conf = data_conf
        self.use_loss_weighting = training_conf.get('use_loss_weighting', False)
        self.boundary_margin = training_conf.get('boundary_margin', 5.0)  # GeV margin to downweight
    
    def _compute_mass_weights(self, x):
        """Compute sample weights based on distance from mass boundaries.
        
        Weighting scheme (for sidebands with mass cut at 122-128 GeV):
        - Outside valid mass ranges: weight = 0
        - Inside signal region (122-128 GeV): weight = 0
        - Lower sideband: 
          * First 10 GeV: linear increase from 0 to 1
          * Remaining region to upper boundary: weight = 1.0 (full weight)
        - Upper sideband: weight = 1.0 throughout (128-180 GeV)
        """
        if not self.use_loss_weighting:
            return torch.ones(x.shape[0], 1, device=x.device)
        
        # Extract mass configuration
        input_proc = self.data_conf['input_processing']
        mass_region = input_proc['mass_region']
        plateau_margin = 10.0  # GeV plateau before/after mass cut
        
        # Compute invariant mass from muon kinematics (assumes specific feature order)
        # Features: Pos_Eta, Neg_Eta, Pos_PT, Neg_PT, Pos_Phi, Neg_Phi
        eta_pos, eta_neg = x[:, 0], x[:, 1]
        pt_pos, pt_neg = x[:, 2], x[:, 3]
        phi_pos, phi_neg = x[:, 4], x[:, 5]
        
        # Calculate invariant mass (simplified, assumes massless muons)
        delta_phi = phi_pos - phi_neg
        delta_phi = torch.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
        delta_phi = torch.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
        
        cosh_deta = torch.cosh(eta_pos - eta_neg)
        cos_dphi = torch.cos(delta_phi)
        mass = torch.sqrt(2 * pt_pos * pt_neg * (cosh_deta - cos_dphi))
        
        # Initialize weights to zero (outside regions)
        weights = torch.zeros_like(mass)
        
        if mass_region == 'full_data':
            min_mass = input_proc['min_mass']
            max_mass = input_proc['max_mass']
            margin = self.boundary_margin
            
            # Only weight points inside the valid range
            in_range = (mass >= min_mass) & (mass <= max_mass)
            
            # Distance from outer boundaries (normalized to [0, 1])
            dist_from_lower = (mass - min_mass) / margin
            dist_from_upper = (max_mass - mass) / margin
            
            # Weight increases from 0 at boundary to 1 at margin distance inward
            # Use minimum of both distances so weight is 0 near BOTH boundaries
            weight_from_lower = torch.clamp(dist_from_lower, 0, 1)
            weight_from_upper = torch.clamp(dist_from_upper, 0, 1)
            range_weight = torch.min(weight_from_lower, weight_from_upper)
            
            weights = torch.where(in_range, range_weight, weights)
            
        elif mass_region == 'sidebands':
            lower_min = input_proc['sideband_lower_min']
            lower_max = input_proc['sideband_lower_max']
            upper_min = input_proc['sideband_upper_min']
            upper_max = input_proc['sideband_upper_max']
            
            # Define ramp-up region for lower sideband
            # Linear increase in the first 10 GeV of the lower sideband
            ramp_up_distance = 10.0  # GeV
            ramp_up_end = lower_min + ramp_up_distance
            
            # Lower sideband with ramp in first 10 GeV
            in_lower = (mass >= lower_min) & (mass <= lower_max)
            if in_lower.any():
                # Ramp up region: first 10 GeV (increases from 0 to 1)
                in_ramp_up = (mass >= lower_min) & (mass < ramp_up_end)
                ramp_up_weight = (mass - lower_min) / ramp_up_distance
                
                # Full weight region: after first 10 GeV to upper boundary (weight = 1)
                in_full_weight = (mass >= ramp_up_end) & (mass <= lower_max)
                
                lower_weight = torch.where(in_ramp_up, torch.clamp(ramp_up_weight, 0, 1), 
                                          torch.where(in_full_weight, torch.ones_like(mass), 
                                                     torch.zeros_like(mass)))
                
                weights = torch.where(in_lower, lower_weight, weights)
            
            # Upper sideband (130-180 GeV)
            in_upper = (mass >= upper_min) & (mass <= upper_max)
            if in_upper.any():
                # Plateau region: 130-140 GeV (weight = 1)
                in_plateau = (mass >= upper_min) & (mass <= upper_plateau_end)
                
                # Ramp down region: 140-180 GeV (decreases from 1 to 0)
                in_ramp_down = (mass > upper_plateau_end) & (mass <= upper_max)
                ramp_down_weight = (upper_max - mass) / (upper_max - upper_plateau_end)
                
                upper_weight = torch.where(in_plateau, torch.ones_like(mass),
                                          torch.where(in_ramp_down, torch.clamp(ramp_down_weight, 0, 1),
                                                     torch.zeros_like(mass)))
                
                weights = torch.where(in_upper, upper_weight, weights)
        
        # Add small epsilon to avoid completely zero weights (can cause numerical issues)
        weights = torch.clamp(weights, min=1e-6)
        
        return weights.unsqueeze(1)  # (N, 1)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        _, log_jac = self.model(x)  # _: (N, L), log_jac: list of [(N, 1),...,(N, C, L)]

        log_jac.pop(self.model.pop_idx)  # accounted for in logsumexp

        mog_nll = self.model.model.bijectors[self.model.pop_idx].log_prob  # (N, 1)

        sum_of_log_det_jacobian = sum(log_jac)  # (N, 1)
        
        # Compute sample losses
        sample_losses = -(sum_of_log_det_jacobian + mog_nll)  # (N, 1)
        
        # Compute sample weights based on mass
        weights = self._compute_mass_weights(x)  # (N, 1)  

        # Weighted loss: focus on core data regions
        weighted_losses = weights * sample_losses
        loss = torch.mean(weighted_losses)

        self.log("train_loss", loss)
        if self.use_loss_weighting:
            self.log("mean_weight", torch.mean(weights))
        self.current_step += 1

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        _, log_jac = self.model(x)

        log_jac.pop(self.model.pop_idx)

        mog_nll = self.model.model.bijectors[self.model.pop_idx].log_prob

        if len(log_jac) != 0:
            sum_of_log_det_jacobian = sum(log_jac)

            mask_nll, mask_jac = self._get_invalid_mask(mog_nll), self._get_invalid_mask(sum_of_log_det_jacobian)
            mask = torch.logical_and(mask_nll, mask_jac)

            sum_of_log_det_jacobian, mog_nll = sum_of_log_det_jacobian[mask], mog_nll[mask]

            loss = -torch.mean(sum_of_log_det_jacobian + mog_nll)

            self.log("val_loss", loss)
            self.log("sum_log_det_jac", torch.mean(sum_of_log_det_jacobian))
            self.log("val_nll", torch.mean(mog_nll))

            return {"val_loss": loss, "sum_log_det_jac": sum_of_log_det_jacobian, "val_nll": mog_nll}
        else:
            loss = -torch.mean(mog_nll)

            self.log("val_loss", loss)
            self.log("val_nll", torch.mean(mog_nll))

            return {"val_loss": loss, "val_nll": mog_nll}
