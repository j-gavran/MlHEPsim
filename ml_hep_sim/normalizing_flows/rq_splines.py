""" Based on the implementation of nflows: normalizing flows in PyTorch, https://github.com/bayesiains/nflows, 2020;
    and on the Durkan et al., Neural Spline Flows, https://arxiv.org/abs/1906.04032, 2019."""


import hydra
import numpy as np
import torch
from torch.nn import functional as F

from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.normalizing_flows.base_flows import NormalizingFlow, PlFlowModel
from ml_hep_sim.normalizing_flows.flow_utils import plot_flow_result_on_event
from ml_hep_sim.normalizing_flows.flows import BatchNormFlow, Conv1x1PLU, ReverseFlow
from ml_hep_sim.normalizing_flows.polynomial_splines import PolynomialSplineFlow

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    tail_bound=None,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


class RqSplineFlow(PolynomialSplineFlow):
    def __init__(
        self,
        input_dim,
        K,
        tail_bound=None,
        hidden_layers=None,
        activation="ReLU",
        use_unet=False,
        use_resnet=False,
        ar=False,
    ):
        super().__init__(input_dim, K, tail_bound, hidden_layers, activation, use_unet, use_resnet, ar)
        self.spline_func = unconstrained_rational_quadratic_spline

        if self.ar:
            self.net = self._build_network(output_activation="Identity", out_mask_multiplier=3 * self.K - 1)
        else:
            self.net = self._build_network(output_activation="Identity", net_output_dim=self.d * 3 * self.K - self.d)

    def _get_predict(self, x_A):
        net_out = self.net(x_A)

        if self.ar:
            transform_params = net_out.view(x_A.shape[0], self.input_dim, 3 * self.K - 1)

            unnormalized_widths = transform_params[..., : self.K]
            unnormalized_heights = transform_params[..., self.K : 2 * self.K]
            unnormalized_derivatives = transform_params[..., 2 * self.K :]
        else:
            unnormalized_widths, unnormalized_heights, unnormalized_derivatives = net_out.split(
                [self.d * self.K, self.d * self.K, self.d * self.K - self.d], dim=-1
            )

            unnormalized_widths = unnormalized_widths.reshape(x_A.shape[0], self.d, self.K)
            unnormalized_heights = unnormalized_heights.reshape(x_A.shape[0], self.d, self.K)
            unnormalized_derivatives = unnormalized_derivatives.reshape(x_A.shape[0], self.d, self.K - 1)

        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def forward(self, x):
        if self.ar:
            unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_predict(x)

            outputs, logabsdet = self.spline_func(
                x,
                unnormalized_widths,
                unnormalized_heights,
                unnormalized_derivatives,
                tail_bound=self.tail_bound,
                inverse=False,
            )

            return outputs, logabsdet.sum(dim=-1, keepdim=True)
        else:
            x_A, x_B = x[:, : self.d], x[:, self.d :]

            unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_predict(x_A)

            outputs, logabsdet = self.spline_func(
                x_B,
                unnormalized_widths,
                unnormalized_heights,
                unnormalized_derivatives,
                tail_bound=self.tail_bound,
                inverse=False,
            )

            cat_outputs = torch.cat((x_A, outputs), dim=1)
            return cat_outputs, logabsdet.sum(dim=-1, keepdim=True)

    def inverse(self, x):
        if self.ar:
            num_inputs = np.prod(x.shape[1:])
            outputs = torch.zeros_like(x)

            logabsdet = None
            for _ in range(num_inputs):
                unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_predict(outputs)

                outputs, logabsdet = self.spline_func(
                    x,
                    unnormalized_widths,
                    unnormalized_heights,
                    unnormalized_derivatives,
                    tail_bound=self.tail_bound,
                    inverse=True,
                )

            return outputs, logabsdet.sum(dim=-1, keepdim=True)
        else:
            x_A, x_B = x[:, : self.d], x[:, self.d :]

            unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_predict(x_A)

            outputs, logabsdet = self.spline_func(
                x_B,
                unnormalized_widths,
                unnormalized_heights,
                unnormalized_derivatives,
                tail_bound=self.tail_bound,
                inverse=True,
            )

            cat_outputs = torch.cat((x_A, outputs), dim=1)
            return cat_outputs, logabsdet.sum(dim=-1, keepdim=True)


class RqSplineFlowModel(PlFlowModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = config["learning_rate"]
        self.lr_scheduler_dct = config.get("lr_scheduler_dct")
        self.activation = config["activation"]
        self.num_flows = config["num_flows"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.hidden_layer_dim = config["hidden_layer_dim"]
        self.bathcnorm_flow = config["batchnorm_flow"]
        self.conv1x1 = config["conv1x1"]

        self.u_net = config.get("u_net")
        self.resnet = config.get("resnet")
        self.tail_bound = config["tail_bound"]
        self.bins = config["bins"]

        self.ar = config.get("ar")

        self.save_hyperparameters()

        self.hidden_layer = []
        for _ in range(self.num_hidden_layers):
            self.hidden_layer.append(self.hidden_layer_dim)
        self.hidden_layer.append(self.input_dim)

        model = RqSplineFlow

        blocks = []
        for _ in range(self.num_flows):
            if self.bathcnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

            blocks.append(
                model(
                    self.input_dim,
                    self.bins,
                    tail_bound=self.tail_bound,
                    hidden_layers=self.hidden_layer,
                    activation=self.activation,
                    use_unet=self.u_net,
                    use_resnet=self.resnet,
                    ar=self.ar,
                )
            )

        if self.bathcnorm_flow:
            blocks.append(BatchNormFlow(self.input_dim))

        if self.conv1x1:
            blocks.append(Conv1x1PLU(self.input_dim, device=self.base_distribution.loc.device))
        else:
            blocks.append(ReverseFlow(self.input_dim))

        self.flow = NormalizingFlow(
            self.input_dim,
            blocks,
            self.base_distribution,
        )

    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            plot_flow_result_on_event(
                self.data_name,
                self.base_distribution,
                self,
                self.logger,
                self._trainer.datamodule.test,
                idx=self.current_epoch,
            )


@hydra.main(config_path="../conf", config_name="rq_splines_config", version_base=None)
def train_rq_splines(config):
    device = "cuda" if config["trainer_config"]["gpus"] else "cpu"
    input_dim = config["datasets"]["input_dim"]

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        device=device,
        pl_model=RqSplineFlowModel,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_rq_splines()
