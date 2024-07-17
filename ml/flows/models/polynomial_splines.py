import logging

import numpy as np
import torch
from torch.nn import functional as F

from ml.common.nn.mlp import BasicMLP, BasicResMLP
from ml.common.nn.u_net import UNet
from ml.flows.models.aux_flows import BatchNormFlow, Conv1x1PLU, Flow, ReverseFlow
from ml.flows.models.base_flows import BaseFlowModel, NormalizingFlow
from ml.flows.models.made import MADE, ResMADE

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_linear_spline(inputs, unnormalized_pdf, inverse=False, tail_bound=1.0, tails="linear"):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = linear_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_pdf=unnormalized_pdf[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
        )

    return outputs, logabsdet


def linear_spline(inputs, unnormalized_pdf, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0, tail_bound=None):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.0
    cdf = F.pad(cdf, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        inv_bin_idx = searchsorted(cdf, inputs)

        bin_boundaries = (
            torch.linspace(0, 1, num_bins + 1, device=inputs.device)
            .view([1] * inputs.dim() + [-1])
            .expand(*inputs.shape, -1)
        )

        slopes = (cdf[..., 1:] - cdf[..., :-1]) / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1])
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1

        alpha = bin_pos - bin_idx.float()

        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = torch.log(input_pdfs) - np.log(bin_width)

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet


def unconstrained_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    tail_bound=1.0,
    tails="linear",
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    num_bins = unnormalized_widths.shape[-1]

    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
        assert unnormalized_heights.shape[-1] == num_bins - 1
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if tail_bound == 0.0:
        left = 0.0
        right = 1.0
        bottom = 0.0
        top = 1.0
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound

    if torch.any(inside_interval_mask):
        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            inverse=inverse,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
        )

    return outputs, logabsdet


def quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    tail_bound=None,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    unnorm_heights_exp = F.softplus(unnormalized_heights) + 1e-3

    if unnorm_heights_exp.shape[-1] == num_bins - 1:
        # Set boundary heights s.t. after normalization they are exactly 1.
        first_widths = 0.5 * widths[..., 0]
        last_widths = 0.5 * widths[..., -1]
        numerator = (
            0.5 * first_widths * unnorm_heights_exp[..., 0]
            + 0.5 * last_widths * unnorm_heights_exp[..., -1]
            + torch.sum(
                ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths[..., 1:-1],
                dim=-1,
            )
        )
        constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
        constant = constant[..., None]
        unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)

    unnormalized_area = torch.sum(
        ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths,
        dim=-1,
    )[..., None]
    heights = unnorm_heights_exp / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights

    bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
    bin_left_cdf[..., -1] = 1.0
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode="constant", value=0.0)

    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.0
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode="constant", value=0.0)

    if inverse:
        bin_idx = searchsorted(bin_left_cdf, inputs)[..., None]
        # added for debuging
        n_outside_bin_idx = len(bin_idx[bin_idx < 0])
        if n_outside_bin_idx > 0:
            logging.warning(f"{n_outside_bin_idx} invalid indices in bin idx, setting them to 0...")
            bin_idx[bin_idx < 0] = 0
    else:
        bin_idx = searchsorted(bin_locations, inputs)[..., None]

    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]

    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx + 1)[..., 0]

    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        alpha = (-b + torch.sqrt(b.pow(2) - 4 * a * c_)) / (2 * a)
        outputs = alpha * input_bin_widths + input_bin_locations
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = -torch.log((alpha * (input_right_heights - input_left_heights) + input_left_heights))
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha.pow(2) + b * alpha + c
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = torch.log((alpha * (input_right_heights - input_left_heights) + input_left_heights))

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet


class PolynomialSplineFlowModel(Flow):
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
        polynomial=None,
        repeats=2,
    ):
        super().__init__()
        self.normalizing_direction = False
        self.input_dim = input_dim
        self.d = self.input_dim // 2
        self.K = K
        self.eps = 1e-6

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.use_unet, self.use_resnet, self.ar = use_unet, use_resnet, ar
        self.repeats = repeats

        self.tail_bound = tail_bound
        if self.tail_bound:
            if polynomial == "linear":
                self.spline_func = unconstrained_linear_spline
            elif polynomial == "quadratic":
                self.spline_func = unconstrained_quadratic_spline
            else:
                self.spline_func = None
        else:
            if polynomial == "linear":
                self.spline_func = linear_spline
            elif polynomial == "quadratic":
                self.spline_func = quadratic_spline
            else:
                self.spline_func = None

    def _build_network(self, output_activation=None, net_output_dim=None, out_mask_multiplier=None, **kwargs):
        if self.ar and self.use_resnet:
            return ResMADE(
                self.input_dim,
                k=self.hidden_layers[0],
                l=self.repeats,
                n_blocks=len(self.hidden_layers) - 1,
                activation=self.activation,
                out_mask_multiplier=out_mask_multiplier,
                **kwargs,
            )

        elif self.ar and not self.use_resnet:
            return MADE(
                self.input_dim,
                self.hidden_layers[0],
                len(self.hidden_layers),
                activation=self.activation,
                out_mask_multiplier=out_mask_multiplier,
                **kwargs,
            )

        if self.use_unet:
            div = int(np.log2(self.hidden_layers[0]))
            return UNet(
                self.d,
                self.hidden_layers[0],
                div,
                activation=self.activation,
                output_activation=output_activation,
                output_dim=net_output_dim,
                **kwargs,
            )

        elif self.use_resnet:
            self.hidden_layers[-1] = net_output_dim
            logging.debug(f"use_resnet is True and has hidden layers: {[self.d] + self.hidden_layers}")
            return BasicResMLP(
                [self.d] + self.hidden_layers,
                act=self.activation,
                act_out=output_activation,
                batchnorm=True,
                repeats=self.repeats,
                **kwargs,
            )

        else:
            self.hidden_layers[-1] = net_output_dim
            logging.debug(f"Using BasicMLP with hidden layers: {[self.d] + self.hidden_layers}")
            return BasicMLP(
                [self.d] + self.hidden_layers,
                act=self.activation,
                act_out=output_activation,
                **kwargs,
            )

    def _get_predict(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def inverse(self):
        raise NotImplementedError


class LinearSplineFlowModel(PolynomialSplineFlowModel):
    def __init__(self, input_dim, K, **kwargs):
        super().__init__(input_dim, K, polynomial="linear", **kwargs)
        if self.ar:
            self.net = self._build_network(output_activation=None, out_mask_multiplier=self.K)
        else:
            self.net = self._build_network(output_activation=None, net_output_dim=self.d * self.K)

    def _get_predict(self, x_A):
        if self.ar:
            unnormalized_pdf = self.net(x_A).reshape(x_A.shape[0], self.input_dim, self.K)
        else:
            unnormalized_pdf = self.net(x_A).reshape(x_A.shape[0], self.d, self.K)

        return unnormalized_pdf

    def forward(self, x):
        if self.ar:
            unnormalized_pdf = self._get_predict(x)
            outputs, logabsdet = self.spline_func(
                x,
                unnormalized_pdf,
                inverse=False,
                tail_bound=self.tail_bound,
            )
            return outputs, logabsdet.sum(dim=-1, keepdim=True)

        else:
            if not self.tail_bound:
                x = torch.sigmoid(x)

            x_A, x_B = x[:, : self.d], x[:, self.d :]

            unnormalized_pdf = self._get_predict(x_A)

            outputs, logabsdet = self.spline_func(
                x_B,
                unnormalized_pdf,
                inverse=False,
                tail_bound=self.tail_bound,
            )

            cat_outputs = torch.cat((x_A, outputs), dim=1)

            if self.tail_bound:
                return cat_outputs, logabsdet.sum(dim=-1, keepdim=True)
            else:
                return torch.logit(torch.clamp(cat_outputs, self.eps, 1 - self.eps)), logabsdet.sum(
                    dim=-1, keepdim=True
                )

    def inverse(self, x):
        if self.ar:
            num_inputs = np.prod(x.shape[1:])
            outputs = torch.zeros_like(x)

            logabsdet = None
            for _ in range(num_inputs):
                unnormalized_pdf = self._get_predict(outputs)
                outputs, logabsdet = self.spline_func(
                    x,
                    unnormalized_pdf,
                    inverse=True,
                    tail_bound=self.tail_bound,
                )

            return outputs, logabsdet.sum(dim=-1, keepdim=True)
        else:
            if not self.tail_bound:
                x = torch.sigmoid(x)

            x_A, x_B = x[:, : self.d], x[:, self.d :]

            unnormalized_pdf = self._get_predict(x_A)

            outputs, logabsdet = self.spline_func(
                x_B,
                unnormalized_pdf,
                inverse=True,
                tail_bound=self.tail_bound,
            )

            cat_outputs = torch.cat((x_A, outputs), dim=1)

            if self.tail_bound:
                return cat_outputs, logabsdet.sum(dim=-1, keepdim=True)
            else:
                return torch.logit(torch.clamp(cat_outputs, self.eps, 1 - self.eps)), logabsdet.sum(
                    dim=-1, keepdim=True
                )


class QuadraticSplineFlowModel(PolynomialSplineFlowModel):
    def __init__(self, input_dim, K, **kwargs):
        super().__init__(input_dim, K, polynomial="quadratic", **kwargs)
        if self.ar:
            self.net = self._build_network(output_activation=None, out_mask_multiplier=2 * self.K - 1)
        else:
            self.net = self._build_network(output_activation=None, net_output_dim=self.d * 2 * self.K - self.d)

    def _get_predict(self, x_A):
        net_out = self.net(x_A)

        if self.ar:
            transform_params = net_out.view(x_A.shape[0], self.input_dim, 2 * self.K - 1)

            unnormalized_widths = transform_params[..., : self.K]
            unnormalized_heights = transform_params[..., self.K :]
        else:
            unnormalized_widths, unnormalized_heights = net_out.split(
                [self.d * self.K, self.d * self.K - self.d], dim=-1
            )

            unnormalized_widths = unnormalized_widths.reshape(x_A.shape[0], self.d, self.K)
            unnormalized_heights = unnormalized_heights.reshape(x_A.shape[0], self.d, self.K - 1)

        return unnormalized_widths, unnormalized_heights

    def forward(self, x):
        if self.ar:
            unnormalized_widths, unnormalized_heights = self._get_predict(x)
            outputs, logabsdet = self.spline_func(
                x,
                unnormalized_widths,
                unnormalized_heights,
                tail_bound=self.tail_bound,
                inverse=False,
            )

            return outputs, logabsdet.sum(dim=-1, keepdim=True)

        else:
            if not self.tail_bound:
                x = torch.sigmoid(x)

            x_A, x_B = x[:, : self.d], x[:, self.d :]

            unnormalized_widths, unnormalized_heights = self._get_predict(x_A)

            outputs, logabsdet = self.spline_func(
                x_B,
                unnormalized_widths,
                unnormalized_heights,
                tail_bound=self.tail_bound,
                inverse=False,
            )

            cat_outputs = torch.cat((x_A, outputs), dim=1)

            if self.tail_bound:
                return cat_outputs, logabsdet.sum(dim=-1, keepdim=True)
            else:
                return torch.logit(torch.clamp(cat_outputs, self.eps, 1 - self.eps)), logabsdet.sum(
                    dim=-1, keepdim=True
                )

    def inverse(self, x):
        if self.ar:
            num_inputs = np.prod(x.shape[1:])
            outputs = torch.zeros_like(x)

            logabsdet = None
            for _ in range(num_inputs):
                unnormalized_widths, unnormalized_heights = self._get_predict(outputs)

                outputs, logabsdet = self.spline_func(
                    x,
                    unnormalized_widths,
                    unnormalized_heights,
                    tail_bound=self.tail_bound,
                    inverse=True,
                )

            return outputs, logabsdet.sum(dim=-1, keepdim=True)

        else:
            if not self.tail_bound:
                x = torch.sigmoid(x)

            x_A, x_B = x[:, : self.d], x[:, self.d :]

            unnormalized_widths, unnormalized_heights = self._get_predict(x_A)

            outputs, logabsdet = self.spline_func(
                x_B,
                unnormalized_widths,
                unnormalized_heights,
                tail_bound=self.tail_bound,
                inverse=True,
            )

            cat_outputs = torch.cat((x_A, outputs), dim=1)

            if self.tail_bound:
                return cat_outputs, logabsdet.sum(dim=-1, keepdim=True)
            else:
                return torch.logit(torch.clamp(cat_outputs, self.eps, 1 - self.eps)), logabsdet.sum(
                    dim=-1, keepdim=True
                )


class PolynomialSplineFlow(BaseFlowModel):
    def __init__(self, *args, **kwargs):
        """Polynomial neural spline flow model (linear or quadratic).

        References
        ----------
        [1] - nflows: normalizing flows in PyTorch: https://github.com/bayesiains/nflows
        [2] - Neural Importance Sampling: https://arxiv.org/abs/1808.03856

        """
        super().__init__(*args, **kwargs)

        self.activation = self.model_conf["activation"]
        self.num_flows = self.model_conf["num_flows"]
        self.num_hidden_layers = self.model_conf["num_hidden_layers"]
        self.hidden_layer_dim = self.model_conf["hidden_layer_dim"]
        self.bathcnorm_flow = self.model_conf["batchnorm_flow"]
        self.conv1x1 = self.model_conf["conv1x1"]
        self.res_layers_in_block = self.model_conf.get("res_layers_in_block")
        self.normaliztion_out = self.model_conf.get("normalization_out")

        self.u_net = self.model_conf.get("u_net")
        self.resnet = self.model_conf.get("resnet")
        self.ar = self.model_conf.get("ar")

        assert not (self.u_net and self.resnet), "Cannot use both u_net and resnet"
        assert not (self.u_net and self.ar), "Cannot use both u_net and ar"

        self.polynomial = self.model_conf["spline_type"]
        self.bins = self.model_conf["bins"]
        self.tail_bound = self.model_conf["tail_bound"]

        self.hidden_layer = []
        for _ in range(self.num_hidden_layers):
            self.hidden_layer.append(self.hidden_layer_dim)
        self.hidden_layer.append(self.input_dim)

        if self.polynomial == "linear":
            model = LinearSplineFlowModel
        elif self.polynomial == "quadratic":
            model = QuadraticSplineFlowModel
        else:
            raise ValueError(f"Polynomial type {self.polynomial} not implemented.")

        blocks = []
        for _ in range(self.num_flows):
            if self.bathcnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
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
                    repeats=self.res_layers_in_block,
                )
            )

        if self.normaliztion_out:
            if self.bathcnorm_flow:
                blocks.append(BatchNormFlow(self.input_dim))

            if self.conv1x1:
                blocks.append(Conv1x1PLU(self.input_dim, device=self.device))
            else:
                blocks.append(ReverseFlow(self.input_dim))

        self.model = NormalizingFlow(
            self.input_dim,
            blocks,
            self.base_distribution,
        )
