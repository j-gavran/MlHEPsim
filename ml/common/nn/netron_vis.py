import netron
import torch


def visualize_model(model, forward_sample, model_name="model", onnx_path=None, input_names=None):
    """Netron viewer for neural networks.

    Parameters
    ----------
    model : nn.Module
        Torch model.
    forward_sample : torch.Tensor
        Dummy batch.
    onnx_path : str, optional
        Path to an already existing onnx file, by default None.
    input_names : list, optional
        Name of the inputs of the model, by default None.

    References
    ----------
    [1] - https://github.com/lutzroeder/netron

    """
    if onnx_path is None:
        onnx_path = f"ml/common/nn/{model_name}.onnx"
    else:
        onnx_path = f"{onnx_path}/{model_name}.onnx"

    torch.onnx.export(model, forward_sample, onnx_path, input_names=input_names)
    netron.start(onnx_path)
