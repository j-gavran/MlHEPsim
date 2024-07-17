import torch
import torch.nn as nn

from ml.common.nn.mlp import BasicMLP, BasicResMLP
from ml.common.nn.modules import Module


class MultilabelClassifier(Module):
    def __init__(self, model_conf, training_conf, tracker=None):
        super().__init__(model_conf, training_conf, model=None, tracker=tracker)

        if model_conf["model"] == "MLP":
            self.model = BasicMLP(
                [model_conf["input_dim"]] + model_conf["hidden_dim"] + [model_conf["output_dim"]],
                act=model_conf["activation_function"],
                act_out=model_conf.get("act_out", None),
                batchnorm=model_conf["batchnorm"],
                act_first=model_conf.get("act_first", True),
                dropout=model_conf["dropout"],
            )
        elif model_conf["model"] == "ResMLP":
            self.model = BasicResMLP(
                [model_conf["input_dim"]] + model_conf["hidden_dim"] + [model_conf["output_dim"]],
                act=model_conf["activation_function"],
                act_out=model_conf.get("act_out", None),
                batchnorm=model_conf["batchnorm"],
                act_first=model_conf.get("act_first", True),
                dropout=model_conf["dropout"],
                repeats=model_conf.get("repeats", 2),
            )
        else:
            raise NotImplementedError("The model set in the params is not yet implemented!")

        self.loss_func = getattr(nn, training_conf["loss"])()

        self.save_hyperparameters(ignore=["tracker", "loss_func"])

    def _get_loss(self, batch):
        yp = self.forward(batch)
        loss = self.loss_func(yp, torch.argmax(batch[1], dim=1))
        return loss
