import pytorch_lightning as pl
import torch
from torch import nn, optim

from ml_hep_sim.nets.mlp import BasicLinearNetwork
from ml_hep_sim.nets.resnet import PreActResNet


class MultiLabelClassifier(pl.LightningModule):
    def __init__(self, config, input_dim, data_name="", **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.data_name = data_name
        self.lr_scheduler_dct = config.get("lr_scheduler_dct")
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]

        hidden_layers = config["hidden_layers"]
        activation = config["activation"]
        output_activation = config["output_activation"]

        if config["resnet"]:
            self.network = PreActResNet(
                input_dim,
                hidden_layers[0],
                l=2,
                n=len(hidden_layers),
                output_dim=1,
                activation=activation,
                output_activation=output_activation,
            )
        else:
            self.network = BasicLinearNetwork(
                input_dim,
                hidden_layers,
                activation,
                output_activation,
            )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=False)

        if self.lr_scheduler_dct:
            get_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_dct["scheduler"])
            scheduler = get_scheduler(optimizer, **self.lr_scheduler_dct["params"])
            sh = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": self.lr_scheduler_dct["interval"],
                },
            }
            return sh
        else:
            return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch

        yhat = self(x)
        loss = self.loss_func(yhat, y)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)

        val_loss = self.loss_func(yhat, y)

        self.log("val_loss", val_loss)

        acc = self.get_accuracy(y, yhat)
        self.log("accuracy", acc)

        return {"val_loss": val_loss, "accuracy": acc}

    @staticmethod
    def get_accuracy(true, predicted):
        return (torch.argmax(predicted, dim=1) == true).sum().item() / predicted.shape[0]

    def on_train_start(self):
        self.logger.experiment.log_text(self.logger.run_id, str(self.network), "model_str.txt")


class BinaryLabelClassifier(MultiLabelClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = nn.MSELoss()

    @staticmethod
    def get_accuracy(true, predicted):
        sig_true = true > 0.5
        sig_predicted = predicted > 0.5
        return (sig_true == sig_predicted).sum().item() / sig_predicted.shape[0]
