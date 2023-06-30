import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler


class BaseVAE(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.encoder, self.decoder = ae.encoder, ae.decoder
        self.latent_dim = ae.latent_dim

    def encode(self, x):
        raise NotImplemented

    def decode(self, x):
        raise NotImplemented

    def loss_function(self):
        raise NotImplemented

    def forward(self, x):
        raise NotImplemented


class PlVAEModel(pl.LightningModule):
    def __init__(self, input_dim, data_name="", learning_rate=1e-3, weight_decay=0.0, lr_scheduler_dct=None):
        """Base lightning VAE model.

        Parameters
        ----------
        input_dim : int
            Feature dimension.
        data_name : str, optional
            Name of the dataset, by default "".
        learning_rate : float, optional
            Learning rate for Adam, by default 1e-3.
        weight_decay : float, optional
            L2, by default 0.0.
        lr_scheduler_dct : dict, optional
           Dict of {"scheduler": <name of scheduler>, "params": <list of params (see docs)>}, by default None.

        Note
        ----
        See https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate.

        from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, StepLR

        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        scheduler = ExponentialLR(optimizer, gamma)
        scheduler = StepLR(optimizer, step_size, gamma=0.5)
        scheduler = CosineAnnealingLR(optimizer, gamma, eta_min=0.0)

        """
        super().__init__()

        self.input_dim = input_dim
        self.data_name = data_name

        self.learning_rate = learning_rate
        self.lr_scheduler_dct = lr_scheduler_dct
        self.weight_decay = weight_decay

        self.save_hyperparameters()
        self.current_step = 0

        self.vae, self.latent_dim = None, None

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=False)

        if self.lr_scheduler_dct:
            get_scheduler = getattr(lr_scheduler, self.lr_scheduler_dct["scheduler"])
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

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_batch, mu, logvar = self(x)
        loss = self.vae.loss_function(recon_batch, x, mu, logvar)
        train_loss = loss[0] + loss[1]

        self.log("train_loss", train_loss)
        self.log("train_rec_loss", loss[0])
        self.log("train_kl_loss", loss[1])

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_batch, mu, logvar = self(x)
        loss = self.vae.loss_function(recon_batch, x, mu, logvar)
        val_loss = loss[0] + loss[1]

        self.log("val_loss", val_loss)
        self.log("val_rec_loss", loss[0])
        self.log("val_kl_loss", loss[1])

        return {"val_loss": val_loss}

    def sample(self, n_samples, device="cpu"):
        noise = torch.randn(n_samples, self.vae.latent_dim, device=device)
        return self.vae.decode(noise)

    def on_train_start(self):
        self.logger.experiment.log_text(self.logger.run_id, str(self.vae), "model_str.txt")
