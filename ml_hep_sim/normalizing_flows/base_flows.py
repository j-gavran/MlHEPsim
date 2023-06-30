import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam, lr_scheduler


class NormalizingFlow(nn.Module):
    def __init__(self, dim, blocks, density=None):
        """Implements base class for chaining together flow layers (bijectors).

        Note
        ----
        Training is implemented within the forward method that calls inverse of a flow because we are interested in
        the generative process and need to train in the normalizing direction (defined as inverse in flows). Also note
        that some flows only support forward in the sense of this class (e.g. batchnorm).

        Parameters
        ----------
        dim: int
            Input dimension.
        blocks : list of flow objects
            Chained together flow of nn modules.
        density : PyTorch distribution
            Base/prior distribution. Must subclass distributions.Distribution or implement log_prob and sample methods.
        """
        super().__init__()
        self.dim = dim
        self.bijectors = nn.ModuleList(blocks)
        self.base_distribution = density
        self.log_det = None

    def forward(self, z):
        self.log_det = []

        for bijector in self.bijectors:
            if bijector.normalizing_direction:
                z, log_abs_det = bijector.inverse(z)
            else:
                z, log_abs_det = bijector.forward(z)

            self.log_det.append(log_abs_det)

        return z, self.log_det

    def inverse(self, z):
        self.log_det = []

        for bijector in self.bijectors[::-1]:
            if bijector.normalizing_direction:
                z, log_abs_det = bijector.forward(z)
            else:
                z, log_abs_det = bijector.inverse(z)

            self.log_det.append(log_abs_det)

        return z, self.log_det

    def sample(self, num_samples):
        z = self.base_distribution.sample((num_samples,))
        xs, _ = self.inverse(z)
        return xs


class MaskedNormalizingFlow(NormalizingFlow):
    def __init__(self, *args, mask_type="checkerboard", mask_device="cpu"):
        """Masked normalizing flow. Shifts between odd and even masks at the begining of every block."""
        super().__init__(*args)
        self.mask_type, self.mask_device = mask_type, mask_device
        self._validate_blocks()
        self._mask_setup()

    def _checkerboard_mask(self):
        mask1 = torch.arange(0, self.dim, 1, dtype=torch.float) % 2
        mask2 = 1 - mask1
        return [mask1.to(self.mask_device), mask2.to(self.mask_device)]

    def _halfhalf_mask(self):
        mask_zeros = torch.zeros(self.dim // 2)
        mask_ones = torch.ones(self.dim // 2)

        mask1 = torch.cat((mask_zeros, mask_ones))
        mask2 = torch.cat((mask_ones, mask_zeros))
        return [mask1.to(self.mask_device), mask2.to(self.mask_device)]

    def _mask_setup(self):
        if self.mask_type == "checkerboard":
            self.masks = self._checkerboard_mask()
        elif self.mask_type == "halfhalf":
            self.masks = self._halfhalf_mask()
        else:
            raise NotImplemented

    def _validate_blocks(self):
        c = 0
        for bijector in self.bijectors:
            if bijector.mask is not False:
                c += 1

        if c % 2 != 0:
            raise ValueError("Number of masked layers must be even!")

    def forward(self, z):
        self.log_det, c = [], 0

        for bijector in self.bijectors:
            if bijector.mask is not False:
                if c % 2 == 0:
                    mask = self.masks[0]
                else:
                    mask = self.masks[1]

                bijector.set_mask(mask)
                c += 1

            if bijector.normalizing_direction:
                z, log_abs_det = bijector.inverse(z)
            else:
                z, log_abs_det = bijector.forward(z)

            self.log_det.append(log_abs_det)

        return z, self.log_det

    def inverse(self, z):
        self.log_det, c = [], 0

        for bijector in self.bijectors[::-1]:
            if bijector.mask is not False:
                if c % 2 != 0:
                    mask = self.masks[0]
                else:
                    mask = self.masks[1]

                bijector.set_mask(mask)
                c += 1

            if bijector.normalizing_direction:
                z, log_abs_det = bijector.forward(z)
            else:
                z, log_abs_det = bijector.inverse(z)

            self.log_det.append(log_abs_det)

        return z, self.log_det


class PlFlowModel(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        device="cpu",
        base_distribution="normal",
        data_name="",
        lr_scheduler_dct=None,
        learning_rate=1e-3,
        weight_decay=0,
    ):
        super().__init__()
        self.input_dim = input_dim

        if base_distribution.lower() == "normal":
            self.base_distribution = Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
        else:
            raise NotImplemented

        self.data_name = data_name

        self.learning_rate = learning_rate
        self.lr_scheduler_dct = lr_scheduler_dct
        self.weight_decay = weight_decay

        self.save_hyperparameters()
        self.current_step = 0

        self.flow = None

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
        z, log_det = self.flow(x)
        return z, log_det

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z, log_jac = self.flow(x)

        jac_loss = sum(log_jac)
        nll = self.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
        loss = -torch.mean(jac_loss + nll)

        self.log("train_loss", loss)
        self.current_step += 1

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z, log_jac = self.flow(x)

        jac_loss = sum(log_jac)
        nll = self.base_distribution.log_prob(z).sum(dim=-1, keepdim=True)
        loss = -torch.mean(jac_loss + nll)

        self.log("val_loss", loss)
        self.log("sum_log_det_jac", torch.mean(jac_loss))
        self.log("val_nll", torch.mean(nll))

        return {"val_loss": loss, "sum_log_det_jac": jac_loss, "val_nll": nll}

    def on_train_start(self):
        try:
            self.logger.experiment.log_text(self.logger.run_id, str(self.flow), "model_str.txt")
        except AttributeError:
            pass
