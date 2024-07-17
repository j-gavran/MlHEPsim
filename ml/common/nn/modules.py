import copy
import logging
import os
from abc import ABC, abstractmethod

import lightning as L
import torch


class Tracker(ABC):
    def __init__(self, experiment_conf, tracker_path):
        self.experiment_conf = experiment_conf
        self.tracker_path = tracker_path

        self.module = None

        self.current_epoch = None  # current epoch number
        self.plotting_dirs = None  # directories for saving plots
        self.stage = None  # set in get_predictions

        self.base_dir = f"{self.tracker_path}/{self.experiment_conf['run_name']}/"
        logging.debug(f"Tracker base directory: {self.base_dir}")

    def __call__(self, module):
        self.module = module
        return self

    def on_first_epoch(self):
        """Create directories if they don't exist yet, should be called after the first epoch in compute method."""
        self.create_dirs()

    def create_dirs(self):
        """Creates the directories where the plots will be saved."""
        self.plotting_dirs = self.make_plotting_dirs()

        # create directories if they don't exist yet
        for d in list(self.plotting_dirs.values()):
            if not os.path.exists(d):
                logging.debug(f"Creating tracker directory after first epoch: {d}")
                os.makedirs(d)

    @abstractmethod
    def make_plotting_dirs(self):
        """Create a dictionary of directories for different plotting graphs."""
        pass

    @abstractmethod
    def get_predictions(self, stage):
        """Needs to be implemented for different tasks. Basically, it is the forward of the model."""
        return None

    @abstractmethod
    def compute(self, stage):
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if self.current_epoch == 0:
            self.on_first_epoch()

        # check if metrics should be calculated this epoch
        if self.current_epoch % self.experiment_conf["check_metrics_n_epoch"] != 0 and self.stage != "test":
            logging.debug(f"Skipping metrics computation for epoch {self.current_epoch}")
            return False

        # get predictions, needs to be implemented
        self.get_predictions(stage)

        return True

    @abstractmethod
    def plot(self):
        """Plot the metrics."""
        self.module.logger.experiment.log_artifact(local_path=self.base_dir, run_id=self.module.logger.run_id)
        return None


class Module(L.LightningModule):
    def __init__(
        self,
        model_conf,
        training_conf,
        model,
        loss_func=None,
        tracker=None,
        split_idx_dct=None,
        scalers=None,
        selection=None,
    ):
        """Base class for MLP models in pytorch-lightning.

        Note
        ----
        If model is passed as None, you need to redefine forward function in your class.

        Parameters
        ----------
        params : dict
            Parameters from src.utils.params are passed here.
        model : nn.Module
            Torch model to use.
        loss_func : method
            Loss function to use.
        tracker : object
            Class for tracking (plots and metrices).
        split_idx_dct : dict
            Dictionary with split indices for train, val and test datasets.
        scalers : dict
            Dictionary with scalers for the data feature scaling.
        selection : pd.DataFrame
            Dictionary with the selection of features used in the model.

        References
        ----------
        [1] - https://pytorch.org/docs/stable/generated/torch.compile.html

        """
        super().__init__()
        self.model_conf = model_conf
        self.training_conf = training_conf
        self.output_dim = model_conf.get("output_dim")
        self.loss_func = loss_func
        self.tracker = tracker

        if self.training_conf.get("compile", False):
            logging.info("[b][red]Torch compile is ON! Model will be compiled in default mode.[/red][/b]")
            self.uncompiled_model = copy.deepcopy(model)  # need for saving with state_dict
            self.model = torch.compile(model, mode="default")
        else:
            self.uncompiled_model = None
            self.model = model

        if self.tracker is not None:
            self.tracker = self.tracker(self)  # initialize tracker

        # save split indices and scalers, if available, on train start from datamodule
        self.split_idx_dct, self.scalers, self.selection = split_idx_dct, scalers, selection

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.training_conf["optimizer"])
        optimizer = optimizer(
            self.parameters(),
            lr=self.training_conf["learning_rate"],
            weight_decay=self.training_conf["weight_decay"],
        )

        if self.training_conf["scheduler"]["scheduler_name"]:
            get_scheduler = getattr(torch.optim.lr_scheduler, self.training_conf["scheduler"]["scheduler_name"])
            scheduler = get_scheduler(optimizer, **self.training_conf["scheduler"]["scheduler_params"])
            scheduler_dct = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": self.training_conf["scheduler"]["interval"],
                },
            }
            return scheduler_dct
        else:
            return {"optimizer": optimizer}

    def forward(self, batch):
        yp = self.model(batch[0])
        return yp

    def _get_loss(self, batch):
        yp = self.forward(batch)
        loss = self.loss_func(batch[1], yp)
        return loss

    def training_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, batch_size=batch[0].size()[0])
        return loss

    def validation_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, batch_size=batch[0].size()[0])

    def test_step(self, batch, *args):
        loss = self._get_loss(batch)
        self.log("test_loss", loss, batch_size=batch[0].size()[0])

    def on_train_start(self):
        # hijack datamodule and module and save split indices and scalers
        dm = self._trainer.datamodule

        try:
            train_idx, val_idx, test_idx = dm.train_idx, dm.val_idx, dm.test_idx
            self.split_idx_dct = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        except Exception as e:
            logging.error(f"Could not save split indices due to {e}")

        try:
            self.scalers = dm.scalers
        except Exception as e:
            logging.error(f"Could not save scalers due to {e}")

        try:
            self.selection = dm.selection
        except Exception as e:
            logging.error(f"Could not save selection due to {e}")

        # log model architecture string to mlflow
        self.logger.experiment.log_text(self.logger.run_id, str(self), "model_str.txt")

    def on_train_epoch_end(self):
        if self.training_conf["scheduler"]["scheduler_name"]:
            reduce_lr_on_epoch = self.training_conf["scheduler"]["reduce_lr_on_epoch"]
            if reduce_lr_on_epoch is not None:
                self.lr_schedulers().base_lrs = [self.lr_schedulers().base_lrs[0] * reduce_lr_on_epoch]

    def on_validation_epoch_end(self):
        if self.tracker:
            self.tracker.compute(stage="val")
            self.tracker.plot()

    def on_test_start(self):
        if self.tracker:
            self.tracker.compute(stage="test")
            self.tracker.plot()
