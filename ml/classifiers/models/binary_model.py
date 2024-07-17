import logging

from torchmetrics.classification import BinaryAccuracy

from ml.classifiers.models.multilabel_model import MultilabelClassifier


class BinaryClassifier(MultilabelClassifier):
    def __init__(self, model_conf, training_conf, tracker=None):
        super().__init__(model_conf, training_conf, tracker=tracker)
        self._check_config()

    def _check_config(self):
        if self.model_conf.get("act_out", None) == "Sigmoid" and self.training_conf["loss"] == "MSELoss":
            logging.warning("Using Sigmoid activation with MSE is not recommended. Consider using BCEWithLogitsLoss.")

        if self.training_conf["loss"] == "BCEWithLogitsLoss":
            logging.info("Using raw logits as output. Use Sigmoid for accuracy inference!")

        if self.model_conf.get("act_out", None) != "Sigmoid" and self.training_conf["loss"] == "BCELoss":
            raise ValueError("Use Sigmoid activation with BCELoss!")

        if self.model_conf.get("act_out", None) == "Sigmoid" and self.training_conf["loss"] == "BCEWithLogitsLoss":
            raise ValueError("Using Sigmoid activation with BCEWithLogitsLoss is not allowed!")

    def _get_loss(self, batch):
        yp = self.forward(batch)
        loss = self.loss_func(yp, batch[1])
        return loss, yp

    def _get_accuracy(self, predicted, target):
        """https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#binaryaccuracy

        If preds is a floating point tensor with values outside [0, 1] range we consider the input to be logits and will
        auto apply sigmoid per element.

        """
        metric = BinaryAccuracy().to(predicted.device)
        return metric(predicted, target)

    def training_step(self, batch, *args):
        loss, _ = self._get_loss(batch)
        self.log("train_loss", loss, batch_size=batch[0].size()[0])
        return loss

    def validation_step(self, batch, *args):
        loss, yp = self._get_loss(batch)
        self.log("val_loss", loss, batch_size=batch[0].size()[0])

        val_acc = self._get_accuracy(yp, batch[1])
        self.log("val_accuracy", val_acc, batch_size=batch[0].size()[0])

    def test_step(self, batch, *args):
        loss, yp = self._get_loss(batch)
        self.log("test_loss", loss, batch_size=batch[0].size()[0])

        test_acc = self._get_accuracy(yp, batch[1])
        self.log("test_accuracy", test_acc, batch_size=batch[0].size()[0])
