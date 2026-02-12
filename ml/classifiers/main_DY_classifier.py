""" Binary classifier for Drell-Yan dataset to distinguish real from generated events. """

import numpy as np
import pandas as pd
import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from ml.classifiers.models import BinaryClassifier
from ml.common.data_utils.data_modules import DataModule, SupervisedDataset


class ClassifierProcessor:
    """Processor to load and combine real and generated data with labels."""
    
    def __init__(self, real_data_path, generated_data_path, n_samples=None):
        """
        Parameters
        ----------
        real_data_path : str
            Path to real MC data .npy file
        generated_data_path : str
            Path to generated data .npy file
        n_samples : int, optional
            Number of samples to use from each dataset (for testing)
        """
        self.real_data_path = real_data_path
        self.generated_data_path = generated_data_path
        self.n_samples = n_samples
    
    def __call__(self):
        # Load real data (label = 1 for real)
        real_data = np.load(self.real_data_path)
        if self.n_samples is not None:
            real_data = real_data[:self.n_samples]
        real_labels = np.ones((len(real_data), 1), dtype=np.float32)
        
        # Load generated data (label = 0 for generated/fake)
        gen_data = np.load(self.generated_data_path)
        if self.n_samples is not None:
            gen_data = gen_data[:self.n_samples]
        gen_labels = np.zeros((len(gen_data), 1), dtype=np.float32)
        
        # Combine datasets
        combined_data = np.vstack([real_data, gen_data])
        combined_labels = np.vstack([real_labels, gen_labels])
        
        # Combine features and labels
        data = np.hstack([combined_data, combined_labels])
        
        # Create selection DataFrame
        n_features = combined_data.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]
        selection = pd.DataFrame({
            "name": feature_names + ["label"],
            "type": ["continuous"] * n_features + ["label"]
        })
        
        print(f"Loaded {len(real_data)} real events and {len(gen_data)} generated events")
        print(f"Total dataset size: {len(data)} events with {n_features} features")
        
        # No scalers needed (data already preprocessed by the flow model)
        scalers = None
        
        return data, selection, scalers


class ClassifierDataModule(DataModule):
    """DataModule for binary classification of real vs generated events."""
    
    def __init__(self, processor, dataset=SupervisedDataset, **kwargs):
        super().__init__(processor, dataset, **kwargs)


@hydra.main(config_path="configs/", config_name="main_config", version_base=None)
def main(config):
    # Get configuration
    model_conf = config.model_config
    training_conf = config.training_config
    data_conf = config.data_config
    experiment_conf = config.experiment_config
    
    # Initialize classifier
    classifier = BinaryClassifier(model_conf, training_conf)
    
    # Create processor to load real and generated data
    processor = ClassifierProcessor(
        real_data_path=data_conf["real_data_path"],
        generated_data_path=data_conf["generated_data_path"],
        n_samples=data_conf.get("n_samples", None)
    )
    
    # Create data module
    data_module = ClassifierDataModule(
        processor=processor,
        train_split=data_conf["train_split"],
        val_split=data_conf["val_split"],
        batch_size=data_conf["batch_size"],
        num_workers=data_conf["num_workers"],
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy"),
        EarlyStopping(
            monitor="val_accuracy", 
            mode="max", 
            patience=training_conf["early_stop_patience"]
        ),
    ]
    
    # Initialize logger
    logger = MLFlowLogger(
        experiment_name=experiment_conf["experiment_name"],
        run_name=experiment_conf["run_name"],
        save_dir=experiment_conf["save_dir"],
    )
    
    # Train using PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=training_conf["epochs"],
        callbacks=callbacks,
        logger=logger,
        accelerator=experiment_conf["accelerator"],
        devices=experiment_conf["devices"],
    )
    
    trainer.fit(classifier, data_module)


if __name__ == "__main__":
    main()