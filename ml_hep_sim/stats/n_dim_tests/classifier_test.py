import hydra
from ml_hep_sim.ml_utils import train_wrapper
from ml_hep_sim.nets.classifiers import BinaryLabelClassifier


@hydra.main(config_path="../../conf", config_name="classifier_config", version_base=None)
def train_classifier(config):
    input_dim = config["datasets"]["input_dim"]

    return train_wrapper(
        config["model_config"],
        input_dim=input_dim,
        pl_model=BinaryLabelClassifier,
        trainer_dict=config["trainer_config"],
        data_name=config["datasets"]["data_name"],
        mlf_trainer_dict=config["logger_config"],
        data_param_dict=config["datasets"]["data_params"],
    )


if __name__ == "__main__":
    train_classifier()
