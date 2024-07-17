import logging
import os

import mlflow
import torch
import yaml


def register_from_checkpoint(trainer, base_module, model_name=None, save_module=True):
    """Register model from checkpoint to mlflow database.

    Note
    ----
    Compiled models cannot be saved. An uncompiled version of the model is saved in the lightning module as
    `uncompiled_model`. The state is then loaded into this model from the compiled one and registered.

    Parameters
    ----------
    trainer : L.Trainer
        Lightning trainer object.
    base_module : L.LightningModule
        Base module that contains the model.
    model_name : str, optional
        Name of the model to register, by default None.
    save_module : bool, optional
        Whether to save lightning module or torch nn model, by default True.

    References
    ----------
    [1] - https://stackoverflow.com/questions/55047065/unexpected-keys-in-state-dict-model-opt
    [2] - https://discuss.pytorch.org/t/how-to-save-load-a-model-with-torch-compile/179739/2

    Returns
    -------
    dict
        Dictionary with model name as key and model as value.
    """
    logger = trainer.logger
    callback = logger._checkpoint_callback

    ckpt_best_model_path = callback.best_model_path

    experiment_id = logger.experiment_id
    run_id = logger.run_id

    state_dict = torch.load(ckpt_best_model_path)["state_dict"]
    checkpoint_dir = f"mlruns/{experiment_id}/{run_id}"

    if base_module.uncompiled_model is not None:
        remove_prefix = "_orig_mod."
        state_dict = {k.replace(remove_prefix, "") if remove_prefix in k else k: v for k, v in state_dict.items()}

        base_module.model = base_module.uncompiled_model
        base_module.uncompiled_model = None

    base_module.load_state_dict(state_dict)
    base_module.tracker = None

    logging.info(f"Registering model {model_name}.")

    mlflow.pytorch.log_model(
        base_module if save_module else base_module.model,
        artifact_path=f"{checkpoint_dir}/artifacts",
        signature=None,
        registered_model_name=model_name,
    )

    logging.info(f"Removing model in checkpoint directory {checkpoint_dir}/.")
    os.system(f"rm -rf {checkpoint_dir}/artifacts/model")
    os.system(f"rm -rf {checkpoint_dir}/checkpoints")

    return {model_name: base_module}


def fetch_registered_module(model_name, model_version=-1, device="cpu"):
    mlflow_models = list_registered_objects()

    if model_version == -1:
        model_version = max(mlflow_models[model_name].keys())

    model_artifact = mlflow_models[model_name][model_version].source
    logging.info(f"Loading version {model_version} of {model_name} model on {device} from: {model_artifact}.")

    return mlflow.pytorch.load_model(model_artifact, map_location=torch.device(device))


def list_registered_objects():
    mlflow_models = dict()

    for r in mlflow.MlflowClient().search_model_versions():
        if r.name not in mlflow_models:
            mlflow_models[r.name] = dict()

        mlflow_models[r.name][r.version] = r

    return mlflow_models


def list_files_walk(start_path=".", return_dirs=False, full_path=True):
    """List files in a directory and its subdirectories.

    Parameters
    ----------
    start_path : str, optional
        Where to start search, by default ".".
    return_dirs : bool, optional
        If True will return list of directories, by default False.
    full_path : bool, optional
        If True will append current directory, by default True.

    Returns
    -------
    list or (lis, list) if return_dirs=True
        List of files or list of files and directories.
    """
    file_lst, dir_lst, current_dir = [], [], os.getcwd()

    for root, _, files in os.walk(start_path):
        if return_dirs:
            if full_path:
                dir_lst.append(f"{current_dir}/{root}")
            else:
                dir_lst.append(root)

        for file in files:
            if full_path:
                file_lst.append(f"{current_dir}/{os.path.join(root, file)}")
            else:
                file_lst.append(os.path.join(root, file))

    if return_dirs:
        return file_lst, dir_lst
    else:
        return file_lst


def change_mlruns_path(new_path, mlruns_dir):
    files = list_files_walk(mlruns_dir + "/0")

    if new_path[-1] == "/":
        new_path = new_path[:-1]

    for f in files:
        if not f.endswith("meta.yaml"):
            continue

        with open(f) as yaml_f:
            meta_dct = yaml.safe_load(yaml_f)

        if "artifact_uri" not in meta_dct:
            continue

        artifact_uri = meta_dct["artifact_uri"]

        if "F9ML" not in artifact_uri:
            continue

        replace_path = "file://" + new_path + "/" + "".join(artifact_uri.partition("F9ML")[1:])
        meta_dct["artifact_uri"] = replace_path

        logging.info(f"Replacing {artifact_uri} with {replace_path}.")

        with open(f, "w") as yaml_f:
            yaml.dump(meta_dct, yaml_f)

    return True


if __name__ == "__main__":
    from ml.common.utils.loggers import setup_logger

    setup_logger()

    change_mlruns_path("/data0/kersevan/", mlruns_dir="mlruns_")
