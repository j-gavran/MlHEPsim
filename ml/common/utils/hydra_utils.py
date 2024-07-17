import time
from collections.abc import Mapping

import hydra
from hydra import compose, initialize


def run_with_config(config_path=".", config_name="main_config", use_ipython=True, update_config=None):
    """A decorator for running a main function with hydra config. Use this decorator inside juptyer notebooks.

    Example
    -------
    @run_with_config(
        config_path="../../custom/HIGGS/config/vae/",
        config_name="main_config",
        update_config={
            "experiment_config": {
                "run_name": "normal",
                "stage": "test",
                "model_version": 1,
            }
        },
    )

    References
    ----------
    [1] - https://github.com/facebookresearch/hydra/issues/2025

    """

    def decorator(main_fn):
        def wrapper(*args, **kwargs):
            if use_ipython:
                # config_path is relative to the parent of the caller
                with initialize(version_base=None, config_path=config_path):
                    config = compose(config_name)
                    if update_config is not None:
                        config = deep_update(config, update_config)

                return main_fn(config)
            else:
                decorator = hydra.main(version_base=None, config_path=config_path, config_name=config_name)
                return decorator(main_fn)()

        return wrapper

    return decorator


def deep_update(source, overrides):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def setup_config(config, experiment_prefix):
    # get configuration
    experiment_conf = config.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    if experiment_conf["experiment_name"] is None:
        t = time.localtime()
        experiment_name = f"{experiment_prefix}_{t.tm_mday:02d}{t.tm_mon:02d}{(t.tm_year%100):02d}"
        experiment_conf["experiment_name"] = experiment_name

    data_conf = config.data_config
    model_conf = config.model_config
    training_conf = config.training_config

    return experiment_conf, data_conf, model_conf, training_conf
