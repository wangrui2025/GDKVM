"""
A helper function to get a default model for quick testing
"""
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from model.cutie import CUTIE
from inference.utils.args_utils import get_dataset_cfg
from scripts.download_models import download_models_if_needed


def get_default_model() -> CUTIE:
    initialize(version_base='1.3.2', config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")

    download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = './weights/cutie-base-mega.pth'
    get_dataset_cfg(cfg)

    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    cutie.load_weights(model_weights)

    return cutie
