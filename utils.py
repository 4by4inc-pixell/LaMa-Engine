from typing import Tuple
import torch
import os
import yaml
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
import cv2
import numpy as np


def load_train_config(train_config_path: str):
    assert os.path.isfile(train_config_path)
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"
        return train_config


def load_model(train_config: dict, checkpoint_path: str, device: torch.device):
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    model.freeze()
    model.to(device)
    return model


def transform_to_input(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    assert len(image.shape) == 3
    assert len(mask.shape) == 2
    ih, iw, ic = image.shape
    mh, mw = mask.shape
    mask_max = np.max(mask)
    assert ih == mh and iw == mw
    assert ic == 3
    assert mask_max != 0
    image = np.transpose(image, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)
    mask = mask / mask_max
    mask_inv = (1 - mask).astype(np.uint8)
    masked_image = np.zeros_like(image)
    [
        np.multiply(
            image[i, :, :], mask_inv[0], out=masked_image[i, :, :], dtype=np.uint8
        )
        for i in range(3)
    ]
    input_image = np.divide(masked_image, 255.0, dtype=np.float32)
    input_mask = mask.astype(np.float32)
    input_data = np.concatenate([input_image, input_mask], axis=0)
    input_data = torch.tensor(input_data)
    return input_data, input_image, input_mask


def transform_to_output(
    pred: torch.Tensor, input_image: np.ndarray, input_mask: np.ndarray
) -> np.ndarray:
    ic, ih, iw = input_image.shape
    pred = pred.detach().cpu().numpy()

    pred_image = np.zeros_like(input_image)
    [
        np.multiply(pred[i, :ih, :iw], input_mask[0], out=pred_image[i, :, :])
        for i in range(ic)
    ]
    result = input_image + pred_image
    result = np.clip((result * 255), 0, 255).astype(np.uint8)
    result = np.transpose(result, (1, 2, 0))
    return result
