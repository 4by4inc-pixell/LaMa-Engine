import argparse
from typing import Tuple
import torch
import os
import yaml
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
import cv2
import numpy as np
import gc
from tqdm import tqdm

REF_DIR = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default=f"{REF_DIR}/configs/predict_config.yaml"
    )

    return parser.parse_args()


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


def main(args: argparse.Namespace):
    print(args)
    config_path = args.config_path

    sample_images_dir = os.path.join(REF_DIR, "assets", "sampes_for_export")
    image_original_file_name = "image_original.png"
    image_mask_1_file_name = "image_mask_1.png"
    image_mask_2_file_name = "image_mask_2.png"
    image_mask_3_file_name = "image_mask_3.png"
    # load predict config
    predict_config = OmegaConf.load(config_path)
    device = torch.device(predict_config.device)
    # load train config
    train_config_path = os.path.join(predict_config.model.path, "config.yaml")
    train_config = load_train_config(train_config_path=train_config_path)
    # load model
    checkpoint_path = os.path.join(
        predict_config.model.path, "models", predict_config.model.checkpoint
    )
    model = load_model(
        train_config=train_config, checkpoint_path=checkpoint_path, device=device
    )
    # make image paths
    image_pair_paths = [
        (
            os.path.join(sample_images_dir, image_original_file_name),
            os.path.join(sample_images_dir, fn),
        )
        for fn in [
            image_mask_1_file_name,
            image_mask_2_file_name,
            image_mask_3_file_name,
        ]
    ]
    # get inference result for original model
    with torch.no_grad():
        for ip, mp in tqdm(image_pair_paths):
            assert os.path.isfile(ip)
            assert os.path.isfile(mp)
            image_original = cv2.imread(ip, cv2.IMREAD_COLOR)
            image_mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            assert image_original is not None
            assert image_mask is not None
            input_data, input_image, input_mask = transform_to_input(
                image=image_original, mask=image_mask
            )
            input_data = torch.unsqueeze(input_data, 0).to(device)
            pred = model(input_data)[0]
            output_data = transform_to_output(
                pred=pred, input_image=input_image, input_mask=input_mask
            )
            cv2.imwrite(mp.replace("sampes_for_export", "outputs"), output_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
