import argparse
import torch
import os
from omegaconf import OmegaConf
import cv2
import numpy as np
from tqdm import tqdm
from utils import load_train_config, load_model, transform_to_input, transform_to_output

REF_DIR = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default=f"{REF_DIR}/configs/predict_config.yaml"
    )

    return parser.parse_args()


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
