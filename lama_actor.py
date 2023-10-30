import ray
import os
from omegaconf import OmegaConf
from utils import load_model, load_train_config, transform_to_input, transform_to_output
import torch
import numpy as np


@ray.remote(num_gpus=1)
class LaMaActor:
    def __init__(
        self,
        config_path: str,
        max_wh: int = 2560,
    ):
        # load predict config
        self.config_path = config_path
        assert os.path.isfile(config_path)
        self.predict_config = OmegaConf.load(config_path)
        self.device = torch.device(self.predict_config.device)

        # load train config
        self.train_config_path = os.path.join(
            self.predict_config.model.path, "config.yaml"
        )
        self.train_config = load_train_config(train_config_path=self.train_config_path)

        # load model
        self.checkpoint_path = os.path.join(
            self.predict_config.model.path,
            "models",
            self.predict_config.model.checkpoint,
        )
        self.model = load_model(
            train_config=self.train_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self.max_wh = max_wh
        # dry run
        self.model(
            torch.tensor(
                np.zeros([1, 4, self.max_wh, self.max_wh], dtype=np.float32)
            ).to(self.device)
        )

    def run(self, image_original: np.ndarray, image_mask: np.ndarray) -> np.ndarray:
        """
        Infill the image using LaMa.

        Args:
            image_original (np.ndarray): original image array(H x W x 3)
            image_mask (np.ndarray): mask image array(H x W x 1)

        Returns:
            np.ndarray: result array(H x W x 3)
        """
        assert image_original is not None
        assert image_mask is not None
        assert (
            len(image_original.shape) == 3
        ), "image_original's shape must be H x W x C."
        assert len(image_mask.shape) == 2, "image_mask's shape must be H x W."
        ih, iw, ic = image_original.shape
        mh, mw = image_mask.shape
        assert (
            ih == mh and iw == mw
        ), "image_original and image_mask must be the same size."
        assert ic == 3, "image_original must be color(C==3)."
        input_data, input_image, input_mask = transform_to_input(
            image=image_original, mask=image_mask
        )
        with torch.no_grad():
            input_data = torch.unsqueeze(input_data, 0).to(self.device)
            pred = self.model(input_data)[0]
            output_data = transform_to_output(
                pred=pred, input_image=input_image, input_mask=input_mask
            )

        result_image = output_data
        return result_image

    def ping(self):
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
