import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lama_engine.saicinpainting.training.data.datasets import (
    make_constant_area_crop_params,
)
from lama_engine.saicinpainting.training.losses.distance_weighting import (
    make_mask_distance_weighter,
)
from lama_engine.saicinpainting.training.losses.feature_matching import (
    feature_matching_loss,
    masked_l1_loss,
)
from lama_engine.saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from lama_engine.saicinpainting.training.trainers.base import (
    BaseInpaintingTrainingModule,
    make_multiscale_noise,
)
from lama_engine.saicinpainting.utils import add_prefix_to_keys, get_ramp

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(
        img_height=batch["image"].shape[2], img_width=batch["image"].shape[3], **kwargs
    )
    batch["image"] = batch["image"][
        :, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
    ]
    batch["mask"] = batch["mask"][
        :, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
    ]
    return batch


class CustomInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(
        self,
        *args,
        concat_mask=True,
        rescale_scheduler_kwargs=None,
        image_to_discriminator="predicted_image",
        add_noise_kwargs=None,
        noise_fill_hole=False,
        const_area_crop_kwargs=None,
        distance_weighter_kwargs=None,
        distance_weighted_mask_for_discr=False,
        fake_fakes_proba=0,
        fake_fakes_generator_kwargs=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = (
            get_ramp(**rescale_scheduler_kwargs)
            if rescale_scheduler_kwargs is not None
            else None
        )
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = (
            make_mask_distance_weighter(**distance_weighter_kwargs)
            if distance_weighter_kwargs is not None
            else None
        )
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(
                **(fake_fakes_generator_kwargs or {})
            )

    def forward(self, input_data):
        predicted_image = self.generator(input_data)

        return predicted_image
