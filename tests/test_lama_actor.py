import torch
from lama_engine import LaMaEngine
import unittest
import os
import cv2
import numpy as np

REF_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(REF_DIR, "assets", "predict_config.yaml")
NUMBER_OF_ACTOR_PER_GPU = 1
TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST = {
    "SD": (480, 640, 3),
    "HD": (720, 1280, 3),
    "FHD": (1080, 1920, 3),
    "2K": (1440, 2560, 3),
    "2K.T": (1440, 2560, 3),
    "FHD.T": (1920, 1080, 3),
    "HD.T": (1280, 720, 3),
    "SD.T": (640, 480, 3),
    # "4K": (2160, 3840, 3),  # NOT SUPPORT!
    # "8K": (4320, 7680, 3), # NOT SUPPORT!
}
TEST_IMAGES_DIR = os.path.join(REF_DIR, "assets", "sampes_for_export")


class TestLaMaActor(unittest.TestCase):
    def setUp(self) -> None:
        """
        모든 unittest 직전에 이 메서드가 호출됩니다.
        """
        self.engine = LaMaEngine(
            config_path=CONFIG_PATH, actors_per_gpu=NUMBER_OF_ACTOR_PER_GPU
        )

        image_original_file_name = "image_original.png"
        image_mask_1_file_name = "image_mask_1.png"
        image_mask_2_file_name = "image_mask_2.png"
        image_mask_3_file_name = "image_mask_3.png"
        # make image paths
        self.image_pair_paths = [
            (
                os.path.join(TEST_IMAGES_DIR, image_original_file_name),
                os.path.join(TEST_IMAGES_DIR, fn),
            )
            for fn in [
                image_mask_1_file_name,
                image_mask_2_file_name,
                image_mask_3_file_name,
            ]
        ]

    def tearDown(self) -> None:
        """
        모든 unittest 직후에 이 메서드가 호출됩니다.
        """
        del self.engine

    def test_single_actor_solid(self):
        inputs = [
            (cv2.imread(ip, cv2.IMREAD_COLOR), cv2.imread(mp, cv2.IMREAD_GRAYSCALE))
            for ip, mp in self.image_pair_paths
        ]
        results = self.engine(inputs=inputs, actor_id=0)
        for im, o in zip(inputs, results):
            self.assertEqual(im[0].shape, o.shape)

    def test_single_actor_dynamic(self):
        __method__ = "test_single_actor_dynamic"
        for key in TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST:
            input_shape = TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST[key]
            image = np.clip(np.random.random(input_shape) * 255, 0, 255).astype(
                np.uint8
            )
            mask = np.clip(np.random.random(input_shape[:-1]) * 255, 0, 255).astype(
                np.uint8
            )
            result = self.engine(inputs=[(image, mask)], actor_id=0)[0]
            print(
                f"{__method__}::{key} [image shape={image.shape} | mask shape={mask.shape} | result shape={result.shape}] done."
            )

    def test_single_actor_processing_time(self):
        __method__ = "test_single_actor_processing_time"
        from tqdm import tqdm
        import time

        number_of_iteration = 100
        for key in TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST:
            input_shape = TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST[key]
            image = np.clip(np.random.random(input_shape) * 255, 0, 255).astype(
                np.uint8
            )
            mask = np.clip(np.random.random(input_shape[:-1]) * 255, 0, 255).astype(
                np.uint8
            )
            t = time.time()
            for i in tqdm(range(number_of_iteration)):
                result = self.engine(inputs=[(image, mask)], actor_id=0)[0]
            t = time.time() - t
            print(
                f"{__method__}::{key} [image shape={image.shape} | mask shape={mask.shape} | result shape={result.shape} | avg proc time = {number_of_iteration/t} inference/sec] done."
            )

    def test_multi_actor_solid(self):
        inputs = [
            (cv2.imread(ip, cv2.IMREAD_COLOR), cv2.imread(mp, cv2.IMREAD_GRAYSCALE))
            for ip, mp in self.image_pair_paths
        ]
        results = self.engine(inputs=inputs)
        for im, o in zip(inputs, results):
            self.assertEqual(im[0].shape, o.shape)

    def test_multi_actor_dynamic(self):
        __method__ = "test_multi_actor_dynamic"
        inputs = [
            (
                np.clip(
                    np.random.random(TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST[key])
                    * 255,
                    0,
                    255,
                ).astype(np.uint8),
                np.clip(
                    np.random.random(
                        TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST[key][:-1]
                    )
                    * 255,
                    0,
                    255,
                ).astype(np.uint8),
            )
            for key in TARGET_RESOLUTIONS_FOR_DYNAMIC_INPUT_TEST
        ]
        results = self.engine(inputs=inputs)
        self.assertEqual(len(inputs), len(results))
        for im, o in zip(inputs, results):
            self.assertEqual(im[0].shape, o.shape)


if __name__ == "__main__":
    unittest.main()
