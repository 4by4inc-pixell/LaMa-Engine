import torch
import ray
from ray.util.actor_pool import ActorPool
import unittest
from lama_actor import LaMaActor
import os
import cv2

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "predict_config.yaml")
CUDA_VISIBLE_DEVICES = torch.cuda.device_count()
NUMBER_OF_ACTOR_PER_GPU = 2
NUMBER_OF_TOTAL_ACTOR = NUMBER_OF_ACTOR_PER_GPU * CUDA_VISIBLE_DEVICES
ACTOR_NUM_GPUS = CUDA_VISIBLE_DEVICES / NUMBER_OF_ACTOR_PER_GPU


class TestLaMaActor(unittest.TestCase):
    def setUp(self) -> None:
        self.actors = [
            LaMaActor.options(num_gpus=ACTOR_NUM_GPUS).remote(config_path=CONFIG_PATH)
            for _ in range(NUMBER_OF_TOTAL_ACTOR)
        ]
        self.actor_pool = ActorPool(self.actors)

        sample_images_dir = os.path.join(
            os.path.dirname(__file__), "assets", "sampes_for_export"
        )
        image_original_file_name = "image_original.png"
        image_mask_1_file_name = "image_mask_1.png"
        image_mask_2_file_name = "image_mask_2.png"
        image_mask_3_file_name = "image_mask_3.png"
        # make image paths
        self.image_pair_paths = [
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

    def tearDown(self) -> None:
        _ = [ray.kill(actor) for actor in self.actors]

    def test_samples_basic(self):
        inputs = [
            (cv2.imread(ip, cv2.IMREAD_COLOR), cv2.imread(mp, cv2.IMREAD_GRAYSCALE))
            for ip, mp in self.image_pair_paths
        ]
        results = self.actor_pool.map(
            lambda actor, v: actor.run.remote(v[0], v[1]), inputs
        )
        results = list(results)
        self.assertEqual(len(inputs), len(results))
        for im, o in zip(inputs, results):
            self.assertEqual(im[0].shape, o.shape)

    def test_single_actor(self):
        inputs = [
            (cv2.imread(ip, cv2.IMREAD_COLOR), cv2.imread(mp, cv2.IMREAD_GRAYSCALE))
            for ip, mp in self.image_pair_paths
        ]
        results = ray.get(
            [self.actors[0].run.remote(image, mask) for image, mask in inputs]
        )
        for im, o in zip(inputs, results):
            self.assertEqual(im[0].shape, o.shape)

    def test_upper2(self):
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    assert int(CUDA_VISIBLE_DEVICES) > 0
    # start ray service
    ray.init(num_gpus=int(CUDA_VISIBLE_DEVICES))
    unittest.main()
    # shutdown ray service
    ray.shutdown(_exiting_interpreter=True)
