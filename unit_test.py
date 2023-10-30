import torch
import ray
import unittest
import os
import cv2
import numpy as np
from tests import *

CUDA_VISIBLE_DEVICES = torch.cuda.device_count()

if __name__ == "__main__":
    assert int(CUDA_VISIBLE_DEVICES) > 0
    # start ray service
    ray.init(num_gpus=int(CUDA_VISIBLE_DEVICES))
    unittest.main()
    # shutdown ray service
    ray.shutdown(_exiting_interpreter=True)
