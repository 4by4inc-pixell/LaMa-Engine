"""
setup script for pip install
"""
import os
from setuptools import setup, find_packages


REQUIREMENTS = [
    "torch",
    "torchvision",
    "albumentations==0.5.2",
    "onnx",
    "omegaconf",
    "webdataset",
    "tqdm",
    "easydict",
    "pandas",
    "lightning",
    "kornia==0.5.0",
    "scikit-learn",
    "ray==2.7.1",
]


def copy_dir():
    dir_path = "YOUR_DIR_HERE"
    base_dir = os.path.join("MODULE_DIR_HERE", dir_path)
    for dirpath, dirnames, files in os.walk(base_dir):
        for f in files:
            yield os.path.join(dirpath.split("/", 1)[1], f)


# INCLUDE_MODULE = ["lama_engine", "lama_engine.models", "lama_engine.saicinpainting"]


setup(
    name="lama-engine",
    version="0.3.5",
    # package_dir={"lama_engine": "lama_engine"},
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    include_package_data=True,
)
