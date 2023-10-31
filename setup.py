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


# INCLUDE_MODULE = ["lama_engine", "lama_engine.models", "lama_engine.saicinpainting"]


setup(
    name="lama-engine",
    version="0.3.10",
    # package_dir={"lama_engine": "lama_engine"},
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    package_data={
        "lama_engine": ["lama_engine/**"],
    },
)
