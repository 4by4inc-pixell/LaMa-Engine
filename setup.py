"""
setup script for pip install
"""

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


setup(
    name="lama-engine",
    version="0.3.0",
    packages=[""],
    install_requires=REQUIREMENTS,
)
