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

INCLUDE_MODULE = ["lama_engine"]


setup(
    name="lama-engine",
    version="0.3.3",
    package_dir={"lama_engine": "lama_engine"},
    packages=find_packages(include=INCLUDE_MODULE),
    install_requires=REQUIREMENTS,
)
