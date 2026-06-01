from setuptools import setup, find_packages

setup(
    name="bevunify",
    version="0.1.0",
    description="Unified Hydra project for BEV segmentation models on GaussianLSS GT",
    packages=find_packages(include=["bevunify", "bevunify.*"]),
    python_requires=">=3.8",
)
