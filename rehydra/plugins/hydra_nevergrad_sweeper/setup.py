# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="rehydra-nevergrad-sweeper",
    version=read_version("rehydra_plugins/rehydra_nevergrad_sweeper", "__init__.py"),
    author="Jeremy Rapin, Omry Yadan, Jieru Hu",
    author_email="jrapin@fb.com, omry@fb.com, jieru@fb.com",
    description="Rehydra Nevergrad Sweeper plugin",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/rehydra/",
    packages=find_namespace_packages(include=["rehydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "rehydra-core>=1.1.0.dev7",
        "nevergrad>=0.4.3.post9",
    ],
    include_package_data=True,
)
