# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="rehydra-submitit-launcher",
    version=read_version("rehydra_plugins/rehydra_submitit_launcher", "__init__.py"),
    author="Jeremy Rapin, Jieru Hu, Omry Yadan",
    author_email="jrapin@fb.com, jieru@fb.com, omry@fb.com",
    description="Submitit Launcher for Rehydra apps",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookincubator/submitit",
    packages=find_namespace_packages(include=["rehydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "rehydra-core>=1.1.0.dev7",
        "submitit>=1.3.3",
    ],
    include_package_data=True,
)
