# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path
from setuptools import find_namespace_packages, setup

setup(
    name="rehydra-joblib-launcher",
    version="1.3.0",
    author="Grigory Yaremenko, Anton Bolychev, Georgiy Malaniya, Pavel Osinenko",
    author_email="p.osinenko@yandex.ru",
    description="Joblib Launcher for Rehydra apps",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/rehydra/",
    packages=find_namespace_packages(include=["rehydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "rehydra>=1.3.2",
        "joblib>=0.14.0",
    ],
    include_package_data=True,
)
