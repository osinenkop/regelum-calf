#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from setuptools import find_packages, setup

setup(
    name="rehydra-configen",
    version="0.9.0.dev8",
    packages=find_packages(include=["configen"]),
    entry_points={"console_scripts": ["configen = configen.configen:main"]},
    author="Omry Yadan, Rosario Scalise",
    author_email="omry@fb.com, rosario@cs.uw.edu",
    url="https://github.com/facebookresearch/rehydra/tree/main/tools/configen",
    include_package_data=True,
    install_requires=[
        "rehydra-core>=1.1.0",
        "jinja2",
    ],
)
