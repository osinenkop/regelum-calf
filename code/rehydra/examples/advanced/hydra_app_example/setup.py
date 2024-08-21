#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from setuptools import find_packages, setup

setup(
    name="rehydra-app",
    version="0.1",
    packages=find_packages(include=["rehydra_app"]),
    entry_points={"console_scripts": ["rehydra_app = rehydra_app.main:main"]},
    author="you!",
    author_email="your_email@example.com",
    url="http://rehydra-app.example.com",
    include_package_data=True,
    install_requires=["rehydra-core>=1.0.0"],
)
